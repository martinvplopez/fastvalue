"""Option chain provider and analytics for the Options Flow Agent.

Network access and computation are split so the math can be tested offline:

* :func:`fetch_option_chains` downloads a bounded set of expirations from Yahoo
  Finance (cached per ticker): up to six between 7 and 60 DTE plus one between
  90 and 180 DTE for term structure.
* :func:`analyze_option_chains` is a pure function that turns chain frames into
  :class:`OptionsFlowData`: ATM IV at ~30 DTE, put/call skew, term structure,
  put/call ratios, premium-weighted flow, unusual activity, max pain, OI walls,
  HV20 and IV rank.
* :func:`fetch_options_flow` glues the two and persists one ATM IV snapshot per
  ticker per day under ``settings.STORE_DIR / "iv_history"`` so a true IV rank
  can be built over time (yfinance does not provide IV history).

Nothing here estimates missing data. If no chain can be loaded the result has
``available=False``. Stale quotes or thin coverage set ``quality="PARTIAL"``
with a note explaining why.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import yfinance as yf

import config.settings as settings
from data.cache_utils import ttl_cache
from data.yahoo import warm_session, yahoo_call

logger = logging.getLogger(__name__)

# Data quality levels (same values as agents.base, kept local so data/ does not import agents/).
LIVE = "LIVE"
PARTIAL = "PARTIAL"
UNAVAILABLE = "UNAVAILABLE"

# Expiration selection.
NEAR_MIN_DTE = 7
NEAR_MAX_DTE = 60
MAX_NEAR_EXPIRATIONS = 6
FAR_MIN_DTE = 90
FAR_MAX_DTE = 180
FAR_TARGET_DTE = 120
FAR_FALLBACK_MAX_DTE = 365
TARGET_DTE = 30

# Contract hygiene.
MIN_IV = 0.01
MAX_IV = 5.0
CONTRACT_MULTIPLIER = 100
MAX_INTERP_GAP = 0.10  # bracketing strikes may be at most 10% of spot apart
MAX_SINGLE_STRIKE_DISTANCE = 0.025  # a lone strike must sit within 2.5% of the target
NEAR_SPOT_BAND = 0.10  # "near the money" for quote freshness checks

# Volatility surface.
SKEW_PUT_MONEYNESS = 0.95
SKEW_CALL_MONEYNESS = 1.05
BACKWARDATION_RATIO = 1.05
CONTANGO_RATIO = 0.95
IV_CRUSH_IV_HV = 1.5
IV_CRUSH_RANK = 80.0
HV_WINDOW = 20
TRADING_DAYS = 252
MIN_HV_PROXY_OBS = 60

# Unusual activity.
UNUSUAL_VOL_OI_MULTIPLE = 2.0
UNUSUAL_MIN_VOLUME = 500
UNUSUAL_MIN_PREMIUM = 100_000.0
UNUSUAL_TOP_N = 5
# Deep ITM prints are mostly intrinsic value (dividend, box and conversion trades), not directional bets.
UNUSUAL_MAX_MONEYNESS = 0.20

# OI walls are only meaningful near spot (far OTM strikes hold long-dated hedges).
OI_WALL_BAND = 0.15

# Data freshness.
STALE_QUOTE_SHARE = 0.5
STALE_TRADE_DAYS = 4

# IV history.
IV_HISTORY_MIN_SNAPSHOTS = 20
IV_HISTORY_WINDOW = 252
IV_HISTORY_MAX_ROWS = 1000

# Call overwriting heuristic (yfinance has no trade side, so this is only a hint).
OVERWRITE_OTM_CALL_SHARE = 0.65
OVERWRITE_MIN_IV_HV = 1.2

_HISTORY_LOCK = threading.Lock()


class ChainFrames(NamedTuple):
    """Raw calls and puts frames for one expiration (yfinance column names)."""

    calls: pd.DataFrame
    puts: pd.DataFrame


@dataclass(frozen=True)
class OptionChainSet:
    """Raw chains returned by :func:`fetch_option_chains`."""

    ticker: str
    chains: dict[str, ChainFrames]  # expiry "YYYY-MM-DD" -> frames
    spot: float | None = None
    market_state: str | None = None
    earnings_date: str | None = None
    expirations_listed: int = 0
    as_of: str | None = None  # exchange-local date the expirations were selected against


@dataclass
class OptionsFlowData:
    """Options positioning and volatility analytics for one ticker.

    Every metric is ``None`` when it could not be computed from real data.
    """

    ticker: str
    available: bool = False
    error: str | None = None
    quality: str = UNAVAILABLE
    quality_notes: list[str] = field(default_factory=list)
    stale_quotes: bool = False

    as_of: str | None = None
    data_as_of: str | None = None  # latest lastTradeDate in the chains
    market_state: str | None = None
    spot: float | None = None
    expirations: list[dict[str, Any]] = field(default_factory=list)
    near_expirations_used: int = 0
    quoted_share: float | None = None  # share of near-the-money contracts with bid > 0 and ask > 0

    # Volatility surface.
    atm_expiry: str | None = None
    atm_dte: int | None = None
    atm_iv: float | None = None
    far_expiry: str | None = None
    far_dte: int | None = None
    atm_iv_far: float | None = None
    term_structure_ratio: float | None = None
    term_structure_state: str | None = None  # BACKWARDATION, CONTANGO, FLAT
    put_iv_otm: float | None = None
    call_iv_otm: float | None = None
    skew: float | None = None  # vol points: 95% put IV minus 105% call IV
    skew_normalized: float | None = None  # skew / ATM IV (both in vol points)
    hv20: float | None = None
    iv_hv_ratio: float | None = None
    iv_rank: float | None = None  # 0 to 100
    iv_percentile: float | None = None  # 0 to 100
    iv_rank_source: str | None = None  # "history" or "hv_proxy"
    iv_history_days: int = 0
    iv_snapshot_date: str | None = None
    iv_snapshot_reliable: bool = False
    iv_crush_warning: bool = False
    earnings_date: str | None = None
    earnings_before_atm_expiry: bool = False

    # Flow over 7 to 60 DTE.
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_call_oi: int = 0
    total_put_oi: int = 0
    put_call_volume_ratio: float | None = None
    put_call_oi_ratio: float | None = None
    call_premium: float = 0.0
    put_premium: float = 0.0
    otm_call_premium: float = 0.0
    otm_put_premium: float = 0.0
    otm_call_volume: int = 0
    otm_put_volume: int = 0

    # Unusual activity (7+ DTE, strikes within 20% of spot).
    unusual_contracts: list[dict[str, Any]] = field(default_factory=list)
    unusual_count: int = 0
    unusual_call_premium: float = 0.0
    unusual_put_premium: float = 0.0

    # Positioning levels.
    max_pain: float | None = None
    max_pain_expiry: str | None = None
    call_wall: float | None = None
    call_wall_oi: int | None = None
    put_wall: float | None = None
    put_wall_oi: int | None = None

    price_change_5d: float | None = None
    call_overwriting_heuristic: bool = False


@dataclass(eq=False)
class _Expiry:
    expiry: str
    dte: int
    calls: pd.DataFrame
    puts: pd.DataFrame

    @property
    def total_oi(self) -> float:
        return float(self.calls["oi"].sum() + self.puts["oi"].sum())


# --------------------------------------------------------------------------- dates


def _market_today() -> date:
    """Today's date on the US exchange calendar (falls back to the local date)."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001 - missing tz database must not break the provider
        return date.today()


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return None


def _is_monthly(d: date) -> bool:
    """Standard monthly expiration: the third Friday of the month."""
    return d.weekday() == 4 and 15 <= d.day <= 21


def select_expirations(
    expirations: Iterable[str], as_of: date | str | None = None
) -> tuple[list[str], str | None]:
    """
    Pick a bounded set of expirations to download.

    Returns up to ``MAX_NEAR_EXPIRATIONS`` expirations between 7 and 60 DTE
    (always the one closest to 30 DTE, then monthlies, then the rest by
    closeness to 30 DTE) and one far expiration for term structure (closest to
    120 DTE within 90 to 180 DTE, else closest to 120 DTE up to one year).
    """
    today = _to_date(as_of) or _market_today()
    parsed: list[tuple[str, date, int]] = []
    for exp in expirations:
        d = _to_date(exp)
        if d is not None:
            parsed.append((d.isoformat(), d, (d - today).days))

    near = [p for p in parsed if NEAR_MIN_DTE <= p[2] <= NEAR_MAX_DTE]
    chosen: list[tuple[str, date, int]] = []
    if near:
        by_target = sorted(near, key=lambda p: (abs(p[2] - TARGET_DTE), p[2]))
        chosen.append(by_target[0])
        for p in [q for q in by_target if _is_monthly(q[1])] + by_target:
            if len(chosen) >= MAX_NEAR_EXPIRATIONS:
                break
            if p not in chosen:
                chosen.append(p)
    near_selected = [p[0] for p in sorted(chosen, key=lambda p: p[2])]

    far_window = [p for p in parsed if FAR_MIN_DTE <= p[2] <= FAR_MAX_DTE]
    if not far_window:
        far_window = [p for p in parsed if NEAR_MAX_DTE < p[2] <= FAR_FALLBACK_MAX_DTE]
    far = min(far_window, key=lambda p: (abs(p[2] - FAR_TARGET_DTE), p[2]))[0] if far_window else None
    return near_selected, far


# --------------------------------------------------------------------------- network


def _earnings_from_underlying(underlying: Mapping[str, Any] | None, today: date) -> str | None:
    if not underlying:
        return None
    upcoming: list[date] = []
    for key in ("earningsTimestampStart", "earningsTimestamp", "earningsTimestampEnd"):
        ts = underlying.get(key)
        if isinstance(ts, (int, float)) and ts > 0:
            d = datetime.fromtimestamp(float(ts), tz=timezone.utc).date()
            if d >= today:
                upcoming.append(d)
    return min(upcoming).isoformat() if upcoming else None


@ttl_cache(settings.CACHE_TTL_OPTIONS)
def fetch_option_chains(ticker: str) -> OptionChainSet:
    """
    Download raw option chains for a bounded set of expirations (at most 7 chain calls).

    Raises on network failure (failures are not cached). Returns an empty
    ``chains`` dict when the ticker has no listed options in range.
    """
    symbol = ticker.upper()
    today = _market_today()
    yf_ticker = yf.Ticker(symbol)
    expirations = tuple(yahoo_call(lambda: yf_ticker.options, f"{symbol} expirations") or ())
    if not expirations and "." not in symbol:
        # yfinance answers a refused request (an expired session) on this endpoint with an empty list and no error.
        # A US listing without options is rare, so ask once more on a fresh session before calling it optionless;
        # listings with an exchange suffix (.L, .AS) have no Yahoo options and skip this.
        warm_session(force=True)
        yf_ticker = yf.Ticker(symbol)
        expirations = tuple(yahoo_call(lambda: yf_ticker.options, f"{symbol} expirations, fresh session") or ())
    near, far = select_expirations(expirations, today)
    wanted = near + ([far] if far else [])

    chains: dict[str, ChainFrames] = {}
    underlying: Mapping[str, Any] | None = None
    last_error: Exception | None = None
    for exp in wanted:
        try:
            chain = yahoo_call(lambda exp=exp: yf_ticker.option_chain(exp), f"{symbol} chain {exp}")
        except Exception as exc:  # noqa: BLE001 - one bad expiry should not drop the rest
            logger.warning("Option chain %s %s failed: %s", symbol, exp, exc)
            last_error = exc
            continue
        chains[exp] = ChainFrames(calls=chain.calls, puts=chain.puts)
        if underlying is None:
            underlying = getattr(chain, "underlying", None)

    if wanted and not chains:
        raise RuntimeError(f"All option chain requests failed for {symbol}: {last_error}")

    spot = None
    market_state = None
    if underlying:
        price = underlying.get("regularMarketPrice")
        if isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
            spot = float(price)
        market_state = underlying.get("marketState")

    return OptionChainSet(
        ticker=symbol,
        chains=chains,
        spot=spot,
        market_state=market_state,
        earnings_date=_earnings_from_underlying(underlying, today),
        expirations_listed=len(expirations),
        as_of=today.isoformat(),
    )


# --------------------------------------------------------------------------- IV history


def iv_history_path(ticker: str) -> Path:
    return Path(settings.STORE_DIR) / "iv_history" / f"{ticker.upper()}.csv"


def load_iv_history(ticker: str) -> pd.Series:
    """Stored daily ATM IV snapshots (index ISO date strings, sorted)."""
    empty = pd.Series(dtype=float, name="atm_iv")
    path = iv_history_path(ticker)
    if not path.exists():
        return empty
    try:
        df = pd.read_csv(path, dtype={"date": str})
    except (OSError, ValueError) as exc:
        logger.warning("Could not read IV history %s: %s", path, exc)
        return empty
    if "date" not in df.columns or "atm_iv" not in df.columns:
        return empty
    df["atm_iv"] = pd.to_numeric(df["atm_iv"], errors="coerce")
    df = df.dropna(subset=["date", "atm_iv"])
    df = df[(df["atm_iv"] > MIN_IV) & (df["atm_iv"] < MAX_IV)]
    df = df.drop_duplicates(subset="date", keep="last")
    return pd.Series(df["atm_iv"].to_numpy(float), index=df["date"].to_numpy(str), name="atm_iv").sort_index()


def record_iv_snapshot(ticker: str, snapshot_date: str, atm_iv: float) -> Path:
    """Store today's ATM IV (overwrites an existing row for the same date)."""
    path = iv_history_path(ticker)
    with _HISTORY_LOCK:
        history = load_iv_history(ticker)
        history.loc[snapshot_date] = round(float(atm_iv), 6)
        history = history.sort_index().tail(IV_HISTORY_MAX_ROWS)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        pd.DataFrame({"date": history.index, "atm_iv": history.to_numpy()}).to_csv(tmp, index=False)
        os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- analytics helpers


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _prepare(df: pd.DataFrame | None, spot: float) -> pd.DataFrame:
    """Normalize a raw chain frame and apply contract hygiene flags."""
    cols = ["strike", "bid", "ask", "last", "volume", "oi", "iv", "quoted", "iv_ok", "mid", "premium", "last_trade"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(index=df.index)
    out["strike"] = _column(df, "strike")
    out["bid"] = _column(df, "bid").fillna(0.0)
    out["ask"] = _column(df, "ask").fillna(0.0)
    out["last"] = _column(df, "lastPrice").fillna(0.0).clip(lower=0.0)
    out["volume"] = _column(df, "volume").fillna(0.0).clip(lower=0.0)
    out["oi"] = _column(df, "openInterest").fillna(0.0).clip(lower=0.0)
    out["iv"] = _column(df, "impliedVolatility")
    out["quoted"] = (out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"])
    out["iv_ok"] = (out["iv"] > MIN_IV) & (out["iv"] < MAX_IV)
    out["mid"] = np.where(out["quoted"], (out["bid"] + out["ask"]) / 2.0, out["last"])
    out["premium"] = out["volume"] * out["mid"] * CONTRACT_MULTIPLIER
    if "lastTradeDate" in df.columns:
        out["last_trade"] = pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce")
    else:
        out["last_trade"] = pd.NaT
    out = out[out["strike"].notna() & (out["strike"] > 0)]
    return out.sort_values("strike").reset_index(drop=True)


def _split_frames(frames: Any) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if hasattr(frames, "calls") and hasattr(frames, "puts"):
        return frames.calls, frames.puts
    if isinstance(frames, Mapping):
        return frames.get("calls"), frames.get("puts")
    calls, puts = frames[0], frames[1]
    return calls, puts


def _interp_strike(pool: pd.DataFrame, target: float, spot: float) -> float | None:
    """IV at ``target`` by linear interpolation between the neighboring strikes."""
    if pool.empty:
        return None
    curve = pool.groupby("strike")["iv"].mean().sort_index()
    strikes = curve.index.to_numpy(float)
    ivs = curve.to_numpy(float)
    exact = np.isclose(strikes, target, rtol=0.0, atol=1e-9)
    if exact.any():
        return float(ivs[exact][0])
    below = strikes < target
    above = strikes > target
    if below.any() and above.any():
        k0, k1 = strikes[below][-1], strikes[above][0]
        if k1 - k0 <= MAX_INTERP_GAP * spot:
            v0, v1 = ivs[below][-1], ivs[above][0]
            return float(v0 + (target - k0) / (k1 - k0) * (v1 - v0))
    nearest = int(np.argmin(np.abs(strikes - target)))
    if abs(strikes[nearest] - target) <= MAX_SINGLE_STRIKE_DISTANCE * spot:
        return float(ivs[nearest])
    return None


def _iv_at(df: pd.DataFrame, target: float, spot: float) -> tuple[float | None, bool]:
    """IV at a strike, preferring contracts with a live two-sided quote.

    Returns (iv, from_quoted_contracts).
    """
    usable = df[df["iv_ok"]] if not df.empty else df
    if usable.empty:
        return None, False
    quoted = usable[usable["quoted"]]
    value = _interp_strike(quoted, target, spot)
    if value is not None:
        return value, True
    return _interp_strike(usable, target, spot), False


def _atm_iv(exp: _Expiry, spot: float) -> tuple[float | None, bool]:
    call_iv, call_q = _iv_at(exp.calls, spot, spot)
    put_iv, put_q = _iv_at(exp.puts, spot, spot)
    values = [v for v in (call_iv, put_iv) if v is not None]
    if not values:
        return None, False
    flags = [q for v, q in ((call_iv, call_q), (put_iv, put_q)) if v is not None]
    return float(np.mean(values)), all(flags)


def max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    """Settlement strike that minimizes the total intrinsic value paid to option holders.

    Frames use the normalized columns ``strike`` and ``oi``.
    """
    c_k = calls["strike"].to_numpy(float) if not calls.empty else np.array([])
    c_oi = calls["oi"].to_numpy(float) if not calls.empty else np.array([])
    p_k = puts["strike"].to_numpy(float) if not puts.empty else np.array([])
    p_oi = puts["oi"].to_numpy(float) if not puts.empty else np.array([])
    if c_oi.sum() + p_oi.sum() <= 0:
        return None
    strikes = np.unique(np.concatenate([c_k, p_k]))
    settle = strikes[:, None]
    call_pay = (np.maximum(settle - c_k[None, :], 0.0) * c_oi[None, :]).sum(axis=1)
    put_pay = (np.maximum(p_k[None, :] - settle, 0.0) * p_oi[None, :]).sum(axis=1)
    return float(strikes[int(np.argmin(call_pay + put_pay))])


def _log_returns(ohlcv: pd.DataFrame | None) -> pd.Series:
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(ohlcv["close"], errors="coerce").dropna()
    close = close[close > 0]
    return np.log(close).diff().dropna()


def _hv20(returns: pd.Series) -> float | None:
    if len(returns) < HV_WINDOW:
        return None
    hv = float(returns.tail(HV_WINDOW).std(ddof=1) * math.sqrt(TRADING_DAYS))
    return hv if hv > 0 and math.isfinite(hv) else None


def _rank_and_percentile(window: pd.Series, current: float, others: pd.Series) -> tuple[float, float]:
    lo, hi = float(window.min()), float(window.max())
    rank = 50.0 if hi - lo < 1e-12 else (current - lo) / (hi - lo) * 100.0
    pct = float((others < current).mean() * 100.0) if len(others) else 50.0
    return float(np.clip(rank, 0.0, 100.0)), pct


def _price_change(ohlcv: pd.DataFrame | None, bars: int = 5) -> float | None:
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return None
    close = pd.to_numeric(ohlcv["close"], errors="coerce").dropna()
    if len(close) <= bars or close.iloc[-1 - bars] <= 0:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - bars] - 1.0)


def _unusual(expiries: list[_Expiry], spot: float) -> pd.DataFrame:
    frames = []
    for exp in expiries:
        for kind, df in (("CALL", exp.calls), ("PUT", exp.puts)):
            if df.empty:
                continue
            mask = (
                (df["volume"] > UNUSUAL_VOL_OI_MULTIPLE * df["oi"])
                & ((df["volume"] >= UNUSUAL_MIN_VOLUME) | (df["premium"] >= UNUSUAL_MIN_PREMIUM))
                & ((df["strike"] / spot - 1.0).abs() <= UNUSUAL_MAX_MONEYNESS)
            )
            hits = df[mask]
            if hits.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "type": kind,
                        "expiry": exp.expiry,
                        "dte": exp.dte,
                        "strike": hits["strike"].to_numpy(float),
                        "moneyness_pct": (hits["strike"].to_numpy(float) / spot - 1.0) * 100.0,
                        "volume": hits["volume"].to_numpy(float),
                        "open_interest": hits["oi"].to_numpy(float),
                        "premium": hits["premium"].to_numpy(float),
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=["type", "expiry", "dte", "strike", "moneyness_pct", "volume", "open_interest", "premium"])
    return pd.concat(frames, ignore_index=True).sort_values("premium", ascending=False).reset_index(drop=True)


def _latest_trade(expiries: list[_Expiry]) -> pd.Timestamp | None:
    stamps = [df["last_trade"].max() for e in expiries for df in (e.calls, e.puts) if not df.empty]
    stamps = [s for s in stamps if s is not None and not pd.isna(s)]
    return max(stamps) if stamps else None


# --------------------------------------------------------------------------- analysis


def analyze_option_chains(
    chains: Mapping[Any, Any],
    spot: float | None,
    ohlcv: pd.DataFrame | None,
    ticker: str,
    *,
    as_of: date | str | None = None,
    earnings_date: date | str | None = None,
    market_state: str | None = None,
    iv_history: pd.Series | None = None,
) -> OptionsFlowData:
    """
    Compute options analytics from raw chain frames. Pure: no network, no disk.

    Args:
        chains: Mapping of expiry (``"YYYY-MM-DD"`` or date) to calls/puts frames
            (a ``ChainFrames``, a ``(calls, puts)`` tuple, a yfinance chain object
            or a ``{"calls":, "puts":}`` dict) with yfinance column names.
        spot: Underlying price used for moneyness.
        ohlcv: Daily OHLCV (for HV20, the HV proxy and 5-day price change).
        ticker: Symbol.
        as_of: Date used for DTE (defaults to today on the US exchange calendar).
        earnings_date: Next earnings date if known.
        market_state: Yahoo ``marketState`` if known (REGULAR, PRE, POST, CLOSED...).
        iv_history: Stored ATM IV snapshots (index ISO dates). The current value
            is merged in for the rank but not persisted here.
    """
    symbol = ticker.upper()
    today = _to_date(as_of) or _market_today()
    data = OptionsFlowData(ticker=symbol, as_of=today.isoformat(), market_state=market_state)

    if spot is None or not math.isfinite(float(spot)) or float(spot) <= 0:
        data.error = "No spot price available to anchor the option chain"
        return data
    spot = float(spot)
    data.spot = round(spot, 4)

    expiries: list[_Expiry] = []
    for key, frames in chains.items():
        exp_date = _to_date(key)
        if exp_date is None:
            continue
        calls_raw, puts_raw = _split_frames(frames)
        exp = _Expiry(exp_date.isoformat(), (exp_date - today).days, _prepare(calls_raw, spot), _prepare(puts_raw, spot))
        if not (exp.calls.empty and exp.puts.empty):
            expiries.append(exp)
    expiries.sort(key=lambda e: e.dte)

    near = [e for e in expiries if NEAR_MIN_DTE <= e.dte <= NEAR_MAX_DTE]
    if not near:
        data.error = f"No option expirations between {NEAR_MIN_DTE} and {NEAR_MAX_DTE} days with contracts"
        return data

    near_calls = pd.concat([e.calls for e in near], ignore_index=True)
    near_puts = pd.concat([e.puts for e in near], ignore_index=True)
    near_all = pd.concat([near_calls, near_puts], ignore_index=True)
    if near_all["volume"].sum() <= 0 and near_all["oi"].sum() <= 0 and not near_all["iv_ok"].any():
        data.error = "Option chain has no volume, open interest or usable implied volatility"
        return data

    notes: list[str] = []

    # ---- quote freshness
    band = near_all[(near_all["strike"] - spot).abs() <= NEAR_SPOT_BAND * spot]
    pool = band if not band.empty else near_all
    data.quoted_share = round(float(pool["quoted"].mean()), 3)
    if data.quoted_share < STALE_QUOTE_SHARE:
        data.stale_quotes = True
        notes.append(
            f"Quotes look stale: only {data.quoted_share:.0%} of near-the-money contracts have a live bid and ask "
            "(market closed or weekend), so IVs may be unreliable"
        )

    far_candidates = [e for e in expiries if e.dte > NEAR_MAX_DTE]
    latest = _latest_trade(near + far_candidates)
    if latest is not None:
        data.data_as_of = latest.strftime("%Y-%m-%d")
        age = (today - latest.date()).days
        if age > STALE_TRADE_DAYS:
            notes.append(f"Latest option trade is from {data.data_as_of} ({age} days old)")
    stale_trades = latest is not None and (today - latest.date()).days > STALE_TRADE_DAYS

    # ---- flow over 7 to 60 DTE
    data.total_call_volume = int(near_calls["volume"].sum())
    data.total_put_volume = int(near_puts["volume"].sum())
    data.total_call_oi = int(near_calls["oi"].sum())
    data.total_put_oi = int(near_puts["oi"].sum())
    if data.total_call_volume > 0:
        data.put_call_volume_ratio = round(data.total_put_volume / data.total_call_volume, 3)
    if data.total_call_oi > 0:
        data.put_call_oi_ratio = round(data.total_put_oi / data.total_call_oi, 3)
    if data.total_call_volume + data.total_put_volume <= 0:
        notes.append(f"No option volume in {NEAR_MIN_DTE}-{NEAR_MAX_DTE} DTE expirations yet (session not started?)")

    otm_calls = near_calls[near_calls["strike"] > spot]
    otm_puts = near_puts[near_puts["strike"] < spot]
    data.call_premium = round(float(near_calls["premium"].sum()), 2)
    data.put_premium = round(float(near_puts["premium"].sum()), 2)
    data.otm_call_premium = round(float(otm_calls["premium"].sum()), 2)
    data.otm_put_premium = round(float(otm_puts["premium"].sum()), 2)
    data.otm_call_volume = int(otm_calls["volume"].sum())
    data.otm_put_volume = int(otm_puts["volume"].sum())

    # ---- ATM IV at the expiry closest to 30 DTE
    atm_from_quoted = False
    usable_near = 0
    atm_exp: _Expiry | None = None
    for exp in sorted(near, key=lambda e: (abs(e.dte - TARGET_DTE), e.dte)):
        iv, quoted = _atm_iv(exp, spot)
        if iv is None:
            continue
        usable_near += 1
        if atm_exp is None:
            atm_exp, data.atm_iv, atm_from_quoted = exp, round(iv, 5), quoted
    data.near_expirations_used = usable_near
    if atm_exp is None:
        notes.append("ATM implied volatility could not be computed near spot")
    else:
        data.atm_expiry, data.atm_dte = atm_exp.expiry, atm_exp.dte
        if usable_near < 2:
            notes.append(f"Only {usable_near} expiration between {NEAR_MIN_DTE} and {NEAR_MAX_DTE} DTE had usable IV near spot")

        put_iv, _ = _iv_at(atm_exp.puts, SKEW_PUT_MONEYNESS * spot, spot)
        call_iv, _ = _iv_at(atm_exp.calls, SKEW_CALL_MONEYNESS * spot, spot)
        data.put_iv_otm = round(put_iv, 5) if put_iv is not None else None
        data.call_iv_otm = round(call_iv, 5) if call_iv is not None else None
        if put_iv is not None and call_iv is not None:
            data.skew = round((put_iv - call_iv) * 100.0, 3)
            data.skew_normalized = round((put_iv - call_iv) / data.atm_iv, 4)

    # ---- term structure
    far_ordered = sorted(
        far_candidates,
        key=lambda e: (not (FAR_MIN_DTE <= e.dte <= FAR_MAX_DTE), abs(e.dte - FAR_TARGET_DTE), e.dte),
    )
    for exp in far_ordered:
        if exp.dte > FAR_FALLBACK_MAX_DTE:
            continue
        iv, _ = _atm_iv(exp, spot)
        if iv is not None:
            data.far_expiry, data.far_dte, data.atm_iv_far = exp.expiry, exp.dte, round(iv, 5)
            break
    if data.atm_iv_far is None:
        notes.append(f"No {FAR_MIN_DTE}-{FAR_MAX_DTE} DTE expiration with usable IV, so term structure is not available")
    elif data.atm_iv is not None:
        ratio = data.atm_iv / data.atm_iv_far
        data.term_structure_ratio = round(ratio, 4)
        data.term_structure_state = (
            "BACKWARDATION" if ratio > BACKWARDATION_RATIO else "CONTANGO" if ratio < CONTANGO_RATIO else "FLAT"
        )

    used = near + ([e for e in far_candidates if e.expiry == data.far_expiry])
    data.expirations = [
        {"expiry": e.expiry, "dte": e.dte, "role": "near" if NEAR_MIN_DTE <= e.dte <= NEAR_MAX_DTE else "far"}
        for e in used
    ]

    # ---- earnings
    earn = _to_date(earnings_date)
    if earn is not None and earn >= today:
        data.earnings_date = earn.isoformat()
        if atm_exp is not None:
            data.earnings_before_atm_expiry = earn <= _to_date(atm_exp.expiry)

    # ---- max pain and OI walls
    by_oi = max(near, key=lambda e: e.total_oi)
    if by_oi.total_oi > 0:
        data.max_pain = max_pain(by_oi.calls, by_oi.puts)
        data.max_pain_expiry = by_oi.expiry
    call_oi = near_calls.groupby("strike")["oi"].sum()
    put_oi = near_puts.groupby("strike")["oi"].sum()
    wall_hi, wall_lo = spot * (1.0 + OI_WALL_BAND), spot * (1.0 - OI_WALL_BAND)
    above = call_oi[(call_oi.index > spot) & (call_oi.index <= wall_hi) & (call_oi > 0)]
    below = put_oi[(put_oi.index < spot) & (put_oi.index >= wall_lo) & (put_oi > 0)]
    if not above.empty:
        data.call_wall, data.call_wall_oi = float(above.idxmax()), int(above.max())
    if not below.empty:
        data.put_wall, data.put_wall_oi = float(below.idxmax()), int(below.max())

    # ---- unusual activity (weeklies under 7 DTE and deep ITM prints are excluded)
    unusual = _unusual([e for e in expiries if e.dte >= NEAR_MIN_DTE], spot)
    data.unusual_count = int(len(unusual))
    data.unusual_call_premium = round(float(unusual.loc[unusual["type"] == "CALL", "premium"].sum()), 2)
    data.unusual_put_premium = round(float(unusual.loc[unusual["type"] == "PUT", "premium"].sum()), 2)
    data.unusual_contracts = [
        {
            "type": row.type,
            "expiry": row.expiry,
            "dte": int(row.dte),
            "strike": float(row.strike),
            "moneyness_pct": round(float(row.moneyness_pct), 2),
            "volume": int(row.volume),
            "open_interest": int(row.open_interest),
            "premium": round(float(row.premium), 2),
        }
        for row in unusual.head(UNUSUAL_TOP_N).itertuples(index=False)
    ]

    # ---- realized volatility
    returns = _log_returns(ohlcv)
    hv20 = _hv20(returns)
    if hv20 is None:
        notes.append(f"Not enough price history for {HV_WINDOW}-day historical volatility")
    else:
        data.hv20 = round(hv20, 5)
        if data.atm_iv is not None:
            data.iv_hv_ratio = round(data.atm_iv / hv20, 3)
    data.price_change_5d = _price_change(ohlcv)

    # ---- IV rank: true history when available, otherwise a labeled HV proxy
    if data.atm_iv is not None:
        data.iv_snapshot_date = data.data_as_of or today.isoformat()
        data.iv_snapshot_reliable = bool(atm_from_quoted and not data.stale_quotes and not stale_trades)
        history = pd.Series(dtype=float) if iv_history is None else iv_history.copy()
        history.index = [str(_to_date(i) or i) for i in history.index]
        history = history[~history.index.duplicated(keep="last")]
        history.loc[data.iv_snapshot_date] = data.atm_iv
        history = history.sort_index().tail(IV_HISTORY_WINDOW)
        data.iv_history_days = int(len(history))
        if len(history) >= IV_HISTORY_MIN_SNAPSHOTS:
            others = history.drop(index=data.iv_snapshot_date)
            rank, pct = _rank_and_percentile(history, data.atm_iv, others)
            data.iv_rank, data.iv_percentile, data.iv_rank_source = round(rank, 1), round(pct, 1), "history"
        else:
            hv_series = (returns.rolling(HV_WINDOW).std(ddof=1) * math.sqrt(TRADING_DAYS)).dropna().tail(TRADING_DAYS)
            if len(hv_series) >= MIN_HV_PROXY_OBS:
                rank, pct = _rank_and_percentile(hv_series, data.atm_iv, hv_series)
                data.iv_rank, data.iv_percentile, data.iv_rank_source = round(rank, 1), round(pct, 1), "hv_proxy"

    data.iv_crush_warning = bool(
        (data.iv_hv_ratio is not None and data.iv_hv_ratio >= IV_CRUSH_IV_HV)
        or (data.iv_rank_source == "history" and data.iv_rank is not None and data.iv_rank >= IV_CRUSH_RANK)
        or (
            data.earnings_before_atm_expiry
            and data.term_structure_ratio is not None
            and data.term_structure_ratio > BACKWARDATION_RATIO
        )
    )

    otm_vol = data.otm_call_volume + data.otm_put_volume
    data.call_overwriting_heuristic = bool(
        otm_vol > 0
        and data.otm_call_volume / otm_vol >= OVERWRITE_OTM_CALL_SHARE
        and data.price_change_5d is not None
        and data.price_change_5d <= 0.0
        and data.iv_hv_ratio is not None
        and data.iv_hv_ratio >= OVERWRITE_MIN_IV_HV
    )

    data.quality_notes = notes
    data.quality = PARTIAL if notes else LIVE
    data.available = True
    return data


# --------------------------------------------------------------------------- glue


def fetch_options_flow(ticker: str, ohlcv: pd.DataFrame | None = None) -> OptionsFlowData:
    """
    Fetch chains, analyze them and persist today's ATM IV snapshot.

    Returns ``OptionsFlowData(available=False, error=...)`` when no usable chain
    exists. Never substitutes estimates for missing data.
    """
    symbol = ticker.upper()
    try:
        raw = fetch_option_chains(symbol)
    except Exception as exc:  # noqa: BLE001 - provider failures become UNAVAILABLE, not crashes
        logger.warning("Option chain fetch failed for %s: %s", symbol, exc)
        return OptionsFlowData(ticker=symbol, error=f"Option chain request failed: {exc}")

    if not raw.chains:
        reason = (
            f"No listed option expirations between {NEAR_MIN_DTE} and {FAR_FALLBACK_MAX_DTE} days"
            if raw.expirations_listed
            else "No listed options for this ticker"
        )
        return OptionsFlowData(ticker=symbol, error=reason, as_of=raw.as_of)

    spot = raw.spot
    if spot is None and ohlcv is not None and not ohlcv.empty and "close" in ohlcv.columns:
        last_close = pd.to_numeric(ohlcv["close"], errors="coerce").dropna()
        spot = float(last_close.iloc[-1]) if len(last_close) else None

    try:
        data = analyze_option_chains(
            raw.chains,
            spot,
            ohlcv,
            symbol,
            as_of=raw.as_of,
            earnings_date=raw.earnings_date,
            market_state=raw.market_state,
            iv_history=load_iv_history(symbol),
        )
    except Exception as exc:  # noqa: BLE001 - malformed chain data must not take down the agent
        logger.exception("Could not analyze option chain for %s", symbol)
        return OptionsFlowData(ticker=symbol, error=f"Could not parse option chain: {exc}", as_of=raw.as_of)

    if data.available and data.iv_snapshot_reliable and data.atm_iv is not None and data.iv_snapshot_date:
        try:
            record_iv_snapshot(symbol, data.iv_snapshot_date, data.atm_iv)
        except OSError as exc:
            logger.warning("Could not store IV snapshot for %s: %s", symbol, exc)
    return data
