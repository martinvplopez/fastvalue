"""Macro indicator data for market regime detection.

Two steps:
1. ``load_macro_history`` downloads daily series once (cached): SPY, VIX,
   VIX3M, RSP, HYG, LQD, ^TNX from Yahoo and the 10Y-2Y / 10Y-3M Treasury
   spreads from FRED (public CSV endpoint, no API key).
2. ``compute_macro_snapshot(history, asof)`` is pure and slices every series at
   ``asof``, so the same code serves live analysis and historical replay.

Missing inputs are skipped (never replaced with made-up values) and reported
in ``missing`` / ``data_quality``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

import config.settings as settings
from data.cache_utils import ttl_cache

logger = logging.getLogger(__name__)

YF_SERIES: dict[str, str] = {
    "spy": "SPY",
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "rsp": "RSP",  # Equal-weight S&P 500 (breadth proxy)
    "hyg": "HYG",  # High yield corporate bonds
    "lqd": "LQD",  # Investment grade corporate bonds
    "tnx": "^TNX",  # 10-Year Treasury yield
}
FRED_SERIES: dict[str, str] = {
    "t10y2y": "T10Y2Y",
    "t10y3m": "T10Y3M",
}
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"

# Backwards-compatible names.
FRED_T10Y2Y = FRED_SERIES["t10y2y"]
FRED_T10Y3M = FRED_SERIES["t10y3m"]
YF_VIX = YF_SERIES["vix"]
YF_SPY = YF_SERIES["spy"]
YF_RSP = YF_SERIES["rsp"]
YF_HYG = YF_SERIES["hyg"]
YF_LQD = YF_SERIES["lqd"]

# Regime thresholds on the summed component points.
REGIME_THRESHOLDS: tuple[tuple[float, str], ...] = ((4.0, "BULL"), (1.0, "CORRECTION"), (-3.0, "BEAR"))
# Highest achievable sum of component points (used to scale the agent's health score).
MAX_REGIME_SCORE = 7.5


@dataclass
class MacroHistory:
    series: dict[str, pd.Series]
    errors: dict[str, str]
    fetched_at: str


@dataclass
class MacroRegimeData:
    """Aggregated macro indicators for regime classification."""

    # Market trend
    spy_price: float
    spy_sma_50: float
    spy_sma_200: float

    # Volatility
    vix_level: float
    vix_sma_20: float
    vix_percentile: float  # % of the last year's sessions with a lower VIX close

    # Yield curve (from FRED)
    yield_spread_10y_2y: float  # Positive = normal, negative = inverted
    yield_spread_10y_3m: float
    yield_curve_status: str  # NORMAL, FLAT, INVERTED, UNKNOWN

    # Market breadth
    breadth_ratio: float  # RSP/SPY ratio, rebased to 1.0 one year ago
    breadth_ratio_sma_50: float
    breadth_trend: str  # IMPROVING, STABLE, DETERIORATING, UNKNOWN

    # Credit spreads
    credit_ratio: float  # HYG/LQD ratio, rebased (rising = risk-on)
    credit_ratio_sma_50: float
    credit_trend: str  # RISK_ON, NEUTRAL, RISK_OFF, UNKNOWN

    # Regime classification
    regime: str  # BULL, CORRECTION, BEAR, CRISIS
    regime_score: float  # Sum of component points
    data_quality: str  # FULL, PARTIAL, ESTIMATED

    # Extended indicators (optional so older callers/tests keep working)
    vix3m_level: float | None = None
    vix_term_ratio: float | None = None  # VIX / VIX3M; > 1 = backwardation (stress)
    ten_year_yield: float | None = None
    ten_year_change_60d: float | None = None  # percentage points
    curve_uninverted: bool = False
    # Buying-conditions score under the contrarian VIX stance (None = use regime_score).
    opportunity_score: float | None = None
    opportunity_components: list[dict[str, Any]] = field(default_factory=list)
    components: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    as_of: str | None = None


# --------------------------------------------------------------------- loading


def _fetch_fred_csv(series_id: str, start: str) -> pd.Series:
    df = pd.read_csv(FRED_CSV_URL.format(series=series_id, start=start))
    date_col = df.columns[0]
    values = pd.to_numeric(df[series_id], errors="coerce")
    s = pd.Series(values.values, index=pd.to_datetime(df[date_col]), name=series_id).dropna()
    if s.empty:
        raise ValueError(f"FRED series {series_id} returned no data")
    return s


def _download_closes(tickers: list[str], start: str) -> dict[str, pd.Series]:
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        return {}
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    out: dict[str, pd.Series] = {}
    for t in tickers:
        if t in closes.columns:
            s = closes[t].dropna()
            if not s.empty:
                s.index = pd.to_datetime(s.index).tz_localize(None)
                out[t] = s.astype(float)
    return out


class MacroDataError(RuntimeError):
    """Raised when the core series (SPY and VIX) could not be downloaded."""


def load_macro_history(years: int = 3) -> MacroHistory:
    """
    Download all macro series covering the last ``years`` years.

    Results are cached, but a failed core download is not, so a network blip
    does not blind the regime agent for the whole cache period.
    """
    try:
        return _load_macro_history_cached(years)
    except MacroDataError as exc:
        return MacroHistory(series={}, errors={"core": str(exc)}, fetched_at=datetime.now().isoformat(timespec="seconds"))


@ttl_cache(settings.CACHE_TTL_MACRO)
def _load_macro_history_cached(years: int) -> MacroHistory:
    start = (datetime.today() - timedelta(days=int(years * 365.25) + 10)).strftime("%Y-%m-%d")
    series: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}

    try:
        closes = _download_closes(list(YF_SERIES.values()), start)
    except Exception as exc:  # noqa: BLE001 - provider errors are reported, not raised
        logger.warning("Macro download failed: %s", exc)
        closes = {}
        for key in YF_SERIES:
            errors[key] = str(exc)
    for key, symbol in YF_SERIES.items():
        if symbol in closes:
            series[key] = closes[symbol]
        else:
            errors.setdefault(key, f"No data for {symbol}")

    for key, series_id in FRED_SERIES.items():
        try:
            series[key] = _fetch_fred_csv(series_id, start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FRED %s unavailable: %s", series_id, exc)
            errors[key] = str(exc)

    if "spy" not in series and "vix" not in series:
        raise MacroDataError(errors.get("spy") or "SPY and VIX unavailable")

    if "tnx" in series and float(series["tnx"].tail(20).median()) > 20:
        series["tnx"] = series["tnx"] / 10.0  # Old Yahoo convention quoted yield x10.

    return MacroHistory(series=series, errors=errors, fetched_at=datetime.now().isoformat(timespec="seconds"))


def history_years_needed(asof: pd.Timestamp | None) -> int:
    """
    History length to load for a snapshot at ``asof``.

    Bucketed so a long replay only triggers a few cached downloads: 3 years for
    live use, the default backtest window, or 25 years for long studies.
    """
    if asof is None:
        return 3
    age_years = (pd.Timestamp.today() - pd.Timestamp(asof)).days / 365.25
    if age_years <= 1:
        return 3
    if age_years <= settings.LOOKBACK_YEARS:
        return settings.LOOKBACK_YEARS + 2
    return 25


# ----------------------------------------------------------------- computation


def _slice(history: MacroHistory, key: str, asof: pd.Timestamp | None, bars: int = 260) -> pd.Series | None:
    s = history.series.get(key)
    if s is None or s.empty:
        return None
    if asof is not None:
        s = s.loc[: pd.Timestamp(asof)]
    s = s.tail(bars)
    return s if not s.empty else None


def _ratio_trend(num: pd.Series | None, den: pd.Series | None, up: float, down: float) -> tuple[float, float, int] | None:
    """Return (rebased ratio, its 50d SMA, +1/0/-1 trend) or None when data is short."""
    if num is None or den is None:
        return None
    joined = pd.concat([num, den], axis=1, join="inner").dropna()
    if len(joined) < 60:
        return None
    ratio = joined.iloc[:, 0] / joined.iloc[:, 1].replace(0, np.nan)
    ratio = (ratio / ratio.iloc[0]).dropna()
    current = float(ratio.iloc[-1])
    sma50 = float(ratio.tail(50).mean())
    trend = 1 if current > sma50 * up else -1 if current < sma50 * down else 0
    return current, sma50, trend


def classify_regime(regime_score: float) -> str:
    for threshold, name in REGIME_THRESHOLDS:
        if regime_score >= threshold:
            return name
    return "CRISIS"


VIX_LONG_RUN_AVERAGE = 20.0
VIX_EXTREME_FEAR = 40.0


def contrarian_vix_points(vix_level: float) -> tuple[float, str]:
    """
    VIX as a buying signal (contrarian stance): fear is where bargains come from.

    The long-run average is about 20; readings above 40 have historically been
    among the best times to buy. Very low readings mean no fear discount.
    """
    if vix_level >= VIX_EXTREME_FEAR:
        return 6.0, "extreme fear, historically a strong time to buy (deploy in tranches over a few weeks)"
    if vix_level >= 30:
        # 1993-2026: 12 months after the first close above 30, SPY returned +8.0% (below its +12.1% average day),
        # against +22.5% after 40. Fear at 30 is not yet a discount, so no points (decision_log.md T-09).
        return 0.0, "high fear, but returns after a first close above 30 were below average; the signal is 40"
    if vix_level >= VIX_LONG_RUN_AVERAGE:
        return 0.5, "above the long-run average of 20"
    if vix_level < 13:
        return -1.0, "complacency, no fear discount"
    return 0.0, "below the long-run average of 20"


def _opportunity_components(components: list[dict[str, Any]], vix_level: float) -> list[dict[str, Any]]:
    """Regime components with the VIX risk reading replaced by the contrarian one."""
    out: list[dict[str, Any]] = []
    for c in components:
        if c["component"] == "VIX level":
            pts, desc = contrarian_vix_points(vix_level)
            out.append({"component": "VIX (contrarian)", "points": pts, "detail": f"VIX {vix_level:.1f}: {desc}"})
        elif c["component"] == "VIX term structure" and vix_level == vix_level and vix_level >= 30 and c["points"] < 0:
            # Backwardation is part of the same fear the contrarian reading rewards; do not count it twice.
            out.append({**c, "points": 0.0, "detail": c["detail"] + " (not penalized while fear is high)"})
        else:
            out.append(dict(c))
    return out


def compute_macro_snapshot(history: MacroHistory, asof: pd.Timestamp | str | None = None) -> MacroRegimeData:
    """Compute all macro indicators and the regime as of ``asof`` (default: latest)."""
    asof_ts = pd.Timestamp(asof).tz_localize(None) if asof is not None else None
    components: list[dict[str, Any]] = []
    missing: list[str] = []

    def add(name: str, points: float, detail: str) -> None:
        components.append({"component": name, "points": points, "detail": detail})

    # 1. SPY trend
    spy = _slice(history, "spy", asof_ts)
    spy_price = spy_sma_50 = spy_sma_200 = float("nan")
    if spy is not None and len(spy) >= 200:
        spy_price = float(spy.iloc[-1])
        spy_sma_50 = float(spy.tail(50).mean())
        spy_sma_200 = float(spy.tail(200).mean())
        add("SPY vs 200-day SMA", 2.0 if spy_price > spy_sma_200 else -2.0,
            f"SPY {spy_price:.2f} vs 200 SMA {spy_sma_200:.2f}")
        add("SPY vs 50-day SMA", 1.0 if spy_price > spy_sma_50 else -1.0,
            f"SPY {spy_price:.2f} vs 50 SMA {spy_sma_50:.2f}")
    else:
        missing.append("spy_trend")

    # 2. VIX level and percentile
    vix = _slice(history, "vix", asof_ts)
    vix_level = vix_sma_20 = vix_percentile = float("nan")
    if vix is not None and len(vix) >= 20:
        vix_level = float(vix.iloc[-1])
        vix_sma_20 = float(vix.tail(20).mean())
        vix_percentile = float((vix < vix_level).mean() * 100.0)
        if vix_level < 18:
            pts = 2.0
        elif vix_level < 22:
            pts = 0.5
        elif vix_level < 28:
            pts = -1.0
        elif vix_level < 35:
            pts = -2.0
        else:
            pts = -3.0
        add("VIX level", pts, f"VIX {vix_level:.1f} ({vix_percentile:.0f}th pct of 1Y)")
    else:
        missing.append("vix")

    # 3. VIX term structure
    vix3m = _slice(history, "vix3m", asof_ts)
    vix3m_level = vix_term_ratio = None
    if vix is not None and vix3m is not None:
        vix3m_level = float(vix3m.iloc[-1])
        vix_term_ratio = float(vix_level / vix3m_level) if vix3m_level > 0 else None
        if vix_term_ratio is not None:
            if vix_term_ratio > 1.1:
                pts = -1.5
            elif vix_term_ratio > 1.0:
                pts = -1.0
            elif vix_term_ratio < 0.9:
                pts = 0.5
            else:
                pts = 0.0
            add("VIX term structure", pts,
                f"VIX/VIX3M {vix_term_ratio:.2f} ({'backwardation' if vix_term_ratio > 1 else 'contango'})")
    else:
        missing.append("vix_term")

    # 4. Yield curve
    t10y2y = _slice(history, "t10y2y", asof_ts)
    t10y3m = _slice(history, "t10y3m", asof_ts)
    spread_2y = float(t10y2y.iloc[-1]) if t10y2y is not None else float("nan")
    spread_3m = float(t10y3m.iloc[-1]) if t10y3m is not None else float("nan")
    curve_uninverted = False
    if t10y2y is not None:
        if spread_2y < -0.10:
            status = "INVERTED"
        elif spread_2y < 0.25:
            status = "FLAT"
        else:
            status = "NORMAL"
        if spread_2y > 0.50:
            pts = 1.0
        elif spread_2y > 0.0:
            pts = 0.0
        elif spread_2y > -0.25:
            pts = -1.0
        else:
            pts = -2.0
        add("Yield curve 10Y-2Y", pts, f"{spread_2y:+.2f} pp ({status.lower()})")
        # Re-steepening after a deep inversion has historically preceded recessions.
        if float(t10y2y.min()) < -0.25 and spread_2y > 0.0:
            curve_uninverted = True
            add("Curve un-inversion", -0.5, f"1Y low {float(t10y2y.min()):+.2f} pp, now {spread_2y:+.2f} pp")
    else:
        status = "UNKNOWN"
        missing.append("fred_yields")

    # 5. Breadth (equal weight vs cap weight)
    breadth = _ratio_trend(_slice(history, "rsp", asof_ts), spy, 1.01, 0.99)
    if breadth is not None:
        breadth_ratio, breadth_sma, trend = breadth
        breadth_trend = {1: "IMPROVING", 0: "STABLE", -1: "DETERIORATING"}[trend]
        add("Breadth RSP/SPY", float(trend), f"{breadth_trend.lower()} ({breadth_ratio:.3f} vs 50d {breadth_sma:.3f})")
    else:
        breadth_ratio = breadth_sma = float("nan")
        breadth_trend = "UNKNOWN"
        missing.append("breadth")

    # 6. Credit (high yield vs investment grade)
    credit = _ratio_trend(_slice(history, "hyg", asof_ts), _slice(history, "lqd", asof_ts), 1.005, 0.995)
    if credit is not None:
        credit_ratio, credit_sma, trend = credit
        credit_trend = {1: "RISK_ON", 0: "NEUTRAL", -1: "RISK_OFF"}[trend]
        add("Credit HYG/LQD", {1: 1.0, 0: 0.0, -1: -1.5}[trend],
            f"{credit_trend.lower().replace('_', '-')} ({credit_ratio:.3f} vs 50d {credit_sma:.3f})")
    else:
        credit_ratio = credit_sma = float("nan")
        credit_trend = "UNKNOWN"
        missing.append("credit")

    # 7. 10Y yield momentum (fast rising yields tighten financial conditions)
    tnx = _slice(history, "tnx", asof_ts)
    ten_year = ten_year_change = None
    if tnx is not None and len(tnx) >= 61:
        ten_year = float(tnx.iloc[-1])
        ten_year_change = float(tnx.iloc[-1] - tnx.iloc[-61])
        pts = -0.5 if ten_year_change > 0.5 else 0.5 if ten_year_change < -0.5 else 0.0
        add("10Y yield 60d change", pts, f"{ten_year:.2f}% ({ten_year_change:+.2f} pp in 60d)")
    else:
        missing.append("ten_year")

    regime_score = float(sum(c["points"] for c in components))
    regime = classify_regime(regime_score)
    opportunity_components = _opportunity_components(components, vix_level)
    opportunity_score = float(sum(c["points"] for c in opportunity_components))

    # Core inputs decide the quality label: trend and VIX are essential.
    if not missing:
        data_quality = "FULL"
    elif "spy_trend" in missing and "vix" in missing:
        data_quality = "ESTIMATED"
    else:
        data_quality = "PARTIAL"

    as_of_candidates = [s.index[-1] for s in (spy, vix) if s is not None]
    as_of = max(as_of_candidates).strftime("%Y-%m-%d") if as_of_candidates else None

    def r(x: float, nd: int = 2) -> float:
        return round(x, nd) if x == x else x  # keep NaN as NaN

    return MacroRegimeData(
        spy_price=r(spy_price),
        spy_sma_50=r(spy_sma_50),
        spy_sma_200=r(spy_sma_200),
        vix_level=r(vix_level),
        vix_sma_20=r(vix_sma_20),
        vix_percentile=r(vix_percentile, 1),
        yield_spread_10y_2y=r(spread_2y),
        yield_spread_10y_3m=r(spread_3m),
        yield_curve_status=status,
        breadth_ratio=r(breadth_ratio, 4),
        breadth_ratio_sma_50=r(breadth_sma, 4),
        breadth_trend=breadth_trend,
        credit_ratio=r(credit_ratio, 4),
        credit_ratio_sma_50=r(credit_sma, 4),
        credit_trend=credit_trend,
        regime=regime,
        regime_score=round(regime_score, 1),
        data_quality=data_quality,
        vix3m_level=round(vix3m_level, 2) if vix3m_level is not None else None,
        vix_term_ratio=round(vix_term_ratio, 3) if vix_term_ratio is not None else None,
        ten_year_yield=round(ten_year, 2) if ten_year is not None else None,
        ten_year_change_60d=round(ten_year_change, 2) if ten_year_change is not None else None,
        curve_uninverted=curve_uninverted,
        opportunity_score=round(opportunity_score, 1),
        opportunity_components=opportunity_components,
        components=components,
        missing=missing,
        as_of=as_of,
    )


def fetch_macro_data(asof: pd.Timestamp | str | None = None) -> MacroRegimeData:
    """Load (cached) macro history and compute the regime snapshot as of ``asof``."""
    asof_ts = pd.Timestamp(asof) if asof is not None else None
    history = load_macro_history(history_years_needed(asof_ts))
    return compute_macro_snapshot(history, asof_ts)
