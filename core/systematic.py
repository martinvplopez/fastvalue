"""Trend and dip rules: hold an asset, or a leveraged version of it, while a rule says so, and T-bills otherwise.

The owner's ask (October 2026): momentum and trend indicators that protect against deep drawdowns, at least as
an alert; the two-engine leveraged Nasdaq strategy (a 3x fund held while the trend and the stress checks are
healthy, cash otherwise); what MSCI does in its momentum indexes; and buying the dip. Nothing here scores a
company or moves a weight (D-10, D-19): these are research tools and an alert (decision_log.md D-48).

Three studies:

- ``run_rule``: one rule read on one asset (the signal), held in that asset, in a synthetic 2x or 3x version
  of it, or in a real leveraged fund spliced onto the synthetic series before its first day. With the checks a
  backtest like this needs before anyone trusts it: both halves of the period, a sweep of the rule's settings
  (does it hold away from the default), walk-forward (settings chosen on the previous years, tested on the
  next one) and a paired block bootstrap (the same days in another order: how much was path luck).
- ``run_savings``: buying the dip with new money. A monthly saving invested at once, against kept in T-bills
  until the price is a set share below its high; the whole period, each half, and every 10-year window.
- ``run_rotation``: the MSCI momentum method across a fixed list of funds (the nine sector SPDRs since 1998 by
  default, so nothing is picked with hindsight): 6- and 12-month price momentum without the latest month, over
  3-year weekly volatility, z-scored; the top names weighted by their score; reviewed in February, May, August
  and November with a 50% turnover buffer, plus MSCI's extra review when volatility jumps.

``trend_check`` gives today's state of the alert rules on a short list of markets; the daily check
(``core/monitor.py``) alerts when one flips.

Conventions: a rule reads closes up to day t and trades at the close of t+1 (``execution="next_close"``;
``"same_close"`` is the optimistic version, trading at the close the rule read). Every change of exposure pays
``cost_bps``. Cash earns 3-month T-bills (FRED DTB3). Returns are total returns (dividends reinvested), except
before a fund's first day when the history is extended with its price index, which leaves dividends out.
Taxes are not modelled. Run ``python -m core.systematic --help``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd

import config.settings as settings
from analysis.trend_indicators import (
    TRADING_DAYS,
    drawdown_from_high,
    excess_momentum,
    msci_momentum,
    msci_scores,
    pb_ema_channel,
    realized_vol,
    sma,
    wilder_rsi,
)
from strategies.base import month_end_mask
from strategies.buy_the_dip import DipInUptrend, DrawdownDip
from strategies.trend_following import AbsoluteMomentum, MSCIMomentumTrend, PBEmaChannel, TrendFilter

logger = logging.getLogger(__name__)

Progress = Callable[[int, int, str], None]

COST_BPS = 10.0
SYNTHETIC_FEE = 0.0095  # yearly fee charged to a synthetic leveraged series (TQQQ's at launch; set per run)
# Borrowing for the synthetic series costs T-bills plus this spread a year. 0.5% matches TQQQ's return over its
# whole life (February 2010 to October 2026: 43.4% a year synthetic against 43.3% real; 0% overstated it by 1.5
# points a year). The fit drifts by about 0.8 points a year between the two halves of that period (D-48).
SYNTHETIC_SPREAD = 0.005
INDEX_PROXIES = {"QQQ": "^NDX", "SPY": "^GSPC", "DIA": "^DJI", "IWM": "^RUT"}
EXECUTION_LAG = {"next_close": 2, "same_close": 1}  # bars from the close a rule reads to the first return it earns
MAX_LEVERAGE = 3
BOOTSTRAP_PATHS = 1000
BOOTSTRAP_BLOCK = 20  # mean block length in days (about a month), so volatility clusters survive the reshuffle
BOOTSTRAP_SEED = 7
TRAIN_YEARS = 5
SAVINGS_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)
SAVINGS_HORIZON_YEARS = 10
SAVINGS_EDGE = 0.001  # a wait is "ahead" only by more than 0.1 point a year; less is noise in a money-weighted return
REVIEW_MONTHS = (2, 5, 8, 11)  # MSCI's index reviews
ADHOC_MIN_HISTORY = 36  # month-ends of volatility changes before the 95th percentile is trusted

_BOOL = {"type": "bool"}
_CHECK = {"type": "choice", "default": "daily", "label": "Decide",
          "choices": {"daily": "Every day", "monthly": "On month-ends only"}}

# A rule: its class, its fixed settings, the settings a user can change (``params``), and the grid its sweep and
# walk-forward use. The sweep varies the grid settings and keeps the user's choice for the others.
RULES: dict[str, dict[str, Any]] = {
    "trend_filter": {
        "label": "Price above its moving average",
        "class": TrendFilter, "fixed": {},
        "params": {
            "window": {"type": "int", "default": 200, "min": 20, "max": 400, "label": "Average (days)"},
            "buffer": {"type": "pct", "default": 0.0, "min": 0.0, "max": 0.10, "label": "Re-entry buffer"},
            "check": _CHECK,
        },
        "grid": {"window": [150, 175, 200, 225, 250], "buffer": [0.0, 0.01, 0.02, 0.03]},
        "description": "Held while the close is above its 200-day average, cash below it: the trend rule with the "
                       "longest record (Faber 2007; Gayed and Bilello 2016 on leveraged funds). A re-entry buffer "
                       "asks the close to clear the average by that much before buying back.",
    },
    "two_engine": {
        "label": "Two engines: trend and stress",
        "class": TrendFilter, "fixed": {},
        "params": {
            "window": {"type": "int", "default": 200, "min": 20, "max": 400, "label": "Slow engine: average (days)"},
            "buffer": {"type": "pct", "default": 0.02, "min": 0.0, "max": 0.10, "label": "Re-entry buffer"},
            "vol_cap": {"type": "pct", "default": 0.30, "min": 0.05, "max": 1.50, "label": "Fast engine: volatility cap"},
            "vol_window": {"type": "int", "default": 20, "min": 5, "max": 120, "label": "Volatility window (days)"},
            "vix_term": {**_BOOL, "default": True, "label": "Out while the VIX is above the 3-month VIX"},
        },
        "grid": {"window": [150, 175, 200, 225, 250], "vol_cap": [0.20, 0.25, 0.30, 0.35, 0.40]},
        "description": "The brochure's rule. Slow engine: the close against its 200-day average. Fast engine: the "
                       "last 20 days' volatility under a cap and the VIX not above the 3-month VIX (an inverted "
                       "curve; that check starts in 2006, when the 3-month VIX history begins). Out when either "
                       "flags trouble, back in only when both are calm and the close is 2% above the average.",
    },
    "pb_ema": {
        "label": "PB EMA channel",
        "class": PBEmaChannel, "fixed": {},
        "params": {"length": {"type": "int", "default": 200, "min": 20, "max": 400, "label": "EMA length"}},
        "grid": {"length": [100, 150, 200, 250, 300]},
        "description": "FFriZz's PlayBit EMA on TradingView: a channel between the 200-period EMA of the highs and "
                       "of the closes. In when the close is above both lines, out when it is below both; inside the "
                       "channel nothing changes, so the channel's width works as the buffer.",
    },
    "abs_momentum": {
        "label": "Absolute momentum, 12 months",
        "class": AbsoluteMomentum, "fixed": {},
        "params": {
            "months": {"type": "int", "default": 12, "min": 1, "max": 24, "label": "Look back (months)"},
            "skip": {"type": "int", "default": 1, "min": 0, "max": 3, "label": "Leave out the latest (months)"},
            "check": {**_CHECK, "default": "monthly"},
        },
        "grid": {"months": [3, 6, 9, 12], "skip": [0, 1]},
        "description": "Held while the last 12 months' return, without the latest month, beat T-bills over the "
                       "same months (time-series momentum: Moskowitz, Ooi and Pedersen 2012; Antonacci). Decided "
                       "at month-ends.",
    },
    "msci_momentum": {
        "label": "MSCI momentum on one asset",
        "class": MSCIMomentumTrend, "fixed": {},
        "params": {"check": {**_CHECK, "default": "monthly"}},
        "grid": {},
        "description": "MSCI's measure: the 6- and 12-month price momentum (each without the latest month, minus "
                       "T-bills) over 3-year weekly volatility, averaged; held while positive. On one asset only "
                       "its sign is left; MSCI uses it to rank and weight a universe, which the momentum rotation "
                       "study does.",
    },
    "dip_rsi2": {
        "label": "Short dip in an uptrend (RSI 2)",
        "class": DipInUptrend, "fixed": {},
        "params": {
            "entry": {"type": "float", "default": 10.0, "min": 1.0, "max": 40.0, "label": "Buy when RSI(2) is under"},
            "trend_window": {"type": "int", "default": 200, "min": 20, "max": 400, "label": "Only above the average (days)"},
            "exit_window": {"type": "int", "default": 5, "min": 2, "max": 50, "label": "Sell above the average (days)"},
            "rsi_period": {"type": "int", "default": 2, "min": 2, "max": 14, "label": "RSI period"},
        },
        "grid": {"entry": [5.0, 10.0, 15.0, 20.0], "exit_window": [3, 5, 10]},
        "description": "Connors' short-term pullback: buy when the 2-day RSI is washed out while the close is above "
                       "its 200-day average, sell when the close gets back above its 5-day average. Trades last days "
                       "to weeks, and the rule sits in cash most of the time.",
    },
    "dip_drawdown": {
        "label": "Buy a fall from the high",
        "class": DrawdownDip, "fixed": {},
        "params": {
            "dip": {"type": "pct", "default": 0.10, "min": 0.02, "max": 0.60, "label": "Buy this far below the high"},
            "lookback": {"type": "int", "default": 252, "min": 20, "max": 2520, "label": "High of the last (days)"},
        },
        "grid": {"dip": [0.05, 0.10, 0.15, 0.20, 0.30], "lookback": [126, 252, 504]},
        "description": "Buy the dip as a switch: cash until the close is 10% below its highest close of the last "
                       "year, then hold until it is back at that high. For buying dips with new savings instead, "
                       "run the savings study.",
    },
}

ROTATION_UNIVERSES: dict[str, dict[str, Any]] = {
    "sectors": {
        "label": "US sectors: the nine Select Sector SPDRs (since December 1998)",
        "tickers": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
    },
    "countries": {
        "label": "Countries: iShares MSCI country funds and SPY (since 1996)",
        "tickers": ["EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWL", "EWN", "EWP", "EWQ", "EWS", "EWU", "SPY"],
    },
}
ROTATION_BENCHMARK = "SPY"  # MSCI's reference index for the volatility trigger is MSCI World; SPY has the longer history


# ================================================================== settings


def validate_params(rule: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """The rule's settings: the defaults, overridden by ``params`` after type and range checks."""
    if rule not in RULES:
        raise ValueError(f"Unknown rule: {rule!r}")
    out: dict[str, Any] = {}
    for key, spec in RULES[rule]["params"].items():
        raw = (params or {}).get(key, spec["default"])
        kind = spec["type"]
        try:
            if kind == "int":
                value: Any = int(raw)
            elif kind in ("float", "pct"):
                value = float(raw)
            elif kind == "bool":
                value = raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
            else:
                value = str(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec.get('label', key)}: {raw!r} is not a valid value") from exc
        if kind == "choice" and value not in spec["choices"]:
            raise ValueError(f"{spec.get('label', key)} must be one of {', '.join(spec['choices'])}")
        if kind in ("int", "float", "pct") and not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{spec.get('label', key)} must be between {spec['min']:g} and {spec['max']:g}")
        out[key] = value
    return out


def build(rule: str, params: dict[str, Any]):
    spec = RULES[rule]
    return spec["class"](**{**spec["fixed"], **params})


def grid_combos(rule: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Every combination of the rule's grid, with the user's other settings, invalid ones left out."""
    grid = RULES[rule]["grid"]
    if not grid:
        return []
    keys = list(grid)
    combos = []
    for values in product(*(grid[k] for k in keys)):
        combo = {**params, **dict(zip(keys, values))}
        try:
            build(rule, combo)
        except ValueError:
            continue
        combos.append(combo)
    return combos


def describe_params(rule: str, params: dict[str, Any]) -> str:
    parts = []
    for key, spec in RULES[rule]["params"].items():
        v = params.get(key)
        if spec["type"] == "pct":
            parts.append(f"{spec['label']} {v:.0%}" if v * 100 == round(v * 100) else f"{spec['label']} {v:.1%}")
        elif spec["type"] == "bool":
            parts.append(f"{spec['label']}: {'yes' if v else 'no'}")
        elif spec["type"] == "choice":
            parts.append(f"{spec['label']}: {spec['choices'].get(v, v).lower()}")
        else:
            parts.append(f"{spec['label']} {v:g}")
    return "; ".join(parts)


# ================================================================== data


def _fred_long(series: str = "DTB3", start: str = "1960-01-01") -> pd.Series:
    """A FRED series since ``start``, cached for a day; a stale copy is used when FRED is unreachable."""
    from core.value_study import load_fred

    path = settings.DATA_DIR / "systematic" / f"fred_{series}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < 86400:
        return pd.read_parquet(path)["value"]
    try:
        s = load_fred(series, start)
        path.parent.mkdir(parents=True, exist_ok=True)
        s.rename("value").to_frame().to_parquet(path)
        return s
    except Exception as exc:  # noqa: BLE001 - cash then earns 0, and the result says so
        logger.warning("FRED %s unavailable (%s)", series, exc)
        return pd.read_parquet(path)["value"] if path.exists() else pd.Series(dtype=float)


def daily_rf(index: pd.DatetimeIndex) -> pd.Series:
    """Daily return of 3-month T-bills on ``index`` (0 where FRED has no rate)."""
    rate = _fred_long("DTB3")
    if rate.empty:
        return pd.Series(0.0, index=index)
    rate = rate[~rate.index.duplicated()].sort_index()
    return rate.reindex(rate.index.union(index)).ffill().reindex(index).fillna(0.0) / 100.0 / TRADING_DAYS


def load_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    from data.price_history import fetch_full_history

    return fetch_full_history(tickers)


def splice_index(fund: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """The fund's history extended backwards with its price index, scaled to meet the fund on its first day.

    Before that day ``adj_close`` follows the index's price alone, so dividends are left out there.
    """
    first = fund.index[0]
    anchor = index["close"].loc[:first]
    pre = index.loc[index.index < first]
    if pre.empty or anchor.empty:
        return fund.assign(source="fund")
    level = float(anchor.iloc[-1])
    k_price, k_total = float(fund["close"].iloc[0]) / level, float(fund["adj_close"].iloc[0]) / level
    head = pd.DataFrame({
        "open": pre["open"] * k_price, "high": pre["high"] * k_price, "low": pre["low"] * k_price,
        "close": pre["close"] * k_price, "adj_close": pre["close"] * k_total, "source": "index",
    }, index=pre.index)
    tail = fund[["open", "high", "low", "close", "adj_close"]].assign(source="fund")
    return pd.concat([head, tail])


@dataclass
class Market:
    """Everything a rule study reads: the signal asset's prices and what is held, on the signal's calendar."""

    signal: str
    frame: pd.DataFrame  # close, high, low, adj_close, rf, source, and vix and vix3m when loaded
    base_ret: pd.Series  # daily total return of the signal asset
    asset_ret: pd.Series  # daily total return of what the rule holds
    vehicle_label: str
    leveraged: bool
    notes: list[str] = field(default_factory=list)
    tracking: dict[str, Any] | None = None
    first_real: pd.Timestamp | None = None  # the real vehicle's first day, if one was given


def _vix(index: pd.DatetimeIndex) -> pd.DataFrame | None:
    from core import regime_study

    try:
        series = regime_study.load_long_history().series
    except Exception as exc:  # noqa: BLE001 - the check is then left out, and the result says so
        logger.warning("VIX history unavailable: %s", exc)
        return None
    if "vix" not in series:
        return None
    out = pd.DataFrame(index=index)
    for key in ("vix", "vix3m"):
        if key in series:
            s = series[key]
            out[key] = s.reindex(s.index.union(index)).ffill(limit=5).reindex(index)
    return out


def _tracking(real: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    both = pd.DataFrame({"real": real, "synthetic": synthetic.reindex(real.index)}).dropna()
    if len(both) < 60:
        return {}
    years = len(both) / TRADING_DAYS
    growth = (1 + both).prod()
    diff = both["real"] - both["synthetic"]
    return {
        "start": both.index[0].strftime("%Y-%m-%d"), "end": both.index[-1].strftime("%Y-%m-%d"),
        "cagr_real": float(growth["real"] ** (1 / years) - 1), "cagr_synthetic": float(growth["synthetic"] ** (1 / years) - 1),
        "correlation": float(both["real"].corr(both["synthetic"])),
        "tracking_error": float(diff.std() * math.sqrt(TRADING_DAYS)),
        "max_drawdown_real": _max_dd(both["real"]), "max_drawdown_synthetic": _max_dd(both["synthetic"]),
    }


def load_market(signal: str, extend: bool = False, leverage: int = 1, vehicle: str | None = None,
                fee: float = SYNTHETIC_FEE, spread: float = SYNTHETIC_SPREAD, need_vix: bool = False) -> Market:
    signal = signal.upper()
    vehicle = (vehicle or "").upper() or None
    if not 1 <= leverage <= MAX_LEVERAGE:
        raise ValueError(f"Leverage must be between 1 and {MAX_LEVERAGE}")
    proxy = INDEX_PROXIES.get(signal) if extend else None
    prices = load_prices([signal] + ([proxy] if proxy else []) + ([vehicle] if vehicle and vehicle != signal else []))
    if signal not in prices:
        raise LookupError(f"No price history for {signal}")
    notes: list[str] = []
    fund = prices[signal]
    if proxy and proxy in prices:
        frame = splice_index(fund, prices[proxy])
        if (frame["source"] == "index").any():
            notes.append(f"Before {fund.index[0]:%B %Y} the history is the {proxy} price index, scaled to meet {signal}: "
                         f"dividends are left out there, which flatters every rule that spends time in cash a little.")
    else:
        frame = fund[["open", "high", "low", "close", "adj_close"]].assign(source="fund")
        if extend:
            notes.append(f"No index history is known for {signal}, so the test starts on its first day ({fund.index[0]:%B %Y}).")
    frame = frame[~frame.index.duplicated()].sort_index()
    frame["rf"] = daily_rf(frame.index)
    if not frame["rf"].any():
        notes.append("T-bill rates could not be loaded, so cash earns 0%.")
    elif frame["rf"].ne(0).idxmax() > frame.index[0] + pd.Timedelta(days=60):
        notes.append(f"T-bill rates start in {frame['rf'].ne(0).idxmax():%B %Y}; before that cash earns 0% here.")
    if need_vix:
        vix = _vix(frame.index)
        if vix is None or "vix3m" not in vix:
            notes.append("The VIX history could not be loaded, so the VIX curve check is left out.")
        else:
            frame = frame.join(vix)
            first = vix["vix3m"].first_valid_index()
            if first is not None and first > frame.index[0]:
                notes.append(f"The VIX curve check starts in {first:%B %Y}, when the 3-month VIX begins; before that only "
                             f"the trend and the volatility cap decide.")

    base = frame["adj_close"].pct_change().fillna(0.0)
    synthetic = base
    if leverage > 1:
        synthetic = leverage * base - (leverage - 1) * (frame["rf"] + spread / TRADING_DAYS) - fee / TRADING_DAYS
    asset, tracking, first_real = synthetic, None, None
    if vehicle and vehicle != signal:
        if vehicle not in prices:
            raise LookupError(f"No price history for {vehicle}")
        v = prices[vehicle]["adj_close"]
        v = v.reindex(v.index.union(frame.index)).ffill(limit=5).reindex(frame.index)
        real = v.pct_change()
        first_real = v.first_valid_index()
        if first_real is None:
            raise LookupError(f"{vehicle} has no prices on {signal}'s trading days")
        real = real.loc[real.index > first_real]
        asset = synthetic.copy()
        asset.loc[real.index] = real.fillna(0.0)
        tracking = _tracking(real, synthetic)
        if leverage > 1:
            notes.append(f"{vehicle} from {first_real:%B %Y}; before that a synthetic {leverage}x {signal}: {leverage} times the "
                         f"daily return, minus {leverage - 1} times the T-bill rate plus {spread:.2%} for the borrowing, minus "
                         f"a {fee:.2%} yearly fee.")
        else:
            notes.append(f"{vehicle} from {first_real:%B %Y}; before that {signal} itself.")
    elif leverage > 1:
        notes.append(f"A synthetic {leverage}x {signal}: {leverage} times the daily return, minus {leverage - 1} times the "
                     f"T-bill rate plus {spread:.2%} for the borrowing, minus a {fee:.2%} yearly fee. "
                     f"Rebalanced daily like a real leveraged fund, so it decays in choppy markets.")
    if leverage > 1 or vehicle:
        held = vehicle if vehicle and leverage == 1 else (f"{leverage}x {signal}" + (f" ({vehicle})" if vehicle else ""))
    else:
        held = signal
    return Market(signal, frame, base, asset, held, leverage > 1, notes, tracking, first_real)


# ================================================================== engine


def simulate(exposure: pd.Series, asset_ret: pd.Series, rf: pd.Series, lag: int, cost_bps: float) -> pd.DataFrame:
    """Daily ``held`` share and ``ret`` (net of costs) for an exposure decided at each close."""
    held = exposure.shift(lag).fillna(0.0)
    turn = held.diff().abs().fillna(held.iloc[0] if len(held) else 0.0)
    ret = held * asset_ret + (1.0 - held) * rf - turn * cost_bps / 10_000.0
    return pd.DataFrame({"held": held, "ret": ret})


def _max_dd(returns: pd.Series) -> float:
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")


def perf(returns: pd.Series, rf: pd.Series | None = None, periods: int = TRADING_DAYS) -> dict[str, float]:
    """Total return, CAGR, volatility, Sharpe (over T-bills), Sortino, max drawdown, Calmar, worst month and year."""
    r = returns.dropna()
    nan = float("nan")
    if len(r) < 3:
        return {k: nan for k in ("total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar",
                                 "worst_month", "worst_year", "best_year", "underwater_days")}
    rf = (rf if rf is not None else pd.Series(0.0, index=r.index)).reindex(r.index).fillna(0.0)
    eq = (1.0 + r).cumprod()
    years = len(r) / periods
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
    sd = float(r.std())
    excess = r - rf
    downside = float(np.sqrt((np.minimum(excess, 0.0) ** 2).mean()))
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    under = (dd < 0).astype(int)
    runs = under.groupby((under == 0).cumsum()).sum()
    monthly = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0 if periods == TRADING_DAYS else r
    yearly = (1.0 + r).groupby(r.index.year).prod() - 1.0
    full_years = yearly.iloc[1:-1] if len(yearly) > 2 else yearly
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "volatility": sd * math.sqrt(periods),
        "sharpe": float(excess.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0,
        "sortino": float(excess.mean() / downside * math.sqrt(periods)) if downside > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < -0.001 else nan,
        "worst_month": float(monthly.min()),
        "worst_year": float(full_years.min()) if len(full_years) else nan,
        "best_year": float(full_years.max()) if len(full_years) else nan,
        "underwater_days": int(runs.max()) if len(runs) else 0,
    }


def trade_list(held: pd.Series, ret: pd.Series, cost_bps: float) -> list[dict[str, Any]]:
    """Round trips: each stretch in the market with its return (the exit cost included)."""
    inside = held.gt(0.5).to_numpy()
    if not inside.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
    out = []
    growth = (1.0 + ret.fillna(0.0)).to_numpy()
    for a, b in zip(edges[::2], edges[1::2]):
        closed = b < len(inside)
        r = float(np.prod(growth[a:b]) * (1.0 - cost_bps / 10_000.0 if closed else 1.0) - 1.0)
        out.append({"entry": held.index[a].strftime("%Y-%m-%d"), "exit": held.index[b].strftime("%Y-%m-%d") if closed else None,
                    "days": int(b - a), "return": r, "open": not closed})
    return out


def trade_stats(trades: list[dict[str, Any]], years: float) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "per_year": 0.0}
    r = np.array([t["return"] for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    win_rate = len(wins) / len(r)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    return {
        "trades": len(r), "per_year": len(r) / max(years, 1e-9), "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "expectancy": win_rate * avg_win + (1 - win_rate) * avg_loss,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        "avg_days": float(np.mean([t["days"] for t in trades])),
        "best": float(r.max()), "worst": float(r.min()),
    }


def attribution(held: pd.Series, asset_ret: pd.Series) -> dict[str, float]:
    """How what the rule holds did while it held it, and while it sat in cash (annualised)."""
    def ann(x: pd.Series) -> float:
        return float((1.0 + x).prod() ** (TRADING_DAYS / len(x)) - 1.0) if len(x) > 20 else float("nan")

    return {"while_in": ann(asset_ret[held > 0.5]), "while_out": ann(asset_ret[held <= 0.5])}


def _halves(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    mid = len(index) // 2
    return index[:mid], index[mid:]


def _row(key: str, label: str, kind: str, ret: pd.Series, rf: pd.Series, held: pd.Series | None = None,
         cost_bps: float = 0.0) -> dict[str, Any]:
    first, second = _halves(ret.index)
    row = {"key": key, "label": label, "kind": kind, "stats": perf(ret, rf),
           "first_half": perf(ret.loc[first], rf), "second_half": perf(ret.loc[second], rf)}
    if held is not None:
        years = len(ret) / TRADING_DAYS
        trades = trade_list(held, ret, cost_bps)
        row.update({"exposure": float(held.mean()), "trade_stats": trade_stats(trades, years)})
    return row


def _sample(series: pd.Series, digits: int = 4) -> list[dict[str, Any]]:
    """A weekly series for charts (the last value of each week): a 40-year daily line is too heavy to draw."""
    s = series.dropna()
    s = s.groupby(s.index.to_period("W-FRI")).tail(1)
    return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), digits)} for d, v in s.items()]


def _equity(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod() * 100.0


def _monthly_table(ret: pd.Series) -> dict[str, Any]:
    m = (1.0 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1.0
    years = sorted({int(y) for y, _ in m.index})
    cells = {str(y): [None] * 12 for y in years}
    for (y, mo), v in m.items():
        cells[str(int(y))][int(mo) - 1] = float(v)
    return {"years": [str(y) for y in years], "cells": cells}


# ================================================================== robustness


def run_sweep(rule: str, params: dict[str, Any], market: Market, window: pd.DatetimeIndex, lag: int,
              cost_bps: float, base: pd.Series) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """Every grid setting over the same window, each against buy and hold of what the rule holds."""
    rf = market.frame["rf"].loc[window]
    first, second = _halves(window)
    ref = {"whole": perf(base, rf), "first": perf(base.loc[first], rf), "second": perf(base.loc[second], rf)}
    rows, sims = [], {}
    for combo in grid_combos(rule, params):
        sim = simulate(build(rule, combo).exposure(market.frame), market.asset_ret, market.frame["rf"], lag, cost_bps)
        sim = sim.loc[window]
        key = json.dumps({k: combo[k] for k in RULES[rule]["grid"]}, sort_keys=True)
        sims[key] = sim
        s, a, b = perf(sim["ret"], rf), perf(sim["ret"].loc[first], rf), perf(sim["ret"].loc[second], rf)
        rows.append({
            "key": key, "values": {k: combo[k] for k in RULES[rule]["grid"]},
            "is_default": all(combo[k] == params[k] for k in RULES[rule]["grid"]),
            "stats": s, "first_half": a, "second_half": b, "exposure": float(sim["held"].mean()),
            "sharpe_both": a["sharpe"] > ref["first"]["sharpe"] and b["sharpe"] > ref["second"]["sharpe"],
            "drawdown_both": a["max_drawdown"] > ref["first"]["max_drawdown"] and b["max_drawdown"] > ref["second"]["max_drawdown"],
            "cagr_both": a["cagr"] > ref["first"]["cagr"] and b["cagr"] > ref["second"]["cagr"],
        })
    return rows, sims


def sweep_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    cagr = np.array([r["stats"]["cagr"] for r in rows])
    dd = np.array([r["stats"]["max_drawdown"] for r in rows])
    sharpe = np.array([r["stats"]["sharpe"] for r in rows])
    default = next((r for r in rows if r["is_default"]), None)
    rank = (1 + int((sharpe > default["stats"]["sharpe"]).sum())) if default else None
    return {
        "settings": len(rows),
        "sharpe_both": int(sum(r["sharpe_both"] for r in rows)),
        "drawdown_both": int(sum(r["drawdown_both"] for r in rows)),
        "cagr_both": int(sum(r["cagr_both"] for r in rows)),
        "cagr_range": [float(cagr.min()), float(np.median(cagr)), float(cagr.max())],
        "drawdown_range": [float(dd.min()), float(np.median(dd)), float(dd.max())],
        "sharpe_range": [float(sharpe.min()), float(np.median(sharpe)), float(sharpe.max())],
        "default_rank": rank,
    }


def walk_forward(rule: str, rows: list[dict[str, Any]], sims: dict[str, pd.DataFrame], market: Market,
                 window: pd.DatetimeIndex, cost_bps: float, base: pd.Series, train_years: int = TRAIN_YEARS) -> dict[str, Any]:
    """Each year, the grid setting with the best Sharpe over the previous ``train_years`` years, held for that year."""
    if len(rows) < 2:
        return {}
    rf = market.frame["rf"]
    years = sorted(set(window.year))
    first_test = years[0] + train_years + (1 if window[0] > pd.Timestamp(year=years[0], month=1, day=10) else 0)
    test_years = [y for y in years if y >= first_test]
    if len(test_years) < 3:
        return {}
    default_key = next((r["key"] for r in rows if r["is_default"]), rows[0]["key"])
    picks, held_parts = [], []
    for y in test_years:
        train = window[(window.year >= y - train_years) & (window.year < y)]
        test = window[window.year == y]
        if len(train) < TRADING_DAYS or not len(test):
            continue
        scores = {k: perf(s["ret"].loc[train], rf.loc[train])["sharpe"] for k, s in sims.items()}
        best = max(scores, key=lambda k: (scores[k], k == default_key))
        held_parts.append(sims[best]["held"].loc[test])
        picks.append({
            "year": int(y), "values": json.loads(best), "train_sharpe": scores[best],
            "test_return": float((1.0 + sims[best]["ret"].loc[test]).prod() - 1.0),
            "default_return": float((1.0 + sims[default_key]["ret"].loc[test]).prod() - 1.0),
            "buy_hold_return": float((1.0 + base.loc[test]).prod() - 1.0),
        })
    if not held_parts:
        return {}
    held = pd.concat(held_parts)
    span = held.index
    turn = held.diff().abs().fillna(0.0)  # a change of setting that changes the position is a trade too
    ret = held * market.asset_ret.loc[span] + (1.0 - held) * rf.loc[span] - turn * cost_bps / 10_000.0
    distinct = len({json.dumps(p["values"], sort_keys=True) for p in picks})
    return {
        "train_years": train_years, "start": span[0].strftime("%Y-%m-%d"), "end": span[-1].strftime("%Y-%m-%d"),
        "selection": "best Sharpe over the training years",
        "walk_forward": perf(ret, rf.loc[span]), "default": perf(sims[default_key]["ret"].loc[span], rf.loc[span]),
        "buy_hold": perf(base.loc[span], rf.loc[span]), "picks": picks, "distinct_settings": distinct,
    }


def block_bootstrap(strategy: pd.Series, reference: pd.Series, paths: int = BOOTSTRAP_PATHS,
                    block: int = BOOTSTRAP_BLOCK, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Stationary block bootstrap of the same days for both series (paired): CAGR and max drawdown percentiles.

    Each path draws blocks of consecutive days (mean length ``block``) from the period, wrapping at the end, and
    applies the same draw to the rule and to buy and hold, so the comparison stays paired.
    """
    a = np.log1p(strategy.fillna(0.0).to_numpy())
    b = np.log1p(reference.fillna(0.0).to_numpy())
    n = len(a)
    if n < 2 * block:
        return {}
    rng = np.random.default_rng(seed)
    years = n / TRADING_DAYS
    out = {"cagr_a": [], "cagr_b": [], "dd_a": [], "dd_b": []}
    pos = np.arange(n)
    for chunk in range(0, paths, 100):
        m = min(100, paths - chunk)
        new = rng.random((m, n)) < 1.0 / block
        new[:, 0] = True
        starts = rng.integers(0, n, size=(m, n))
        first_pos = np.where(new, pos, 0)
        first_pos = np.maximum.accumulate(first_pos, axis=1)  # position where the current block began
        start_of_block = np.take_along_axis(np.where(new, starts, 0), first_pos, axis=1)
        idx = (start_of_block + (pos - first_pos)) % n
        for key, series in (("a", a), ("b", b)):
            logs = np.cumsum(series[idx], axis=1)
            out[f"cagr_{key}"].append(np.expm1(logs[:, -1] / years))
            peak = np.maximum.accumulate(np.maximum(logs, 0.0), axis=1)
            out[f"dd_{key}"].append(np.expm1(logs - peak).min(axis=1))
    res = {k: np.concatenate(v) for k, v in out.items()}
    pct = (5, 25, 50, 75, 95)

    def q(x: np.ndarray) -> dict[str, float]:
        return {f"p{p}": float(np.percentile(x, p)) for p in pct}

    return {
        "paths": paths, "block_days": block,
        "rule": {"cagr": q(res["cagr_a"]), "max_drawdown": q(res["dd_a"])},
        "buy_hold": {"cagr": q(res["cagr_b"]), "max_drawdown": q(res["dd_b"])},
        "p_shallower_drawdown": float((res["dd_a"] > res["dd_b"]).mean()),
        "p_higher_cagr": float((res["cagr_a"] > res["cagr_b"]).mean()),
    }


# ================================================================== the rule study


def _fall(x: float) -> str:
    """A worst fall in words: one decimal near a total loss (3x Nasdaq lost 99.9% in 2000 to 2002, not 100%)."""
    return f"{x:.1%}" if x < -0.99 else f"{x:.0%}"


def _verdict(main: dict[str, Any], bh: dict[str, Any], summary: dict[str, Any], wf: dict[str, Any],
             held_label: str) -> dict[str, Any]:
    halves = ("first_half", "second_half")
    dd_both = all(main[h]["max_drawdown"] > bh[h]["max_drawdown"] for h in halves)
    sharpe_both = all(main[h]["sharpe"] > bh[h]["sharpe"] for h in halves)
    cagr_both = all(main[h]["cagr"] > bh[h]["cagr"] for h in halves)
    s, b = main["stats"], bh["stats"]
    # Robust only if most settings around the default also beat holding on Sharpe in both halves.
    fragile = bool(summary) and summary["sharpe_both"] * 2 < summary["settings"]
    if dd_both and sharpe_both and fragile:
        level, title = "mixed", "Better in both halves at this setting, but not at most settings around it"
    elif dd_both and sharpe_both and cagr_both:
        level, title = "strong", "Smaller falls and a higher return, in both halves"
    elif dd_both and sharpe_both:
        level, title = "strong", "Smaller falls and better risk-adjusted returns in both halves"
    elif dd_both:
        level, title = "mixed", "Smaller falls in both halves, but not a better trade-off in both"
    else:
        level, title = "weak", "No consistent edge over holding throughout"
    lines = [
        f"Worst fall {_fall(s['max_drawdown'])} against {_fall(b['max_drawdown'])} holding {held_label} throughout; "
        f"{s['cagr']:.1%} a year against {b['cagr']:.1%}, Sharpe {s['sharpe']:.2f} against {b['sharpe']:.2f}.",
        "By half: " + "; ".join(
            f"{name} {main[h]['cagr']:.1%} a year against {bh[h]['cagr']:.1%}, worst fall {_fall(main[h]['max_drawdown'])} "
            f"against {_fall(bh[h]['max_drawdown'])}" for name, h in (("first", "first_half"), ("second", "second_half"))) + ".",
    ]
    if summary:
        lines.append(f"Settings around the default: {summary['sharpe_both']} of {summary['settings']} had a higher Sharpe "
                     f"than holding throughout in both halves, {summary['drawdown_both']} a smaller worst fall in both; "
                     f"returns ranged from {summary['cagr_range'][0]:.1%} to {summary['cagr_range'][2]:.1%} a year.")
    if wf:
        lines.append(f"Walk-forward from {wf['start'][:4]} (the setting with the best Sharpe over the previous "
                     f"{wf['train_years']} years, each year): {wf['walk_forward']['cagr']:.1%} a year, worst fall "
                     f"{_fall(wf['walk_forward']['max_drawdown'])}; the default {wf['default']['cagr']:.1%} and "
                     f"{_fall(wf['default']['max_drawdown'])}; holding throughout {wf['buy_hold']['cagr']:.1%} and "
                     f"{_fall(wf['buy_hold']['max_drawdown'])}.")
    return {"level": level, "title": title, "lines": lines}


def run_rule(rule: str = "two_engine", params: dict[str, Any] | None = None, signal: str = "QQQ", leverage: int = 1,
             vehicle: str | None = None, extend: bool = False, execution: str = "next_close",
             cost_bps: float = COST_BPS, fee: float = SYNTHETIC_FEE, spread: float = SYNTHETIC_SPREAD, start: str | None = None,
             end: str | None = None, robustness: bool = True, train_years: int = TRAIN_YEARS,
             progress: Progress | None = None) -> dict[str, Any]:
    """One rule on one signal asset, against holding the same thing throughout, with the robustness checks."""
    progress = progress or (lambda *_: None)
    if execution not in EXECUTION_LAG:
        raise ValueError(f"execution must be one of {', '.join(EXECUTION_LAG)}")
    if not 0 <= cost_bps <= 200:
        raise ValueError("Cost per trade must be between 0 and 200 bps")
    p = validate_params(rule, params)
    lag = EXECUTION_LAG[execution]
    progress(0, 5, "Loading prices")
    market = load_market(signal, extend, int(leverage), vehicle, fee, spread, need_vix=bool(p.get("vix_term")))
    frame = market.frame
    strategy = build(rule, p)
    combos = grid_combos(rule, p) if robustness else []
    warm = max([strategy.warmup] + [build(rule, c).warmup for c in combos]) + lag + 1
    if len(frame) <= warm + TRADING_DAYS:
        raise ValueError(f"Not enough history for this rule: {len(frame)} days, it needs more than {warm + TRADING_DAYS}")
    index = frame.index[warm:]
    if start == "vehicle" and market.first_real is not None:
        index = index[index > market.first_real]
    elif start:
        index = index[index >= pd.Timestamp(start)]
    if end:
        index = index[index <= pd.Timestamp(end)]
    if len(index) < TRADING_DAYS:
        raise ValueError("Less than a year of history in the chosen period")

    progress(1, 5, "Running the rule")
    rf = frame["rf"].loc[index]
    exposure = strategy.exposure(frame)
    sim = simulate(exposure, market.asset_ret, frame["rf"], lag, cost_bps).loc[index]
    held_ret = market.asset_ret.loc[index]
    label = RULES[rule]["label"]
    rows = [
        _row("rule", f"{label}, {market.vehicle_label}", "rule", sim["ret"], rf, sim["held"], cost_bps),
        _row("buy_hold", f"{market.vehicle_label} held throughout", "benchmark", held_ret, rf),
    ]
    equity = {"rule": _sample(_equity(sim["ret"])), "buy_hold": _sample(_equity(held_ret))}
    drawdowns = {}
    for key, r in (("rule", sim["ret"]), ("buy_hold", held_ret)):
        eq = _equity(r)
        drawdowns[key] = _sample(eq / eq.cummax() - 1.0)
    if market.leveraged or market.vehicle_label != market.signal:
        plain = simulate(exposure, market.base_ret, frame["rf"], lag, cost_bps).loc[index]
        rows.append(_row("rule_1x", f"{label}, {market.signal}", "rule", plain["ret"], rf, plain["held"], cost_bps))
        rows.append(_row("signal_hold", f"{market.signal} held throughout", "benchmark", market.base_ret.loc[index], rf))
        equity["rule_1x"] = _sample(_equity(plain["ret"]))
        equity["signal_hold"] = _sample(_equity(market.base_ret.loc[index]))
    rows.append(_row("cash", "T-bills", "cash", rf, rf))
    equity["cash"] = _sample(_equity(rf))
    main, bh = rows[0], rows[1]
    main["attribution"] = attribution(sim["held"], held_ret)
    first, second = _halves(index)

    sweep_rows, summary, wf, boot = [], {}, {}, {}
    if robustness:
        progress(2, 5, "Sweeping the settings")
        sweep_rows, sims = run_sweep(rule, p, market, index, lag, cost_bps, held_ret)
        summary = sweep_summary(sweep_rows)
        progress(3, 5, "Walk-forward")
        wf = walk_forward(rule, sweep_rows, sims, market, index, cost_bps, held_ret, train_years)
        progress(4, 5, "Reshuffling the history")
        boot = block_bootstrap(sim["ret"], held_ret)

    trades = trade_list(sim["held"], sim["ret"], cost_bps)
    yearly_rule = (1.0 + sim["ret"]).groupby(index.year).prod() - 1.0
    yearly_bh = (1.0 + held_ret).groupby(index.year).prod() - 1.0
    price = frame["close"].loc[index]
    overlays = getattr(strategy, "overlays", None)
    lines = {name: _sample(s.loc[index], 2) for name, s in (overlays(frame) if overlays else {}).items()}
    notes = list(market.notes)
    from_index = frame["source"].loc[index] == "index"
    if from_index.any():
        out_share = 1.0 - float(sim["held"][from_index].mean())
        notes.append(f"Over the index stretch ({index[0]:%Y} to {index[from_index.to_numpy()][-1]:%Y}) the rule sat in cash "
                     f"{out_share:.0%} of the days, earning T-bills while holding earned no dividends: against holding, the "
                     f"rule is flattered there by about the index's dividend yield times {out_share:.0%}. Compare returns on the "
                     f"fund's own history; the worst falls are fair either way.")
    notes.append(f"Decided at each close and traded at the {'next' if lag == 2 else 'same'} close; {cost_bps:g} bps per "
                 f"change of position; cash earns 3-month T-bills. Taxes are not modelled: every exit realises gains.")
    if market.leveraged:
        notes.append("Leveraged funds reset daily: overnight gaps cannot be exited, and a fall of a third in one day "
                     "would wipe a 3x fund out. US leveraged ETFs such as TQQQ are not sold to private investors in the "
                     "EU; the European 3x Nasdaq products charge more.")
    if rule.startswith("dip"):
        notes.append("A dip rule that sells on the rebound spends most of the time in cash; compare it with the savings "
                     "study, which asks the question with new money.")
    progress(5, 5, "Done")
    return _clean({
        "study": "rule", "rule": rule, "label": label, "params": p, "params_text": describe_params(rule, p),
        "signal": market.signal, "held": market.vehicle_label, "leverage": int(leverage), "vehicle": vehicle,
        "execution": execution, "cost_bps": cost_bps,
        "period": {"start": index[0].strftime("%Y-%m-%d"), "end": index[-1].strftime("%Y-%m-%d"),
                   "split": second[0].strftime("%Y-%m-%d"), "years": len(index) / TRADING_DAYS,
                   "index_until": (frame.index[frame["source"] == "fund"][0].strftime("%Y-%m-%d")
                                   if (frame["source"].loc[index] == "index").any() else None)},
        "verdict": _verdict(main, bh, summary, wf, market.vehicle_label),
        "rows": rows, "equity": equity, "drawdowns": drawdowns,
        "price": {"close": _sample(price, 2), "lines": lines, "held": _sample(sim["held"], 2)},
        "yearly": [{"year": int(y), "rule": float(yearly_rule[y]), "buy_hold": float(yearly_bh[y])} for y in yearly_rule.index],
        "monthly": _monthly_table(sim["ret"]),
        "trades": trades[-60:], "trade_count": len(trades),
        "sweep": {"grid": RULES[rule]["grid"], "rows": sweep_rows, "summary": summary},
        "walk_forward": wf, "bootstrap": boot, "tracking": market.tracking,
        "now": _now_state(rule, strategy, frame),
        "notes": notes,
    })


def _now_state(rule: str, strategy: Any, frame: pd.DataFrame) -> dict[str, Any]:
    exposure = strategy.exposure(frame)
    state = int(exposure.iloc[-1] > 0.5)
    changes = exposure.index[exposure.diff().fillna(0.0) != 0]
    return {"date": frame.index[-1].strftime("%Y-%m-%d"), "in_market": bool(state),
            "since": changes[-1].strftime("%Y-%m-%d") if len(changes) else None,
            "reason": explain_state(rule, strategy, frame)}


def explain_state(rule: str, strategy: Any, frame: pd.DataFrame) -> str:
    """One sentence on why the rule is in or out today."""
    close = frame["close"]
    last = float(close.iloc[-1])
    if isinstance(strategy, TrendFilter):
        avg = float(sma(close, strategy.window).iloc[-1])
        parts = [f"close {last / avg - 1:+.1%} against its {strategy.window}-day average"]
        if strategy.vol_cap > 0:
            vol = float(realized_vol(close, strategy.vol_window).iloc[-1])
            parts.append(f"{strategy.vol_window}-day volatility {vol:.0%} (cap {strategy.vol_cap:.0%})")
        if strategy.vix_term and {"vix", "vix3m"} <= set(frame.columns) and pd.notna(frame["vix3m"].iloc[-1]):
            parts.append(f"VIX {frame['vix'].iloc[-1]:.1f} against the 3-month VIX {frame['vix3m'].iloc[-1]:.1f}")
        return "; ".join(parts) + "."
    if isinstance(strategy, PBEmaChannel):
        ch = pb_ema_channel(frame["high"], close, strategy.length).iloc[-1]
        where = {1: "above both lines", -1: "below both lines", 0: "inside the channel"}[int(ch["state"])]
        return f"Close {where} (EMA of highs {ch['ema_high']:.2f}, of closes {ch['ema_close']:.2f})."
    if isinstance(strategy, AbsoluteMomentum):
        v = float(strategy.value(frame).iloc[-1])
        return f"{strategy.months}-month return without the latest {strategy.skip}, minus T-bills: {v:+.1%}."
    if isinstance(strategy, MSCIMomentumTrend):
        v = float(strategy.value(frame).iloc[-1])
        return f"MSCI momentum (6- and 12-month excess return over volatility, averaged): {v:+.2f}."
    if isinstance(strategy, DipInUptrend):
        rsi = float(wilder_rsi(close, strategy.rsi_period).iloc[-1])
        return f"RSI({strategy.rsi_period}) {rsi:.0f}; close {last / float(sma(close, strategy.trend_window).iloc[-1]) - 1:+.1%} against its {strategy.trend_window}-day average."
    if isinstance(strategy, DrawdownDip):
        dd = float(drawdown_from_high(close, strategy.lookback).iloc[-1])
        return f"Close {dd:.1%} from its {strategy.lookback}-day high (buys at -{strategy.dip:.0%})."
    return ""


# ================================================================== buying the dip with savings


def _xirr(values: np.ndarray, years_held: np.ndarray, final: float) -> float:
    """Yearly money-weighted return of deposits ``values`` held ``years_held`` years that ended at ``final``."""
    if final <= 0 or not len(values):
        return float("nan")
    lo, hi = -0.99, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        fv = float((values * (1 + mid) ** years_held).sum())
        lo, hi = (mid, hi) if fv < final else (lo, mid)
    return (lo + hi) / 2


def _savings_values(dep: np.ndarray, invest_at: np.ndarray, asset: np.ndarray, cash: np.ndarray, end: int) -> np.ndarray:
    """Value at ``end`` of 1 deposited at each position in ``dep``, waiting in cash until ``invest_at``."""
    in_time = invest_at <= end
    tau = np.where(in_time, invest_at, end)
    return np.where(in_time, cash[tau] / cash[dep] * asset[end] / asset[tau], cash[end] / cash[dep])


def run_savings(signal: str = "SPY", extend: bool = False, leverage: int = 1, vehicle: str | None = None,
                fee: float = SYNTHETIC_FEE, spread: float = SYNTHETIC_SPREAD, thresholds: tuple[float, ...] = SAVINGS_THRESHOLDS,
                lookback: int | None = None, horizon_years: int = SAVINGS_HORIZON_YEARS, start: str | None = None,
                progress: Progress | None = None) -> dict[str, Any]:
    """The same saving each month: invested at once, or kept in T-bills until the price is X% below its high.

    A dip is read at a close and the waiting cash is invested at the next close. The deposit of each month
    arrives on its first trading day; invested at once it buys that close.
    """
    progress = progress or (lambda *_: None)
    thresholds = tuple(sorted({round(float(t), 4) for t in thresholds}))
    if not thresholds or any(not 0 < t < 0.9 for t in thresholds):
        raise ValueError("Dip thresholds must be between 0 and 90%")
    progress(0, 3, "Loading prices")
    market = load_market(signal, extend, int(leverage), vehicle, fee, spread)
    frame = market.frame
    index = frame.index if not start else frame.index[frame.index >= pd.Timestamp(start)]
    if len(index) < 3 * TRADING_DAYS:
        raise ValueError("Need at least three years of history")
    asset = (1.0 + market.asset_ret.loc[index].fillna(0.0)).cumprod().to_numpy()
    cash = (1.0 + frame["rf"].loc[index]).cumprod().to_numpy()
    close = frame["close"]
    dd = drawdown_from_high(close, lookback).loc[index].to_numpy()  # the high includes the years before ``start``
    month = index.to_period("M")
    dep = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])  # first trading day of each month
    n = len(index)
    strategies = [("now", "Invest at once", None)] + [(f"dip_{t:g}", f"Wait for {t:.0%} below the high", t) for t in thresholds]
    invest_at: dict[str, np.ndarray] = {}
    for key, _, t in strategies:
        if t is None:
            invest_at[key] = dep.copy()
        else:
            trigger = np.flatnonzero(dd <= -t)
            if not len(trigger):  # never that far below the high: every deposit waits to the end
                invest_at[key] = np.full(len(dep), n + 1)
                continue
            k = np.searchsorted(trigger, dep)
            invest_at[key] = np.where(k < len(trigger), trigger[np.minimum(k, len(trigger) - 1)] + 1, n + 1)

    progress(1, 3, "Whole period and halves")
    def window_result(key: str, mask: np.ndarray, end: int) -> dict[str, Any]:
        d = dep[mask]
        values = _savings_values(d, invest_at[key][mask], asset, cash, end)
        held_years = (index[end] - index[d]).days / 365.25
        waited = np.minimum(invest_at[key][mask], end + 1) - d
        return {"multiple": float(values.sum() / len(d)), "irr": _xirr(np.ones(len(d)), np.asarray(held_years), float(values.sum())),
                "deposits": int(len(d)), "still_waiting": float(np.mean(invest_at[key][mask] > end)),
                "longest_wait_months": float(waited.max() / 21.0), "average_wait_months": float(waited.mean() / 21.0)}

    mid = n // 2
    whole_mask = np.ones(len(dep), bool)
    first_mask, second_mask = dep < mid, dep >= mid
    rows = []
    for key, label, t in strategies:
        rows.append({"key": key, "label": label, "threshold": t,
                     "whole": window_result(key, whole_mask, n - 1),
                     "first_half": window_result(key, first_mask, mid - 1),
                     "second_half": window_result(key, second_mask, n - 1),
                     "dip_buys": int(len(np.unique(invest_at[key][invest_at[key] < n]))) if t is not None else None,
                     "cash_share": _cash_share(dep, invest_at[key], asset, cash) if t is not None else 0.0})

    progress(2, 3, f"Every {horizon_years}-year window")
    span = horizon_years * TRADING_DAYS
    windows = [i for i, p in enumerate(dep) if p + span < n]
    rolling: dict[str, dict[str, Any]] = {}
    if len(windows) >= 12:
        diffs = {key: [] for key, _, t in strategies if t is not None}
        starts = []
        for i in windows:
            p0 = dep[i]
            end = p0 + span
            mask = (dep >= p0) & (dep < end)
            starts.append(index[p0])
            base = window_result("now", mask, end)["irr"]
            for key in diffs:
                diffs[key].append(window_result(key, mask, end)["irr"] - base)
        half = len(starts) // 2
        for key, vals in diffs.items():
            v = np.array(vals)
            rolling[key] = {"windows": len(v), "beat_share": float((v > 0).mean()), "median": float(np.median(v)),
                            "worst": float(v.min()), "best": float(v.max()),
                            "beat_share_first": float((v[:half] > 0).mean()), "beat_share_second": float((v[half:] > 0).mean())}
        rolling_meta = {"horizon_years": horizon_years, "first_start": starts[0].strftime("%Y-%m"),
                        "last_start": starts[-1].strftime("%Y-%m"), "split_start": starts[half].strftime("%Y-%m")}
    else:
        rolling_meta = {"horizon_years": horizon_years}
    progress(3, 3, "Done")
    now = rows[0]["whole"]
    best = max(rows[1:], key=lambda r: r["whole"]["irr"])
    edge = best["whole"]["irr"] - now["irr"]
    most_windows = bool(rolling) and max(r["beat_share"] for r in rolling.values()) > 0.5
    if edge > SAVINGS_EDGE and most_windows:
        level, title = "strong", "Waiting for a dip beat investing at once, over the whole period and in most windows"
    elif edge > SAVINGS_EDGE:
        level, title = "mixed", "A wait was ahead over the whole period, but lost in most windows"
    elif edge > -SAVINGS_EDGE:
        level, title = "weak", "Waiting for a dip made no real difference over the whole period"
    else:
        level, title = "weak", "Investing each saving at once beat waiting for every dip size"
    lines = [f"Invested at once: {now['irr']:.2%} a year on the money put in. Best wait ({best['label'].lower()}): "
             f"{best['whole']['irr']:.2%}, with {best['cash_share']:.0%} of the account in cash on average."]
    if rolling:
        k = max(rolling, key=lambda x: rolling[x]["beat_share"])
        r = rolling[k]
        label = next(lab for key, lab, _ in strategies if key == k)
        lines.append(f"Across {r['windows']} {horizon_years}-year windows, the best of the waits ({label.lower()}) beat investing "
                     f"at once in {r['beat_share']:.0%} of them ({r['beat_share_first']:.0%} of the earlier windows, "
                     f"{r['beat_share_second']:.0%} of the later ones); median difference {r['median']:+.1%} a year.")
    notes = list(market.notes) + [
        "A dip is measured from the highest close so far" + (f" in the last {lookback} days" if lookback else "")
        + "; the waiting cash is invested at the next close after a close at or below the threshold.",
        "Money still waiting at the end counts at its cash value. Costs and taxes are left out of both.",
    ]
    return _clean({
        "study": "savings", "signal": market.signal, "held": market.vehicle_label, "thresholds": list(thresholds),
        "period": {"start": index[0].strftime("%Y-%m-%d"), "end": index[-1].strftime("%Y-%m-%d"),
                   "split": index[mid].strftime("%Y-%m-%d"), "deposits": int(len(dep))},
        "verdict": {"level": level, "title": title, "lines": lines},
        "rows": rows, "rolling": rolling, "rolling_meta": rolling_meta,
        "drawdown": _sample(pd.Series(dd, index=index), 4),
        "notes": notes,
    })


def _cash_share(dep: np.ndarray, invest_at: np.ndarray, asset: np.ndarray, cash: np.ndarray) -> float:
    """Average share of the account waiting in cash, over the days after the first deposit."""
    n = len(asset)
    w = np.zeros(n + 2)
    np.add.at(w, dep, 1.0 / cash[dep])
    np.add.at(w, np.minimum(invest_at, n + 1), -1.0 / cash[dep])
    cash_value = np.cumsum(w)[:n] * cash
    inv = invest_at < n
    units = np.zeros(n + 1)
    np.add.at(units, invest_at[inv], cash[invest_at[inv]] / (cash[dep[inv]] * asset[invest_at[inv]]))
    invested = np.cumsum(units)[:n] * asset
    total = cash_value + invested
    ok = total > 0
    return float((cash_value[ok] / total[ok]).mean()) if ok.any() else 0.0


# ================================================================== MSCI momentum rotation


def run_rotation(universe: str = "sectors", tickers: list[str] | None = None, top: int = 3, review: str = "quarterly",
                 weighting: str = "score", buffer: bool = True, adhoc: bool = True, trend_filter: bool = False,
                 cost_bps: float = COST_BPS, start: str | None = None, progress: Progress | None = None) -> dict[str, Any]:
    """MSCI momentum across a fixed list of funds, against holding them all equally and against SPY."""
    progress = progress or (lambda *_: None)
    if tickers:
        names = [t.strip().upper() for t in tickers if t.strip()]
        label = "Your list: " + ", ".join(names)
    elif universe in ROTATION_UNIVERSES:
        names, label = list(ROTATION_UNIVERSES[universe]["tickers"]), ROTATION_UNIVERSES[universe]["label"]
    else:
        raise ValueError(f"Unknown universe: {universe!r}")
    names = list(dict.fromkeys(names))
    if not 3 <= len(names) <= 40:
        raise ValueError("The universe needs between 3 and 40 funds")
    if not 1 <= top < len(names):
        raise ValueError("Hold at least 1 and fewer than all of the funds")
    if review not in ("quarterly", "monthly") or weighting not in ("score", "equal"):
        raise ValueError("review is quarterly or monthly; weighting is score or equal")
    progress(0, 3, "Loading prices")
    prices = load_prices(names + [ROTATION_BENCHMARK])
    missing = [t for t in names if t not in prices]
    if ROTATION_BENCHMARK not in prices:
        raise LookupError(f"No price history for {ROTATION_BENCHMARK}")
    names = [t for t in names if t in prices]
    if len(names) <= top:
        raise LookupError("Too few funds with prices: " + ", ".join(missing))
    bench = prices[ROTATION_BENCHMARK]
    cal = bench.index
    rf_d = daily_rf(cal)
    progress(1, 3, "Momentum scores")
    level = pd.DataFrame({t: prices[t]["adj_close"].reindex(cal).ffill(limit=5) for t in names})
    close = pd.DataFrame({t: prices[t]["close"].reindex(cal).ffill(limit=5) for t in names})
    mom = {t: msci_momentum(close[t].dropna(), rf_d).reindex(cal) for t in names}
    ra6 = pd.DataFrame({t: m["ra6"] for t, m in mom.items()})
    ra12 = pd.DataFrame({t: m["ra12"] for t, m in mom.items()})
    decisions = cal[month_end_mask(cal).to_numpy()]
    decisions = decisions[(ra6.loc[decisions].notna().sum(axis=1) > top).to_numpy()]
    if start:
        decisions = decisions[decisions >= pd.Timestamp(start)]
    if len(decisions) < 36:
        raise ValueError("Need at least three years of month-ends with momentum values")
    ends = decisions if decisions[-1] == cal[-1] else decisions.append(cal[-1:])  # the month so far, if not over

    def forward(levels: pd.Series | pd.DataFrame):
        lv = levels.reindex(ends)
        return (lv.shift(-1) / lv - 1.0).iloc[:-1]

    fwd = forward(level)
    bench_fwd = forward(bench["adj_close"])
    cash_fwd = forward((1.0 + rf_d).cumprod())
    trend_ok = (bench["close"] > sma(bench["close"], 200)).reindex(fwd.index)
    vol = bench["adj_close"].pct_change().rolling(63, min_periods=60).std().mul(math.sqrt(250)).reindex(fwd.index)
    delta = vol / vol.shift(1) - 1.0

    def simulate_rotation(k: int, rev: str, use_trend: bool) -> dict[str, Any]:
        target = pd.Series(0.0, index=names)
        current = pd.Series(0.0, index=names)
        rets, turnover, log = [], 0.0, []
        was_inside = False
        for i, d in enumerate(fwd.index):
            scheduled = rev == "monthly" or d.month in REVIEW_MONTHS or i == 0
            past = delta.iloc[:i].dropna()
            spike = (adhoc and not scheduled and len(past) >= ADHOC_MIN_HISTORY and pd.notna(delta.iloc[i])
                     and delta.iloc[i] > np.percentile(past, 95))
            trade = False
            if scheduled or spike:
                sc = msci_scores(ra6.loc[d], None if spike else ra12.loc[d])
                if len(sc) > k:
                    picks = sc["z"].sort_values(ascending=False).index[:k]
                    raw = sc.loc[picks, "score"] if weighting == "score" else pd.Series(1.0, index=picks)
                    new = pd.Series(0.0, index=names)
                    new[picks] = raw / raw.sum()
                    if buffer and target.sum() > 0:
                        held_now = current / current.sum() if current.sum() > 0 else target
                        new = (held_now + (new - held_now) / 2.0).where(new > 0, 0.0)
                        new = new / new.sum()
                    target = new
                    trade = True
                    log.append({"date": d.strftime("%Y-%m-%d"), "kind": "volatility review" if spike else "review",
                                "holdings": [{"ticker": t, "weight": float(target[t]), "z": float(sc.loc[t, "z"])}
                                             for t in target[target > 0].sort_values(ascending=False).index]})
            inside = not use_trend or bool(trend_ok.loc[d])
            cost = 0.0
            if trade or inside != was_inside:
                want = target if inside else target * 0.0
                cost = float((want - current).abs().sum())
                turnover += cost
                current = want.copy()
            was_inside = inside
            r = fwd.loc[d].reindex(names).fillna(0.0)
            invested = float(current.sum())
            port = float((current * r).sum() + (1.0 - invested) * cash_fwd.loc[d]) - cost * cost_bps / 10_000.0
            rets.append(port)
            grown = current * (1.0 + r)
            total = float(grown.sum() + (1.0 - invested) * (1.0 + cash_fwd.loc[d]))
            current = grown / total if total > 0 else grown
        years = len(rets) / 12.0
        return {"returns": pd.Series(rets, index=fwd.index), "turnover": turnover / years, "log": log}

    def equal_weight() -> pd.Series:
        current = pd.Series(0.0, index=names)
        rets = []
        for i, d in enumerate(fwd.index):
            if i == 0 or d.month in REVIEW_MONTHS:
                avail = fwd.loc[d].dropna().index
                current = pd.Series(0.0, index=names)
                current[avail] = 1.0 / len(avail)
            r = fwd.loc[d].reindex(names).fillna(0.0)
            port = float((current * r).sum())
            rets.append(port)
            grown = current * (1.0 + r)
            current = grown / grown.sum() if grown.sum() > 0 else grown
        return pd.Series(rets, index=fwd.index)

    progress(2, 3, "Reviews and sweep")
    rf_m = cash_fwd
    main = simulate_rotation(top, review, trend_filter)
    rows = [
        _mrow("rotation", f"MSCI momentum, top {top}, {review}" + (", 200-day filter on SPY" if trend_filter else ""),
              "rule", main["returns"], rf_m, main["turnover"]),
    ]
    other = simulate_rotation(top, review, not trend_filter)
    rows.append(_mrow("rotation_alt", f"MSCI momentum, top {top}, {review}" + ("" if trend_filter else ", 200-day filter on SPY"),
                      "rule", other["returns"], rf_m, other["turnover"]))
    ew = equal_weight()
    rows.append(_mrow("equal_weight", "All funds, equal weight", "benchmark", ew, rf_m))
    rows.append(_mrow("spy", "S&P 500 (SPY)", "benchmark", bench_fwd, rf_m))
    for r in rows:
        r["excess_first"] = r["first_half"]["cagr"] - rows[2]["first_half"]["cagr"]
        r["excess_second"] = r["second_half"]["cagr"] - rows[2]["second_half"]["cagr"]
    sweep = []
    for k in range(1, min(6, len(names))):
        for rev in ("quarterly", "monthly"):
            s = simulate_rotation(k, rev, trend_filter)
            row = _mrow(f"top{k}_{rev}", f"Top {k}, {rev}", "sweep", s["returns"], rf_m, s["turnover"])
            row["beats_both"] = (row["first_half"]["cagr"] > rows[2]["first_half"]["cagr"]
                                 and row["second_half"]["cagr"] > rows[2]["second_half"]["cagr"])
            row["is_default"] = k == top and rev == review
            sweep.append(row)
    today = cal[-1]
    latest = msci_scores(ra6.loc[today], ra12.loc[today])
    latest_rows = [{"ticker": t, "m6": _f(mom[t]["m6"].loc[today]), "m12": _f(mom[t]["m12"].loc[today]),
                    "sigma": _f(mom[t]["sigma"].loc[today]), "z": _f(latest.loc[t, "z"]) if t in latest.index else None,
                    "score": _f(latest.loc[t, "score"]) if t in latest.index else None}
                   for t in names]
    latest_rows.sort(key=lambda r: -(r["z"] if r["z"] is not None else -1e9))
    progress(3, 3, "Done")
    ours, eq_row = rows[0], rows[2]
    both = ours["excess_first"] > 0 and ours["excess_second"] > 0
    title = ("Ahead of the equal-weight funds in both halves" if both
             else "Not ahead of the equal-weight funds in both halves")
    lines = [f"{ours['stats']['cagr']:.1%} a year, worst fall {ours['stats']['max_drawdown']:.0%}, against "
             f"{eq_row['stats']['cagr']:.1%} and {eq_row['stats']['max_drawdown']:.0%} for all funds held equally; "
             f"by half {ours['excess_first']:+.1%} and {ours['excess_second']:+.1%} a year.",
             f"Settings tried: {sum(r['beats_both'] for r in sweep)} of {len(sweep)} (top 1 to 5, quarterly or monthly) "
             f"were ahead of the equal-weight funds in both halves."]
    notes = [f"Momentum from price closes, returns with dividends; decided on the last trading day of each month and "
             f"traded at that close; {cost_bps:g} bps on the share of the portfolio that changes; cash earns T-bills.",
             "MSCI weights the chosen names by score times their weight in the parent index; funds have no index "
             "weight, so here the score alone sets the weight." if weighting == "score" else
             "Equal weights among the chosen names (MSCI weights by score times the parent index weight).",
             "The volatility review follows MSCI's Appendix III with SPY as the reference: when SPY's 3-month "
             "volatility rises more in a month than in 95% of the earlier months, the portfolio is reviewed with "
             "6-month momentum alone." if adhoc else "No volatility reviews."]
    if missing:
        notes.append("No prices for: " + ", ".join(missing) + ".")
    if tickers:
        notes.append("A list you chose today may hold funds picked with hindsight; the fixed sector list avoids that.")
    return _clean({
        "study": "rotation", "universe": label, "tickers": names, "top": top, "review": review, "weighting": weighting,
        "period": {"start": fwd.index[0].strftime("%Y-%m-%d"), "end": ends[-1].strftime("%Y-%m-%d"),
                   "split": fwd.index[len(fwd) // 2].strftime("%Y-%m-%d"), "months": len(fwd)},
        "verdict": {"level": "strong" if both else "weak", "title": title, "lines": lines},
        "rows": rows, "sweep": sweep,
        "equity": {r["key"]: [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 3)}
                              for d, v in _equity(s).items()]
                   for r, s in zip(rows, (main["returns"], other["returns"], ew, bench_fwd))},
        "reviews": main["log"][-24:], "latest": latest_rows, "latest_date": today.strftime("%Y-%m-%d"),
        "notes": notes,
    })


def _mrow(key: str, label: str, kind: str, monthly: pd.Series, rf: pd.Series, turnover: float | None = None) -> dict[str, Any]:
    mid = len(monthly) // 2
    return {"key": key, "label": label, "kind": kind, "stats": perf(monthly, rf, 12),
            "first_half": perf(monthly.iloc[:mid], rf, 12), "second_half": perf(monthly.iloc[mid:], rf, 12),
            "turnover": turnover}


def _f(x: Any) -> float | None:
    return float(x) if x is not None and pd.notna(x) else None


# ================================================================== today's state, for the alert


def trend_check(tickers: list[str] | None = None, rules: tuple[str, ...] | None = None,
                max_age_days: float = 0.25) -> dict[str, Any]:
    """Today's state of the alert rules (default settings) on each watched market, with the readings behind it."""
    from data.macro_data import load_macro_history
    from data.price_history import fetch_full_history

    tickers = [t.upper() for t in (tickers or settings.TREND_WATCH)]
    rules = rules or tuple(dict.fromkeys(settings.TREND_ALERT_RULES + settings.TREND_CHECK_RULES))  # the alerting rule first
    prices = fetch_full_history(tickers, max_age_days=max_age_days)
    vix = vix3m = None
    try:
        hist = load_macro_history(2)
        vix, vix3m = hist.series.get("vix"), hist.series.get("vix3m")
    except Exception as exc:  # noqa: BLE001 - the VIX curve check is then skipped
        logger.warning("VIX unavailable for the trend check: %s", exc)
    out = []
    for t in tickers:
        df = prices.get(t)
        if df is None or len(df) < 260:
            out.append({"ticker": t, "error": "No price history"})
            continue
        frame = df[["open", "high", "low", "close", "adj_close"]].copy()
        frame["rf"] = daily_rf(frame.index)
        for key, s in (("vix", vix), ("vix3m", vix3m)):
            if s is not None:
                frame[key] = s.reindex(s.index.union(frame.index)).ffill(limit=5).reindex(frame.index)
        close = frame["close"]
        readings = {
            "close": _f(close.iloc[-1]), "sma200": _f(sma(close, 200).iloc[-1]),
            "vol20": _f(realized_vol(close, 20).iloc[-1]),
            "vix_ratio": _f((frame["vix"] / frame["vix3m"]).iloc[-1]) if {"vix", "vix3m"} <= set(frame.columns) else None,
            "momentum_12_1": _f(excess_momentum(close, frame["rf"], 12, 1).iloc[-1]),
            "from_high": _f(drawdown_from_high(close, 252).iloc[-1]),
            "rsi2": _f(wilder_rsi(close, 2).iloc[-1]),
            "pb_state": int(pb_ema_channel(frame["high"], close, 200)["state"].iloc[-1]),
        }
        states = {}
        for rule in rules:
            strategy = build(rule, validate_params(rule, None))
            states[rule] = {"label": RULES[rule]["label"], **_now_state(rule, strategy, frame)}
        out.append({"ticker": t, "date": frame.index[-1].strftime("%Y-%m-%d"), "readings": readings, "rules": states})
    return _clean({"checked_at": datetime.now().isoformat(timespec="seconds"), "markets": out,
                   "rules": {r: RULES[r]["label"] for r in rules}, "rule_order": list(rules),
                   "alert_rules": list(settings.TREND_ALERT_RULES)})


# ================================================================== helpers and CLI


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN and infinities become None, numpy scalars become Python numbers."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def options() -> dict[str, Any]:
    """What the UI needs to draw the forms."""
    return {
        "rules": {k: {"label": v["label"], "description": v["description"], "params": v["params"], "grid": v["grid"]}
                  for k, v in RULES.items()},
        "rule_order": list(RULES),
        "param_order": {k: list(v["params"]) for k, v in RULES.items()},  # JSON objects arrive with sorted keys
        "index_proxies": INDEX_PROXIES,
        "executions": {"next_close": "At the next close", "same_close": "At the same close (optimistic)"},
        "universes": {k: v["label"] for k, v in ROTATION_UNIVERSES.items()},
        "savings_thresholds": list(SAVINGS_THRESHOLDS),
        "fee": SYNTHETIC_FEE, "spread": SYNTHETIC_SPREAD, "cost_bps": COST_BPS, "max_leverage": MAX_LEVERAGE,
        "watch": list(settings.TREND_WATCH), "alert_rules": list(settings.TREND_ALERT_RULES),
    }


def _cell(v: float | None, spec: str, width: int = 8) -> str:
    return f"{v:>{width}{spec}}" if v is not None else " " * width


def _print_rule(r: dict[str, Any]) -> None:
    head = f"{'':<44}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}{'1st':>8}{'2nd':>8}{'In mkt':>8}"
    print(f"\n{r['label']} ({r['params_text']}), signal {r['signal']}, held {r['held']}")
    print(f"{r['period']['start']} to {r['period']['end']}, halves split at {r['period']['split']}\n")
    print(head)
    for row in r["rows"]:
        s = row["stats"]
        print(f"{row['label'][:43]:<44}{_cell(s['cagr'], '.1%')}{_cell(s['volatility'], '.1%')}{_cell(s['sharpe'], '.2f')}"
              f"{_cell(s['max_drawdown'], '.1%')}{_cell(s['calmar'], '.2f')}{_cell(row['first_half']['cagr'], '.1%')}"
              f"{_cell(row['second_half']['cagr'], '.1%')}{_cell(row.get('exposure'), '.0%')}")
    ts = r["rows"][0].get("trade_stats") or {}
    if ts.get("trades"):
        print(f"\nTrades: {ts['trades']} ({ts['per_year']:.1f} a year), win rate {ts['win_rate']:.0%}, average win "
              f"{ts['avg_win']:.1%}, average loss {ts['avg_loss']:.1%}, expectancy {ts['expectancy']:+.2%} a trade, "
              f"{ts['avg_days']:.0f} days on average.")
    print(f"\n{r['verdict']['title']}.")
    for line in r["verdict"]["lines"]:
        print(f"  {line}")
    boot = r.get("bootstrap") or {}
    if boot:
        print(f"  Reshuffled ({boot['paths']} paths, {boot['block_days']}-day blocks): worst fall 5-95% "
              f"{boot['rule']['max_drawdown']['p5']:.0%} to {boot['rule']['max_drawdown']['p95']:.0%} for the rule, "
              f"{boot['buy_hold']['max_drawdown']['p5']:.0%} to {boot['buy_hold']['max_drawdown']['p95']:.0%} holding; "
              f"the rule fell less in {boot['p_shallower_drawdown']:.0%} of paths and earned more in {boot['p_higher_cagr']:.0%}.")
    tr = r.get("tracking") or {}
    if tr:
        print(f"  Synthetic against the real fund {tr['start']} to {tr['end']}: {tr['cagr_synthetic']:.1%} against "
              f"{tr['cagr_real']:.1%} a year, correlation {tr['correlation']:.3f}, tracking error {tr['tracking_error']:.1%}.")
    for n in r["notes"]:
        print(f"  Note: {n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trend, momentum and dip rules, with robustness checks")
    parser.add_argument("--study", choices=("rule", "savings", "rotation", "check"), default="rule")
    parser.add_argument("--rule", choices=list(RULES), default="two_engine")
    parser.add_argument("--param", action="append", default=[], help="key=value, for example window=200")
    parser.add_argument("--signal", default="QQQ")
    parser.add_argument("--leverage", type=int, default=1)
    parser.add_argument("--vehicle", default=None, help="A real fund to hold from its first day, e.g. TQQQ")
    parser.add_argument("--extend", action="store_true", help="Extend with the price index (QQQ: ^NDX since 1985)")
    parser.add_argument("--execution", choices=list(EXECUTION_LAG), default="next_close")
    parser.add_argument("--cost-bps", type=float, default=COST_BPS)
    parser.add_argument("--fee", type=float, default=SYNTHETIC_FEE)
    parser.add_argument("--spread", type=float, default=SYNTHETIC_SPREAD, help="Borrowing spread over T-bills, a year")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD, or 'vehicle' for the real fund's first day")
    parser.add_argument("--universe", choices=list(ROTATION_UNIVERSES), default="sectors")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--trend-filter", action="store_true")
    parser.add_argument("--json", action="store_true", help="Also write the full result to analysis/output/")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.study == "check":
        print(json.dumps(trend_check(), indent=2))
        return 0
    if args.study == "savings":
        result = run_savings(args.signal, args.extend, args.leverage, args.vehicle, args.fee, args.spread, start=args.start)
        print(f"\nSavings into {result['held']}, {result['period']['start']} to {result['period']['end']}")
        print(f"{'':<34}{'IRR':>8}{'1st':>8}{'2nd':>8}{'x money':>9}{'Cash':>7}{'Max wait':>10}")
        for row in result["rows"]:
            w = row["whole"]
            print(f"{row['label']:<34}{w['irr']:>8.1%}{row['first_half']['irr']:>8.1%}{row['second_half']['irr']:>8.1%}"
                  f"{w['multiple']:>9.2f}{row['cash_share']:>7.0%}{w['longest_wait_months']:>9.0f}m")
        for key, r in result["rolling"].items():
            print(f"  {key}: beat investing at once in {r['beat_share']:.0%} of {r['windows']} windows "
                  f"(earlier {r['beat_share_first']:.0%}, later {r['beat_share_second']:.0%}), median {r['median']:+.2%} a year")
        print(f"\n{result['verdict']['title']}.")
        for line in result["verdict"]["lines"]:
            print(f"  {line}")
    elif args.study == "rotation":
        result = run_rotation(args.universe, top=args.top, trend_filter=args.trend_filter, cost_bps=args.cost_bps,
                              start=args.start)
        print(f"\n{result['universe']}, {result['period']['start']} to {result['period']['end']}")
        print(f"{'':<52}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'1st':>8}{'2nd':>8}")
        for row in result["rows"] + result["sweep"]:
            s = row["stats"]
            print(f"{row['label'][:51]:<52}{_cell(s['cagr'], '.1%')}{_cell(s['sharpe'], '.2f')}{_cell(s['max_drawdown'], '.1%')}"
                  f"{_cell(row['first_half']['cagr'], '.1%')}{_cell(row['second_half']['cagr'], '.1%')}")
        print(f"\n{result['verdict']['title']}.")
        for line in result["verdict"]["lines"]:
            print(f"  {line}")
    else:
        params = dict(p.split("=", 1) for p in args.param)
        result = run_rule(args.rule, params, args.signal, args.leverage, args.vehicle, args.extend, args.execution,
                          args.cost_bps, args.fee, args.spread, start=args.start)
        _print_rule(result)
    if args.json:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = settings.OUTPUT_DIR / f"systematic_{args.study}_{args.rule if args.study == 'rule' else args.signal}.json"
        path.write_text(json.dumps(result, indent=1))
        print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
