"""The market regime over time: what it said, why it says it today, and what followed (October 2026).

A view for the Analyze tab, built on the point in time replay of ``core.regime_study`` (one classifier,
``data.macro_data.compute_macro_snapshot``, which reads nothing after its date). Nothing here scores or
decides anything (D-10, D-38): the label orders risk, not returns.

``build_history`` returns plain JSON-safe data:
  today       each input with its rule, value and points, the sum, the cut-offs and the Macro agent score
  weekly      dates, regime codes, regime score, Macro agent score and SPY and QQQ closes, for the chart
  monthly     one row per month-end: label, scores, each input's value and points, which inputs existed
  stats       per regime, for the whole period and each half, SPY and QQQ forward 3-month volatility,
              forward 3-month maximum drawdown and forward 12-month return (the D-38 numbers; monthly sample)
  eras        stretches of time with the same set of inputs (inputs appear over time: D-38)

The result is cached for a day under data/cache/regime_history.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from core import regime_study as rs
from data.macro_data import (
    MAX_REGIME_SCORE,
    REGIME_THRESHOLDS,
    MacroHistory,
    MacroRegimeData,
    _download_closes,
    compute_macro_snapshot,
)

logger = logging.getLogger(__name__)

START = "2000-01-01"
QQQ_START = "1999-01-01"  # QQQ trades from March 1999
REGIMES = rs.REGIMES
PERIODS = ("whole", "first_half", "second_half")
CACHE_SECONDS = 86400

# Which snapshot components belong to which input of the classifier.
COMPONENT_INPUT = {
    "SPY vs 200-day SMA": "spy_trend", "SPY vs 50-day SMA": "spy_trend",
    "VIX level": "vix", "VIX (contrarian)": "vix",
    "VIX term structure": "vix_term",
    "Yield curve 10Y-2Y": "fred_yields", "Curve un-inversion": "fred_yields",
    "Breadth RSP/SPY": "breadth", "Credit HYG/LQD": "credit", "10Y yield 60d change": "ten_year",
}

INPUT_TEXT: dict[str, dict[str, str]] = {
    "spy_trend": {
        "label": "Market trend",
        "rule": "SPY above its 200-day average: +2, below: -2. SPY above its 50-day average: +1, below: -1.",
    },
    "vix": {
        "label": "Fear (VIX)",
        "rule": "VIX under 18: +2. 18 to 22: +0.5. 22 to 28: -1. 28 to 35: -2. 35 and above: -3.",
        "agent_rule": "The Macro agent reads fear the other way round: 40 and above: +6. 30 to 40: 0. 20 to 30: +0.5. 13 to 20: 0. Under 13: -1.",
    },
    "vix_term": {
        "label": "Fear, short against long term",
        "rule": "VIX divided by the 3-month VIX: above 1.1: -1.5. Above 1.0: -1. Under 0.9: +0.5. Otherwise 0.",
    },
    "fred_yields": {
        "label": "Yield curve (10 year minus 2 year)",
        "rule": "Spread above 0.5 points: +1. Above 0: 0. Down to -0.25: -1. Lower: -2. Back above 0 after a deep inversion: a further -0.5.",
    },
    "breadth": {
        "label": "Breadth (equal weight against cap weight)",
        "rule": "RSP/SPY more than 1% above its 50-day average: +1. More than 1% below: -1. Otherwise 0.",
    },
    "credit": {
        "label": "Credit (junk against investment grade bonds)",
        "rule": "HYG/LQD more than 0.5% above its 50-day average: +1. More than 0.5% below: -1.5. Otherwise 0.",
    },
    "ten_year": {
        "label": "10 year yield, 60-day change",
        "rule": "Up more than 0.5 points: -0.5. Down more than 0.5 points: +0.5. Otherwise 0.",
    },
}


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _input_value(key: str, snap: MacroRegimeData) -> tuple[float | None, str]:
    """The reading behind an input, as a number and as plain text."""
    if key == "spy_trend":
        if _num(snap.spy_price) is None or not _num(snap.spy_sma_200):
            return None, ""
        gap = (snap.spy_price / snap.spy_sma_200 - 1) * 100
        gap50 = (snap.spy_price / snap.spy_sma_50 - 1) * 100
        return gap, f"{gap:+.1f}% against the 200-day average, {gap50:+.1f}% against the 50-day"
    if key == "vix":
        v = _num(snap.vix_level)
        return v, "" if v is None else f"VIX {v:.1f}"
    if key == "vix_term":
        v = _num(snap.vix_term_ratio)
        return v, "" if v is None else f"VIX / 3-month VIX {v:.2f}"
    if key == "fred_yields":
        v = _num(snap.yield_spread_10y_2y)
        return v, "" if v is None else f"{v:+.2f} points"
    if key == "breadth":
        r, s = _num(snap.breadth_ratio), _num(snap.breadth_ratio_sma_50)
        if r is None or not s:
            return None, ""
        return (r / s - 1) * 100, f"{(r / s - 1) * 100:+.1f}% against its 50-day average ({snap.breadth_trend.lower()})"
    if key == "credit":
        r, s = _num(snap.credit_ratio), _num(snap.credit_ratio_sma_50)
        if r is None or not s:
            return None, ""
        return (r / s - 1) * 100, f"{(r / s - 1) * 100:+.1f}% against its 50-day average"
    v = _num(snap.ten_year_change_60d)
    return v, "" if v is None else f"{v:+.2f} points in 60 days (yield {snap.ten_year_yield:.2f}%)"


def explain(snap: MacroRegimeData) -> dict[str, Any]:
    """Every input of one snapshot with its rule, value and points; the sum; the cut-offs; the Macro agent score."""
    regime_pts: dict[str, float] = {}
    agent_pts: dict[str, float] = {}
    for c in snap.components:
        k = COMPONENT_INPUT.get(c["component"])
        if k:
            regime_pts[k] = regime_pts.get(k, 0.0) + float(c["points"])
    contrarian = settings.VIX_STANCE == "contrarian" and snap.opportunity_score is not None
    for c in (snap.opportunity_components if contrarian else snap.components):
        k = COMPONENT_INPUT.get(c["component"])
        if k:
            agent_pts[k] = agent_pts.get(k, 0.0) + float(c["points"])
    inputs = []
    for key in rs.INPUTS:
        hi, lo = rs.INPUT_POINTS[key]
        available = key not in snap.missing
        value, text = _input_value(key, snap) if available else (None, "")
        inputs.append({
            "key": key, "label": INPUT_TEXT[key]["label"], "rule": INPUT_TEXT[key]["rule"],
            "agent_rule": INPUT_TEXT[key].get("agent_rule"), "available": available,
            "value": value, "value_text": text,
            "points": regime_pts.get(key) if available else None,
            "agent_points": agent_pts.get(key) if available else None,
            "best": hi, "worst": lo,
        })
    total = float(sum(c["points"] for c in snap.components))
    agent_total = float(snap.opportunity_score if contrarian else snap.regime_score)
    cutoffs = [{"regime": name, "from": thr} for thr, name in REGIME_THRESHOLDS]
    cutoffs.append({"regime": "CRISIS", "from": None})
    up = next((thr for thr, name in reversed(REGIME_THRESHOLDS) if thr > total), None)
    down = next((thr for thr, name in REGIME_THRESHOLDS if thr <= total), None)
    return {
        "as_of": snap.as_of, "regime": snap.regime, "total": round(total, 2), "cutoffs": cutoffs,
        "points_to_next_up": None if up is None else round(up - total, 2),
        "margin_above_cutoff": None if down is None else round(total - down, 2),
        "inputs": inputs, "missing": list(snap.missing), "data_quality": snap.data_quality,
        "agent": {"score": float(np.clip(agent_total / MAX_REGIME_SCORE * 100.0, -100.0, 100.0)),
                  "points": round(agent_total, 2), "max_points": MAX_REGIME_SCORE,
                  "stance": "contrarian" if contrarian else "risk"},
    }


# ------------------------------------------------------------------ the replay


def month_ends(spy: pd.Series, start: str) -> list[pd.Timestamp]:
    return [d for d in rs.sample_dates(spy, "M") if d >= pd.Timestamp(start)]


def week_ends(spy: pd.Series, start: str) -> list[pd.Timestamp]:
    return [d for d in rs.sample_dates(spy, "W") if d >= pd.Timestamp(start)]


def monthly_table(history: MacroHistory, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Per month-end: regime, scores, input era (from regime_study.macro_table) and each input's value and points."""
    table = rs.macro_table(history, dates)
    agent = rs.macro_rows(history, list(table.index)).set_index("date")
    table["macro_score"] = agent["macro_score"]
    eras = rs.input_eras(table)
    table["era"] = eras
    detail = {d: explain(compute_macro_snapshot(history, d)) for d in table.index}
    table["detail"] = pd.Series({d: {i["key"]: {"v": i["value"], "p": i["points"]} for i in e["inputs"]} for d, e in detail.items()})
    return table


def _regime_stats(table: pd.DataFrame, fwd: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    """Per regime on the rows in ``mask``: months, share, mean forward 3-month volatility and drawdown, 12-month return."""
    out: dict[str, Any] = {}
    total = int(mask.sum())
    for r in REGIMES:
        m = mask & (table["regime"] == r).to_numpy()
        entry: dict[str, Any] = {"months": int(m.sum()), "share": float(m.sum() / total) if total else None}
        for col in ("vol_3m", "mdd_3m", "ret_12m"):
            vals = fwd[col].to_numpy(dtype=float)[m]
            vals = vals[~np.isnan(vals)]
            entry[col] = float(vals.mean()) if len(vals) else None
            entry[col + "_n"] = int(len(vals))
        out[r] = entry
    return out


def regime_stats(table: pd.DataFrame, closes: dict[str, pd.Series]) -> dict[str, Any]:
    """{asset: {period: {regime: stats}}} plus the split date between the halves."""
    n = len(table)
    half = n // 2
    masks = {"whole": np.ones(n, bool), "first_half": np.arange(n) < half, "second_half": np.arange(n) >= half}
    out: dict[str, Any] = {}
    for asset, s in closes.items():
        s = s.dropna()
        dates = [d for d in table.index if d >= s.index[0]]
        sub = table.loc[dates]
        fwd = rs.forward_spy(s, dates)
        sub_masks = {k: v[len(table) - len(sub):] for k, v in masks.items()}
        out[asset] = {p: _regime_stats(sub, fwd, sub_masks[p]) for p in PERIODS}
        out[asset]["first_month"] = str(sub.index[0].date()) if len(sub) else None
    return {"assets": out, "split": str(table.index[half].date()) if n else None, "months": n,
            "first": str(table.index[0].date()) if n else None, "last": str(table.index[-1].date()) if n else None}


def era_summary(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Stretches with the same inputs: when, how many months, which inputs were missing, the regime shares."""
    out = []
    for era, g in table.groupby("era", sort=False):
        have = era.split("+") if era else []
        shares = g["regime"].value_counts(normalize=True)
        lo, hi = rs.theoretical_score_range(era)
        out.append({"first": str(g.index[0].date()), "last": str(g.index[-1].date()), "months": int(len(g)),
                    "inputs": have, "missing": [i for i in rs.INPUTS if i not in have],
                    "possible_min": lo, "possible_max": hi,
                    "shares": {r: float(shares.get(r, 0.0)) for r in REGIMES}})
    out.sort(key=lambda e: e["first"])
    return out


def build_history(history: MacroHistory, qqq: pd.Series | None, start: str = START) -> dict[str, Any]:
    """Everything the panel needs, from the macro history (and QQQ closes) with no look-ahead."""
    spy = history.series["spy"]
    months = month_ends(spy, start)
    weeks = week_ends(spy, start)
    table = monthly_table(history, months)

    wtable = rs.macro_table(history, weeks)
    wagent = rs.macro_rows(history, list(wtable.index)).set_index("date")
    wdates = list(wtable.index)

    def on(series: pd.Series | None) -> list[float | None]:
        if series is None:
            return [None] * len(wdates)
        s = series.dropna()
        v = s.reindex(wdates, method="ffill")
        v[[d < s.index[0] for d in wdates]] = np.nan
        return [None if not np.isfinite(x) else round(float(x), 2) for x in v.to_numpy(dtype=float)]

    closes = {"SPY": spy}
    if qqq is not None and len(qqq):
        closes["QQQ"] = qqq
    today = explain(compute_macro_snapshot(history, None))
    eras = era_summary(table)
    return {
        "generated": time.strftime("%Y-%m-%d"),
        "start": start,
        "regimes": list(REGIMES),
        "weekly": {
            "dates": [str(d.date()) for d in wdates],
            "regime": [REGIMES.index(r) for r in wtable["regime"]],
            "regime_score": [round(float(x), 2) for x in wtable["regime_score"]],
            "macro_score": [None if not np.isfinite(x) else round(float(x), 1) for x in wagent["macro_score"]],
            "spy": on(spy),
            "qqq": on(closes.get("QQQ")),
        },
        "monthly": [
            {"date": str(d.date()), "regime": r.regime, "regime_score": float(r.regime_score),
             "macro_score": None if not np.isfinite(r.macro_score) else float(r.macro_score),
             "era": r.era, "inputs": r.detail}
            for d, r in table.iterrows()
        ],
        "stats": regime_stats(table, closes),
        "eras": eras,
        "today": today,
        "note": ("Month-end sample. The label at each date uses only data up to that date. Inputs appear over time "
                 "(breadth 2003, VIX3M 2006, credit 2007), so the same score means a slightly different thing in each era."),
    }


# ------------------------------------------------------------------ loading and cache


def _cache_path():
    return settings.DATA_DIR / "regime_history" / "history.json"


def load_qqq() -> pd.Series | None:
    """QQQ adjusted closes since 1999, or None when Yahoo refuses (the view then shows SPY only)."""
    try:
        return _download_closes(["QQQ"], QQQ_START).get("QQQ")
    except Exception as exc:  # noqa: BLE001 - provider errors are not fatal here
        logger.warning("QQQ download failed: %s", exc)
        return None


def get_regime_history(refresh: bool = False) -> dict[str, Any]:
    """The cached view (a day), rebuilt from the day-cached long macro history when stale."""
    path = _cache_path()
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < CACHE_SECONDS:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            pass
    history = rs.load_long_history(refresh=refresh)
    qqq = load_qqq()
    result = rs._clean(build_history(history, qqq))
    if qqq is None:
        result["warning"] = "QQQ could not be loaded, so only SPY is shown. Try again in a minute."
        return result  # not cached: a blip should not hide QQQ for a day
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result
