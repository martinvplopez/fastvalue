"""Does the market regime make sense, and does it earn its place in the weights? (October 2026)

The regime (``data.macro_data.compute_macro_snapshot``: BULL, CORRECTION, BEAR, CRISIS) picks the timing
panel's weight profile (``regime_weights`` in config/agent_weights.json). The owner score is regime
independent by design (decision_log.md D-11) and the timing panel never votes. This study measures four
things and proposes; it changes nothing live (D-10).

ACCEPTANCE TESTS, written down before any result was looked at
==============================================================
Samples: Q1 uses every month-end from the first date SPY has 200 bars (1993-11) to the last month with
data; the halves split that list in the middle. Q2 to Q5 use the S&P 500 point-in-time panel (601
companies, 2011-01 to 2026-08, cached by core.value_study / core.all_agents_study) and split its month list
in the middle. Newey-West t uses 3 lags for 1-month horizons and 12 for 12-month ones. "Passes" always means
the whole period AND both halves, so a result that holds in one half only is reported as failing.

Q1. Does the classifier describe the market?
  1a. Risk. The label "separates forward risk" if the forward 3-month realised volatility and 3-month maximum
      drawdown of SPY, grouped by label, have an adjusted R-squared larger than BOTH baselines (SPY above or
      below its 200-day mean; VIX buckets <15, 15-20, 20-30, >=30) for the whole period and in each half, and
      mean forward volatility rises BULL < CORRECTION < BEAR < CRISIS in the whole period and in each half.
  1b. Returns. The contrarian view "fear regimes precede better returns" passes if BEAR+CRISIS minus BULL
      forward 12-month SPY return is positive in both halves and has a whole-period Newey-West t of at least 2.
      The opposite sign (stress precedes worse returns) is reported the same way, not hidden.
  1c. Stability. Report flips a year, spell lengths, and whether the meaning drifts as inputs appear
      (regime shares by input era; agreement between the full-input regime and a core-inputs-only regime).
      A drift is a finding if the BULL share differs by more than 10 points between eras with the same market.

Q2. Should timing signals be weighted differently by regime?
  2a. Per regime, rank IC (1 month ahead, primary; 12 months shown) of the momentum, volume and pullback agent
      scores and the timing composite, and the pullback-setup buckets, each with a Newey-West t computed on
      the full month index, and the two halves. Descriptive: no pass mark, the numbers feed 2b.
  2b. Decisive: the shipped per-regime profiles against one flat profile (the average of the four), all else
      equal. Four paired monthly series, shipped minus flat: (i) the timing composite's rank IC, (ii) mean
      forward 1-month return of owner BUYs labelled FAVORABLE minus UNFAVORABLE, (iii) the return of owner top
      20% held only while the label is not UNFAVORABLE, (iv) the same held only while FAVORABLE (the all-agents
      study's rules). Per-regime profiles are justified only if EVERY one of the four has a positive mean
      difference over the whole period and in each half, with Newey-West t of at least 2 in all three
      (whole, first, second). Otherwise the verdict is "one profile is enough" and nothing changes.

Q3. Should the owner score value companies differently by regime?
  3a. Per regime at the rating date, the owner score's 12-month rank IC and the BUY minus SELL 12-month
      return. The score "works in a regime" if the mean is positive in both halves and the whole-period t is
      at least 2 (a regime with fewer than 6 months in a half is "too thin"). If it works everywhere,
      regime independence (D-11) stays. A regime where it fails is a lead, not a change.

Q4. "Buy owner BUYs in BEAR or CRISIS" (research_results.md section 2 says CRISIS 48.9%, t 3.6).
  4a. Months per regime, runs of consecutive months, independent episodes (stress months less than 12 months
      apart are one episode, because their 12-month windows overlap), both halves, and leave-one-episode-out.
      The claim "owner BUYs bought in BEAR or CRISIS do better over 12 months" is supported only if the stress
      minus calm difference is positive in both halves, whole-period t at least 2, there are at least 5
      independent episodes, and it stays positive with every single episode removed. Fewer episodes is
      reported as an anecdote, whatever the t.

Q5. The double count.
  5a. The macro score is the same for every ticker on a date. Report the share of timing-label changes (same
      ticker, consecutive months) that happen only because the macro state changed, only because the ticker's
      own signals changed, both, or neither alone.
  5b. A timing label without the macro agent "loses nothing" if the paired differences (shipped minus
      no-macro) of the same four series as 2b are not positive with t of at least 2 in the whole period.
      Both a regime-profile and a flat-profile no-macro version are tested, and a version whose label shares
      are matched to the shipped label to remove the threshold effect.

Other checks. (i) compute_macro_snapshot reads nothing after ``asof`` (tests/test_regime_study.py proves it
on synthetic data by rewriting the future). (ii) Where the thresholds (4, 1, -3) came from: searched in
decision_log.md; the answer goes to the report. (iii) Any classifier bug is reported with a failing test in
the report, not fixed here.

Run: ``python -m core.regime_study`` (cached value, insider and timing panels from the earlier studies are
reused; market data comes from Yahoo and FRED, SEC is not needed). ``--refresh`` redownloads the macro
history, ``--rebuild-panel`` rebuilds the scored panel, ``--classifier-only`` skips Q2 to Q5.
Report: analysis/output/regime_study.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import time
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from agents.base import UNAVAILABLE
from agents.regime_agent import MacroRegimeAgent
from core.all_agents_study import (
    BUYS,
    SCALES,
    SELLS,
    add_composites,
    portfolio_returns,
)
from core.value_study import add_insider_composite, forward_returns, month_end_panel, newey_west_t
from data.macro_data import (
    REGIME_THRESHOLDS,
    MacroHistory,
    classify_regime,
    compute_macro_snapshot,
    load_macro_history,
)

logger = logging.getLogger(__name__)

REGIMES: tuple[str, ...] = ("BULL", "CORRECTION", "BEAR", "CRISIS")
STRESS = ("BEAR", "CRISIS")
SETUPS = ("pullback_in_uptrend", "neutral", "extended", "trend_broken")
INPUTS = ("spy_trend", "vix", "vix_term", "fred_yields", "breadth", "credit", "ten_year")
CORE_SERIES = ("spy", "vix", "t10y2y", "t10y3m", "tnx")  # the inputs that exist since 1993
HORIZON_DAYS = {"1m": 21, "3m": 63, "12m": 252}
HISTORY_YEARS = 40
VIX_BUCKETS = (15.0, 20.0, 30.0)
VIX_LABELS = ("<15", "15-20", "20-30", ">=30")
TIMING_KEYS = ("momentum", "volume", "pullback", "macro")
SCORE_COLUMNS = {"momentum": ("mom_score", "mom_conf"), "volume": ("vol_score", "vol_conf"),
                 "pullback": ("pb_score", "pb_conf"), "macro": ("macro_score", "macro_conf")}
# Largest and smallest points each input can add to the summed score (data/macro_data.py compute_macro_snapshot).
INPUT_POINTS: dict[str, tuple[float, float]] = {
    "spy_trend": (3.0, -3.0), "vix": (2.0, -3.0), "vix_term": (0.5, -1.5), "fred_yields": (1.0, -2.0),
    "breadth": (1.0, -1.0), "credit": (1.0, -1.5), "ten_year": (0.5, -0.5),
}
EPISODE_GAP_MONTHS = 12

# D-40 out-of-sample check. C1's label cut-offs were frozen on the S&P 500 panel (shipped label shares FAVORABLE 56.44%,
# UNFAVORABLE 19.00%; flat profile without macro, rounded to one decimal) BEFORE any of the 32 tickers below was scored.
C1_FAVORABLE_CUTOFF = 6.1
C1_UNFAVORABLE_CUTOFF = -40.0
# The 32 untuned US tickers of core.timing_validation (the cache files of 2026-09-27, before any other ticker was cached;
# the cache now also holds the owner's holdings, which are not used).
OOS_TICKERS = ("AA", "AIG", "AMD", "AXP", "BA", "C", "CAT", "DD", "DIS", "GE", "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "KO", "MCD",
               "MMM", "MO", "MRK", "NEE", "NKE", "NVDA", "O", "PFE", "PG", "RTX", "T", "TSLA", "VZ", "WMT")
MIN_EPISODES = 5
THIN_MONTHS = 6
CONFIG_PATH = settings.CONFIG_DIR / "agent_weights.json"


# ------------------------------------------------------------------ statistics


def _hac_sum(g: np.ndarray, lags: int) -> float:
    """Bartlett-weighted long-run variance numerator of a moment series (NaN treated as 0)."""
    g = np.nan_to_num(g.astype(float))
    s = float(g @ g)
    for k in range(1, lags + 1):
        if k < len(g):
            s += 2 * (1 - k / (lags + 1)) * float(g[k:] @ g[:-k])
    return s


def subset_mean_t(x: np.ndarray, mask: np.ndarray, lags: int) -> tuple[float, float, int]:
    """(mean, Newey-West t, n) of ``x`` over the months where ``mask`` is true.

    The HAC runs on the full time index, with the other months contributing zero, so the gaps between
    non-adjacent months of one regime are respected (compacting the subset would pretend they are
    consecutive and misstate the autocorrelation).
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(mask, dtype=bool) & ~np.isnan(x)
    n = int(d.sum())
    if n < 3:
        return (float(x[d].mean()) if n else float("nan")), float("nan"), n
    m = float(x[d].mean())
    g = np.where(d, x - m, 0.0)
    s = _hac_sum(g, lags)
    t = m / (math.sqrt(s) / n) if s > 0 else float("nan")
    return m, float(t), n


def diff_t(x: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, lags: int, y: np.ndarray | None = None) -> tuple[float, float, int, int]:
    """(mean_a - mean_b, Newey-West t, n_a, n_b) for two month subsets, on the full time index.

    Group a is read from ``x``, group b from ``y`` (default: also ``x``, the two subsets of one series).
    """
    x = np.asarray(x, dtype=float)
    yb = x if y is None else np.asarray(y, dtype=float)
    da = np.asarray(mask_a, dtype=bool) & ~np.isnan(x)
    db = np.asarray(mask_b, dtype=bool) & ~np.isnan(yb)
    na, nb = int(da.sum()), int(db.sum())
    if na < 3 or nb < 3:
        return float("nan"), float("nan"), na, nb
    ma, mb = float(x[da].mean()), float(yb[db].mean())
    g = np.where(da, (x - ma) / na, 0.0) - np.where(db, (yb - mb) / nb, 0.0)
    s = _hac_sum(g, lags)
    t = (ma - mb) / math.sqrt(s) if s > 0 else float("nan")
    return ma - mb, float(t), na, nb


def paired_diff(a: pd.Series, b: pd.Series, lags: int = 3, split: pd.Timestamp | None = None) -> dict[str, Any]:
    """Mean of a - b over the common months, whole period and both halves, with Newey-West t."""
    d = (a - b).dropna().sort_index()
    if len(d) < 12:
        return {"months": int(len(d)), "mean": None, "t": None, "first_half": None, "second_half": None,
                "t_first": None, "t_second": None}
    cut = split if split is not None else d.index[len(d) // 2]
    first, second = d[d.index < cut], d[d.index >= cut]
    out = {"months": int(len(d)), "mean": float(d.mean()), "t": newey_west_t(d.to_numpy(), lags)}
    for name, part in (("first", first), ("second", second)):
        out[f"{name}_half" if name == "first" else "second_half"] = float(part.mean()) if len(part) else None
        out[f"t_{name}"] = newey_west_t(part.to_numpy(), lags) if len(part) >= 10 else None
    return out


def passes_both_halves(res: dict[str, Any], t_min: float = 2.0) -> bool:
    """Positive mean in the whole period and each half, every t at or above ``t_min``."""
    keys = ("mean", "first_half", "second_half")
    ts = ("t", "t_first", "t_second")
    if any(res.get(k) is None or res[k] <= 0 for k in keys):
        return False
    return all(res.get(k) is not None and res[k] == res[k] and res[k] >= t_min for k in ts)


def group_adj_r2(y: np.ndarray, labels: np.ndarray) -> float:
    """Adjusted R-squared of the group-mean model of ``y`` on ``labels`` (rows with a missing value are dropped)."""
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels, dtype=object)
    ok = ~np.isnan(y) & np.array([isinstance(v, str) for v in labels])
    y, labels = y[ok], labels[ok]
    n = len(y)
    groups = pd.Series(y).groupby(pd.Series(labels)).agg(["mean", "count"])
    k = len(groups)
    if n <= k + 1 or k < 2:
        return float("nan")
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    means = pd.Series(labels).map(groups["mean"]).to_numpy(dtype=float)
    sse = float(((y - means) ** 2).sum())
    r2 = 1 - sse / sst
    return float(1 - (1 - r2) * (n - 1) / (n - k))


def runs(labels: list[Any]) -> list[tuple[Any, int, int]]:
    """Maximal runs of equal consecutive labels as (label, start index, length)."""
    out: list[tuple[Any, int, int]] = []
    for i, lab in enumerate(labels):
        if out and out[-1][0] == lab:
            out[-1] = (lab, out[-1][1], out[-1][2] + 1)
        else:
            out.append((lab, i, 1))
    return out


def episodes(month_index: np.ndarray, gap: int = EPISODE_GAP_MONTHS) -> list[list[int]]:
    """Cluster sorted integer month numbers: a new episode starts when the gap to the previous one exceeds ``gap`` months."""
    clusters: list[list[int]] = []
    for m in sorted(int(v) for v in month_index):
        if clusters and m - clusters[-1][-1] <= gap:
            clusters[-1].append(m)
        else:
            clusters.append([m])
    return clusters


def month_number(dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return (d.year * 12 + d.month).to_numpy()


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN and inf become None, numpy scalars become Python ones."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)[:10]
    return obj


# ------------------------------------------------------------------ Q1: the classifier on its own


def load_long_history(refresh: bool = False) -> MacroHistory:
    """Macro series since the 1980s (Yahoo and FRED), cached for a day under data/cache/regime_study/."""
    path = settings.DATA_DIR / "regime_study" / "macro_history.pkl"
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < 86400:
        return pickle.loads(path.read_bytes())
    history = load_macro_history(HISTORY_YEARS)
    if history.errors.get("core") or "spy" not in history.series:
        raise RuntimeError(f"Macro history unavailable: {history.errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(history))
    return history


def macro_table(history: MacroHistory, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Regime, scores and which inputs existed, at each date (one compute_macro_snapshot per date)."""
    rows = []
    for d in dates:
        s = compute_macro_snapshot(history, d)
        if "spy_trend" in s.missing or "vix" in s.missing:
            continue  # the label is defined here only when the trend and the VIX both exist
        row = {
            "date": d, "regime": s.regime, "regime_score": s.regime_score, "opportunity_score": s.opportunity_score,
            "vix": s.vix_level, "spy": s.spy_price, "sma200": s.spy_sma_200, "missing": ",".join(s.missing),
            "n_components": len(s.components),
        }
        for c in s.components:
            row["pts " + c["component"]] = c["points"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def sample_dates(spy: pd.Series, freq: str) -> list[pd.Timestamp]:
    """Last SPY trading day of each month ("M") or week ("W"), from the 201st bar on."""
    s = spy.iloc[200:]
    key = s.index.to_period("M") if freq == "M" else s.index.to_period("W-FRI")
    return list(s.groupby(key).tail(1).index)


def forward_spy(spy: pd.Series, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Forward return, realised volatility (annualised) and maximum drawdown of SPY after each date.

    Horizons are 21, 63 and 252 trading days; a date too close to the end of the data gets NaN.
    """
    p = spy.to_numpy(dtype=float)
    idx = spy.index
    lr = np.diff(np.log(p))
    rows = []
    for d in dates:
        i = int(idx.searchsorted(d, side="right")) - 1
        row: dict[str, Any] = {"date": d}
        for name, h in HORIZON_DAYS.items():
            if i < 0 or i + h >= len(p):
                row[f"ret_{name}"] = row[f"vol_{name}"] = row[f"mdd_{name}"] = np.nan
                continue
            path = p[i: i + h + 1]
            row[f"ret_{name}"] = float(path[-1] / path[0] - 1)
            row[f"vol_{name}"] = float(np.std(lr[i: i + h], ddof=1) * math.sqrt(252))
            row[f"mdd_{name}"] = float((path / np.maximum.accumulate(path) - 1).min())
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def flip_drivers(table: pd.DataFrame) -> dict[str, Any]:
    """What moved when the label changed from one sample date to the next: which components changed their points."""
    pts = [c for c in table.columns if c.startswith("pts ")]
    lab = table["regime"].to_numpy()
    change = lab[1:] != lab[:-1]
    n = int(change.sum())
    out: dict[str, Any] = {"label_changes": n, "components": {}}
    delta = table[pts].fillna(0.0).diff().iloc[1:]
    for c in pts:
        moved = (delta[c].abs().to_numpy() > 0) & change
        out["components"][c[4:]] = float(moved.sum() / n) if n else None
    trend = ((delta["pts SPY vs 200-day SMA"].abs() > 0) & change).to_numpy() if "pts SPY vs 200-day SMA" in delta else np.zeros(len(change), bool)
    out["spy_200d_side_changed"] = float(trend.sum() / n) if n else None
    ratio = (table["spy"] / table["sma200"] - 1).abs()
    out["share_of_dates_within_1pct_of_200d"] = float((ratio < 0.01).mean())
    out["share_of_dates_within_2pct_of_200d"] = float((ratio < 0.02).mean())
    out["score_swing_when_200d_and_50d_both_flip"] = 6.0
    return out


def baseline_labels(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=table.index)
    out["regime4"] = table["regime"]
    out["ma200"] = np.where(table["spy"] > table["sma200"], "above 200d", "below 200d")
    out["vix4"] = pd.cut(table["vix"], [-np.inf, *VIX_BUCKETS, np.inf], labels=VIX_LABELS, right=False).astype(object)
    return out


def regime_description(labels: pd.Series, fwd: pd.DataFrame, lags: dict[str, int], order: tuple[str, ...]) -> dict[str, Any]:
    """Share of time, spells, flips and forward statistics by label (one sample, one frequency)."""
    n = len(labels)
    years = (labels.index[-1] - labels.index[0]).days / 365.25
    rs = runs(list(labels))
    out: dict[str, Any] = {"observations": n, "flips_per_year": (len(rs) - 1) / years if years > 0 else None, "labels": {}}
    for lab in order:
        mask = (labels == lab).to_numpy()
        spells = [r[2] for r in rs if r[0] == lab]
        entry: dict[str, Any] = {"share": float(mask.mean()), "n": int(mask.sum()), "spells": len(spells),
                                 "mean_spell": float(np.mean(spells)) if spells else None,
                                 "longest_spell": int(max(spells)) if spells else None}
        for col in ("ret_1m", "ret_3m", "ret_12m"):
            h = col.split("_")[1]
            m, t, k = subset_mean_t(fwd[col].to_numpy(), mask, lags[h])
            entry[col] = m
            entry[f"{col}_t"] = t
            entry[f"{col}_n"] = k
        for col in ("vol_1m", "vol_3m", "mdd_3m", "mdd_12m"):
            vals = fwd[col].to_numpy(dtype=float)[mask]
            vals = vals[~np.isnan(vals)]
            entry[col] = float(vals.mean()) if len(vals) else None
            entry[f"{col}_median"] = float(np.median(vals)) if len(vals) else None
        out["labels"][lab] = entry
    return out


def risk_separation(labels_df: pd.DataFrame, fwd: pd.DataFrame) -> dict[str, Any]:
    """Adjusted R-squared of forward risk grouped by each labelling, plus the monotone-volatility check."""
    out: dict[str, Any] = {}
    for y in ("vol_1m", "vol_3m", "mdd_3m", "mdd_12m"):
        out[y] = {name: group_adj_r2(fwd[y].to_numpy(), labels_df[name].to_numpy()) for name in ("regime4", "ma200", "vix4")}
    means = fwd.groupby(labels_df["regime4"])["vol_3m"].mean()
    present = [means.get(r) for r in REGIMES if r in means.index]
    out["vol_3m_by_regime"] = {r: float(means[r]) for r in REGIMES if r in means.index}
    out["vol_3m_monotone"] = bool(all(a < b for a, b in zip(present, present[1:]))) and len(present) == 4
    return out


def returns_separation(labels: pd.Series, fwd: pd.DataFrame) -> dict[str, Any]:
    """Forward SPY returns: stress (BEAR+CRISIS) minus BULL, and each regime against all other months."""
    out: dict[str, Any] = {}
    stress = labels.isin(STRESS).to_numpy()
    bull = (labels == "BULL").to_numpy()
    for h, lag in (("1m", 3), ("3m", 3), ("12m", 12)):
        d, t, na, nb = diff_t(fwd[f"ret_{h}"].to_numpy(), stress, bull, lag)
        out[f"stress_minus_bull_{h}"] = {"diff": d, "t": t, "n_stress": na, "n_bull": nb}
    return out


def input_eras(table: pd.DataFrame) -> pd.DataFrame:
    """Which inputs existed at each date, as an era label such as 'spy_trend+vix+fred_yields+ten_year'."""
    present = table["missing"].map(lambda m: "+".join(i for i in INPUTS if i not in set(filter(None, m.split(",")))))
    return present.rename("era")


def theoretical_score_range(era: str) -> tuple[float, float]:
    """(minimum, maximum) summed score possible when only the inputs named in ``era`` (joined by '+') exist."""
    keys = [k for k in era.split("+") if k in INPUT_POINTS]
    return sum(INPUT_POINTS[k][1] for k in keys), sum(INPUT_POINTS[k][0] for k in keys)


def score_range_by_era(history: MacroHistory, table: pd.DataFrame) -> dict[str, Any]:
    """Observed minimum and maximum summed score per input era, and the regime shares: the meaning of the cut-offs."""
    eras = input_eras(table)
    out: dict[str, Any] = {}
    for era, g in table.groupby(eras):
        shares = g["regime"].value_counts(normalize=True)
        lo, hi = theoretical_score_range(era)
        out[era] = {"first": str(g.index[0].date()), "last": str(g.index[-1].date()), "months": int(len(g)),
                    "possible_min": lo, "possible_max": hi,
                    "bull_cutoff_position": float((REGIME_THRESHOLDS[0][0] - lo) / (hi - lo)),  # 0 = lowest possible score, 1 = highest
                    "score_min": float(g["regime_score"].min()), "score_max": float(g["regime_score"].max()),
                    "score_mean": float(g["regime_score"].mean()),
                    "shares": {r: float(shares.get(r, 0.0)) for r in REGIMES}}
    return out


def core_only_history(history: MacroHistory) -> MacroHistory:
    return MacroHistory(series={k: v for k, v in history.series.items() if k in CORE_SERIES}, errors={}, fetched_at="")


def core_vs_full(history: MacroHistory, table: pd.DataFrame) -> dict[str, Any]:
    """The regime from the always-available inputs only (SPY, VIX, curve, 10Y) against the full-input regime."""
    core = macro_table(core_only_history(history), list(table.index))
    full_era = table.index[input_eras(table) == "+".join(INPUTS)]
    joined = pd.DataFrame({"full": table["regime"], "core": core["regime"]}).dropna()
    out: dict[str, Any] = {"core_shares": {r: float((core["regime"] == r).mean()) for r in REGIMES},
                           "full_shares": {r: float((table["regime"] == r).mean()) for r in REGIMES}}
    if len(full_era):
        sub = joined.loc[joined.index.isin(full_era)]
        out["full_era_start"] = str(full_era[0].date())
        out["agreement_in_full_era"] = float((sub["full"] == sub["core"]).mean())
        out["confusion_full_era"] = {f: {c: int(((sub["full"] == f) & (sub["core"] == c)).sum()) for c in REGIMES} for f in REGIMES}
        out["core_shares_full_era"] = {r: float((sub["core"] == r).mean()) for r in REGIMES}
        out["full_shares_full_era"] = {r: float((sub["full"] == r).mean()) for r in REGIMES}
        out["max_share_gap_points"] = float(max(abs(out["core_shares_full_era"][r] - out["full_shares_full_era"][r]) for r in REGIMES) * 100)
        out["one_step_or_less"] = float(np.mean([abs(REGIMES.index(f) - REGIMES.index(c)) <= 1 for f, c in zip(sub["full"], sub["core"])]))
    return out, core


def first_dates(table: pd.DataFrame, history: MacroHistory) -> dict[str, Any]:
    out = {k: str(v.index[0].date()) for k, v in history.series.items()}
    avail = {}
    for i in INPUTS:
        ok = table.index[~table["missing"].map(lambda m, i=i: i in set(filter(None, m.split(","))))]
        avail[i] = str(ok[0].date()) if len(ok) else None
    return {"series_start": out, "input_in_snapshot_from": avail}


def analyze_classifier(history: MacroHistory) -> dict[str, Any]:
    """Question 1."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    spy = history.series["spy"]
    report: dict[str, Any] = {}
    for freq, name in (("M", "monthly"), ("W", "weekly")):
        dates = sample_dates(spy, freq)
        table = macro_table(history, dates)
        fwd = forward_spy(spy, list(table.index))
        labels_df = baseline_labels(table)
        cuts = table["opportunity_score"].quantile([1 / 3, 2 / 3]).to_numpy()
        labels_df["opp3"] = np.where(table["opportunity_score"] <= cuts[0], "low", np.where(table["opportunity_score"] <= cuts[1], "mid", "high"))
        lags = {"1m": 3, "3m": 3, "12m": 12} if freq == "M" else {"1m": 4, "3m": 13, "12m": 52}
        cut = len(table) // 2
        parts = {"whole": slice(None), "first_half": slice(0, cut), "second_half": slice(cut, None)}
        full_start = int(np.argmax((input_eras(table) == "+".join(INPUTS)).to_numpy()))
        parts["full_input_era"] = slice(full_start, None)
        res: dict[str, Any] = {"first": str(table.index[0].date()), "last": str(table.index[-1].date()),
                               "split": str(table.index[cut].date())}
        for part, sl in parts.items():
            sub_t, sub_f, sub_l = table.iloc[sl], fwd.iloc[sl], labels_df.iloc[sl]
            res[part] = {
                "description": regime_description(sub_l["regime4"], sub_f, lags, REGIMES),
                "risk": risk_separation(sub_l, sub_f),
                "returns": returns_separation(sub_l["regime4"], sub_f),
                "ma200_description": regime_description(sub_l["ma200"], sub_f, lags, ("above 200d", "below 200d")),
                "vix_description": regime_description(sub_l["vix4"], sub_f, lags, VIX_LABELS),
                "opportunity_description": regime_description(sub_l["opp3"], sub_f, lags, ("low", "mid", "high")),
            }
        res["_table"], res["_fwd"], res["_labels"] = table, fwd, labels_df
        report[name] = res
    m = report["monthly"]
    table = m["_table"]
    report["inputs"] = first_dates(table, history)
    report["eras"] = score_range_by_era(history, table)
    report["core_vs_full"], core = core_vs_full(history, table)
    # Does the core-only label (identical meaning over time) describe forward risk as well as the full one?
    core_labels = pd.DataFrame({"regime4": core["regime"].reindex(table.index)})
    report["core_risk"] = {"vol_3m": group_adj_r2(m["_fwd"]["vol_3m"].to_numpy(), core_labels["regime4"].to_numpy()),
                           "mdd_3m": group_adj_r2(m["_fwd"]["mdd_3m"].to_numpy(), core_labels["regime4"].to_numpy()),
                           "full_vol_3m": group_adj_r2(m["_fwd"]["vol_3m"].to_numpy(), table["regime"].to_numpy()),
                           "full_mdd_3m": group_adj_r2(m["_fwd"]["mdd_3m"].to_numpy(), table["regime"].to_numpy())}
    report["acceptance_1"] = acceptance_q1(m)
    report["flip_drivers_monthly"] = flip_drivers(table)
    report["threshold_sensitivity"] = threshold_sensitivity(table, m["_fwd"])
    return report


def relabel(scores: pd.Series, shift: float) -> pd.Series:
    """Regime labels from the summed score with every cut-off moved by ``shift`` points (descriptive sensitivity only)."""
    cuts = [(c + shift, n) for c, n in REGIME_THRESHOLDS]
    out = []
    for v in scores:
        out.append(next((n for c, n in cuts if v >= c), "CRISIS"))
    return pd.Series(out, index=scores.index, dtype=object)


def linear_adj_r2(y: np.ndarray, x: np.ndarray) -> float:
    ok = ~np.isnan(y) & ~np.isnan(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 10 or np.std(x) == 0:
        return float("nan")
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return 1 - (1 - r2) * (n - 1) / (n - 2)


def threshold_sensitivity(table: pd.DataFrame, fwd: pd.DataFrame) -> dict[str, Any]:
    """How the label's behaviour moves when the cut-offs move, and what fewer states or the raw score would give.

    Descriptive only: nothing is chosen from it (the cut-offs have no recorded origin, and picking the best
    shift here would be tuning on the sample).
    """
    cut = len(table) // 2
    out: dict[str, Any] = {}
    for part, sl in (("whole", slice(None)), ("first_half", slice(0, cut)), ("second_half", slice(cut, None))):
        t, f = table.iloc[sl], fwd.iloc[sl]
        rows: dict[str, Any] = {}
        for shift in (-1.0, -0.5, 0.0, 0.5, 1.0):
            lab = relabel(t["regime_score"], shift)
            rows[f"shift_{shift:+.1f}"] = {
                "shares": {r: float((lab == r).mean()) for r in REGIMES},
                "flips_per_year": (len(runs(list(lab))) - 1) / ((lab.index[-1] - lab.index[0]).days / 365.25),
                "adj_r2_vol_3m": group_adj_r2(f["vol_3m"].to_numpy(), lab.to_numpy()),
                "adj_r2_mdd_3m": group_adj_r2(f["mdd_3m"].to_numpy(), lab.to_numpy())}
        two = np.where(t["regime"].isin(STRESS), "stress", "calm").astype(object)
        rows["two_states"] = {"adj_r2_vol_3m": group_adj_r2(f["vol_3m"].to_numpy(), two), "adj_r2_mdd_3m": group_adj_r2(f["mdd_3m"].to_numpy(), two)}
        rows["score_linear"] = {"adj_r2_vol_3m": linear_adj_r2(f["vol_3m"].to_numpy(), t["regime_score"].to_numpy()),
                                "adj_r2_mdd_3m": linear_adj_r2(f["mdd_3m"].to_numpy(), t["regime_score"].to_numpy())}
        rows["vix_linear"] = {"adj_r2_vol_3m": linear_adj_r2(f["vol_3m"].to_numpy(), t["vix"].to_numpy()),
                              "adj_r2_mdd_3m": linear_adj_r2(f["mdd_3m"].to_numpy(), t["vix"].to_numpy())}
        out[part] = rows
    return out


def acceptance_q1(monthly: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    risk_ok = True
    detail: dict[str, Any] = {}
    for part in ("whole", "first_half", "second_half"):
        r = monthly[part]["risk"]
        beats = all(r[y]["regime4"] > r[y]["ma200"] and r[y]["regime4"] > r[y]["vix4"] for y in ("vol_3m", "mdd_3m"))
        detail[part] = {"beats_both_baselines": bool(beats), "monotone_vol": bool(r["vol_3m_monotone"]),
                        "adj_r2": {y: r[y] for y in ("vol_3m", "mdd_3m")}}
        risk_ok = risk_ok and beats and r["vol_3m_monotone"]
    out["1a_risk"] = {"passes": bool(risk_ok), "detail": detail}
    d12 = {p: monthly[p]["returns"]["stress_minus_bull_12m"] for p in ("whole", "first_half", "second_half")}
    contrarian = (d12["whole"]["t"] is not None and d12["whole"]["t"] >= 2 and d12["first_half"]["diff"] > 0
                  and d12["second_half"]["diff"] > 0)
    opposite = (d12["whole"]["t"] is not None and d12["whole"]["t"] <= -2 and d12["first_half"]["diff"] < 0
                and d12["second_half"]["diff"] < 0)
    out["1b_returns"] = {"contrarian_passes": bool(contrarian), "opposite_passes": bool(opposite), "stress_minus_bull_12m": d12}
    return out


# ------------------------------------------------------------------ the S&P 500 panel


# The timing profiles and cut-offs shipped until D-40 (October 2026): the "shipped" setup this study tested. Frozen here
# so the study stays reproducible after config/agent_weights.json moved to one profile without the macro score.
STUDIED_PROFILES: dict[str, dict[str, float]] = {
    "BULL": {"momentum": 0.30, "volume": 0.15, "options": 0.15, "macro": 0.25, "pullback": 0.15, "sentiment": 0.0},
    "CORRECTION": {"momentum": 0.25, "volume": 0.15, "options": 0.15, "macro": 0.30, "pullback": 0.15, "sentiment": 0.0},
    "BEAR": {"momentum": 0.20, "volume": 0.15, "options": 0.15, "macro": 0.35, "pullback": 0.15, "sentiment": 0.0},
    "CRISIS": {"momentum": 0.15, "volume": 0.10, "options": 0.15, "macro": 0.45, "pullback": 0.15, "sentiment": 0.0},
}
STUDIED_THRESHOLDS = {"strong_buy": 45.0, "buy": 15.0, "sell": -15.0, "strong_sell": -45.0}


def shipped_profiles() -> dict[str, Any]:
    """The setup the study tested (the profiles and cut-offs shipped before D-40), with the rest from config/agent_weights.json."""
    raw = json.loads(CONFIG_PATH.read_text())
    profiles = {r: dict(STUDIED_PROFILES[r]) for r in REGIMES}
    flat = {k: float(np.mean([profiles[r].get(k, 0.0) for r in REGIMES])) for k in profiles["BULL"]}
    return {"profiles": profiles, "flat": flat, "thresholds": dict(STUDIED_THRESHOLDS),
            "floor": float(raw.get("confidence_floor", 0.4)), "fallback": raw.get("fallback_regime", "CORRECTION"),
            "enabled": raw.get("enabled", {}), "scales": raw.get("score_scales", {})}


def macro_rows(history: MacroHistory, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Same columns as core.all_agents_study.macro_by_date, from the long cached history (no second download)."""
    agent = MacroRegimeAgent()
    rows = []
    for date in dates:
        snap = compute_macro_snapshot(history, date)
        a = agent.evaluate_with_data(snap)
        ok = a.data_quality != UNAVAILABLE and snap.data_quality != "ESTIMATED"
        rows.append({"date": date, "macro_score": a.score if ok else np.nan, "macro_conf": a.confidence if ok else 0.0,
                     "regime": snap.regime if ok else None, "vix": snap.vix_level})
    return pd.DataFrame(rows)


def load_panel(history: MacroHistory, rebuild: bool = False) -> pd.DataFrame:
    """The scored S&P 500 panel of the all-agents study, built from the cached parquet panels."""
    cache = settings.DATA_DIR / "regime_study" / "scored_panel.parquet"
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    value_path = settings.DATA_DIR / "value_panel.parquet"
    timing_path = settings.DATA_DIR / "timing_panel.parquet"
    insider_path = settings.DATA_DIR / "insider_panel.parquet"
    if not value_path.exists() or not timing_path.exists():
        raise RuntimeError("Missing value or timing panel: run core.value_study and core.all_agents_study first.")
    panel = pd.read_parquet(value_path)
    if insider_path.exists():
        panel = panel.merge(pd.read_parquet(insider_path), on=["date", "ticker"], how="left")
    else:
        for col in ("ins_score", "ins_conf", "ins_quality"):
            panel[col] = np.nan
    panel = add_insider_composite(panel)
    dates = sorted(panel["date"].unique())
    macro = macro_rows(history, [pd.Timestamp(d) for d in dates])
    panel = panel.merge(pd.read_parquet(timing_path), on=["date", "ticker"], how="left").merge(macro, on="date", how="left")
    panel = add_composites(panel)
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache, index=False)
    return panel


def load_forward_frames(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forward 1, 3 and 12 month total returns by company, from the cached daily price files (no network)."""
    dates = [pd.Timestamp(d) for d in sorted(panel["date"].unique())]
    folder = settings.DATA_DIR / "history"
    prices: dict[str, pd.DataFrame] = {}
    for t in sorted(set(panel["ticker"]) | {"SPY"}):
        path = folder / f"{t}.parquet"
        if path.exists():
            prices[t] = pd.read_parquet(path, columns=["adj_close"])
    level = month_end_panel(prices, dates + [dates[-1] + pd.offsets.MonthEnd(k) for k in range(1, 13)])
    level.loc[level.index > prices["SPY"].index[-1] + pd.Timedelta(days=3)] = np.nan
    return forward_returns(level, 1), forward_returns(level, 3), forward_returns(level, 12)


# ------------------------------------------------------------------ timing composite, vectorised


def timing_composite(frame: pd.DataFrame, cfg: dict[str, Any], per_regime: bool = True, include_macro: bool = True,
                     macro_frozen: bool = False) -> pd.Series:
    """The timing composite of agents.scoring.aggregate for a whole frame at once.

    ``per_regime`` False uses the flat profile; ``include_macro`` False drops the macro agent. Same
    steps as ``aggregate``: tanh normalisation, weight times (floor + (1 - floor) * confidence), weighted mean,
    clip, round to one decimal. Rows where no agent has data are NaN. The equality with aggregate is tested.
    """
    floor = cfg["floor"]
    scales = {**SCALES, **cfg["scales"]}
    regimes = frame["regime"].where(frame["regime"].isin(REGIMES), cfg["fallback"]) if "regime" in frame else pd.Series(cfg["fallback"], index=frame.index)
    num = np.zeros(len(frame))
    den = np.zeros(len(frame))
    for key in TIMING_KEYS:
        if key == "macro" and not include_macro:
            continue
        if not cfg["enabled"].get(key, True):
            continue
        score_col, conf_col = SCORE_COLUMNS[key]
        score = frame[score_col].to_numpy(dtype=float)
        conf = np.clip(frame[conf_col].fillna(0.0).to_numpy(dtype=float), 0.0, 100.0)
        if per_regime:
            w = regimes.map(lambda r, k=key: cfg["profiles"][r].get(k, 0.0)).to_numpy(dtype=float)
        else:
            w = np.full(len(frame), cfg["flat"].get(key, 0.0))
        ok = ~np.isnan(score) & (w > 0)
        eff = np.round(np.where(ok, w * (floor + (1 - floor) * conf / 100.0), 0.0), 4)  # aggregate stores 4 decimals
        norm = np.tanh(np.nan_to_num(score) / max(scales.get(key, 50.0), 1e-6)) * 100.0
        num += norm * eff
        den += eff
    comp = np.where(den > 0, np.clip(num / np.where(den > 0, den, 1.0), -100.0, 100.0), np.nan)
    return pd.Series(np.round(comp, 1), index=frame.index)


def label_series(comp: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    lab = pd.Series(np.where(comp >= thresholds["buy"], "FAVORABLE", np.where(comp <= thresholds["sell"], "UNFAVORABLE", "NEUTRAL")),
                    index=comp.index, dtype=object)
    lab[comp.isna()] = None
    return lab


def matched_label(comp: pd.Series, reference: pd.Series) -> pd.Series:
    """Label whose FAVORABLE and UNFAVORABLE shares equal those of ``reference`` (removes the threshold effect)."""
    ref = reference.dropna()
    fav, unf = float((ref == "FAVORABLE").mean()), float((ref == "UNFAVORABLE").mean())
    hi, lo = comp.quantile(1 - fav), comp.quantile(unf)
    lab = pd.Series(np.where(comp >= hi, "FAVORABLE", np.where(comp <= lo, "UNFAVORABLE", "NEUTRAL")), index=comp.index, dtype=object)
    lab[comp.isna()] = None
    return lab


# ------------------------------------------------------------------ monthly series from the panel


def ic_series(panel: pd.DataFrame, fwd: pd.DataFrame, column: str, subset: pd.Series | None = None, min_names: int = 50) -> pd.Series:
    """Monthly cross-sectional rank IC of ``column`` against the forward returns ``fwd``."""
    frame = panel if subset is None else panel[subset]
    frame = frame.dropna(subset=[column])
    out: dict[pd.Timestamp, float] = {}
    for date, g in frame.groupby("date"):
        if date not in fwd.index:
            continue
        x = g.set_index("ticker")[column].astype(float)
        r = fwd.loc[date].reindex(x.index)
        ok = x.notna() & r.notna()
        if ok.sum() >= min_names:
            out[date] = float(x[ok].rank().corr(r[ok].rank()))
    return pd.Series(out, dtype=float).sort_index()


def group_return_series(panel: pd.DataFrame, fwd: pd.DataFrame, mask: pd.Series, min_names: int = 5) -> pd.Series:
    """Monthly equal-weight mean forward return of the rows in ``mask``."""
    out: dict[pd.Timestamp, float] = {}
    for date, g in panel[mask].groupby("date"):
        if date not in fwd.index:
            continue
        r = fwd.loc[date].reindex(g["ticker"]).dropna()
        if len(r) >= min_names:
            out[date] = float(r.mean())
    return pd.Series(out, dtype=float).sort_index()


def label_spread_series(panel: pd.DataFrame, fwd1: pd.DataFrame, label: pd.Series, is_buy: pd.Series, min_names: int = 2) -> pd.Series:
    fav = group_return_series(panel, fwd1, is_buy & (label == "FAVORABLE"), min_names)
    unf = group_return_series(panel, fwd1, is_buy & (label == "UNFAVORABLE"), min_names)
    return (fav - unf).dropna()


RULES = (
    ("top20_not_unfavorable", "Owner top 20%, held only while timing not unfavorable"),
    ("top20_favorable", "Owner top 20%, held only while timing favorable"),
    ("buys_not_unfavorable", "Owner BUY names, held only while timing not unfavorable"),
    ("buys_favorable", "Owner BUY names, held only while timing favorable"),
)


def rule_series(panel: pd.DataFrame, fwd1: pd.DataFrame, label: pd.Series, is_buy: pd.Series) -> dict[str, pd.Series]:
    """Monthly returns of the all-agents study's label rules, for one labelling (its own portfolio_returns)."""
    p = panel.copy()
    p["timing_ok"] = np.where(label.eq("UNFAVORABLE"), -1.0, 1.0)
    p["timing_favorable"] = np.where(label.eq("FAVORABLE"), 1.0, -1.0)
    return {
        "top20_not_unfavorable": portfolio_returns(p, fwd1, "composite", "top_q", min_names=50, keep_if="timing_ok")[0],
        "top20_favorable": portfolio_returns(p, fwd1, "composite", "top_q", min_names=50, keep_if="timing_favorable")[0],
        "buys_not_unfavorable": portfolio_returns(p, fwd1, "composite", "all", mask=is_buy, keep_if="timing_ok")[0],
        "buys_favorable": portfolio_returns(p, fwd1, "composite", "all", mask=is_buy, keep_if="timing_favorable")[0],
    }


def variant_series(panel: pd.DataFrame, fwd1: pd.DataFrame, comp: pd.Series, label: pd.Series, is_buy: pd.Series) -> dict[str, pd.Series]:
    p = panel.assign(_comp=comp.to_numpy())
    out = {"ic_1m": ic_series(p, fwd1, "_comp"), "label_spread": label_spread_series(panel, fwd1, label, is_buy)}
    out.update(rule_series(panel, fwd1, label, is_buy))
    return out


def regime_means(series: pd.Series, regime_by_date: pd.Series, shifted: bool = False) -> dict[str, Any]:
    """Mean of a monthly series by the regime at the decision date (``shifted``: the series is dated one month later)."""
    idx = series.index - pd.offsets.MonthEnd(1) if shifted else series.index
    reg = regime_by_date.reindex(idx).to_numpy()
    return {r: (float(series.to_numpy()[reg == r].mean()) if (reg == r).any() else None) for r in REGIMES}


def pooled_label_buckets(panel: pd.DataFrame, fwd1: pd.DataFrame, label: pd.Series, is_buy: pd.Series) -> dict[str, Any]:
    """Owner BUYs by timing label, pooled over months as in the all-agents study: annualised 1-month return and months."""
    dates = [pd.Timestamp(d) for d in sorted(panel["date"].unique())]
    series = {k: group_return_series(panel, fwd1, is_buy & (label == k), 2).reindex(dates).to_numpy() for k in ("FAVORABLE", "NEUTRAL", "UNFAVORABLE")}
    out: dict[str, Any] = {}
    for k, x in series.items():
        ok = ~np.isnan(x)
        out[k] = {"months": int(ok.sum()), "annualized_1m": float((1 + np.nanmean(x)) ** 12 - 1) if ok.any() else None}
    d, t, na, nb = diff_t(series["FAVORABLE"], ~np.isnan(series["FAVORABLE"]), ~np.isnan(series["UNFAVORABLE"]), 3, y=series["UNFAVORABLE"])
    out["favorable_minus_unfavorable_monthly"] = {"diff": d, "t": t, "n_fav": na, "n_unf": nb}
    return out


# ------------------------------------------------------------------ Q2: timing signals by regime


def stats_by_regime(series: pd.Series, dates: list[pd.Timestamp], regime_by_date: pd.Series, lags: int, cut: pd.Timestamp) -> dict[str, Any]:
    """Mean of a monthly series in each regime with Newey-West t on the full month index, and each half."""
    full = series.reindex(dates).to_numpy(dtype=float)
    reg = regime_by_date.reindex(dates).to_numpy()
    first = np.array([d < cut for d in dates])
    out: dict[str, Any] = {}
    for name, mask in [(r, reg == r) for r in REGIMES] + [("ALL", np.ones(len(dates), dtype=bool))]:
        m, t, n = subset_mean_t(full, mask, lags)
        m1, _, n1 = subset_mean_t(full, mask & first, lags)
        m2, _, n2 = subset_mean_t(full, mask & ~first, lags)
        out[name] = {"mean": m, "t": t, "months": n, "first_half": m1, "months_first": n1, "second_half": m2, "months_second": n2}
    stress, bull = np.isin(reg, STRESS), reg == "BULL"
    d, td, na, nb = diff_t(full, stress, bull, lags)
    d1, _, _, _ = diff_t(full, stress & first, bull & first, lags)
    d2, _, _, _ = diff_t(full, stress & ~first, bull & ~first, lags)
    out["STRESS_minus_BULL"] = {"diff": d, "t": td, "months_stress": na, "months_bull": nb, "first_half": d1, "second_half": d2}
    return out


def timing_by_regime(panel: pd.DataFrame, fwd1: pd.DataFrame, fwd12: pd.DataFrame, regime_by_date: pd.Series, is_buy: pd.Series,
                     dates: list[pd.Timestamp], cut: pd.Timestamp) -> dict[str, Any]:
    out: dict[str, Any] = {"ic_1m": {}, "ic_12m": {}}
    mn = month_number(pd.DatetimeIndex(dates))
    reg = regime_by_date.reindex(dates).to_numpy()
    stress_eps = episodes(mn[np.isin(reg, STRESS)])
    out["stress_ic_12m_by_episode"] = {}
    for key, col in (("momentum", "mom_score"), ("volume", "vol_score"), ("pullback", "pb_score"), ("timing", "timing_score")):
        out["ic_1m"][key] = stats_by_regime(ic_series(panel, fwd1, col), dates, regime_by_date, 3, cut)
        ic12 = ic_series(panel, fwd12, col)
        out["ic_12m"][key] = stats_by_regime(ic12, dates, regime_by_date, 12, cut)
        x = ic12.reindex(dates).to_numpy(dtype=float)
        per_episode = []
        for e in stress_eps:
            sel = np.isin(mn, e) & np.isin(reg, STRESS) & ~np.isnan(x)
            if sel.any():
                per_episode.append({"first": f"{pd.Timestamp(dates[list(mn).index(e[0])]):%Y-%m}", "months": int(sel.sum()), "mean_ic_12m": float(x[sel].mean())})
        out["stress_ic_12m_by_episode"][key] = {"episodes": per_episode, "negative": int(sum(1 for p in per_episode if p["mean_ic_12m"] < 0)), "of": len(per_episode)}
    # Pullback-setup buckets within each regime (forward 1 month; all names and owner BUYs).
    buckets: dict[str, Any] = {}
    for scope, scope_mask in (("all", pd.Series(True, index=panel.index)), ("owner_buys", is_buy)):
        per_setup = {s: group_return_series(panel, fwd1, scope_mask & (panel["pb_setup"] == s), 5) for s in SETUPS}
        per_setup12 = {s: group_return_series(panel, fwd12, scope_mask & (panel["pb_setup"] == s), 5) for s in SETUPS}
        entry: dict[str, Any] = {}
        for s in SETUPS:
            entry[s] = stats_by_regime(per_setup[s], dates, regime_by_date, 3, cut)
            by12 = stats_by_regime(per_setup12[s], dates, regime_by_date, 12, cut)
            for r in (*REGIMES, "ALL"):
                entry[s][r]["mean_12m"] = by12[r]["mean"]
        spread = (per_setup["pullback_in_uptrend"] - per_setup["extended"]).dropna()
        entry["pullback_minus_extended"] = stats_by_regime(spread, dates, regime_by_date, 3, cut)
        buckets[scope] = entry
    out["setup_by_regime"] = buckets
    return out


def profiles_vs_flat(panel: pd.DataFrame, fwd1: pd.DataFrame, cfg: dict[str, Any], is_buy: pd.Series, regime_by_date: pd.Series,
                     cut: pd.Timestamp) -> tuple[dict[str, Any], dict[str, pd.Series]]:
    """The decisive test of question 2 (and, with the same machinery, the no-macro variants of question 5)."""
    th = cfg["thresholds"]
    variants: dict[str, tuple[pd.Series, pd.Series]] = {}
    comp = timing_composite(panel, cfg, per_regime=True, include_macro=True)
    variants["shipped"] = (comp, label_series(comp, th))
    comp = timing_composite(panel, cfg, per_regime=False, include_macro=True)
    variants["flat"] = (comp, label_series(comp, th))
    comp = timing_composite(panel, cfg, per_regime=True, include_macro=False)
    variants["no_macro_profile"] = (comp, label_series(comp, th))
    variants["no_macro_profile_matched"] = (comp, matched_label(comp, variants["shipped"][1]))
    comp = timing_composite(panel, cfg, per_regime=False, include_macro=False)
    variants["no_macro_flat"] = (comp, label_series(comp, th))
    variants["no_macro_flat_matched"] = (comp, matched_label(comp, variants["shipped"][1]))
    series: dict[str, dict[str, pd.Series]] = {}
    out: dict[str, Any] = {"label_shares": {}, "series_means": {}}
    for name, (c, lab) in variants.items():
        series[name] = variant_series(panel, fwd1, c, lab, is_buy)
        out["label_shares"][name] = {k: float((lab == k).sum() / lab.notna().sum()) for k in ("FAVORABLE", "NEUTRAL", "UNFAVORABLE")}
        out["series_means"][name] = {k: (float(v.mean()) if len(v) else None) for k, v in series[name].items()}
        out.setdefault("label_buckets_in_owner_buys", {})[name] = pooled_label_buckets(panel, fwd1, lab, is_buy)
    metrics = ["ic_1m", "label_spread"] + [k for k, _ in RULES]

    def compare(a: str, b: str) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for m in metrics:
            r = paired_diff(series[a][m], series[b][m], 3, cut)
            r["passes"] = passes_both_halves(r)
            sample = series[a][m]
            r["by_regime"] = regime_means(series[a][m] - series[b][m].reindex(series[a][m].index), regime_by_date,
                                           shifted=m in dict(RULES))
            res[m] = r
        return res

    out["shipped_vs_flat"] = compare("shipped", "flat")
    out["profiles_justified"] = bool(all(out["shipped_vs_flat"][m]["passes"] for m in metrics))
    out["shipped_vs_no_macro_profile"] = compare("shipped", "no_macro_profile")
    out["shipped_vs_no_macro_profile_matched"] = compare("shipped", "no_macro_profile_matched")
    out["shipped_vs_no_macro_flat"] = compare("shipped", "no_macro_flat")
    out["shipped_vs_no_macro_flat_matched"] = compare("shipped", "no_macro_flat_matched")
    out["macro_helps_somewhere"] = {k: [m for m in metrics if out[k][m]["mean"] is not None and out[k][m]["mean"] > 0 and (out[k][m]["t"] or 0) >= 2]
                                    for k in ("shipped_vs_no_macro_profile", "shipped_vs_no_macro_profile_matched",
                                              "shipped_vs_no_macro_flat", "shipped_vs_no_macro_flat_matched")}
    out["flat_profile"] = cfg["flat"]
    out["shipped_composite"] = variants["shipped"][0]
    return out, {n: v[0] for n, v in variants.items()} | {f"label_{n}": v[1] for n, v in variants.items()}


# ------------------------------------------------------------------ Q3: the owner score by regime


def owner_by_regime(panel: pd.DataFrame, fwd1: pd.DataFrame, fwd12: pd.DataFrame, regime_by_date: pd.Series,
                    dates: list[pd.Timestamp], cut: pd.Timestamp) -> dict[str, Any]:
    ic12 = ic_series(panel, fwd12, "composite")
    ic1 = ic_series(panel, fwd1, "composite")
    buy12 = group_return_series(panel, fwd12, panel["owner_decision"].isin(BUYS), 10)
    sell12 = group_return_series(panel, fwd12, panel["owner_decision"].isin(SELLS), 10)
    all12 = group_return_series(panel, fwd12, panel["owner_decision"].notna() & (panel["owner_decision"] != "NOT_RATED"), 10)
    spread = (buy12 - sell12).dropna()
    out = {"ic_12m": stats_by_regime(ic12, dates, regime_by_date, 12, cut), "ic_1m": stats_by_regime(ic1, dates, regime_by_date, 3, cut),
           "buy_minus_sell_12m": stats_by_regime(spread, dates, regime_by_date, 12, cut),
           "buy_12m": stats_by_regime(buy12, dates, regime_by_date, 12, cut),
           "universe_12m": stats_by_regime(all12, dates, regime_by_date, 12, cut)}
    verdict: dict[str, Any] = {}
    for r in REGIMES:
        row = {}
        for metric in ("ic_12m", "buy_minus_sell_12m"):
            s = out[metric][r]
            thin = s["months_first"] < THIN_MONTHS or s["months_second"] < THIN_MONTHS
            ok = (s["mean"] is not None and s["mean"] > 0 and (s["first_half"] or 0) > 0 and (s["second_half"] or 0) > 0
                  and s["t"] == s["t"] and s["t"] >= 2)
            row[metric] = "too thin" if thin else ("works" if ok else "does not pass")
        verdict[r] = row
    out["verdict"] = verdict
    return out


# ------------------------------------------------------------------ Q4: buying owner BUYs in stress


def entry_by_regime(panel: pd.DataFrame, fwd1: pd.DataFrame, fwd12: pd.DataFrame, regime_by_date: pd.Series,
                    dates: list[pd.Timestamp], cut: pd.Timestamp) -> dict[str, Any]:
    is_buy = panel["owner_decision"].isin(BUYS)
    rated = panel["owner_decision"].notna() & (panel["owner_decision"] != "NOT_RATED")
    b1 = group_return_series(panel, fwd1, is_buy, 5)
    b12 = group_return_series(panel, fwd12, is_buy, 5)
    u1 = group_return_series(panel, fwd1, rated, 5)
    u12 = group_return_series(panel, fwd12, rated, 5)
    first = np.array([d < cut for d in dates])
    reg = regime_by_date.reindex(dates).to_numpy()
    mn = month_number(pd.DatetimeIndex(dates))
    ann = lambda m: None if m is None or m != m else float((1 + m) ** 12 - 1)  # noqa: E731
    out: dict[str, Any] = {"entry_by_regime": {}}
    for r in REGIMES:
        mask = reg == r
        x1, x12 = b1.reindex(dates).to_numpy(), b12.reindex(dates).to_numpy()
        m1, t1, n1 = subset_mean_t(x1, mask, 3)
        m12, t12, n12 = subset_mean_t(x12, mask, 12)
        h1, _, nh1 = subset_mean_t(x1, mask & first, 3)
        h2, _, nh2 = subset_mean_t(x1, mask & ~first, 3)
        compact = newey_west_t(b1.reindex(dates).to_numpy()[mask], 3)  # the all-agents study's own way of computing t
        u1m, _, _ = subset_mean_t(u1.reindex(dates).to_numpy(), mask, 3)
        u12m, _, _ = subset_mean_t(u12.reindex(dates).to_numpy(), mask, 12)
        ep = episodes(mn[mask])
        rr = runs([bool(v) for v in mask])
        out["entry_by_regime"][r] = {
            "fwd_12m": m12, "fwd_12m_t": t12, "fwd_1m_ann": ann(m1), "t": t1, "t_compact_all_agents_way": compact,
            "n_months": int(mask.sum()), "months_with_fwd_1m": n1, "months_with_fwd_12m": n12,
            "episodes": len(ep), "runs": int(sum(1 for lab, _, _ in rr if lab)),
            "first_half": ann(h1), "second_half": ann(h2), "months_first": int((mask & first).sum()), "months_second": int((mask & ~first).sum()),
            "universe_fwd_1m_ann": ann(u1m), "universe_fwd_12m": u12m,
            "episode_months": [[f"{pd.Timestamp(dates[list(mn).index(m)]):%Y-%m}" for m in e] for e in ep],
        }
    # Stress (BEAR or CRISIS) against calm (BULL or CORRECTION) at 12 months, with leave-one-episode-out.
    stress = np.isin(reg, STRESS)
    calm = ~stress
    x12 = b12.reindex(dates).to_numpy()
    d, t, na, nb = diff_t(x12, stress, calm, 12)
    ep = episodes(mn[stress])
    eps_detail = []
    for e in ep:
        in_e = np.isin(mn, e) & stress
        mean_e = float(np.nanmean(x12[in_e])) if np.isfinite(x12[in_e]).any() else None
        d2, t2, na2, _ = diff_t(x12, stress & ~in_e, calm, 12)
        eps_detail.append({"first": f"{pd.Timestamp(dates[list(mn).index(e[0])]):%Y-%m}", "last": f"{pd.Timestamp(dates[list(mn).index(e[-1])]):%Y-%m}",
                           "months": len(e), "months_with_fwd_12m": int(np.isfinite(x12[in_e]).sum()), "mean_fwd_12m": mean_e,
                           "diff_without": d2, "t_without": t2})
    halves = {}
    for name, hm in (("first_half", first), ("second_half", ~first)):
        dh, th, nah, nbh = diff_t(x12, stress & hm, calm & hm, 12)
        halves[name] = {"diff": dh, "t": th, "n_stress": nah, "n_calm": nbh}
    contributing = [e for e in eps_detail if e["months_with_fwd_12m"] > 0]
    supported = bool(d == d and d > 0 and t == t and t >= 2 and all((halves[h]["diff"] or 0) > 0 for h in halves)
                     and len(contributing) >= MIN_EPISODES and all((e["diff_without"] or -1) > 0 for e in eps_detail))
    out["stress_vs_calm_12m"] = {"diff": d, "t": t, "n_stress": na, "n_calm": nb, "episodes": len(ep),
                                 "episodes_with_fwd_12m": len(contributing), "halves": halves, "leave_one_out": eps_detail,
                                 "supported": supported, "stress_mean": float(np.nanmean(x12[stress])), "calm_mean": float(np.nanmean(x12[calm])),
                                 "universe_stress_mean": float(np.nanmean(u12.reindex(dates).to_numpy()[stress])),
                                 "universe_calm_mean": float(np.nanmean(u12.reindex(dates).to_numpy()[calm]))}
    # The same difference for the whole universe: is it the market, not the owner picks?
    du, tu, _, _ = diff_t(u12.reindex(dates).to_numpy(), stress, calm, 12)
    out["stress_vs_calm_12m"]["universe_diff"], out["stress_vs_calm_12m"]["universe_t"] = du, tu
    # One-month returns after BUYs in stress, and how much each episode carries (the 48.9% figure rests on few months).
    x1 = b1.reindex(dates).to_numpy()
    u1x = u1.reindex(dates).to_numpy()
    for name, sel in (("stress_1m", stress), ("crisis_1m", reg == "CRISIS")):
        m, t, n = subset_mean_t(x1, sel, 3)
        eps = episodes(mn[sel])
        without = []
        for e in eps:
            in_e = np.isin(mn, e) & sel
            mw, tw, nw = subset_mean_t(x1, sel & ~in_e, 3)
            without.append({"first": f"{pd.Timestamp(dates[list(mn).index(e[0])]):%Y-%m}", "last": f"{pd.Timestamp(dates[list(mn).index(e[-1])]):%Y-%m}",
                            "months": len(e), "ann_without": ann(mw), "t_without": tw, "months_left": nw})
        ex = sel & (~np.isnan(x1)) & (~np.isnan(u1x))
        out[name] = {"ann": ann(m), "t": t, "months": n, "episodes": len(eps), "leave_one_episode_out": without,
                     "buy_minus_universe_monthly": float(np.mean(x1[ex] - u1x[ex])) if ex.any() else None,
                     "best_month": float(np.nanmax(x1[sel])) if np.isfinite(x1[sel]).any() else None,
                     "ann_without_best_month": ann(float(np.mean(np.delete(x1[sel][~np.isnan(x1[sel])], np.argmax(x1[sel][~np.isnan(x1[sel])]))))) if np.isfinite(x1[sel]).sum() > 1 else None}
    # Months per regime for the whole panel.
    out["months_per_regime"] = {r: int((reg == r).sum()) for r in REGIMES}
    out["episodes_per_regime"] = {r: len(episodes(mn[reg == r])) for r in REGIMES}
    return out


# ------------------------------------------------------------------ Q5: the double count


def label_change_decomposition(panel: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Share of timing-label changes (same ticker, consecutive months) caused by the macro state alone."""
    p = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    prev = p.groupby("ticker").shift(1)
    consecutive = (prev["date"] == p["date"] - pd.offsets.MonthEnd(1)).to_numpy()
    th = cfg["thresholds"]
    tick_cols = ["mom_score", "mom_conf", "vol_score", "vol_conf", "pb_score", "pb_conf"]
    macro_cols = ["macro_score", "macro_conf", "regime"]

    def label_of(tick_src: pd.DataFrame, macro_src: pd.DataFrame) -> pd.Series:
        f = pd.concat([tick_src[tick_cols].reset_index(drop=True), macro_src[macro_cols].reset_index(drop=True)], axis=1)
        return label_series(timing_composite(f, cfg), th)

    actual_prev = label_of(prev, prev)
    actual_now = label_of(p, p)
    macro_only_cf = label_of(prev, p)  # last month's ticker signals, this month's macro state
    ticker_only_cf = label_of(p, prev)  # this month's ticker signals, last month's macro state
    valid = consecutive & actual_prev.notna().to_numpy() & actual_now.notna().to_numpy()
    changed = valid & (actual_prev.to_numpy() != actual_now.to_numpy())
    a = (macro_only_cf.to_numpy() != actual_prev.to_numpy()) & changed
    b = (ticker_only_cf.to_numpy() != actual_prev.to_numpy()) & changed
    n = int(changed.sum())
    shares = {"macro_only": float((a & ~b).sum() / n), "ticker_only": float((b & ~a).sum() / n),
              "either_alone": float((a & b).sum() / n), "only_together": float((~a & ~b & changed).sum() / n)}
    # Same, as a function of the regime moving (profile switch) or not.
    regime_moved = (prev["regime"] != p["regime"]).to_numpy()
    out = {"ticker_months_compared": int(valid.sum()), "label_changes": n, "change_rate": float(n / max(int(valid.sum()), 1)),
           "shares": shares, "changes_in_months_regime_moved": float((changed & regime_moved).sum() / n),
           "share_of_all_ticker_months_with_regime_move": float((valid & regime_moved).sum() / max(int(valid.sum()), 1))}
    # How much of the month-to-month macro score movement is common to all tickers (it is, by construction).
    m = p.groupby("date")["macro_score"].agg(["mean", "std"]).dropna()
    out["macro_cross_section_std_max"] = float(m["std"].max()) if len(m) else None
    return out


# ------------------------------------------------------------------ report


def run_panel_questions(history: MacroHistory, rebuild: bool) -> dict[str, Any]:
    cfg = shipped_profiles()
    panel = load_panel(history, rebuild)
    print(f"Panel: {len(panel):,} company-months, {panel['date'].nunique()} months, {panel['ticker'].nunique()} companies.")
    fwd1, fwd3, fwd12 = load_forward_frames(panel)
    dates = [pd.Timestamp(d) for d in sorted(panel["date"].unique())]
    cut = dates[len(dates) // 2]
    regime_by_date = panel.drop_duplicates("date").set_index("date")["regime"].sort_index()
    is_buy = panel["owner_decision"].isin(BUYS)
    report: dict[str, Any] = {"months": len(dates), "first": str(dates[0].date()), "last": str(dates[-1].date()), "split": str(cut.date()),
                              "profiles": cfg["profiles"], "thresholds": cfg["thresholds"]}
    shipped = timing_composite(panel, cfg)
    diff = (shipped - panel["timing_score"]).abs()
    report["vectorised_matches_aggregate"] = {"max_abs_diff": float(diff.max()), "rows_differing_over_0.11": int((diff > 0.11).sum()),
                                              "rows": int(panel["timing_score"].notna().sum())}
    print(f"Vectorised composite vs aggregate: max abs diff {report['vectorised_matches_aggregate']['max_abs_diff']:.2f}")
    print("Q2a timing signals by regime...")
    report["q2_timing_by_regime"] = timing_by_regime(panel, fwd1, fwd12, regime_by_date, is_buy, dates, cut)
    print("Q2b shipped profiles against the flat profile (and Q5b no-macro variants)...")
    comparison, _ = profiles_vs_flat(panel, fwd1, cfg, is_buy, regime_by_date, cut)
    comparison.pop("shipped_composite")
    report["q2_profiles_vs_flat"] = {k: v for k, v in comparison.items() if k in ("shipped_vs_flat", "profiles_justified", "flat_profile")}
    report["q2_profiles_vs_flat"]["label_shares"] = comparison["label_shares"]
    report["q2_profiles_vs_flat"]["series_means"] = comparison["series_means"]
    report["q2_profiles_vs_flat"]["label_buckets_in_owner_buys"] = comparison["label_buckets_in_owner_buys"]
    report["q5_no_macro"] = {k: v for k, v in comparison.items() if k.startswith("shipped_vs_no_macro") or k == "macro_helps_somewhere"}
    print("Q3 owner score by regime...")
    report["q3_owner_by_regime"] = owner_by_regime(panel, fwd1, fwd12, regime_by_date, dates, cut)
    print("Q4 entries in stress...")
    report["q4_entries"] = entry_by_regime(panel, fwd1, fwd12, regime_by_date, dates, cut)
    print("Q5a label change decomposition...")
    report["q5_label_changes"] = label_change_decomposition(panel, cfg)
    return report


def regime_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """The machine-readable block other parts of the app read (see REPORT.md)."""
    q2, q3, q4 = report["q2_timing_by_regime"], report["q3_owner_by_regime"], report["q4_entries"]
    out: dict[str, Any] = {"entry_by_regime": {}, "timing_ic_by_regime": {}, "owner_ic_by_regime": {}}
    for r in REGIMES:
        e = q4["entry_by_regime"][r]
        out["entry_by_regime"][r] = {k: e[k] for k in ("fwd_12m", "fwd_1m_ann", "t", "n_months", "episodes", "first_half", "second_half")}
        out["timing_ic_by_regime"][r] = {k: {"ic": q2["ic_1m"][k][r]["mean"], "t": q2["ic_1m"][k][r]["t"],
                                             "first_half": q2["ic_1m"][k][r]["first_half"], "second_half": q2["ic_1m"][k][r]["second_half"]}
                                         for k in ("momentum", "volume", "pullback", "timing")}
        out["owner_ic_by_regime"][r] = {"ic_12m": q3["ic_12m"][r]["mean"], "t": q3["ic_12m"][r]["t"],
                                        "buy_minus_sell_12m": q3["buy_minus_sell_12m"][r]["mean"]}
    out["verdicts"] = report.get("verdicts", [])
    vf = report["q2_profiles_vs_flat"]["shipped_vs_flat"]
    out["profiles_vs_flat"] = {"justified": report["q2_profiles_vs_flat"]["profiles_justified"],
                               "metrics": {m: {k: vf[m][k] for k in ("mean", "t", "first_half", "second_half", "t_first", "t_second", "passes")} for m in vf}}
    return out


def verdicts(report: dict[str, Any]) -> list[str]:
    """One plain sentence per question, built from the acceptance results (the REPORT.md wording is longer)."""
    out: list[str] = []
    q1 = report["q1_classifier"]
    a = q1["acceptance_1"]
    r = a["1a_risk"]["detail"]["whole"]["adj_r2"]
    out.append(f"Q1a risk: the label orders forward volatility correctly in every sample (3-month volatility adjusted R-squared {r['vol_3m']['regime4']:.2f}) "
               + ("and beats" if a["1a_risk"]["passes"] else "but does not beat both baselines in the whole period and both halves: VIX buckets reach "
                  f"{r['vol_3m']['vix4']:.2f} and the 200-day mean {r['vol_3m']['ma200']:.2f}") + ".")
    b = a["1b_returns"]["stress_minus_bull_12m"]
    out.append("Q1b returns: " + ("high-fear regimes preceded better 12-month SPY returns (passes)" if a["1b_returns"]["contrarian_passes"]
                                 else "stress regimes preceded worse 12-month returns (passes)" if a["1b_returns"]["opposite_passes"]
                                 else f"BEAR+CRISIS minus BULL 12-month SPY return is {b['whole']['diff']:+.1%} (t {b['whole']['t']:.1f}), "
                                      f"{b['first_half']['diff']:+.1%} in the first half and {b['second_half']['diff']:+.1%} in the second: no stable relation") + ".")
    if "q2_profiles_vs_flat" in report:
        out.append("Q2: per-regime profiles " + ("beat" if report["q2_profiles_vs_flat"]["profiles_justified"] else "did not beat")
                   + " the flat profile on every pre-registered test; " + ("keep them." if report["q2_profiles_vs_flat"]["profiles_justified"] else "one profile is enough."))
        q3 = report["q3_owner_by_regime"]
        ics = ", ".join(f"{reg} {q3['ic_12m'][reg]['mean']:+.3f}" for reg in REGIMES)
        diff = q3["ic_12m"]["STRESS_minus_BULL"]
        positive = all((q3["ic_12m"][reg]["mean"] or 0) > 0 for reg in REGIMES)
        differs = diff["t"] is not None and abs(diff["t"]) >= 2
        out.append(f"Q3: the owner score's 12-month IC is {'positive in every regime' if positive else 'not positive in every regime'} ({ics}); "
                   f"stress minus BULL {diff['diff']:+.3f} (t {diff['t']:.1f}), {'a difference to look at' if differs else 'not significant'}"
                   f"{': regime independence stays.' if positive and not differs else '.'}")
        s = report["q4_entries"]["stress_vs_calm_12m"]
        out.append("Q4: buying owner BUYs in BEAR or CRISIS is " + ("supported" if s["supported"] else "not supported")
                   + f" by the pre-registered rule (12-month difference {s['diff']:+.1%}, t {s['t']:.1f}, {s['episodes']} independent episodes; "
                   f"owner BUYs beat the whole universe by {report['q4_entries']['stress_1m']['buy_minus_universe_monthly']:+.2%} a month in those months).")
        c = report["q5_label_changes"]["shares"]
        helps = report["q5_no_macro"]["macro_helps_somewhere"]
        out.append(f"Q5: {c['macro_only']:.0%} of timing-label changes come from the macro state alone, {c['ticker_only']:.0%} from the ticker alone; "
                   + ("dropping the macro agent loses nothing measurable." if not any(helps.values()) else f"the macro agent helps on: {helps}."))
    return out


# ------------------------------------------------------------------ D-40: out-of-sample check of the simplification


def c1_cutoffs_from_panel(panel: pd.DataFrame, cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    """(favorable cut-off, unfavorable cut-off, shipped FAVORABLE share, shipped UNFAVORABLE share) on the panel.

    One flat profile with macro 0, cut-offs such that the label shares equal the shipped label's, rounded to one decimal.
    """
    lab = label_series(timing_composite(panel, cfg), cfg["thresholds"])
    n = lab.notna().sum()
    fav, unf = float((lab == "FAVORABLE").sum() / n), float((lab == "UNFAVORABLE").sum() / n)
    c1 = timing_composite(panel, cfg, per_regime=False, include_macro=False).dropna()
    return round(float(c1.quantile(1 - fav)), 1), round(float(c1.quantile(unf)), 1), fav, unf


def _oos_score(task: tuple) -> pd.DataFrame:
    """Weekly Momentum, Volume and Pullback scores and confidences with forward returns for one ticker (worker process)."""
    from agents.momentum_agent import MomentumAgent
    from agents.pullback_agent import PullbackAgent
    from agents.volume_agent import VolumeAgent
    from core.timing_validation import STEP, WARMUP_BARS, WINDOW_BARS

    ticker, ohlcv, years = task
    agents = {"mom": MomentumAgent(), "vol": VolumeAgent(), "pb": PullbackAgent()}
    close = ohlcv["close"].to_numpy(dtype=float)
    start = max(WARMUP_BARS, int(ohlcv.index.searchsorted(ohlcv.index[-1] - pd.Timedelta(days=int(years * 365.25)))))
    rows = []
    for i in range(start, len(ohlcv) - 60, STEP):
        window = ohlcv.iloc[max(0, i - WINDOW_BARS + 1): i + 1]
        row: dict[str, Any] = {"ticker": ticker, "date": ohlcv.index[i]}
        for key, agent in agents.items():
            try:
                a = agent.evaluate(ticker, window)
            except Exception:  # noqa: BLE001
                a = agent.unavailable("agent raised")
            ok = a.data_quality != UNAVAILABLE
            row[f"{key}_score"] = a.score if ok else np.nan
            row[f"{key}_conf"] = a.confidence if ok else 0.0
        for h in (5, 20, 60):
            row[f"fwd_{h}"] = close[i + h] / close[i] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def oos_label_frames(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Composite and label per variant: S shipped, C1 flat without macro (frozen cut-offs), C2 flat with macro (shipped cut-offs)."""
    th = cfg["thresholds"]
    s = timing_composite(frame, cfg)
    c1 = timing_composite(frame, cfg, per_regime=False, include_macro=False)
    lab1 = pd.Series(np.where(c1 >= C1_FAVORABLE_CUTOFF, "FAVORABLE", np.where(c1 <= C1_UNFAVORABLE_CUTOFF, "UNFAVORABLE", "NEUTRAL")),
                     index=c1.index, dtype=object)
    lab1[c1.isna()] = None
    c2 = timing_composite(frame, cfg, per_regime=False, include_macro=True)
    return {"S": (s, label_series(s, th)), "C1": (c1, lab1), "C2": (c2, label_series(c2, th))}


def oos_series(frame: pd.DataFrame, comp: pd.Series, label: pd.Series) -> dict[str, pd.Series]:
    """Weekly series of one variant: cross-sectional IC and label spread at 20 and 60 sessions, two holding rules."""
    f = frame[["ticker", "date", "fwd_5", "fwd_20", "fwd_60"]].copy()
    f["comp"], f["label"] = comp.to_numpy(), label.to_numpy()
    for h in (20, 60):
        f[f"ex_{h}"] = f[f"fwd_{h}"] - f.groupby("ticker")[f"fwd_{h}"].transform("mean")
    out: dict[str, dict[Any, float]] = {k: {} for k in ("ic_20", "ic_60", "spread_20", "spread_60", "hold_not_unfavorable", "hold_favorable")}
    for date, g in f.groupby("date"):
        g = g.dropna(subset=["comp"])
        if len(g) < 10:
            continue
        for h in (20, 60):
            ok = g[f"fwd_{h}"].notna()
            if ok.sum() >= 10 and g.loc[ok, "comp"].nunique() > 1:
                out[f"ic_{h}"][date] = float(g.loc[ok, "comp"].rank().corr(g.loc[ok, f"fwd_{h}"].rank()))
            fav, unf = g[g["label"] == "FAVORABLE"], g[g["label"] == "UNFAVORABLE"]
            if len(fav) and len(unf):
                out[f"spread_{h}"][date] = float(fav[f"ex_{h}"].mean() - unf[f"ex_{h}"].mean())
        keep = g[g["label"] != "UNFAVORABLE"]
        out["hold_not_unfavorable"][date] = float(keep["fwd_5"].mean()) if len(keep) else 0.0  # cash when nothing qualifies
        keep = g[g["label"] == "FAVORABLE"]
        out["hold_favorable"][date] = float(keep["fwd_5"].mean()) if len(keep) else 0.0
    return {k: pd.Series(v, dtype=float).sort_index() for k, v in out.items()}


OOS_LAGS = {"ic_20": 4, "ic_60": 12, "spread_20": 4, "spread_60": 12, "hold_not_unfavorable": 4, "hold_favorable": 4}


def label_changes_per_ticker_year(frame: pd.DataFrame, label: pd.Series) -> float:
    f = pd.DataFrame({"ticker": frame["ticker"].to_numpy(), "date": frame["date"].to_numpy(), "label": label.to_numpy()}).sort_values(["ticker", "date"])
    prev = f.groupby("ticker")["label"].shift(1)
    ok = prev.notna() & f["label"].notna()
    changes = int((ok & (prev != f["label"])).sum())
    years = f.groupby("ticker")["date"].agg(lambda d: (d.max() - d.min()).days / 365.25).sum()
    return float(changes / years) if years > 0 else float("nan")


def timeseries_ic_diff(frame: pd.DataFrame, a: pd.Series, b: pd.Series, h: int, cut: pd.Timestamp) -> dict[str, Any]:
    """Per-ticker time-series IC of variant a minus variant b, mean and across-ticker t, whole and halves."""
    f = frame[["ticker", "date", f"fwd_{h}"]].copy()
    f["a"], f["b"] = a.to_numpy(), b.to_numpy()
    res: dict[str, Any] = {}
    for name, sel in (("whole", f["date"].notna()), ("first_half", f["date"] < cut), ("second_half", f["date"] >= cut)):
        diffs = []
        for _, g in f[sel].dropna().groupby("ticker"):
            if len(g) >= 40 and g["a"].nunique() > 1 and g["b"].nunique() > 1:
                diffs.append(g["a"].rank().corr(g[f"fwd_{h}"].rank()) - g["b"].rank().corr(g[f"fwd_{h}"].rank()))
        d = np.array(diffs)
        res[name] = {"mean": float(d.mean()) if len(d) else None, "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std(ddof=1) > 0 else None,
                     "tickers": int(len(d))}
    ok = all(res[k]["mean"] is not None and res[k]["mean"] > 0 and (res[k]["t"] or 0) >= 2 for k in res)
    res["favours_S"] = bool(ok)
    return res


def oos_compare(frame: pd.DataFrame, variants: dict[str, tuple[pd.Series, pd.Series]], cand: str, cut: pd.Timestamp) -> dict[str, Any]:
    """S minus a candidate on every pre-registered series, the adoption rule's inputs."""
    s_series = oos_series(frame, *variants["S"])
    c_series = oos_series(frame, *variants[cand])
    out: dict[str, Any] = {"series": {}}
    for k, lag in OOS_LAGS.items():
        r = paired_diff(s_series[k], c_series[k], lag, cut)
        r["favours_S"] = passes_both_halves(r)
        out["series"][k] = r
    out["timeseries_ic_20"] = timeseries_ic_diff(frame, variants["S"][0], variants[cand][0], 20, cut)
    out["timeseries_ic_60"] = timeseries_ic_diff(frame, variants["S"][0], variants[cand][0], 60, cut)
    out["label_changes_per_ticker_year"] = {"S": label_changes_per_ticker_year(frame, variants["S"][1]),
                                            cand: label_changes_per_ticker_year(frame, variants[cand][1])}
    out["label_shares"] = {v: {k: float((variants[v][1] == k).sum() / variants[v][1].notna().sum()) for k in ("FAVORABLE", "NEUTRAL", "UNFAVORABLE")}
                           for v in ("S", cand)}
    out["means"] = {"S": {k: float(v.mean()) for k, v in s_series.items()}, cand: {k: float(v.mean()) for k, v in c_series.items()}}
    favoured = [k for k, v in out["series"].items() if v["favours_S"]] + [k for k in ("timeseries_ic_20", "timeseries_ic_60") if out[k]["favours_S"]]
    out["series_favouring_S"] = favoured
    out["changes_less_often"] = bool(out["label_changes_per_ticker_year"][cand] < out["label_changes_per_ticker_year"]["S"])
    out["passes"] = bool(not favoured and out["changes_less_often"])
    return out


def run_oos(history: MacroHistory, years: float = 10.0, workers: int = 6) -> dict[str, Any]:
    """The D-40 protocol, as written: S against C1; C2 only if C1 fails."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from data.fetcher import fetch_ohlcv

    cfg = shipped_profiles()
    panel = load_panel(history)
    hi, lo, fav, unf = c1_cutoffs_from_panel(panel, cfg)
    print(f"C1 cut-offs recomputed on the panel: FAVORABLE >= {hi}, UNFAVORABLE <= {lo} (frozen constants {C1_FAVORABLE_CUTOFF}, {C1_UNFAVORABLE_CUTOFF})")
    assert (hi, lo) == (C1_FAVORABLE_CUTOFF, C1_UNFAVORABLE_CUTOFF), "frozen cut-offs no longer match the panel"
    data = fetch_ohlcv(list(OOS_TICKERS))
    from core.timing_validation import WARMUP_BARS
    tasks = [(t, data[t], years) for t in OOS_TICKERS if t in data and len(data[t]) > WARMUP_BARS + 110]
    print(f"Replaying {len(tasks)} tickers weekly over {years:g} years")
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(_oos_score, t) for t in tasks]):
            frames.append(fut.result())
    frame = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    agent = MacroRegimeAgent()
    macro = {}
    for d in sorted(frame["date"].unique()):
        snap = compute_macro_snapshot(history, d)
        a = agent.evaluate_with_data(snap)
        ok = a.data_quality != UNAVAILABLE and snap.data_quality != "ESTIMATED"
        macro[d] = (a.score if ok else np.nan, a.confidence if ok else 0.0, snap.regime if ok else None)
    frame["macro_score"] = frame["date"].map(lambda d: macro[d][0])
    frame["macro_conf"] = frame["date"].map(lambda d: macro[d][1])
    frame["regime"] = frame["date"].map(lambda d: macro[d][2])
    spread = frame.groupby("date")["macro_score"].std().max()
    dates = sorted(frame["date"].unique())
    cut = dates[len(dates) // 2]
    variants = oos_label_frames(frame, cfg)
    report: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds"), "tickers": sorted(frame["ticker"].unique()),
                              "evaluations": int(len(frame)), "weeks": len(dates), "first": str(dates[0].date()), "last": str(dates[-1].date()),
                              "split": str(cut.date()), "frozen_cutoffs": {"favorable": C1_FAVORABLE_CUTOFF, "unfavorable": C1_UNFAVORABLE_CUTOFF,
                                                                           "panel_shipped_shares": {"FAVORABLE": fav, "UNFAVORABLE": unf}},
                              "macro_score_max_cross_ticker_std": float(spread),
                              "regime_share": {r: float((frame.drop_duplicates("date")["regime"] == r).mean()) for r in REGIMES},
                              "s_vs_c1": oos_compare(frame, variants, "C1", cut)}
    if report["s_vs_c1"]["passes"]:
        report["decision"] = "ADOPT C1"
    else:
        report["s_vs_c2"] = oos_compare(frame, variants, "C2", cut)
        report["decision"] = "ADOPT C2" if report["s_vs_c2"]["passes"] else "KEEP the shipped profiles and cut-offs"
    print("Decision:", report["decision"])
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "regime_oos.json").write_text(json.dumps(_clean(report), indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Market regime study: classifier, timing weights by regime, owner score by regime")
    parser.add_argument("--refresh", action="store_true", help="Redownload the macro history")
    parser.add_argument("--rebuild-panel", action="store_true", help="Rebuild the scored panel from the cached panels")
    parser.add_argument("--classifier-only", action="store_true", help="Run question 1 only")
    parser.add_argument("--oos", action="store_true", help="D-40 out-of-sample check on the 32 untuned tickers (writes regime_oos.json)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    history = load_long_history(args.refresh)
    if args.oos:
        run_oos(history)
        return 0
    print("Q1 classifier on SPY history...")
    q1 = analyze_classifier(history)
    report: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                              "thresholds": {"regime_cutoffs": [list(t) for t in REGIME_THRESHOLDS], "classify_regime(0.9)": classify_regime(0.9)}}
    report["q1_classifier"] = {k: v for k, v in q1.items()}
    for part in ("monthly", "weekly"):
        for k in ("_table", "_fwd", "_labels"):
            report["q1_classifier"][part].pop(k, None)
    if not args.classifier_only:
        report.update(run_panel_questions(history, args.rebuild_panel))
    report["verdicts"] = verdicts(report)
    if not args.classifier_only:
        report["regime_evidence"] = regime_evidence(report)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "regime_study.json").write_text(json.dumps(_clean(report), indent=2))
    for v in report["verdicts"]:
        print(" -", v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
