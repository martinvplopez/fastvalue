"""Out-of-sample check for the Backtest tab: does a setup that won on your holdings also win on the S&P 500?

The setup search ranks setups on the holdings you own today, which were picked with hindsight. This replays a
setup and your saved one on the value study's point-in-time S&P 500 panel (2011 onward, about 600 companies,
none of them chosen for this test) with the same trading rule as the lab and the exit-rule study: a book of
every name the setup rates BUY (optional entry gate), held until it turns SELL or leaves the index, equal
weight, rebalanced monthly, 10 bps per trade.

Acceptance rule, fixed before the first run (decision_log.md D-27), the one the exit-rule study used (T-14):
the setup generalizes only if it beats your saved setup over the whole period and in both halves, with a
Newey-West t of at least 2 on the monthly difference. Beating it in both halves with a lower t is reported as
consistent but not significant; anything else says to keep your setup.

Needs the panels the studies cache (``python -m core.value_study`` and ``python -m core.all_agents_study`` once);
prices come from the study's price cache. Nothing about your portfolio is used.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

import config.settings as settings
from agents.scoring import NOT_RATED
from config.weights import ScoringConfig, load_scoring_config
from core.lab import (BUYS, MACRO_COLUMNS, SELLS, Rules, _fred, _halves, _perf, _read_part, _write_part, apply_overrides,
                      describe_setup, evaluate_setup)

logger = logging.getLogger(__name__)

Progress = Callable[[int, int, str], None]
MIN_NAMES = 10
MIN_T = 2.0
COST_BPS = 10.0


def _macro(dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Macro agent at each month-end, from the lab's cache (scored once, reused by every check)."""
    from core.lab import _macro_rows

    macro, done, _ = _read_part("macro")
    todo = sorted(set(dates) - done)
    if todo:
        fresh = _macro_rows(todo)
        good = fresh[fresh["macro_score"].notna()] if not fresh.empty else fresh
        if not good.empty:
            macro = pd.concat([macro, good], ignore_index=True)
            done |= {pd.Timestamp(d) for d in good["date"]}
            _write_part("macro", macro, done)
    if macro.empty:
        return pd.DataFrame(columns=["date", *MACRO_COLUMNS])
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    return macro.drop_duplicates("date", keep="last")[["date", *MACRO_COLUMNS]]


def load_panel() -> pd.DataFrame:
    """The value study panel with timing and macro scores. Raises LookupError when the studies have not run."""
    value_path = settings.DATA_DIR / "value_panel.parquet"
    timing_path = settings.DATA_DIR / "timing_panel.parquet"
    if not value_path.exists() or not timing_path.exists():
        raise LookupError("The S&P 500 panel is not built yet: run `python -m core.value_study` and "
                          "`python -m core.all_agents_study` once (they need SEC_USER_AGENT).")
    panel = pd.read_parquet(value_path).merge(pd.read_parquet(timing_path), on=["date", "ticker"], how="left")
    panel["date"] = pd.to_datetime(panel["date"])
    dates = sorted(panel["date"].unique())
    return panel.merge(_macro([pd.Timestamp(d) for d in dates]), on="date", how="left")


def load_forward(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return over the month after each panel date (dividends included) and the monthly T-bill rate."""
    from core.value_study import PRICE_START, forward_returns, month_end_panel
    from data.price_history import fetch_price_history

    dates = sorted(pd.Timestamp(d) for d in panel["date"].unique())
    tickers = sorted(panel["ticker"].unique())
    prices = fetch_price_history(tickers, PRICE_START, max_age_days=math.inf)  # the study's cache; history is fixed
    level = month_end_panel(prices, dates + [dates[-1] + pd.offsets.MonthEnd(1)])
    last = max((df.index[-1] for df in prices.values() if len(df)), default=dates[-1])
    level.loc[level.index > last + pd.Timedelta(days=3)] = np.nan
    fwd1 = forward_returns(level, 1).reindex(dates)
    rate = _fred("DTB3")
    idx = pd.DatetimeIndex([d + pd.offsets.MonthEnd(1) for d in dates])
    rf = (rate.reindex(rate.index.union(idx)).ffill().reindex(idx).fillna(0.0) / 100.0 / 12.0) if len(rate) else pd.Series(0.0, index=idx)
    return fwd1, rf


def book(decided: pd.DataFrame, fwd1: pd.DataFrame, entry_gate: str = "none", cost_bps: float = COST_BPS) -> dict[str, Any]:
    """Equal-weight book: enter on BUY (through the entry gate), exit on SELL or when the name leaves the panel."""
    holdings: set[str] = set()
    returns: dict[pd.Timestamp, float] = {}
    names: list[int] = []
    turnovers: list[float] = []
    for date, g in decided.groupby("date"):
        g = g.drop_duplicates("ticker").set_index("ticker")
        dec, label, setup = g["decision"], g["timing_label"], g["setup"]
        kept = {t for t in holdings if t in g.index and dec[t] not in SELLS}
        for t in g.index[dec.isin(BUYS)]:
            if t in kept:
                continue
            if (entry_gate == "none" or (entry_gate == "not_unfavorable" and label[t] != "UNFAVORABLE")
                    or (entry_gate == "favorable" and label[t] == "FAVORABLE")
                    or (entry_gate == "not_extended" and setup[t] != "extended")):
                kept.add(t)
        turnover = len(kept ^ holdings) / (2 * max(len(kept), 1)) if holdings else 1.0
        holdings = kept
        if date not in fwd1.index:
            continue
        r = fwd1.loc[date].reindex(sorted(holdings)).dropna()
        if len(r) < MIN_NAMES:
            continue
        returns[date] = float(r.mean()) - turnover * 2 * cost_bps / 10_000
        names.append(len(r))
        turnovers.append(turnover)
    s = pd.Series(returns, dtype=float).sort_index()
    s.index = s.index + pd.offsets.MonthEnd(1)  # earned over the following month
    return {"returns": s, "avg_names": float(np.mean(names)) if names else float("nan"),
            "turnover": float(np.mean(turnovers)) * 12 if turnovers else float("nan")}


def _universe(panel: pd.DataFrame, fwd1: pd.DataFrame) -> pd.Series:
    out = {}
    for date, g in panel.groupby("date"):
        if date in fwd1.index:
            r = fwd1.loc[date].reindex(g["ticker"].unique()).dropna()
            if len(r) >= MIN_NAMES:
                out[date] = float(r.mean())
    s = pd.Series(out, dtype=float).sort_index()
    s.index = s.index + pd.offsets.MonthEnd(1)
    return s


def _row(label: str, sim: dict[str, Any] | None, monthly: pd.Series, rf: pd.Series) -> dict[str, Any]:
    first, second = _halves(monthly, rf)
    out = {"label": label, "stats": _perf(monthly, rf), "first_half": first, "second_half": second}
    if sim is not None:
        out.update({"avg_names": sim["avg_names"], "turnover": sim["turnover"]})
    return out


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def compare(panel: pd.DataFrame, fwd1: pd.DataFrame, rf: pd.Series, candidate: ScoringConfig, saved: ScoringConfig,
            candidate_gate: str = "none", saved_gate: str = "none") -> dict[str, Any]:
    from core.value_study import newey_west_t

    a = book(evaluate_setup(panel, saved), fwd1, saved_gate)
    b = book(evaluate_setup(panel, candidate), fwd1, candidate_gate)
    common = a["returns"].index.intersection(b["returns"].index)
    ra, rb = a["returns"].reindex(common), b["returns"].reindex(common)
    universe = _universe(panel, fwd1).reindex(common)
    diff = (rb - ra).to_numpy(float)
    mid = len(common) // 2
    t = newey_west_t(diff, 3) if len(diff) else float("nan")
    full = _perf(rb, rf)["cagr"] - _perf(ra, rf)["cagr"]
    halves = [_perf(rb.iloc[:mid], rf)["cagr"] - _perf(ra.iloc[:mid], rf)["cagr"],
              _perf(rb.iloc[mid:], rf)["cagr"] - _perf(ra.iloc[mid:], rf)["cagr"]]
    same = bool(len(diff)) and float(np.nanmax(np.abs(diff))) < 1e-12
    wins = full > 0 and all(h > 0 for h in halves)
    if same:
        verdict, level = "On the S&P 500 this setup trades exactly like your saved one: nothing to choose.", "same"
    elif wins and t == t and t >= MIN_T:
        verdict, level = ("It beat your saved setup on 500 companies it was not tuned on, in both halves, with t "
                          f"{t:.1f}. Worth adopting: note these numbers before you apply it."), "generalizes"
    elif wins:
        verdict, level = (f"It beat your saved setup in both halves, but the difference is not significant (t {t:.1f} "
                          f"below {MIN_T:.0f}). Re-test when more history exists."), "consistent"
    else:
        verdict, level = ("It did not beat your saved setup on the S&P 500 in both halves: what won on your holdings "
                          "does not carry over."), "no"
    return _clean({
        "period": {"start": common[0].strftime("%Y-%m") if len(common) else None,
                   "end": common[-1].strftime("%Y-%m") if len(common) else None,
                   "split": common[mid].strftime("%Y-%m") if len(common) else None, "months": int(len(common))},
        "companies": int(panel["ticker"].nunique()),
        "rows": [_row("This setup", b, rb, rf), _row("Your saved setup", a, ra, rf),
                 _row("Equal-weight S&P 500 (covered companies)", None, universe, rf)],
        "difference": {"cagr": full, "first_half": halves[0], "second_half": halves[1], "t": None if t != t else round(t, 2)},
        "verdict": verdict, "level": level,
    })


def check_setup(combo: dict[str, Any], rules: Rules | None = None, progress: Progress | None = None) -> dict[str, Any]:
    """Compare the search row ``combo`` (applied to the saved setup) with the saved setup on the S&P 500 panel."""
    progress = progress or (lambda *_: None)
    saved = load_scoring_config()
    candidate = apply_overrides(saved, combo or {})
    if candidate is None:
        raise ValueError("This combination of settings is not valid")
    base_gate = (rules or Rules()).entry_gate
    progress(0, 3, "Loading the S&P 500 panel")
    panel = load_panel()
    progress(1, 3, "Loading prices")
    fwd1, rf = load_forward(panel)
    progress(2, 3, "Replaying both setups")
    out = compare(panel, fwd1, rf, candidate, saved, combo.get("entry_gate", base_gate), base_gate)
    progress(3, 3, "Done")
    return {**out, "setup": describe_setup(candidate), "saved_setup": describe_setup(saved), "combo": combo,
            "notes": ["Survivorship: companies later acquired or bankrupt are mostly missing from the panel (about half "
                      "the index in 2011), so compare the two setups with each other more than with the market.",
                      "Only the agents with point-in-time history are replayed (Fundamentals, Shareholders, Momentum, "
                      "Volume, Pullback, Macro), as in the lab."]}
