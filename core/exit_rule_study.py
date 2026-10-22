"""Exit-rule study: what an owner SELL should do to a held position.

The owner decision ranks forward returns (``core/all_agents_study.py``) and the
portfolio review exits a position when the decision turns SELL, that is when the
owner score is at or below -15. SELL names still made about 11% a year in the
panel, so the level is a choice worth testing rather than assuming. This module
replays a held book on the same point-in-time panel under alternative rules:

1. Book simulation. Enter every owner BUY (capped at 35x) at the month-end it
   appears, equal weight, and hold until the exit rule fires or the name leaves
   the index. A month without a rating (statements missing) keeps the position.
   Rules: exit at or below a score level (-15 is SELL, -45 is STRONG SELL), the
   same level confirmed over consecutive month-ends, and never (index exits
   only). Reported: CAGR, Sharpe, drawdown, turnover, names held, both halves of
   the sample, and the monthly return difference against the current rule with
   a Newey-West t.
2. Crossing view. For held names, the 12-month return after the first month-end
   at or below -15, by how deep the score went, against the equal-weight
   universe over the same months. If shallow SELLs do as well as the universe,
   exiting them gives up nothing; if they lag, the exit is right.
3. Red flags (reverse splits) are not in the panel and are not replayed; the
   red-flag EXIT stays whatever the level.

Acceptance rule, fixed before the run (decision_log.md T-14): the exit level
changes only if a rule beats the current one over the whole period and in both
halves, with a Newey-West |t| of at least 2 on the monthly difference. Otherwise
the current rule stays. One small grid, one pre-stated criterion, no tuning.

Run: ``SEC_USER_AGENT="app you@example.com" python -m core.exit_rule_study``
(needs the value panel from ``core.value_study``). Report:
analysis/output/exit_rule_study.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from core.all_agents_study import BUYS, forward_frames, load_scored_panel
from core.value_study import COST_BPS, newey_west_t, performance, prepare

logger = logging.getLogger(__name__)

CURRENT_RULE = "exit on SELL (score <= -15)"
# (label, exit level on the owner score or None for never, consecutive month-ends at or below the level)
RULES: tuple[tuple[str, float | None, int], ...] = (
    (CURRENT_RULE, -15.0, 1),
    ("exit at or below -20", -20.0, 1),
    ("exit at or below -25", -25.0, 1),
    ("exit at or below -30", -30.0, 1),
    ("exit at or below -35", -35.0, 1),
    ("exit on STRONG SELL (<= -45)", -45.0, 1),
    ("exit on SELL 2 month-ends running", -15.0, 2),
    ("exit on SELL 3 month-ends running", -15.0, 3),
    ("exit at or below -30, 2 month-ends running", -30.0, 2),
    ("never (index exits only)", None, 1),
)
CROSSING_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("-15 to -20", -20.0, -15.0),
    ("-20 to -30", -30.0, -20.0),
    ("-30 to -45", -45.0, -30.0),
    ("below -45", -np.inf, -45.0),
)
MIN_NAMES = 10
T_REQUIRED = 2.0


def held_book(panel: pd.DataFrame, fwd1: pd.DataFrame, level: float | None, confirm: int = 1,
              cost_bps: float = COST_BPS) -> dict[str, Any]:
    """Equal-weight book of owner BUY entries held until the rule fires or the name leaves the index."""
    holdings: dict[str, int] = {}  # ticker -> consecutive month-ends at or below the level
    returns: dict[pd.Timestamp, float] = {}
    turnovers: list[float] = []
    counts: list[int] = []
    exits: list[dict[str, Any]] = []
    for date, group in panel.groupby("date"):
        g = group.drop_duplicates("ticker").set_index("ticker")
        kept: dict[str, int] = {}
        for ticker, streak in holdings.items():
            if ticker not in g.index:
                exits.append({"date": date, "ticker": ticker, "score": np.nan, "why": "left"})
                continue
            score = g.at[ticker, "composite"]
            if pd.isna(score):
                kept[ticker] = streak  # no rating this month: hold
                continue
            score = float(score)
            streak = streak + 1 if (level is not None and score <= level) else 0
            if level is not None and streak >= confirm:
                exits.append({"date": date, "ticker": ticker, "score": score, "why": "rule"})
                continue
            kept[ticker] = streak
        for ticker in g.index[g["owner_decision_capped"].isin(BUYS)]:
            kept.setdefault(ticker, 0)
        new, old = set(kept), set(holdings)
        turnover = len(new ^ old) / (2 * max(len(new), 1)) if old else 1.0
        holdings = kept
        if not holdings or date not in fwd1.index:
            continue
        r = fwd1.loc[date].reindex(sorted(holdings)).dropna()
        if len(r) < MIN_NAMES:
            continue
        returns[date] = float(r.mean()) - turnover * 2 * cost_bps / 10_000
        turnovers.append(turnover)
        counts.append(len(r))
    s = pd.Series(returns).sort_index()
    s.index = s.index + pd.offsets.MonthEnd(1)  # earned over the following month
    return {
        "returns": s,
        "turnover": float(np.mean(turnovers)) if turnovers else float("nan"),
        "avg_names": float(np.mean(counts)) if counts else float("nan"),
        "exits": pd.DataFrame(exits, columns=["date", "ticker", "score", "why"]),
    }


def _universe_means(panel: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    out = {}
    for date, group in panel.groupby("date"):
        if date in fwd.index:
            r = fwd.loc[date].reindex(group["ticker"].unique()).dropna()
            if len(r) >= MIN_NAMES:
                out[date] = float(r.mean())
    return pd.Series(out).sort_index()


def post_exit_returns(exits: pd.DataFrame, fwd12: pd.DataFrame, universe12: pd.Series) -> dict[str, Any] | None:
    """12-month return of names sold by the rule, against the universe over the same months."""
    rule_exits = exits[exits["why"] == "rule"]
    if rule_exits.empty:
        return None
    rows = []
    for e in rule_exits.itertuples(index=False):
        if e.date not in fwd12.index or e.ticker not in fwd12.columns or e.date not in universe12.index:
            continue
        r = fwd12.at[e.date, e.ticker]
        if pd.isna(r):
            continue
        rows.append({"date": e.date, "ticker": e.ticker, "score": e.score, "r12": float(r),
                     "excess12": float(r) - float(universe12.loc[e.date])})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    monthly = df.groupby("date")["excess12"].mean()
    return {
        "exits": int(len(df)),
        "mean_12m": float(df["r12"].mean()),
        "mean_excess_12m": float(df["excess12"].mean()),
        "share_excess_positive": float((df["excess12"] > 0).mean()),
        "t_excess_monthly": newey_west_t(monthly.to_numpy(), 11),
    }


def crossing_view(exits: pd.DataFrame, fwd12: pd.DataFrame, universe12: pd.Series,
                  halves: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]) -> dict[str, Any]:
    """After the first month-end at or below -15 while held: 12-month return by depth of the score."""
    rule_exits = exits[exits["why"] == "rule"].copy()
    rows = []
    for e in rule_exits.itertuples(index=False):
        if e.date not in fwd12.index or e.ticker not in fwd12.columns or e.date not in universe12.index:
            continue
        r = fwd12.at[e.date, e.ticker]
        if pd.isna(r):
            continue
        rows.append({"date": e.date, "score": e.score, "r12": float(r), "excess12": float(r) - float(universe12.loc[e.date])})
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    for label, low, high in CROSSING_BUCKETS:
        sub = df[(df["score"] > low) & (df["score"] <= high)] if not df.empty else df
        if len(sub) < 20:
            continue
        monthly = sub.groupby("date")["excess12"].mean()
        entry = {
            "events": int(len(sub)),
            "mean_12m": float(sub["r12"].mean()),
            "mean_excess_12m": float(sub["excess12"].mean()),
            "share_excess_positive": float((sub["excess12"] > 0).mean()),
            "t_excess_monthly": newey_west_t(monthly.to_numpy(), 11),
        }
        for name, (a, b) in zip(("first_half", "second_half"), halves):
            part = sub[(sub["date"] >= a) & (sub["date"] <= b)]
            entry[f"mean_excess_12m_{name}"] = float(part["excess12"].mean()) if len(part) >= 10 else None
        out[label] = entry
    return out


def analyze(panel: pd.DataFrame, prep: dict[str, Any]) -> dict[str, Any]:
    fwd1, fwd12, spy_ret, rf_m = forward_frames(prep)
    universe12 = _universe_means(panel, fwd12)
    books = {label: held_book(panel, fwd1, level, confirm) for label, level, confirm in RULES}

    common = None
    for b in books.values():
        if b["returns"].empty:
            continue
        common = b["returns"].index if common is None else common.intersection(b["returns"].index)
    common = common[common <= spy_ret.dropna().index[-1]]
    halves = ((common[0], common[len(common) // 2 - 1]), (common[len(common) // 2], common[-1]))
    base = books[CURRENT_RULE]["returns"].reindex(common)

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fcf_definition": settings.FCF_DEFINITION,
        "period": {"start": str(common[0].date()), "end": str(common[-1].date())},
        "halves": [[str(a.date()), str(b.date())] for a, b in halves],
        "acceptance_rule": (f"change the level only if a rule beats '{CURRENT_RULE}' over the whole period and in both "
                            f"halves with Newey-West |t| >= {T_REQUIRED:g} on the monthly difference"),
        "rules": {},
    }
    print(f"\nEXIT RULES  held book of owner BUY entries, {common[0]:%Y-%m} to {common[-1]:%Y-%m}, "
          f"equal weight, {COST_BPS:g} bps per trade, FCF definition '{settings.FCF_DEFINITION}'\n")
    print(f"{'':<44}{'CAGR':>7}{'Sharpe':>8}{'MaxDD':>8}{'Turn':>6}{'Names':>6}{'1st':>7}{'2nd':>7}"
          f"{'vs current':>11}{'t':>6}{'1st':>8}{'2nd':>8}{'exit 12m excess':>17}")
    for label, level, confirm in RULES:
        b = books[label]
        s = b["returns"].reindex(common)
        if s.isna().all():
            continue
        perf = performance(s, rf_m.reindex(common))
        diff = (s - base).dropna()
        first = [performance(s.loc[a:b_], rf_m.reindex(common).loc[a:b_])["cagr"] for a, b_ in halves]
        diff_halves = [float(diff.loc[a:b_].mean() * 12) for a, b_ in halves]
        post = post_exit_returns(b["exits"], fwd12, universe12)
        row = {
            **perf, "turnover": b["turnover"], "avg_names": b["avg_names"], "level": level, "confirm_months": confirm,
            "cagr_first_half": first[0], "cagr_second_half": first[1],
            "diff_vs_current_annual": float(diff.mean() * 12), "diff_t": newey_west_t(diff.to_numpy(), 3),
            "diff_first_half_annual": diff_halves[0], "diff_second_half_annual": diff_halves[1],
            "rule_exits": int((b["exits"]["why"] == "rule").sum()), "index_exits": int((b["exits"]["why"] == "left").sum()),
            "post_exit": post,
        }
        row["qualifies"] = bool(label != CURRENT_RULE and row["diff_vs_current_annual"] > 0 and diff_halves[0] > 0
                                and diff_halves[1] > 0 and row["diff_t"] >= T_REQUIRED)
        report["rules"][label] = row
        post_txt = f"{post['mean_excess_12m']:>+9.1%} ({post['exits']})" if post else "n/a"
        print(f"{label:<44}{perf['cagr']:>7.1%}{perf['sharpe']:>8.2f}{perf['max_drawdown']:>8.1%}{b['turnover']:>6.1%}"
              f"{b['avg_names']:>6.0f}{first[0]:>7.1%}{first[1]:>7.1%}{row['diff_vs_current_annual']:>+11.1%}"
              f"{row['diff_t']:>6.1f}{diff_halves[0]:>+8.1%}{diff_halves[1]:>+8.1%}{post_txt:>17}"
              + ("   <- qualifies" if row["qualifies"] else ""))
    print(f"  halves: {halves[0][0]:%Y-%m} to {halves[0][1]:%Y-%m} and {halves[1][0]:%Y-%m} to {halves[1][1]:%Y-%m}."
          " 'exit 12m excess': 12-month return of the names the rule sold, minus the universe (count).")

    report["crossing"] = crossing_view(books[CURRENT_RULE]["exits"], fwd12, universe12, halves)
    print("\nAFTER THE FIRST SELL WHILE HELD  12-month return of the name by how deep the owner score went, vs the universe\n")
    print(f"{'score at the SELL':<20}{'events':>8}{'12m':>8}{'excess':>9}{'excess>0':>10}{'t':>6}{'1st half':>10}{'2nd half':>10}")
    for label, v in report["crossing"].items():
        f = lambda x: "n/a" if x is None else f"{x:+.1%}"  # noqa: E731
        print(f"{label:<20}{v['events']:>8}{v['mean_12m']:>8.1%}{v['mean_excess_12m']:>+9.1%}{v['share_excess_positive']:>10.0%}"
              f"{v['t_excess_monthly']:>6.1f}{f(v['mean_excess_12m_first_half']):>10}{f(v['mean_excess_12m_second_half']):>10}")

    qualifying = [k for k, v in report["rules"].items() if v["qualifies"]]
    if qualifying:
        best = max(qualifying, key=lambda k: report["rules"][k]["diff_t"])
        report["decision"] = {"change": True, "rule": best, "level": report["rules"][best]["level"],
                              "confirm_months": report["rules"][best]["confirm_months"], "qualifying": qualifying}
        print(f"\nDECISION: '{best}' meets the acceptance rule (qualifying: {', '.join(qualifying)}).")
    else:
        report["decision"] = {"change": False, "rule": CURRENT_RULE, "level": -15.0, "confirm_months": 1, "qualifying": []}
        print(f"\nDECISION: no rule meets the acceptance rule; keep '{CURRENT_RULE}'.")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exit-rule study for held positions on the S&P 500 panel")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    end = pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)
    prep = prepare(end)
    panel = load_scored_panel(prep, args.workers)
    if panel is None:
        return 1
    report = analyze(panel, prep)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "exit_rule_study.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
