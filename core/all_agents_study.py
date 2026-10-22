"""All-agents point-in-time backtest on the S&P 500, 2011-2026.

The value study (``core/value_study.py``) scores the owner agents on each
company-month from SEC filings. This module puts the timing panel next to it on
the same company-months and tests the two-layer decision as a whole:

1. Owner decision buckets (STRONG_BUY .. STRONG_SELL), with and without the
   margin-of-safety cap: forward returns 1 and 12 months ahead.
2. The timing panel (Momentum, Volume, Macro; Pullback shown) replayed from
   as-traded prices: its own signal, and whether it helps inside owner BUYs
   (label, trend, pullback setup, 12-1 momentum, distance from the high).
3. A single all-agents composite (the pre-September-2026 design, v1 BULL
   weights) against the owner score.
4. Portfolios: owner BUY names, capped or not, yearly or monthly; the owner
   top 20% held only while the timing panel is not unfavorable; the legacy
   all-agents composite; the timing composite alone.

Inputs: the cached value and insider panels from the value study and the same
price files. Prices are the price that traded on the day (splits undone);
returns include dividends; 10 bps per trade.

Run: ``SEC_USER_AGENT="app you@example.com" python -m core.all_agents_study``
(``--rebuild-timing`` rescores the timing panel). Report:
analysis/output/all_agents_study.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from agents.base import UNAVAILABLE, AgentAssessment, signal_from_score
from agents.momentum_agent import MomentumAgent
from agents.pullback_agent import PullbackAgent
from agents.regime_agent import MacroRegimeAgent
from agents.scoring import aggregate, decide, timing_label
from agents.volume_agent import VolumeAgent
from config.weights import load_scoring_config
from core.value_study import (
    COST_BPS,
    PRICE_START,
    add_insider_composite,
    apply_framework_exclusions,
    factor_test,
    forward_returns,
    load_fred,
    month_end_panel,
    newey_west_t,
    performance,
    prepare,
)
from data.macro_data import compute_macro_snapshot, load_macro_history
from data.price_history import as_traded, fetch_price_history

logger = logging.getLogger(__name__)

WINDOW_BARS = 300
MIN_BARS = 210
BUYS = ("BUY", "STRONG_BUY")
SELLS = ("SELL", "STRONG_SELL")
DECISION_ORDER = ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "NOT_RATED")
# The single-composite design before September 2026 (BULL profile; options cannot be replayed).
LEGACY_WEIGHTS = {"momentum": 0.24, "volume": 0.12, "insider": 0.10, "fundamentals": 0.18, "shareholder": 0.08, "macro": 0.16}
SCALES = {
    "momentum": MomentumAgent.score_scale, "volume": VolumeAgent.score_scale, "pullback": PullbackAgent.score_scale,
    "macro": MacroRegimeAgent.score_scale, "fundamentals": 45.0, "shareholder": 35.0, "insider": 35.0,
}


# ------------------------------------------------------------------ timing panel


def _score_timing(task: tuple) -> list[dict[str, Any]]:
    """Momentum, Volume and Pullback agents at each month-end of one ticker (worker process)."""
    ticker, dates = task
    prices = fetch_price_history([ticker], PRICE_START).get(ticker)
    if prices is None or prices.empty:
        return []
    traded = as_traded(prices)[["open", "high", "low", "close", "volume"]]
    agents = {"mom": MomentumAgent(), "vol": VolumeAgent(), "pb": PullbackAgent()}
    rows: list[dict[str, Any]] = []
    for date in dates:
        i = int(traded.index.searchsorted(date, side="right")) - 1
        if i < MIN_BARS or (date - traded.index[i]).days > 7:
            continue
        window = traded.iloc[max(0, i - WINDOW_BARS + 1): i + 1]
        close = window["close"]
        row: dict[str, Any] = {"date": date, "ticker": ticker}
        for key, agent in agents.items():
            try:
                a = agent.evaluate(ticker, window)
            except Exception:  # noqa: BLE001 - one bad window must not sink the ticker
                a = agent.unavailable("agent raised")
            ok = a.data_quality != UNAVAILABLE
            row[f"{key}_score"] = a.score if ok else np.nan
            row[f"{key}_conf"] = a.confidence if ok else 0.0
            if key == "pb":
                row["pb_setup"] = a.raw_data.get("setup") if ok else None
        latest = float(close.iloc[-1])
        sma50 = float(close.tail(50).mean())
        sma200 = float(close.tail(200).mean())
        row["trend"] = "UP" if latest > sma200 and sma50 > sma200 else "DOWN" if latest < sma200 and sma50 < sma200 else "MIXED"
        row["from_high_252"] = latest / float(window["high"].tail(252).max()) - 1.0
        row["ret_1m"] = latest / float(close.iloc[-22]) - 1.0 if len(close) > 22 else np.nan
        rows.append(row)
    return rows


def build_timing_panel(prep: dict[str, Any], tickers: list[str], workers: int) -> pd.DataFrame:
    dates = prep["dates"]
    tasks = []
    for t in tickers:
        member_dates = [d for d in dates if t in prep["per_date"][d]]
        if member_dates:
            tasks.append((t, member_dates))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score_timing, task) for task in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if n % 100 == 0:
                print(f"  timing scored {n}/{len(tasks)} tickers")
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def macro_by_date(dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Macro agent score and regime at each month-end (market-wide, one row per date)."""
    history = load_macro_history(25)
    agent = MacroRegimeAgent()
    rows = []
    for date in dates:
        try:
            snap = compute_macro_snapshot(history, date)
            a = agent.evaluate_with_data(snap)
            ok = a.data_quality != UNAVAILABLE and snap.data_quality != "ESTIMATED"
            rows.append({"date": date, "macro_score": a.score if ok else np.nan, "macro_conf": a.confidence if ok else 0.0,
                         "regime": snap.regime if ok else None, "vix": snap.vix_level})
        except Exception as exc:  # noqa: BLE001 - macro is optional
            logger.warning("Macro snapshot failed for %s: %s", date, exc)
            rows.append({"date": date, "macro_score": np.nan, "macro_conf": 0.0, "regime": None, "vix": np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ composites


def _assess(key: str, score: float, conf: float) -> AgentAssessment:
    ok = pd.notna(score)
    value = float(score) if ok else 0.0
    return AgentAssessment(agent_name=key, score=value, signal=signal_from_score(value), confidence=float(conf or 0.0),
                           summary="", agent_key=key, data_quality="LIVE" if ok else UNAVAILABLE)


def add_composites(panel: pd.DataFrame) -> pd.DataFrame:
    """Timing composite and label, owner decision (raw and capped), legacy all-agents composite."""
    config = load_scoring_config()
    cap = config.owner_max_ev_to_fcf
    panel = panel.copy()
    timing_scores, labels, legacy = [], [], []
    for r in panel.itertuples(index=False):
        timing_in = [_assess("momentum", r.mom_score, r.mom_conf), _assess("volume", r.vol_score, r.vol_conf),
                     _assess("pullback", r.pb_score, r.pb_conf), _assess("macro", r.macro_score, r.macro_conf)]
        t = aggregate(timing_in, config, r.regime, SCALES)
        has_timing = any(c.included for c in t.contributions)
        timing_scores.append(t.composite_score if has_timing else np.nan)
        labels.append(timing_label(t.composite_score, config.thresholds) if has_timing else None)
        owner_in = [_assess("fundamentals", r.fund_score, r.fund_conf), _assess("shareholder", r.sh_score, r.sh_conf),
                    _assess("insider", r.ins_score, r.ins_conf)]
        legacy.append(aggregate(timing_in + owner_in, config, r.regime, SCALES, weight_overrides=LEGACY_WEIGHTS).composite_score)
    panel["timing_score"] = timing_scores
    panel["timing_label"] = labels
    panel["legacy_composite"] = legacy
    panel = apply_framework_exclusions(panel)
    panel["owner_decision"] = [decide(c, config.owner_thresholds) if pd.notna(c) else None for c in panel["composite"]]
    panel.loc[~panel["framework_applicable"], "owner_decision"] = "NOT_RATED"
    expensive = pd.Series(False, index=panel.index)
    if cap is not None:
        ev = panel["ev_to_fcf"]
        expensive = ((ev >= cap) | (ev.isna() & (panel["fcf"] <= 0))) & ~panel["is_financial"].astype(bool)
    panel["capped"] = expensive & panel["owner_decision"].isin(BUYS)
    panel["owner_decision_capped"] = np.where(panel["capped"], "HOLD", panel["owner_decision"])
    panel["timing_ok"] = np.where(panel["timing_label"].eq("UNFAVORABLE"), -1.0, 1.0)
    panel["timing_favorable"] = np.where(panel["timing_label"].eq("FAVORABLE"), 1.0, -1.0)
    return panel


# ------------------------------------------------------------------ analyses


def bucket_returns(panel: pd.DataFrame, fwd1: pd.DataFrame, fwd12: pd.DataFrame, column: str,
                   order: tuple[str, ...] | None = None, subset: pd.Series | None = None) -> dict[str, Any]:
    """Mean forward return by bucket of ``column`` (1 month annualized and 12 months), with a monthly t-stat on 1m."""
    frame = panel if subset is None else panel[subset]
    frame = frame.dropna(subset=[column])
    out: dict[str, Any] = {}
    keys = order or sorted(frame[column].dropna().unique(), key=str)
    for key in keys:
        sub = frame[frame[column] == key]
        r1s, r12s, months = [], [], []
        for date, g in sub.groupby("date"):
            if date not in fwd1.index:
                continue
            r1 = fwd1.loc[date].reindex(g["ticker"]).dropna()
            r12 = fwd12.loc[date].reindex(g["ticker"]).dropna()
            if len(r1) >= 5:
                r1s.append(float(r1.mean()))
                months.append(date)
            if len(r12) >= 5:
                r12s.append(float(r12.mean()))
        if not r1s:
            continue
        r1a = np.array(r1s)
        out[str(key)] = {
            "rows": int(len(sub)),
            "share": round(float(len(sub) / len(frame)), 4),
            "months": len(r1s),
            "mean_1m": float(r1a.mean()),
            "annualized_1m": float((1 + r1a.mean()) ** 12 - 1),
            "t_1m": newey_west_t(r1a, 3),
            "mean_12m": float(np.mean(r12s)) if r12s else None,
            "hit_12m": float(np.mean([r > 0 for r in r12s])) if r12s else None,
        }
    return out


def portfolio_returns(panel: pd.DataFrame, fwd1: pd.DataFrame, column: str, pick: str,
                      rebalance_months: tuple[int, ...] | None = None, mask: pd.Series | None = None,
                      keep_if: str | None = None, min_names: int = 10, cost_bps: float = COST_BPS) -> tuple[pd.Series, float]:
    """Equal-weight monthly returns; like the value study's, with a lower minimum name count for BUY-only sets."""
    out: dict[pd.Timestamp, float] = {}
    turnovers: list[float] = []
    holdings: set[str] = set()
    for date, group in panel.groupby("date"):
        if mask is not None:
            group = group[mask.loc[group.index]]
        valid = group.dropna(subset=[column])
        if len(valid) < min_names:
            continue
        refresh = not holdings or rebalance_months is None or date.month in rebalance_months
        if refresh:
            ranked = valid.sort_values(column, ascending=False)["ticker"].tolist()
            q = max(1, len(ranked) // 5)
            new = set(ranked[:q] if pick == "top_q" else ranked[-q:] if pick == "bottom_q" else ranked)
            if keep_if is not None:
                passing = set(valid.loc[valid[keep_if] > 0, "ticker"])
                new = (new & passing) or new
            turnover = len(new ^ holdings) / (2 * max(len(new), 1)) if holdings else 1.0
            holdings = new
        else:
            holdings = {t for t in holdings if t in set(group["ticker"])}
            if keep_if is not None:
                passing = set(valid.loc[valid[keep_if] > 0, "ticker"])
                holdings = (holdings & passing) or holdings
            turnover = 0.0
        r = fwd1.loc[date, sorted(holdings)] if date in fwd1.index else pd.Series(dtype=float)
        r = r.dropna()
        if r.empty:
            continue
        out[date] = float(r.mean()) - turnover * 2 * cost_bps / 10_000
        turnovers.append(turnover)
    s = pd.Series(out).sort_index()
    s.index = s.index + pd.offsets.MonthEnd(1)
    return s, float(np.mean(turnovers)) if turnovers else float("nan")


def analyze(panel: pd.DataFrame, prep: dict[str, Any]) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    dates = prep["dates"]
    prices = prep["prices"]
    level = month_end_panel(prices, dates + [dates[-1] + pd.offsets.MonthEnd(k) for k in range(1, 13)])
    level.loc[level.index > prices["SPY"].index[-1] + pd.Timedelta(days=3)] = np.nan
    fwd1, fwd12 = forward_returns(level, 1), forward_returns(level, 12)
    rf_m = (load_fred("DTB3", "2010-01-01") / 100 / 12).resample("ME").last().ffill()
    spy = month_end_panel({"SPY": prices["SPY"]}, list(level.index))["SPY"]
    spy.loc[spy.index > prices["SPY"].index[-1] + pd.Timedelta(days=3)] = np.nan
    spy_ret = spy.shift(-1) / spy - 1
    spy_ret.index = spy_ret.index + pd.offsets.MonthEnd(1)

    mom = (level.shift(1) / level.shift(12) - 1).stack().rename("mom_12_1").reset_index()
    mom.columns = ["date", "ticker", "mom_12_1"]
    panel = panel.merge(mom, on=["date", "ticker"], how="left")
    panel["mom_12_1_bucket"] = pd.cut(panel["mom_12_1"], [-np.inf, -0.2, 0.0, 0.2, np.inf],
                                      labels=["below -20%", "-20% to 0", "0 to +20%", "above +20%"]).astype(object)
    panel["from_high_bucket"] = pd.cut(panel["from_high_252"], [-np.inf, -0.3, -0.15, -0.05, np.inf],
                                       labels=["30%+ below high", "15-30% below", "5-15% below", "within 5%"]).astype(object)
    panel["inv_ev_to_ebit"] = 1.0 / panel["ev_to_ebit"].where(panel["ev_to_ebit"] > 0)
    nonfin = ~panel["is_financial"].astype(bool)
    is_buy = panel["owner_decision"].isin(BUYS)
    is_buy_capped = panel["owner_decision_capped"].isin(BUYS)

    report: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                              "rows": int(len(panel)), "months": int(panel["date"].nunique()),
                              "timing_coverage": float(panel["timing_score"].notna().mean())}
    print(f"\nPanel: {len(panel):,} company-months, timing scores on {report['timing_coverage']:.0%} of them.")

    # 1. Owner decision buckets
    print("\nOWNER DECISION BUCKETS (equal-weight forward returns; 1m annualized, 12m raw)\n")
    print(f"{'':<34}{'rows':>8}{'share':>7}{'1m ann.':>9}{'t':>6}{'12m':>8}{'12m>0':>7}")
    for label, column in (("Owner decision", "owner_decision"), ("Owner decision, capped at 35x", "owner_decision_capped")):
        res = bucket_returns(panel, fwd1, fwd12, column, DECISION_ORDER)
        report[column] = res
        print(f"  {label}")
        for k, v in res.items():
            print(f"    {k:<30}{v['rows']:>8,}{v['share']:>7.0%}{v['annualized_1m']:>9.1%}{v['t_1m']:>6.1f}"
                  f"{(v['mean_12m'] or 0):>8.1%}{(v['hit_12m'] or 0):>7.0%}")
    capped = panel[panel["capped"]]
    report["capped_rows"] = {"rows": int(len(capped)), "share_of_buys": float(len(capped) / max(is_buy.sum(), 1))}
    print(f"  Cap: {len(capped):,} BUY rows ({report['capped_rows']['share_of_buys']:.0%} of BUYs) turned into HOLD.")

    # 2. Timing inside owner BUYs and overall
    print("\nTIMING CONDITIONS (forward returns; 'BUYs' = owner BUY or STRONG BUY rows)\n")
    conditions = (
        ("Timing label", "timing_label", ("FAVORABLE", "NEUTRAL", "UNFAVORABLE")),
        ("Trend (50/200 structure)", "trend", ("UP", "MIXED", "DOWN")),
        ("Pullback setup", "pb_setup", ("pullback_in_uptrend", "neutral", "extended", "trend_broken")),
        ("12-1 month momentum", "mom_12_1_bucket", ("below -20%", "-20% to 0", "0 to +20%", "above +20%")),
        ("Distance from 52-week high", "from_high_bucket", ("30%+ below high", "15-30% below", "5-15% below", "within 5%")),
        ("Macro regime", "regime", ("BULL", "CORRECTION", "BEAR", "CRISIS")),
    )
    report["conditions"] = {}
    for label, column, order in conditions:
        within = bucket_returns(panel, fwd1, fwd12, column, order, subset=is_buy)
        overall = bucket_returns(panel, fwd1, fwd12, column, order)
        report["conditions"][column] = {"within_owner_buys": within, "all": overall}
        print(f"  {label:<30}{'BUYs 1m ann.':>13}{'t':>6}{'12m':>8}{'All 1m ann.':>13}{'t':>6}{'12m':>8}")
        for k in order:
            w, a = within.get(k), overall.get(k)
            if not w and not a:
                continue
            f = lambda d, key, fmt: (fmt.format(d[key]) if d and d.get(key) is not None else "n/a")  # noqa: E731
            print(f"    {k:<28}{f(w, 'annualized_1m', '{:>13.1%}')}{f(w, 't_1m', '{:>6.1f}')}{f(w, 'mean_12m', '{:>8.1%}')}"
                  f"{f(a, 'annualized_1m', '{:>13.1%}')}{f(a, 't_1m', '{:>6.1f}')}{f(a, 'mean_12m', '{:>8.1%}')}")

    # 3. Factor tests
    tests = [
        ("Owner score (composite)", "composite", True, None),
        ("Owner + insider composite", "composite_ins", True, None),
        ("Legacy all-agents composite (v1 weights)", "legacy_composite", True, None),
        ("Timing composite (momentum, volume, macro)", "timing_score", True, None),
        ("Momentum agent", "mom_score", True, None),
        ("Volume agent", "vol_score", True, None),
        ("Pullback agent", "pb_score", True, None),
        ("12-1 month momentum", "mom_12_1", True, None),
        ("1-month return (reversal check)", "ret_1m", True, None),
        ("Distance from 52-week high", "from_high_252", True, None),
        ("EBIT / EV", "inv_ev_to_ebit", True, nonfin),
        ("Net dilution cost (stock pay - buybacks) / mcap", "net_dilution_cost_m", False, None),
        ("Asset growth", "asset_growth", False, nonfin),
        ("Incremental ROIC", "incremental_roic", True, nonfin),
    ]
    report["factors"] = {}
    print("\nFACTORS  (rank IC vs next 1 and 12 months, Newey-West t; top minus bottom quintile per year)\n")
    print(f"{'':<48}{'names':>6}{'IC 1m':>8}{'t':>6}{'IC 12m':>8}{'t':>6}{'Q5-Q1':>8}{'1st half':>9}{'2nd half':>9}{'hit':>6}")
    for label, col, higher, subset in tests:
        if col not in panel:
            continue
        res = factor_test(panel, fwd1, fwd12, col, higher, subset)
        if res is None:
            continue
        report["factors"][label] = res
        print(f"{label:<48}{res['avg_names']:>6}{res['ic_1m']:>+8.3f}{res['ic_1m_t']:>6.1f}{res['ic_12m']:>+8.3f}"
              f"{res['ic_12m_t']:>6.1f}{res['q5_minus_q1_annual']:>+8.1%}{res['q5_minus_q1_first_half']:>+9.1%}"
              f"{res['q5_minus_q1_second_half']:>+9.1%}{res['hit_rate']:>6.0%}")

    # 4. Portfolios
    variants: dict[str, tuple[pd.Series, float]] = {
        "Equal-weight universe": portfolio_returns(panel, fwd1, "composite", "all", min_names=50, cost_bps=0),
        "Owner top 20%, monthly": portfolio_returns(panel, fwd1, "composite", "top_q", min_names=50),
        "Owner top 20%, yearly (April)": portfolio_returns(panel, fwd1, "composite", "top_q", (4,), min_names=50),
        "Owner BUY names (uncapped), monthly": portfolio_returns(panel, fwd1, "composite", "all", mask=is_buy),
        "Owner BUY names (uncapped), yearly (April)": portfolio_returns(panel, fwd1, "composite", "all", (4,), mask=is_buy),
        "Owner BUY names, capped at 35x, monthly": portfolio_returns(panel, fwd1, "composite", "all", mask=is_buy_capped),
        "Owner BUY names, capped at 35x, yearly (April)": portfolio_returns(panel, fwd1, "composite", "all", (4,), mask=is_buy_capped),
        "Owner BUY names, held only while timing favorable": portfolio_returns(panel, fwd1, "composite", "all", mask=is_buy, keep_if="timing_favorable"),
        "Owner BUY names, held only while timing not unfavorable": portfolio_returns(panel, fwd1, "composite", "all", mask=is_buy, keep_if="timing_ok"),
        "Owner top 20%, held only while timing not unfavorable": portfolio_returns(panel, fwd1, "composite", "top_q", min_names=50, keep_if="timing_ok"),
        "Owner top 20%, held only while timing favorable": portfolio_returns(panel, fwd1, "composite", "top_q", min_names=50, keep_if="timing_favorable"),
        "Owner top 20%, only 12-1 momentum > 0": portfolio_returns(panel, fwd1, "composite", "top_q", min_names=50, keep_if="mom_12_1"),
        "Legacy all-agents composite top 20%, monthly": portfolio_returns(panel, fwd1, "legacy_composite", "top_q", min_names=50),
        "Legacy all-agents composite top 20%, yearly (April)": portfolio_returns(panel, fwd1, "legacy_composite", "top_q", (4,), min_names=50),
        "Timing composite top 20%, monthly": portfolio_returns(panel, fwd1, "timing_score", "top_q", min_names=50),
        "Timing composite bottom 20%, monthly": portfolio_returns(panel, fwd1, "timing_score", "bottom_q", min_names=50),
        "Owner SELL names, monthly": portfolio_returns(panel, fwd1, "composite", "all", mask=panel["owner_decision"].isin(SELLS)),
    }
    common = None
    for s, _ in variants.values():
        if s.empty:
            continue
        common = s.index if common is None else common.intersection(s.index)
    common = common[common <= spy_ret.dropna().index[-1]]
    halves = [(common[0], common[len(common) // 2 - 1]), (common[len(common) // 2], common[-1])]
    series_by_name = {"S&P 500 (SPY)": spy_ret, **{k: v[0] for k, v in variants.items() if not v[0].empty}}
    rows = {"S&P 500 (SPY)": performance(spy_ret.reindex(common), rf_m.reindex(common))}
    for name, (s, turnover) in variants.items():
        if s.empty:
            continue
        rows[name] = {**performance(s.reindex(common), rf_m.reindex(common)), "turnover": turnover,
                      "months_held": int(s.reindex(common).notna().sum())}
    for name, st in rows.items():
        sub = [performance(series_by_name[name].reindex(common).loc[a:b], rf_m.reindex(common).loc[a:b])["cagr"] for a, b in halves]
        st["cagr_first_half"], st["cagr_second_half"] = sub
    report["portfolios"] = rows
    report["period"] = {"start": str(common[0].date()), "end": str(common[-1].date())}
    print(f"\nPORTFOLIOS {common[0]:%Y-%m} to {common[-1]:%Y-%m} (equal weight, {COST_BPS:g} bps per trade)\n")
    print(f"{'':<58}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'Turn':>6}{'1st half':>10}{'2nd half':>10}")
    for name, st in rows.items():
        turn = f"{st['turnover']:.0%}" if "turnover" in st else ""
        print(f"{name:<58}{st['cagr']:>7.1%}{st['volatility']:>7.1%}{st['sharpe']:>8.2f}{st['max_drawdown']:>8.1%}{turn:>6}"
              f"{st['cagr_first_half']:>10.1%}{st['cagr_second_half']:>10.1%}")
    print(f"  halves: {halves[0][0]:%Y-%m} to {halves[0][1]:%Y-%m} and {halves[1][0]:%Y-%m} to {halves[1][1]:%Y-%m}")

    # Share of the universe rated BUY by year, and how many were capped
    by_year = panel.groupby(panel["date"].dt.year)
    yearly = pd.DataFrame({
        "buy_share": by_year["owner_decision"].apply(lambda s: float(s.isin(BUYS).mean())),
        "buy_share_capped": by_year["owner_decision_capped"].apply(lambda s: float(s.isin(BUYS).mean())),
        "sell_share": by_year["owner_decision"].apply(lambda s: float(s.isin(SELLS).mean())),
        "median_timing": by_year["timing_score"].median(),
    })
    report["by_year"] = {int(y): {k: (None if pd.isna(v) else float(v)) for k, v in r.items()} for y, r in yearly.iterrows()}
    print("\nOWNER RATINGS BY YEAR\n\n  Year  BUY share  BUY share (capped)  SELL share  median timing")
    for y, r in yearly.iterrows():
        print(f"  {y}  {r['buy_share']:>9.0%}  {r['buy_share_capped']:>18.0%}  {r['sell_share']:>10.0%}  {r['median_timing']:>13.0f}")
    return report


def load_scored_panel(prep: dict[str, Any], workers: int, rebuild_timing: bool = False) -> pd.DataFrame | None:
    """The value panel with insider, timing and macro scores and every composite; None without a value panel."""
    value_path = settings.DATA_DIR / "value_panel.parquet"
    insider_path = settings.DATA_DIR / "insider_panel.parquet"
    if not value_path.exists():
        print("No value panel: run `python -m core.value_study` first.")
        return None
    panel = pd.read_parquet(value_path)
    if insider_path.exists():
        panel = panel.merge(pd.read_parquet(insider_path), on=["date", "ticker"], how="left")
    else:
        for col in ("ins_score", "ins_conf", "ins_quality"):
            panel[col] = np.nan
    panel = add_insider_composite(panel)

    timing_path = settings.DATA_DIR / "timing_panel.parquet"
    if timing_path.exists() and not rebuild_timing:
        timing = pd.read_parquet(timing_path)
    else:
        print("Scoring the timing agents on every company-month (cached afterwards)...")
        timing = build_timing_panel(prep, sorted(panel["ticker"].unique()), workers)
        timing.to_parquet(timing_path, index=False)
    macro = macro_by_date(prep["dates"])
    panel = panel.merge(timing, on=["date", "ticker"], how="left").merge(macro, on="date", how="left")
    return add_composites(panel)


def forward_frames(prep: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """(fwd1, fwd12, SPY monthly return, monthly risk-free rate) on the study's month-ends."""
    dates = prep["dates"]
    prices = prep["prices"]
    level = month_end_panel(prices, dates + [dates[-1] + pd.offsets.MonthEnd(k) for k in range(1, 13)])
    level.loc[level.index > prices["SPY"].index[-1] + pd.Timedelta(days=3)] = np.nan
    fwd1, fwd12 = forward_returns(level, 1), forward_returns(level, 12)
    rf_m = (load_fred("DTB3", "2010-01-01") / 100 / 12).resample("ME").last().ffill()
    spy = month_end_panel({"SPY": prices["SPY"]}, list(level.index))["SPY"]
    spy.loc[spy.index > prices["SPY"].index[-1] + pd.Timedelta(days=3)] = np.nan
    spy_ret = spy.shift(-1) / spy - 1
    spy_ret.index = spy_ret.index + pd.offsets.MonthEnd(1)
    return fwd1, fwd12, spy_ret, rf_m


# ------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="All-agents point-in-time backtest on the S&P 500")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--rebuild-timing", action="store_true", help="Rescore the timing agents even if cached")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    end = pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)
    prep = prepare(end)
    panel = load_scored_panel(prep, args.workers, rebuild_timing=args.rebuild_timing)
    if panel is None:
        return 1
    report = analyze(panel, prep)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "all_agents_study.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
