"""Long-horizon benchmark study: do the agents beat the S&P 500?

Two tests, both using only point-in-time agents (the data each agent sees at a
date is data that existed on that date):

1. Market timing: hold SPY while the composite score is at or above the entry
   threshold, move to T-bills when it falls to the exit threshold. Cash earns
   the 3-month T-bill rate (FRED DTB3); every change in exposure costs
   ``cost_bps``.
2. Stock selection: each month-end, rank a fixed historical universe (the Dow
   30 members as of 2006, so the list is not chosen with hindsight) by the
   composite score and hold the top N equally weighted for the next month.
   Compared with SPY and with an equal-weight portfolio of the same universe,
   which separates the agents' skill from the universe itself.

Run: ``python -m core.benchmark_study --years 20``. A JSON report is written to
analysis/output/benchmark_study.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from config.weights import ScoringConfig, load_scoring_config
from core.score_backtest import WARMUP_BARS, replay_composite_scores, resolve_agents
from data.fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

# Dow Jones Industrial Average members in 2006 (a universe fixed before the test period).
# Old GM (bankrupt 2009) has no price history on Yahoo and is excluded; the
# equal-weight benchmark excludes it too, so the picks-vs-universe comparison is fair.
DOW_2006: tuple[str, ...] = (
    "MMM", "AA", "MO", "AIG", "AXP", "T", "BA", "CAT", "C", "KO", "DD", "XOM", "GE", "HPQ", "HD",
    "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT", "PFE", "PG", "RTX", "VZ", "WMT", "DIS",
)
TBILL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd={start}"


# ------------------------------------------------------------------ data


def load_tbill_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily simple return of 3-month T-bills aligned to ``index`` (0 if unavailable)."""
    try:
        df = pd.read_csv(TBILL_URL.format(start=(index[0] - pd.Timedelta(days=10)).strftime("%Y-%m-%d")))
        rate = pd.Series(pd.to_numeric(df["DTB3"], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]))
        rate = rate.reindex(index, method="ffill").fillna(0.0)
        return rate / 100.0 / 252.0
    except Exception as exc:  # noqa: BLE001 - study still runs, noted in the report
        logger.warning("T-bill rates unavailable (%s); cash earns 0%%", exc)
        return pd.Series(0.0, index=index)


# --------------------------------------------------------------- statistics


def performance(returns: pd.Series, rf: pd.Series, periods: int) -> dict[str, float]:
    """Total return, CAGR, volatility, Sharpe (excess over T-bills), max drawdown, Calmar."""
    returns = returns.dropna()
    rf = rf.reindex(returns.index).fillna(0.0)
    equity = (1 + returns).cumprod()
    years = len(returns) / periods
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    vol = float(returns.std() * math.sqrt(periods))
    excess = returns - rf
    sharpe = float(excess.mean() / returns.std() * math.sqrt(periods)) if returns.std() > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else 0.0,
    }


def yearly(returns: pd.Series) -> pd.Series:
    return (1 + returns).groupby(returns.index.year).prod() - 1


# ------------------------------------------------------------ market timing


def timing_returns(
    close: pd.Series,
    scores: pd.Series,
    rf: pd.Series,
    entry: float,
    exit_: float,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    """Daily strategy returns and exposure from scores at evaluation dates."""
    state, positions = 0.0, {}
    for date, value in scores.items():
        if state == 0.0 and value >= entry:
            state = 1.0
        elif state == 1.0 and value <= exit_:
            state = 0.0
        positions[date] = state
    pos = pd.Series(positions).reindex(close.index).ffill().fillna(0.0)
    held = pos.shift(1).fillna(0.0)  # decided at the close, earns from the next bar
    asset = close.pct_change().fillna(0.0)
    cost = held.diff().abs().fillna(held.iloc[0]) * cost_bps / 10_000
    return held * asset + (1 - held) * rf - cost, held


def attribution(close: pd.Series, held: pd.Series) -> dict[str, float]:
    """How SPY behaved while the strategy was in versus out of the market."""
    asset = close.pct_change().fillna(0.0)
    inside, outside = asset[held == 1], asset[held == 0]
    ann = lambda r: float((1 + r).prod() ** (252 / len(r)) - 1) if len(r) else float("nan")  # noqa: E731
    switches = int((held.diff().abs() > 0).sum())
    return {
        "exposure": float(held.mean()),
        "spy_annualized_while_invested": ann(inside),
        "spy_annualized_while_in_cash": ann(outside),
        "round_trips": switches // 2,
    }


def run_timing(
    spy: pd.DataFrame,
    agent_keys: list[str],
    config: ScoringConfig,
    start: pd.Timestamp,
    rf: pd.Series,
    entry: float,
    exit_: float,
    cost_bps: float,
    step: int = 5,
) -> dict[str, Any]:
    keys, excluded = resolve_agents(agent_keys, config)
    first = max(WARMUP_BARS, int(spy.index.searchsorted(start)))
    positions = list(range(first, len(spy), step))
    scores = replay_composite_scores("SPY", spy, keys, config, positions)
    close = spy["close"].loc[scores.index[0]:]
    strat, held = timing_returns(close, scores["composite"], rf.reindex(close.index).fillna(0.0), entry, exit_, cost_bps)
    return {"keys": keys, "excluded": excluded, "scores": scores, "returns": strat, "held": held, "close": close}


# ---------------------------------------------------------- stock selection


def month_end_positions(index: pd.DatetimeIndex, start: pd.Timestamp) -> list[int]:
    dates = pd.Series(index, index=index)
    ends = dates.groupby([index.year, index.month]).max()
    return [int(index.get_loc(d)) for d in ends if d >= start and index.get_loc(d) >= WARMUP_BARS]


def run_selection(
    universe: dict[str, pd.DataFrame],
    agent_keys: list[str],
    config: ScoringConfig,
    start: pd.Timestamp,
    top_n: int,
    cost_bps: float,
) -> dict[str, Any]:
    keys, excluded = resolve_agents(agent_keys, config)
    scores: dict[str, pd.Series] = {}
    monthly_close: dict[str, pd.Series] = {}
    for ticker, df in universe.items():
        pos = month_end_positions(df.index, start)
        if not pos:
            continue
        s = replay_composite_scores(ticker, df, keys, config, pos)["composite"]
        s.index = s.index.to_period("M")
        scores[ticker] = s
        mc = df["close"].groupby([df.index.year, df.index.month]).last()
        mc.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq="M") for y, m in mc.index])
        monthly_close[ticker] = mc

    score_panel = pd.DataFrame(scores).sort_index()
    close_panel = pd.DataFrame(monthly_close).sort_index()
    fwd = close_panel.pct_change().shift(-1)  # return over the month after the ranking date

    rows = []
    prev_top: set[str] = set()
    ics = []
    for month in score_panel.index[:-1]:
        s = score_panel.loc[month].dropna()
        r = fwd.loc[month].reindex(s.index).dropna()
        s = s.reindex(r.index)
        if len(s) < top_n * 2:
            continue
        ranked = s.sort_values(ascending=False)
        top, bottom = list(ranked.index[:top_n]), list(ranked.index[-top_n:])
        turnover = len(set(top) ^ prev_top) / (2 * top_n) if prev_top else 1.0
        prev_top = set(top)
        rows.append({
            "month": month,
            "top": float(r[top].mean()) - turnover * 2 * cost_bps / 10_000,
            "bottom": float(r[bottom].mean()),
            "universe_ew": float(r.mean()),
            "holdings": top,
            "turnover": turnover,
        })
        if s.nunique() > 1:
            ics.append(float(s.rank().corr(r.rank())))

    frame = pd.DataFrame(rows).set_index("month")
    frame.index = frame.index.to_timestamp(how="end").normalize()
    ic = np.array(ics)
    return {
        "keys": keys,
        "excluded": excluded,
        "monthly": frame,
        "mean_ic": float(ic.mean()) if len(ic) else float("nan"),
        "ic_t_stat": float(ic.mean() / ic.std() * math.sqrt(len(ic))) if len(ic) > 2 and ic.std() > 0 else float("nan"),
        "ic_positive_share": float((ic > 0).mean()) if len(ic) else float("nan"),
        "universe_size": len(score_panel.columns),
    }


def monthly_from_daily(returns: pd.Series) -> pd.Series:
    out = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    out.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq="M") for y, m in out.index]).to_timestamp(how="end").normalize()
    return out


# ------------------------------------------------------------------ report


def _fmt_stats(name: str, s: dict[str, float], extra: str = "") -> str:
    return (f"{name:<34}{s['total_return']:>9.0%}{s['cagr']:>8.1%}{s['volatility']:>8.1%}"
            f"{s['sharpe']:>8.2f}{s['max_drawdown']:>9.1%}{s['calmar']:>8.2f}  {extra}")


HEADER = f"{'':<34}{'Total':>9}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Do the agents beat the S&P 500?")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--entry", type=float, default=None, help="Timing entry threshold (default: buy threshold)")
    parser.add_argument("--exit", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=10, help="Stocks held in the selection test")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="One-way cost per trade for SPY timing")
    parser.add_argument("--stock-cost-bps", type=float, default=10.0, help="One-way cost per trade for stocks")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_scoring_config()
    entry = args.entry if args.entry is not None else config.thresholds["buy"]
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=args.years)
    data_start = (start - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    spy = fetch_ohlcv(["SPY"], start=data_start)["SPY"]
    rf_daily = load_tbill_daily(spy.index)
    report: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds"), "years": args.years,
                              "entry": entry, "exit": args.exit, "timing": {}, "selection": {}}

    # ---- 1. Market timing on SPY
    variants = {
        "Composite (momentum+volume+macro)": ["momentum", "volume", "macro"],
        "Momentum agent only": ["momentum"],
        "Volume agent only": ["volume"],
        "Macro agent only (contrarian VIX)": ["macro"],
    }
    results = {name: run_timing(spy, keys, config, start, rf_daily, entry, args.exit, args.cost_bps)
               for name, keys in variants.items()}
    first = max(r["returns"].index[0] for r in results.values())
    spy_ret = spy["close"].pct_change().loc[first:].fillna(0.0)
    rf = rf_daily.loc[first:]
    bench = performance(spy_ret, rf, 252)
    print(f"\n1. MARKET TIMING ON SPY  {first:%Y-%m-%d} to {spy.index[-1]:%Y-%m-%d}  "
          f"(enter >= {entry:g}, exit <= {args.exit:g}, weekly, cash in T-bills, {args.cost_bps:g} bps per trade)\n")
    print(HEADER)
    print(_fmt_stats("S&P 500 buy and hold (SPY)", bench))
    print(_fmt_stats("T-bills only", performance(rf, rf, 252)))
    report["timing"]["benchmark"] = bench
    for name, r in results.items():
        ret = r["returns"].loc[first:]
        stats = performance(ret, rf, 252)
        attr = attribution(r["close"].loc[first:], r["held"].loc[first:])
        report["timing"][name] = {**stats, **attr}
        print(_fmt_stats(name, stats, f"in market {attr['exposure']:.0%}, {attr['round_trips']} round trips"))

    comp = results["Composite (momentum+volume+macro)"]
    comp_ret = comp["returns"].loc[first:]
    attr = attribution(comp["close"].loc[first:], comp["held"].loc[first:])
    print(f"\n   While the composite was invested, SPY returned {attr['spy_annualized_while_invested']:.1%} a year; "
          f"while it sat in cash, SPY returned {attr['spy_annualized_while_in_cash']:.1%} a year.")
    years_tbl = pd.DataFrame({"SPY": yearly(spy_ret), "Composite": yearly(comp_ret)})
    years_tbl["Diff"] = years_tbl["Composite"] - years_tbl["SPY"]
    report["timing"]["yearly"] = {int(y): {k: float(v) for k, v in row.items()} for y, row in years_tbl.iterrows()}
    print("\n   Year     SPY  Composite   Diff")
    for y, row in years_tbl.iterrows():
        print(f"   {y}  {row['SPY']:>6.1%}  {row['Composite']:>8.1%}  {row['Diff']:>+6.1%}")

    halves = [(first, first + (spy.index[-1] - first) / 2), (first + (spy.index[-1] - first) / 2, spy.index[-1])]
    print("\n   Sub-periods (composite vs SPY, CAGR and max drawdown):")
    for a, b in halves:
        s1 = performance(comp_ret.loc[a:b], rf.loc[a:b], 252)
        s2 = performance(spy_ret.loc[a:b], rf.loc[a:b], 252)
        print(f"   {a:%Y-%m} to {b:%Y-%m}: composite {s1['cagr']:.1%} / {s1['max_drawdown']:.0%}, SPY {s2['cagr']:.1%} / {s2['max_drawdown']:.0%}")

    # ---- 2. Stock selection on the 2006 Dow
    raw = fetch_ohlcv(list(DOW_2006), start=data_start)
    universe = {t: df for t, df in raw.items() if len(df) > WARMUP_BARS + 60}
    sel = run_selection(universe, ["momentum", "volume", "macro"], config, start, args.top, args.stock_cost_bps)
    m = sel["monthly"]
    spy_m = monthly_from_daily(spy["close"].pct_change().fillna(0.0)).reindex(m.index).fillna(0.0)
    rf_m = monthly_from_daily(rf_daily).reindex(m.index).fillna(0.0)
    print(f"\n2. STOCK SELECTION  {m.index[0]:%Y-%m} to {m.index[-1]:%Y-%m}  (2006 Dow members, {sel['universe_size']} with data; "
          f"monthly top {args.top} by composite, {args.stock_cost_bps:g} bps per trade)\n")
    print(HEADER)
    rows = {
        "S&P 500 (SPY)": spy_m,
        "Equal-weight 2006 Dow universe": m["universe_ew"],
        f"Agents' top {args.top}": m["top"],
        f"Agents' bottom {args.top} (control)": m["bottom"],
    }
    for name, series in rows.items():
        stats = performance(series, rf_m, 12)
        report["selection"][name] = stats
        print(_fmt_stats(name, stats))
    print(f"\n   Monthly rank correlation between score and next-month return: mean {sel['mean_ic']:+.3f} "
          f"(t-stat {sel['ic_t_stat']:+.2f}, positive in {sel['ic_positive_share']:.0%} of months); "
          f"average monthly turnover {m['turnover'].mean():.0%}.")
    report["selection"]["ic"] = {"mean": sel["mean_ic"], "t_stat": sel["ic_t_stat"], "positive_share": sel["ic_positive_share"]}
    report["selection"]["turnover"] = float(m["turnover"].mean())

    from agents.registry import get_agent_classes

    excluded = [cls.name for cls in get_agent_classes().values() if not cls.point_in_time]
    print("\nNot replayed (no point-in-time history): " + (", ".join(excluded) or "none") + "\n")
    report["excluded"] = excluded

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "benchmark_study.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
