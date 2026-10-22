"""Main CLI entry point for running backtests and multi-agent assessments."""

from __future__ import annotations

import argparse
import logging
import sys

from analysis.metrics import (
    compute_buy_and_hold_returns,
    compute_metrics,
    format_metrics_comparison,
    print_tear_sheet,
)
from config.settings import (
    COMMISSION_PER_SHARE,
    INITIAL_CASH,
    OUTPUT_DIR,
)
from core.comparator import compare_strategies
from core.engine import run_backtest
from core.portfolio import assess_candidate, entry_snapshot, review_portfolio
from core.score_backtest import FREQUENCIES, run_score_backtest
from core.watchlist import analyze_ticker, scan_watchlist
from data.fetcher import fetch_ohlcv
from data.history import load_watchlist
from data.portfolio import load_portfolio, remove_position, set_cash, upsert_position
from strategies import STRATEGY_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _strategy_names() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a backtest or multi-agent evaluation")
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Ticker symbol to backtest (default: SPY)",
    )
    parser.add_argument(
        "--strategy",
        default="sma_crossover",
        choices=_strategy_names(),
        help="Strategy to backtest (default: sma_crossover)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all registered strategies side-by-side",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Run Multi-Agent Decision System for systematic entry/exit analysis",
    )
    parser.add_argument(
        "--watchlist",
        nargs="*",
        metavar="TICKER",
        help="Scan and rank a watchlist (defaults to the saved watchlist)",
    )
    parser.add_argument(
        "--score-backtest",
        action="store_true",
        help="Backtest the entry timing composite (point-in-time agents only)",
    )
    parser.add_argument("--portfolio", action="store_true", help="Review every held position (owner thesis, risk, action)")
    parser.add_argument(
        "--portfolio-add", nargs=2, metavar=("SHARES", "AVG_COST"),
        help="Add or replace the --ticker position, e.g. --ticker AAPL --portfolio-add 10 150.25 [--opened 2026-01-15]",
    )
    parser.add_argument("--opened", default=None, help="ISO date the position was opened (with --portfolio-add)")
    parser.add_argument("--portfolio-remove", action="store_true", help="Remove the --ticker position")
    parser.add_argument("--cash", type=float, default=None, help="Set the portfolio cash balance")
    parser.add_argument("--candidate", action="store_true", help="Should I enter --ticker (or, if held, what is the risk)?")
    parser.add_argument(
        "--frequency",
        default="weekly",
        choices=sorted(FREQUENCIES),
        help="Evaluation frequency for --score-backtest (default: weekly)",
    )
    parser.add_argument("--entry", type=float, default=None, help="Score backtest entry threshold (default: buy threshold)")
    parser.add_argument("--exit", type=float, default=0.0, help="Score backtest exit threshold (default: 0)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of market data",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print available strategies and exit",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Save HTML tear sheet to analysis/output/",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strategy parameter override (e.g. --param fast_window=100)",
    )
    return parser.parse_args(argv)


def _parse_params(raw: list[str], strategy_name: str) -> dict:
    """Parse --param KEY=VALUE args into a typed dict."""
    registry_entry = STRATEGY_REGISTRY[strategy_name]
    param_specs = registry_entry.get("params", {})
    result = dict(registry_entry["defaults"])

    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid param format: {item!r} (expected KEY=VALUE)")
        key, value = item.split("=", 1)
        if key in param_specs:
            ptype = param_specs[key]["type"]
            if ptype == "int":
                result[key] = int(value)
            elif ptype == "float":
                result[key] = float(value)
            else:
                result[key] = value
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value

    return result


def main(argv: list[str] | None = None) -> int:
    """Load data, run selected strategy or multi-agent evaluation, and print report."""
    args = parse_args(argv)

    if args.list_strategies:
        print("\nAvailable strategies:\n")
        for name in _strategy_names():
            entry = STRATEGY_REGISTRY[name]
            print(f"  {name:20s}  {entry['description']}")
            if entry.get("params"):
                for pname, pinfo in entry["params"].items():
                    print(f"    --param {pname}={pinfo['default']}  ({pinfo['label']})")
        print()
        return 0

    if args.watchlist is not None:
        tickers = [t.upper() for t in args.watchlist] or load_watchlist()
        logger.info("Scanning watchlist: %s", ", ".join(tickers))
        result = scan_watchlist(tickers)
        print(f"\n{'Ticker':<8}{'Decision':<13}{'Score':>7}{'Change':>8}{'Conf':>6}  {'Regime':<11}Flags")
        print("-" * 72)
        for r in result["rows"]:
            change = f"{r['score_change']:+.1f}" if r["score_change"] is not None else "new"
            flags = []
            if r["decision_changed"]:
                flags.append(f"{r['previous_decision']} -> {r['decision']}")
            if r["conflict"]:
                flags.append("conflict")
            if r["coverage"] < 0.7:
                flags.append("low coverage")
            print(
                f"{r['ticker']:<8}{r['decision']:<13}{r['composite_score']:>+7.1f}{change:>8}"
                f"{r['confidence']:>5.0f}%  {r['regime']:<11}{', '.join(flags)}"
            )
        for e in result["errors"]:
            print(f"{e['ticker']:<8}ERROR: {e['error']}")
        print()
        return 0

    ticker = args.ticker.upper()

    if args.cash is not None:
        p = set_cash(args.cash)
        print(f"Cash set to {p['cash']:,.2f} {p['currency']}")
        if not (args.portfolio or args.portfolio_add or args.portfolio_remove or args.candidate):
            return 0

    if args.portfolio_add:
        shares, cost = args.portfolio_add
        position = {"ticker": ticker, "shares": shares, "avg_cost": cost, "opened": args.opened}
        try:
            rec, _ = analyze_ticker(ticker)
            position.update(entry_snapshot(rec))
            print(f"{ticker}: owner {rec.decision} ({rec.composite_score:+.1f}) stored as the entry rating")
        except Exception as exc:  # noqa: BLE001 - the snapshot is optional
            logger.warning("Could not rate %s at entry: %s", ticker, exc)
        p = upsert_position(position)
        print(f"Portfolio now holds {len(p['positions'])} position(s), cash {p['cash']:,.2f}")
        return 0

    if args.portfolio_remove:
        p = remove_position(ticker)
        print(f"Removed {ticker}. Portfolio now holds {len(p['positions'])} position(s).")
        return 0

    if args.portfolio:
        p = load_portfolio()
        if not p["positions"]:
            print("The portfolio is empty. Add a position with --ticker AAPL --portfolio-add SHARES AVG_COST [--opened DATE].")
            return 0
        result = review_portfolio(p)
        s = result["summary"]
        print(f"\nPORTFOLIO REVIEW {result['reviewed_at']}  total {s['total_value']:,.0f}  invested {s['invested_value']:,.0f}  "
              f"cash {s['cash']:,.0f} ({(s['cash_weight'] or 0):.0%})  unrealized {s['unrealized_pnl']:+,.0f}")
        vol = f"{s['volatility']:.0%}" if s.get("volatility") is not None else "n/a"
        mdd = f"{s['max_drawdown_1y']:.0%}" if s.get("max_drawdown_1y") is not None else "n/a"
        corr = f"{s['avg_correlation']:.2f}" if s.get("avg_correlation") is not None else "n/a"
        print(f"Volatility {vol}  max drawdown 1y {mdd}  avg correlation {corr}  effective positions {s.get('effective_positions') or 'n/a'}")
        for n in result["notes"]:
            print(f"  NOTE: {n}")
        print(f"\n{'Ticker':<8}{'Action':<8}{'Weight':>7}{'P&L':>8}{'Owner':>14}{'vs entry':>9}{'Timing':>13}{'DD':>7}  Flags")
        print("-" * 96)
        for r in result["positions"]:
            change = f"{r['owner_change']:+.0f}" if r["owner_change"] is not None else "n/a"
            dd = f"{r['drawdown']['from_high']:.0%}" if r["drawdown"].get("from_high") is not None else "n/a"
            print(f"{r['ticker']:<8}{r['action']:<8}{r['weight']:>7.1%}{r['pnl_pct']:>+8.1%}"
                  f"{r['decision']:>11} {r['owner_score']:>+3.0f}{change:>9}{(r['timing_label'] or 'n/a'):>13}{dd:>7}  {', '.join(r['flags'])}")
            for reason in r["reasons"]:
                print(f"         {reason}")
        for e in result["errors"]:
            print(f"{e['ticker']:<8}ERROR: {e['error']}")
        print()
        return 0

    if args.candidate:
        try:
            result = assess_candidate(ticker)
        except LookupError as exc:
            logger.error("%s", exc)
            return 1
        a = result["analysis"]
        print(f"\n{ticker}: {result['verdict']}  (owner {a['decision']} {a['composite_score']:+.1f}, timing {a['timing']['label']} "
              f"{a['timing']['score']:+.0f}, {'held' if result['held'] else 'not held'})")
        for reason in result["reasons"]:
            print(f"  - {reason}")
        if result.get("sizing"):
            sz = result["sizing"]
            if sz.get("shares") is not None:
                print(f"  Size: {sz['shares']:,} shares (~{sz['value']:,.0f}, {sz['weight']:.1%} of the portfolio), binding: {sz['binding']}")
            print(f"  {sz['note']}")
        st = result.get("staging") or {}
        if st.get("tranches", 1) > 1:
            print(f"  Staging: {st['tranches']} tranches over about {st['over_weeks']} weeks")
        for c in result.get("concerns") or []:
            print(f"  Concern: {c}")
        for n in result.get("timing_notes") or []:
            print(f"  Timing: {n}")
        print()
        return 0

    if args.multi_agent:
        logger.info("Executing Multi-Agent Decision Framework on %s", ticker)
        try:
            rec, _ = analyze_ticker(ticker)
        except LookupError as exc:
            logger.error("%s", exc)
            return 1
        fresh = rec.freshness or {}
        line = "=" * 72
        print(f"\n{line}\n OWNER-FIRST DECISION FOR {ticker}\n{line}")
        if rec.decision == "NOT_RATED":
            print(f" Owner decision:   NOT_RATED (owner evidence {rec.composite_score:+.1f} / 100 shown for information only, "
                  f"{rec.horizon} horizon)")
        else:
            print(f" Owner decision:   {rec.decision} (owner score {rec.composite_score:+.1f} / 100, confidence "
                  f"{rec.overall_confidence:.0f}%, {rec.horizon} horizon)")
        t = rec.timing
        print(f" Entry timing:     {t.get('label')} ({t.get('score', 0):+.1f}), trend {t.get('trend')}, setup {t.get('setup') or 'n/a'}, "
              f"regime {rec.market_regime}")
        print(f" Data:             {fresh.get('status', 'n/a')} as of {fresh.get('as_of', 'n/a')}, owner coverage {rec.coverage:.0%}")
        print(f" Risk Rating:      {rec.risk_rating} ({rec.risk_score:.1f}/10)")
        print(f" Price / Stop ref / Target: ${rec.suggested_entry:.2f} / ${rec.suggested_stop_loss:.2f} / "
              f"${rec.suggested_target_price:.2f} (R:R {rec.reward_risk_ratio}x)")
        print(f" Stop method:      {rec.stop_loss_method}")
        for w in rec.warnings:
            print(f" WARNING: {w}")
        for n in t.get("notes", []):
            print(f" TIMING: {n}")
        print(f"\n {rec.executive_summary}\n")
        print(f" {'Agent':<32}{'Layer':<8}{'Quality':<12}{'Score':>7}{'Conf':>6}{'Weight':>8}{'Contrib':>9}")
        for c in rec.contributions:
            if c["included"]:
                print(f" {c['agent_name']:<32}{c.get('layer', ''):<8}{c['data_quality']:<12}{c['raw_score']:>+7.1f}{c['confidence']:>5.0f}%"
                      f"{c['weight_share']:>8.0%}{c['contribution']:>+9.1f}")
            else:
                print(f" {c['agent_name']:<32}{c.get('layer', ''):<8}{c['data_quality']:<12}{'':>7}{'':>6}{'':>8}  ({c['exclusion_reason']})")
        print()
        for a in rec.agent_assessments:
            print(f" {a.agent_name}: {a.summary}")
            for f in a.key_factors:
                print(f"   - {f}")
        print(f"{line}\n")
        return 0

    logger.info("Fetching market data for %s", ticker)
    data = fetch_ohlcv([ticker], force_refresh=args.refresh)
    if ticker not in data:
        logger.error("No market data available for %s", ticker)
        return 1

    ohlcv = data[ticker]

    if args.score_backtest:
        logger.info("Backtesting the composite score on %s (%s)", ticker, args.frequency)
        res = run_score_backtest(ticker, ohlcv, frequency=args.frequency, entry_threshold=args.entry, exit_threshold=args.exit)
        s, b = res["strategy_stats"], res["benchmark_stats"]
        cal = res["calibration"]
        print(f"\nComposite score backtest for {ticker}: {res['evaluations']} {res['frequency']} evaluations, "
              f"{res['date_range']['start']} to {res['date_range']['end']}")
        print(f"Agents: {', '.join(a['name'] for a in res['agents_used'])}")
        for e in res["agents_excluded"]:
            print(f"  excluded {e['agent']}: {e['reason']}")
        print(f"\n{'':16}{'Strategy':>12}{'Buy & Hold':>12}")
        for key, label, pct in (("total_return", "Total return", True), ("cagr", "CAGR", True),
                                ("sharpe", "Sharpe", False), ("max_drawdown", "Max drawdown", True)):
            fmt = "{:>12.1%}" if pct else "{:>12.2f}"
            print(f"{label:<16}{fmt.format(s[key])}{fmt.format(b[key])}")
        print(f"{'Trades':<16}{s['num_trades']:>12}   exposure {s['exposure']:.0%}")
        print(f"\nCalibration ({cal['horizon_bars']}-bar forward returns), IC {cal['information_coefficient']}")
        for bk in cal["buckets"]:
            mean = f"{bk['mean_forward_return']:+.2%}" if bk["mean_forward_return"] is not None else "n/a"
            hit = f"{bk['hit_rate']:.0%}" if bk["hit_rate"] is not None else "n/a"
            print(f"  {bk['decision']:<12} n={bk['count']:<5} mean {mean:>8}  hit {hit}")
        for n in res["notes"]:
            print(f"\nNote: {n}")
        print()
        return 0

    if args.compare:
        logger.info("Comparing all strategies for %s", ticker)
        comp = compare_strategies(ticker, ohlcv=ohlcv)

        print(f"\n=======================================================")
        print(f" STRATEGY COMPARISON MATRIX FOR {ticker}")
        print(f"=======================================================")
        print(f"{'Rank':<5}{'Strategy':<22}{'Return':<10}{'CAGR':<10}{'Sharpe':<10}{'Drawdown':<10}{'WinRate':<10}")
        print("-" * 77)
        for r in comp.rankings:
            print(
                f"{r['rank']:<5}{r['display_name']:<22}"
                f"{r['total_return']:>8.1%}  "
                f"{r['cagr']:>8.1%}  "
                f"{r['sharpe']:>8.2f}  "
                f"{r['max_drawdown']:>8.1%}  "
                f"{r['win_rate']:>8.1%}"
            )
        print(f"=======================================================\n")
        return 0

    strategy_name = args.strategy
    params = _parse_params(args.param, strategy_name)
    registry_entry = STRATEGY_REGISTRY[strategy_name]
    strategy = registry_entry["class"](**params)
    display_name = f"{strategy_name.replace('_', ' ').title()}"

    logger.info("Generating signals with %s", display_name)
    signals = strategy.generate_signals(ohlcv)

    logger.info("Running backtest")
    result = run_backtest(
        ohlcv=ohlcv,
        signals=signals,
        commission_per_share=COMMISSION_PER_SHARE,
        init_cash=INITIAL_CASH,
    )

    benchmark_returns = compute_buy_and_hold_returns(ohlcv)
    benchmark_returns = benchmark_returns.reindex(result.returns.index).fillna(0.0)
    benchmark_stats = compute_metrics(benchmark_returns)

    print(
        format_metrics_comparison(
            strategy_stats=result.stats,
            benchmark_stats=benchmark_stats,
            ticker=ticker,
            strategy_name=display_name,
        )
    )

    if args.html:
        output_path = OUTPUT_DIR / f"{ticker}_tearsheet.html"
        print_tear_sheet(
            result.returns,
            benchmark=benchmark_returns,
            output_path=output_path,
        )
        logger.info("Tear sheet saved to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
