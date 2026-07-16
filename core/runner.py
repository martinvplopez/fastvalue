"""Main CLI entry point for running backtests."""

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
    DEFAULT_TICKERS,
    INITIAL_CASH,
    OUTPUT_DIR,
)
from core.engine import run_backtest
from data.fetcher import fetch_ohlcv
from strategies.sma_crossover import SMACrossover

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a backtest on cached market data")
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Ticker symbol to backtest (default: SPY)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of market data",
    )
    parser.add_argument(
        "--fast",
        type=int,
        default=50,
        help="Fast SMA window (default: 50)",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=200,
        help="Slow SMA window (default: 200)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Save HTML tear sheet to analysis/output/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Load data, run SMA crossover backtest, and print metrics.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    args = parse_args(argv)
    ticker = args.ticker.upper()

    logger.info("Fetching data for %s", ticker)
    data = fetch_ohlcv([ticker], force_refresh=args.refresh)
    if ticker not in data:
        logger.error("No data available for %s", ticker)
        return 1

    ohlcv = data[ticker]
    strategy = SMACrossover(fast_window=args.fast, slow_window=args.slow)
    strategy_name = f"Dual SMA ({args.fast}/{args.slow})"

    logger.info("Generating signals")
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
            strategy_name=strategy_name,
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
