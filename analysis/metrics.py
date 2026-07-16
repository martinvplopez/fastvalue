"""Performance metrics and tear sheet generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import quantstats as qs


def compute_metrics(returns: pd.Series) -> dict[str, float]:
    """
    Compute core performance metrics from a daily returns series.

    Args:
        returns: Daily percentage returns (e.g. 0.01 = 1%).

    Returns:
        Dict with keys: total_return, cagr, max_drawdown, sharpe.
    """
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    total_return = float((1 + returns).prod() - 1)
    cagr = float(qs.stats.cagr(returns))
    max_drawdown = float(qs.stats.max_drawdown(returns))
    sharpe = float(qs.stats.sharpe(returns))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def compute_buy_and_hold_returns(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Compute daily buy-and-hold returns from close prices.

    Assumes a single purchase on the first bar and no trading afterward.

    Args:
        ohlcv: OHLCV DataFrame with a ``close`` column.

    Returns:
        Daily percentage returns aligned to ``ohlcv`` index.
    """
    return ohlcv["close"].pct_change().fillna(0.0)


def format_metrics(stats: dict[str, float], ticker: str, strategy_name: str) -> str:
    """
    Format metrics dict as a human-readable summary block.

    Args:
        stats: Output from ``compute_metrics``.
        ticker: Symbol being reported.
        strategy_name: Strategy label for the header.

    Returns:
        Formatted multi-line string.
    """
    return (
        f"=== {ticker} — {strategy_name} ===\n"
        f"Total Return:  {stats['total_return']:>8.1%}\n"
        f"CAGR:          {stats['cagr']:>8.1%}\n"
        f"Max Drawdown:  {stats['max_drawdown']:>8.1%}\n"
        f"Sharpe Ratio:  {stats['sharpe']:>8.2f}"
    )


def format_metrics_comparison(
    strategy_stats: dict[str, float],
    benchmark_stats: dict[str, float],
    ticker: str,
    strategy_name: str,
    benchmark_name: str = "Buy & Hold",
) -> str:
    """
    Format strategy metrics alongside a passive benchmark.

    Args:
        strategy_stats: Metrics for the backtested strategy.
        benchmark_stats: Metrics for the buy-and-hold benchmark.
        ticker: Symbol being reported.
        strategy_name: Strategy label for the header.
        benchmark_name: Benchmark label for the comparison column.

    Returns:
        Formatted multi-line comparison table.
    """
    header = f"=== {ticker} — {strategy_name} vs {benchmark_name} ==="
    col_strategy = "Strategy"
    col_benchmark = benchmark_name
    width = max(len(col_strategy), len(col_benchmark), 12)

    return (
        f"{header}\n"
        f"{'':16}{col_strategy:>{width}}  {col_benchmark:>{width}}\n"
        f"{'Total Return:':<16}{strategy_stats['total_return']:>{width}.1%}  "
        f"{benchmark_stats['total_return']:>{width}.1%}\n"
        f"{'CAGR:':<16}{strategy_stats['cagr']:>{width}.1%}  "
        f"{benchmark_stats['cagr']:>{width}.1%}\n"
        f"{'Max Drawdown:':<16}{strategy_stats['max_drawdown']:>{width}.1%}  "
        f"{benchmark_stats['max_drawdown']:>{width}.1%}\n"
        f"{'Sharpe Ratio:':<16}{strategy_stats['sharpe']:>{width}.2f}  "
        f"{benchmark_stats['sharpe']:>{width}.2f}"
    )


def print_tear_sheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    output_path: Path | None = None,
) -> None:
    """
    Print quantstats summary and optionally save an HTML tear sheet.

    Args:
        returns: Daily returns series for the strategy.
        benchmark: Optional benchmark returns for comparison.
        output_path: If provided, write HTML tear sheet to this path.
    """
    qs.extend_pandas()
    qs.reports.metrics(returns, benchmark=benchmark, display=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        qs.reports.html(returns, benchmark=benchmark, output=str(output_path))
