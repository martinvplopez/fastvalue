"""Performance metrics and tear sheet generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import quantstats as qs


def compute_metrics(
    returns: pd.Series,
    signals: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Compute performance metrics from a daily returns series.

    Args:
        returns: Daily percentage returns (e.g. 0.01 = 1%).
        signals: Optional DataFrame with ``entries`` / ``exits`` columns
                 used to compute trade-level statistics.

    Returns:
        Dict with keys: total_return, cagr, max_drawdown, sharpe,
        sortino, calmar, win_rate, profit_factor, num_trades.
    """
    empty: dict[str, float] = {
        "total_return": 0.0,
        "cagr": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "num_trades": 0,
    }

    returns = returns.dropna()
    if returns.empty:
        return empty

    total_return = float((1 + returns).prod() - 1)
    cagr = float(qs.stats.cagr(returns))
    max_drawdown = float(qs.stats.max_drawdown(returns))
    sharpe = float(qs.stats.sharpe(returns))

    # Sortino — penalises only downside volatility
    downside = returns[returns < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    mean_return = float(returns.mean())
    sortino = (mean_return / downside_std * (252 ** 0.5)) if downside_std > 0 else 0.0

    # Calmar — CAGR / |max drawdown|
    calmar = (cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # Trade-level metrics from signals
    win_rate = 0.0
    profit_factor = 0.0
    num_trades = 0

    if signals is not None and "entries" in signals.columns and "exits" in signals.columns:
        trade_stats = _compute_trade_metrics(returns, signals)
        win_rate = trade_stats["win_rate"]
        profit_factor = trade_stats["profit_factor"]
        num_trades = trade_stats["num_trades"]

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
    }


def _compute_trade_metrics(
    returns: pd.Series, signals: pd.DataFrame
) -> dict[str, float]:
    """
    Compute win rate, profit factor, and trade count from signals.

    A "trade" is defined as the period between an entry signal and the
    next exit signal.
    """
    entries = signals["entries"].reindex(returns.index).fillna(False).astype(bool)
    exits = signals["exits"].reindex(returns.index).fillna(False).astype(bool)

    in_trade = False
    trade_returns: list[float] = []
    current_trade_return = 1.0

    for i in range(len(returns)):
        if entries.iloc[i] and not in_trade:
            in_trade = True
            current_trade_return = 1.0
        if in_trade:
            current_trade_return *= 1 + returns.iloc[i]
        if exits.iloc[i] and in_trade:
            trade_returns.append(current_trade_return - 1)
            in_trade = False
            current_trade_return = 1.0

    # Close any open trade
    if in_trade:
        trade_returns.append(current_trade_return - 1)

    num_trades = len(trade_returns)
    if num_trades == 0:
        return {"win_rate": 0.0, "profit_factor": 0.0, "num_trades": 0}

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]

    win_rate = len(wins) / num_trades
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
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
