"""VectorBT backtesting engine wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

from analysis.metrics import compute_metrics
from config.settings import COMMISSION_PER_SHARE, INITIAL_CASH


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    portfolio: vbt.Portfolio
    returns: pd.Series
    stats: dict[str, float]


def run_backtest(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    commission_per_share: float = COMMISSION_PER_SHARE,
    init_cash: float = INITIAL_CASH,
) -> BacktestResult:
    """
    Run a long-only backtest from entry/exit signals.

    Args:
        ohlcv: OHLCV DataFrame with a ``close`` column.
        signals: DataFrame with boolean ``entries`` and ``exits`` columns.
        commission_per_share: Fixed commission per share traded (USD).
        init_cash: Starting portfolio cash (USD).

    Returns:
        BacktestResult with portfolio, daily returns, and summary stats.
    """
    close = ohlcv["close"]
    entries = signals["entries"].reindex(close.index).fillna(False).astype(bool)
    exits = signals["exits"].reindex(close.index).fillna(False).astype(bool)

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fixed_fees=commission_per_share,
        freq="1D",
    )

    returns = portfolio.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    stats = compute_metrics(returns)
    return BacktestResult(portfolio=portfolio, returns=returns, stats=stats)
