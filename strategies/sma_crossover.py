"""Dual simple moving average crossover strategy."""

from __future__ import annotations

import pandas as pd


class SMACrossover:
    """
    Long-only dual SMA crossover strategy.

    Enters on a golden cross (fast SMA crosses above slow SMA) and exits
    on a death cross. Signals are lagged by one bar: a crossover detected
    at the close of day T produces a trade signal for day T+1, preventing
    look-ahead bias from using same-bar close prices for execution.
    """

    def __init__(self, fast_window: int = 50, slow_window: int = 200) -> None:
        """
        Args:
            fast_window: Period for the fast SMA (default 50).
            slow_window: Period for the slow SMA (default 200).

        Raises:
            ValueError: If fast_window >= slow_window or windows < 1.
        """
        if fast_window < 1 or slow_window < 1:
            raise ValueError("Window sizes must be positive integers")
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window")

        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lagged entry/exit signals from OHLCV close prices.

        Args:
            ohlcv: DataFrame with at least a ``close`` column.

        Returns:
            DataFrame with ``entries`` and ``exits`` boolean columns.
        """
        close = ohlcv["close"]
        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()
        bullish = fast_ma > slow_ma

        prev_bullish = bullish.shift(1).fillna(False).astype(bool)
        entries = bullish & (~prev_bullish)
        exits = (~bullish.astype(bool)) & prev_bullish

        # Lag signals by one bar: observe crossover at close T, trade on T+1
        entries = entries.shift(1).fillna(False).astype(bool)
        exits = exits.shift(1).fillna(False).astype(bool)

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
