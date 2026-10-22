"""MACD crossover strategy."""

from __future__ import annotations

import pandas as pd


class MACDCrossover:
    """
    Long-only MACD crossover strategy.

    The Moving Average Convergence Divergence (MACD) measures the
    relationship between two exponential moving averages.  A buy signal
    fires when the MACD line crosses above the signal line; a sell signal
    fires on the reverse cross.

    All signals are lagged by one bar to prevent look-ahead bias.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        """
        Args:
            fast_period: Fast EMA period (default 12).
            slow_period: Slow EMA period (default 26).
            signal_period: Signal line EMA period (default 9).

        Raises:
            ValueError: If parameters are invalid.
        """
        if fast_period < 1 or slow_period < 1 or signal_period < 1:
            raise ValueError("All periods must be positive integers")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lagged entry/exit signals from MACD crossover.

        Args:
            ohlcv: DataFrame with at least a ``close`` column.

        Returns:
            DataFrame with ``entries`` and ``exits`` boolean columns.
        """
        close = ohlcv["close"]

        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        bullish = macd_line > signal_line
        prev_bullish = bullish.shift(1).fillna(False).astype(bool)

        entries = bullish & (~prev_bullish)   # MACD crosses above signal
        exits = (~bullish.astype(bool)) & prev_bullish  # MACD crosses below signal

        # Lag signals by one bar
        entries = entries.shift(1).fillna(False).astype(bool)
        exits = exits.shift(1).fillna(False).astype(bool)

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
