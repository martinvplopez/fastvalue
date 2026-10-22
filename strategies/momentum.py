"""Momentum (Rate of Change) trend-following strategy."""

from __future__ import annotations

import pandas as pd


class Momentum:
    """
    Long-only momentum / rate-of-change strategy.

    Measures the percentage change in price over a lookback window.  A buy
    signal fires when momentum exceeds an entry threshold (the trend is
    strong); a sell signal fires when momentum drops below an exit
    threshold.

    All signals are lagged by one bar to prevent look-ahead bias.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 0.05,
        exit_threshold: float = -0.02,
    ) -> None:
        """
        Args:
            lookback: Number of bars for rate-of-change calculation (default 20).
            entry_threshold: Enter when ROC exceeds this value (default 0.05 = 5%).
            exit_threshold: Exit when ROC drops below this value (default -0.02 = -2%).

        Raises:
            ValueError: If parameters are invalid.
        """
        if lookback < 1:
            raise ValueError("Lookback must be a positive integer")
        if entry_threshold <= exit_threshold:
            raise ValueError("entry_threshold must be greater than exit_threshold")

        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lagged entry/exit signals from momentum.

        Args:
            ohlcv: DataFrame with at least a ``close`` column.

        Returns:
            DataFrame with ``entries`` and ``exits`` boolean columns.
        """
        close = ohlcv["close"]
        roc = close.pct_change(periods=self.lookback)

        entries = roc > self.entry_threshold
        exits = roc < self.exit_threshold

        # Lag signals by one bar
        entries = entries.shift(1).fillna(False).astype(bool)
        exits = exits.shift(1).fillna(False).astype(bool)

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
