"""Buy and hold (passive) strategy — the baseline benchmark."""

from __future__ import annotations

import pandas as pd


class BuyAndHold:
    """
    Passive buy-and-hold strategy.

    Enters a long position on the first available bar and never exits.
    This is the simplest strategy and serves as the benchmark that any
    active strategy must beat to justify its complexity and transaction costs.
    """

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals: enter on bar 0, never exit.

        Args:
            ohlcv: DataFrame with at least a ``close`` column.

        Returns:
            DataFrame with ``entries`` and ``exits`` boolean columns.
        """
        entries = pd.Series(False, index=ohlcv.index)
        exits = pd.Series(False, index=ohlcv.index)

        if not ohlcv.empty:
            entries.iloc[0] = True

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
