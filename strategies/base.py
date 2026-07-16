"""Strategy protocol and shared types."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    """Protocol for alpha strategies that produce lagged entry/exit signals."""

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry and exit signals from OHLCV data.

        Args:
            ohlcv: DataFrame with columns open, high, low, close, volume.

        Returns:
            DataFrame with boolean columns ``entries`` and ``exits``,
            index-aligned to ``ohlcv``. Signals MUST be lagged by at least
            one bar so trades execute after the signal bar closes.
        """
        ...
