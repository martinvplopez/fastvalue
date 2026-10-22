"""Bollinger Bands mean reversion strategy."""

from __future__ import annotations

import pandas as pd


class BollingerBands:
    """
    Long-only Bollinger Bands mean reversion strategy.

    Bollinger Bands consist of a simple moving average with upper and lower
    bands placed a configurable number of standard deviations away.  When
    the price touches the lower band the asset is considered "cheap"
    (entry signal); when it touches the upper band it is considered
    "expensive" (exit signal).

    All signals are lagged by one bar to prevent look-ahead bias.
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
    ) -> None:
        """
        Args:
            window: SMA lookback window for the middle band (default 20).
            num_std: Number of standard deviations for the bands (default 2.0).

        Raises:
            ValueError: If parameters are invalid.
        """
        if window < 2:
            raise ValueError("Bollinger window must be >= 2")
        if num_std <= 0:
            raise ValueError("num_std must be positive")

        self.window = window
        self.num_std = num_std

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lagged entry/exit signals from Bollinger Bands.

        Args:
            ohlcv: DataFrame with at least a ``close`` column.

        Returns:
            DataFrame with ``entries`` and ``exits`` boolean columns.
        """
        close = ohlcv["close"]

        middle = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper = middle + self.num_std * std
        lower = middle - self.num_std * std

        entries = close <= lower
        exits = close >= upper

        # Lag signals by one bar
        entries = entries.shift(1).fillna(False).astype(bool)
        exits = exits.shift(1).fillna(False).astype(bool)

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
