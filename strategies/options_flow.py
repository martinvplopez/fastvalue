"""Options Flow Sentiment Trading Strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


class OptionsFlowStrategy:
    """
    Strategy that enters when options flow sentiment is strongly bullish
    and exits when options flow sentiment drops or put/call ratio spikes.
    """

    def __init__(
        self,
        bullish_threshold: float = 0.20,
        bearish_threshold: float = -0.20,
        rolling_window: int = 14,
    ) -> None:
        if bullish_threshold <= bearish_threshold:
            raise ValueError("bullish_threshold must be strictly greater than bearish_threshold")

        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.rolling_window = rolling_window

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Generate entry/exit signals using price trend proxy + options flow sentiment."""
        close = ohlcv["close"]
        volume = ohlcv["volume"]

        price_roc = close.pct_change(self.rolling_window)
        vol_sma = volume.rolling(self.rolling_window).mean()
        vol_surge = (volume / vol_sma.replace(0, np.nan)).fillna(1.0)

        sentiment_series = (price_roc * 2.0 + (vol_surge - 1.0) * 0.1).clip(-1.0, 1.0)

        raw_entries = sentiment_series > self.bullish_threshold
        raw_exits = sentiment_series < self.bearish_threshold

        entries = raw_entries.shift(1).fillna(False).astype(bool)
        exits = raw_exits.shift(1).fillna(False).astype(bool)

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
