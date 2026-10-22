"""RSI mean reversion strategy."""

from __future__ import annotations

import pandas as pd


class RSIMeanReversion:
    """
    Long-only RSI mean reversion strategy.

    Enters when the Relative Strength Index (RSI) drops below an oversold
    threshold and exits when it rises above an overbought threshold. RSI
    measures the speed and magnitude of recent price changes on a 0–100
    scale; extreme readings suggest a reversal may be imminent.

    All signals are lagged by one bar to prevent look-ahead bias.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        """
        Args:
            period: RSI lookback window (default 14).
            oversold: Buy when RSI drops below this level (default 30).
            overbought: Sell when RSI rises above this level (default 70).

        Raises:
            ValueError: If parameters are invalid.
        """
        if period < 1:
            raise ValueError("RSI period must be a positive integer")
        if not (0 < oversold < overbought < 100):
            raise ValueError(
                "Thresholds must satisfy 0 < oversold < overbought < 100"
            )

        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        """Compute Wilder-smoothed RSI."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lagged entry/exit signals from RSI.

        Args:
            ohlcv: DataFrame with at least a ``close`` column.

        Returns:
            DataFrame with ``entries`` and ``exits`` boolean columns.
        """
        rsi = self._compute_rsi(ohlcv["close"])

        entries = rsi < self.oversold
        exits = rsi > self.overbought

        # Lag signals by one bar
        entries = entries.shift(1).fillna(False).astype(bool)
        exits = exits.shift(1).fillna(False).astype(bool)

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
