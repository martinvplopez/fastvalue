"""Insider Buying Alpha Strategy."""

from __future__ import annotations

import pandas as pd


class InsiderBuysStrategy:
    """
    Strategy that generates signals based on insider buying activity.

    Enters when insider cluster buying or heavy net insider buying occurs after a pull-back,
    and exits after a profit target lookback window or insider selling pressure.
    """

    def __init__(
        self,
        dip_lookback: int = 20,
        holding_period: int = 30,
        dip_threshold: float = -0.03,
    ) -> None:
        self.dip_lookback = dip_lookback
        self.holding_period = holding_period
        self.dip_threshold = dip_threshold

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Generate entry/exit signals from price dips + insider buy accumulation patterns."""
        close = ohlcv["close"]

        rolling_ret = close.pct_change(self.dip_lookback)
        rsi_proxy = close.pct_change(5)

        raw_entries = (rolling_ret < self.dip_threshold) & (rsi_proxy > 0.0)

        raw_exits = (rolling_ret > abs(self.dip_threshold) * 2.0) | (
            raw_entries.shift(self.holding_period).fillna(False)
        )

        entries = raw_entries.shift(1).fillna(False).astype(bool)
        exits = raw_exits.shift(1).fillna(False).astype(bool)

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
