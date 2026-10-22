"""Price and volume confluence: trend, volume surge and dip recovery, from OHLCV only.

It is not an aggregate of the agents and reads no options or insider data: the
"options flow" and "insider" factors below are price-and-volume proxies named
after the ideas they imitate. The agents are backtested in the Backtest tab.
"""

from __future__ import annotations

import pandas as pd


class MultiFactorStrategy:
    """
    Enters when at least ``min_factor_confluence`` of three price-and-volume factors agree:
    1. Trend: fast SMA above slow SMA and rate of change above a threshold
    2. Volume surge on an up day (a proxy named after options flow; no options data)
    3. Dip recovery: price back above the fast SMA after a dip (a proxy named after insider buying; no Form 4 data)
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        momentum_threshold: float = 0.02,
        min_factor_confluence: int = 2,
    ) -> None:
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.momentum_threshold = momentum_threshold
        self.min_factor_confluence = min_factor_confluence

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Generate entry and exit signals based on factor confluence."""
        close = ohlcv["close"]
        volume = ohlcv["volume"]

        fast_sma = close.rolling(self.fast_window).mean()
        slow_sma = close.rolling(self.slow_window).mean()
        roc = close.pct_change(self.fast_window)
        momentum_factor = (fast_sma > slow_sma) & (roc > self.momentum_threshold)

        vol_sma = volume.rolling(self.fast_window).mean()
        vol_surge = volume > (vol_sma * 1.2)
        price_up = close.diff() > 0
        options_flow_factor = vol_surge & price_up

        dip = close < fast_sma
        insider_factor = dip.shift(3).fillna(False) & (close > fast_sma)

        factor_score = (
            momentum_factor.astype(int)
            + options_flow_factor.astype(int)
            + insider_factor.astype(int)
        )

        raw_entries = factor_score >= self.min_factor_confluence
        raw_exits = (fast_sma < slow_sma) | (factor_score == 0)

        entries = raw_entries.shift(1).fillna(False).astype(bool)
        exits = raw_exits.shift(1).fillna(False).astype(bool)

        return pd.DataFrame({"entries": entries, "exits": exits}, index=ohlcv.index)
