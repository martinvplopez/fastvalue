"""Strategy protocol and shared types."""

from __future__ import annotations

from typing import Protocol

import numpy as np
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


def month_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of each month in ``index``.

    The final bar counts only when the next business day falls in a new month, so a series that ends
    mid-month does not take a month-end decision on its last day.
    """
    if not len(index):
        return pd.Series([], index=index, dtype=bool)
    months = np.asarray(index.to_period("M"))
    following = np.append(months[1:], (index[-1] + pd.offsets.BDay(1)).to_period("M"))
    return pd.Series(months != following, index=index)


def latch(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Exposure that switches to 1 on ``enter`` and to 0 on ``exit_`` and otherwise keeps its state.

    Starts in cash; an exit on the same bar as an entry wins. Missing values in either input count as False.
    """
    enter = enter.fillna(False).astype(bool)
    exit_ = exit_.fillna(False).astype(bool)
    state = pd.Series(np.where(exit_, 0.0, np.where(enter, 1.0, np.nan)), index=enter.index)
    return state.ffill().fillna(0.0)


def exposure_signals(exposure: pd.Series) -> pd.DataFrame:
    """Entry and exit signals from an exposure decided at each close, lagged one bar (trade the next bar)."""
    step = exposure.diff().fillna(exposure.iloc[0] if len(exposure) else 0.0)
    entries = (step > 0).shift(1, fill_value=False).astype(bool)
    exits = (step < 0).shift(1, fill_value=False).astype(bool)
    return pd.DataFrame({"entries": entries, "exits": exits}, index=exposure.index)
