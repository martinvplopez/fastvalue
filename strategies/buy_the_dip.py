"""Buy-the-dip rules that hold an asset or sit in cash (see ``strategies/trend_following.py`` for the interface).

- ``DipInUptrend``: Connors' short-term pullback. Buy when RSI(2) is washed out while the close is above its
  200-day average, sell when the close gets back above its 5-day average. Days to weeks per trade. The
  Pullback agent's "pullback inside an uptrend" reading is the same idea measured in ATR units (D-19).
- ``DrawdownDip``: buy when the close is a set share below its high of the last year, sell when it is back
  at that high. The plain "buy the dip" of a long-term investor, as a switch between the asset and cash.

Dip buying with new money (save cash, invest it on a dip, against investing each month at once) is a
different question, answered by the savings study in ``core/systematic.py``.
"""

from __future__ import annotations

import pandas as pd

from analysis.trend_indicators import drawdown_from_high, sma, wilder_rsi
from strategies.base import exposure_signals, latch


class DipInUptrend:
    """Buy a short, sharp dip (RSI below ``entry``) above the ``trend_window``-day average; sell above the
    ``exit_window``-day average."""

    def __init__(self, rsi_period: int = 2, entry: float = 10.0, trend_window: int = 200, exit_window: int = 5) -> None:
        if rsi_period < 1 or trend_window < 2 or exit_window < 1:
            raise ValueError("Periods must be positive (trend window at least 2)")
        if not 0 < entry < 50:
            raise ValueError("entry must be between 0 and 50")
        self.rsi_period, self.entry = int(rsi_period), float(entry)
        self.trend_window, self.exit_window = int(trend_window), int(exit_window)

    @property
    def warmup(self) -> int:
        return self.trend_window

    def exposure(self, frame: pd.DataFrame) -> pd.Series:
        close = frame["close"]
        rsi = wilder_rsi(close, self.rsi_period)
        enter = rsi.lt(self.entry) & close.gt(sma(close, self.trend_window))
        exit_ = close.gt(sma(close, self.exit_window))
        return latch(enter, exit_)

    def overlays(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        close = frame["close"]
        return {f"{self.trend_window}-day average": sma(close, self.trend_window),
                f"{self.exit_window}-day average": sma(close, self.exit_window)}

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return exposure_signals(self.exposure(ohlcv))


class DrawdownDip:
    """Buy when the close is ``dip`` or more below its highest close of the last ``lookback`` days; sell when it
    is back at that high."""

    def __init__(self, dip: float = 0.10, lookback: int = 252) -> None:
        if not 0 < dip < 0.9:
            raise ValueError("dip must be between 0 and 0.9")
        if lookback < 5:
            raise ValueError("lookback must be at least 5 days")
        self.dip, self.lookback = float(dip), int(lookback)

    @property
    def warmup(self) -> int:
        return self.lookback

    def exposure(self, frame: pd.DataFrame) -> pd.Series:
        dd = drawdown_from_high(frame["close"], self.lookback)
        return latch(dd.le(-self.dip), dd.ge(0.0))

    def overlays(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        high = frame["close"].rolling(self.lookback, min_periods=1).max()
        return {f"{self.lookback}-day high": high, f"Buy line (-{self.dip:.0%})": high * (1.0 - self.dip)}

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return exposure_signals(self.exposure(ohlcv))
