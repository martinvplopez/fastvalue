"""Trend-following rules that hold an asset or sit in cash: the moving-average filter (and the two-engine rule
built on it), the PB EMA channel, absolute momentum and MSCI-style momentum on one asset.

Each rule has ``exposure(frame)``: 1 or 0 decided with closes up to each day, unlagged, for the daily
engine in ``core/systematic.py``, which trades it a bar later. ``generate_signals(ohlcv)`` turns it into
the lagged entries and exits of the Strategy protocol for the Classic strategy view and the lab.

``frame`` holds ``close`` and ``high`` of the asset the rule reads, and optionally ``rf`` (the daily T-bill
return; momentum is measured above it, 0 when missing), ``vix`` and ``vix3m`` (the two-engine rule's term
structure check; skipped where either is missing).
"""

from __future__ import annotations

import pandas as pd

from analysis.trend_indicators import (
    MONTH_BARS,
    excess_momentum,
    msci_momentum,
    pb_ema_channel,
    realized_vol,
    sma,
)
from strategies.base import exposure_signals, latch, month_end_mask

CHECKS = ("daily", "monthly")


class TrendFilter:
    """
    Hold while the close is above its moving average, cash below it (the 200-day rule: Faber 2007; Gayed and
    Bilello, "Leverage for the Long Run", 2016).

    Optional parts turn it into the two-engine rule: ``buffer`` asks the close to clear the average by that
    share before re-entering (fewer whipsaws), ``vol_cap`` exits while the annualised volatility of the last
    ``vol_window`` days is at or above the cap, and ``vix_term`` exits while the VIX is at or above the
    3-month VIX (an inverted volatility curve). Out when either engine flags trouble, back in only when both
    are calm. ``check="monthly"`` decides on the last trading day of each month only.
    """

    def __init__(self, window: int = 200, buffer: float = 0.0, vol_cap: float = 0.0, vol_window: int = 20,
                 vix_term: bool = False, check: str = "daily") -> None:
        if window < 2 or vol_window < 2:
            raise ValueError("Windows must be at least 2 days")
        if not 0.0 <= buffer < 0.5:
            raise ValueError("buffer must be between 0 and 0.5")
        if vol_cap < 0.0:
            raise ValueError("vol_cap must be 0 (off) or positive")
        if check not in CHECKS:
            raise ValueError(f"check must be one of {', '.join(CHECKS)}")
        self.window, self.buffer, self.vol_cap, self.vol_window = int(window), float(buffer), float(vol_cap), int(vol_window)
        self.vix_term, self.check = bool(vix_term), check

    @property
    def warmup(self) -> int:
        return self.window

    def calm(self, frame: pd.DataFrame) -> pd.Series:
        """The fast engine: True while volatility is under the cap and the VIX curve is not inverted."""
        close = frame["close"]
        ok = pd.Series(True, index=close.index)
        if self.vol_cap > 0:
            ok &= realized_vol(close, self.vol_window).lt(self.vol_cap)
        if self.vix_term and {"vix", "vix3m"} <= set(frame.columns):
            ratio = frame["vix"] / frame["vix3m"]
            ok &= ~ratio.ge(1.0)  # no 3-month VIX yet (Yahoo's history starts in 2006): this check is skipped
        return ok

    def exposure(self, frame: pd.DataFrame) -> pd.Series:
        close = frame["close"]
        avg = sma(close, self.window)
        calm = self.calm(frame)
        enter = close.gt(avg * (1.0 + self.buffer)) & calm
        exit_ = close.lt(avg) | (~calm & avg.notna())
        if self.check == "monthly":
            month_end = month_end_mask(close.index)
            enter, exit_ = enter & month_end, exit_ & month_end
        return latch(enter, exit_)

    def overlays(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        avg = sma(frame["close"], self.window)
        lines = {f"{self.window}-day average": avg}
        if self.buffer > 0:
            lines[f"Re-entry line (+{self.buffer:.0%})"] = avg * (1.0 + self.buffer)
        return lines

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return exposure_signals(self.exposure(ohlcv))


class PBEmaChannel:
    """
    The PB EMA channel (FFriZz's PlayBit EMA on TradingView): a 200-period EMA of the highs and one of the
    closes. In when the close is above both lines, out when it is below both; inside the channel the rule
    keeps what it was doing, so the channel's width is the buffer.
    """

    def __init__(self, length: int = 200) -> None:
        if length < 2:
            raise ValueError("length must be at least 2")
        self.length = int(length)

    @property
    def warmup(self) -> int:
        return self.length

    def exposure(self, frame: pd.DataFrame) -> pd.Series:
        state = pb_ema_channel(frame["high"], frame["close"], self.length)["state"]
        return latch(state.eq(1), state.eq(-1))

    def overlays(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        ch = pb_ema_channel(frame["high"], frame["close"], self.length)
        return {f"EMA {self.length} of highs": ch["ema_high"], f"EMA {self.length} of closes": ch["ema_close"]}

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return exposure_signals(self.exposure(ohlcv))


def _rf(frame: pd.DataFrame) -> pd.Series | None:
    return frame["rf"] if "rf" in frame.columns else None


class AbsoluteMomentum:
    """
    Absolute (time-series) momentum: hold while the return over the last ``months`` months, leaving out the
    latest ``skip`` months, beat T-bills over the same months; cash otherwise. Decided at month-ends by default.
    Without T-bill data (the Classic strategy view) the bar is a return above 0.
    """

    def __init__(self, months: int = 12, skip: int = 1, check: str = "monthly") -> None:
        if months < 1 or skip < 0:
            raise ValueError("months must be at least 1 and skip at least 0")
        if check not in CHECKS:
            raise ValueError(f"check must be one of {', '.join(CHECKS)}")
        self.months, self.skip, self.check = int(months), int(skip), check

    @property
    def warmup(self) -> int:
        return (self.months + self.skip) * MONTH_BARS

    def value(self, frame: pd.DataFrame) -> pd.Series:
        return excess_momentum(frame["close"], _rf(frame), self.months, self.skip)

    def exposure(self, frame: pd.DataFrame) -> pd.Series:
        m = self.value(frame)
        enter, exit_ = m.gt(0), m.le(0)
        if self.check == "monthly":
            month_end = month_end_mask(frame.index)
            enter, exit_ = enter & month_end, exit_ & month_end
        return latch(enter, exit_)

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return exposure_signals(self.exposure(ohlcv))


class MSCIMomentumTrend:
    """
    MSCI's momentum measure on one asset: the 6- and 12-month price momentum (each leaving out the latest
    month, minus T-bills) over 3-year weekly volatility, averaged; hold while it is positive. On one asset
    the volatility scales both terms alike, so the sign is that of the two excess returns added up; the
    z-scores and the score weights only exist across a universe (the rotation study in ``core/systematic.py``).
    """

    def __init__(self, check: str = "monthly") -> None:
        if check not in CHECKS:
            raise ValueError(f"check must be one of {', '.join(CHECKS)}")
        self.check = check

    @property
    def warmup(self) -> int:
        return 13 * MONTH_BARS

    def value(self, frame: pd.DataFrame) -> pd.Series:
        m = msci_momentum(frame["close"], _rf(frame))
        return ((m["ra6"] + m["ra12"]) / 2.0).where(m["ra12"].notna(), m["ra6"])

    def exposure(self, frame: pd.DataFrame) -> pd.Series:
        v = self.value(frame)
        enter, exit_ = v.gt(0), v.le(0)
        if self.check == "monthly":
            month_end = month_end_mask(frame.index)
            enter, exit_ = enter & month_end, exit_ & month_end
        return latch(enter, exit_)

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return exposure_signals(self.exposure(ohlcv))
