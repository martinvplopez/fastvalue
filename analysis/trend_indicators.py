"""Trend, momentum and dip indicators for the systematic rules (price only, no I/O).

Every value at day t reads closes up to and including t, never later ones; the rules that use them trade a
bar later (``strategies/trend_following.py``, ``core/systematic.py``).

- Moving average and the PB EMA channel: FFriZz's PlayBit ([PB]) EMA on TradingView is a channel between a
  200-period EMA of the highs and a 200-period EMA of the closes, coloured by where the close sits: above
  both, below both, or inside. Read as a trend filter it has a built-in buffer: the close has to clear the
  upper line to turn up and break the lower one to turn down.
- Absolute (time-series) momentum: the return over the last 12 months without the latest one, minus what
  T-bills paid over the same months (Moskowitz, Ooi and Pedersen 2012; Antonacci's absolute momentum).
- MSCI momentum (MSCI Momentum Indexes Methodology, July 2025, section 2.2): 6- and 12-month price
  performance ending one month before the review, minus the local risk-free rate, each divided by the
  annualised standard deviation of weekly returns over 3 years; across a universe each is z-scored, the two
  are averaged, standardised again, winsorised at +/-3 and mapped to a score of 1 + Z (Z > 0) or 1 / (1 - Z).
- Dips: RSI(2) (Connors' short-term pullback inside a 200-day uptrend) and the fall from the high.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTH_BARS = 21  # trading days in a month, for daily windows measured in months
WEEKS_A_YEAR = 52


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def pb_ema_channel(high: pd.Series, close: pd.Series, length: int = 200) -> pd.DataFrame:
    """The PB EMA channel: EMA of the closes, EMA of the highs, and the state of the close against both.

    ``state`` is +1 above both lines, -1 below both, 0 inside the channel or before the EMAs exist.
    """
    ema_close, ema_high = ema(close, length), ema(high, length)
    upper, lower = np.fmax(ema_close, ema_high), np.fmin(ema_close, ema_high)
    ready = ema_close.notna() & ema_high.notna()
    state = np.where(ready & (close > upper), 1, np.where(ready & (close < lower), -1, 0))
    return pd.DataFrame({"ema_close": ema_close, "ema_high": ema_high, "state": state}, index=close.index)


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualised standard deviation of daily returns over ``window`` days."""
    return close.pct_change().rolling(window, min_periods=window).std() * math.sqrt(TRADING_DAYS)


def drawdown_from_high(close: pd.Series, lookback: int | None = None) -> pd.Series:
    """Close against its highest close over ``lookback`` days (all history when None); 0 at a new high."""
    high = close.cummax() if not lookback else close.rolling(lookback, min_periods=1).max()
    return close / high - 1.0


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return rsi.mask((loss == 0) & gain.notna(), 100.0)  # no down days in the window: 100; warm-up stays NaN


def _window_return(level: pd.Series, months: int, skip: int) -> pd.Series:
    end = level.shift(skip * MONTH_BARS)
    start = level.shift((months + skip) * MONTH_BARS)
    return end / start - 1.0


def excess_momentum(close: pd.Series, rf_daily: pd.Series | None, months: int = 12, skip: int = 1) -> pd.Series:
    """Return over ``months`` months ending ``skip`` months ago, minus T-bills over the same months."""
    ret = _window_return(close, months, skip)
    if rf_daily is None:
        return ret
    bills = (1.0 + rf_daily.reindex(close.index).fillna(0.0)).cumprod()
    return ret - _window_return(bills, months, skip)


def weekly_volatility(close: pd.Series, years: int = 3, min_weeks: int = WEEKS_A_YEAR) -> pd.Series:
    """Annualised standard deviation of weekly returns over ``years`` years, on the daily index.

    A week counts once it has closed (its last trading day), so a mid-week value uses the weeks before it.
    """
    weekly = close.groupby(close.index.to_period("W-FRI")).tail(1)
    vol = weekly.pct_change().rolling(WEEKS_A_YEAR * years, min_periods=min_weeks).std() * math.sqrt(WEEKS_A_YEAR)
    return vol.reindex(close.index).ffill()


def msci_momentum(close: pd.Series, rf_daily: pd.Series | None, vol_years: int = 3) -> pd.DataFrame:
    """MSCI's 6- and 12-month price momentum (latest month left out, minus T-bills) and both over volatility."""
    m6 = excess_momentum(close, rf_daily, 6, 1)
    m12 = excess_momentum(close, rf_daily, 12, 1)
    sigma = weekly_volatility(close, vol_years)
    return pd.DataFrame({"m6": m6, "m12": m12, "sigma": sigma, "ra6": m6 / sigma, "ra12": m12 / sigma},
                        index=close.index)


def _zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd and math.isfinite(sd) and sd > 0 else x * 0.0


def msci_scores(ra6: pd.Series, ra12: pd.Series | None = None) -> pd.DataFrame:
    """MSCI momentum Z-score and score across a universe at one date (one row per asset).

    An asset without a 6-month value is left out; one without a 12-month value uses its 6-month z-score alone.
    ``z`` is the unwinsorised Z (MSCI ranks on it); ``score`` is 1 + Z for Z > 0 and 1 / (1 - Z) below, from
    Z winsorised at +/-3.
    """
    ra6 = ra6.dropna()
    if ra6.empty:
        return pd.DataFrame(columns=["z", "score"])
    z6 = _zscore(ra6)
    if ra12 is not None:
        ra12 = ra12.reindex(ra6.index)
        z12 = _zscore(ra12.dropna()).reindex(ra6.index)
        combined = (0.5 * z6 + 0.5 * z12).where(z12.notna(), z6)
    else:
        combined = z6
    z = _zscore(combined)
    capped = z.clip(-3.0, 3.0)
    score = np.where(capped > 0, 1.0 + capped, 1.0 / (1.0 - capped))
    return pd.DataFrame({"z": z, "score": score}, index=ra6.index)
