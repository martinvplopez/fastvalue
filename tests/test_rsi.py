"""Tests for RSI Mean Reversion strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.rsi import RSIMeanReversion


def _make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=index,
    )


def test_rsi_produces_correct_shape() -> None:
    ohlcv = _make_ohlcv([100.0 + i * 0.5 for i in range(100)])
    strategy = RSIMeanReversion()
    signals = strategy.generate_signals(ohlcv)

    assert set(signals.columns) == {"entries", "exits"}
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool
    assert signals.index.equals(ohlcv.index)


def test_rsi_signals_are_lagged() -> None:
    """Entry/exit signals must not appear on the first bar."""
    ohlcv = _make_ohlcv([100.0 + i for i in range(50)])
    strategy = RSIMeanReversion(period=5)
    signals = strategy.generate_signals(ohlcv)

    # First bar should never have a signal (lagged)
    assert not signals["entries"].iloc[0]
    assert not signals["exits"].iloc[0]


def test_rsi_invalid_period() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RSIMeanReversion(period=0)


def test_rsi_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="0 < oversold < overbought < 100"):
        RSIMeanReversion(oversold=70, overbought=30)


def test_rsi_triggers_on_extreme_moves() -> None:
    """After a sharp drop, RSI should drop low enough to trigger an entry."""
    closes = [100.0] * 20 + [100.0 - i * 3 for i in range(1, 21)]
    ohlcv = _make_ohlcv(closes)
    strategy = RSIMeanReversion(period=14, oversold=30, overbought=70)
    signals = strategy.generate_signals(ohlcv)

    # Somewhere after the drop, there should be at least one entry
    assert signals["entries"].any()
