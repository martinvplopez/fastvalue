"""Tests for MACD Crossover strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.macd import MACDCrossover


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


def test_macd_produces_correct_shape() -> None:
    ohlcv = _make_ohlcv([100.0 + i * 0.5 for i in range(100)])
    strategy = MACDCrossover()
    signals = strategy.generate_signals(ohlcv)

    assert set(signals.columns) == {"entries", "exits"}
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool
    assert signals.index.equals(ohlcv.index)


def test_macd_signals_are_lagged() -> None:
    ohlcv = _make_ohlcv([100.0 + i for i in range(60)])
    strategy = MACDCrossover(fast_period=5, slow_period=10, signal_period=3)
    signals = strategy.generate_signals(ohlcv)

    assert not signals["entries"].iloc[0]
    assert not signals["exits"].iloc[0]


def test_macd_invalid_periods() -> None:
    with pytest.raises(ValueError, match="fast_period must be less"):
        MACDCrossover(fast_period=26, slow_period=12)


def test_macd_negative_period() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        MACDCrossover(fast_period=-1, slow_period=26, signal_period=9)


def test_macd_crossover_generates_entries() -> None:
    """A strong uptrend should eventually produce a MACD crossover entry."""
    closes = [50.0] * 30 + [50.0 + i * 2 for i in range(1, 31)]
    ohlcv = _make_ohlcv(closes)
    strategy = MACDCrossover(fast_period=5, slow_period=10, signal_period=3)
    signals = strategy.generate_signals(ohlcv)

    assert signals["entries"].any()
