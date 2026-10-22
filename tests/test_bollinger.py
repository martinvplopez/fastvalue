"""Tests for Bollinger Bands strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.bollinger_bands import BollingerBands


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


def test_bollinger_produces_correct_shape() -> None:
    ohlcv = _make_ohlcv([100.0 + i * 0.1 for i in range(50)])
    strategy = BollingerBands()
    signals = strategy.generate_signals(ohlcv)

    assert set(signals.columns) == {"entries", "exits"}
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool
    assert signals.index.equals(ohlcv.index)


def test_bollinger_signals_are_lagged() -> None:
    ohlcv = _make_ohlcv([100.0 + i for i in range(50)])
    strategy = BollingerBands(window=10)
    signals = strategy.generate_signals(ohlcv)

    assert not signals["entries"].iloc[0]
    assert not signals["exits"].iloc[0]


def test_bollinger_invalid_window() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        BollingerBands(window=1)


def test_bollinger_invalid_num_std() -> None:
    with pytest.raises(ValueError, match="num_std must be positive"):
        BollingerBands(num_std=0)


def test_bollinger_entry_on_price_drop() -> None:
    """A sharp price drop below the lower band should trigger an entry."""
    closes = [100.0] * 25 + [80.0]  # Sudden drop
    ohlcv = _make_ohlcv(closes)
    strategy = BollingerBands(window=20, num_std=2.0)
    signals = strategy.generate_signals(ohlcv)

    # The entry is lagged, so it may appear one bar after the drop
    # But we should see at least one entry signal in the data
    # Due to lag the entry appears at bar 26 (if there were one)
    # With only 26 bars the signal from bar 25 appears nowhere (lagged off end)
    # So let's extend the data
    closes_extended = [100.0] * 25 + [80.0, 80.0]
    ohlcv = _make_ohlcv(closes_extended)
    signals = strategy.generate_signals(ohlcv)
    assert signals["entries"].any()
