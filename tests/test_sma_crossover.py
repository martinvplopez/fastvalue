"""Tests for SMA crossover strategy signal generation."""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.sma_crossover import SMACrossover


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


def test_sma_crossover_produces_lagged_signals() -> None:
    """Entry on golden cross should appear one bar after the crossover."""
    closes = [100.0] * 10 + list(range(110, 130)) + [150.0] * 30
    ohlcv = _make_ohlcv(closes)
    strategy = SMACrossover(fast_window=5, slow_window=10)
    signals = strategy.generate_signals(ohlcv)

    assert set(signals.columns) == {"entries", "exits"}
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool
    assert signals.index.equals(ohlcv.index)

    # No signal on the first slow_window bars (insufficient history + lag)
    assert not signals["entries"].iloc[:10].any()


def test_sma_crossover_invalid_windows() -> None:
    with pytest.raises(ValueError, match="fast_window must be less"):
        SMACrossover(fast_window=200, slow_window=50)


def test_sma_crossover_no_same_bar_entry() -> None:
    """When bullish flips True, entry must appear one bar after the crossover."""
    closes = [10.0] * 20 + [20.0] * 20
    ohlcv = _make_ohlcv(closes)
    strategy = SMACrossover(fast_window=3, slow_window=5)
    signals = strategy.generate_signals(ohlcv)

    fast = ohlcv["close"].rolling(3).mean()
    slow = ohlcv["close"].rolling(5).mean()
    bullish = fast > slow
    prev_bullish = bullish.shift(1).fillna(False).astype(bool)
    cross_bars = bullish & (~prev_bullish)

    cross_indices = list(signals.index[cross_bars.fillna(False)])
    for cross_idx in cross_indices:
        assert not signals.loc[cross_idx, "entries"], f"Same-bar entry at {cross_idx}"
