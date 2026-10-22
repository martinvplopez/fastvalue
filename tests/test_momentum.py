"""Tests for Momentum (Rate of Change) strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.momentum import Momentum


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


def test_momentum_produces_correct_shape() -> None:
    ohlcv = _make_ohlcv([100.0 + i for i in range(50)])
    strategy = Momentum()
    signals = strategy.generate_signals(ohlcv)

    assert set(signals.columns) == {"entries", "exits"}
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool
    assert signals.index.equals(ohlcv.index)


def test_momentum_signals_are_lagged() -> None:
    ohlcv = _make_ohlcv([100.0 + i for i in range(50)])
    strategy = Momentum(lookback=5)
    signals = strategy.generate_signals(ohlcv)

    assert not signals["entries"].iloc[0]
    assert not signals["exits"].iloc[0]


def test_momentum_invalid_lookback() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Momentum(lookback=0)


def test_momentum_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="entry_threshold must be greater"):
        Momentum(entry_threshold=-0.01, exit_threshold=0.05)


def test_momentum_entry_on_strong_uptrend() -> None:
    """A steady uptrend with >5% over 20 days should trigger momentum entry."""
    closes = [100.0] * 20 + [100.0 + i * 1.5 for i in range(1, 21)]
    ohlcv = _make_ohlcv(closes)
    strategy = Momentum(lookback=10, entry_threshold=0.05, exit_threshold=-0.02)
    signals = strategy.generate_signals(ohlcv)

    assert signals["entries"].any()


def test_momentum_exit_on_decline() -> None:
    """A decline should trigger a momentum exit."""
    closes = [100.0 + i * 2 for i in range(20)] + [138.0 - i * 3 for i in range(1, 21)]
    ohlcv = _make_ohlcv(closes)
    strategy = Momentum(lookback=10, entry_threshold=0.05, exit_threshold=-0.02)
    signals = strategy.generate_signals(ohlcv)

    assert signals["exits"].any()
