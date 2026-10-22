"""Tests for Buy and Hold strategy."""

from __future__ import annotations

import pandas as pd

from strategies.buy_and_hold import BuyAndHold


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


def test_buy_and_hold_enters_on_first_bar() -> None:
    ohlcv = _make_ohlcv([100.0, 101.0, 102.0, 103.0])
    strategy = BuyAndHold()
    signals = strategy.generate_signals(ohlcv)

    assert signals["entries"].iloc[0] is True or signals["entries"].iloc[0] == True
    assert not signals["entries"].iloc[1:].any()


def test_buy_and_hold_never_exits() -> None:
    ohlcv = _make_ohlcv([100.0] * 50)
    strategy = BuyAndHold()
    signals = strategy.generate_signals(ohlcv)

    assert not signals["exits"].any()


def test_buy_and_hold_correct_shape() -> None:
    ohlcv = _make_ohlcv([100.0, 110.0, 90.0])
    strategy = BuyAndHold()
    signals = strategy.generate_signals(ohlcv)

    assert set(signals.columns) == {"entries", "exits"}
    assert signals.index.equals(ohlcv.index)
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool


def test_buy_and_hold_empty_dataframe() -> None:
    ohlcv = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    ohlcv.index = pd.DatetimeIndex([])
    strategy = BuyAndHold()
    signals = strategy.generate_signals(ohlcv)

    assert len(signals) == 0
