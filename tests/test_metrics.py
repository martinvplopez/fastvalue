"""Tests for performance metrics helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.metrics import compute_buy_and_hold_returns, compute_metrics


def test_compute_buy_and_hold_returns_matches_price_move() -> None:
    ohlcv = pd.DataFrame(
        {"close": [100.0, 110.0, 99.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    returns = compute_buy_and_hold_returns(ohlcv)

    assert returns.iloc[0] == 0.0
    assert returns.iloc[1] == pytest.approx(0.1)
    assert returns.iloc[2] == pytest.approx(-0.1)


def test_compute_metrics_from_buy_and_hold() -> None:
    ohlcv = pd.DataFrame(
        {"close": [100.0, 110.0, 121.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    returns = compute_buy_and_hold_returns(ohlcv)
    stats = compute_metrics(returns)

    assert stats["total_return"] == pytest.approx(0.21)
