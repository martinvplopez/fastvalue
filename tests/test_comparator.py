"""Unit tests for the Strategy Comparison Engine."""

import numpy as np
import pandas as pd
from core.comparator import compare_strategies


def test_compare_strategies():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(100))
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [100000] * 100,
        },
        index=dates,
    )

    res = compare_strategies(
        ticker="TEST",
        strategy_names=["sma_crossover", "rsi", "momentum", "multi_factor"],
        ohlcv=ohlcv,
    )

    assert res.ticker == "TEST"
    assert len(res.strategy_results) == 4
    assert len(res.rankings) == 4
    assert res.rankings[0]["rank"] == 1
    assert "total_return" in res.benchmark_stats
