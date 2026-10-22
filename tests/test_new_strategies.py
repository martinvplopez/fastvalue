"""Unit tests for Options Flow, Insider Buys, and Multi-Factor Strategies."""

import numpy as np
import pandas as pd
from strategies.insider_buys import InsiderBuysStrategy
from strategies.multi_factor import MultiFactorStrategy
from strategies.options_flow import OptionsFlowStrategy


def _sample_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(100))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [100000] * 100,
        },
        index=dates,
    )


def test_options_flow_strategy():
    ohlcv = _sample_ohlcv()
    strat = OptionsFlowStrategy()
    signals = strat.generate_signals(ohlcv)

    assert "entries" in signals.columns
    assert "exits" in signals.columns
    assert len(signals) == len(ohlcv)
    assert signals["entries"].dtype == bool
    assert signals["exits"].dtype == bool


def test_insider_buys_strategy():
    ohlcv = _sample_ohlcv()
    strat = InsiderBuysStrategy()
    signals = strat.generate_signals(ohlcv)

    assert "entries" in signals.columns
    assert "exits" in signals.columns
    assert len(signals) == len(ohlcv)


def test_multi_factor_strategy():
    ohlcv = _sample_ohlcv()
    strat = MultiFactorStrategy(fast_window=10, slow_window=30)
    signals = strat.generate_signals(ohlcv)

    assert "entries" in signals.columns
    assert "exits" in signals.columns
    assert len(signals) == len(ohlcv)
