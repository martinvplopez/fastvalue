"""Unit tests for the Macro Regime Agent and data layer."""

import numpy as np
import pandas as pd

from data.macro_data import MacroRegimeData
from agents.regime_agent import MacroRegimeAgent


def _make_macro_data(**overrides) -> MacroRegimeData:
    """Create a MacroRegimeData with sensible defaults, optionally overridden."""
    defaults = dict(
        spy_price=550.0,
        spy_sma_50=540.0,
        spy_sma_200=500.0,
        vix_level=15.0,
        vix_sma_20=16.0,
        vix_percentile=30.0,
        yield_spread_10y_2y=0.80,
        yield_spread_10y_3m=1.20,
        yield_curve_status="NORMAL",
        breadth_ratio=1.02,
        breadth_ratio_sma_50=1.01,
        breadth_trend="IMPROVING",
        credit_ratio=1.005,
        credit_ratio_sma_50=1.002,
        credit_trend="RISK_ON",
        regime="BULL",
        regime_score=6.0,
        data_quality="FULL",
    )
    defaults.update(overrides)
    return MacroRegimeData(**defaults)


def _sample_ohlcv(periods: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    close = np.linspace(100.0, 140.0, periods)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": [500000] * periods,
        },
        index=dates,
    )


def test_regime_agent_bull_market():
    """Bull market macro data should produce bullish signal with high score."""
    agent = MacroRegimeAgent()
    macro = _make_macro_data(regime="BULL", regime_score=6.0)
    assessment = agent.evaluate_with_data(macro)

    assert assessment.agent_name == "Macro Regime Agent"
    assert assessment.signal == "BULLISH"
    assert assessment.score > 0
    assert assessment.confidence >= 80.0
    assert "BULL" in assessment.raw_data["regime"]


def test_regime_agent_bear_market():
    """Bear market macro data should produce bearish signal."""
    agent = MacroRegimeAgent()
    macro = _make_macro_data(
        spy_price=450.0,
        spy_sma_50=480.0,
        spy_sma_200=500.0,
        vix_level=30.0,
        vix_percentile=75.0,
        yield_spread_10y_2y=-0.30,
        yield_curve_status="INVERTED",
        breadth_trend="DETERIORATING",
        credit_trend="RISK_OFF",
        regime="BEAR",
        regime_score=-3.0,
    )
    assessment = agent.evaluate_with_data(macro)

    assert assessment.signal == "BEARISH"
    assert assessment.score < 0
    assert "inverted" in assessment.summary


def test_regime_agent_crisis():
    """Crisis conditions should produce very negative score."""
    agent = MacroRegimeAgent()
    macro = _make_macro_data(
        spy_price=400.0,
        spy_sma_50=480.0,
        spy_sma_200=520.0,
        vix_level=45.0,
        vix_percentile=95.0,
        yield_spread_10y_2y=-0.50,
        yield_curve_status="INVERTED",
        breadth_trend="DETERIORATING",
        credit_trend="RISK_OFF",
        regime="CRISIS",
        regime_score=-7.0,
    )
    assessment = agent.evaluate_with_data(macro)

    assert assessment.signal == "BEARISH"
    assert assessment.score < -50
    assert "CRISIS" in assessment.raw_data["regime"]


def test_regime_agent_correction():
    """Correction conditions: SPY above 200 SMA but VIX elevated."""
    agent = MacroRegimeAgent()
    macro = _make_macro_data(
        spy_price=510.0,
        spy_sma_50=520.0,
        spy_sma_200=500.0,
        vix_level=24.0,
        vix_percentile=60.0,
        yield_spread_10y_2y=0.30,
        yield_curve_status="NORMAL",
        breadth_trend="STABLE",
        credit_trend="NEUTRAL",
        regime="CORRECTION",
        regime_score=2.0,
    )
    assessment = agent.evaluate_with_data(macro)

    assert -100.0 <= assessment.score <= 100.0
    assert assessment.signal in ("BULLISH", "BEARISH", "NEUTRAL")


def test_regime_agent_partial_data_quality():
    """Partial data quality should lower confidence."""
    agent = MacroRegimeAgent()
    macro_full = _make_macro_data(data_quality="FULL", regime_score=5.0)
    macro_partial = _make_macro_data(data_quality="PARTIAL", regime_score=5.0)

    a_full = agent.evaluate_with_data(macro_full)
    a_partial = agent.evaluate_with_data(macro_partial)

    assert a_full.confidence > a_partial.confidence


def test_regime_agent_metrics_table():
    """Raw data should include a metrics_table for UI display."""
    agent = MacroRegimeAgent()
    macro = _make_macro_data()
    assessment = agent.evaluate_with_data(macro)

    assert "metrics_table" in assessment.raw_data
    metrics = assessment.raw_data["metrics_table"]
    assert len(metrics) > 10
    metric_names = [m["metric"] for m in metrics]
    assert "VIX Level" in metric_names
    assert "10Y-2Y Yield Spread (FRED)" in metric_names
    assert "Market Regime" in metric_names


def test_regime_agent_score_range():
    """Score should always be in [-100, 100]."""
    agent = MacroRegimeAgent()

    # Extreme bull
    macro_bull = _make_macro_data(regime_score=10.0)
    a = agent.evaluate_with_data(macro_bull)
    assert -100.0 <= a.score <= 100.0

    # Extreme bear
    macro_bear = _make_macro_data(regime_score=-10.0)
    a = agent.evaluate_with_data(macro_bear)
    assert -100.0 <= a.score <= 100.0
