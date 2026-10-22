"""Tests for the Volume & Accumulation Agent (no network, synthetic OHLCV only)."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from agents.base import BEARISH, BULLISH, LIVE, NEUTRAL, PARTIAL, UNAVAILABLE, iso_date
from agents.registry import get_agent_classes
from agents.volume_agent import VolumeAgent
from tests.conftest import make_ohlcv

DASHES = ("\u2013", "\u2014")  # en dash, em dash


def _noisy(periods: int = 300, seed: int = 5) -> pd.DataFrame:
    df = make_ohlcv(periods, noise=0.012, seed=seed)
    rng = np.random.default_rng(seed)
    df["volume"] = rng.uniform(300_000, 900_000, periods)
    return df


def _flow_frame(direction: int, periods: int = 200, seed: int = 11) -> pd.DataFrame:
    """Trend with heavy volume on bars that move with the trend and closes near the bar extreme."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.002 * direction, 0.012, periods)
    close = 100.0 * np.cumprod(1.0 + rets)
    up = rets > 0
    with_trend = up if direction > 0 else ~up
    return pd.DataFrame(
        {
            "open": np.concatenate(([100.0], close[:-1])),
            "high": np.where(up, close * 1.002, close * 1.012),
            "low": np.where(up, close * 0.988, close * 0.998),
            "close": close,
            "volume": np.where(with_trend, 1_500_000.0, 500_000.0),
        },
        index=pd.date_range("2024-01-01", periods=periods, freq="B"),
    )


def _with_selling_climax(volume: float = 3_500_000.0) -> pd.DataFrame:
    """Steady downtrend ending in a wide, high-volume down bar that closes in its upper half."""
    df = make_ohlcv(120, start=120.0, end=90.0, volume=1_000_000.0)
    prev_close = float(df["close"].iloc[-2])
    df.iloc[-1, df.columns.get_loc("open")] = prev_close - 1.0
    df.iloc[-1, df.columns.get_loc("high")] = prev_close + 0.25
    df.iloc[-1, df.columns.get_loc("low")] = prev_close - 5.25
    df.iloc[-1, df.columns.get_loc("close")] = prev_close - 0.75
    df.iloc[-1, df.columns.get_loc("volume")] = volume
    return df


def _with_buying_climax() -> pd.DataFrame:
    """Steady uptrend ending in a wide, high-volume up bar at the high that closes in its lower half."""
    df = make_ohlcv(120, start=90.0, end=120.0, volume=1_000_000.0)
    prev_close = float(df["close"].iloc[-2])
    df.iloc[-1, df.columns.get_loc("open")] = prev_close + 0.5
    df.iloc[-1, df.columns.get_loc("high")] = prev_close + 6.0
    df.iloc[-1, df.columns.get_loc("low")] = prev_close - 0.5
    df.iloc[-1, df.columns.get_loc("close")] = prev_close + 1.0
    df.iloc[-1, df.columns.get_loc("volume")] = 3_500_000.0
    return df


def _factors(result) -> dict[str, float]:
    return {b["factor"]: b["points"] for b in result.raw_data["score_breakdown"]}


def test_registered_with_expected_metadata():
    classes = get_agent_classes()
    assert "volume" in classes
    cls = classes["volume"]
    assert cls is VolumeAgent
    assert cls.name == "Volume & Accumulation Agent"
    assert cls.category == "Technical"
    assert cls.score_scale == 45.0
    assert cls.default_weight == 0.15
    assert cls.point_in_time is True


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_score_range_and_output_contract(seed):
    df = _noisy(seed=seed)
    result = VolumeAgent().evaluate("TEST", df)

    assert -100.0 <= result.score <= 100.0
    assert result.signal in (BULLISH, BEARISH, NEUTRAL)
    assert 0.0 < result.confidence <= 90.0
    assert result.data_quality == LIVE
    assert result.data_as_of == iso_date(df.index[-1])
    assert result.agent_key == "volume"

    raw = result.raw_data
    assert sum(b["points"] for b in raw["score_breakdown"]) == pytest.approx(result.score, abs=0.05)
    assert all(isinstance(row["value"], str) for row in raw["metrics_table"])
    assert {lv["kind"] for lv in raw["levels"]} <= {"support", "resistance", "pivot"}
    assert [lv["label"] for lv in raw["levels"]] == ["VWAP 20", "Point of control", "Value area high", "Value area low"]
    assert raw["val"] <= raw["poc"] <= raw["vah"]
    assert raw["links"] == [{"label": "TradingView chart", "url": "https://www.tradingview.com/symbols/TEST/"}]
    assert len(raw["table"]["rows"]) == 3

    texts = [result.summary, *result.key_factors]
    texts += [row["metric"] + row["value"] for row in raw["metrics_table"]]
    texts += [b["factor"] for b in raw["score_breakdown"]]
    assert not any(d in t for t in texts for d in DASHES)


def test_accumulation_scores_higher_than_distribution():
    agent = VolumeAgent()
    accumulation = agent.evaluate("ACC", _flow_frame(+1))
    distribution = agent.evaluate("DIST", _flow_frame(-1))

    assert accumulation.score > 0 > distribution.score
    assert accumulation.score > distribution.score + 30
    assert accumulation.raw_data["cmf_20"] > 0 > distribution.raw_data["cmf_20"]
    assert accumulation.raw_data["up_down_volume_ratio"] > 1 > distribution.raw_data["up_down_volume_ratio"]


def test_selling_climax_detected():
    result = VolumeAgent().evaluate("CAP", _with_selling_climax())
    assert result.raw_data["climax"] == "selling"
    assert _factors(result)["Selling climax"] == 12.0
    assert any("Selling climax" in f for f in result.key_factors)
    # The climax rule, not the heavy-volume down-day rule, interprets the bar.
    assert "Heavy volume down day" not in _factors(result)


def test_no_climax_without_volume_spike():
    result = VolumeAgent().evaluate("CAP", _with_selling_climax(volume=1_200_000.0))
    assert result.raw_data["climax"] is None
    assert "Selling climax" not in _factors(result)


def test_selling_climax_invalidated_by_lower_close():
    df = _with_selling_climax()
    climax_low = float(df["low"].iloc[-1])
    extra = df.iloc[[-1]].copy()
    extra.index = [df.index[-1] + pd.offsets.BDay(1)]
    extra[["open", "high", "low", "close"]] = [climax_low - 0.5, climax_low, climax_low - 2.0, climax_low - 1.5]
    extra["volume"] = 1_000_000.0
    result = VolumeAgent().evaluate("CAP", pd.concat([df, extra]))
    assert result.raw_data["climax"] is None


def test_buying_climax_detected():
    result = VolumeAgent().evaluate("TOP", _with_buying_climax())
    assert result.raw_data["climax"] == "buying"
    assert _factors(result)["Buying climax"] == -12.0


def test_insufficient_bars_unavailable():
    result = VolumeAgent().evaluate("NEW", make_ohlcv(59))
    assert result.data_quality == UNAVAILABLE
    assert result.confidence == 0.0


def test_minimum_history_is_partial():
    result = VolumeAgent().evaluate("NEW", make_ohlcv(60))
    assert result.data_quality == PARTIAL
    assert result.raw_data["rvol_60"] is None
    assert result.raw_data["profile_bars"] == 60


def test_zero_volume_unavailable():
    result = VolumeAgent().evaluate("IDX", make_ohlcv(120, volume=0.0))
    assert result.data_quality == UNAVAILABLE


def test_intraday_partial_uses_previous_bar_for_rvol():
    df = make_ohlcv(120, volume=1_000_000.0)
    df.iloc[-2, df.columns.get_loc("volume")] = 3_000_000.0
    df.iloc[-1, df.columns.get_loc("volume")] = 100_000.0  # today's bar, still forming

    full_day = VolumeAgent().evaluate("LIVE", df)
    assert full_day.raw_data["rvol_20"] < 0.2
    assert full_day.raw_data["volume_bar_date"] == iso_date(df.index[-1])

    df.attrs["intraday_partial"] = True
    intraday = VolumeAgent().evaluate("LIVE", df)
    assert intraday.raw_data["rvol_20"] == pytest.approx(3.0, rel=1e-6)
    assert intraday.raw_data["volume_bar_date"] == iso_date(df.index[-2])
    assert intraday.data_as_of == iso_date(df.index[-1])
    assert any("still forming" in f for f in intraday.key_factors)
    assert _factors(intraday).get("Heavy volume up day") == 6.0


def test_no_look_ahead():
    base = _noisy(300, seed=9)
    altered = base.copy()
    altered.iloc[200:, :4] *= 1.6
    altered.iloc[200:, 4] *= 12.0
    agent = VolumeAgent()
    for n in (80, 150, 200):
        a = agent.evaluate("PIT", base.iloc[:n])
        b = agent.evaluate("PIT", altered.iloc[:n])
        assert asdict(a) == asdict(b)
        assert a.data_as_of == iso_date(base.index[n - 1])


def test_only_recent_window_matters():
    base = _noisy(300, seed=9)
    agent = VolumeAgent()
    assert asdict(agent.evaluate("PIT", base)) == asdict(agent.evaluate("PIT", base.iloc[40:]))


def test_speed():
    df = _noisy(300, seed=3)
    agent = VolumeAgent()
    start = time.perf_counter()
    for _ in range(200):
        agent.evaluate("FAST", df)
    assert time.perf_counter() - start < 2.0
