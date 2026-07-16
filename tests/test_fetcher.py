"""Tests for data fetcher normalization and cleaning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.fetcher import (
    DataFetchError,
    _cache_covers_range,
    _clean_ohlcv,
    _normalize_columns,
    fetch_ohlcv,
)


def test_normalize_columns_lowercases_and_selects_ohlcv() -> None:
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000_000, 1_100_000],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )
    result = _normalize_columns(raw)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.iloc[0]["close"] == 101.0


def test_normalize_columns_raises_on_missing() -> None:
    raw = pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(DataFetchError, match="Missing required columns"):
        _normalize_columns(raw)


def test_clean_ohlcv_drops_nan_close_and_forward_fills() -> None:
    df = pd.DataFrame(
        {
            "open": [100.0, None, 102.0],
            "high": [101.0, None, 103.0],
            "low": [99.0, None, 101.0],
            "close": [100.5, None, 102.5],
            "volume": [1000, None, 1200],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    result = _clean_ohlcv(df)
    assert len(result) == 2
    assert result.iloc[1]["open"] == 102.0


def test_cache_covers_range() -> None:
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    assert _cache_covers_range(df, "2020-01-01", "2020-01-03")
    assert not _cache_covers_range(df, "2019-01-01", None)


def test_fetch_ohlcv_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("data.fetcher.DATA_DIR", tmp_path)

    index = pd.date_range("2015-01-01", "2015-12-31", freq="B")
    n = len(index)
    cached = pd.DataFrame(
        {
            "open": [400.0 + i * 0.1 for i in range(n)],
            "high": [402.0 + i * 0.1 for i in range(n)],
            "low": [399.0 + i * 0.1 for i in range(n)],
            "close": [401.0 + i * 0.1 for i in range(n)],
            "volume": [50_000_000] * n,
        },
        index=index,
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(tmp_path / "SPY.parquet")

    result = fetch_ohlcv(["SPY"], start="2015-01-01", end="2015-01-10")
    assert "SPY" in result
    assert len(result["SPY"]) == 7
