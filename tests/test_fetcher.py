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


def test_a_short_request_never_shrinks_a_stale_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("data.fetcher.DATA_DIR", tmp_path)
    old = pd.date_range("2016-01-01", "2026-08-31", freq="B")
    frame = lambda idx: pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1e6}, index=idx)  # noqa: E731
    frame(old).to_parquet(tmp_path / "XYZ.parquet")
    asked = {}

    def download(symbol, start, end):
        asked["start"] = start
        return frame(pd.date_range(start, "2026-10-02", freq="B"))

    monkeypatch.setattr("data.fetcher._download_ticker", download)
    out = fetch_ohlcv(["XYZ"], start="2026-09-15")
    assert asked["start"] == "2026-08-24"  # from a week before the cached end, not just the requested window
    assert out["XYZ"].index.min() == pd.Timestamp("2026-09-15")
    kept = pd.read_parquet(tmp_path / "XYZ.parquet")
    assert kept.index.min() == pd.Timestamp("2016-01-01") and kept.index.max() == pd.Timestamp("2026-10-02")
    assert not kept.index.duplicated().any()


def _bars(idx: pd.DatetimeIndex, close: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx)


def test_an_uncoverable_start_downloads_the_full_range_once_a_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The asked start is a Sunday; the first session is the Monday after, so no cache file can ever "cover" it.
    monkeypatch.setattr("data.fetcher.DATA_DIR", tmp_path)
    today = pd.Timestamp.today().normalize()
    start = (today - pd.Timedelta(days=400) - pd.Timedelta(days=(today - pd.Timedelta(days=400)).weekday() + 1))
    assert start.weekday() == 6
    calls = []

    def download(symbol, frm, end):
        calls.append(frm)
        return _bars(pd.bdate_range(frm, today))

    monkeypatch.setattr("data.fetcher._download_ticker", download)
    first = fetch_ohlcv(["XYZ"], start=start.strftime("%Y-%m-%d"))
    second = fetch_ohlcv(["XYZ"], start=start.strftime("%Y-%m-%d"))
    assert calls == [start.strftime("%Y-%m-%d")]  # today: one full download per call
    assert first["XYZ"].equals(second["XYZ"]) and first["XYZ"].index.min() > start
    fetch_ohlcv(["XYZ"], start=start.strftime("%Y-%m-%d"), force_refresh=True)
    assert len(calls) == 2  # a forced refresh still downloads


def test_several_tickers_download_side_by_side_and_keep_their_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("data.fetcher.DATA_DIR", tmp_path)
    today = pd.Timestamp.today().normalize()

    def download(symbol, frm, end):
        if symbol == "BAD":
            raise DataFetchError("no data")
        return _bars(pd.bdate_range(frm, today), close=float(len(symbol)))

    monkeypatch.setattr("data.fetcher._download_ticker", download)
    out = fetch_ohlcv(["ccc", "BAD", "aa", "B", "aa"], start=(today - pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
    assert list(out) == ["CCC", "AA", "B"]
    assert [float(out[t]["close"].iloc[-1]) for t in out] == [3.0, 2.0, 1.0]
