"""Shared test fixtures.

Tests never touch the network: yfinance and URL reads raise, so any agent that
forgets to handle a provider failure shows up as a test error instead of a
silently different result. On-disk state (history DB, local weights, IV
snapshots, watchlist) is redirected to a temporary directory.
"""

from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd
import pytest
import yfinance

import config.settings as settings
from data.cache_utils import clear_all_caches


class NetworkDisabledError(RuntimeError):
    pass


def _blocked(*_args, **_kwargs):
    raise NetworkDisabledError("Network access is disabled in tests; mock the provider instead.")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yfinance, "Ticker", _blocked)
    monkeypatch.setattr(yfinance, "download", _blocked)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    clear_all_caches()
    yield
    clear_all_caches()


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(settings, "STORE_DIR", store)
    # Exchange rates and quote currencies: a test must never write its offline guesses into the real cache.
    import data.fx as fx

    fx_dir = tmp_path / "fx"
    fx_dir.mkdir()
    monkeypatch.setattr(fx, "_dir", lambda: fx_dir)
    fx._GUESSES.clear()
    # The daily check's trend watch needs live prices; a test that wants it sets its own list and trend_fn.
    monkeypatch.setattr(settings, "TREND_WATCH", [])
    yield store


def make_ohlcv(
    periods: int = 300,
    start: float = 100.0,
    end: float = 140.0,
    volume: float = 500_000.0,
    noise: float = 0.0,
    seed: int = 7,
    freq: str = "B",
) -> pd.DataFrame:
    """Synthetic OHLCV frame (linear drift plus optional noise)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=periods, freq=freq)
    close = np.linspace(start, end, periods)
    if noise:
        close = close * np.cumprod(1 + rng.normal(0, noise, periods))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, volume),
        },
        index=dates,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()
