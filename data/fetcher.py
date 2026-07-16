"""Download and cache OHLCV market data via yfinance."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import (
    DATA_DIR,
    LOOKBACK_YEARS,
    MAX_FORWARD_FILL_DAYS,
    MIN_BARS_WARNING,
)

logger = logging.getLogger(__name__)

OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


class DataFetchError(Exception):
    """Raised when market data cannot be fetched or validated for a ticker."""


def _default_start_date() -> str:
    """Return ISO start date for the default lookback window."""
    start = datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)
    return start.strftime("%Y-%m-%d")


def _cache_path(ticker: str) -> Path:
    """Return the Parquet cache path for a ticker."""
    return DATA_DIR / f"{ticker.upper()}.parquet"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output to lowercase OHLCV columns.

    Args:
        df: Raw DataFrame from yfinance.

    Returns:
        DataFrame with columns: open, high, low, close, volume.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise DataFetchError(f"Missing required columns: {sorted(missing)}")

    return df[OHLCV_COLUMNS]


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data: drop NaN closes, forward-fill minor gaps, drop remainder.

    Args:
        df: Normalized OHLCV DataFrame.

    Returns:
        Cleaned DataFrame indexed by datetime.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = df.dropna(subset=["close"])
    df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].ffill(limit=MAX_FORWARD_FILL_DAYS)
    df = df.dropna(subset=OHLCV_COLUMNS)

    return df


def _cache_covers_range(df: pd.DataFrame, start: str, end: str | None) -> bool:
    """Return True if cached data covers the requested date range."""
    if df.empty:
        return False

    cache_start = df.index.min()
    cache_end = df.index.max()
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end) if end else pd.Timestamp.today()

    return cache_start <= requested_start and cache_end >= requested_end - pd.Timedelta(days=5)


def _download_ticker(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from yfinance.

    Args:
        ticker: Stock symbol.
        start: Start date (YYYY-MM-DD).
        end: Optional end date (YYYY-MM-DD).

    Returns:
        Cleaned OHLCV DataFrame.

    Raises:
        DataFetchError: If download fails or returns empty data.
    """
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise DataFetchError(f"Failed to download {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        raise DataFetchError(f"No data returned for {ticker}")

    df = _normalize_columns(raw)
    df = _clean_ohlcv(df)

    if df.empty:
        raise DataFetchError(f"No valid OHLCV rows after cleaning for {ticker}")

    return df


def _validate_bar_count(ticker: str, df: pd.DataFrame) -> None:
    """Log a warning if the DataFrame has fewer bars than expected."""
    if len(df) < MIN_BARS_WARNING:
        logger.warning(
            "%s has only %d bars (minimum recommended: %d)",
            ticker,
            len(df),
            MIN_BARS_WARNING,
        )


def fetch_ohlcv(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download or load cached OHLCV data for a list of tickers.

    Args:
        tickers: List of stock symbols.
        start: Start date (YYYY-MM-DD). Defaults to LOOKBACK_YEARS ago.
        end: Optional end date (YYYY-MM-DD).
        force_refresh: If True, re-download even when cache exists.

    Returns:
        Dict mapping ticker symbol to cleaned OHLCV DataFrame.
        Failed tickers are skipped with a logged warning.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start_date = start or _default_start_date()
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        symbol = ticker.upper()
        path = _cache_path(symbol)

        try:
            if not force_refresh and path.exists():
                cached = pd.read_parquet(path)
                cached.index = pd.to_datetime(cached.index)
                if _cache_covers_range(cached, start_date, end):
                    df = cached.loc[start_date:end] if end else cached.loc[start_date:]
                    _validate_bar_count(symbol, df)
                    result[symbol] = df
                    logger.info("Loaded %s from cache (%d bars)", symbol, len(df))
                    continue

            df = _download_ticker(symbol, start_date, end)
            df.to_parquet(path, index=True)
            _validate_bar_count(symbol, df)
            result[symbol] = df
            logger.info("Downloaded and cached %s (%d bars)", symbol, len(df))

        except DataFetchError as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
        except Exception as exc:
            logger.warning("Unexpected error for %s: %s", symbol, exc)

    return result
