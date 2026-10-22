"""Download and cache OHLCV market data via yfinance.

``fetch_ohlcv`` serves long histories from a Parquet cache (backtests).
``fetch_live_ohlcv`` additionally refreshes the most recent month so analysis
always runs on the latest session, including today's still-forming bar while
the market is open (flagged via ``df.attrs["intraday_partial"]``).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config.settings import (
    DATA_DIR,
    LOOKBACK_YEARS,
    MAX_FORWARD_FILL_DAYS,
    MIN_BARS_WARNING,
)
from data.cache_utils import _ALL_CACHES
from data.exchange_calendar import NYSE, exchange_for

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


# (symbol, start) -> the day a full-range download last succeeded. On days when the 10-year start falls on a weekend
# or holiday, and for listings younger than 10 years, the cache file can never reach back to the start, so every
# fetch re-downloaded the whole range; one full download per ticker and day gives the same bars.
_FULL_DOWNLOADS: dict[tuple[str, str], date] = {}
_ALL_CACHES.append(_FULL_DOWNLOADS)  # cleared with the other caches (tests)
DOWNLOAD_WORKERS = 4


def _fetch_one(symbol: str, start_date: str, end: str | None, force_refresh: bool) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    try:
        cached = None
        if path.exists():
            cached = pd.read_parquet(path)
            cached.index = pd.to_datetime(cached.index)
        if not force_refresh and cached is not None and _cache_covers_range(cached, start_date, end):
            df = cached.loc[start_date:end] if end else cached.loc[start_date:]
            _validate_bar_count(symbol, df)
            logger.info("Loaded %s from cache (%d bars)", symbol, len(df))
            return df
        if (not force_refresh and end is None and cached is not None and not cached.empty
                and cached.index.max() >= pd.Timestamp.today() - pd.Timedelta(days=5)
                and _FULL_DOWNLOADS.get((symbol, start_date)) == date.today()):
            df = cached.loc[start_date:]  # only the start is uncovered, and the whole range came down today
            _validate_bar_count(symbol, df)
            logger.info("Loaded %s from cache, fully downloaded today (%d bars)", symbol, len(df))
            return df

        # The cache only grows: when it already reaches back far enough, fetch from a week before its last
        # bar; either way merge (new bars win), so a short request can never replace years of history.
        fetch_from = start_date
        if (not force_refresh and cached is not None and not cached.empty
                and pd.Timestamp(start_date) >= cached.index.min()):
            fetch_from = (cached.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        new = _download_ticker(symbol, fetch_from, end)
        if fetch_from == start_date and end is None:
            _FULL_DOWNLOADS[(symbol, start_date)] = date.today()
        if cached is not None and not cached.empty:
            new = pd.concat([cached[~cached.index.isin(new.index)], new]).sort_index()
        new.to_parquet(path, index=True)
        df = new.loc[start_date:end] if end else new.loc[start_date:]
        _validate_bar_count(symbol, df)
        logger.info("Downloaded and cached %s (%d bars)", symbol, len(df))
        return df

    except DataFetchError as exc:
        logger.warning("Skipping %s: %s", symbol, exc)
    except Exception as exc:
        logger.warning("Unexpected error for %s: %s", symbol, exc)
    return None


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
        Dict mapping ticker symbol to cleaned OHLCV DataFrame, in the order asked.
        Failed tickers are skipped with a logged warning.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start_date = start or _default_start_date()
    symbols = list(dict.fromkeys(t.upper() for t in tickers))
    if len(symbols) > 1:
        # Each ticker has its own file and download (yfinance keeps per-call state), so misses run side by side
        # instead of one after another; results keep the order asked.
        with ThreadPoolExecutor(max_workers=min(DOWNLOAD_WORKERS, len(symbols)), thread_name_prefix="ohlcv") as pool:
            frames = list(pool.map(lambda s: _fetch_one(s, start_date, end, force_refresh), symbols))
    else:
        frames = [_fetch_one(s, start_date, end, force_refresh) for s in symbols]
    return {s: df for s, df in zip(symbols, frames) if df is not None}


# ---------------------------------------------------------------- live data

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

LIVE = "LIVE"  # Intraday bar from the session in progress.
CLOSE = "CLOSE"  # Up to date with the last completed session.
STALE = "STALE"  # Missing one or more completed sessions.


@dataclass
class DataFreshness:
    """How current the data behind an analysis is."""

    ticker: str
    as_of: str  # Date of the last bar
    last_price: float
    market_open: bool
    is_intraday: bool  # Last bar is today's still-forming session
    sessions_behind: int  # Completed sessions of the listing's exchange missing from the data
    status: str  # LIVE, CLOSE or STALE
    fetched_at: str
    note: str
    exchange: str = NYSE.name

    def to_dict(self) -> dict:
        return asdict(self)


def _now_et(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(MARKET_TZ)
    return now.astimezone(MARKET_TZ) if now.tzinfo else now.replace(tzinfo=MARKET_TZ)


def market_is_open(now: datetime | None = None) -> bool:
    """Regular NYSE hours on a NYSE session day."""
    return NYSE.is_open(_now_et(now))


def last_completed_session(now: datetime | None = None) -> date:
    """Most recent NYSE session whose regular hours have ended."""
    return NYSE.last_completed_session(_now_et(now))


def is_forming(ticker: str, df: pd.DataFrame, now: datetime | None = None) -> bool:
    """True when the last bar is today's session of the listing's exchange and that session is still open."""
    ex = exchange_for(ticker)
    local = ex.local(_now_et(now))
    return ex.is_open(local) and pd.Timestamp(df.index[-1]).date() == local.date()


def assess_freshness(ticker: str, df: pd.DataFrame, now: datetime | None = None) -> DataFreshness:
    ex = exchange_for(ticker)
    et = _now_et(now)
    local = ex.local(et)
    last_bar = pd.Timestamp(df.index[-1]).date()
    is_open = ex.is_open(local)
    is_intraday = is_open and last_bar == local.date()
    behind = 0 if is_intraday else ex.sessions_between(last_bar, ex.last_completed_session(local))

    if is_intraday:
        status, note = LIVE, "Intraday price from the session in progress (quotes may be delayed)."
    elif behind == 0:
        status, note = CLOSE, "Up to date with the last completed session."
    else:
        status = STALE
        note = (f"Missing {behind} {'session' if behind == 1 else 'sessions'} of {ex.name}; the provider may be lagging."
                if ex.holidays_modeled else
                f"Missing {behind} weekday {'session' if behind == 1 else 'sessions'}; the provider may be lagging or it was a holiday "
                f"(holidays are not modeled for {ex.name}).")

    return DataFreshness(
        ticker=ticker,
        as_of=last_bar.isoformat(),
        last_price=round(float(df["close"].iloc[-1]), 4),
        market_open=is_open,
        is_intraday=is_intraday,
        sessions_behind=behind,
        status=status,
        fetched_at=et.isoformat(timespec="seconds"),
        note=note,
        exchange=ex.name,
    )


def _download_recent(tickers: list[str], period: str = "1mo") -> dict[str, pd.DataFrame]:
    """Download recent daily bars for several tickers in one request."""
    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = raw.columns.get_level_values(0)
                frame = raw[t] if t in level0 else raw.xs(t, axis=1, level=1)
            else:
                frame = raw
            frame = _clean_ohlcv(_normalize_columns(frame))
            if not frame.empty:
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                out[t] = frame
        except (KeyError, DataFetchError) as exc:
            logger.warning("No recent bars for %s: %s", t, exc)
    return out


def fetch_live_ohlcv(
    tickers: list[str],
    now: datetime | None = None,
) -> dict[str, tuple[pd.DataFrame, DataFreshness]]:
    """
    Load history from cache and refresh the latest month from the provider.

    Completed sessions are written back to the Parquet cache; today's partial
    bar (market open) is returned but never cached.

    Returns:
        ticker -> (OHLCV DataFrame, DataFreshness). Tickers with no data are omitted.
    """
    symbols = [t.upper() for t in tickers]
    history = fetch_ohlcv(symbols)
    try:
        recent = _download_recent(symbols)
    except Exception as exc:  # noqa: BLE001 - fall back to cached history, flagged as stale
        logger.warning("Recent-bar refresh failed for %s: %s", symbols, exc)
        recent = {}

    et = _now_et(now)
    out: dict[str, tuple[pd.DataFrame, DataFreshness]] = {}
    for symbol in symbols:
        hist = history.get(symbol)
        new = recent.get(symbol)
        if hist is None and new is None:
            continue
        if hist is None:
            merged = new
        elif new is None:
            merged = hist
        else:
            merged = pd.concat([hist[~hist.index.isin(new.index)], new]).sort_index()

        # The listing's own exchange decides: a Xetra bar fetched before New York opens is still forming.
        partial = is_forming(symbol, merged, et)
        completed = merged.iloc[:-1] if partial else merged
        if new is not None and not completed.empty:
            try:
                completed.to_parquet(_cache_path(symbol), index=True)
            except OSError as exc:
                logger.warning("Could not update cache for %s: %s", symbol, exc)

        merged = merged.copy()
        merged.attrs["intraday_partial"] = bool(partial)
        out[symbol] = (merged, assess_freshness(symbol, merged, et))
    return out
