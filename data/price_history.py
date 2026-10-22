"""Long daily price histories with split and dividend data, for point-in-time studies.

``data.fetcher`` serves total-return adjusted OHLCV, which is right for charts
and signals but wrong for valuation on a past date: a 2012 Apple close of $24
(split-adjusted) times the 2012 share count gives a market cap 28 times too
small. This module keeps Yahoo's split-adjusted close, the dividend-adjusted
close for returns, and the split events, so ``as_traded`` can restore the price
and volume that actually printed on each day.

Frames are cached as Parquet under data/cache/history/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

import config.settings as settings

logger = logging.getLogger(__name__)

COLUMNS = ("open", "high", "low", "close", "adj_close", "volume", "dividends", "splits")
CHUNK = 40
MAX_AGE_DAYS = 3


def _cache_dir(folder: str = "history") -> Path:
    path = settings.DATA_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(ticker: str, folder: str = "history") -> Path:
    return _cache_dir(folder) / f"{ticker.upper().replace('/', '_')}.parquet"


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
              "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits"}
    df = frame.rename(columns=rename)
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    for col in ("dividends", "splits"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    return df.sort_index()


def fetch_price_history(tickers: list[str], start: str, max_age_days: float = MAX_AGE_DAYS) -> dict[str, pd.DataFrame]:
    """Daily history per ticker since ``start``. Tickers Yahoo does not know are left out."""
    out: dict[str, pd.DataFrame] = {}
    todo: list[str] = []
    for t in dict.fromkeys(x.upper() for x in tickers):
        path = _path(t)
        if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
            df = pd.read_parquet(path)
            if not df.empty:
                out[t] = df.loc[pd.Timestamp(start):]
                continue
        todo.append(t)

    out.update(_download(todo, "history", start=start))
    return out


def fetch_full_history(tickers: list[str], max_age_days: float = MAX_AGE_DAYS) -> dict[str, pd.DataFrame]:
    """Every day Yahoo has for each ticker (QQQ since 1999, ^NDX since 1985), cached under data/cache/history_max/.

    A separate cache from ``fetch_price_history``, whose files start where their first caller asked.
    """
    out: dict[str, pd.DataFrame] = {}
    todo: list[str] = []
    for t in dict.fromkeys(x.upper() for x in tickers):
        path = _path(t, "history_max")
        if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
            df = pd.read_parquet(path)
            if not df.empty:
                out[t] = df
                continue
        todo.append(t)
    fresh = _download(todo, "history_max", period="max")
    for t in todo:
        if t not in fresh and _path(t, "history_max").exists():  # Yahoo refused: a stale copy beats nothing
            fresh[t] = pd.read_parquet(_path(t, "history_max"))
    out.update(fresh)
    return out


def _download(tickers: list[str], folder: str, **span: str) -> dict[str, pd.DataFrame]:
    """Download in chunks (``span`` is yfinance's ``start`` or ``period``), normalise and cache each ticker."""
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(chunk, auto_adjust=False, actions=True, group_by="ticker",
                                  progress=False, threads=True, **span)
                break
            except Exception as exc:  # noqa: BLE001 - retried, then the chunk is skipped
                logger.warning("Price download failed for %s... (%s), attempt %d", chunk[0], exc, attempt + 1)
                time.sleep(3 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        for t in chunk:
            try:
                frame = raw[t] if len(chunk) > 1 else raw.droplevel(0, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = _normalize(frame)
            if df.empty:
                continue
            df.to_parquet(_path(t, folder))
            out[t] = df
    return out


def as_traded(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV as it printed on each day (undoing later split adjustments).

    Yahoo divides past prices (and multiplies past volume) by every split that
    happened afterwards. ``split_factor`` is the product of those later splits.
    Dollar volume is unchanged by the adjustment.
    """
    splits = df["splits"].where(df["splits"] > 0, 1.0)
    # Product of split ratios strictly after each day.
    after = splits[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out = pd.DataFrame(index=df.index)
    for col in ("open", "high", "low", "close"):
        out[col] = df[col] * after
    out["volume"] = df["volume"] / after
    out["split_factor"] = after
    return out


def split_events(df: pd.DataFrame) -> pd.Series:
    s = df["splits"]
    return s[s > 0].astype(float)
