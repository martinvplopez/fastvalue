"""Shared, rate-friendly access to Yahoo Finance endpoints.

Many agents hit Yahoo at the same time (7 agents per ticker, several tickers
per watchlist scan). Unthrottled, that produces intermittent 401 "Invalid
Crumb" and 429 errors, which silently drop agents from the composite and move
scores for reasons that have nothing to do with the market. Route every
crumb-authenticated call (quoteSummary, options, insider, news) through
``yahoo_call``:

- the session cookie/crumb is fetched once, under a lock, before parallel use;
- at most ``MAX_CONCURRENT`` requests run at once;
- transient failures are retried with a short backoff;
- a refusal yfinance only logs is raised: with its default ``hide_exceptions``,
  yfinance logs "401 Invalid Crumb" or "429" and returns an empty result, which
  the callers' caches would otherwise keep for hours as "no insider trades".
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import yfinance as yf

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_CONCURRENT = 4
RETRIES = 2
BACKOFF_SECONDS = (0.6, 1.8)
WARM_TTL_SECONDS = 1800

_TRANSIENT_MARKERS = ("401", "crumb", "429", "too many requests", "timed out", "timeout", "connection", "reset by peer", "502", "503")
# yfinance ERROR records that mean "Yahoo refused the request", not "Yahoo has no such data" (a 404 is the latter).
_REFUSAL_MARKERS = ("http error 401", "invalid crumb", "unauthorized", "http error 429", "too many requests", "rate limit")

_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT)
_warm_lock = threading.Lock()
_warmed_at = 0.0


class YahooRefusedError(RuntimeError):
    """Yahoo refused the request; yfinance logged it and returned an empty result instead of raising."""


class _ThreadErrors(logging.Handler):
    """The yfinance ERROR records one thread emits during one call (other threads' calls run in parallel)."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.thread = threading.get_ident()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread:
            self.messages.append(record.getMessage())


def is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _run_watched(fn: Callable[[], T], label: str) -> T:
    """Run ``fn``; raise ``YahooRefusedError`` if yfinance logged a refusal for it in this thread."""
    handler = _ThreadErrors()
    yf_logger = logging.getLogger("yfinance")
    yf_logger.addHandler(handler)
    try:
        result = fn()
    finally:
        yf_logger.removeHandler(handler)
    refusal = next((m for m in handler.messages if any(k in m.lower() for k in _REFUSAL_MARKERS)), None)
    if refusal is not None:
        raise YahooRefusedError(f"{label}: {' '.join(refusal.split())[:200]}")
    return result


def warm_session(force: bool = False) -> None:
    """Fetch Yahoo's cookie and crumb once so parallel requests share them."""
    global _warmed_at
    if not force and time.monotonic() - _warmed_at < WARM_TTL_SECONDS:
        return
    with _warm_lock:
        if not force and time.monotonic() - _warmed_at < WARM_TTL_SECONDS:
            return
        try:
            # Lightweight crumb-authenticated request; the crumb is cached inside yfinance.
            _ = yf.Ticker("SPY").options
            _warmed_at = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            logger.warning("Yahoo session warm-up failed: %s", exc)


def yahoo_call(fn: Callable[[], T], label: str = "yahoo") -> T:
    """Run ``fn`` with the shared session warm, throttled, and retried on transient errors."""
    warm_session()
    attempt = 0
    while True:
        try:
            with _semaphore:
                return _run_watched(fn, label)
        except Exception as exc:
            if attempt >= RETRIES or not is_transient(exc):
                raise
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            logger.info("%s: transient Yahoo error (%s); retrying in %.1fs", label, exc, delay)
            time.sleep(delay)
            if "crumb" in str(exc).lower() or "401" in str(exc):
                warm_session(force=True)
            attempt += 1
