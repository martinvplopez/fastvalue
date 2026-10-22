"""Small thread-safe in-process TTL cache for network-bound data providers.

Provider functions (options chains, insider filings, fundamentals, macro series,
news) are slow and rate limited, and the same data is requested repeatedly when
scanning a watchlist. Wrap them with ``@ttl_cache(seconds)`` and key them on
hashable arguments only (usually the ticker).
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_ALL_CACHES: list[dict] = []


def ttl_cache(seconds: float, maxsize: int = 512) -> Callable[[F], F]:
    """
    Cache a function's return value per argument tuple for ``seconds``.

    Exceptions are not cached. Call ``fn.cache_clear()`` to reset one cache or
    ``clear_all_caches()`` to reset every cache (useful in tests).
    """

    def decorator(fn: F) -> F:
        store: dict[Any, tuple[float, Any]] = {}
        lock = threading.Lock()
        key_locks: dict[Any, threading.Lock] = {}
        _ALL_CACHES.append(store)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                hit = store.get(key)
                if hit is not None and now - hit[0] < seconds:
                    return hit[1]
                key_lock = key_locks.setdefault(key, threading.Lock())

            # One fetch per key at a time, so parallel watchlist scans share results.
            with key_lock:
                with lock:
                    hit = store.get(key)
                    if hit is not None and time.monotonic() - hit[0] < seconds:
                        return hit[1]
                value = fn(*args, **kwargs)
                with lock:
                    if len(store) >= maxsize:
                        oldest = min(store, key=lambda k: store[k][0])
                        store.pop(oldest, None)
                    store[key] = (time.monotonic(), value)
                return value

        def cache_clear() -> None:
            with lock:
                store.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def clear_all_caches() -> None:
    for store in _ALL_CACHES:
        store.clear()
