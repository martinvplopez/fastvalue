"""Score history store (SQLite) and watchlist persistence.

Every analysis is recorded so the UI can chart how a ticker's composite score
and decision evolved, and so a watchlist scan can flag decision changes
(e.g. HOLD -> BUY) since the previous run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import config.settings as settings

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    data_as_of TEXT,
    price REAL,
    decision TEXT NOT NULL,
    composite_score REAL NOT NULL,
    confidence REAL NOT NULL,
    regime TEXT,
    agent_scores TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_ticker_time ON analyses (ticker, generated_at);
"""

# Columns added after the first schema (owner score and timing panel); added in place when missing.
_EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("owner_score", "REAL"),
    ("timing_score", "REAL"),
    ("timing_label", "TEXT"),
)

DEFAULT_WATCHLIST: list[str] = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


def _db_path() -> Path:
    return settings.STORE_DIR / "history.db"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
        for column, kind in _EXTRA_COLUMNS:
            if column not in existing:
                conn.execute(f"ALTER TABLE analyses ADD COLUMN {column} {kind}")
        yield conn
        conn.commit()
    finally:
        conn.close()


_INSERT = (
    "INSERT INTO analyses (ticker, generated_at, data_as_of, price, decision, composite_score,"
    " confidence, regime, agent_scores, owner_score, timing_score, timing_label)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _values(rec: Any) -> tuple:
    agent_scores = {
        c["agent_key"]: {
            "score": c["raw_score"],
            "normalized": c["normalized_score"],
            "confidence": c["confidence"],
            "quality": c["data_quality"],
            "included": c["included"],
        }
        for c in rec.contributions
    }
    freshness = rec.freshness or {}
    owner = getattr(rec, "owner", None) or {}
    timing = getattr(rec, "timing", None) or {}
    return (
        rec.ticker,
        rec.generated_at,
        freshness.get("as_of"),
        rec.suggested_entry,
        rec.decision,
        rec.composite_score,
        rec.overall_confidence,
        rec.market_regime,
        json.dumps(agent_scores),
        owner.get("score", rec.composite_score),
        timing.get("score"),
        timing.get("label"),
    )


def record_analysis(rec: Any) -> None:
    """Persist a TradeRecommendation (only scores, no raw provider data)."""
    record_analyses([rec])


def record_analyses(recs: list[Any]) -> None:
    """Persist several recommendations in one transaction (a scan or a review: one connection, not one per ticker)."""
    rows = [_values(rec) for rec in recs]
    if not rows:
        return
    with _LOCK, _connect() as conn:
        conn.executemany(_INSERT, rows)


def get_history(ticker: str, limit: int = 250) -> list[dict[str, Any]]:
    """Recorded analyses for ``ticker``, oldest first (``composite_score`` is the owner score)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE ticker = ? ORDER BY generated_at DESC, id DESC LIMIT ?",
            (ticker.upper(), limit),
        ).fetchall()
    out = []
    for row in reversed(rows):
        item = dict(row)
        item["agent_scores"] = json.loads(item["agent_scores"])
        out.append(item)
    return out


_PREVIOUS = "SELECT * FROM analyses WHERE ticker = ? AND generated_at < ? ORDER BY generated_at DESC, id DESC LIMIT 1"


def _item(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    item["agent_scores"] = json.loads(item["agent_scores"])
    return item


def get_previous(ticker: str, before: str) -> dict[str, Any] | None:
    """Most recent analysis for ``ticker`` generated strictly before ``before``."""
    with _connect() as conn:
        return _item(conn.execute(_PREVIOUS, (ticker.upper(), before)).fetchone())


def get_previous_many(pairs: list[tuple[str, str]]) -> list[dict[str, Any] | None]:
    """``get_previous`` for each (ticker, before) pair, over one connection."""
    if not pairs:
        return []
    with _connect() as conn:
        return [_item(conn.execute(_PREVIOUS, (ticker.upper(), before)).fetchone()) for ticker, before in pairs]


def get_first_after(ticker: str, since: str) -> dict[str, Any] | None:
    """Earliest recorded analysis for ``ticker`` generated on or after ``since`` (ISO date or datetime)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE ticker = ? AND generated_at >= ? ORDER BY generated_at ASC, id ASC LIMIT 1",
            (ticker.upper(), since),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["agent_scores"] = json.loads(item["agent_scores"])
    return item


def get_latest(ticker: str) -> dict[str, Any] | None:
    """Most recent recorded analysis for ``ticker``."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE ticker = ? ORDER BY generated_at DESC, id DESC LIMIT 1",
            (ticker.upper(),),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["agent_scores"] = json.loads(item["agent_scores"])
    return item


def get_latest_all(since: str | None = None) -> list[dict[str, Any]]:
    """The most recent recorded analysis of every ticker (generated on or after ``since`` when given), by ticker."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT a.* FROM analyses a WHERE a.id = (SELECT b.id FROM analyses b WHERE b.ticker = a.ticker"
            " ORDER BY b.generated_at DESC, b.id DESC LIMIT 1) AND a.generated_at >= ? ORDER BY a.ticker",
            (since or "",),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["agent_scores"] = json.loads(item["agent_scores"])
        out.append(item)
    return out


# ------------------------------------------------------------------ watchlist


def _watchlist_path() -> Path:
    return settings.STORE_DIR / "watchlist.json"


def load_watchlist() -> list[str]:
    path = _watchlist_path()
    if not path.exists():
        return list(DEFAULT_WATCHLIST)
    try:
        data = json.loads(path.read_text())
        return [str(t).upper() for t in data.get("tickers", [])]
    except (OSError, json.JSONDecodeError):
        return list(DEFAULT_WATCHLIST)


def save_watchlist(tickers: list[str]) -> list[str]:
    clean: list[str] = []
    for t in tickers:
        symbol = str(t).strip().upper()
        if symbol and symbol not in clean and len(symbol) <= 12:
            clean.append(symbol)
    path = _watchlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tickers": clean}, indent=2))
    return clean
