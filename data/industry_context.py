"""Industry context for the Analyze tab: where to read the sector outlook, and the latest earnings release and
investor presentation of the company and its closest peers.

Two parts, both plain links, no scoring:

* Sources: ``config/industry_sources.json`` is a short, curated list of free outlooks (agency forecasts, trade
  statistics, big investor conferences), checked by hand. A company gets the sources of its Yahoo industry, then
  of its Yahoo sector, then the general macro ones. Nothing here is fetched at run time.
* Documents: the latest earnings release and presentation from ``data.sec_reports.recent_filings`` for the company
  and up to five peers. Peers are chosen the way the Competitive Position agent does (the company's Yahoo industry,
  nearest in market value, else the SEC industry code). Slides often sit only on the company's investor-relations
  site; the SEC carries them when the company attaches them to an 8-K, so a missing presentation is not proof that
  none exists.

The whole call has a time budget (``BUDGET_SECONDS``). Each company's filing list is cached for an hour, the fetches
share the SEC throttle in ``data/sec_edgar.py``, and a fetch that misses the budget keeps running and fills the
cache for the next call. A peer that fails is skipped with a note, never fatal. Providers raise on failure so an
error is never cached. Nothing is written to the history database.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from typing import Any

import config.settings as settings
from analysis.competitive_metrics import Candidate, choose_peers, choose_peers_by_sic
from data import fundamentals as _fundamentals
from data import peers as _peers
from data.cache_utils import ttl_cache

logger = logging.getLogger(__name__)

SOURCES_PATH = Path(settings.PROJECT_ROOT) / "config" / "industry_sources.json"
MAX_PEERS = 5
BUDGET_SECONDS = 25.0
FILINGS_TTL_SECONDS = 3600
DOCUMENT_KINDS = ("earnings", "presentation")
POOL_WORKERS = 3

_POOL = ThreadPoolExecutor(max_workers=POOL_WORKERS, thread_name_prefix="industry")

NO_CONNECTOR = "The SEC reports connector is not available in this build"


class _NoAccess(Exception):
    """The SEC contact email is not set, so no filing list can be read."""

    def __init__(self, message: str | None, fix: str | None) -> None:
        super().__init__(message or "SEC access is not set up")
        self.fix = fix


# ------------------------------------------------------------------ sources


@lru_cache(maxsize=4)
def _load_sources(path: str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Industry sources unreadable (%s): %s", path, exc)
        return {"version": 1, "general": [], "sectors": {}}


def sources_for(sector: str | None, industry: str | None, path: Path | str | None = None) -> list[dict[str, Any]]:
    """Industry sources first, then sector, then general; a link listed at two levels shows once, most specific."""
    data = _load_sources(str(path or SOURCES_PATH))
    block = (data.get("sectors") or {}).get(sector or "") or {}
    levels = (
        ("industry", (block.get("industries") or {}).get(industry or "") or []),
        ("sector", block.get("sources") or []),
        ("general", data.get("general") or []),
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scope, items in levels:
        for item in items:
            if item["url"] in seen:
                continue
            seen.add(item["url"])
            out.append({"name": item["name"], "publisher": item["publisher"], "url": item["url"],
                        "what": item["what"], "cadence": item["cadence"], "scope": scope,
                        "conference": bool(item.get("conference"))})
    return out


# ------------------------------------------------------------------ filings


def _recent_filings(ticker: str) -> dict[str, Any]:
    try:
        from data import sec_reports
    except ImportError as exc:
        raise RuntimeError(NO_CONNECTOR) from exc
    return sec_reports.recent_filings(ticker, limit=40)


@ttl_cache(FILINGS_TTL_SECONDS)
def _filings_for(ticker: str) -> dict[str, Any]:
    """The connector's answer for one ticker; raises on ``no_access`` and ``error`` so they are not cached."""
    result = _recent_filings(ticker)
    status = result.get("status")
    if status == "no_access":
        raise _NoAccess(result.get("message"), result.get("fix"))
    if status == "error":
        raise RuntimeError(result.get("message") or "the SEC filing list could not be read")
    return result


def latest_documents(result: dict[str, Any], ticker: str, name: str, is_peer: bool) -> list[dict[str, Any]]:
    """The newest earnings release and the newest presentation in a filing list (a filing itself or an exhibit)."""
    found: dict[str, dict[str, Any]] = {}
    for filing in result.get("filings") or []:  # newest first
        candidates: list[tuple[str, str, str | None]] = []
        for exhibit in filing.get("exhibits") or []:  # the exhibit is the document; the 8-K itself is a cover page
            if exhibit.get("kind") in DOCUMENT_KINDS and exhibit.get("url"):
                candidates.append((exhibit["kind"], exhibit.get("label") or "Exhibit", exhibit["url"]))
        if filing.get("kind") in DOCUMENT_KINDS and filing.get("url"):
            candidates.append((filing["kind"], filing.get("title") or filing.get("form") or "Filing", filing["url"]))
        for kind, label, url in candidates:
            if kind not in found:
                found[kind] = {"ticker": ticker, "name": name, "is_peer": is_peer, "kind": kind,
                               "filed": filing.get("filed"), "label": label, "url": url}
    return [found[k] for k in DOCUMENT_KINDS if k in found]


# ------------------------------------------------------------------ peers


def _choose_peers(symbol: str, name: str, info: dict[str, Any], members: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Up to ``MAX_PEERS`` (ticker, name): the Yahoo industry first, the SEC industry code when it is thin."""
    cap = _peers.market_cap_usd(info)
    identity = {"subject_ticker": symbol, "subject_name": name}
    candidates = [Candidate(m["ticker"], m["name"], m["market_cap"]) for m in members]
    choice = choose_peers(cap, candidates, MAX_PEERS, **identity)
    pool = {c.ticker: c.name or c.ticker for c in candidates}
    if len(choice.tickers) < 4:
        try:
            directory = _peers.sic_directory()
            sic = (_peers.sic_records().get(symbol) or {}).get("sic")
            by_sic = choose_peers_by_sic(cap, sic, directory, MAX_PEERS, **identity) if sic is not None else None
        except Exception as exc:  # noqa: BLE001 - the SEC industry code is an extra, not a requirement
            logger.info("SEC industry code peers unavailable for %s: %s", symbol, exc)
            by_sic = None
        if by_sic and len(by_sic.tickers) > len(choice.tickers):
            choice = by_sic
            pool.update({c.ticker: c.name or c.ticker for c in directory})
    return [(t, pool.get(t, t)) for t in choice.tickers[:MAX_PEERS]]


# ------------------------------------------------------------------ the payload


def industry_context(ticker: str) -> dict[str, Any]:
    """The ``/api/industry/<ticker>`` payload. Never raises for a provider failure."""
    start = time.monotonic()
    deadline = start + BUDGET_SECONDS
    symbol = ticker.strip().upper()
    payload: dict[str, Any] = {"ticker": symbol, "status": "ok", "message": None, "fix": None, "sector": None,
                               "industry": None, "sources": [], "documents": [], "notes": []}
    notes: list[str] = payload["notes"]

    try:
        info = _fundamentals._load_info(symbol)
    except Exception as exc:  # noqa: BLE001 - general sources still help; the company is unknown
        logger.info("Industry context: no Yahoo profile for %s: %s", symbol, exc)
        payload["sources"] = sources_for(None, None)
        payload["status"] = "error"
        payload["message"] = "Yahoo Finance has no profile for this ticker, so only the general sources are shown"
        return payload

    sector, industry = info.get("sector"), info.get("industry")
    payload["sector"], payload["industry"] = sector, industry
    payload["sources"] = sources_for(sector, industry)
    if not sector:
        notes.append("No sector on Yahoo for this ticker (a fund or a trust has none), so only general sources are shown")
    name = info.get("shortName") or info.get("longName") or symbol
    quote_type = str(info.get("quoteType") or "").upper()
    if quote_type and quote_type != "EQUITY":
        notes.append("Earnings releases and presentations apply to companies, not funds")
        return payload

    peers: list[tuple[str, str]] = []
    if info.get("industryKey"):
        members_future = _peers.submit_once(f"industry:{info['industryKey']}", _peers.industry_members, info["industryKey"])
        try:
            members = members_future.result(timeout=max(0.0, deadline - time.monotonic()))
        except Exception as exc:  # noqa: BLE001 - timed out or Yahoo refused: no peers, the company still shows
            logger.info("Industry members unavailable for %s: %s", symbol, exc)
            members = []
        peers = _choose_peers(symbol, name, info, members)
    if not peers:
        notes.append("No peers found, so only the company's own documents are shown")

    names = {symbol: name, **dict(peers)}
    order = [symbol] + [t for t, _ in peers]
    futures: dict[str, Future] = {t: _POOL.submit(_filings_for, t) for t in order}
    wait(list(futures.values()), timeout=max(0.0, deadline - time.monotonic()))

    documents: list[dict[str, Any]] = []
    for t in order:
        future, is_peer = futures[t], t != symbol
        label = names[t]
        if not future.done():
            notes.append(f"{label}: still loading, open the tab again in a minute")
            continue
        try:
            result = future.result()
        except _NoAccess as exc:
            if not is_peer:
                payload["status"], payload["message"], payload["fix"] = "no_access", str(exc), exc.fix
            break  # no contact email: no company can be read, so stop here and keep the sources
        except Exception as exc:  # noqa: BLE001 - a peer that fails is skipped
            notes.append(f"{label}: skipped ({str(exc)[:120]})")
            continue
        if result.get("status") == "not_sec_filer":
            notes.append(f"{label}: files no 10-K or 10-Q with the SEC, so no documents here")
            continue
        found = latest_documents(result, t, result.get("name") or label, is_peer)
        if not found:
            notes.append(f"{label}: no earnings release or presentation among the latest SEC filings")
        documents.extend(found)
    payload["documents"] = documents
    if any(d["kind"] == "earnings" for d in documents) and not any(d["kind"] == "presentation" for d in documents):
        notes.append("No investor presentation among the latest SEC filings: companies often post slides only on their "
                     "investor-relations site")
    return payload
