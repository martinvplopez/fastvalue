"""Peer companies and their figures for the Competitive Position agent.

Sources, in the order the agent tries them: the members of the company's Yahoo industry (market value from
Yahoo's market weight times the industry total, so no call per peer), then the SEC industry code (SIC) of the
companies the value study already follows. Each company's figures come as a "card": annual revenue, cost of
revenue, gross profit, operating income, receivables, payables, inventory and ROIC for up to ten years (SEC
10-K filings for US filers when the SEC contact is set, else Yahoo's four years).

Cards live a week on disk (``data/cache/competitive/``, git-ignored); a failure is never written, and a peer that
failed is skipped for 30 minutes so a refusing Yahoo is not hammered. One shared pool of three threads fetches
peers, deduplicated by key, so a portfolio review of many holdings fetches a shared peer once; fetches that miss
an analysis deadline keep running and fill the cache for the next run (decision_log.md D-34).
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yfinance as yf

import config.settings as settings
from analysis.competitive_metrics import Candidate, card_years
from data import fundamentals as _fundamentals
from data.financial_statements import fetch_statements
from data.fx import convert_latest, major
from data.sec_edgar import (
    SecConfigError,
    SecDataError,
    filed_since,
    has_sec_access,
    load_company_facts,
    load_ticker_map,
    sec_cache_dir,
)
from data.sec_filings import competitive_dir
from data.sec_margins import load_margin_facts, margins_asof
from data.yahoo import yahoo_call

logger = logging.getLogger(__name__)

CARD_MAX_AGE_DAYS = 7
CARD_VERSION = 2  # bump when the card's figures or fields change, so cached cards are rebuilt
INDUSTRY_MAX_AGE_DAYS = 7
FAILURE_PAUSE_SECONDS = 1800
YAHOO_NEW_CARDS = 4  # new Yahoo-only cards fetched per evaluation, so the other agents are not starved
DIRECTORY_TTL_SECONDS = 3600
POOL_WORKERS = 3

_POOL = ThreadPoolExecutor(max_workers=POOL_WORKERS, thread_name_prefix="peers")
_LOCK = threading.Lock()
_INFLIGHT: dict[str, Future] = {}
_FAILED: dict[str, float] = {}
_MEMO: dict[str, tuple[float, Any]] = {}


def submit_once(key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    """Run ``fn`` in the shared peer pool; a second request for the same key gets the running future."""
    with _LOCK:
        running = _INFLIGHT.get(key)
        if running is not None and not running.done():
            return running
        future = _POOL.submit(fn, *args, **kwargs)
        _INFLIGHT[key] = future
    future.add_done_callback(lambda f, k=key: _forget(k, f))
    return future


def _forget(key: str, future: Future) -> None:
    with _LOCK:
        if _INFLIGHT.get(key) is future:
            del _INFLIGHT[key]


def _memo(key: str, ttl: float, build: Callable[[], Any]) -> Any:
    now = time.monotonic()
    with _LOCK:
        hit = _MEMO.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = build()
    with _LOCK:
        _MEMO[key] = (now, value)
    return value


def market_cap_usd(info: dict[str, Any]) -> float | None:
    """Yahoo's ``marketCap`` in US dollars, or None without a positive figure or a rate.

    Yahoo gives the cap in the major unit of the quote currency (pounds for a London listing quoted in pence), so
    the amount is converted from the major currency, never from the minor unit a price is quoted in.
    """
    cap = _num(info.get("marketCap"))
    if not cap or cap <= 0:
        return None
    code, _ = major(info.get("currency") or "USD")
    try:
        return convert_latest(cap, code, "USD")
    except Exception as exc:  # noqa: BLE001 - no rate: no size band, so no peers rather than wrong ones
        logger.info("No %s rate for the market cap: %s", code, exc)
        return None


def _fresh(path: Path, max_age_days: float) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ------------------------------------------------------------------ Yahoo industries


def industry_members(industry_key: str | None) -> list[dict[str, Any]]:
    """Members of a Yahoo industry: ticker, name, market cap in US dollars (market weight x industry total).

    Cached a week on disk. Rows without a name are dropped (Yahoo sends a few). Returns an empty list when Yahoo
    refuses; that is not cached.
    """
    if not industry_key:
        return []
    path = competitive_dir() / f"industry_{re.sub(r'[^a-z0-9-]', '_', industry_key.lower())}.json"
    if _fresh(path, INDUSTRY_MAX_AGE_DAYS):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            pass
    try:
        industry = yf.Industry(industry_key)
        top = yahoo_call(lambda: industry.top_companies, f"{industry_key} industry")
        total = _num((yahoo_call(lambda: industry.overview, f"{industry_key} overview") or {}).get("market_cap"))
    except Exception as exc:  # noqa: BLE001 - the SEC industry code is the fallback
        logger.info("Yahoo industry %s unavailable: %s", industry_key, exc)
        return []
    if top is None or not isinstance(top, pd.DataFrame) or top.empty or not total:
        return []
    members = []
    for symbol, row in top.iterrows():
        name, weight = row.get("name"), _num(row.get("market weight"))
        if not isinstance(name, str) or not name.strip() or not weight or weight <= 0:
            continue
        members.append({"ticker": str(symbol).upper(), "name": name.strip(), "market_cap": weight * total})
    if members:
        path.write_text(json.dumps(members))
    return members


# ------------------------------------------------------------------ SEC industry codes


def sic_records() -> dict[str, dict[str, Any]]:
    """Ticker -> name, sic, sic_description, cik for every company whose SEC extract is cached (no network)."""

    def build() -> dict[str, dict[str, Any]]:
        try:
            ticker_map = load_ticker_map(max_age_days=math.inf)
        except Exception as exc:  # noqa: BLE001 - no cached SEC ticker list: no SIC peers
            logger.info("SEC ticker list unavailable: %s", exc)
            return {}
        by_cik: dict[int, str] = {}
        for ticker, cik in ticker_map.items():
            by_cik.setdefault(cik, ticker)
        out: dict[str, dict[str, Any]] = {}
        for path in sec_cache_dir().glob("meta_*.json"):
            try:
                cik = int(path.stem.split("_", 1)[1])
                meta = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            ticker = by_cik.get(cik)
            if ticker and meta.get("sic"):
                out[ticker] = {"name": meta.get("name"), "sic": int(meta["sic"]),
                               "sic_description": meta.get("sic_description"), "cik": cik}
        return out

    return _memo(f"sic_records:{sec_cache_dir()}", DIRECTORY_TTL_SECONDS, build)


def latest_market_caps() -> dict[str, float]:
    """Ticker -> market cap (US dollars) at the latest month of the value study panel, else from cached cards."""
    panel = settings.DATA_DIR / "value_panel.parquet"

    def build() -> dict[str, float]:
        if panel.exists():
            try:
                frame = pd.read_parquet(panel, columns=["date", "ticker", "market_cap"])
                last = frame[frame["date"] == frame["date"].max()].dropna(subset=["market_cap"])
                return {str(t): float(c) for t, c in zip(last["ticker"], last["market_cap"]) if c > 0}
            except Exception as exc:  # noqa: BLE001 - fall back to the cards
                logger.info("Value panel unreadable: %s", exc)
        out: dict[str, float] = {}
        for path in competitive_dir().glob("card_*.json"):
            try:
                card = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            cap = _num(card.get("market_cap"))
            if cap and cap > 0 and card.get("ticker"):
                out[card["ticker"]] = cap
        return out

    return _memo(f"market_caps:{panel}", DIRECTORY_TTL_SECONDS, build)


def sic_directory() -> list[Candidate]:
    """Companies with an SEC industry code and a market value, as peer candidates (empty without the caches)."""
    caps = latest_market_caps()
    return [Candidate(t, r["name"], caps[t], r["sic"], r["cik"]) for t, r in sic_records().items() if t in caps]


# ------------------------------------------------------------------ company cards


def _card_path(ticker: str) -> Path:
    return competitive_dir() / f"card_{re.sub(r'[^A-Z0-9.-]', '_', ticker.upper())}.json"


def cached_card(ticker: str, max_age_days: float = CARD_MAX_AGE_DAYS) -> dict[str, Any] | None:
    path = _card_path(ticker)
    if not _fresh(path, max_age_days):
        return None
    try:
        card = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return card if isinstance(card, dict) and card.get("years") and card.get("version") == CARD_VERSION else None


def recently_failed(ticker: str) -> bool:
    with _LOCK:
        return time.monotonic() - _FAILED.get(ticker.upper(), -math.inf) < FAILURE_PAUSE_SECONDS


def _ticker_map() -> dict[str, int]:
    if not has_sec_access():
        return {}
    try:
        return _memo(f"ticker_map:{sec_cache_dir()}", DIRECTORY_TTL_SECONDS, lambda: load_ticker_map(max_age_days=30))
    except Exception as exc:  # noqa: BLE001 - Yahoo statements are the fallback
        logger.info("SEC ticker list unavailable: %s", exc)
        return {}


def is_sec_filer(ticker: str) -> bool:
    return ticker.upper() in _ticker_map()


def _sec_years(ticker: str, cik: int, new_filings: bool = False) -> tuple[list[dict[str, Any]], int | None, str | None]:
    today = pd.Timestamp.today().normalize()
    cf = load_company_facts(ticker, cik, max_age_days=CARD_MAX_AGE_DAYS, new_filings=new_filings)
    st = cf.statements_asof(today, max_years=10)
    margins = margins_asof(load_margin_facts(cik, max_age_days=CARD_MAX_AGE_DAYS, new_filings=new_filings), today)
    return card_years(st, margins), cf.sic, cf.name


def _reason(exc: Exception) -> str:
    """The failure in one line; the lookup errors raised here already carry their reason, so no type prefix on them."""
    text = str(exc)
    if isinstance(exc, LookupError):
        return re.sub(r"^(?:\w+(?:Error|Exception): )+", "", text)[:200]
    return f"{type(exc).__name__}: {text}"[:200]


def company_card(
    ticker: str, info: dict[str, Any] | None = None, max_age_days: float = CARD_MAX_AGE_DAYS, market_cap: float | None = None,
    new_filings: bool = False,
) -> dict[str, Any]:
    """One company's figures (see the module docstring). Never raises: ``error`` is set instead.

    ``info`` is Yahoo's quote summary when the caller has it; ``market_cap`` is a hint (US dollars) used when
    Yahoo gives none. ``new_filings`` rebuilds a cached card once the company files a 10-K or 10-Q after it
    (the company being analysed; peers keep the weekly refresh).
    """
    symbol = ticker.strip().upper()
    card = cached_card(symbol, max_age_days)
    if card is not None and new_filings and card.get("cik") and filed_since(int(card["cik"]), _card_path(symbol).stat().st_mtime):
        card = None
    if card is not None:
        return card
    base: dict[str, Any] = {"version": CARD_VERSION, "ticker": symbol, "name": None, "source": None, "cik": None, "sic": None, "currency": "USD",
                            "market_cap": market_cap, "ev_to_fcf": None, "years": [], "error": None}
    try:
        if info is None:
            try:
                info = _fundamentals._load_info(symbol)
            except Exception as exc:  # noqa: BLE001 - statements may still come
                logger.info("No Yahoo info for %s: %s", symbol, exc)
                info = {}
        base["name"] = info.get("shortName") or info.get("longName")
        base["currency"] = info.get("financialCurrency") or info.get("currency") or "USD"
        base["market_cap"] = market_cap_usd(info) or market_cap
        ev, fcf = _num(info.get("enterpriseValue")), _num(info.get("freeCashflow"))
        mismatch = bool(info.get("currency") and info.get("financialCurrency") and major(info["currency"])[0] != info["financialCurrency"])
        base["ev_to_fcf"] = ev / fcf if ev and fcf and ev > 0 and fcf > 0 and not mismatch else None  # as Fundamentals: no ratio across currencies

        years: list[dict[str, Any]] = []
        cik = _ticker_map().get(symbol)
        if cik is not None:
            try:
                years, sic, sec_name = _sec_years(symbol, cik, new_filings)
                base.update(source="SEC" if years and years[0].get("revenue") else None, cik=cik, sic=sic)
                base["name"] = base["name"] or sec_name
            except (SecDataError, SecConfigError, OSError, ValueError) as exc:
                logger.info("SEC figures unavailable for %s: %s", symbol, exc)
            if base["source"] is None:
                years = []
        if not years:
            st = fetch_statements(symbol)
            if not st.available:
                raise LookupError(st.error or "annual statements unavailable")
            years = card_years(st, None, max_years=5)
            base["source"] = "Yahoo"
        if not years or years[0].get("revenue") is None:
            raise LookupError("no revenue in the annual statements")
        base["years"] = [{k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in y.items()} for y in years]
        base["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _card_path(symbol).write_text(json.dumps(base))
        with _LOCK:
            _FAILED.pop(symbol, None)
        return base
    except Exception as exc:  # noqa: BLE001 - one peer must not sink the analysis
        logger.info("No figures for %s: %s", symbol, exc)
        with _LOCK:
            _FAILED[symbol] = time.monotonic()
        base["years"] = []
        base["error"] = _reason(exc)
        return base


def peer_cards(
    tickers: list[str], deadline: float, market_caps: dict[str, float] | None = None, yahoo_budget: int = YAHOO_NEW_CARDS
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Cards of ``tickers`` (cache first, then the shared pool until ``deadline``, a ``time.monotonic()`` time).

    Returns (cards that have figures, tickers still loading, tickers that failed). A cold run starts at most
    ``yahoo_budget`` new Yahoo-only cards; the rest stay pending for the next run.
    """
    caps = market_caps or {}
    cards: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    failed: list[str] = []
    futures: dict[str, Future] = {}
    new_yahoo = 0
    for ticker in tickers:
        card = cached_card(ticker)
        if card is not None:
            cards[ticker] = card
        elif recently_failed(ticker):
            failed.append(ticker)
        elif not is_sec_filer(ticker) and new_yahoo >= yahoo_budget:
            pending.append(ticker)
        else:
            new_yahoo += 0 if is_sec_filer(ticker) else 1
            futures[ticker] = submit_once(f"card:{ticker}", company_card, ticker, None, CARD_MAX_AGE_DAYS, caps.get(ticker))
    if futures:
        wait(list(futures.values()), timeout=max(0.0, deadline - time.monotonic()))
    for ticker, future in futures.items():
        if not future.done():
            pending.append(ticker)
            continue
        try:
            card = future.result()
        except Exception:  # noqa: BLE001 - company_card never raises; this is belt and braces
            failed.append(ticker)
            continue
        if card.get("years"):
            cards[ticker] = card
        else:
            failed.append(ticker)
    return cards, pending, failed
