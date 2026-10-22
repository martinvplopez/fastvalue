"""News headline sentiment for a ticker.

Headlines come from pluggable sources (see :class:`SentimentSource`). Yahoo
Finance news needs no key. Two API sources switch on when their key is in the
environment: Alpha Vantage NEWS_SENTIMENT (``ALPHAVANTAGE_API_KEY``), which
attaches a sentiment score and a relevance score per ticker to every item, and
Finnhub company news (``FINNHUB_API_KEY``). A provider's own score is used when
present; otherwise the title is scored with a small finance-specific lexicon
(multi-word phrases first, simple negation such as "not", "no" or "fails to").
Items are weighted by recency with a 3-day half-life and by the provider's
relevance when given; items older than 14 days or near-identical to a newer
title are dropped. One sentiment snapshot per ticker per day is appended to
``data/store/sentiment_history/`` so a point-in-time history builds up for a
later backtest (the agent has no history to replay today).

Adding a source (Reddit, X, StockTwits, ...)
--------------------------------------------
1. Write a class with a ``name`` attribute and a ``fetch(ticker) -> list[Headline]``
   method. Raise on failure; never return placeholder or made-up items.
2. Append an instance to ``SOURCES``. Results are cached per (source name, ticker).

Reddit (the OAuth API, e.g. via PRAW) and X (API v2 recent search) both need API
credentials. Read them from environment variables, and only add the source to
``SOURCES`` when the credentials are set. Social posts are far noisier than news
headlines, so consider a per-source weight before mixing them into one average.

All titles, publishers and URLs are untrusted external text: they are kept as plain
strings (the UI escapes them) and non-http(s) URLs are dropped.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import yfinance as yf

import config.settings as settings
from data.cache_utils import ttl_cache
from data.yahoo import yahoo_call

logger = logging.getLogger(__name__)

HALF_LIFE_DAYS = 3.0
MAX_AGE_DAYS = 14.0
NEGATION_WINDOW = 3  # tokens after a negator that it can flip
NEGATION_FACTOR = -0.8  # a negated term flips sign and loses a little strength
QUESTION_DAMPING = 0.5  # "Is X a buy after its earnings beat?" is speculation, not news
DUPLICATE_JACCARD = 0.8
MAX_TITLE_LENGTH = 300
ALPHA_VANTAGE_KEY_ENV = "ALPHAVANTAGE_API_KEY"
FINNHUB_KEY_ENV = "FINNHUB_API_KEY"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
HTTP_TIMEOUT = 20.0

# Finance-specific lexicon. Values are per-term polarity in -1..1. Multi-word phrases
# are matched before single words, so "record loss" beats "record" and
# "cuts guidance" is one term.
LEXICON: dict[str, float] = {
    # Earnings and guidance
    "beat": 0.6, "beats": 0.6, "beat estimates": 0.7, "beats estimates": 0.7,
    "tops estimates": 0.7, "top estimates": 0.7, "beats expectations": 0.7,
    "tops expectations": 0.7, "exceeds expectations": 0.7, "blowout": 0.6,
    "raises guidance": 0.9, "raised guidance": 0.9, "lifts guidance": 0.9,
    "boosts guidance": 0.9, "raises outlook": 0.8, "raises forecast": 0.8,
    "record": 0.5, "record high": 0.6, "record revenue": 0.7, "all time high": 0.6,
    "miss": -0.6, "misses": -0.6, "missed": -0.6, "misses estimates": -0.7,
    "missed estimates": -0.7, "misses expectations": -0.7, "falls short": -0.6,
    "cuts guidance": -0.9, "cut guidance": -0.9, "lowers guidance": -0.9,
    "slashes guidance": -1.0, "cuts outlook": -0.8, "lowers outlook": -0.8,
    "cuts forecast": -0.8, "lowers forecast": -0.8, "profit warning": -0.8,
    "record loss": -0.8, "disappoints": -0.6, "disappointing": -0.6,
    "impress": 0.4, "impresses": 0.4, "strong demand": 0.5, "weak demand": -0.5,
    # Analyst actions
    "upgrade": 0.8, "upgrades": 0.8, "upgraded": 0.8, "outperform": 0.6,
    "outperforms": 0.6, "overweight": 0.5, "strong buy": 0.6, "bullish": 0.5,
    "raises price target": 0.5, "price target raised": 0.5,
    "downgrade": -0.8, "downgrades": -0.8, "downgraded": -0.8, "underperform": -0.6,
    "underperforms": -0.6, "underweight": -0.5, "bearish": -0.5,
    "cuts price target": -0.5, "lowers price target": -0.5, "price target cut": -0.5,
    # Price action
    "surge": 0.7, "surges": 0.7, "surged": 0.7, "surging": 0.6, "soar": 0.8, "soars": 0.8,
    "soared": 0.8, "soaring": 0.7, "skyrocket": 0.8, "skyrockets": 0.8, "skyrocketed": 0.8,
    "jumps": 0.5, "rally": 0.4, "rallies": 0.4, "climbs": 0.4, "rebounds": 0.4, "gains": 0.3,
    "plunge": -0.8, "plunges": -0.8, "plunged": -0.8, "plunging": -0.7, "plummets": -0.8,
    "tumble": -0.6, "tumbles": -0.6, "slumps": -0.6, "sinks": -0.5, "slides": -0.4,
    "falls": -0.3, "drops": -0.3, "crash": -0.7, "crashes": -0.7, "selloff": -0.5,
    "sell off": -0.5,
    # Corporate actions
    "buyback": 0.5, "buybacks": 0.5, "share repurchase": 0.5, "raises dividend": 0.6,
    "dividend hike": 0.6, "approval": 0.6, "approved": 0.5, "approves": 0.5,
    "fda approval": 0.8, "partnership": 0.4, "partners with": 0.3,
    "wins contract": 0.5, "contract win": 0.5, "breakthrough": 0.5,
    "layoffs": -0.5, "job cuts": -0.5, "cuts jobs": -0.5,
    # Legal, regulatory and distress
    "lawsuit": -0.6, "sued": -0.6, "class action": -0.6, "probe": -0.6,
    "investigation": -0.6, "subpoena": -0.6, "antitrust": -0.4, "fined": -0.5,
    "recall": -0.6, "recalls": -0.6, "bankruptcy": -1.0, "chapter 11": -1.0,
    "fraud": -0.9, "halt": -0.5, "halts": -0.5, "halted": -0.5, "delisting": -0.9,
    "delisted": -0.9, "short seller": -0.5, "short report": -0.6, "warning": -0.4,
}

NEGATORS = frozenset({"not", "no", "never", "without", "neither", "nor", "cannot"})
NEGATOR_PHRASES = frozenset({("fails", "to"), ("failed", "to"), ("fail", "to"), ("unable", "to")})

_PHRASES: dict[tuple[str, ...], float] = {tuple(term.split()): value for term, value in LEXICON.items()}
_MAX_PHRASE = max(len(p) for p in _PHRASES)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


# ---------------------------------------------------------------- data classes


@dataclass(frozen=True)
class Headline:
    """One news item as returned by a source. All text fields are untrusted plain strings."""

    title: str
    source: str  # name of the SentimentSource that produced it
    publisher: str | None = None
    url: str | None = None
    published: datetime | None = None  # timezone-aware UTC
    summary: str | None = None
    provider_score: float | None = None  # sentiment attached by the provider, -1..1 (Alpha Vantage)
    relevance: float | None = None  # how much the item is about the ticker, 0..1 (Alpha Vantage)


@dataclass(frozen=True)
class ScoredHeadline:
    """A kept headline with its score (provider's or lexicon) and its weight."""

    title: str
    source: str
    publisher: str | None
    url: str | None
    published: datetime
    score: float  # -1..1
    age_days: float
    weight: float  # 0.5 ** (age_days / HALF_LIFE_DAYS), times the provider relevance when given
    terms: tuple[str, ...]  # matched lexicon terms; negated ones are prefixed "not "
    score_source: str = "lexicon"  # "lexicon" or "provider"
    relevance: float | None = None

    @property
    def published_iso(self) -> str:
        return self.published.isoformat(timespec="seconds")


@dataclass
class NewsSentimentData:
    """Recency-weighted headline sentiment for one ticker."""

    ticker: str
    available: bool
    error: str | None = None
    headlines: list[ScoredHeadline] = field(default_factory=list)  # newest first
    weighted_score: float | None = None  # recency-weighted mean score, -1..1
    sources: list[str] = field(default_factory=list)  # sources that answered
    source_errors: dict[str, str] = field(default_factory=dict)
    fetched_count: int = 0
    dropped_stale: int = 0
    dropped_undated: int = 0
    dropped_duplicates: int = 0
    provider_scored: int = 0  # kept headlines that carried a provider sentiment score
    as_of: str | None = None  # ISO timestamp of the newest kept headline


class SentimentSource(Protocol):
    """A pluggable headline source. ``fetch`` raises on failure."""

    name: str

    def fetch(self, ticker: str) -> list[Headline]: ...


# ---------------------------------------------------------------- parsing helpers


def _clean_text(value: Any, limit: int = MAX_TITLE_LENGTH) -> str | None:
    if value is None:
        return None
    text = " ".join(_CONTROL_RE.sub(" ", str(value)).split())
    return text[:limit] or None


def _clean_url(value: Any) -> str | None:
    url = _clean_text(value, limit=2000)
    if url and url.lower().startswith(("https://", "http://")):
        return url
    return None


def _parse_time(value: Any) -> datetime | None:
    """Epoch seconds (or milliseconds) or an ISO 8601 string to an aware UTC datetime."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            seconds = float(value)
            if not math.isfinite(seconds) or seconds <= 0:
                return None
            if seconds > 1e12:
                seconds /= 1000.0
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        if isinstance(value, str) and value.strip():
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return None


def _nested_url(value: Any) -> str | None:
    return _clean_url(value.get("url")) if isinstance(value, dict) else None


def parse_yahoo_news_item(item: Any, source: str = "Yahoo Finance") -> Headline | None:
    """
    Parse one entry of ``yf.Ticker(t).news``.

    Handles the current nested shape (``{"id", "content": {"title", "summary",
    "pubDate", "provider": {"displayName"}, "canonicalUrl": {"url"},
    "clickThroughUrl": {"url"}}}``) and the older flat shape (``title``,
    ``publisher``, ``link``, ``providerPublishTime`` epoch seconds).
    """
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if isinstance(content, dict):
        title = content.get("title")
        summary = content.get("summary") or content.get("description")
        published = _parse_time(content.get("pubDate") or content.get("displayTime"))
        provider = content.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else None
        url = _nested_url(content.get("canonicalUrl")) or _nested_url(content.get("clickThroughUrl"))
    else:
        title = item.get("title")
        summary = item.get("summary")
        published = _parse_time(item.get("providerPublishTime"))
        publisher = item.get("publisher")
        url = _clean_url(item.get("link"))

    clean_title = _clean_text(title)
    if not clean_title:
        return None
    return Headline(
        title=clean_title,
        source=source,
        publisher=_clean_text(publisher, limit=100),
        url=url,
        published=published,
        summary=_clean_text(summary, limit=1000),
    )


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _get_json(url: str, label: str, timeout: float = HTTP_TIMEOUT) -> Any:
    """GET a JSON document. Errors never echo the URL, which may carry an API key."""
    request = urllib.request.Request(url, headers={"User-Agent": "fastvalue", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{label}: HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label}: {exc.reason}") from None
    except json.JSONDecodeError:
        raise RuntimeError(f"{label}: response is not JSON") from None


def _parse_compact_time(value: Any) -> datetime | None:
    """Alpha Vantage timestamps look like 20260927T120000 (UTC)."""
    if not isinstance(value, str) or len(value) < 13:
        return None
    try:
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(value[:13], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def parse_alpha_vantage_item(item: Any, symbol: str, source: str = "Alpha Vantage") -> Headline | None:
    """One ``feed`` entry of the NEWS_SENTIMENT endpoint, with the ticker's own sentiment and relevance."""
    if not isinstance(item, dict):
        return None
    title = _clean_text(item.get("title"))
    if not title:
        return None
    score = relevance = None
    for entry in item.get("ticker_sentiment") or []:
        if isinstance(entry, dict) and str(entry.get("ticker", "")).upper() == symbol.upper():
            score = _num(entry.get("ticker_sentiment_score"))
            relevance = _num(entry.get("relevance_score"))
            break
    if score is None:
        score = _num(item.get("overall_sentiment_score"))
    return Headline(
        title=title,
        source=source,
        publisher=_clean_text(item.get("source"), limit=100),
        url=_clean_url(item.get("url")),
        published=_parse_compact_time(item.get("time_published")),
        summary=_clean_text(item.get("summary"), limit=1000),
        provider_score=max(-1.0, min(1.0, score)) if score is not None else None,
        relevance=max(0.0, min(1.0, relevance)) if relevance is not None else None,
    )


def parse_finnhub_item(item: Any, source: str = "Finnhub") -> Headline | None:
    """One entry of Finnhub's company-news list (epoch ``datetime``, ``headline``, ``source``, ``url``)."""
    if not isinstance(item, dict):
        return None
    title = _clean_text(item.get("headline"))
    if not title:
        return None
    return Headline(
        title=title,
        source=source,
        publisher=_clean_text(item.get("source"), limit=100),
        url=_clean_url(item.get("url")),
        published=_parse_time(item.get("datetime")),
        summary=_clean_text(item.get("summary"), limit=1000),
    )


class AlphaVantageNewsSource:
    """Alpha Vantage NEWS_SENTIMENT: news with a sentiment score and relevance per ticker (free key, 25 requests a day)."""

    name = "Alpha Vantage"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch(self, ticker: str, start: datetime | None = None, end: datetime | None = None) -> list[Headline]:
        params = {"function": "NEWS_SENTIMENT", "tickers": ticker.upper(), "limit": "50", "sort": "LATEST", "apikey": self.api_key}
        if start is not None:
            params["time_from"] = start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")
        if end is not None:
            params["time_to"] = end.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")
        data = _get_json(f"{ALPHA_VANTAGE_URL}?{urlencode(params)}", self.name)
        if not isinstance(data, dict) or not isinstance(data.get("feed"), list):
            note = data.get("Information") or data.get("Note") or data.get("Error Message") if isinstance(data, dict) else None
            note = str(note).replace(self.api_key, "<key>") if note and self.api_key else note  # its limit notice quotes the key
            raise RuntimeError(f"{self.name}: {_clean_text(note, limit=160) or 'no feed in the response'}")
        parsed = (parse_alpha_vantage_item(item, ticker, self.name) for item in data["feed"])
        return [h for h in parsed if h is not None]


class FinnhubNewsSource:
    """Finnhub company news for the last ``NEWS_LOOKBACK_DAYS`` (free key, 60 requests a minute)."""

    name = "Finnhub"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch(self, ticker: str, start: datetime | None = None, end: datetime | None = None) -> list[Headline]:
        end = end or datetime.now(timezone.utc)
        start = start or end - timedelta(days=settings.NEWS_LOOKBACK_DAYS)
        params = {"symbol": ticker.upper(), "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"), "token": self.api_key}
        data = _get_json(f"{FINNHUB_NEWS_URL}?{urlencode(params)}", self.name)
        if not isinstance(data, list):
            note = data.get("error") if isinstance(data, dict) else None
            raise RuntimeError(f"{self.name}: {_clean_text(note, limit=160) or 'unexpected payload'}")
        parsed = (parse_finnhub_item(item, self.name) for item in data)
        return [h for h in parsed if h is not None]


class YahooNewsSource:
    """Headlines from Yahoo Finance via ``yf.Ticker(t).news`` (no credentials needed)."""

    name = "Yahoo Finance"

    def fetch(self, ticker: str) -> list[Headline]:
        items = yahoo_call(lambda: yf.Ticker(ticker).news, f"{ticker} news")
        if items is None:
            return []
        if not isinstance(items, list):
            raise TypeError(f"unexpected news payload of type {type(items).__name__}")
        parsed = (parse_yahoo_news_item(item, self.name) for item in items)
        return [h for h in parsed if h is not None]


SOURCES: list[SentimentSource] = [YahooNewsSource()]


def extra_sources() -> list[SentimentSource]:
    """API sources whose key is set in the environment (read at call time, never stored)."""
    out: list[SentimentSource] = []
    key = os.environ.get(ALPHA_VANTAGE_KEY_ENV, "").strip()
    if key:
        out.append(AlphaVantageNewsSource(key))
    key = os.environ.get(FINNHUB_KEY_ENV, "").strip()
    if key:
        out.append(FinnhubNewsSource(key))
    return out


def all_sources() -> list[SentimentSource]:
    return list(SOURCES) + extra_sources()


def configured_source_names() -> list[str]:
    return [s.name for s in all_sources()]


# ---------------------------------------------------------------- scoring


def _tokens(text: str) -> list[str]:
    normalized = text.lower().replace("’", "'").replace("‘", "'")
    return _TOKEN_RE.findall(normalized)


def _negator_length(tokens: list[str], i: int) -> int:
    if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in NEGATOR_PHRASES:
        return 2
    token = tokens[i]
    if token == "not" and i + 1 < len(tokens) and tokens[i + 1] == "only":
        return 0  # "not only X but Y" is emphasis, not negation
    if token in NEGATORS or token.endswith("n't"):
        return 1
    return 0


def _match_term(tokens: list[str], i: int) -> tuple[tuple[str, ...], float] | None:
    for length in range(min(_MAX_PHRASE, len(tokens) - i), 0, -1):
        candidate = tuple(tokens[i : i + length])
        if candidate in _PHRASES:
            return candidate, _PHRASES[candidate]
    return None


def score_text(text: str) -> tuple[float, tuple[str, ...]]:
    """
    Lexicon score of one headline in -1..1 and the terms that matched.

    A negator ("not", "no", "never", "without", "...n't", "fails to") flips the
    next lexicon term within ``NEGATION_WINDOW`` tokens. Headlines phrased as
    questions are damped.
    """
    tokens = _tokens(text)
    total = 0.0
    matched: list[str] = []
    negate_left = 0
    i = 0
    while i < len(tokens):
        neg = _negator_length(tokens, i)
        if neg:
            negate_left = NEGATION_WINDOW
            i += neg
            continue
        hit = _match_term(tokens, i)
        if hit is not None:
            term, value = hit
            phrase = " ".join(term)
            if negate_left > 0:
                value *= NEGATION_FACTOR
                phrase = f"not {phrase}"
                negate_left = 0
            total += value
            matched.append(phrase)
            i += len(term)
            continue
        negate_left = max(0, negate_left - 1)
        i += 1

    if text.rstrip().endswith("?"):
        total *= QUESTION_DAMPING
    return max(-1.0, min(1.0, total)), tuple(matched)


def recency_weight(age_days: float) -> float:
    """Exponential decay with a ``HALF_LIFE_DAYS`` half-life."""
    return 0.5 ** (max(0.0, age_days) / HALF_LIFE_DAYS)


def _dedupe_tokens(title: str) -> frozenset[str]:
    return frozenset(_tokens(title))


def _is_duplicate(tokens: frozenset[str], seen: list[frozenset[str]]) -> bool:
    for other in seen:
        union = tokens | other
        if union and len(tokens & other) / len(union) >= DUPLICATE_JACCARD:
            return True
    return False


def summarize_headlines(
    ticker: str, headlines: Iterable[Headline], now: datetime | None = None
) -> NewsSentimentData:
    """Filter, dedupe, score and recency-weight headlines (pure; no network)."""
    now = now or datetime.now(timezone.utc)
    items = list(headlines)
    result = NewsSentimentData(ticker=ticker, available=True, fetched_count=len(items))

    dated = [h for h in items if h.published is not None]
    result.dropped_undated = len(items) - len(dated)
    dated.sort(key=lambda h: h.published, reverse=True)  # newest first, so the newest copy is kept

    seen_tokens: list[frozenset[str]] = []
    seen_urls: set[str] = set()
    kept: list[ScoredHeadline] = []
    for h in dated:
        age = max(0.0, (now - h.published).total_seconds() / 86400.0)
        if age > MAX_AGE_DAYS:
            result.dropped_stale += 1
            continue
        tokens = _dedupe_tokens(h.title)
        if (h.url and h.url in seen_urls) or _is_duplicate(tokens, seen_tokens):
            result.dropped_duplicates += 1
            continue
        seen_tokens.append(tokens)
        if h.url:
            seen_urls.add(h.url)
        lexicon_score, terms = score_text(h.title)
        if h.provider_score is not None:
            score, score_source = h.provider_score, "provider"
        else:
            score, score_source = lexicon_score, "lexicon"
        weight = recency_weight(age)
        if h.relevance is not None:
            weight *= max(settings.NEWS_MIN_RELEVANCE, h.relevance)
        kept.append(
            ScoredHeadline(
                title=h.title,
                source=h.source,
                publisher=h.publisher,
                url=h.url,
                published=h.published,
                score=round(score, 3),
                age_days=round(age, 2),
                weight=round(weight, 4),
                terms=terms,
                score_source=score_source,
                relevance=h.relevance,
            )
        )

    result.headlines = kept
    result.provider_scored = sum(1 for h in kept if h.score_source == "provider")
    total_weight = sum(h.weight for h in kept)
    if kept and total_weight > 0:
        result.weighted_score = sum(h.weight * h.score for h in kept) / total_weight
        result.as_of = kept[0].published_iso
    return result


# ---------------------------------------------------------------- public API


def _short_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message[:200]}" if message else type(exc).__name__


@ttl_cache(settings.CACHE_TTL_NEWS)
def _fetch_source(source_name: str, symbol: str) -> tuple[Headline, ...]:
    """Cached headlines from one source. Raises on failure, so errors are not cached."""
    for source in all_sources():
        if source.name == source_name:
            return tuple(source.fetch(symbol))
    raise LookupError(f"unknown news source {source_name}")


# ---------------------------------------------------------------- daily snapshots


def _history_path(ticker: str) -> Path:
    return settings.STORE_DIR / "sentiment_history" / f"{ticker.upper()}.jsonl"


def record_sentiment_snapshot(data: NewsSentimentData, today: date | None = None, social: dict[str, Any] | None = None) -> bool:
    """Append today's weighted score for the ticker (one line a day). Returns True when a line was written.

    ``social`` (from ``SocialSentimentData.snapshot_fields``) is stored under its own key; ``score`` stays the news tone.
    """
    if not data.available or data.weighted_score is None:
        return False
    day = (today or date.today()).isoformat()
    path = _history_path(data.ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lines = path.read_text().strip().splitlines()
        if lines:
            try:
                if json.loads(lines[-1]).get("date") == day:
                    return False
            except json.JSONDecodeError:
                pass
    row = {"date": day, "score": round(float(data.weighted_score), 4), "headlines": len(data.headlines),
           "provider_scored": data.provider_scored, "sources": list(data.sources)}
    if social:
        row["social"] = social
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    return True


def load_sentiment_history(ticker: str, days: int = 90) -> list[dict[str, Any]]:
    """Stored daily snapshots for ``ticker``, oldest first, at most ``days`` of them."""
    path = _history_path(ticker)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "date" in item and "score" in item:
            rows.append(item)
    return rows[-days:]


def fetch_news_sentiment(ticker: str) -> NewsSentimentData:
    """
    Collect headlines from every source in ``SOURCES`` and score them.

    Never raises. Each source fails independently (recorded in ``source_errors``);
    when every source fails the result has ``available=False``.
    """
    symbol = ticker.strip().upper()
    sources = all_sources()
    if not sources:
        return NewsSentimentData(ticker=symbol, available=False, error="No news sources are configured")

    collected: list[Headline] = []
    answered: list[str] = []
    errors: dict[str, str] = {}
    for source in sources:
        try:
            collected.extend(_fetch_source(source.name, symbol))
            answered.append(source.name)
        except Exception as exc:  # noqa: BLE001 - provider failures become unavailable data
            logger.warning("News source %s failed for %s: %s", source.name, symbol, exc)
            errors[source.name] = _short_error(exc)

    if not answered:
        detail = "; ".join(f"{name}: {err}" for name, err in errors.items())
        return NewsSentimentData(
            ticker=symbol,
            available=False,
            error=f"News request failed ({detail})",
            source_errors=errors,
        )

    data = summarize_headlines(symbol, collected)
    data.sources = answered
    data.source_errors = errors
    return data
