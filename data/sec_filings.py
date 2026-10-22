"""The latest 10-K annual report of a company: download, section extraction and a small on-disk cache.

``analysis/filing_text.py`` holds the pure parsers; this module does the I/O. Only the extracted sections (12 KB
each) and the findings are stored, never the whole document, in ``data/cache/competitive/``. Requests share the
SEC throttle of ``data/sec_edgar.py`` (the same lock, so filings and company facts together stay under 8 requests
a second) and need the contact email in the User-Agent. Foreign companies file a 20-F or nothing: the notes then
say so instead of guessing (decision_log.md D-34).
"""

from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config.settings as settings
from analysis.filing_text import (
    customer_concentration,
    excerpt,
    html_to_text,
    named_competitors,
    no_major_customer,
    normalize_company_name,
    sentences,
    sourcing_sentences,
    split_items,
    subsections,
)
from data.sec_edgar import (
    MIN_REQUEST_INTERVAL,
    SUBMISSIONS_URL,
    SecConfigError,
    SecDataError,
    _get_json,
    _last_request,
    _rate_lock,
    has_sec_access,
    sec_cache_dir,
    user_agent,
)

logger = logging.getLogger(__name__)

TENK_FORMS = ("10-K", "10-KT")  # not 10-K/A: an amendment is often only Part III
INDEX_MAX_AGE_DAYS = 7
DOWNLOAD_TIMEOUT = 20
MAX_DOCUMENT_BYTES = 25_000_000
KEEP_FILINGS = 2  # extracts kept per company
MAX_ITEMS = 5  # passages kept per finding
MIN_DOWNLOAD_SECONDS = 2.0  # not worth starting a download with less time than this left in the budget

NO_ACCESS = "Set the SEC contact email to read the 10-K"
NO_TENK = "No 10-K: foreign companies file a 20-F or nothing with the SEC"
NOT_READ_YET = "The 10-K is still being read; analyze again in a minute"


def competitive_dir() -> Path:
    path = settings.DATA_DIR / "competitive"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class FilingNotes:
    """What the latest 10-K says about competitors, suppliers and large customers (all optional)."""

    ref: dict[str, Any] | None = None  # accession, filed, period, primary_doc, url
    competition: str = ""
    suppliers: str = ""
    customers: str = ""
    competitors: list[dict[str, Any]] = field(default_factory=list)  # name, ticker
    sourcing: list[str] = field(default_factory=list)
    concentration: list[dict[str, Any]] = field(default_factory=list)  # kind, n, pct, sentence
    no_major_customer: str | None = None
    raw_materials: str | None = None
    error: str | None = None  # why there are no notes (not stored)
    pending: bool = False  # the document is still being read (not stored)

    @property
    def found(self) -> bool:
        return self.ref is not None and self.error is None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("error"), out.pop("pending")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilingNotes":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("error", "pending")}
        return cls(**known)


# ------------------------------------------------------------------ the filing index


def latest_10k(cik: int, max_age_days: float = INDEX_MAX_AGE_DAYS) -> dict[str, Any] | None:
    """The newest 10-K (or 10-KT) of a company: accession, filed, period, primary_doc, url; None when it has none.

    The recent-filings index is cached for a week. Raises ``SecDataError`` / ``SecConfigError``.
    """
    path = competitive_dir() / f"tenk_index_{cik}.json"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        try:
            return json.loads(path.read_text()).get("ref")
        except (OSError, ValueError):
            pass
    recent = (_get_json(SUBMISSIONS_URL.format(cik=cik)).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    ref = None
    for i, form in enumerate(forms):
        if form in TENK_FORMS:
            accession, doc = recent["accessionNumber"][i], recent["primaryDocument"][i]
            ref = {
                "accession": accession, "filed": recent["filingDate"][i],
                "period": (recent.get("reportDate") or [None] * len(forms))[i] or None, "primary_doc": doc,
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc}",
            }
            break
    path.write_text(json.dumps({"ref": ref}))
    return ref


def _get_bytes(url: str, timeout: float = DOWNLOAD_TIMEOUT, max_bytes: int = MAX_DOCUMENT_BYTES) -> bytes:
    """One document through the shared SEC throttle, with one retry. Raises ``SecDataError``."""
    headers = {"User-Agent": user_agent(), "Accept-Encoding": "gzip"}
    last: Exception | None = None
    for attempt in range(2):
        with _rate_lock:
            wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request[0])
            if wait > 0:
                time.sleep(wait)
            _last_request[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
                body = resp.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise SecDataError(f"SEC document is larger than {max_bytes // 1_000_000} MB: {url}")
                return gzip.decompress(body) if resp.headers.get("Content-Encoding") == "gzip" else body
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                raise SecConfigError("SEC refused the request (403); check the contact email") from exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise SecDataError(f"SEC request failed ({exc.code}) for {url}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        time.sleep(1.0)
    raise SecDataError(f"SEC request failed for {url}: {last}")


def fetch_document(ref: dict[str, Any]) -> str:
    """The 10-K's primary document as HTML text."""
    return _get_bytes(ref["url"]).decode("utf-8", errors="replace")


def title_map() -> dict[str, str]:
    """Normalized company title -> ticker, from the cached SEC ticker list (empty when it was never downloaded)."""
    path = sec_cache_dir() / "company_tickers.json.gz"
    try:
        with gzip.open(path, "rt") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    out: dict[str, str] = {}
    for row in raw.values():
        out.setdefault(normalize_company_name(row.get("title", "")), str(row["ticker"]).upper().replace(".", "-"))
    out.pop("", None)
    return out


# ------------------------------------------------------------------ reading a report


def read_report(html: str, ref: dict[str, Any], own_names: tuple[str, ...] = ()) -> FilingNotes:
    """Sections and findings of one 10-K document."""
    text = html_to_text(html)
    spans = split_items(text)
    business = spans.get("1")
    body = text[business[0]:business[1]] if business and business[1] - business[0] > 5000 else text
    sections = subsections(body)
    risks = spans.get("1A")
    risk_text = text[risks[0]:risks[1]] if risks else ""

    sourcing: list[str] = []
    for part in (body, risk_text):  # the business description first: it is the company's own words
        for sentence in sourcing_sentences(part):
            if sentence not in sourcing:
                sourcing.append(sentence)
    supply = sections.get("suppliers", "")
    first = next(iter(sentences(supply)), None) if supply else None
    return FilingNotes(
        ref=ref,
        competition=sections.get("competition", ""),
        suppliers=supply,
        customers=sections.get("customers", ""),
        competitors=named_competitors(sections.get("competition", ""), title_map(), own_names),
        sourcing=[excerpt(s, 400) for s in sourcing[:MAX_ITEMS]],
        concentration=[c.to_dict() for c in customer_concentration(text)[:MAX_ITEMS]],
        no_major_customer=no_major_customer(text),
        raw_materials=excerpt(first) if first else None,
    )


def _prune(cik: int) -> None:
    files = sorted(competitive_dir().glob(f"tenk_{cik}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[KEEP_FILINGS:]:
        old.unlink(missing_ok=True)


def tenk_notes(cik: int | None, deadline: float | None = None, name: str | None = None, ticker: str | None = None) -> FilingNotes:
    """Notes from the latest 10-K, read once per filing and cached by its accession number.

    Never raises: when there are no notes, ``error`` says why and what to do. ``deadline`` is a
    ``time.monotonic()`` time; with too little left, a document that is not cached is not started (``pending``).
    """
    if not has_sec_access():
        return FilingNotes(error=NO_ACCESS)
    if cik is None:
        return FilingNotes(error=NO_TENK)
    try:
        ref = latest_10k(cik)
        if ref is None:
            return FilingNotes(error=NO_TENK)
        path = competitive_dir() / f"tenk_{cik}_{ref['accession']}.json"
        if path.exists():
            try:
                return FilingNotes.from_dict(json.loads(path.read_text()))
            except (OSError, ValueError, TypeError):
                path.unlink(missing_ok=True)
        if deadline is not None and deadline - time.monotonic() < MIN_DOWNLOAD_SECONDS:
            return FilingNotes(ref=ref, error=NOT_READ_YET, pending=True)
        notes = read_report(fetch_document(ref), ref, tuple(n for n in (name, ticker) if n))
        path.write_text(json.dumps(notes.to_dict()))
        _prune(cik)
        return notes
    except (SecDataError, SecConfigError, OSError) as exc:
        logger.info("10-K notes unavailable for CIK %s: %s", cik, exc)
        return FilingNotes(error=f"The 10-K could not be read ({type(exc).__name__}); analyze again later")
    except Exception as exc:  # noqa: BLE001 - a parser or index surprise must not read as "still being read"
        logger.warning("10-K notes failed for CIK %s: %s: %s", cik, type(exc).__name__, exc)
        return FilingNotes(error="The 10-K could not be read; analyze again later")
