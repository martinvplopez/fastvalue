"""Company reports from SEC EDGAR for the company page: a financial overview and the latest filings.

Two public functions, both of which return a status instead of raising for expected conditions (no SEC contact
email, a ticker the SEC does not know, a network failure):

- ``financial_overview(ticker, period)``: up to ten fiscal years (plus an LTM column, the last twelve months) or the
  last twelve quarters, with the metrics the owner checks as a value investor. Every figure comes from the XBRL
  facts of the company's 10-K and 10-Q filings.
- ``recent_filings(ticker, limit)``: the latest 10-K, 10-Q, 8-K earnings releases and investor presentations,
  proxy statements (DEF 14A, where pay and incentives are) and, for foreign filers, 20-F, 40-F and 6-K, with links.

Decisions (decision_log.md D-37):

- One definition of every metric. The XBRL values of each period are mapped to the Yahoo-style statements by
  ``data.sec_edgar.statements_from_values`` (the function the value study uses, same concept lists) and the
  metrics come from ``analysis.fundamental_metrics.compute_owner_metrics`` (FCF per ``settings.FCF_DEFINITION``,
  ROIC, reinvestment, EV, working capital, net dilution). Nothing is recomputed with a second formula.
- Quarters are built in this module from the same companyfacts JSON, in a separate extract
  (``data/cache/sec/reports_v2_<cik>.parquet``), so the annual extract the point-in-time study reads is untouched.
  A discrete three-month figure is used when the filing tags one; otherwise it is worked out from year-to-date
  figures (cash flow statements are year to date; the fourth quarter is the fiscal year minus nine months) and the
  column is marked ``derived``. Balance sheet items are instants.
- Ratios use trailing twelve months. A quarter column shows the quarter's own flows, but ROIC, EV/FCF, PER and the
  other ratios use the twelve months that end at the quarter, because a quarter alone is too noisy and seasonal.
- Valuation needs a price: the split-adjusted close on the period end (``data.price_history``) times the diluted
  share count of the period, restated for later splits. The LTM column uses the latest close and, when the filing
  has one, the latest cover-page share count.
- A value the filings do not tag stays ``None`` and is listed in the notes (decision_log.md D-01). Nothing is
  estimated.
- Call transcripts and investor relations pages are not on EDGAR. This module only links to EDGAR's own full-text
  search and the company's own website; it does not read third-party transcript sites.

Requests share the SEC throttle of ``data/sec_edgar.py`` (8 a second) and need the contact email in the
User-Agent. Filing indexes are cached per CIK for a day; exhibit lists are cached for good (a filing never
changes). A network failure is never cached.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

import config.settings as settings
from analysis.fundamental_metrics import FCF_DEFINITIONS, OwnerMetrics, compute_owner_metrics, reinvestment_explanation
from data.cache_utils import ttl_cache
from data.financial_statements import FinancialStatements
from data.fundamentals import FundamentalsData
from data.price_history import fetch_price_history, split_events
from data.sec_edgar import (
    ANCHOR_CONCEPTS,
    ANNUAL_DAYS,
    ANNUAL_FORMS,
    AMORTIZATION,
    BALANCE_ROWS,
    DEBT_CONCEPTS,
    DEPRECIATION,
    DURATION_CONCEPTS,
    FACTS_URL,
    INCOME_ROWS,
    INSTANT_CONCEPTS,
    SHARE_ROWS,
    SHARES_OUTSTANDING_DEI,
    CompanyFacts,
    SecConfigError,
    SecDataError,
    _fresh,
    _get_json,
    _split_product,
    has_sec_access,
    is_financial_sic,
    load_submissions,
    load_ticker_map,
    sec_cache_dir,
    statements_from_values,
)
from data.sec_filings import NO_ACCESS, _get_bytes
from data.sec_margins import MARGIN_CONCEPTS

logger = logging.getLogger(__name__)

PRICE_START = "2008-06-01"  # the same start as the value study, so both share one price cache file
FACTS_MAX_AGE_DAYS = 1
SUBMISSIONS_MAX_AGE_DAYS = 1
TICKER_MAP_MAX_AGE_DAYS = 30
MODEL_TTL_SECONDS = 600
MAX_ANNUAL_COLUMNS = 10
MAX_QUARTER_COLUMNS = 12
HISTORY_YEARS_FOR_RATIOS = 5  # years of same-period history given to the metrics (incremental ROIC, working capital)
PRICE_MAX_GAP_DAYS = 10  # a close further than this from the period end is not used
EXHIBIT_FILINGS = 12  # most recent 8-Ks with Item 2.02, 7.01 or 8.01 whose exhibits are listed (press releases)
DAYS_PER_QUARTER = 91.3
VERDICT_BAND = 0.20  # a ratio within 20% of its reference reads "in line" (presentation only, not a scoring rule)

# Market references of the framework (README "Investing framework", agents/fundamental_agent.py).
REF_EV_FCF = 16.5
REF_PER = 16.0
REF_ROIC = 0.135
HIGH_ROIC = 0.15
GOOD_REINVESTMENT = 0.20
HIGH_SBC_TO_REVENUE = 0.06
NET_DILUTION_WORSE = 0.015  # shareholder agent: above +1.5% of market cap costs points
REF_ROE = 0.13  # fundamentals agent: 13% and above scores well for banks and insurers
REF_ROA = 0.010

# Extra concepts the statements do not carry, with the same priority rule (first with a value wins).
EXTRA_CONCEPTS: dict[str, tuple[str, ...]] = {
    "gross_profit": ("GrossProfit",),
    "cogs": MARGIN_CONCEPTS["cogs"],
    "dividends": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "rd": ("ResearchAndDevelopmentExpense",),
}
PER_SHARE_CONCEPTS = frozenset(EXTRA_CONCEPTS["eps"])
SHARE_CONCEPTS = frozenset(c for r in SHARE_ROWS for c in INCOME_ROWS[r])  # weighted averages: averaged, not added
_EXTRA_FLOWS = frozenset(c for concepts in EXTRA_CONCEPTS.values() for c in concepts)
FLOW_CONCEPTS = DURATION_CONCEPTS | {DEPRECIATION, AMORTIZATION} | _EXTRA_FLOWS
WANTED_CONCEPTS = FLOW_CONCEPTS | INSTANT_CONCEPTS
ANCHORS = ANCHOR_CONCEPTS
FACT_COLUMNS = ["concept", "unit", "start", "end", "val", "filed", "form", "accn"]
UNITS = ("USD", "shares", "USD/shares")

FOREIGN_ANNUAL = {"20-F", "20-F/A", "40-F", "40-F/A"}
NOT_SEC_FILER = (
    "This company is not in the SEC's list of filers. Foreign listings, funds and indexes file with their own "
    "regulators; FastValue reads SEC filings only."
)
FOREIGN_FILER = (
    "This is a foreign filer (20-F or 40-F). Its statements use IFRS tags and its own currency, and the overview "
    "reads US GAAP tags only. The filings list still works."
)
NO_US_GAAP = (
    "The filings carry no US GAAP statement tags, so the overview has nothing to read. The filings list still works."
)


# ------------------------------------------------------------------ status payloads


def _edgar_url(cik: int, form_type: str = "") -> str:
    return (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik:010d}"
            f"&type={form_type}&dateb=&owner=include&count=40")


def _base(ticker: str, cik: int | None = None, name: str | None = None) -> dict[str, Any]:
    return {"ticker": ticker, "name": name, "cik": cik, "status": "ok", "message": None, "fix": None}


def _no_urls(text: str) -> str:
    """An error text for the page: a raw address runs past the card and means nothing to the reader."""
    return re.sub(r"https?://\S+", "the SEC", text)


def _failure(payload: dict[str, Any], status: str, message: str, fix: str | None = None) -> dict[str, Any]:
    payload.update(status=status, message=message, fix=fix)
    return payload


def _no_access(payload: dict[str, Any]) -> dict[str, Any]:
    return _failure(
        payload, "no_access",
        "The SEC asks for a contact email with every request, so filings cannot be read yet.", NO_ACCESS,
    )


@ttl_cache(3600, maxsize=4)
def _ticker_map() -> dict[str, int]:
    return load_ticker_map(max_age_days=TICKER_MAP_MAX_AGE_DAYS)


def _resolve(ticker: str, payload: dict[str, Any]) -> int | None:
    """The CIK of a ticker, or None with ``payload`` set to the matching failure status."""
    if not has_sec_access():
        _no_access(payload)
        return None
    try:
        cik = _ticker_map().get(ticker.upper().replace(".", "-"))
    except SecConfigError:
        _no_access(payload)
        return None
    except (SecDataError, OSError, ValueError) as exc:
        _failure(payload, "error", f"The SEC company list could not be loaded ({_reason(exc)}).", "Try again in a minute.")
        return None
    if cik is None:
        _failure(payload, "not_sec_filer", NOT_SEC_FILER)
        return None
    payload["cik"] = cik
    return cik


def _reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


# ------------------------------------------------------------------ SEC downloads


def _submissions(cik: int) -> dict[str, Any]:
    """The company's submissions JSON (name, SIC code, website and recent filings), cached for a day."""
    return load_submissions(cik, SUBMISSIONS_MAX_AGE_DAYS)


def extract_report_facts(raw: dict[str, Any]) -> pd.DataFrame:
    """Facts of the 10-K and 10-Q filings that the overview uses, one row per filed value."""
    rows: list[tuple] = []
    for taxonomy, concepts in (raw.get("facts") or {}).items():
        for concept, body in concepts.items():
            if not ((taxonomy == "us-gaap" and concept in WANTED_CONCEPTS) or (taxonomy == "dei" and concept == SHARES_OUTSTANDING_DEI)):
                continue
            for unit, entries in (body.get("units") or {}).items():
                if unit not in UNITS:
                    continue
                for e in entries:
                    form = e.get("form") or ""
                    if e.get("val") is None or not e.get("end") or not e.get("filed") or not form.startswith("10-"):
                        continue
                    rows.append((concept, unit, e.get("start"), e["end"], float(e["val"]), e["filed"], form, e.get("accn") or ""))
    frame = pd.DataFrame(rows, columns=FACT_COLUMNS)
    for col in ("start", "end", "filed"):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame.dropna(subset=["end", "filed"]).reset_index(drop=True)


def load_report_facts(cik: int, max_age_days: float = FACTS_MAX_AGE_DAYS) -> pd.DataFrame:
    """Facts of one company (cached for a day). Raises ``SecDataError`` / ``SecConfigError``; failures are not cached."""
    path = sec_cache_dir() / f"reports_v2_{cik:010d}.parquet"
    if _fresh(path, max_age_days):
        return pd.read_parquet(path)
    frame = extract_report_facts(_get_json(FACTS_URL.format(cik=cik)))
    frame.to_parquet(path, index=False)
    return frame


def _price_frame(symbol: str) -> pd.DataFrame | None:
    """Daily history with split-adjusted close and split events; None when Yahoo has no prices."""
    try:
        return fetch_price_history([symbol], PRICE_START).get(symbol)
    except Exception as exc:  # noqa: BLE001 - valuation rows stay empty, the statements still show
        logger.info("No prices for %s: %s", symbol, exc)
        return None


# ------------------------------------------------------------------ fiscal calendar and flows


@dataclass
class _Year:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class _Flows:
    """One concept in one fiscal year: the full year, year-to-date figures by quarter slot, discrete quarters."""

    fy: float | None = None
    ytd: dict[int, float] = field(default_factory=dict)
    q: dict[int, float] = field(default_factory=dict)

    def ytd_value(self, k: int) -> tuple[float, bool] | None:
        """Year-to-date figure through quarter ``k`` and whether it was added up from quarters."""
        if k == 4 and self.fy is not None:
            return self.fy, False
        if k in self.ytd:
            return self.ytd[k], False
        quarters = [self.q.get(i) for i in range(1, k + 1)]
        if all(v is not None for v in quarters):
            return float(sum(quarters)), True
        return None

    def quarter(self, k: int, average: bool) -> tuple[float, bool] | None:
        """The discrete quarter and whether it had to be worked out from year-to-date figures."""
        if k in self.q:
            return self.q[k], False
        current = self.ytd_value(k)
        if current is None:
            return None
        if k == 1:
            return current[0], current[1]
        earlier = self.ytd_value(k - 1)
        if earlier is None:
            return None
        if average:  # weighted average shares: k * (average to date) - (k - 1) * (average a quarter earlier)
            return k * current[0] - (k - 1) * earlier[0], True
        return current[0] - earlier[0], True


def fiscal_years(frame: pd.DataFrame) -> list[_Year]:
    """Fiscal years from the annual anchor facts, oldest first, plus the year in progress."""
    days = (frame["end"] - frame["start"]).dt.days
    anchors = frame[frame["concept"].isin(ANCHORS) & frame["form"].isin(ANNUAL_FORMS) & days.between(*ANNUAL_DAYS)]
    if anchors.empty:
        return []
    by_end = anchors.groupby("end")["start"].agg(lambda s: s.value_counts().idxmax())
    years = [_Year(pd.Timestamp(s), pd.Timestamp(e)) for e, s in sorted(by_end.items())]
    for prev, cur in zip(years, years[1:]):  # a fiscal year starts the day after the previous one ends
        cur.start = prev.end + pd.Timedelta(days=1)
    nxt = years[-1].end + pd.Timedelta(days=1)
    years.append(_Year(nxt, nxt + pd.Timedelta(days=364)))
    return years


class _Calendar:
    """Places a (start, end) period into a fiscal year and quarter slot."""

    def __init__(self, years: list[_Year]):
        self.years = years
        self._limits = [y.end + pd.Timedelta(days=10) for y in years]

    def slot(self, end: pd.Timestamp) -> tuple[int, int] | None:
        i = bisect.bisect_left(self._limits, end)
        if i >= len(self.years):
            return None
        span = (end - self.years[i].start).days
        k = round(span / DAYS_PER_QUARTER)
        if not 1 <= k <= 4 or abs(span - k * DAYS_PER_QUARTER) > 25:
            return None
        return i, k


def _collect_flows(frame: pd.DataFrame, calendar: _Calendar) -> tuple[dict[str, dict[int, _Flows]], dict[tuple[int, int], pd.Timestamp]]:
    """Flows per concept and fiscal year, plus the quarter end dates (taken from the anchor concepts)."""
    flows: dict[str, dict[int, _Flows]] = {}
    end_votes: dict[tuple[int, int], Counter] = {}
    durations = frame[frame["start"].notna() & frame["concept"].isin(FLOW_CONCEPTS)]
    latest = durations.sort_values("filed", kind="stable").groupby(["concept", "unit", "start", "end"], sort=False).tail(1)
    for concept, start, end, val in zip(latest["concept"], latest["start"], latest["end"], latest["val"]):
        place = calendar.slot(end)
        if place is None:
            continue
        i, k = place
        year = calendar.years[i]
        days = (end - start).days
        entry = flows.setdefault(concept, {}).setdefault(i, _Flows())
        near_start = abs((start - year.start).days) <= 10
        if k == 4 and near_start and days >= ANNUAL_DAYS[0]:
            entry.fy = val
        elif near_start and abs(days - k * DAYS_PER_QUARTER) <= 25 and k < 4:
            entry.ytd[k] = val
            if k == 1:
                entry.q[1] = val
        elif 80 <= days <= 100:
            entry.q[k] = val
        else:
            continue
        if concept in ANCHORS:
            end_votes.setdefault((i, k), Counter())[end] += 1
    ends = {key: votes.most_common(1)[0][0] for key, votes in end_votes.items()}
    return flows, ends


def _ttm(entries: dict[int, _Flows], i: int, k: int, average: bool) -> float | None:
    """Value for the twelve months that end with quarter ``k`` of fiscal year ``i``."""
    cur = entries.get(i)
    if cur is None:
        return None
    if k == 4:
        year = cur.ytd_value(4)
        if year is not None:
            return year[0]
    elif not average:
        prev = entries.get(i - 1)
        if prev is not None:
            now, then, prev_year = cur.ytd_value(k), prev.ytd_value(k), prev.ytd_value(4)
            if now and then and prev_year:
                return prev_year[0] + now[0] - then[0]
    quarters: list[float] = []
    for step in range(4):  # the four discrete quarters that end here
        slot = k - step
        year_index = i
        while slot < 1:
            slot += 4
            year_index -= 1
        entry = entries.get(year_index)
        got = entry.quarter(slot, average) if entry is not None else None
        if got is None:
            return None
        quarters.append(got[0])
    return sum(quarters) / 4 if average else sum(quarters)


@dataclass
class _Model:
    """Everything the payloads need about one company, built once and kept for a few minutes."""

    ticker: str
    cik: int
    name: str | None
    sic: int | None
    is_financial: bool
    quarters: list[tuple[pd.Timestamp, int, int]]  # (quarter end, fiscal year index, slot), oldest first
    fiscal_years: list[_Year]
    values_ttm: dict[tuple[str, pd.Timestamp], float]
    values_q: dict[tuple[str, pd.Timestamp], float]
    derived: dict[pd.Timestamp, bool]
    st_ttm: FinancialStatements
    st_q: FinancialStatements
    filings: dict[pd.Timestamp, dict[str, Any]]  # period end -> form, filed, url
    prices: pd.Series | None
    cover_shares: float | None
    notes: list[str] = field(default_factory=list)

    def chain(self, end: pd.Timestamp, length: int = HISTORY_YEARS_FOR_RATIOS) -> list[pd.Timestamp]:
        """The same period in earlier fiscal years: ``end``, a year before, two years before, and so on."""
        index = {(i, k): e for e, i, k in self.quarters}
        place = next(((i, k) for e, i, k in self.quarters if e == end), None)
        if place is None:
            return [end]
        i, k = place
        out = []
        for back in range(length):
            e = index.get((i - back, k))
            if e is None:
                break
            out.append(e)
        return out


def _first_value(values: dict[tuple[str, pd.Timestamp], float], concepts: tuple[str, ...], end: pd.Timestamp) -> float | None:
    for c in concepts:
        v = values.get((c, end))
        if v is not None:
            return v
    return None


def _build_model(symbol: str, cik: int) -> _Model:
    """Download (or read from the cache) and organise one company's facts. Raises ``SecDataError`` / ``SecConfigError``."""
    subs = _submissions(cik)
    sic = int(subs["sic"]) if str(subs.get("sic") or "").isdigit() else None
    forms = set(_recent_arrays(subs).get("form") or [])
    if not forms & {"10-K", "10-Q", "10-KT"} and forms & FOREIGN_ANNUAL:
        raise SecDataError(FOREIGN_FILER)  # skip the large IFRS facts download
    frame = load_report_facts(cik)
    if frame.empty or not frame["concept"].isin(ANCHORS).any():
        raise SecDataError(NO_US_GAAP)

    prices = _price_frame(symbol)
    splits = split_events(prices) if prices is not None else pd.Series(dtype=float)
    today = pd.Timestamp.today().normalize()
    frame = _restate_for_splits(frame, splits, today)

    calendar = _Calendar(fiscal_years(frame))
    flows, quarter_end = _collect_flows(frame, calendar)

    quarters = sorted(((end, i, k) for (i, k), end in quarter_end.items()), key=lambda t: t[0])
    values_ttm: dict[tuple[str, pd.Timestamp], float] = {}
    values_q: dict[tuple[str, pd.Timestamp], float] = {}
    derived_flags: dict[pd.Timestamp, bool] = {}
    for concept, entries in flows.items():
        average = concept in SHARE_CONCEPTS
        for end, i, k in quarters:
            ttm = _ttm(entries, i, k, average)
            if ttm is not None:
                values_ttm[(concept, end)] = ttm
            entry = entries.get(i)
            got = entry.quarter(k, average) if entry is not None else None
            if got is not None:
                values_q[(concept, end)] = got[0]
                if got[1] and concept in ANCHORS:
                    derived_flags[end] = True
    instants = frame[frame["start"].isna() & frame["concept"].isin(INSTANT_CONCEPTS)]
    latest = instants.sort_values("filed", kind="stable").groupby(["concept", "end"], sort=False).tail(1)
    for concept, end, val in zip(latest["concept"], latest["end"], latest["val"]):
        values_ttm[(concept, end)] = val
        values_q[(concept, end)] = val

    ends = [e for e, _, _ in quarters]
    notes: list[str] = [] if prices is not None else ["No price history was found, so the valuation rows are empty."]
    mixed = _drop_broader_tags(values_q, values_ttm, ends, [e for e, _, k in quarters if k == 4])
    if mixed:
        notes.append("Working capital is empty in quarters where the 10-Q only gives a broader figure than the 10-K: "
                     + ", ".join(mixed) + ". Mixing them would make the line jump.")
    stale = _stale_note(frame, subs)
    if stale:
        notes.append(stale)
    st_ttm = statements_from_values(symbol, values_ttm, ends)
    st_q = statements_from_values(symbol, values_q, ends)

    cf = CompanyFacts(ticker=symbol, cik=cik, name=subs.get("name"), sic=sic, sic_description=subs.get("sicDescription"), facts=frame)
    cover = cf.shares_outstanding_asof(frame["filed"].max())  # the newest filing's cover-page count
    cover_shares = cover[0] if cover else None  # already on today's basis: the frame was restated above

    recent = _recent_arrays(subs)
    return _Model(
        ticker=symbol, cik=cik, name=subs.get("name"), sic=sic, is_financial=is_financial_sic(sic), quarters=quarters,
        fiscal_years=calendar.years, values_ttm=values_ttm, values_q=values_q, derived=derived_flags, st_ttm=st_ttm,
        st_q=st_q, filings=_period_filings(frame, ends, recent, cik),
        prices=None if prices is None else prices["close"], cover_shares=cover_shares,
        notes=notes,
    )


def _stale_note(frame: pd.DataFrame, subs: dict[str, Any]) -> str | None:
    """A note when the newest 10-K or 10-Q is not in the SEC's XBRL data feed yet (the feed lags a little)."""
    recent = _recent_arrays(subs)
    newest = next(((f, d) for f, d in zip(recent.get("form") or [], recent.get("filingDate") or [])
                   if f in ("10-K", "10-Q", "10-KT")), None)
    if newest is None or frame.empty:
        return None
    in_feed = frame["filed"].max()
    if pd.Timestamp(newest[1]) - in_feed > pd.Timedelta(days=2):
        return (f"The newest {newest[0]} (filed {newest[1]}) is not in the SEC's XBRL data yet, so the latest figures "
                f"come from the report filed {in_feed:%Y-%m-%d}. Open the filing for the newest numbers.")
    return None


def _drop_broader_tags(values_q: dict, values_ttm: dict, ends: list[pd.Timestamp], year_ends: list[pd.Timestamp]) -> list[str]:
    """Keep each working capital line on the tag the 10-K uses.

    A 10-K may tag trade payables where the 10-Qs only tag payables and accrued liabilities together (a lower
    priority concept). Taking whichever tag exists would make the line jump between a year end and the next
    quarter, so a quarter end whose tag is broader than the latest year end's is left empty instead.
    """
    dropped: list[str] = []
    if not year_ends:
        return dropped
    for row in ("Accounts Payable", "Accounts Receivable", "Inventory"):
        concepts = BALANCE_ROWS[row]
        index = {c: i for i, c in enumerate(concepts)}
        reference = next((index[c] for c in concepts if (c, year_ends[-1]) in values_ttm), None)
        if reference is None:
            continue
        for end in ends:
            if end in year_ends:
                continue
            used = next((index[c] for c in concepts if (c, end) in values_ttm), None)
            if used is None or used <= reference:
                continue
            for c in concepts[reference + 1:]:
                values_q.pop((c, end), None)
                values_ttm.pop((c, end), None)
            if row.lower() not in dropped:
                dropped.append(row.lower())
    return dropped


def _restate_for_splits(frame: pd.DataFrame, splits: pd.Series, today: pd.Timestamp) -> pd.DataFrame:
    """Put share counts and per-share figures on today's share basis (a filing already restates for earlier splits)."""
    if splits.empty:
        return frame
    out = frame.copy()
    shares = out["concept"].isin(SHARE_CONCEPTS | {SHARES_OUTSTANDING_DEI, "CommonStockSharesOutstanding"})
    per_share = out["concept"].isin(PER_SHARE_CONCEPTS)
    if not (shares | per_share).any():
        return out
    factors = {d: _split_product(splits, d, today) for d in out.loc[shares | per_share, "filed"].unique()}
    out.loc[shares, "val"] = out.loc[shares, "val"] * out.loc[shares, "filed"].map(factors)
    out.loc[per_share, "val"] = out.loc[per_share, "val"] / out.loc[per_share, "filed"].map(factors)
    return out


def _recent_arrays(subs: dict[str, Any]) -> dict[str, list]:
    return (subs.get("filings") or {}).get("recent") or {}


def _filing_url(cik: int, accession: str, document: str | None) -> str:
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
    return f"{base}/{document}" if document else f"{base}/{accession}-index.htm"


def _period_filings(frame: pd.DataFrame, ends: list[pd.Timestamp], recent: dict[str, list], cik: int) -> dict[pd.Timestamp, dict[str, Any]]:
    """For each period end, the first 10-K or 10-Q that reported it: form, filing date and link."""
    documents = dict(zip(recent.get("accessionNumber") or [], recent.get("primaryDocument") or []))
    wanted = frame[frame["end"].isin(ends) & frame["concept"].isin(ANCHORS)].sort_values("filed", kind="stable")
    first = wanted.groupby("end", sort=False).head(1)
    out: dict[pd.Timestamp, dict[str, Any]] = {}
    for end, filed, form, accn in zip(first["end"], first["filed"], first["form"], first["accn"]):
        out[end] = {"form": form, "filed": filed.strftime("%Y-%m-%d"),
                    "url": _filing_url(cik, accn, documents.get(accn)) if accn else None}
    return out


@ttl_cache(MODEL_TTL_SECONDS, maxsize=32)
def _model(symbol: str, cik: int) -> _Model:
    return _build_model(symbol, cik)


# ------------------------------------------------------------------ one column of numbers


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _price_at(prices: pd.Series | None, end: pd.Timestamp) -> float | None:
    if prices is None or prices.empty:
        return None
    window = prices.loc[:end]
    if window.empty or (end - window.index[-1]).days > PRICE_MAX_GAP_DAYS:
        return None
    return float(window.iloc[-1])


def _slice(st: FinancialStatements, ends: list[pd.Timestamp]) -> FinancialStatements:
    """The statements restricted to ``ends`` (newest first)."""
    def pick(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        cols = [c for c in frame.columns if pd.Timestamp(c) in set(ends)]
        return frame[sorted(cols, reverse=True)]

    return FinancialStatements(ticker=st.ticker, available=st.available, income=pick(st.income), balance=pick(st.balance),
                               cashflow=pick(st.cashflow), source=st.source)


def _leases(values: dict[tuple[str, pd.Timestamp], float], end: pd.Timestamp) -> float | None:
    total = _first_value(values, DEBT_CONCEPTS["operating_lease"], end)
    if total is not None:
        return total
    parts = [_first_value(values, DEBT_CONCEPTS[k], end) for k in ("operating_lease_current", "operating_lease_noncurrent")]
    return None if all(p is None for p in parts) else float(sum(p for p in parts if p is not None))


def _abs(v: float | None) -> float | None:
    return None if v is None else abs(v)


def _column(model: _Model, end: pd.Timestamp, quarterly: bool, latest_price: bool = False) -> dict[str, float | None]:
    """Every figure of one column. Flows are the period's own (a quarter, a fiscal year or the last twelve months);
    ratios use the twelve months that end on ``end``."""
    st_show = model.st_q if quarterly else model.st_ttm
    values = model.values_q if quarterly else model.values_ttm
    chain = model.chain(end)
    before = chain[1] if len(chain) > 1 else None

    def inc(rows: str | tuple[str, ...], when: pd.Timestamp | None = end) -> float | None:
        return None if when is None else st_show.value("income", rows, when)

    def cash(rows: str | tuple[str, ...]) -> float | None:
        return st_show.value("cashflow", rows, end)

    revenue = inc("Total Revenue")
    gross = _first_value(values, EXTRA_CONCEPTS["gross_profit"], end)
    if gross is None:
        cogs = _first_value(values, EXTRA_CONCEPTS["cogs"], end)
        gross = revenue - cogs if revenue is not None and cogs is not None else None
    ebit = inc("EBIT")
    ocf = cash("Operating Cash Flow")
    capex = _abs(cash("Capital Expenditure"))
    shares = inc("Diluted Average Shares")
    shares_before = inc("Diluted Average Shares", before)
    sbc = cash("Stock Based Compensation")
    c: dict[str, float | None] = {
        "revenue": revenue, "gross_profit": gross, "ebit": ebit, "ebitda": inc("EBITDA"),
        "net_income": inc(("Net Income Common Stockholders", "Net Income")),
        "eps_diluted": _first_value(values, EXTRA_CONCEPTS["eps"], end),
        "revenue_growth": None if before is None else _growth(revenue, inc("Total Revenue", before)),
        "operating_cash_flow": ocf, "capex": capex,
        "depreciation_amortization": cash(("Depreciation And Amortization", "Depreciation Amortization Depletion")),
        "sbc": sbc, "acquisitions": _abs(cash(("Purchase Of Business", "Net Business Purchase And Sale"))),
        "buybacks": _abs(cash(("Repurchase Of Capital Stock", "Common Stock Payments"))),
        "dividends": _abs(_first_value(values, EXTRA_CONCEPTS["dividends"], end)),
        "shares_diluted": shares, "shares_change": _growth(shares, shares_before),
        "operating_leases": _leases(values, end),
    }
    if model.is_financial:  # EBIT adds back interest expense, which is a bank's main cost
        c["ebit"], c["ebitda"], ebit = None, None, None
    c["gross_margin"], c["ebit_margin"] = _div(gross, revenue), _div(ebit, revenue)
    c["net_margin"] = _div(c["net_income"], revenue)
    c["sbc_to_revenue"] = _div(sbc, revenue) if revenue and revenue > 0 else None
    c["reported_fcf"] = ocf - capex if ocf is not None and capex is not None else None

    metrics = _metrics(model, end, quarterly, latest_price, shares)
    c["price"] = metrics[1]
    m = metrics[0]
    fin = model.is_financial
    c["_metrics_ok"] = 1.0 if m is not None else 0.0
    if m is not None:
        c.update({
            "cash_and_sti": m.cash, "total_debt": m.total_debt, "equity": m.equity,
            "working_capital": m.working_capital, "invested_capital": None if fin else m.invested_capital,
            "roic": None if fin else m.roic, "roe": m.roe, "roa": m.roa,
            "reinvestment_rate": m.reinvestment_rate, "incremental_roic": m.incremental_roic,
            "_acq_reported": None if m.acquisitions_reported is None else float(m.acquisitions_reported),
            "_growth_floored": float(m.growth_capex_floored), "_growth_capex": m.growth_capex,
            "market_cap": m.market_cap, "enterprise_value": m.enterprise_value, "ev_to_fcf": m.ev_to_fcf,
            "pe": m.per, "pe_ex_cash": m.per_ex_cash, "fcf_yield": m.fcf_yield_on_ev, "ev_to_ebit": m.ev_to_ebit,
            "net_dilution": m.net_dilution_cost,
        })
        ttm_fcf = m.fcf
    else:
        ttm_fcf = None
    has_debt = c.get("total_debt") is not None and c.get("cash_and_sti") is not None
    c["net_debt"] = c["total_debt"] - c["cash_and_sti"] if has_debt and not fin else None
    c["ttm_fcf"] = ttm_fcf
    # The row's own FCF: the configured definition on twelve months, or the quarter's reported FCF for a quarter column.
    c["fcf"] = None if fin else (c["reported_fcf"] if quarterly else ttm_fcf)
    c["fcf_margin"] = _div(c["fcf"], revenue)
    rd = _first_value(values, EXTRA_CONCEPTS["rd"], end)
    c["rd"] = rd
    c["rd_to_fcf"] = _div(rd, c["fcf"]) if rd is not None and c["fcf"] is not None and c["fcf"] > 0 else None
    c["fcf_per_share"] = _div(c["fcf"], shares)
    return c


def _growth(now: float | None, before: float | None) -> float | None:
    if now is None or before is None or before <= 0:
        return None
    return now / before - 1.0


def _metrics(model: _Model, end: pd.Timestamp, quarterly: bool, latest_price: bool, shares_show: float | None) -> tuple[OwnerMetrics | None, float | None]:
    """The owner metrics of the twelve months ending at ``end`` and the price used for them."""
    st = _slice(model.st_ttm, model.chain(end))
    if end not in set(pd.Timestamp(c) for c in st.income.columns) or not st.available:
        return None, None
    price = None
    if model.prices is not None and not model.prices.empty:
        price = float(model.prices.iloc[-1]) if latest_price else _price_at(model.prices, end)
    shares = model.cover_shares if latest_price and model.cover_shares else shares_show
    f = FundamentalsData(
        ticker=model.ticker, available=True, sector="Financial Services" if model.is_financial else None,
        currency="USD", financial_currency="USD", current_price=price, shares_outstanding=shares,
    )
    m = compute_owner_metrics(st, f, price=price, today=end)
    if m.fiscal_year_end != end.strftime("%Y-%m-%d"):
        return None, price  # the column lacked income statement rows: do not show an older period's ratios
    return m, price


# ------------------------------------------------------------------ the financial overview


@dataclass(frozen=True)
class _Row:
    key: str
    label: str
    unit: str
    tip: str
    highlight: bool = False


SECTIONS: tuple[tuple[str, str, tuple[_Row, ...]], ...] = (
    ("income", "Income statement", (
        _Row("revenue", "Revenue", "money", "Total net sales or revenue as reported."),
        _Row("revenue_growth", "Revenue growth", "pct", "Change against the same period a year earlier (the previous fiscal year for yearly columns)."),
        _Row("gross_profit", "Gross profit", "money", "Revenue minus cost of revenue. Empty when the company does not report a cost of revenue."),
        _Row("gross_margin", "Gross margin", "pct", "Gross profit divided by revenue."),
        _Row("ebit", "EBIT", "money", "Operating profit before interest and tax: pre-tax income plus interest expense (the EBIT the value agents use)."),
        _Row("ebit_margin", "EBIT margin", "pct", "EBIT divided by revenue."),
        _Row("ebitda", "EBITDA", "money", "EBIT plus depreciation and amortization."),
        _Row("net_income", "Net income", "money", "Profit after tax and interest, to common shareholders."),
        _Row("net_margin", "Net margin", "pct", "Net income divided by revenue."),
        _Row("eps_diluted", "Diluted EPS", "per_share", "Reported earnings per share after dilution, on today's share basis (later stock splits restated)."),
    )),
    ("cash_flow", "Cash flow", (
        _Row("operating_cash_flow", "Operating cash flow", "money", "Cash the business generated from operations."),
        _Row("capex", "CapEx", "money", "Cash spent on property, plant and equipment, shown as a positive number."),
        _Row("fcf", "Free cash flow", "money", "", True),
        _Row("fcf_margin", "FCF margin", "pct", "Free cash flow divided by revenue."),
        _Row("fcf_per_share", "FCF per share", "per_share", "Free cash flow divided by diluted shares."),
        _Row("depreciation_amortization", "Depreciation and amortization", "money", "Non-cash charge for wear on assets and acquired intangibles."),
        _Row("sbc", "Stock-based compensation", "money", "Pay in shares and options. It dilutes owners, so the owner's net dilution check charges it."),
        _Row("acquisitions", "Acquisitions", "money", "Cash paid to buy businesses, net of cash acquired. Empty means no standard tag was found (the company may not report it or may use its own tag), not a reported zero."),
        _Row("rd", "R&D expense", "money", "Research and development, expensed. Information only: it sits outside the owner's reinvestment formula."),
        _Row("rd_to_fcf", "R&D / FCF", "pct", "R&D as a share of free cash flow. Information only, outside the owner's reinvestment formula and never scored."),
    )),
    ("balance", "Balance sheet", (
        _Row("cash_and_sti", "Cash and short-term investments", "money", "Cash plus short-term investments at the period end."),
        _Row("total_debt", "Total debt", "money", "Financial debt plus lease liabilities, as the owner's enterprise value counts it."),
        _Row("net_debt", "Net debt", "money", "Total debt minus cash and short-term investments. Negative means net cash."),
        _Row("operating_leases", "Operating leases", "money", "Lease liabilities for rented property and equipment (already inside total debt)."),
        _Row("equity", "Shareholders' equity", "money", "Book value owned by shareholders."),
        _Row("working_capital", "Working capital", "money", "Inventories plus receivables minus payables (the owner's definition, not current assets minus current liabilities)."),
        _Row("invested_capital", "Invested capital", "money", "Equity plus financial debt plus leases minus short-term investments: the base ROIC is measured on."),
    )),
    ("returns", "Returns", (
        _Row("roic", "ROIC", "pct", "Return on invested capital: EBIT x (1 - tax rate) divided by invested capital. Market average 13 to 14%. Uses the last twelve months.", True),
        _Row("roe", "ROE", "pct", "Return on equity: net income divided by equity."),
        _Row("roa", "ROA", "pct", "Return on assets: net income divided by total assets."),
        _Row("reinvestment_rate", "Reinvestment rate", "pct", "(CapEx minus depreciation and amortization, plus acquisitions) divided by FCF. A missing acquisitions line counts as zero. Above 20% with a high ROIC marks a compounder.", True),
        _Row("incremental_roic", "Incremental ROIC", "pct", "Change in after-tax operating profit divided by change in invested capital over the last years: what new capital earned."),
    )),
    ("shareholders", "Shareholders", (
        _Row("shares_diluted", "Diluted shares", "shares", "Weighted average shares after options and RSUs, on today's share basis. Falling is good.", True),
        _Row("shares_change", "Share count change", "pct", "Change in diluted shares against the same period a year earlier."),
        _Row("buybacks", "Buybacks", "money", "Cash spent buying back the company's own shares."),
        _Row("dividends", "Dividends", "money", "Cash paid out as dividends."),
        _Row("sbc_to_revenue", "Stock pay / revenue", "pct", "Stock-based compensation divided by revenue. Above 6% is high."),
        _Row("net_dilution", "Net dilution", "pct", "(Stock-based compensation minus buybacks) divided by market cap, last twelve months. Negative means buybacks exceed stock pay.", True),
    )),
    ("valuation", "Valuation", (
        _Row("price", "Share price", "per_share", "Closing price on the period end date, adjusted for later stock splits. The LTM column uses the latest close."),
        _Row("market_cap", "Market cap", "money", "Price times the diluted share count of the period (the LTM column uses the latest cover-page share count when the filing has one)."),
        _Row("enterprise_value", "Enterprise value", "money", "Market cap plus total debt minus cash and short-term investments."),
        _Row("ev_to_fcf", "EV / FCF", "ratio", "Enterprise value divided by the last twelve months of free cash flow. Market average 16 to 17x, lower is cheaper.", True),
        _Row("pe", "PER", "ratio", "Market cap divided by net income (last twelve months). Market average 16x.", True),
        _Row("pe_ex_cash", "PER ex-cash", "ratio", "PER after taking net cash out of the market cap."),
        _Row("fcf_yield", "FCF yield", "pct", "Free cash flow divided by enterprise value (last twelve months)."),
        _Row("ev_to_ebit", "EV / EBIT", "ratio", "Enterprise value divided by EBIT. Shown for reference, not scored."),
    )),
)
FINANCIAL_NOT_SHOWN = {"ebit", "ebit_margin", "ebitda", "net_debt", "capex", "gross_profit", "gross_margin", "fcf", "fcf_margin", "fcf_per_share", "ev_to_fcf", "fcf_yield", "ev_to_ebit", "enterprise_value", "roic",
                       "invested_capital", "working_capital", "reinvestment_rate", "incremental_roic", "pe_ex_cash"}


def _fcf_tip(quarterly: bool) -> str:
    definition = str(getattr(settings, "FCF_DEFINITION", "standard"))
    label = FCF_DEFINITIONS.get(definition, definition)
    if quarterly:
        return (f"Operating cash flow minus CapEx for the quarter. The valuation rows use the last twelve months under the "
                f"configured definition: {label}.")
    return f"Free cash flow under the configured definition: {label}. The last twelve months for the LTM column."


def _fiscal_label(end: pd.Timestamp) -> int:
    return end.year - 1 if end.month == 1 and end.day <= 7 else end.year  # a 52-week year ending in early January


def _column_meta(model: _Model, end: pd.Timestamp, key: str, label: str, derived: bool) -> dict[str, Any]:
    info = model.filings.get(end) or {}
    return {"key": key, "label": label, "end": end.strftime("%Y-%m-%d"), "filed": info.get("filed"), "form": info.get("form"),
            "url": info.get("url"), "derived": derived}


def _reinvestment_detail(latest: dict[str, float | None]) -> str | None:
    """Why the reinvestment rate is what it is: the components, a floored growth CapEx, a missing acquisitions line
    and R&D as information (outside the owner's formula). Display only."""
    acq_reported = latest.get("_acq_reported")
    text = reinvestment_explanation(latest.get("capex"), latest.get("depreciation_amortization"), latest.get("acquisitions"),
                                    None if acq_reported is None else bool(acq_reported), latest.get("reinvestment_rate"), "USD")
    rd_share = latest.get("rd_to_fcf")
    if rd_share is not None:
        text = (text + ". " if text else "") + f"For information, outside the formula: R&D is {rd_share:.0%} of free cash flow."
    return text


def _checklist(model: _Model, annual: list[dict[str, float | None]], ltm: dict[str, float | None] | None) -> list[dict[str, Any]]:
    """The owner's checks on the latest figures (LTM when there is one) against the framework's references."""
    latest = ltm or (annual[-1] if annual else {})
    fin = model.is_financial

    def item(key: str, label: str, value: float | None, unit: str, tip: str, reference: float | None = None,
             reference_label: str | None = None, better: str | None = None, verdict: str | None = None,
             texts: dict[str, str] | None = None, **extra: Any) -> dict[str, Any]:
        if verdict is None and value is not None and reference is not None and better in ("lower", "higher"):
            band = abs(reference) * VERDICT_BAND
            if abs(value - reference) <= band:
                verdict = "in line"
            else:
                good = value < reference if better == "lower" else value > reference
                verdict = "better" if good else "worse"
        # verdict_text: the verdict in words when the reference is not a market average (a rule or the company's own past).
        return {"key": key, "label": label, "value": value, "unit": unit, "reference": reference,
                "reference_label": reference_label, "better": better, "verdict": verdict,
                "verdict_text": (texts or {}).get(verdict or ""), "tip": tip, **extra}

    out: list[dict[str, Any]] = []
    not_for_banks = "Does not apply to banks and insurers: their cash and debt are the business. Use PER, ROE and ROA."
    if fin:
        out.append(item("ev_fcf", "EV / FCF", None, "ratio", not_for_banks, REF_EV_FCF, "market average 16-17x", "lower"))
    else:
        ev_fcf = latest.get("ev_to_fcf")
        out.append(item("ev_fcf", "EV / FCF", ev_fcf, "ratio",
                        "Enterprise value divided by free cash flow, the primary price check. Empty when free cash flow is not positive.",
                        REF_EV_FCF, "market average 16-17x", "lower"))
    pe = latest.get("pe")
    out.append(item("pe", "PER", pe, "ratio", "Market cap divided by net income. Empty when net income is not positive.", REF_PER, "market average 16x", "lower"))
    if fin:
        out.append(item("roic", "ROIC", None, "pct", not_for_banks, REF_ROIC, "market average 13-14%", "higher"))
        level = {"better": "Above the level that scores well", "worse": "Below the level that scores well", "in line": "Near the level that scores well"}
        out.append(item("roe", "ROE", latest.get("roe"), "pct", "Net income divided by equity: the main return check for a bank or insurer.", REF_ROE, "13% or more scores well", "higher", texts=level))
        out.append(item("roa", "ROA", latest.get("roa"), "pct", "Net income divided by total assets.", REF_ROA, "1% or more scores well", "higher", texts=level))
    else:
        out.append(item("roic", "ROIC", latest.get("roic"), "pct", "After-tax operating profit on invested capital. Above the market average means the business earns more than most.",
                        REF_ROIC, "market average 13-14%", "higher"))
        roics = [c.get("roic") for c in annual[-10:] if c.get("roic") is not None]
        if roics:
            above = sum(1 for r in roics if r >= REF_ROIC)
            high = sum(1 for r in roics if r >= HIGH_ROIC)
            share = above / len(roics)
            out.append(item("roic_persistence", "ROIC persistence", float(above), "years",
                            f"Years with ROIC at or above the market average of 13.5%, out of the last {len(roics)} years with a figure. "
                            "A durable moat shows as a high ROIC in most years.",
                            None, "at or above 13.5% in most years", "higher",
                            "better" if share >= 2 / 3 else "worse" if share <= 1 / 3 else "in line",
                            texts={"better": "Persistent", "worse": "Rarely above the average", "in line": "Some years"},
                            value_text=f"{above} of {len(roics)} years",
                            detail=f"At or above 15% (the fundamentals agent's moat test): {high} of {len(roics)} years."))
        else:
            out.append(item("roic_persistence", "ROIC persistence", None, "years", "Needs at least one year with an ROIC figure.", None, None, "higher"))
        rate, roic = latest.get("reinvestment_rate"), latest.get("roic")
        verdict = None
        if rate is not None and roic is not None:
            verdict = ("better" if roic >= HIGH_ROIC and rate >= GOOD_REINVESTMENT
                       else "worse" if roic < 0.08 and rate >= GOOD_REINVESTMENT else "in line")
        out.append(item("reinvestment_rate", "Reinvestment rate", rate, "pct",
                        "Share of free cash flow put back into growth. Above 20% with a high ROIC is a great sign; "
                        "the same reinvestment at an ROIC below 8% destroys value.",
                        GOOD_REINVESTMENT, "above 20% with a high ROIC is a great sign", "higher", verdict,
                        texts={"better": "Great sign", "worse": "Reinvesting at a low return", "in line": "Neutral"},
                        detail=_reinvestment_detail(latest)))
    nd = latest.get("net_dilution")
    nd_verdict = None if nd is None else "better" if nd <= 0 else "worse" if nd >= NET_DILUTION_WORSE else "in line"
    out.append(item("net_dilution", "Net dilution", nd, "pct",
                    "(Stock pay minus buybacks) divided by market cap. Negative means buybacks more than cover stock pay; "
                    "above +1.5% of market cap costs points in the shareholder check.",
                    0.0, "buybacks cover stock pay at 0 or below", "lower", nd_verdict,
                    texts={"better": "Buybacks cover stock pay", "worse": "Dilutes shareholders", "in line": "Small dilution"}))
    sbc = latest.get("sbc_to_revenue")
    out.append(item("sbc_to_revenue", "Stock pay / revenue", sbc, "pct", "Stock-based compensation divided by revenue. Shown as information; above 6% is high.",
                    HIGH_SBC_TO_REVENUE, "above 6% is high", "lower", None if sbc is None else "worse" if sbc >= HIGH_SBC_TO_REVENUE else "in line",
                    texts={"worse": "High", "in line": "Below 6%"}))
    if fin:
        out.append(item("net_debt_to_fcf", "Years to repay net debt", None, "years", not_for_banks))
        out.append(item("fcf_margin", "FCF margin trend", None, "pct", not_for_banks))
        return out
    net_debt, fcf = latest.get("net_debt"), latest.get("fcf")
    years, nd_note = None, "Net debt divided by free cash flow: how many years of free cash flow it would take to repay the debt. The framework sets no reference level."
    verdict = None
    if net_debt is not None and net_debt <= 0:
        years, verdict, nd_note = 0.0, "better", "Net cash: the company holds more cash than debt, so there is nothing to repay."
    elif net_debt is not None and fcf is not None and fcf > 0:
        years = net_debt / fcf
    elif net_debt is not None and fcf is not None:
        verdict, nd_note = "worse", "Free cash flow is not positive, so net debt cannot be repaid from it."
    out.append(item("net_debt_to_fcf", "Years to repay net debt", years, "years", nd_note, verdict=verdict,
                    texts={"better": "Net cash", "worse": "Free cash flow not positive"}))
    margins = [c.get("fcf_margin") for c in annual if c.get("fcf_margin") is not None]
    if len(margins) >= 4:
        latest_margin, prior = margins[-1], sum(margins[-4:-1]) / 3
        trend = "better" if latest_margin - prior > 0.01 else "worse" if latest_margin - prior < -0.01 else "in line"
        out.append(item("fcf_margin", "FCF margin trend", latest_margin, "pct",
                        "Latest free cash flow margin against the average of the three years before. A rising margin means the business keeps more of each sale.",
                        prior, "average of the three years before", "higher", trend,
                        texts={"better": "Rising", "worse": "Falling", "in line": "Steady"}))
    else:
        out.append(item("fcf_margin", "FCF margin trend", margins[-1] if margins else None, "pct", "Needs four years of free cash flow to show a trend.", None, None, "higher"))
    return out


def _empty_rows_notes(sections: list[dict[str, Any]], fin: bool) -> list[str]:
    notes = []
    for section in sections:
        for row in section["rows"]:
            if all(v is None for v in row["values"]) and not (fin and row["key"] in FINANCIAL_NOT_SHOWN):
                notes.append(f"{row['label']} is not tagged in this company's filings, or cannot be worked out, so it is left empty.")
    return notes


def financial_overview(ticker: str, period: str = "annual") -> dict[str, Any]:
    """The /api/financials payload (see the module docstring). Never raises for expected conditions."""
    symbol = ticker.strip().upper()
    period = "quarterly" if period == "quarterly" else "annual"
    payload: dict[str, Any] = {
        **_base(symbol), "currency": "USD", "period": period, "is_financial": False, "source": None,
        "as_of": date.today().isoformat(), "columns": [], "sections": [], "checklist": [], "notes": [],
    }
    cik = _resolve(symbol, payload)
    if cik is None:
        return payload
    payload["source"] = {"label": "SEC EDGAR XBRL", "url": _edgar_url(cik)}
    try:
        model = _model(symbol, cik)
    except SecConfigError:
        return _no_access(payload)
    except SecDataError as exc:
        text = str(exc)
        if text in (FOREIGN_FILER, NO_US_GAAP):  # not a failure to retry: this company has nothing to read here
            payload["reason"] = "foreign_filer" if text == FOREIGN_FILER else "no_us_gaap_tags"
            return _failure(payload, "error", text)
        if text.startswith("SEC has no data at"):  # a fund, or a company with no annual statements yet: retrying cannot help
            payload["reason"] = "no_company_facts"
            return _failure(payload, "error", f"The SEC has no reported figures for {symbol}. Funds, and companies that have not filed annual statements yet, have none; the Filings view lists what it does file.")
        return _failure(payload, "error", f"The SEC data could not be loaded ({_no_urls(text)[:160]}).", "Try again in a minute.")
    except Exception as exc:  # noqa: BLE001 - one company's odd data must not break the page
        logger.warning("Financial overview failed for %s: %s", symbol, _reason(exc), exc_info=True)
        return _failure(payload, "error", f"The financial overview could not be built ({_reason(exc)}).", "Try again later.")
    try:
        built = _assemble(model, period)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Financial overview failed for %s: %s", symbol, _reason(exc), exc_info=True)
        return _failure(payload, "error", f"The financial overview could not be built ({_reason(exc)}).", "Try again later.")
    payload.update(name=model.name, is_financial=model.is_financial, **built)
    return payload


def _assemble(model: _Model, period: str) -> dict[str, Any]:
    quarterly = period == "quarterly"
    fy_ends = _fiscal_year_ends(model)
    annual_cols = [_column(model, e, False) for e in fy_ends]
    ltm_end = _ltm_end(model, fy_ends)
    ltm = _column(model, ltm_end, False, latest_price=True) if ltm_end is not None else None

    metas: list[dict[str, Any]] = []
    cols: list[dict[str, float | None]] = []
    notes: list[str] = []
    if quarterly:
        shown = _quarter_ends(model)[-MAX_QUARTER_COLUMNS:]
        for end, i, k in shown:
            label_year = _fiscal_label(model.fiscal_years[i].end)
            cols.append(_column(model, end, True))
            metas.append(_column_meta(model, end, f"FY{label_year}Q{k}", f"Q{k} FY{label_year}", model.derived.get(end, False)))
        notes.append("Each quarter column shows the quarter's own flows. ROIC, EV/FCF, PER and the other ratios use the twelve months that end with that quarter.")
        notes.append("Columns marked derived contain figures worked out from year-to-date reports: cash flow statements are filed year to date, "
                     "and the fourth quarter is the fiscal year minus the first nine months.")
    else:
        for end, c in zip(fy_ends, annual_cols):
            metas.append(_column_meta(model, end, f"FY{_fiscal_label(end)}", f"FY {_fiscal_label(end)}", False))
        cols = list(annual_cols)
        if ltm is not None and ltm_end is not None:
            meta = _column_meta(model, ltm_end, "LTM", "LTM", True)
            metas.append(meta)
            cols.append(ltm)
            notes.append(f"LTM is the last twelve months to {meta['end']}: the latest 10-K plus the quarters filed since, worked out from year-to-date reports.")
    if not cols:
        raise SecDataError("The filings contain no complete fiscal year yet")

    sections = [{"key": key, "label": label, "rows": [_row(r, cols, quarterly) for r in rows]} for key, label, rows in SECTIONS]
    fin = model.is_financial
    if fin:
        notes.append("Bank or insurer: EV/FCF, ROIC, free cash flow, EBIT and net debt do not apply, so those rows are empty. "
                     "Use PER, ROE and ROA. EBIT is left out because it adds back interest expense, a bank's main cost.")
    notes.extend(model.notes)
    newest = cols[-1]
    if not fin and newest.get("reinvestment_rate") is not None:
        why = _reinvestment_detail(newest)
        if why and (newest.get("_growth_floored") or newest.get("_acq_reported") == 0.0):
            notes.append(f"Reinvestment rate in the latest column, {why}")
    notes.extend(_empty_rows_notes(sections, fin))
    if any(c.get("_metrics_ok") == 0.0 for c in cols):
        notes.append("Some columns lack enough income statement rows, so their ratios are empty.")
    # The checklist reads the newest figures at the latest close. Right after a 10-K there is no LTM column, and the
    # fiscal-year column is priced at the year end, so the year is priced again at the latest close.
    latest = ltm if ltm is not None else _column(model, fy_ends[-1], False, latest_price=True) if fy_ends else None
    return {"columns": metas, "sections": sections, "checklist": _checklist(model, annual_cols, latest),
            "checklist_basis": _checklist_basis(model, fy_ends, ltm_end), "notes": notes}


def _checklist_basis(model: _Model, fy_ends: list[pd.Timestamp], ltm_end: pd.Timestamp | None) -> dict[str, Any] | None:
    """Which period and which close the checklist uses, so its ratios can be lined up with the agents'."""
    if not fy_ends:
        return None
    if ltm_end is not None:
        period = {"kind": "ltm", "end": ltm_end.strftime("%Y-%m-%d"), "label": f"last twelve months to {ltm_end:%Y-%m-%d}"}
    else:
        period = {"kind": "fy", "end": fy_ends[-1].strftime("%Y-%m-%d"), "label": f"fiscal year to {fy_ends[-1]:%Y-%m-%d}"}
    if model.prices is not None and not model.prices.empty:
        period.update(price=float(model.prices.iloc[-1]), price_date=model.prices.index[-1].strftime("%Y-%m-%d"))
    return period


def _row(row: _Row, cols: list[dict[str, float | None]], quarterly: bool) -> dict[str, Any]:
    tip = _fcf_tip(quarterly) if row.key == "fcf" else row.tip
    return {"key": row.key, "label": row.label, "unit": row.unit, "values": [c.get(row.key) for c in cols], "tip": tip,
            "highlight": row.highlight}


def _fiscal_year_ends(model: _Model) -> list[pd.Timestamp]:
    """Fiscal year ends with a complete income statement anchor set, newest MAX_ANNUAL_COLUMNS, oldest first."""
    ends = [e for e, _, k in model.quarters if k == 4]
    keep = [e for e in ends if _has_year_data(model, e)]
    return keep[-MAX_ANNUAL_COLUMNS:]


def _has_year_data(model: _Model, end: pd.Timestamp) -> bool:
    return sum(1 for c in ANCHORS if (c, end) in model.values_ttm) >= 2


def _quarter_ends(model: _Model) -> list[tuple[pd.Timestamp, int, int]]:
    return [(e, i, k) for e, i, k in model.quarters
            if sum(1 for c in ANCHORS if (c, e) in model.values_q) >= 2]


def _ltm_end(model: _Model, fy_ends: list[pd.Timestamp]) -> pd.Timestamp | None:
    """The latest quarter end after the last fiscal year end that has twelve months of data."""
    if not fy_ends:
        return None
    for end, i, k in reversed(model.quarters):
        if end <= fy_ends[-1]:
            return None
        if k < 4 and _has_year_data(model, end):
            return end
    return None


# ------------------------------------------------------------------ filings

ITEM_NAMES = {
    "1.01": "material agreement", "1.02": "agreement ended", "1.03": "bankruptcy", "2.01": "acquisition or sale completed",
    "2.02": "results of operations", "2.03": "new financial obligation", "2.04": "debt acceleration",
    "2.05": "exit or restructuring costs", "2.06": "impairment", "3.01": "listing notice", "3.02": "unregistered share sale",
    "3.03": "change in shareholder rights", "4.01": "auditor change", "4.02": "financial statements no longer relied on",
    "5.01": "change of control", "5.02": "executive or director change", "5.03": "bylaw change", "5.07": "shareholder vote results",
    "7.01": "Regulation FD disclosure", "8.01": "other events",
}
FILING_FORMS = {"10-K", "10-K/A", "10-KT", "10-Q", "10-Q/A", "8-K", "8-K/A", "DEF 14A", "DEF 14A/A", "6-K"} | FOREIGN_ANNUAL
EXHIBIT_CACHE_VERSION = 1
FILINGS_CACHE_VERSION = 1


def _classify_form(form: str, items: list[str]) -> tuple[str, str]:
    """(kind, title) of one filing from its form and 8-K items."""
    base = form.replace("/A", "")
    suffix = " (amendment)" if form.endswith("/A") else ""
    if base in ("10-K", "10-KT"):
        return "annual", f"Annual report ({base}){suffix}"
    if base == "10-Q":
        return "quarterly", f"Quarterly report (10-Q){suffix}"
    if base == "DEF 14A":
        return "proxy", f"Proxy statement: pay, incentives and votes (DEF 14A){suffix}"
    if base in ("20-F", "40-F"):
        return "foreign_annual", f"Annual report for a foreign filer ({base}){suffix}"
    if base == "6-K":
        return "foreign_current", "Report of a foreign filer (6-K)"
    if base == "8-K":
        if "2.02" in items:
            return "earnings", f"Quarterly results (Item 2.02){suffix}"
        if "7.01" in items:
            return "current", f"Regulation FD disclosure (Item 7.01){suffix}"
        named = [ITEM_NAMES[i] for i in items if i in ITEM_NAMES][:2]
        return "current", (f"Current report: {', '.join(named)}{suffix}" if named else f"Current report (8-K){suffix}")
    return "other", form


def _exhibit_kind(description: str, document: str, number: str, items: list[str]) -> tuple[str, str]:
    """(kind, label) of one exhibit 99.x from its description and file name."""
    text = f"{description} {document}".lower()
    tag = f"(Ex. {number})"
    if re.search(r"transcript", text):
        return "transcript", f"Call transcript {tag}"
    if re.search(r"prepared remarks|\bscript\b|commentary", text):
        return "transcript", f"Prepared remarks {tag}"
    if re.search(r"presentation|slides|slide deck|\bdeck\b|supplement|investor day|webcast", text):
        return "presentation", f"Investor presentation or slides {tag}"
    if re.search(r"press release|news release|earnings release|results|announces", text):
        return "earnings", f"Earnings press release {tag}"
    if number == "99.1" and "2.02" in items:  # by convention the results press release
        return "earnings", f"Earnings press release {tag}"
    return "other", f"Other exhibit {tag}"


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def parse_filing_index(html: str, cik: int, items: list[str]) -> list[dict[str, str]]:
    """Exhibits 99.x listed on a filing's index page, labelled by what they are."""
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S | re.I)
        if len(cells) < 4:
            continue
        kind_text = _strip(cells[3]).upper()
        match = re.match(r"^EX-(99(?:\.\d+)?)", kind_text)
        link = re.search(r'href="([^"]+)"', cells[2])
        if not match or not link:
            continue
        href = link.group(1)
        if "doc=" in href:
            href = href.split("doc=", 1)[1]
        url = href if href.startswith("http") else f"https://www.sec.gov{href}"
        kind, label = _exhibit_kind(_strip(cells[1]), _strip(cells[2]), match.group(1), items)
        out.append({"label": label, "kind": kind, "url": url})
    return out


def _fetch_text(url: str) -> str:
    return _get_bytes(url).decode("utf-8", errors="replace")


def _read_json(path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _exhibits(cik: int, filing: dict[str, Any], cache: dict[str, list], failures: list[str]) -> None:
    """Fill ``filing['exhibits']`` from the cache or the filing index (one request, kept for good)."""
    accession = filing["accession"]
    if accession in cache:
        filing["exhibits"] = cache[accession]
        return
    try:
        html = _fetch_text(filing["index_url"])
    except SecConfigError:
        raise
    except SecDataError as exc:
        failures.append(accession)
        logger.info("Filing index %s unavailable: %s", accession, exc)
        return
    filing["exhibits"] = cache[accession] = parse_filing_index(html, cik, filing["items"])


def _filing_entry(cik: int, recent: dict[str, list], i: int) -> dict[str, Any] | None:
    form = recent["form"][i]
    if form not in FILING_FORMS:
        return None
    accession = recent["accessionNumber"][i]
    document = (recent.get("primaryDocument") or [None] * (i + 1))[i] or None
    items = [x.strip() for x in str((recent.get("items") or [""] * (i + 1))[i] or "").split(",") if x.strip()]
    kind, title = _classify_form(form, items)
    return {
        "form": form, "kind": kind, "filed": recent["filingDate"][i],
        "period": (recent.get("reportDate") or [None] * (i + 1))[i] or None, "title": title, "items": items,
        "url": _filing_url(cik, accession, document), "index_url": _filing_url(cik, accession, None), "exhibits": [],
        "accession": accession,
    }


def _build_filings(cik: int, subs: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """All recent filings of the forms we show, newest first, with exhibits for the latest results 8-Ks.

    The flag says whether every exhibit lookup worked (a payload with a gap is not cached).
    """
    recent = _recent_arrays(subs)
    entries = [e for i in range(len(recent.get("form") or [])) if (e := _filing_entry(cik, recent, i)) is not None]
    entries.sort(key=lambda e: e["filed"], reverse=True)
    path = sec_cache_dir() / f"exhibits_v{EXHIBIT_CACHE_VERSION}_{cik:010d}.json"
    cache: dict[str, list] = _read_json(path) or {}
    before = len(cache)
    failures: list[str] = []
    todo = [e for e in entries if e["form"] in ("8-K", "8-K/A") and ("2.02" in e["items"] or "7.01" in e["items"] or "8.01" in e["items"])][:EXHIBIT_FILINGS]
    for entry in todo:
        _exhibits(cik, entry, cache, failures)
        kinds = {x["kind"] for x in entry["exhibits"]}
        if entry["kind"] == "current":  # a 7.01 filing is named after what it carries
            entry["kind"] = "presentation" if "presentation" in kinds else "transcript" if "transcript" in kinds else "current"
            if entry["kind"] == "presentation":
                entry["title"] = "Investor presentation (Item 7.01)"
            elif entry["kind"] == "transcript":
                entry["title"] = "Call transcript (Item 7.01)"
    if len(cache) != before:
        path.write_text(json.dumps(cache))
    for entry in entries:
        entry.pop("accession")
    return entries, not failures


def _latest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    def first(*kinds: str) -> dict[str, Any] | None:
        return next((e for e in entries if e["kind"] in kinds), None)

    presentation = next((e for e in entries if any(x["kind"] == "presentation" for x in e["exhibits"])), None) or first("presentation")
    return {"annual": first("annual", "foreign_annual"), "quarterly": first("quarterly"), "earnings": first("earnings"),
            "presentation": presentation, "proxy": first("proxy")}


def _website(symbol: str, subs: dict[str, Any]) -> str | None:
    for key in ("investorWebsite", "website"):
        site = str(subs.get(key) or "").strip()
        if site:
            return site if site.startswith("http") else f"https://{site}"
    try:  # the company's own profile on Yahoo carries its website
        from data import fundamentals as _fundamentals

        site = str(_fundamentals._load_info(symbol).get("website") or "").strip()
    except Exception as exc:  # noqa: BLE001 - the link is optional
        logger.info("No website for %s: %s", symbol, exc)
        return None
    return site or None


def _links(symbol: str, cik: int, subs: dict[str, Any], foreign: bool) -> list[dict[str, Any]]:
    forms = "6-K" if foreign else "8-K"
    links: list[dict[str, Any]] = [
        {"label": "All filings on EDGAR", "url": _edgar_url(cik), "note": None},
        {"label": "Insider trades (Form 4)", "url": _edgar_url(cik, "4"),
         "note": "Form 4 reports buys and sales by executives and directors. Only a link here."},
        {"label": "Call transcripts", "url": f"https://www.sec.gov/edgar/search/#/q=transcript&ciks={cik:010d}&forms={forms}",
         "note": "Earnings call transcripts are not filed with the SEC. A few companies attach one to an 8-K; this searches this company's filings for the word transcript."},
    ]
    site = _website(symbol, subs)
    if site:
        links.append({"label": "Company website", "url": site,
                      "note": "Look for an Investors or Investor Relations section: earnings call webcasts, transcripts and slides are usually there."})
    return links


def recent_filings(ticker: str, limit: int = 40) -> dict[str, Any]:
    """The /api/filings payload (see the module docstring). Never raises for expected conditions."""
    symbol = ticker.strip().upper()
    limit = max(1, min(int(limit), 200))
    payload: dict[str, Any] = {**_base(symbol), "filings": [], "latest": {}, "links": [], "notes": []}
    cik = _resolve(symbol, payload)
    if cik is None:
        return payload
    try:
        subs = _submissions(cik)
        payload["name"] = subs.get("name")
        cache_path = sec_cache_dir() / f"filings_v{FILINGS_CACHE_VERSION}_{cik:010d}.json"
        cached = _read_json(cache_path) if _fresh(cache_path, SUBMISSIONS_MAX_AGE_DAYS) else None
        if cached is None:
            entries, complete = _build_filings(cik, subs)
            if complete:
                cache_path.write_text(json.dumps(entries))
        else:
            entries = cached
    except SecConfigError:
        return _no_access(payload)
    except SecDataError as exc:
        return _failure(payload, "error", f"The SEC filings could not be loaded ({_no_urls(str(exc))[:160]}).","Try again in a minute.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Filings failed for %s: %s", symbol, _reason(exc), exc_info=True)
        return _failure(payload, "error", f"The filings could not be loaded ({_reason(exc)}).", "Try again later.")
    foreign = any(e["kind"].startswith("foreign") for e in entries) and not any(e["kind"] == "annual" for e in entries)
    payload.update(filings=entries[:limit], latest=_latest(entries), links=_links(symbol, cik, subs, foreign))
    if any(e["kind"] == "foreign_current" for e in entries):
        payload["notes"].append("6-K exhibits are not itemised: foreign filers use the 6-K for everything, so open the filing to see whether it holds results or slides.")
    if not entries:
        payload["notes"].append("No annual, quarterly or current reports were found in the recent filings.")
    return payload
