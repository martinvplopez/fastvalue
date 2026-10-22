"""Point-in-time company fundamentals from SEC EDGAR XBRL.

The SEC companyfacts API returns every XBRL fact a company has filed since
2009, each stamped with the date it was filed. That makes it possible to
rebuild the annual statements exactly as an investor could have read them on
any past date, which a backtest of the fundamentals agents needs: Yahoo only
serves today's statements, restated, for the last four years.

``CompanyFacts.statements_asof`` maps us-gaap concepts to the Yahoo line-item
names that ``analysis.fundamental_metrics`` reads, so the agents run unchanged
on either source. Only facts from annual reports (10-K) filed on or before the
requested date are used, and a figure restated in a later 10-K replaces the
original only from the restatement's filing date.

SEC fair-access rules: every request must send a User-Agent with a contact
email, and clients must stay under 10 requests per second. Set the header with
``export SEC_USER_AGENT="fastvalue you@example.com"``. Responses are
cached under data/cache/sec/ (git-ignored).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

import config.settings as settings
from data.cache_utils import _ALL_CACHES, ttl_cache
from data.financial_statements import FinancialStatements

logger = logging.getLogger(__name__)

SEC_USER_AGENT_ENV = "SEC_USER_AGENT"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FACTS_URL_PREFIX = FACTS_URL.split("CIK", 1)[0]
FACTS_REUSE_SECONDS = 120.0
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
MIN_REQUEST_INTERVAL = 0.125  # 8 requests per second, under the SEC limit of 10
CACHE_MAX_AGE_DAYS = 7
SUBMISSIONS_MAX_AGE_DAYS = 1
SOURCE_LABEL = "SEC EDGAR XBRL (10-K, point-in-time)"

ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-KT", "10-KT/A", "10-K405"})
PERIODIC_FORMS = ANNUAL_FORMS | {"10-Q", "10-Q/A"}
ANNUAL_DAYS = (340, 380)  # 52/53-week fiscal years included
SHARE_SCALE_OUTLIER = 20.0  # a share count 20x off the other years is a tagging error (thousands vs units)
SHARES_MAX_AGE_DAYS = 460  # ignore a cover-page share count older than about 15 months
IMMATERIAL_INVENTORY = 0.02  # of total assets: a missing year is treated as zero

# Row name (as used by analysis.fundamental_metrics) -> us-gaap concepts in priority order.
# The first concept that has a value for a fiscal year wins, so a company that
# switched tags (SalesRevenueNet before 2018, RevenueFromContract... after) keeps one row.
INCOME_ROWS: dict[str, tuple[str, ...]] = {
    "Total Revenue": (
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet", "SalesRevenueServicesNet",
        "RevenuesNetOfInterestExpense",
    ),
    "Operating Income": ("OperatingIncomeLoss",),
    "Pretax Income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    "Tax Provision": ("IncomeTaxExpenseBenefit",),
    "Interest Expense": (
        "InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense",
        "InterestExpenseBorrowings",
    ),
    "Net Income Common Stockholders": ("NetIncomeLossAvailableToCommonStockholdersBasic",),
    "Net Income": ("NetIncomeLoss", "ProfitLoss"),
    "Diluted Average Shares": (
        "WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ),
    "Basic Average Shares": (
        "WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ),
}
SHARE_ROWS = ("Diluted Average Shares", "Basic Average Shares")
CASHFLOW_ROWS: dict[str, tuple[str, ...]] = {
    "Depreciation And Amortization": (
        "DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ),
    "Operating Cash Flow": (
        "NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "Stock Based Compensation": ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"),
    "Interest Paid Supplemental Data": ("InterestPaidNet", "InterestPaid"),
    "Income Tax Paid Supplemental Data": ("IncomeTaxesPaidNet", "IncomeTaxesPaid"),
    "Issuance Of Capital Stock": (
        "ProceedsFromIssuanceOfCommonStock", "ProceedsFromStockOptionsExercised",
        "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions",
    ),
}
# Cash outflows that Yahoo reports as negative numbers (the metrics code expects that sign).
CASHFLOW_OUTFLOW_ROWS: dict[str, tuple[str, ...]] = {
    "Capital Expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    ),
    "Repurchase Of Capital Stock": ("PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"),
    "Purchase Of Business": (
        "PaymentsToAcquireBusinessesNetOfCashAcquired", "PaymentsToAcquireBusinessesGross",
        "PaymentsToAcquireBusinessesAndInterestInAffiliates",
    ),
}
BALANCE_ROWS: dict[str, tuple[str, ...]] = {
    "Total Assets": ("Assets",),
    "Stockholders Equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "Inventory": ("InventoryNet", "InventoryGross", "InventoryFinishedGoodsNetOfReserves"),
    "Accounts Receivable": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent"),
    "Accounts Payable": ("AccountsPayableCurrent", "AccountsPayableTradeCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"),
    "Cash And Cash Equivalents": (
        "CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash", "CashAndDueFromBanks",
    ),
    "Other Short Term Investments": (
        "ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "AvailableForSaleSecuritiesCurrent",
    ),
}
# Balance-sheet concepts combined into Total Debt (financial debt plus lease liabilities).
DEBT_CONCEPTS: dict[str, tuple[str, ...]] = {
    "long_term_total": ("LongTermDebt",),
    "long_term_noncurrent": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"),
    "long_term_current": ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    "debt_current": ("DebtCurrent",),
    "short_term": ("ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings"),
    "operating_lease": ("OperatingLeaseLiability",),
    "operating_lease_current": ("OperatingLeaseLiabilityCurrent",),
    "operating_lease_noncurrent": ("OperatingLeaseLiabilityNoncurrent",),
    "finance_lease": ("FinanceLeaseLiability",),
    "finance_lease_current": ("FinanceLeaseLiabilityCurrent",),
    "finance_lease_noncurrent": ("FinanceLeaseLiabilityNoncurrent",),
    "notes": ("LongTermNotesPayable", "SeniorNotes", "NotesPayable", "ConvertibleNotesPayable"),
}
# Used when no combined D&A line is tagged: depreciation plus intangible amortization.
DEPRECIATION = "Depreciation"
AMORTIZATION = "AmortizationOfIntangibleAssets"
CASH_AND_STI = ("CashCashEquivalentsAndShortTermInvestments",)
SHARES_OUTSTANDING_DEI = "EntityCommonStockSharesOutstanding"
SHARES_OUTSTANDING_GAAP = "CommonStockSharesOutstanding"

# Concepts whose value is a flow over the fiscal year (the rest are balance-sheet instants).
DURATION_CONCEPTS = frozenset(
    {c for rows in (INCOME_ROWS, CASHFLOW_ROWS, CASHFLOW_OUTFLOW_ROWS) for concepts in rows.values() for c in concepts}
    | {DEPRECIATION, AMORTIZATION}
)
INSTANT_CONCEPTS = frozenset(
    {c for concepts in BALANCE_ROWS.values() for c in concepts}
    | {c for concepts in DEBT_CONCEPTS.values() for c in concepts}
    | set(CASH_AND_STI)
    | {SHARES_OUTSTANDING_GAAP}
)
# Concepts that anchor a fiscal year (a 10-K always reports at least some of them).
ANCHOR_CONCEPTS = frozenset({
    "NetIncomeLoss", "ProfitLoss", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
    "OperatingIncomeLoss", "IncomeTaxExpenseBenefit", "WeightedAverageNumberOfDilutedSharesOutstanding",
    "NetCashProvidedByUsedInOperatingActivities",
})
CONCEPT_VERSION = 2  # bump when the concept lists change so cached extracts are rebuilt

# SIC codes for banks, credit, brokers, insurers and holding/investment offices.
# REITs (6798) and real estate (65xx) are excluded, as Yahoo files them under Real Estate.
FINANCIAL_SIC_RANGES = ((6000, 6499), (6700, 6797), (6799, 6799))


class SecConfigError(RuntimeError):
    """SEC access is not configured (no contact email in the User-Agent)."""


class SecDataError(RuntimeError):
    """The SEC returned no usable data."""


# ------------------------------------------------------------------ network


CONTACT_FILE = "local_settings.json"  # under STORE_DIR (git-ignored): the SEC contact saved from the app
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def _stored_user_agent() -> str:
    try:
        data = json.loads((settings.STORE_DIR / CONTACT_FILE).read_text())
    except (OSError, ValueError):
        return ""
    return str(data.get("sec_user_agent") or "").strip() if isinstance(data, dict) else ""


def sec_access_source() -> str | None:
    """Where the SEC contact comes from: 'environment', 'saved' (set in the app) or None (not set)."""
    if "@" in os.environ.get(SEC_USER_AGENT_ENV, ""):
        return "environment"
    return "saved" if "@" in _stored_user_agent() else None


def has_sec_access() -> bool:
    return sec_access_source() is not None


def save_contact(email: str | None) -> str | None:
    """Save (or with an empty value, forget) the contact email the SEC asks for. Stored only on this machine."""
    path = settings.STORE_DIR / CONTACT_FILE
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    email = (email or "").strip()
    if email:
        if not EMAIL_RE.match(email):
            raise ValueError("Enter a valid email address")
        data["sec_user_agent"] = f"fastvalue {email}"
    else:
        data.pop("sec_user_agent", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return sec_access_source()


def user_agent() -> str:
    ua = os.environ.get(SEC_USER_AGENT_ENV, "").strip() or _stored_user_agent()
    if "@" not in ua:
        raise SecConfigError(
            "The SEC asks for a contact email with every request. Add one in the app (Backtest tab, "
            f'"Set contact email") or start it with {SEC_USER_AGENT_ENV}="fastvalue you@example.com".'
        )
    return ua


_rate_lock = threading.Lock()
_last_request = [0.0]


def _request(url: str, retries: int = 4) -> tuple[bool, bytes]:
    """The response body as sent and whether it is gzip-compressed (throttled, retried)."""
    headers = {"User-Agent": user_agent(), "Accept-Encoding": "gzip"}
    for attempt in range(retries):
        with _rate_lock:
            wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request[0])
            if wait > 0:
                time.sleep(wait)
            _last_request[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
                return resp.headers.get("Content-Encoding") == "gzip", resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise SecDataError(f"SEC has no data at {url}") from exc
            if exc.code == 403:
                raise SecConfigError(f"SEC refused the request (403); check {SEC_USER_AGENT_ENV}") from exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise SecDataError(f"SEC request failed ({exc.code}) for {url}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise SecDataError(f"SEC request failed for {url}: {exc}") from exc
        time.sleep(2.0 * 2 ** attempt)
    raise SecDataError(f"SEC request failed for {url}")


@ttl_cache(FACTS_REUSE_SECONDS, maxsize=8)
def _facts_body(url: str) -> tuple[bool, bytes]:
    # One companyfacts download feeds three extracts (facts, margins, reports) whose loaders refresh on their own
    # schedules, often within the same analysis; the compressed body (about 0.3 MB) is kept briefly and shared, and
    # the per-key lock makes concurrent loaders wait for the one download.
    return _request(url)


def _get_json(url: str, retries: int = 4) -> Any:
    gz, body = _facts_body(url) if url.startswith(FACTS_URL_PREFIX) else _request(url, retries)
    return json.loads(gzip.decompress(body) if gz else body)


def sec_cache_dir() -> Path:
    path = settings.DATA_DIR / "sec"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fresh(path: Path, max_age_days: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400


def _cached_json(path: Path, url: str, max_age_days: float) -> Any:
    if _fresh(path, max_age_days):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    data = _get_json(url)
    with gzip.open(path, "wt") as fh:
        json.dump(data, fh)
    return data


def load_submissions(cik: int, max_age_days: float = SUBMISSIONS_MAX_AGE_DAYS) -> dict[str, Any]:
    """The company's submissions JSON (name, SIC code, website and recent filings), cached for a day."""
    return _cached_json(sec_cache_dir() / f"submissions_{cik:010d}.json.gz", SUBMISSIONS_URL.format(cik=cik), max_age_days)


def latest_periodic_filing(subs: dict[str, Any]) -> pd.Timestamp | None:
    """When the newest 10-K or 10-Q in a submissions JSON was accepted (the filing date when no time is given)."""
    recent = (subs.get("filings") or {}).get("recent") or {}
    accepted, filed = recent.get("acceptanceDateTime") or [], recent.get("filingDate") or []
    out: pd.Timestamp | None = None
    for i, form in enumerate(recent.get("form") or []):
        if form not in PERIODIC_FORMS:
            continue
        raw = (accepted[i] if i < len(accepted) else None) or (filed[i] if i < len(filed) else None)
        try:
            when = pd.Timestamp(raw)
        except (TypeError, ValueError):
            continue
        if pd.isna(when):
            continue
        when = when.tz_localize("UTC") if when.tzinfo is None else when
        out = when if out is None or when > out else out
    return out


def filed_since(cik: int, since: float) -> bool:
    """True when the SEC lists a 10-K or 10-Q accepted after ``since`` (epoch seconds). Never raises."""
    try:
        latest = latest_periodic_filing(load_submissions(cik))
    except Exception as exc:  # noqa: BLE001 - the cached copy is kept when the check fails
        logger.info("SEC filing list unavailable for CIK %s: %s", cik, exc)
        return False
    return latest is not None and latest.timestamp() > since


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


# path -> (file stamp, parsed map). Parsing the 10,000-row file took 5-9 ms and ran twice per analysis.
_TICKER_MAPS: dict[str, tuple[tuple[int, int], dict[str, int]]] = {}
_ALL_CACHES.append(_TICKER_MAPS)


def load_ticker_map(max_age_days: float = CACHE_MAX_AGE_DAYS) -> dict[str, int]:
    """Current ticker -> CIK map (Yahoo-style tickers such as BRK-B)."""
    path = sec_cache_dir() / "company_tickers.json.gz"
    before = _stamp(path) if _fresh(path, max_age_days) else None
    hit = _TICKER_MAPS.get(str(path))
    if before is not None and hit is not None and hit[0] == before:
        return dict(hit[1])
    raw = _cached_json(path, TICKERS_URL, max_age_days)
    out: dict[str, int] = {}
    for row in raw.values():
        out.setdefault(str(row["ticker"]).upper().replace(".", "-"), int(row["cik_str"]))
    after = _stamp(path)
    if after is not None and before in (None, after):  # not rewritten by another thread while this one read it
        _TICKER_MAPS[str(path)] = (after, out)
    return dict(out)


# ------------------------------------------------------------------ parsing


def extract_facts(raw: dict[str, Any]) -> pd.DataFrame:
    """Compact frame of the facts this module uses, one row per filed value."""
    wanted = DURATION_CONCEPTS | INSTANT_CONCEPTS
    rows: list[tuple] = []
    for taxonomy, concepts in (raw.get("facts") or {}).items():
        for concept, body in concepts.items():
            if not ((taxonomy == "us-gaap" and concept in wanted) or (taxonomy == "dei" and concept == SHARES_OUTSTANDING_DEI)):
                continue
            for unit, entries in (body.get("units") or {}).items():
                if unit not in ("USD", "shares"):
                    continue
                for e in entries:
                    if e.get("val") is None or not e.get("end") or not e.get("filed"):
                        continue
                    rows.append((concept, unit, e.get("start"), e["end"], float(e["val"]), e["filed"], e.get("form") or "", e.get("accn") or ""))
    frame = pd.DataFrame(rows, columns=["concept", "unit", "start", "end", "val", "filed", "form", "accn"])
    for col in ("start", "end", "filed"):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame.dropna(subset=["end", "filed"]).reset_index(drop=True)


def is_financial_sic(sic: int | None) -> bool:
    return sic is not None and any(lo <= sic <= hi for lo, hi in FINANCIAL_SIC_RANGES)


@dataclass
class CompanyFacts:
    """Every relevant XBRL fact one company filed, with point-in-time accessors."""

    ticker: str
    cik: int
    name: str | None
    sic: int | None
    sic_description: str | None
    facts: pd.DataFrame
    splits: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # split ratios by date (Yahoo)
    _annual: pd.DataFrame = field(init=False, repr=False)
    _shares: pd.DataFrame = field(init=False, repr=False)
    _snapshots: dict[tuple[pd.Timestamp, int], FinancialStatements] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        f = self.facts
        days = (f["end"] - f["start"]).dt.days
        is_duration = f["concept"].isin(DURATION_CONCEPTS)
        annual_duration = is_duration & days.between(*ANNUAL_DAYS)
        instant = f["concept"].isin(INSTANT_CONCEPTS) & f["start"].isna()
        annual = f[f["form"].isin(ANNUAL_FORMS) & (annual_duration | instant)].copy()
        annual["duration"] = annual["concept"].isin(DURATION_CONCEPTS)
        self._annual = annual.sort_values("filed", kind="stable").reset_index(drop=True)
        shares = f[f["concept"].isin({SHARES_OUTSTANDING_DEI, SHARES_OUTSTANDING_GAAP}) & (f["unit"] == "shares")]
        self._shares = shares.sort_values("filed", kind="stable").reset_index(drop=True)

    @property
    def is_financial(self) -> bool:
        return is_financial_sic(self.sic)

    @property
    def first_filed(self) -> pd.Timestamp | None:
        return None if self._annual.empty else self._annual["filed"].iloc[0]

    # -------------------------------------------------------------- statements

    def filing_dates(self) -> list[pd.Timestamp]:
        """Dates on which the annual statements changed (each 10-K or 10-K/A)."""
        return sorted(self._annual["filed"].unique())

    def statements_asof(self, date: pd.Timestamp, max_years: int = 4) -> FinancialStatements:
        """Annual statements as known on ``date`` (newest fiscal year first).

        Share counts are expressed on the share basis of ``date``: a figure is
        multiplied by every split between its filing date and ``date`` (filings
        already restate for splits that happened before they were filed).
        """
        date = pd.Timestamp(date)
        known = self._annual[self._annual["filed"] <= date]
        if known.empty:
            return FinancialStatements(ticker=self.ticker, available=False, error=f"No 10-K filed by {date:%Y-%m-%d}",
                                       source=SOURCE_LABEL)
        last_filed = known["filed"].iloc[-1]
        key = (last_filed, max_years)
        base = self._snapshots.get(key)
        if base is None:
            base = self._build(known, max_years, last_filed)
            self._snapshots[key] = base
        income = base.income
        factor = _split_product(self.splits, last_filed, date)
        if factor != 1.0 and not income.empty:
            income = income.copy()
            rows = [r for r in SHARE_ROWS if r in income.index]
            income.loc[rows] = income.loc[rows] * factor
        out = FinancialStatements(
            ticker=base.ticker, available=base.available, income=income, balance=base.balance,
            cashflow=base.cashflow, error=base.error, source=SOURCE_LABEL,
        )
        if not self.splits.empty:
            out.splits = self.splits[self.splits.index <= date]
        return out

    def _build(self, known: pd.DataFrame, max_years: int, last_filed: pd.Timestamp) -> FinancialStatements:
        latest = known.groupby(["concept", "end"], sort=False).tail(1)  # latest filing wins (restatements)
        values: dict[tuple[str, pd.Timestamp], float] = {
            (c, e): v for c, e, v in zip(latest["concept"], latest["end"], latest["val"])
        }
        # Put every share count on the basis of the newest filing in this snapshot.
        share_concepts = {c for r in SHARE_ROWS for c in INCOME_ROWS[r]}
        if not self.splits.empty:
            for c, e, filed in zip(latest["concept"], latest["end"], latest["filed"]):
                if c in share_concepts:
                    values[(c, e)] *= _split_product(self.splits, filed, last_filed)
        anchors = latest[latest["duration"] & latest["concept"].isin(ANCHOR_CONCEPTS)]
        counts = anchors.groupby("end")["concept"].nunique()
        year_ends = sorted(counts[counts >= 2].index, reverse=True)[:max_years]
        if not year_ends:
            return FinancialStatements(ticker=self.ticker, available=False, error="10-K facts lack income statement anchors")
        return statements_from_values(self.ticker, values, year_ends)

    # -------------------------------------------------------------- shares

    def shares_outstanding_asof(self, date: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
        """(shares outstanding, as-of date) from the latest 10-K or 10-Q filed on or before ``date``.

        The cover-page count (dei) is preferred; the balance-sheet count is used when
        the filing has no cover-page figure. Counts older than about 15 months are ignored.
        """
        date = pd.Timestamp(date)
        known = self._shares[(self._shares["filed"] <= date) & (self._shares["end"] <= date)]
        if known.empty:
            return None
        filing = known[known["filed"] == known["filed"].iloc[-1]]
        for concept in (SHARES_OUTSTANDING_DEI, SHARES_OUTSTANDING_GAAP):
            rows = filing[filing["concept"] == concept]
            if rows.empty:
                continue
            latest_end = rows["end"].max()
            same = rows[rows["end"] == latest_end]
            if (date - latest_end).days > SHARES_MAX_AGE_DAYS:
                return None
            # Several share classes are reported as separate facts of the same filing: add them up.
            total = float(same.drop_duplicates(subset=["val"])["val"].sum())
            if total > 0:
                return total, pd.Timestamp(latest_end)
        return None


def statements_from_values(
    ticker: str, values: dict[tuple[str, pd.Timestamp], float], year_ends: list[pd.Timestamp]
) -> FinancialStatements:
    """Map us-gaap values keyed by (concept, period end) to the Yahoo-style statements for ``year_ends``.

    ``CompanyFacts`` calls this with annual values; ``data/sec_reports.py`` calls it with trailing-twelve-month
    and single-quarter values, so every period goes through the same concept lists and derived lines.
    """
    def first(concepts: tuple[str, ...], end: pd.Timestamp) -> float | None:
        for c in concepts:
            v = values.get((c, end))
            if v is not None:
                return v
        return None

    income: dict[str, dict[pd.Timestamp, float | None]] = {}
    cashflow: dict[str, dict[pd.Timestamp, float | None]] = {}
    balance: dict[str, dict[pd.Timestamp, float | None]] = {}
    for end in year_ends:
        for row, concepts in INCOME_ROWS.items():
            income.setdefault(row, {})[end] = first(concepts, end)
        for row, concepts in CASHFLOW_ROWS.items():
            cashflow.setdefault(row, {})[end] = first(concepts, end)
        if cashflow["Depreciation And Amortization"][end] is None and values.get((DEPRECIATION, end)) is not None:
            cashflow["Depreciation And Amortization"][end] = values[(DEPRECIATION, end)] + values.get((AMORTIZATION, end), 0.0)
        for row, concepts in CASHFLOW_OUTFLOW_ROWS.items():
            v = first(concepts, end)
            cashflow.setdefault(row, {})[end] = -abs(v) if v is not None else None
        for row, concepts in BALANCE_ROWS.items():
            balance.setdefault(row, {})[end] = first(concepts, end)

        # EBIT = pretax income + interest expense (Yahoo's definition); EBITDA = EBIT + D&A.
        pretax, interest = income["Pretax Income"][end], income["Interest Expense"][end]
        ebit = pretax + (interest or 0.0) if pretax is not None else None
        dna = cashflow["Depreciation And Amortization"][end]
        income.setdefault("EBIT", {})[end] = ebit
        income.setdefault("EBITDA", {})[end] = ebit + dna if ebit is not None and dna is not None else None
        income.setdefault("Reconciled Depreciation", {})[end] = dna

        cash = balance["Cash And Cash Equivalents"][end]
        sti = balance["Other Short Term Investments"][end]
        combined = first(CASH_AND_STI, end)
        if combined is None and cash is not None:
            combined = cash + (sti or 0.0)
        balance.setdefault("Cash Cash Equivalents And Short Term Investments", {})[end] = combined
        debt = _total_debt({k: first(v, end) for k, v in DEBT_CONCEPTS.items()})
        if debt is None and balance["Total Assets"][end] is not None:
            debt = 0.0  # a full 10-K balance sheet without any debt or lease line: no debt
        balance.setdefault("Total Debt", {})[end] = debt

    # Companies that tag only basic shares (equal to diluted, as at Exxon) still get a share trend.
    for end in year_ends:
        if income["Diluted Average Shares"][end] is None:
            income["Diluted Average Shares"][end] = income["Basic Average Shares"][end]
    for row in SHARE_ROWS:
        _drop_scale_outliers(income[row])
    _fill_immaterial_inventory(balance)

    def frame(rows: dict[str, dict[pd.Timestamp, float | None]]) -> pd.DataFrame:
        df = pd.DataFrame(rows).T.astype(float)
        df = df.dropna(how="all")  # keep has_row() meaningful: a row exists only if reported
        return df[sorted(df.columns, reverse=True)] if not df.empty else df

    return FinancialStatements(
        ticker=ticker, available=True, income=frame(income), balance=frame(balance), cashflow=frame(cashflow),
        source=SOURCE_LABEL,
    )


def _drop_scale_outliers(row: dict[pd.Timestamp, float | None]) -> None:
    """Blank share counts that are off by a scale factor (for example reported in thousands)."""
    known = [v for v in row.values() if v]
    if len(known) < 3:
        return
    median = float(pd.Series(known).median())
    for end, v in row.items():
        if v and (v / median > SHARE_SCALE_OUTLIER or median / v > SHARE_SCALE_OUTLIER):
            row[end] = None


def _fill_immaterial_inventory(balance: dict[str, dict[pd.Timestamp, float | None]]) -> None:
    """A year without an inventory line counts as zero when inventory is immaterial in other years."""
    inventory, assets = balance["Inventory"], balance["Total Assets"]
    shares = [inventory[e] / assets[e] for e in inventory if inventory[e] is not None and assets.get(e)]
    if not shares or max(shares) >= IMMATERIAL_INVENTORY:
        return
    for end, v in inventory.items():
        if v is None and assets.get(end) is not None:
            inventory[end] = 0.0


def _split_product(splits: pd.Series, after: pd.Timestamp, until: pd.Timestamp) -> float:
    """Product of split ratios dated in (after, until]."""
    if splits.empty or until <= after:
        return 1.0
    s = splits[(splits.index > after) & (splits.index <= until)]
    return float(s.prod()) if len(s) else 1.0


def _total_debt(parts: dict[str, float | None]) -> float | None:
    """Financial debt plus lease liabilities from whichever debt tags the company uses."""
    if parts["long_term_total"] is not None:
        long_term = parts["long_term_total"]
        current_from_debt_current = False
    else:
        noncurrent, current = parts["long_term_noncurrent"], parts["long_term_current"]
        current_from_debt_current = current is None and parts["debt_current"] is not None
        if current_from_debt_current:
            current = parts["debt_current"]  # DebtCurrent already includes short-term borrowings
        long_term = None if noncurrent is None and current is None else (noncurrent or 0.0) + (current or 0.0)
    short_term = None if current_from_debt_current else parts["short_term"]

    def lease(total_key: str, current_key: str, noncurrent_key: str) -> float | None:
        if parts[total_key] is not None:
            return parts[total_key]
        if parts[current_key] is None and parts[noncurrent_key] is None:
            return None
        return (parts[current_key] or 0.0) + (parts[noncurrent_key] or 0.0)

    if long_term is None and parts["notes"] is not None:
        long_term = parts["notes"]
    operating = lease("operating_lease", "operating_lease_current", "operating_lease_noncurrent")
    finance = lease("finance_lease", "finance_lease_current", "finance_lease_noncurrent")
    items = [long_term, short_term, operating, finance]
    if all(v is None for v in items):
        return None
    return float(sum(v for v in items if v is not None))


# ------------------------------------------------------------------ loading


def load_company_facts(
    ticker: str,
    cik: int,
    predecessors: tuple[int, ...] = (),
    max_age_days: float = CACHE_MAX_AGE_DAYS,
    splits: pd.Series | None = None,
    new_filings: bool = False,
) -> CompanyFacts:
    """Facts plus SIC code for one company (cached compact extract). Raises SecDataError.

    ``predecessors`` are CIKs of the same business before a reorganization into a new
    holding company (Google Inc. before Alphabet, for example). Their facts are
    merged in; point-in-time selection then uses whichever filing was latest.
    ``new_filings`` also refreshes the cached copy as soon as the company files a 10-K
    or 10-Q after it was written (the live agents ask for this; studies do not).
    """
    facts, meta = _load_one(cik, max_age_days, new_filings)
    frames = [facts]
    for old in predecessors:
        try:
            frames.append(_load_one(old, max_age_days)[0])
        except SecDataError as exc:
            logger.warning("Predecessor CIK %s of %s unavailable: %s", old, ticker, exc)
    if len(frames) > 1:
        facts = pd.concat(frames, ignore_index=True)
    return CompanyFacts(
        ticker=ticker.upper(), cik=cik, name=meta.get("name"), sic=meta.get("sic"),
        sic_description=meta.get("sic_description"), facts=facts,
        splits=splits if splits is not None else pd.Series(dtype=float),
    )


def _load_one(cik: int, max_age_days: float, new_filings: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = sec_cache_dir()
    facts_path = base / f"facts_v{CONCEPT_VERSION}_{cik:010d}.parquet"
    meta_path = base / f"meta_{cik:010d}.json"
    fresh = _fresh(facts_path, max_age_days) and _fresh(meta_path, max_age_days)
    if fresh and new_filings and filed_since(cik, facts_path.stat().st_mtime):
        fresh = False  # a 10-K or 10-Q came out after the cached copy: read it now, not when the week is up
    if fresh:
        facts = pd.read_parquet(facts_path)
        meta = json.loads(meta_path.read_text())
    else:
        raw = _get_json(FACTS_URL.format(cik=cik))
        facts = extract_facts(raw)
        subs = load_submissions(cik)
        sic = subs.get("sic")
        meta = {
            "name": raw.get("entityName") or subs.get("name"),
            "sic": int(sic) if str(sic or "").isdigit() else None,
            "sic_description": subs.get("sicDescription"),
        }
        facts.to_parquet(facts_path, index=False)
        meta_path.write_text(json.dumps(meta))
    return facts, meta
