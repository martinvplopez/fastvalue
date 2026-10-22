"""Cost of revenue and gross profit from SEC EDGAR XBRL, as a sidecar of ``data/sec_edgar.py``.

The Competitive Position agent needs days payable (payables / cost of revenue) and gross margin. The main SEC
extract does not carry those two concepts, and adding them there would bump ``CONCEPT_VERSION``, rebuild all
cached extracts and could move ``filing_dates()`` (a 10-K/A that only tags gross profit would add a date), which
the value study replays. This module therefore keeps its own small extract per company, under
``data/cache/sec/margins_v1_<cik>.parquet``, from one more companyfacts download through the same throttle and
contact header. Nothing that exists changes (decision_log.md D-34).

A figure restated in a later filing replaces the original only from the restatement's filing date, as in
``CompanyFacts.statements_asof``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.sec_edgar import (
    ANNUAL_DAYS,
    ANNUAL_FORMS,
    CACHE_MAX_AGE_DAYS,
    FACTS_URL,
    _fresh,
    _get_json,
    filed_since,
    sec_cache_dir,
)

logger = logging.getLogger(__name__)

# Row name -> us-gaap concepts in priority order (the first with a value for a fiscal year wins).
MARGIN_CONCEPTS: dict[str, tuple[str, ...]] = {
    "cogs": (
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
    ),
    "gross_profit": ("GrossProfit",),
}
COLUMNS = ["concept", "start", "end", "val", "filed", "form"]
_WANTED = {c for concepts in MARGIN_CONCEPTS.values() for c in concepts}


def _margin_path(cik: int) -> Path:
    return sec_cache_dir() / f"margins_v1_{cik:010d}.parquet"


def extract_margin_facts(raw: dict) -> pd.DataFrame:
    """Frame of the cost of revenue and gross profit facts of one companyfacts payload."""
    rows: list[tuple] = []
    for concept, body in ((raw.get("facts") or {}).get("us-gaap") or {}).items():
        if concept not in _WANTED:
            continue
        for entry in (body.get("units") or {}).get("USD", []):
            if entry.get("val") is None or not entry.get("end") or not entry.get("filed") or not entry.get("start"):
                continue
            rows.append((concept, entry["start"], entry["end"], float(entry["val"]), entry["filed"], entry.get("form") or ""))
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("start", "end", "filed"):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame.dropna(subset=["start", "end", "filed"]).reset_index(drop=True)


def load_margin_facts(cik: int, max_age_days: float = CACHE_MAX_AGE_DAYS, new_filings: bool = False) -> pd.DataFrame:
    """Margin facts of one company (cached). Raises ``SecDataError`` / ``SecConfigError`` like ``load_company_facts``,
    and refreshes on a new 10-K or 10-Q the same way when ``new_filings`` is set."""
    path = _margin_path(cik)
    if _fresh(path, max_age_days) and not (new_filings and filed_since(cik, path.stat().st_mtime)):
        return pd.read_parquet(path)
    frame = extract_margin_facts(_get_json(FACTS_URL.format(cik=cik)))
    frame.to_parquet(path, index=False)
    return frame


def margins_asof(frame: pd.DataFrame, date: pd.Timestamp) -> dict[pd.Timestamp, dict[str, float]]:
    """Cost of revenue and gross profit per fiscal year end, as known on ``date``.

    Only annual facts (10-K, 10-K/A, 10-KT; 340 to 380 days long) filed on or before ``date`` count. Per fiscal
    year end the first concept in priority order wins, and a later filing replaces an earlier one only from its
    own filing date.
    """
    if frame is None or frame.empty:
        return {}
    days = (frame["end"] - frame["start"]).dt.days
    known = frame[frame["form"].isin(ANNUAL_FORMS) & days.between(*ANNUAL_DAYS) & (frame["filed"] <= pd.Timestamp(date))]
    if known.empty:
        return {}
    latest = known.sort_values("filed", kind="stable").groupby(["concept", "end"], sort=False).tail(1)
    values = {(c, e): v for c, e, v in zip(latest["concept"], latest["end"], latest["val"])}
    out: dict[pd.Timestamp, dict[str, float]] = {}
    for end in sorted({e for _, e in values}, reverse=True):
        row: dict[str, float] = {}
        for name, concepts in MARGIN_CONCEPTS.items():
            value = next((values[(c, end)] for c in concepts if (c, end) in values), None)
            if value is not None:
                row[name] = value
        if row:
            out[end] = row
    return out
