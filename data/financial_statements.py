"""Annual financial statements and split history from Yahoo Finance.

Owner-style metrics (FCF from EBITDA, working capital, ROIC, reinvestment,
share count trends) need the full annual income statement, balance sheet and
cash flow statement, not the summary ratios in ``Ticker.info``. Annual (10-K)
figures are used because they are audited and complete; quarterly statements
often omit lines such as interest expense.

Each network call is cached and raises on failure (failures are not cached);
``fetch_statements`` never raises and reports ``available=False`` instead.

Yahoo can take weeks to add a newly filed 10-K. ``fetch_current_statements`` (the
live agents) uses the SEC 10-K figures instead when the SEC already has a newer
fiscal year, so today's price is never paired with last year's cash flow.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yfinance as yf

import config.settings as settings
from data.cache_utils import ttl_cache
from data.yahoo import yahoo_call

logger = logging.getLogger(__name__)


@dataclass
class FinancialStatements:
    """Annual statements with rows as line items and columns as fiscal year ends (newest first)."""

    ticker: str
    available: bool
    income: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    splits: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    error: str | None = None
    splits_error: str | None = None
    source: str = "Yahoo Finance"
    basis_note: str | None = None  # why this source was used, when it is not the usual one

    def years(self) -> list[pd.Timestamp]:
        """Fiscal year ends that have an income statement, newest first."""
        if self.income.empty:
            return []
        cols = [c for c in self.income.columns if self.income[c].notna().sum() > 3]
        return sorted((pd.Timestamp(c) for c in cols), reverse=True)

    def value(self, statement: str, rows: str | tuple[str, ...], year: pd.Timestamp | None = None) -> float | None:
        """First available finite value among ``rows`` for ``year`` (default: latest year)."""
        frame: pd.DataFrame = getattr(self, statement)
        if frame.empty:
            return None
        if year is None:
            years = self.years()
            if not years:
                return None
            year = years[0]
        col = next((c for c in frame.columns if pd.Timestamp(c) == pd.Timestamp(year)), None)
        if col is None:
            return None
        for row in (rows,) if isinstance(rows, str) else rows:
            if row in frame.index:
                v = frame.at[row, col]
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f):
                    return f
        return None

    def has_row(self, statement: str, row: str) -> bool:
        frame: pd.DataFrame = getattr(self, statement)
        return not frame.empty and row in frame.index


def _clean(frame: Any) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    df.columns = [pd.Timestamp(c).tz_localize(None) if pd.Timestamp(c).tzinfo else pd.Timestamp(c) for c in df.columns]
    return df.sort_index(axis=1, ascending=False)


@ttl_cache(settings.CACHE_TTL_FUNDAMENTALS)
def _load_statements(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tk = yf.Ticker(symbol)
    income = _clean(yahoo_call(lambda: tk.income_stmt, f"{symbol} income statement"))
    balance = _clean(yahoo_call(lambda: tk.balance_sheet, f"{symbol} balance sheet"))
    cashflow = _clean(yahoo_call(lambda: tk.cashflow, f"{symbol} cash flow"))
    if income.empty and balance.empty and cashflow.empty:
        raise LookupError(f"Yahoo Finance returned no financial statements for {symbol}")
    return income, balance, cashflow


@ttl_cache(settings.CACHE_TTL_FUNDAMENTALS)
def _load_splits(symbol: str) -> pd.Series:
    splits = yahoo_call(lambda: yf.Ticker(symbol).splits, f"{symbol} splits")
    if splits is None or not isinstance(splits, pd.Series):
        return pd.Series(dtype=float)
    out = splits.astype(float).copy()
    out.index = pd.to_datetime(out.index).tz_localize(None) if getattr(out.index, "tz", None) else pd.to_datetime(out.index)
    return out[out > 0].sort_index()


def fetch_statements(ticker: str) -> FinancialStatements:
    """Annual statements plus split history. Never raises."""
    symbol = ticker.strip().upper()
    try:
        income, balance, cashflow = _load_statements(symbol)
    except Exception as exc:  # noqa: BLE001 - provider failures become unavailable data
        logger.warning("Could not fetch statements for %s: %s", symbol, exc)
        return FinancialStatements(ticker=symbol, available=False, error=f"{type(exc).__name__}: {exc}"[:240])

    statements = FinancialStatements(ticker=symbol, available=True, income=income, balance=balance, cashflow=cashflow)
    try:
        statements.splits = _load_splits(symbol)
    except Exception as exc:  # noqa: BLE001 - split history is optional
        logger.warning("Could not fetch split history for %s: %s", symbol, exc)
        statements.splits_error = f"{type(exc).__name__}: {exc}"[:240]
    return statements


NEWER_YEAR_DAYS = 60  # fiscal year ends closer than this are the same year (52/53-week years, Yahoo's month-end dates)
CURRENT_TTL_SECONDS = 600  # the fundamentals and shareholder agents ask for the same company in one analysis


def _sec_statements(symbol: str, splits: pd.Series) -> FinancialStatements | None:
    """The SEC 10-K statements as known today, refreshed when a new 10-K or 10-Q is out; None without access."""
    from data.sec_edgar import has_sec_access, load_company_facts, load_ticker_map

    if not has_sec_access():
        return None
    try:
        cik = load_ticker_map().get(symbol)
        if cik is None:
            return None
        facts = load_company_facts(symbol, cik, splits=splits, new_filings=True)
        st = facts.statements_asof(pd.Timestamp.today().normalize(), max_years=4)
    except Exception as exc:  # noqa: BLE001 - Yahoo's statements stay in use
        logger.info("SEC statements unavailable for %s: %s", symbol, exc)
        return None
    return st if st.available and st.years() else None


@ttl_cache(CURRENT_TTL_SECONDS)
def fetch_current_statements(ticker: str) -> FinancialStatements:
    """The newest annual statements for live scoring: Yahoo's, or the SEC 10-K when it has a newer fiscal year. Never raises.

    The point-in-time panel scores 10-K figures from the day they are filed (core/value_study.py), so the live agents
    should not wait for Yahoo. On the same fiscal year Yahoo is kept (decision_log.md D-14 and D-47).
    """
    yahoo = fetch_statements(ticker)
    sec = _sec_statements(yahoo.ticker, yahoo.splits)
    if sec is None:
        return yahoo
    yahoo_years = yahoo.years() if yahoo.available else []
    newest = sec.years()[0]
    if yahoo_years and newest - yahoo_years[0] < pd.Timedelta(days=NEWER_YEAR_DAYS):
        return yahoo
    sec.splits, sec.splits_error = yahoo.splits, yahoo.splits_error
    sec.basis_note = (f"Statements from the SEC 10-K for the fiscal year to {newest:%Y-%m-%d}: "
                      + (f"Yahoo's still end at {yahoo_years[0]:%Y-%m-%d}" if yahoo_years else "Yahoo has none"))
    return sec
