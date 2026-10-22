"""Company fundamentals, analyst consensus, earnings surprises and the 10Y Treasury yield.

All values come from Yahoo Finance via yfinance. Every field is optional: Yahoo omits
fields that do not apply (banks report no current ratio, loss makers no trailing P/E),
and a missing value stays ``None`` instead of being estimated.

The three network sub-fetches (quote summary, earnings history, ``^TNX``) are cached
independently and raise on failure, so a transient Yahoo error is retried on the next
call instead of being cached for hours. :func:`fetch_fundamentals` itself never raises:
it returns ``available=False`` with an ``error`` when the quote summary cannot be
loaded, and records earnings or Treasury failures on the result (partial data).
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

TREASURY_10Y_SYMBOL = "^TNX"
EARNINGS_QUARTERS = 4

# Metric fields that prove Yahoo returned real data (an unknown symbol returns
# a near-empty dict such as {"trailingPegRatio": None}).
_METRIC_FIELDS = (
    "marketCap",
    "currentPrice",
    "regularMarketPrice",
    "trailingPE",
    "forwardPE",
    "priceToSalesTrailing12Months",
    "revenueGrowth",
    "operatingMargins",
    "returnOnEquity",
)


@dataclass(frozen=True)
class EarningsSurprise:
    """One reported quarter versus the consensus EPS estimate."""

    quarter: str | None  # ISO date of the fiscal quarter end
    eps_actual: float | None
    eps_estimate: float | None
    surprise_pct: float | None  # percent: 4.5 means EPS beat the estimate by 4.5%

    @property
    def outcome(self) -> str | None:
        """Return "beat", "miss", "in line", or None when either EPS figure is missing."""
        if self.eps_actual is None or self.eps_estimate is None:
            return None
        if self.eps_actual > self.eps_estimate:
            return "beat"
        if self.eps_actual < self.eps_estimate:
            return "miss"
        return "in line"


@dataclass
class FundamentalsData:
    """Fundamental snapshot for one ticker. Ratios and growth rates are fractions (0.12 = 12%)."""

    ticker: str
    available: bool
    error: str | None = None

    quote_type: str | None = None  # EQUITY, ETF, INDEX, MUTUALFUND, CRYPTOCURRENCY, ...
    short_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None  # quote currency (price, market cap, targets)
    financial_currency: str | None = None  # currency of the reported statements (FCF)

    current_price: float | None = None
    market_cap: float | None = None

    trailing_pe: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_sales: float | None = None
    price_to_book: float | None = None
    ev_to_ebitda: float | None = None
    free_cashflow: float | None = None

    revenue_growth: float | None = None
    earnings_growth: float | None = None
    earnings_quarterly_growth: float | None = None

    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    return_on_equity: float | None = None
    debt_to_equity: float | None = None  # ratio: 0.8 means debt is 80% of equity (Yahoo reports percent)
    current_ratio: float | None = None

    recommendation_mean: float | None = None  # 1 = strong buy ... 5 = sell
    recommendation_key: str | None = None
    analyst_count: int | None = None
    target_mean_price: float | None = None
    target_low_price: float | None = None
    target_high_price: float | None = None
    short_percent_of_float: float | None = None

    most_recent_quarter: str | None = None  # ISO date of the latest reported fiscal quarter

    # Capital structure and ownership (latest reported quarter)
    shares_outstanding: float | None = None
    float_shares: float | None = None
    total_debt: float | None = None  # financial debt incl. leases, most recent quarter
    total_cash: float | None = None  # cash + short-term investments, most recent quarter
    net_income_ttm: float | None = None  # net income to common, trailing twelve months
    held_percent_insiders: float | None = None  # fraction of shares held by insiders
    officers: list[dict[str, Any]] = field(default_factory=list)  # name, title, age, total_pay

    earnings: list[EarningsSurprise] = field(default_factory=list)  # oldest first, up to 4
    earnings_error: str | None = None

    treasury_10y: float | None = None  # percent, e.g. 4.25
    treasury_as_of: str | None = None
    treasury_error: str | None = None

    @property
    def currency_mismatch(self) -> bool:
        """True when statements are reported in a different currency than the quote (many ADRs)."""
        return bool(self.currency and self.financial_currency and self.currency != self.financial_currency)

    @property
    def fcf_yield(self) -> float | None:
        """Free cash flow / market cap as a fraction, or None when missing or not comparable."""
        if self.free_cashflow is None or self.market_cap is None or self.market_cap <= 0:
            return None
        if self.currency_mismatch:
            return None
        return self.free_cashflow / self.market_cap


# ---------------------------------------------------------------- parsing helpers


def _num(value: Any) -> float | None:
    """Finite float or None (Yahoo sometimes sends strings such as 'Infinity')."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    out = " ".join(str(value).split())
    return out or None


def _iso_date(value: Any) -> str | None:
    """ISO date from an epoch-seconds number or any timestamp-like value."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return pd.Timestamp(int(value), unit="s").strftime("%Y-%m-%d")
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        return None


def _short_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message[:200]}" if message else type(exc).__name__


def _from_info(symbol: str, info: dict[str, Any]) -> FundamentalsData:
    peg = _num(info.get("pegRatio"))
    if peg is None:
        peg = _num(info.get("trailingPegRatio"))
    debt_to_equity = _num(info.get("debtToEquity"))
    analysts = _num(info.get("numberOfAnalystOpinions"))
    price = _num(info.get("currentPrice"))
    if price is None:
        price = _num(info.get("regularMarketPrice"))
    quote_type = _text(info.get("quoteType"))

    return FundamentalsData(
        ticker=symbol,
        available=True,
        quote_type=quote_type.upper() if quote_type else None,
        short_name=_text(info.get("shortName") or info.get("longName")),
        sector=_text(info.get("sector")),
        industry=_text(info.get("industry")),
        currency=_text(info.get("currency")),
        financial_currency=_text(info.get("financialCurrency")),
        current_price=price,
        market_cap=_num(info.get("marketCap")),
        trailing_pe=_num(info.get("trailingPE")),
        forward_pe=_num(info.get("forwardPE")),
        peg_ratio=peg,
        price_to_sales=_num(info.get("priceToSalesTrailing12Months")),
        price_to_book=_num(info.get("priceToBook")),
        ev_to_ebitda=_num(info.get("enterpriseToEbitda")),
        free_cashflow=_num(info.get("freeCashflow")),
        revenue_growth=_num(info.get("revenueGrowth")),
        earnings_growth=_num(info.get("earningsGrowth")),
        earnings_quarterly_growth=_num(info.get("earningsQuarterlyGrowth")),
        gross_margin=_num(info.get("grossMargins")),
        operating_margin=_num(info.get("operatingMargins")),
        profit_margin=_num(info.get("profitMargins")),
        return_on_equity=_num(info.get("returnOnEquity")),
        debt_to_equity=debt_to_equity / 100.0 if debt_to_equity is not None else None,
        current_ratio=_num(info.get("currentRatio")),
        recommendation_mean=_num(info.get("recommendationMean")),
        recommendation_key=_text(info.get("recommendationKey")),
        analyst_count=int(analysts) if analysts is not None else None,
        target_mean_price=_num(info.get("targetMeanPrice")),
        target_low_price=_num(info.get("targetLowPrice")),
        target_high_price=_num(info.get("targetHighPrice")),
        short_percent_of_float=_num(info.get("shortPercentOfFloat")),
        most_recent_quarter=_iso_date(_num(info.get("mostRecentQuarter"))),
        shares_outstanding=_num(info.get("sharesOutstanding")) or _num(info.get("impliedSharesOutstanding")),
        float_shares=_num(info.get("floatShares")),
        total_debt=_num(info.get("totalDebt")),
        total_cash=_num(info.get("totalCash")),
        net_income_ttm=_num(info.get("netIncomeToCommon")),
        held_percent_insiders=_num(info.get("heldPercentInsiders")),
        officers=_parse_officers(info.get("companyOfficers")),
    )


def _parse_officers(raw: Any) -> list[dict[str, Any]]:
    """Name, title, age and total pay of listed officers (untrusted text, kept plain)."""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw[:12]:
        if not isinstance(item, dict):
            continue
        name, title = _text(item.get("name")), _text(item.get("title"))
        if not name and not title:
            continue
        out.append({
            "name": name,
            "title": title,
            "age": _num(item.get("age")),
            "total_pay": _num(item.get("totalPay")),
        })
    return out


def _parse_earnings(frame: Any) -> tuple[EarningsSurprise, ...]:
    """Last ``EARNINGS_QUARTERS`` rows of ``Ticker.earnings_history``, oldest first."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return ()
    df = frame.copy()
    if "quarter" in df.columns:
        df = df.set_index("quarter")
    try:
        df = df.sort_index()
    except TypeError:
        pass

    rows: list[EarningsSurprise] = []
    for quarter, row in df.tail(EARNINGS_QUARTERS).iterrows():
        actual = _num(row.get("epsActual"))
        estimate = _num(row.get("epsEstimate"))
        if actual is not None and estimate is not None and estimate != 0:
            surprise = (actual - estimate) / abs(estimate) * 100.0
        else:
            # yfinance reports surprisePercent as a fraction (0.045 = 4.5%).
            fraction = _num(row.get("surprisePercent"))
            surprise = fraction * 100.0 if fraction is not None else None
        if actual is None and estimate is None and surprise is None:
            continue
        rows.append(EarningsSurprise(_iso_date(quarter), actual, estimate, surprise))
    return tuple(rows)


# ---------------------------------------------------------------- cached network calls


@ttl_cache(settings.CACHE_TTL_FUNDAMENTALS)
def _load_info(symbol: str) -> dict[str, Any]:
    """Raw ``Ticker.info``; raises when Yahoo returns nothing usable."""
    info = yahoo_call(lambda: yf.Ticker(symbol).info, f"{symbol} info")
    if not isinstance(info, dict):
        raise TypeError(f"unexpected info payload of type {type(info).__name__}")
    if not info.get("quoteType") and all(info.get(k) is None for k in _METRIC_FIELDS):
        raise LookupError(f"Yahoo Finance returned no fundamentals for {symbol}")
    return dict(info)


@ttl_cache(settings.CACHE_TTL_FUNDAMENTALS)
def _load_earnings(symbol: str) -> tuple[EarningsSurprise, ...]:
    return _parse_earnings(yahoo_call(lambda: yf.Ticker(symbol).earnings_history, f"{symbol} earnings"))


@ttl_cache(settings.CACHE_TTL_MACRO)
def _load_treasury_10y() -> tuple[float, str | None]:
    """Latest 10Y Treasury yield in percent and its date."""
    history = yahoo_call(lambda: yf.Ticker(TREASURY_10Y_SYMBOL).history(period="10d"), "10Y yield")
    if history is None or history.empty or "Close" not in history.columns:
        raise LookupError(f"no {TREASURY_10Y_SYMBOL} quotes returned")
    closes = history["Close"].dropna()
    if closes.empty:
        raise LookupError(f"no {TREASURY_10Y_SYMBOL} closing values returned")
    value = float(closes.iloc[-1])
    if value > 20:  # some feeds quote ^TNX in tenths of a percent (42.5 = 4.25%)
        value /= 10.0
    if not 0 < value <= 20:
        raise ValueError(f"implausible 10Y Treasury yield {value}")
    return value, _iso_date(closes.index[-1])


# ---------------------------------------------------------------- public API


def fetch_fundamentals(ticker: str) -> FundamentalsData:
    """
    Fetch fundamentals, the last four earnings surprises and the 10Y Treasury yield.

    Never raises. The earnings history and Treasury yield fail independently:
    their errors are recorded in ``earnings_error`` / ``treasury_error``.
    Non-equity symbols return early with only the profile fields filled.
    """
    symbol = ticker.strip().upper()
    try:
        info = _load_info(symbol)
    except Exception as exc:  # noqa: BLE001 - provider failures become unavailable data
        logger.warning("Could not fetch fundamentals for %s: %s", symbol, exc)
        return FundamentalsData(
            ticker=symbol,
            available=False,
            error=f"Yahoo Finance fundamentals request failed ({_short_error(exc)})",
        )

    data = _from_info(symbol, info)
    if data.quote_type and data.quote_type != "EQUITY":
        return data

    try:
        data.earnings = list(_load_earnings(symbol))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch earnings history for %s: %s", symbol, exc)
        data.earnings_error = _short_error(exc)

    try:
        data.treasury_10y, data.treasury_as_of = _load_treasury_10y()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch the 10Y Treasury yield: %s", exc)
        data.treasury_error = _short_error(exc)

    return data
