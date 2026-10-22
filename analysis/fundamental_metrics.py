"""Owner-style fundamental metrics computed from annual statements.

Definitions follow the owner's investing notes:

- FCF = EBITDA - Maintenance CapEx - Interest - Taxes - Change in Working Capital
  (maintenance CapEx is proxied by D&A, capped at total CapEx; interest and
  taxes use cash paid when reported, else the income statement figures; the
  change in working capital is averaged over up to
  ``settings.WORKING_CAPITAL_AVERAGE_YEARS`` yearly changes to remove one-year noise)
- Two comparison variants are always computed: FCF after all CapEx (growth CapEx
  subtracted too) and the reported FCF (operating cash flow - CapEx).
  ``settings.FCF_DEFINITION`` picks which one feeds EV/FCF (decision_log.md T-05).
- Working capital = Inventories + Accounts Receivable - Accounts Payable
- Market cap = Shares outstanding x Share price (live price)
- EV = Market cap + Debt - Cash (debt includes lease liabilities; cash includes
  short-term investments; both from the latest reported quarter)
- ROIC = EBIT x (1 - T) / Invested Capital, with Invested Capital =
  Equity + Financial Debt + Operating Leases - Short-term Investments
- Reinvestment rate = (Growth CapEx + Acquisitions) / FCF, Growth CapEx = CapEx - D&A
- Net dilution cost = (Stock-based compensation - Buybacks) / Market cap; net
  buyback yield = (Buybacks - Share issuance) / Market cap
- Information only: EV/EBIT, incremental ROIC (change in NOPAT / change in
  invested capital across the available years) and asset growth
- Financial companies use PER, ROE = Net Income / Equity, ROA = Net Income / Total Assets

Everything here is pure (no I/O). Missing inputs stay None and are listed in
``missing``; any simplifying assumption is recorded in ``notes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import config.settings as settings
from data.financial_statements import FinancialStatements
from data.fundamentals import FundamentalsData

FINANCIAL_INDUSTRY_WORDS = ("bank", "insurance", "capital markets", "credit services", "asset management",
                            "mortgage", "financial conglomerates", "financial data")
DEFAULT_TAX_RATE = 0.21
REVERSE_SPLIT_LOOKBACK_YEARS = 20
LIQUIDITY_WINDOW = 20

# Sectors outside the value framework (owner's call, decision_log.md T-12): the debt and ROIC rules exclude
# utilities and REITs by construction, and EV/FCF on REIT depreciation is not meaningful. They are NOT_RATED.
EXCLUDED_SECTORS: tuple[str, ...] = ("Utilities", "Real Estate")


def is_excluded_sector(sector: str | None) -> bool:
    """True when ``sector`` (Yahoo or SIC naming, any case) is outside the value framework."""
    if not sector:
        return False
    return sector.strip().lower() in {x.lower() for x in EXCLUDED_SECTORS}


FCF_DEFINITIONS: dict[str, str] = {
    "owner": "owner FCF (maintenance CapEx)",
    "owner_all_capex": "owner FCF after all CapEx",
    "standard": "reported FCF (operating cash flow - CapEx)",
}


@dataclass
class YearMetrics:
    fiscal_year_end: str
    ebit: float | None = None
    tax_rate: float | None = None
    invested_capital: float | None = None
    roic: float | None = None
    diluted_shares: float | None = None
    basic_shares: float | None = None


@dataclass
class OwnerMetrics:
    ticker: str
    is_financial: bool
    fiscal_year_end: str | None = None
    currency: str | None = None
    currency_mismatch: bool = False

    # Market value
    price: float | None = None
    shares_outstanding: float | None = None
    market_cap: float | None = None
    total_debt: float | None = None
    cash: float | None = None
    enterprise_value: float | None = None
    ev_to_market_cap: float | None = None

    # FCF (owner formula) and its parts, latest fiscal year
    ebitda: float | None = None
    d_and_a: float | None = None
    capex: float | None = None
    maintenance_capex: float | None = None
    growth_capex: float | None = None
    interest: float | None = None
    interest_source: str | None = None
    taxes: float | None = None
    taxes_source: str | None = None
    inventory: float | None = None
    receivables: float | None = None
    payables: float | None = None
    working_capital: float | None = None
    working_capital_prev: float | None = None
    delta_working_capital: float | None = None  # latest year
    delta_working_capital_avg: float | None = None  # average used in the FCF
    delta_working_capital_years: int = 0
    fcf: float | None = None  # the definition selected by settings.FCF_DEFINITION
    fcf_definition: str = "standard"
    fcf_owner: float | None = None
    fcf_all_capex: float | None = None
    reported_fcf: float | None = None
    ev_to_fcf: float | None = None
    ev_to_fcf_owner: float | None = None
    ev_to_fcf_all_capex: float | None = None
    ev_to_reported_fcf: float | None = None
    fcf_yield_on_ev: float | None = None
    ebit: float | None = None
    ev_to_ebit: float | None = None

    # Earnings multiples
    net_income: float | None = None
    net_income_source: str | None = None
    per: float | None = None
    net_cash: float | None = None
    per_ex_cash: float | None = None

    # Returns
    roic: float | None = None
    tax_rate: float | None = None
    invested_capital: float | None = None
    years: list[YearMetrics] = field(default_factory=list)  # newest first
    roe: float | None = None
    roa: float | None = None
    equity: float | None = None
    total_assets: float | None = None
    total_assets_prev: float | None = None

    # Reinvestment
    acquisitions: float | None = None
    acquisitions_reported: bool | None = None  # False: the cash-flow line is missing and counted as 0 (the documented convention)
    growth_capex_floored: bool = False  # CapEx at or below D&A: the proxy gives no growth CapEx
    r_and_d: float | None = None  # information only, outside the owner's reinvestment formula
    reinvestment_rate: float | None = None
    incremental_roic: float | None = None  # change in NOPAT / change in invested capital, oldest to latest year
    incremental_roic_years: float | None = None
    asset_growth: float | None = None  # total assets, latest year vs the year before

    # Shareholder structure
    revenue: float | None = None
    sbc: float | None = None
    sbc_to_revenue: float | None = None
    sbc_to_fcf: float | None = None
    buybacks: float | None = None
    stock_issuance: float | None = None
    buyback_yield: float | None = None
    net_buyback_yield: float | None = None  # (buybacks - issuance) / market cap
    net_dilution_cost: float | None = None  # (stock pay - buybacks) / market cap; negative = buybacks exceed pay
    share_cagr: float | None = None  # per year, diluted average shares
    share_cagr_years: float | None = None
    dilutive_gap: float | None = None  # diluted / basic - 1
    reverse_splits: list[dict[str, Any]] = field(default_factory=list)

    # Trading liquidity
    avg_volume: float | None = None
    avg_dollar_volume: float | None = None
    daily_turnover: float | None = None  # average volume / float shares

    notes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _short_money(value: float, currency: str | None = None) -> str:
    sign = "$" if currency in (None, "USD") else f"{currency} "
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= divisor:
            return f"{sign}{abs(value) / divisor:.1f}{suffix}"
    return f"{sign}{abs(value):,.0f}"


def reinvestment_explanation(capex: float | None, d_and_a: float | None, acquisitions: float | None,
                             acquisitions_reported: bool | None, rate: float | None, currency: str | None = None) -> str | None:
    """Plain-language reason behind a reinvestment rate: the components, a floored growth CapEx and a missing
    acquisitions line. Display only: it never feeds a score."""
    if capex is None or d_and_a is None:
        return None
    capex, d_and_a = abs(capex), abs(d_and_a)
    parts = []
    if capex < d_and_a:
        parts.append(f"CapEx {_short_money(capex, currency)} is below D&A {_short_money(d_and_a, currency)}, so the proxy counts no growth CapEx")
    else:
        parts.append(f"growth CapEx is CapEx {_short_money(capex, currency)} minus D&A {_short_money(d_and_a, currency)}")
    if acquisitions_reported is False:
        parts.append("acquisitions not reported, counted as zero")
    elif acquisitions:
        parts.append(f"acquisitions {_short_money(acquisitions, currency)}")
    else:
        parts.append("no acquisitions")
    head = "n/a" if rate is None else f"{rate * 100:.0f}%"
    return f"{head}: " + "; ".join(parts)


def is_financial_company(sector: str | None, industry: str | None) -> bool:
    if sector and sector.strip().lower() == "financial services":
        return True
    ind = (industry or "").lower()
    return any(word in ind for word in FINANCIAL_INDUSTRY_WORDS)


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _tax_rate(st: FinancialStatements, year: pd.Timestamp) -> tuple[float, bool]:
    """Effective tax rate for ``year`` and whether it was actually reported."""
    provision = st.value("income", "Tax Provision", year)
    pretax = st.value("income", "Pretax Income", year)
    if provision is not None and pretax is not None and pretax > 0:
        return min(max(provision / pretax, 0.0), 0.5), True
    reported = st.value("income", "Tax Rate For Calcs", year)
    if reported is not None and 0 <= reported <= 0.5:
        return reported, True
    return DEFAULT_TAX_RATE, False


def _total_debt(st: FinancialStatements, year: pd.Timestamp) -> float | None:
    debt = st.value("balance", "Total Debt", year)
    if debt is not None:
        return debt
    current = st.value("balance", ("Current Debt And Capital Lease Obligation", "Current Debt"), year)
    long_term = st.value("balance", ("Long Term Debt And Capital Lease Obligation", "Long Term Debt"), year)
    if current is None and long_term is None:
        return None
    return (current or 0.0) + (long_term or 0.0)


def _short_term_investments(st: FinancialStatements, year: pd.Timestamp) -> float | None:
    sti = st.value("balance", ("Other Short Term Investments", "Short Term Investments"), year)
    if sti is not None:
        return sti
    total = st.value("balance", "Cash Cash Equivalents And Short Term Investments", year)
    cash = st.value("balance", "Cash And Cash Equivalents", year)
    if total is not None and cash is not None:
        return max(total - cash, 0.0)
    return None


def _interest(
    st: FinancialStatements, years: list[pd.Timestamp], total_debt: float | None, notes: list[str]
) -> tuple[float | None, str | None]:
    """Interest for the latest year: cash paid, else expense, else the latest year that reports it."""
    for i, year in enumerate(years):
        paid = st.value("cashflow", "Interest Paid Supplemental Data", year)
        expense = st.value("income", ("Interest Expense", "Interest Expense Non Operating"), year)
        value, source = (abs(paid), "cash interest paid") if paid is not None else (
            (abs(expense), "interest expense") if expense is not None else (None, None))
        if value is None:
            continue
        if i > 0:
            label = year.strftime("%Y-%m-%d")
            notes.append(f"Latest 10-K does not report interest; used the FY {label} figure (most recent reported)")
            source = f"{source}, FY {label}"
        return value, source
    if total_debt is not None and total_debt <= 0:
        return 0.0, "no debt"
    return None, None


def _working_capital(st: FinancialStatements, year: pd.Timestamp, notes: list[str]) -> float | None:
    receivables = st.value("balance", ("Accounts Receivable", "Receivables"), year)
    payables = st.value("balance", ("Accounts Payable", "Payables"), year)
    if receivables is None or payables is None:
        return None
    inventory = st.value("balance", "Inventory", year)
    if inventory is None:
        if st.has_row("balance", "Inventory"):
            return None
        inventory = 0.0
        note = "No inventory line reported (treated as 0, typical for service and software companies)"
        if note not in notes:
            notes.append(note)
    return inventory + receivables - payables


def _roic_for_year(st: FinancialStatements, year: pd.Timestamp) -> YearMetrics:
    ym = YearMetrics(fiscal_year_end=year.strftime("%Y-%m-%d"))
    ym.ebit = st.value("income", ("EBIT", "Operating Income"), year)
    ym.tax_rate, _ = _tax_rate(st, year)
    equity = st.value("balance", ("Stockholders Equity", "Common Stock Equity"), year)
    debt = _total_debt(st, year)
    sti = _short_term_investments(st, year) or 0.0
    if equity is not None and debt is not None:
        ym.invested_capital = equity + debt - sti
        if ym.ebit is not None and ym.invested_capital > 0:
            ym.roic = ym.ebit * (1.0 - ym.tax_rate) / ym.invested_capital
    ym.diluted_shares = st.value("income", "Diluted Average Shares", year)
    ym.basic_shares = st.value("income", "Basic Average Shares", year)
    return ym


def _split_adjusted_shares(years: list[YearMetrics], splits: pd.Series) -> list[tuple[pd.Timestamp, float]]:
    """(year end, diluted shares) oldest first, restated for splits Yahoo did not restate."""
    points = [(pd.Timestamp(y.fiscal_year_end), y.diluted_shares) for y in years if y.diluted_shares]
    points.sort()
    if len(points) < 2 or splits.empty:
        return points
    adjusted = [list(p) for p in points]
    for i in range(1, len(adjusted)):
        prev_date, curr_date = adjusted[i - 1][0], adjusted[i][0]
        between = splits[(splits.index > prev_date) & (splits.index <= curr_date)]
        for ratio in between.values:
            observed = adjusted[i][1] / adjusted[i - 1][1]
            if ratio > 0 and abs(observed / ratio - 1.0) < 0.1:
                for j in range(i):  # restate older years to the post-split basis
                    adjusted[j][1] *= ratio
    return [(d, v) for d, v in adjusted]


def _liquidity(ohlcv: pd.DataFrame | None, float_shares: float | None, m: OwnerMetrics) -> None:
    if ohlcv is None or ohlcv.empty or not {"close", "volume"} <= set(ohlcv.columns):
        return
    bars = ohlcv.iloc[:-1] if ohlcv.attrs.get("intraday_partial") else ohlcv
    window = bars.tail(LIQUIDITY_WINDOW)
    if len(window) < 5:
        return
    m.avg_volume = float(window["volume"].mean())
    m.avg_dollar_volume = float((window["close"] * window["volume"]).mean())
    if float_shares and float_shares > 0:
        m.daily_turnover = m.avg_volume / float_shares


def compute_owner_metrics(
    st: FinancialStatements,
    f: FundamentalsData,
    price: float | None = None,
    ohlcv: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
) -> OwnerMetrics:
    """Compute every owner metric that the available data supports."""
    m = OwnerMetrics(ticker=f.ticker, is_financial=is_financial_company(f.sector, f.industry))
    m.currency = f.financial_currency or f.currency
    m.currency_mismatch = f.currency_mismatch
    notes, missing = m.notes, m.missing
    today = today or pd.Timestamp.today()

    _liquidity(ohlcv, f.float_shares, m)

    # Market value (live price x current share count)
    m.price = price if price is not None else f.current_price
    m.shares_outstanding = f.shares_outstanding
    if m.price is not None and m.shares_outstanding:
        m.market_cap = m.price * m.shares_outstanding
    elif f.market_cap is not None:
        m.market_cap = f.market_cap
        notes.append("Share count unavailable; used Yahoo's reported market cap")

    years = st.years() if st.available else []
    latest = years[0] if years else None
    previous = years[1] if len(years) > 1 else None
    if latest is not None:
        m.fiscal_year_end = latest.strftime("%Y-%m-%d")

    # Reverse splits (red flag) come from split history, statements not required.
    cutoff = today - pd.DateOffset(years=REVERSE_SPLIT_LOOKBACK_YEARS)
    for date, ratio in st.splits.items():
        if ratio < 1 and pd.Timestamp(date) >= cutoff:
            m.reverse_splits.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "ratio": float(ratio),
                "label": f"1-for-{round(1 / ratio):g}",
                "years_ago": round((today - pd.Timestamp(date)).days / 365.25, 1),
            })

    if latest is None:
        missing.append("annual statements")
        return m

    # History (ROIC per year, share counts)
    m.years = [_roic_for_year(st, y) for y in years]
    cur = m.years[0]
    m.roic, m.tax_rate, m.invested_capital = cur.roic, cur.tax_rate, cur.invested_capital
    if not _tax_rate(st, latest)[1]:
        notes.append(f"Tax rate not reported; assumed {DEFAULT_TAX_RATE:.0%} for ROIC")
    if cur.invested_capital is not None and cur.invested_capital <= 0:
        notes.append("Invested capital is negative (equity wiped out by buybacks or losses); ROIC not meaningful")

    m.equity = st.value("balance", ("Stockholders Equity", "Common Stock Equity"), latest)
    m.total_assets = st.value("balance", "Total Assets", latest)
    m.revenue = st.value("income", "Total Revenue", latest)
    annual_ni = st.value("income", ("Net Income Common Stockholders", "Net Income"), latest)

    # Net income for PER: TTM when Yahoo has it, else the last fiscal year
    if f.net_income_ttm is not None:
        m.net_income, m.net_income_source = f.net_income_ttm, "TTM"
    elif annual_ni is not None:
        m.net_income, m.net_income_source = annual_ni, f"FY {m.fiscal_year_end}"

    # Debt and cash for EV: latest quarter from Yahoo, else the last annual balance sheet
    m.total_debt = f.total_debt if f.total_debt is not None else _total_debt(st, latest)
    if f.total_cash is not None:
        m.cash = f.total_cash
    else:
        m.cash = st.value("balance", ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"), latest)

    # Returns for financial companies
    if m.net_income is not None:
        if m.equity and m.equity > 0:
            m.roe = m.net_income / m.equity
        if m.total_assets and m.total_assets > 0:
            m.roa = m.net_income / m.total_assets

    comparable = not m.currency_mismatch
    if not comparable:
        notes.append(
            f"Statements are in {f.financial_currency} but the price is in {f.currency}; "
            "market-value ratios (EV, PER, EV/FCF) are not computed"
        )

    # PER and PER ex-cash
    if comparable and m.market_cap and m.net_income:
        if m.net_income > 0:
            m.per = m.market_cap / m.net_income
            # Cash is inventory for banks and insurers, so PER ex-cash only applies to other companies.
            if not m.is_financial and m.cash is not None and m.total_debt is not None:
                m.net_cash = m.cash - m.total_debt
                adjusted_cap = m.market_cap - max(m.net_cash, 0.0)
                m.per_ex_cash = adjusted_cap / m.net_income
        else:
            notes.append("Net income is negative: PER not meaningful")

    # Shareholder structure
    m.sbc = st.value("cashflow", "Stock Based Compensation", latest)
    buybacks = st.value("cashflow", ("Repurchase Of Capital Stock", "Common Stock Payments"), latest)
    m.buybacks = abs(buybacks) if buybacks is not None else None
    issuance = st.value("cashflow", ("Issuance Of Capital Stock", "Common Stock Issuance"), latest)
    m.stock_issuance = abs(issuance) if issuance is not None else None
    m.sbc_to_revenue = _div(m.sbc, m.revenue) if m.revenue and m.revenue > 0 else None
    if comparable and m.market_cap:
        if m.buybacks is not None:
            m.buyback_yield = m.buybacks / m.market_cap
            m.net_buyback_yield = (m.buybacks - (m.stock_issuance or 0.0)) / m.market_cap
        if m.sbc is not None:
            m.net_dilution_cost = (m.sbc - (m.buybacks or 0.0)) / m.market_cap
    m.total_assets_prev = st.value("balance", "Total Assets", previous) if previous is not None else None
    if m.total_assets and m.total_assets_prev and m.total_assets_prev > 0:
        m.asset_growth = m.total_assets / m.total_assets_prev - 1.0
    shares = _split_adjusted_shares(m.years, st.splits)
    if len(shares) >= 2:
        span = (shares[-1][0] - shares[0][0]).days / 365.25
        if span > 0.5 and shares[0][1] > 0:
            m.share_cagr = (shares[-1][1] / shares[0][1]) ** (1 / span) - 1.0
            m.share_cagr_years = round(span, 1)
    if cur.diluted_shares and cur.basic_shares:
        m.dilutive_gap = cur.diluted_shares / cur.basic_shares - 1.0

    if m.is_financial:
        # Working capital, EBITDA-based FCF, EV and ROIC do not describe banks and insurers.
        return m

    # EV = Market cap + Debt - Cash
    if comparable and m.market_cap is not None and m.total_debt is not None and m.cash is not None:
        m.enterprise_value = m.market_cap + m.total_debt - m.cash
        m.ev_to_market_cap = m.enterprise_value / m.market_cap if m.market_cap else None
    else:
        missing.append("enterprise value")

    # FCF = EBITDA - Maintenance CapEx - Interest - Taxes - Change in WC
    m.ebitda = st.value("income", ("EBITDA", "Normalized EBITDA"), latest)
    m.d_and_a = st.value("cashflow", ("Depreciation And Amortization", "Depreciation Amortization Depletion"), latest)
    if m.d_and_a is None:
        m.d_and_a = st.value("income", "Reconciled Depreciation", latest)
    if m.ebitda is None:
        ebit = cur.ebit
        if ebit is not None and m.d_and_a is not None:
            m.ebitda = ebit + m.d_and_a
            notes.append("EBITDA computed as EBIT + D&A")
    capex = st.value("cashflow", ("Capital Expenditure", "Purchase Of PPE"), latest)
    m.capex = abs(capex) if capex is not None else None
    if m.capex is not None and m.d_and_a is not None:
        m.maintenance_capex = min(m.capex, abs(m.d_and_a))
        m.growth_capex = max(m.capex - abs(m.d_and_a), 0.0)

    m.interest, m.interest_source = _interest(st, years, m.total_debt, notes)
    taxes_paid = st.value("cashflow", "Income Tax Paid Supplemental Data", latest)
    if taxes_paid is not None:
        m.taxes, m.taxes_source = abs(taxes_paid), "cash taxes paid"
    else:
        provision = st.value("income", "Tax Provision", latest)
        if provision is not None:
            m.taxes, m.taxes_source = provision, "tax provision"

    m.working_capital = _working_capital(st, latest, notes)
    if previous is not None:
        m.working_capital_prev = _working_capital(st, previous, notes)
    if m.working_capital is not None and m.working_capital_prev is not None:
        m.delta_working_capital = m.working_capital - m.working_capital_prev
    m.inventory = st.value("balance", "Inventory", latest)
    m.receivables = st.value("balance", ("Accounts Receivable", "Receivables"), latest)
    m.payables = st.value("balance", ("Accounts Payable", "Payables"), latest)

    # Change in working capital averaged over the last N yearly changes (one-year swings are noise).
    max_deltas = max(1, int(getattr(settings, "WORKING_CAPITAL_AVERAGE_YEARS", 3)))
    wc_levels = [_working_capital(st, y, notes) for y in years[: max_deltas + 1]]
    deltas = [a - b for a, b in zip(wc_levels, wc_levels[1:]) if a is not None and b is not None]
    if deltas:
        m.delta_working_capital_avg = sum(deltas) / len(deltas)
        m.delta_working_capital_years = len(deltas)
        if len(deltas) > 1:
            notes.append(f"Change in working capital averaged over {len(deltas)} years")

    parts = {
        "EBITDA": m.ebitda,
        "maintenance CapEx": m.maintenance_capex,
        "interest": m.interest,
        "taxes": m.taxes,
        "change in working capital": m.delta_working_capital_avg,
    }
    absent = [k for k, v in parts.items() if v is None]
    if absent:
        missing.append(f"FCF inputs ({', '.join(absent)})")
    else:
        m.fcf_owner = m.ebitda - m.maintenance_capex - m.interest - m.taxes - m.delta_working_capital_avg
        if m.growth_capex is not None:
            m.fcf_all_capex = m.fcf_owner - m.growth_capex

    reported = st.value("cashflow", "Free Cash Flow", latest)
    if reported is None:
        ocf = st.value("cashflow", "Operating Cash Flow", latest)
        if ocf is not None and m.capex is not None:
            reported = ocf - m.capex
    m.reported_fcf = reported

    definition = str(getattr(settings, "FCF_DEFINITION", "standard"))
    if definition not in FCF_DEFINITIONS:
        definition = "standard"
    m.fcf_definition = definition
    m.fcf = {"owner": m.fcf_owner, "owner_all_capex": m.fcf_all_capex, "standard": m.reported_fcf}[definition]

    if m.enterprise_value is not None:
        for attr, value in (("ev_to_fcf_owner", m.fcf_owner), ("ev_to_fcf_all_capex", m.fcf_all_capex),
                            ("ev_to_reported_fcf", m.reported_fcf)):
            if value is not None and value > 0:
                setattr(m, attr, m.enterprise_value / value)
        if m.fcf is not None:
            if m.fcf > 0:
                m.ev_to_fcf = m.enterprise_value / m.fcf
            if m.enterprise_value > 0:
                m.fcf_yield_on_ev = m.fcf / m.enterprise_value
        m.ebit = cur.ebit
        if m.ebit is not None and m.ebit > 0:
            m.ev_to_ebit = m.enterprise_value / m.ebit
    if m.fcf is not None and m.sbc is not None and m.fcf > 0:
        m.sbc_to_fcf = m.sbc / m.fcf

    # Reinvestment rate = (Growth CapEx + Acquisitions) / FCF
    acquisitions = st.value("cashflow", ("Purchase Of Business", "Net Business Purchase And Sale"), latest)
    m.acquisitions = abs(acquisitions) if acquisitions is not None and acquisitions < 0 else 0.0
    m.acquisitions_reported = acquisitions is not None  # a missing line is counted as 0 for the rate, but flagged
    m.growth_capex_floored = m.capex is not None and m.d_and_a is not None and m.capex < abs(m.d_and_a)
    m.r_and_d = st.value("income", ("Research And Development", "Research Development"), latest)
    if m.growth_capex is not None and m.fcf is not None and m.fcf > 0:
        m.reinvestment_rate = (m.growth_capex + m.acquisitions) / m.fcf

    # Incremental ROIC: NOPAT change over invested-capital change, oldest to latest year (information)
    with_ic = [y for y in m.years if y.ebit is not None and y.invested_capital is not None and y.tax_rate is not None]
    if len(with_ic) >= 2:
        newest, oldest = with_ic[0], with_ic[-1]
        d_ic = newest.invested_capital - oldest.invested_capital
        d_nopat = newest.ebit * (1 - newest.tax_rate) - oldest.ebit * (1 - oldest.tax_rate)
        span = (pd.Timestamp(newest.fiscal_year_end) - pd.Timestamp(oldest.fiscal_year_end)).days / 365.25
        if d_ic > 0 and span > 0.5:
            m.incremental_roic = d_nopat / d_ic
            m.incremental_roic_years = round(span, 1)

    return m
