"""Fundamentals & Valuation Domain Agent (owner's value framework).

Non-financial companies:
- Valuation: EV/FCF (the primary ratio, market average 16-17x), PER or PER
  ex-cash (market average 16x), and EV versus market cap (debt load).
- Returns: ROIC (Damodaran NOPAT on invested capital, market average 13-14%)
  and how consistently it stayed high (moat signal).
- Reinvestment: the compounder profile (high ROIC with a reinvestment rate
  above 20%) is reported but no longer rewarded: in the 2011-2026 S&P 500
  point-in-time test the organic reinvestment rate scored backwards (t -2.4)
  and the compounder flag lost 1% a year, in line with the asset-growth
  literature (decision_log.md T-07). Heavy reinvestment at a low ROIC is
  still penalized.
- Cash yield: FCF / EV versus the 10-year Treasury (small weight).
- ROIC history: Yahoo gives about four years; when ``SEC_USER_AGENT`` is set
  the agent extends it to ten years from SEC XBRL filings (T-06).

Financial companies (banks, insurers): PER, ROE and ROA. EV, working capital
and EV/FCF do not describe them. Utilities and REITs are scored but flagged:
the debt and ROIC rules exclude them by construction (T-12).

All formulas live in ``analysis/fundamental_metrics.py``; every rule reports its
signed points in ``score_breakdown`` and every input appears in the metrics table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import quote

import pandas as pd

from agents.base import LIVE, OWNER, PARTIAL, AgentAssessment, BaseAgent
from agents.registry import register_agent
from analysis.fundamental_metrics import (
    EXCLUDED_SECTORS,
    FCF_DEFINITIONS,
    OwnerMetrics,
    YearMetrics,
    _roic_for_year,
    compute_owner_metrics,
    is_excluded_sector,
    reinvestment_explanation,
)
from data.financial_statements import FinancialStatements, fetch_current_statements
from data.fundamentals import FundamentalsData, fetch_fundamentals

logger = logging.getLogger(__name__)

# Owner's reference levels
MARKET_AVG_PER = 16.0
MARKET_AVG_ROIC = 0.135
HIGH_ROIC = 0.15
GOOD_REINVESTMENT = 0.20

MAX_CONFIDENCE = 85.0
BASE_CONFIDENCE = 30.0
CONFIDENCE_PER_COMPONENT = 12.0
ROIC_HISTORY_YEARS = 10  # from SEC filings when available (Yahoo serves about 4)
DURABLE_MOAT_YEARS = 8

COMPONENTS = ("valuation", "returns", "reinvestment", "cash_yield")


@dataclass(frozen=True)
class _Rule:
    component: str
    factor: str
    points: float


class _Scorecard:
    def __init__(self) -> None:
        self.rules: list[_Rule] = []
        self.notes: list[str] = []

    def add(self, component: str, factor: str, points: float) -> None:
        self.rules.append(_Rule(component, factor, float(points)))

    def components(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {c: None for c in COMPONENTS}
        for r in self.rules:
            out[r.component] = (out[r.component] or 0.0) + r.points
        return out

    @property
    def total(self) -> float:
        return sum(r.points for r in self.rules)


def _below(value: float, steps: tuple[tuple[float, float], ...], otherwise: float) -> float:
    """Points of the first (bound, points) step with ``value < bound``."""
    for bound, points in steps:
        if value < bound:
            return points
    return otherwise


def _at_least(value: float, steps: tuple[tuple[float, float], ...], otherwise: float) -> float:
    """Points of the first (threshold, points) step with ``value >= threshold``."""
    for threshold, points in steps:
        if value >= threshold:
            return points
    return otherwise


# ---------------------------------------------------------------- formatting


def _money(value: float | None, currency: str | None = None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    size = abs(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if size >= divisor:
            body = f"{size / divisor:.2f}{suffix}"
            break
    else:
        body = f"{size:,.0f}"
    return f"{sign}${body}" if currency in (None, "USD") else f"{sign}{body} {currency}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _mult(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}x"


# ---------------------------------------------------------------- scoring


def _period(m: OwnerMetrics) -> str:
    """The fiscal year the statement figures cover, in words ("year to Sep 2026"), so a rule can be checked against the filings."""
    if not m.fiscal_year_end:
        return "latest fiscal year"
    return f"year to {pd.Timestamp(m.fiscal_year_end):%b %Y}"


def _ni_period(m: OwnerMetrics) -> str:
    return "last twelve months" if m.net_income_source == "TTM" else _period(m)


def _score_valuation(m: OwnerMetrics, card: _Scorecard) -> None:
    if m.fcf is not None and m.enterprise_value is not None:
        label = FCF_DEFINITIONS.get(m.fcf_definition, m.fcf_definition)
        if m.fcf <= 0:
            card.add("valuation", f"FCF is negative ({_money(m.fcf, m.currency)}, {label}, {_period(m)}): no EV/FCF", -12)
        else:
            pts = _below(m.ev_to_fcf, ((8, 14), (12, 10), (16, 5), (20, 0), (25, -4), (35, -8)), -12)
            card.add("valuation", f"EV/FCF {m.ev_to_fcf:.1f}x, {label}, {_period(m)} (market average 16-17x)", pts)

    # Earnings multiple: PER ex-cash when the company holds net cash, else plain PER.
    multiple, label = (m.per_ex_cash, "PER ex-cash") if (m.per_ex_cash is not None and m.net_cash and m.net_cash > 0) else (m.per, "PER")
    if m.net_income is not None and m.net_income <= 0:
        card.add("valuation", "Net income is negative: no PER", -6)
    elif multiple is not None:
        pts = _below(multiple, ((10, 6), (14, 3), (18, 0), (25, -3), (35, -6)), -8)
        card.add("valuation", f"{label} {multiple:.1f}x, {_ni_period(m)} (market average {MARKET_AVG_PER:.0f}x)", pts)

    if m.ev_to_market_cap is not None:
        ratio = m.ev_to_market_cap
        pts = _below(ratio, ((1.0, 4), (1.15, 0), (1.4, -3), (1.8, -6)), -10)
        desc = "net cash" if ratio < 1.0 else "little debt" if ratio < 1.15 else "notable debt" if ratio < 1.4 else "a lot of debt"
        card.add("valuation", f"EV is {ratio:.2f}x market cap ({desc})", pts)


def _score_returns(m: OwnerMetrics, card: _Scorecard) -> None:
    if m.roic is not None:
        pts = _at_least(m.roic, ((0.25, 12), (0.18, 8), (0.135, 4), (0.08, -2), (0.0, -6)), -10)
        card.add("returns", f"ROIC {m.roic:.1%}, {_period(m)} (market average 13-14%)", pts)

    history = [y.roic for y in m.years if y.roic is not None]
    if len(history) >= 3:
        high = sum(1 for r in history if r >= HIGH_ROIC)
        share = high / len(history)
        pts = 5 if share == 1.0 else 2 if share >= 2 / 3 else -3 if share <= 1 / 3 else 0
        if share == 1.0 and len(history) >= DURABLE_MOAT_YEARS:
            pts += 2  # a decade of high returns is stronger evidence than four years (T-06)
        card.add("returns", f"ROIC at or above 15% in {high} of {len(history)} years (moat persistence)", pts)


def _score_reinvestment(m: OwnerMetrics, card: _Scorecard) -> None:
    """Reinvestment is reported, not rewarded (T-07); only value-destroying reinvestment costs points."""
    if m.reinvestment_rate is None or m.roic is None:
        return
    rate = m.reinvestment_rate
    if m.roic >= HIGH_ROIC and rate >= GOOD_REINVESTMENT:
        card.add("reinvestment", f"High ROIC with {rate:.0%} reinvestment rate: compounder profile (reported, not scored)", 0)
    elif m.roic >= HIGH_ROIC and rate < 0.10:
        card.add("reinvestment", f"High ROIC but only {rate:.0%} reinvested: limited room to grow (Coca-Cola case)", 0)
    elif m.roic >= HIGH_ROIC:
        card.add("reinvestment", f"High ROIC with a moderate {rate:.0%} reinvestment rate", 0)
    elif m.roic < 0.08 and rate >= GOOD_REINVESTMENT:
        card.add("reinvestment", f"Reinvesting {rate:.0%} of FCF at a low {m.roic:.1%} ROIC destroys value", -4)
    else:
        card.add("reinvestment", f"Reinvestment rate {rate:.0%} at {m.roic:.1%} ROIC", 0)


def _score_cash_yield(m: OwnerMetrics, treasury: float | None, card: _Scorecard) -> None:
    if m.fcf_yield_on_ev is None or treasury is None:
        return
    spread = m.fcf_yield_on_ev * 100.0 - treasury
    pts = 3 if spread >= 2 else 1 if spread >= 0 else -2 if spread <= -2 else 0
    card.add("cash_yield", f"FCF yield on EV {m.fcf_yield_on_ev:.1%} vs 10Y Treasury {treasury:.2f}% ({spread:+.1f} pts)", pts)


def _score_financial(m: OwnerMetrics, card: _Scorecard) -> None:
    if m.net_income is not None and m.net_income <= 0:
        card.add("valuation", "Net income is negative: no PER", -10)
    elif m.per is not None:
        pts = _below(m.per, ((8, 10), (11, 6), (14, 3), (18, 0), (25, -4)), -8)
        card.add("valuation", f"PER {m.per:.1f}x (market average {MARKET_AVG_PER:.0f}x)", pts)
    if m.roe is not None:
        pts = _at_least(m.roe, ((0.18, 10), (0.13, 6), (0.10, 2), (0.06, -3)), -8)
        card.add("returns", f"ROE {m.roe:.1%} (net income / equity)", pts)
    if m.roa is not None:
        pts = _at_least(m.roa, ((0.015, 6), (0.010, 3), (0.006, -1)), -5)
        card.add("returns", f"ROA {m.roa:.2%} (net income / total assets, banks target about 1%)", pts)


# ---------------------------------------------------------------- raw_data


def _metrics_table(d: FundamentalsData, m: OwnerMetrics, source: str) -> list[dict[str, str]]:
    c = m.currency
    rows = [
        {"metric": "Company", "value": d.short_name or d.ticker},
        {"metric": "Sector / industry", "value": " / ".join(x for x in (d.sector, d.industry) if x) or "n/a"},
        {"metric": "Statements basis", "value": f"FY ending {m.fiscal_year_end} ({source})" if m.fiscal_year_end else "n/a"},
        {"metric": "Price (live)", "value": "n/a" if m.price is None else f"${m.price:,.2f}"},
        {"metric": "Shares outstanding", "value": "n/a" if m.shares_outstanding is None else f"{m.shares_outstanding / 1e6:,.1f}M"},
        {"metric": "Market cap = shares x price", "value": _money(m.market_cap)},
    ]
    if m.is_financial:
        rows += [
            {"metric": f"Net income ({m.net_income_source or 'n/a'})", "value": _money(m.net_income, c)},
            {"metric": "PER = market cap / net income", "value": _mult(m.per)},
            {"metric": "Equity", "value": _money(m.equity, c)},
            {"metric": "Total assets", "value": _money(m.total_assets, c)},
            {"metric": "ROE = net income / equity", "value": _pct(m.roe)},
            {"metric": "ROA = net income / total assets", "value": _pct(m.roa)},
        ]
        return rows

    wc_formula = "WC = inventory + receivables - payables"
    rows += [
        {"metric": "Debt incl. leases (latest quarter)", "value": _money(m.total_debt, c)},
        {"metric": "Cash + short-term investments", "value": _money(m.cash, c)},
        {"metric": "EV = market cap + debt - cash", "value": _money(m.enterprise_value)},
        {"metric": "EV / market cap", "value": "n/a" if m.ev_to_market_cap is None else f"{m.ev_to_market_cap:.2f}x"},
        {"metric": "EBITDA", "value": _money(m.ebitda, c)},
        {"metric": "Maintenance CapEx (D&A, capped at CapEx)", "value": _money(m.maintenance_capex, c)},
        {"metric": f"Interest ({m.interest_source or 'n/a'})", "value": _money(m.interest, c)},
        {"metric": f"Taxes ({m.taxes_source or 'n/a'})", "value": _money(m.taxes, c)},
        {"metric": f"Working capital, {wc_formula}", "value": f"{_money(m.working_capital, c)} (prior year {_money(m.working_capital_prev, c)})"},
        {"metric": "Change in working capital (latest year)", "value": _money(m.delta_working_capital, c)},
        {"metric": f"Change in working capital used ({m.delta_working_capital_years}-year average)", "value": _money(m.delta_working_capital_avg, c)},
        {"metric": "FCF = EBITDA - maint. CapEx - interest - taxes - change in WC", "value": _money(m.fcf_owner, c)},
        {"metric": "FCF after all CapEx (growth CapEx subtracted too)", "value": _money(m.fcf_all_capex, c)},
        {"metric": "Reported FCF (operating cash flow - total CapEx)", "value": _money(m.reported_fcf, c)},
        {"metric": f"EV / FCF scored ({FCF_DEFINITIONS.get(m.fcf_definition, m.fcf_definition)}; market average 16-17x)", "value": _mult(m.ev_to_fcf)},
        {"metric": "EV / FCF: owner / after all CapEx / reported", "value": f"{_mult(m.ev_to_fcf_owner)} / {_mult(m.ev_to_fcf_all_capex)} / {_mult(m.ev_to_reported_fcf)}"},
        {"metric": "EV / EBIT (reference: best single value metric in the literature, not scored)", "value": _mult(m.ev_to_ebit)},
        {"metric": f"PER (market average {MARKET_AVG_PER:.0f}x), net income {m.net_income_source or 'n/a'}", "value": _mult(m.per)},
        {"metric": "PER ex-cash = (market cap - net cash) / net income", "value": _mult(m.per_ex_cash)},
        {"metric": "Tax rate used for ROIC", "value": _pct(m.tax_rate)},
        {"metric": "Invested capital = equity + debt + leases - ST investments", "value": _money(m.invested_capital, c)},
        {"metric": "ROIC = EBIT x (1 - T) / invested capital (market 13-14%)", "value": _pct(m.roic)},
        {"metric": "Growth CapEx = CapEx - D&A", "value": _money(m.growth_capex, c)},
        {"metric": "Acquisitions", "value": _money(m.acquisitions, c) + ("" if m.acquisitions_reported is not False else " (line not reported, counted as 0)")},
        {"metric": "Reinvestment rate = (growth CapEx + acquisitions) / FCF (reported, not scored)", "value": _pct(m.reinvestment_rate)},
        {"metric": "Why this reinvestment rate", "value": reinvestment_explanation(m.capex, m.d_and_a, m.acquisitions, m.acquisitions_reported, m.reinvestment_rate, c) or "n/a"},
        {"metric": "R&D as a share of FCF (information only, outside the owner's reinvestment formula)", "value": "n/a" if not m.r_and_d or not m.fcf or m.fcf <= 0 else f"{m.r_and_d / m.fcf:.0%} ({_money(m.r_and_d, c)})"},
        {"metric": "Incremental ROIC = change in NOPAT / change in invested capital (reference)", "value": "n/a" if m.incremental_roic is None else f"{_pct(m.incremental_roic)} over {m.incremental_roic_years:g} years"},
        {"metric": "Asset growth YoY (reference: high asset growth predicts lower returns)", "value": "n/a" if m.asset_growth is None else f"{m.asset_growth:+.1%}"},
        {"metric": "FCF yield on EV", "value": _pct(m.fcf_yield_on_ev)},
        {"metric": "10Y Treasury yield", "value": "n/a" if d.treasury_10y is None else f"{d.treasury_10y:.2f}%"},
        {"metric": "Operating margin (reference, not scored)", "value": _pct(d.operating_margin)},
        {"metric": "Revenue growth YoY (reference, not scored)", "value": "n/a" if d.revenue_growth is None else f"{d.revenue_growth:+.1%}"},
    ]
    return rows


def _roic_table(m: OwnerMetrics) -> dict | None:
    if m.is_financial or not any(y.roic is not None for y in m.years):
        return None
    return {
        "title": f"ROIC by fiscal year ({len(m.years)} years)",
        "columns": ["Fiscal year end", "EBIT", "Tax rate", "Invested capital", "ROIC"],
        "rows": [
            [y.fiscal_year_end, _money(y.ebit, m.currency), _pct(y.tax_rate), _money(y.invested_capital, m.currency), _pct(y.roic)]
            for y in m.years
            if y.ebit is not None or y.invested_capital is not None
        ],
    }


def _links(ticker: str) -> list[dict[str, str]]:
    t = quote(ticker, safe="")
    return [
        {"label": "Yahoo income statement", "url": f"https://finance.yahoo.com/quote/{t}/financials/"},
        {"label": "Yahoo balance sheet", "url": f"https://finance.yahoo.com/quote/{t}/balance-sheet/"},
        {"label": "Yahoo cash flow", "url": f"https://finance.yahoo.com/quote/{t}/cash-flow/"},
        {"label": "SEC EDGAR 10-K filings", "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type=10-K&dateb=&owner=include&count=40"},
    ]


def _last_close(ohlcv: pd.DataFrame | None) -> float | None:
    try:
        if ohlcv is not None and not ohlcv.empty and "close" in ohlcv.columns:
            value = float(ohlcv["close"].iloc[-1])
            return value if value > 0 else None
    except (TypeError, ValueError, IndexError):
        return None
    return None


# ---------------------------------------------------------------- SEC ROIC history (T-06)


def _extend_roic_history(ticker: str, m: OwnerMetrics, statements) -> None:
    """Append older fiscal years from SEC XBRL filings so ROIC persistence covers up to ten years.

    Yahoo serves about four annual statements. When ``SEC_USER_AGENT`` is set,
    the SEC companyfacts API (cached for a week) supplies the earlier years.
    Only years older than Yahoo's oldest are added, so every other metric keeps
    using Yahoo's latest figures. Never raises; silently skipped offline.
    """
    from data.sec_edgar import has_sec_access, load_company_facts, load_ticker_map

    if not has_sec_access() or not m.years:
        return
    try:
        cik = load_ticker_map().get(ticker.upper())
        if cik is None:
            return
        facts = load_company_facts(ticker, cik, splits=statements.splits)
        sec = facts.statements_asof(pd.Timestamp.today().normalize(), max_years=ROIC_HISTORY_YEARS)
        if not sec.available:
            return
        oldest = min(pd.Timestamp(y.fiscal_year_end) for y in m.years)
        added: list[YearMetrics] = []
        for year in sec.years():
            if year < oldest - pd.Timedelta(days=200):
                ym = _roic_for_year(sec, year)
                if ym.roic is not None:
                    added.append(ym)
        if added:
            m.years = m.years + sorted(added, key=lambda y: y.fiscal_year_end, reverse=True)
            m.notes.append(f"ROIC history extended to {len(m.years)} years with SEC EDGAR 10-K filings")
    except Exception as exc:  # noqa: BLE001 - the extension is optional
        logger.info("SEC ROIC history unavailable for %s: %s", ticker, exc)


# ---------------------------------------------------------------- agent


@register_agent
class FundamentalAgent(BaseAgent):
    """Owner-style valuation and returns: EV/FCF, PER, EV vs market cap, ROIC, reinvestment."""

    key = "fundamentals"
    name = "Fundamentals & Valuation Agent"
    description = (
        "Value framework: EV/FCF with FCF = EBITDA - maintenance CapEx - interest - taxes - change in "
        "working capital (market average 16-17x), PER and PER ex-cash (16x), EV vs market cap (debt), "
        "ROIC with up to ten years of persistence (13-14% average; SEC filings when SEC_USER_AGENT is set). "
        "Reinvestment rate, EV/EBIT and incremental ROIC are reported, not scored. Banks and insurers use "
        "PER, ROE and ROA. Annual 10-K figures with the live price. Owner layer. Not point-in-time."
    )
    category = "Fundamental"
    score_scale = 45.0
    default_weight = 0.45  # share of the owner score (config owner_weights)
    point_in_time = False
    role = OWNER

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        try:
            data = fetch_fundamentals(ticker)
            if not data.available or (data.quote_type and data.quote_type != "EQUITY"):
                return self.evaluate_with_data(data, FinancialStatements(ticker=ticker, available=False), ohlcv)
            statements = fetch_current_statements(ticker)
        except Exception as exc:  # noqa: BLE001 - providers should not raise, but never crash the run
            logger.warning("Fundamentals providers raised for %s: %s", ticker, exc)
            return self.unavailable("fundamentals provider error", error=str(exc))
        return self.evaluate_with_data(data, statements, ohlcv)

    def evaluate_with_data(
        self,
        data: FundamentalsData,
        statements: FinancialStatements,
        ohlcv: pd.DataFrame | None = None,
        asof: pd.Timestamp | None = None,
    ) -> AgentAssessment:
        """Score already fetched data (used by tests, batch scans and point-in-time backtests).

        ``asof`` is the evaluation date (default today); ``ohlcv`` must end on it.
        """
        if not data.available:
            return self.unavailable(data.error or "fundamentals unavailable", error=data.error)
        if data.quote_type and data.quote_type != "EQUITY":
            return self.unavailable(f"Fundamentals do not apply to {data.quote_type}")
        if not statements.available:
            return self.unavailable(statements.error or "annual statements unavailable", error=statements.error)

        m = compute_owner_metrics(statements, data, _last_close(ohlcv), ohlcv, today=asof)
        if asof is None and not m.is_financial:
            _extend_roic_history(data.ticker, m, statements)
        card = _Scorecard()
        if m.is_financial:
            _score_financial(m, card)
        else:
            _score_valuation(m, card)
            _score_returns(m, card)
            _score_reinvestment(m, card)
            _score_cash_yield(m, data.treasury_10y, card)

        components = card.components()
        scored = [c for c in COMPONENTS if components[c] is not None]
        if not scored:
            return self.unavailable("Statements lack the inputs for EV/FCF, PER and ROIC")

        expected = ("valuation", "returns") if m.is_financial else COMPONENTS
        missing = [c for c in expected if components[c] is None]
        confidence = min(MAX_CONFIDENCE, BASE_CONFIDENCE + CONFIDENCE_PER_COMPONENT * len(scored))
        if m.notes:
            confidence -= 5
        quality = LIVE if not missing and not m.missing else PARTIAL

        key_rules = sorted((r for r in card.rules if r.points), key=lambda r: abs(r.points), reverse=True)
        key_factors = [f"{r.factor} ({r.points:+.0f})" for r in key_rules[:6]]
        if statements.basis_note:
            key_factors.append(statements.basis_note)
        key_factors += m.notes
        if m.missing:
            key_factors.append(f"Missing inputs: {', '.join(m.missing)}")
        if m.is_financial:
            key_factors.append("Financial company: EV/FCF, working capital and ROIC do not apply; scored on PER, ROE and ROA")
        excluded = is_excluded_sector(data.sector)
        if excluded:
            key_factors.append(
                f"{data.sector}: outside the value framework, so the ticker is not rated (owner's call, decision_log.md T-12). "
                "The debt and ROIC rules exclude utilities and REITs by construction and EV/FCF on REIT depreciation "
                "is not meaningful; this score is shown for information only"
            )

        label = data.short_name or data.ticker
        if m.is_financial:
            summary = f"{label} (financial): PER {_mult(m.per)}, ROE {_pct(m.roe)}, ROA {_pct(m.roa)}; fundamentals score {card.total:+.0f}."
        else:
            summary = (
                f"{label}: EV/FCF {_mult(m.ev_to_fcf)}, PER {_mult(m.per)}, EV/market cap "
                f"{'n/a' if m.ev_to_market_cap is None else f'{m.ev_to_market_cap:.2f}x'}, ROIC {_pct(m.roic)}, "
                f"reinvestment {_pct(m.reinvestment_rate)}; fundamentals score {card.total:+.0f}."
            )

        raw_data: dict = {
            "metrics_table": _metrics_table(data, m, statements.source),
            "score_breakdown": [{"factor": r.factor, "points": r.points} for r in card.rules],
            "components": components,
            "links": _links(data.ticker),
            "data_source_label": f"{statements.source} annual statements",
            "is_financial": m.is_financial,
            "sector": data.sector,
            "industry": data.industry,
            "excluded_sector": excluded,
            "fcf_definition": m.fcf_definition,
            "fcf": m.fcf,
            "ev_to_fcf": m.ev_to_fcf,
            "ev_to_ebit": m.ev_to_ebit,
            "per": m.per,
            "roic": m.roic,
            "roic_years": len([y for y in m.years if y.roic is not None]),
            "net_dilution_cost": m.net_dilution_cost,
            "fiscal_year_end": m.fiscal_year_end,
            "statements_note": statements.basis_note,
        }
        table = _roic_table(m)
        if table is not None:
            raw_data["table"] = table

        return self.assessment(
            score=card.total,
            confidence=confidence,
            summary=summary,
            key_factors=key_factors,
            raw_data=raw_data,
            data_quality=quality,
            data_as_of=m.fiscal_year_end,
        )
