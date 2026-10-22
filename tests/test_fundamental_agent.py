"""Tests for the owner-framework fundamentals, the shareholder agent and their providers (no network)."""

from __future__ import annotations

import pandas as pd
import pytest
import yfinance

import config.settings as settings
import data.fundamentals as fundamentals
from agents.base import LIVE, PARTIAL, UNAVAILABLE
from agents.fundamental_agent import FundamentalAgent
from agents.master_agent import MasterDecisionAgent
from agents.registry import get_agent_classes
from agents.shareholder_agent import ShareholderAgent
from analysis.fundamental_metrics import compute_owner_metrics, is_financial_company, reinvestment_explanation
from config.weights import REGIMES, ScoringConfig
from data.financial_statements import FinancialStatements
from data.fundamentals import EarningsSurprise, FundamentalsData
from tests.conftest import make_ohlcv

YEARS = [pd.Timestamp(f"{y}-12-31") for y in (2025, 2024, 2023, 2022)]


def _frame(rows: dict[str, list[float | None]]) -> pd.DataFrame:
    """Rows of values newest first, one per fiscal year in YEARS."""
    return pd.DataFrame({YEARS[i]: {k: v[i] for k, v in rows.items()} for i in range(len(YEARS))})


def _statements(
    ebit=(80.0, 75.0, 70.0, 65.0),
    diluted=(90.0, 93.0, 96.0, 100.0),
    splits: pd.Series | None = None,
    interest_latest: float | None = 5.0,
    inventory: bool = True,
    **overrides,
) -> FinancialStatements:
    """A compounder: EBITDA 100, D&A 20, CapEx 30, taxes 15, WC 50 vs 45 the year before."""
    income = {
        "EBIT": list(ebit),
        "EBITDA": [100.0, 95.0, 90.0, 85.0],
        "Pretax Income": [75.0, 70.0, 65.0, 60.0],
        "Tax Provision": [15.0, 14.0, 13.0, 12.0],
        "Interest Expense": [interest_latest, 5.0, 5.0, 5.0],
        "Net Income": [60.0, 56.0, 52.0, 48.0],
        "Total Revenue": [400.0, 380.0, 360.0, 340.0],
        "Diluted Average Shares": list(diluted),
        "Basic Average Shares": [d * 0.99 for d in diluted],
    }
    balance = {
        "Stockholders Equity": [200.0, 190.0, 180.0, 170.0],
        "Total Debt": [60.0, 60.0, 60.0, 60.0],
        "Other Short Term Investments": [10.0, 10.0, 10.0, 10.0],
        "Accounts Receivable": [30.0, 28.0, 26.0, 24.0],
        "Accounts Payable": [20.0, 20.0, 20.0, 20.0],
        "Total Assets": [500.0, 480.0, 460.0, 440.0],
        "Cash Cash Equivalents And Short Term Investments": [50.0, 45.0, 40.0, 35.0],
    }
    if inventory:
        balance["Inventory"] = [40.0, 37.0, 35.0, 33.0]
    cashflow = {
        "Depreciation And Amortization": [20.0, 19.0, 18.0, 17.0],
        "Capital Expenditure": [-30.0, -28.0, -26.0, -24.0],
        "Free Cash Flow": [45.0, 42.0, 40.0, 38.0],
        "Stock Based Compensation": [4.0, 4.0, 4.0, 4.0],
        "Repurchase Of Capital Stock": [-20.0, -20.0, -20.0, -20.0],
        "Purchase Of Business": [-5.0, 0.0, 0.0, 0.0],
    }
    for key, value in overrides.items():
        for table in (income, balance, cashflow):
            if key in table:
                table[key] = value
    return FinancialStatements(
        ticker="GOOD",
        available=True,
        income=_frame(income),
        balance=_frame(balance),
        cashflow=_frame(cashflow),
        splits=splits if splits is not None else pd.Series(dtype=float),
    )


def _profile(**overrides) -> FundamentalsData:
    values = dict(
        ticker="GOOD", available=True, quote_type="EQUITY", short_name="Good Co",
        sector="Technology", industry="Software", currency="USD", financial_currency="USD",
        current_price=10.0, shares_outstanding=100.0, float_shares=95.0,
        total_debt=60.0, total_cash=50.0, net_income_ttm=60.0,
        held_percent_insiders=0.12, treasury_10y=4.0,
        officers=[{"name": "Ada Founder", "title": "Co-Founder, CEO & Director", "age": 50, "total_pay": 2.0}],
        earnings=[EarningsSurprise(q, 1.1, 1.0, 10.0) for q in ("2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30")],
    )
    values.update(overrides)
    return FundamentalsData(**values)


def _ohlcv(price: float = 10.0, volume: float = 5_000_000.0) -> pd.DataFrame:
    df = make_ohlcv(periods=60, start=price, end=price, volume=volume)
    return df


def _breakdown(result) -> dict[str, float]:
    return {row["factor"]: row["points"] for row in result.raw_data["score_breakdown"]}


# ------------------------------------------------------------------ formulas


def test_owner_formulas_match_definitions():
    m = compute_owner_metrics(_statements(), _profile(), price=10.0, ohlcv=_ohlcv())
    # Market cap = shares x price; EV = market cap + debt - cash
    assert m.market_cap == pytest.approx(1000.0)
    assert m.enterprise_value == pytest.approx(1000.0 + 60.0 - 50.0)
    assert m.ev_to_market_cap == pytest.approx(1.01)
    # WC = inventory + receivables - payables: 50, 45, 41, 37 over the four years
    assert m.working_capital == pytest.approx(50.0)
    assert m.delta_working_capital == pytest.approx(5.0)  # latest year
    # The FCF uses the change averaged over the last 3 yearly changes: (5 + 4 + 4) / 3
    assert m.delta_working_capital_years == 3
    assert m.delta_working_capital_avg == pytest.approx(13.0 / 3.0)
    # FCF = EBITDA - maintenance CapEx (D&A 20, capped at CapEx 30) - interest - taxes - change in WC
    assert m.maintenance_capex == pytest.approx(20.0)
    fcf = 100.0 - 20.0 - 5.0 - 15.0 - 13.0 / 3.0
    assert m.fcf_owner == pytest.approx(fcf)
    assert m.fcf_all_capex == pytest.approx(fcf - 10.0)  # growth CapEx = 30 - 20
    assert m.reported_fcf == pytest.approx(45.0)
    assert m.ev_to_fcf_owner == pytest.approx(1010.0 / fcf)
    # T-05 (owner's call, September 2026): reported FCF feeds the scored EV/FCF by default
    assert settings.FCF_DEFINITION == "standard"
    assert m.fcf_definition == "standard" and m.fcf == pytest.approx(45.0) and m.ev_to_fcf == pytest.approx(1010.0 / 45.0)
    assert m.ev_to_fcf_all_capex == pytest.approx(1010.0 / (fcf - 10.0))
    assert m.ev_to_reported_fcf == pytest.approx(1010.0 / 45.0)
    assert m.ev_to_ebit == pytest.approx(1010.0 / 80.0)
    # ROIC = EBIT (1 - T) / (equity + debt - ST investments), T = 15 / 75
    assert m.tax_rate == pytest.approx(0.2)
    assert m.invested_capital == pytest.approx(200.0 + 60.0 - 10.0)
    assert m.roic == pytest.approx(80.0 * 0.8 / 250.0)
    # Reinvestment = (CapEx - D&A + acquisitions) / FCF, on the scored FCF definition (reported FCF = 45)
    assert m.reinvestment_rate == pytest.approx((10.0 + 5.0) / 45.0)
    # Net dilution cost = (stock pay - buybacks) / market cap; net buyback yield = (buybacks - issuance) / market cap
    assert m.net_dilution_cost == pytest.approx((4.0 - 20.0) / 1000.0)
    assert m.net_buyback_yield == pytest.approx(20.0 / 1000.0)
    assert m.asset_growth == pytest.approx(500.0 / 480.0 - 1.0)
    assert m.incremental_roic is not None and m.incremental_roic > 0
    # PER and PER ex-cash (no net cash here: debt 60 > cash 50, so no adjustment)
    assert m.per == pytest.approx(1000.0 / 60.0)
    assert m.per_ex_cash == pytest.approx(m.per)
    # Shareholder structure
    assert m.share_cagr == pytest.approx((90.0 / 100.0) ** (1 / 3.0) - 1.0, rel=1e-2)
    assert m.sbc_to_revenue == pytest.approx(0.01)
    assert m.avg_dollar_volume == pytest.approx(10.0 * 5_000_000.0)


def test_per_ex_cash_subtracts_net_cash():
    m = compute_owner_metrics(_statements(), _profile(total_cash=300.0), price=10.0)
    assert m.net_cash == pytest.approx(240.0)
    assert m.per_ex_cash == pytest.approx((1000.0 - 240.0) / 60.0)


def test_missing_inventory_line_counts_as_zero_with_note():
    m = compute_owner_metrics(_statements(inventory=False), _profile(), price=10.0)
    assert m.working_capital == pytest.approx(30.0 - 20.0)
    assert any("inventory" in n.lower() for n in m.notes)


def test_interest_falls_back_to_latest_reported_year():
    m = compute_owner_metrics(_statements(interest_latest=None), _profile(), price=10.0)
    assert m.interest == pytest.approx(5.0)
    assert "FY 2024-12-31" in m.interest_source
    assert any("does not report interest" in n for n in m.notes)


def test_fcf_definition_setting_selects_the_scored_fcf(monkeypatch):
    monkeypatch.setattr(settings, "FCF_DEFINITION", "owner_all_capex")
    m = compute_owner_metrics(_statements(), _profile(), price=10.0)
    assert m.fcf_definition == "owner_all_capex" and m.fcf == pytest.approx(m.fcf_all_capex)
    monkeypatch.setattr(settings, "FCF_DEFINITION", "standard")
    m = compute_owner_metrics(_statements(), _profile(), price=10.0)
    assert m.fcf == pytest.approx(45.0) and m.ev_to_fcf == pytest.approx(1010.0 / 45.0)
    result = FundamentalAgent().evaluate_with_data(_profile(), _statements(), _ohlcv())
    assert any(k.startswith("EV/FCF") and "reported FCF" in k for k in _breakdown(result))
    monkeypatch.setattr(settings, "FCF_DEFINITION", "nonsense")
    assert compute_owner_metrics(_statements(), _profile(), price=10.0).fcf_definition == "standard"  # invalid: default


def test_working_capital_average_window_setting(monkeypatch):
    monkeypatch.setattr(settings, "WORKING_CAPITAL_AVERAGE_YEARS", 1)
    m = compute_owner_metrics(_statements(), _profile(), price=10.0)
    assert m.delta_working_capital_years == 1 and m.delta_working_capital_avg == pytest.approx(5.0)
    assert m.fcf_owner == pytest.approx(55.0)


def test_currency_mismatch_skips_market_value_ratios():
    m = compute_owner_metrics(_statements(), _profile(financial_currency="TWD"), price=10.0)
    assert m.enterprise_value is None and m.ev_to_fcf is None and m.per is None
    assert m.roic is not None  # ratios within the statements are still valid


def test_split_guard_restates_unadjusted_share_counts():
    splits = pd.Series([10.0], index=[pd.Timestamp("2024-06-10")])
    unadjusted = (100.0, 101.0, 10.2, 10.3)  # pre-split years reported on the old basis
    m = compute_owner_metrics(_statements(diluted=unadjusted, splits=splits), _profile(), price=10.0)
    assert m.share_cagr is not None and abs(m.share_cagr) < 0.02


def test_financial_detection():
    assert is_financial_company("Financial Services", "Banks - Diversified")
    assert is_financial_company(None, "Insurance - Life")
    assert not is_financial_company("Technology", "Software")


# ------------------------------------------------------------ fundamentals agent


def test_registered_agents():
    classes = get_agent_classes()
    assert classes["fundamentals"].category == "Fundamental" and not classes["fundamentals"].point_in_time
    assert classes["shareholder"].default_weight > 0


def test_cheap_compounder_outscores_expensive_value_trap():
    good = FundamentalAgent().evaluate_with_data(_profile(), _statements(), _ohlcv(price=10.0))
    trap = FundamentalAgent().evaluate_with_data(
        _profile(total_debt=900.0, net_income_ttm=10.0),
        _statements(ebit=(8.0, 9.0, 10.0, 11.0), **{"Total Debt": [900.0] * 4, "EBITDA": [20.0, 21.0, 22.0, 23.0]}),
        _ohlcv(price=10.0),
    )
    assert good.score > trap.score
    rules = _breakdown(trap)
    assert any("a lot of debt" in k for k in rules)
    assert sum(_breakdown(good).values()) == pytest.approx(good.score, abs=0.1)
    assert good.data_quality == LIVE
    metrics = {row["metric"] for row in good.raw_data["metrics_table"]}
    assert "EV = market cap + debt - cash" in metrics
    assert "FCF = EBITDA - maint. CapEx - interest - taxes - change in WC" in metrics


def test_ev_fcf_anchor_rewards_cheapness():
    cheap = FundamentalAgent().evaluate_with_data(_profile(), _statements(), _ohlcv(price=5.0))
    rich = FundamentalAgent().evaluate_with_data(_profile(), _statements(), _ohlcv(price=40.0))
    ev_rule = lambda r: next(v for k, v in _breakdown(r).items() if k.startswith("EV/FCF"))
    assert ev_rule(cheap) > 0 > ev_rule(rich)


def test_financial_company_uses_per_roe_roa():
    bank = _profile(sector="Financial Services", industry="Banks - Regional")
    result = FundamentalAgent().evaluate_with_data(bank, _statements(), _ohlcv())
    factors = list(_breakdown(result))
    assert any(f.startswith("PER") for f in factors)
    assert any(f.startswith("ROE") for f in factors) and any(f.startswith("ROA") for f in factors)
    assert not any("EV" in f or "ROIC" in f for f in factors)


def test_non_equity_and_missing_statements_are_unavailable():
    etf = FundamentalAgent().evaluate_with_data(_profile(quote_type="ETF"), _statements())
    assert etf.data_quality == UNAVAILABLE
    none = FundamentalAgent().evaluate_with_data(_profile(), FinancialStatements("GOOD", available=False, error="down"))
    assert none.data_quality == UNAVAILABLE


# ------------------------------------------------------------ shareholder agent


def test_cannibal_founder_led_scores_well():
    result = ShareholderAgent().evaluate_with_data(_profile(), _statements(), _ohlcv())
    rules = _breakdown(result)
    assert any("cannibal" in k or "buybacks reduce" in k for k in rules)
    assert any(k.startswith("Founder-led") and v == 5 for k, v in rules.items())
    assert result.score > 10
    assert result.raw_data["table"]["title"] == "Management checklist"


def test_diluter_with_heavy_stock_pay_scores_badly():
    diluter = _statements(diluted=(130.0, 118.0, 108.0, 100.0), **{"Stock Based Compensation": [60.0] * 4})
    result = ShareholderAgent().evaluate_with_data(_profile(officers=[], held_percent_insiders=0.001), diluter, _ohlcv())
    rules = _breakdown(result)
    assert any("serial diluter" in k and v == -15 for k, v in rules.items())
    # Net dilution is scored: (60 pay - 20 buybacks) / 1000 market cap = +4.0% of market cap.
    assert any(k.startswith("Stock pay net of buybacks +4.0%") and v == -10 for k, v in rules.items())
    # Gross stock pay / revenue is information only (it scored backwards in the S&P 500 test).
    assert not any(k.startswith("Stock-based compensation") for k in rules)
    assert any("Stock-based compensation 15.0% of revenue" in f and "not scored" in f for f in result.key_factors)
    assert result.score < -10


def test_buybacks_above_stock_pay_are_rewarded():
    result = ShareholderAgent().evaluate_with_data(_profile(), _statements(), _ohlcv())
    rules = _breakdown(result)
    # (4 pay - 20 buybacks) / 1000 = -1.6% of market cap
    assert any(k.startswith("Stock pay net of buybacks -1.6%") and v == 4 for k, v in rules.items())


def test_reinvestment_is_reported_not_rewarded():
    good = FundamentalAgent().evaluate_with_data(_profile(), _statements(), _ohlcv())
    rules = _breakdown(good)
    compounder = next((v for k, v in rules.items() if "compounder profile" in k), None)
    assert compounder == 0  # T-07: reinvestment scored backwards in the point-in-time test
    metrics = {row["metric"]: row["value"] for row in good.raw_data["metrics_table"]}
    assert any("Incremental ROIC" in k for k in metrics)
    assert any("Asset growth" in k for k in metrics)
    assert good.raw_data["sector"] == "Technology" and good.raw_data["fcf_definition"] == "standard"


def test_low_roic_heavy_reinvestment_is_penalized():
    weak = _statements(ebit=(10.0, 10.0, 10.0, 10.0), **{"Capital Expenditure": [-60.0] * 4})
    result = FundamentalAgent().evaluate_with_data(_profile(), weak, _ohlcv())
    assert any("destroys value" in k and v == -4 for k, v in _breakdown(result).items())


def test_utilities_and_reits_are_outside_the_framework():
    utility = _profile(sector="Utilities", industry="Utilities - Regulated Electric")
    result = FundamentalAgent().evaluate_with_data(utility, _statements(), _ohlcv())
    assert result.raw_data["excluded_sector"] is True and result.raw_data["sector"] == "Utilities"
    assert any("not rated" in f and "T-12" in f for f in result.key_factors)
    assert result.data_quality != UNAVAILABLE  # the evidence is still shown
    reit = ShareholderAgent().evaluate_with_data(_profile(sector="Real Estate", industry="REIT - Retail"), _statements(), _ohlcv())
    assert reit.raw_data["excluded_sector"] is True
    normal = FundamentalAgent().evaluate_with_data(_profile(), _statements(), _ohlcv())
    assert normal.raw_data["excluded_sector"] is False


def test_recent_reverse_split_is_a_red_flag():
    splits = pd.Series([0.125], index=[pd.Timestamp.today() - pd.DateOffset(years=3)])
    result = ShareholderAgent().evaluate_with_data(_profile(), _statements(splits=splits), _ohlcv())
    assert any("Reverse split 1-for-8" in k and v == -20 for k, v in _breakdown(result).items())
    assert result.key_factors[0].startswith("Red flag")


def test_low_liquidity_is_penalized():
    thin = ShareholderAgent().evaluate_with_data(_profile(), _statements(), _ohlcv(price=2.0, volume=100_000))
    assert any("illiquid" in k and v == -15 for k, v in _breakdown(thin).items())


# ------------------------------------------------------------ master + macro


def test_master_warns_on_low_liquidity():
    from tests.test_multi_agent import StubAgent

    cfg = ScoringConfig(regime_weights={r: {"a": 1.0} for r in REGIMES},
                        thresholds={"strong_buy": 45, "buy": 15, "sell": -15, "strong_sell": -45})
    thin = make_ohlcv(volume=10_000)  # about $1.3M a day
    rec = MasterDecisionAgent(config=cfg, agents=[StubAgent("a", 30)]).analyze("THIN", thin)
    assert any("Low trading liquidity" in w for w in rec.warnings)
    liquid = MasterDecisionAgent(config=cfg, agents=[StubAgent("a", 30)]).analyze("BIG", make_ohlcv(volume=5e6))
    assert not any("liquidity" in w for w in liquid.warnings)
    assert rec.risk_score > liquid.risk_score


def test_contrarian_vix_rewards_extreme_fear(monkeypatch):
    from agents.regime_agent import MacroRegimeAgent
    from tests.test_platform import _macro_history
    from data.macro_data import compute_macro_snapshot

    panic = compute_macro_snapshot(_macro_history(spy_trend=-0.3, vix=45.0))
    calm = compute_macro_snapshot(_macro_history(spy_trend=-0.3, vix=12.0))
    assert panic.regime in ("BEAR", "CRISIS")  # regime still reflects the stress
    assert panic.opportunity_score > calm.opportunity_score
    assert any(c["component"] == "VIX (contrarian)" and c["points"] == 6.0 for c in panic.opportunity_components)

    monkeypatch.setattr(settings, "VIX_STANCE", "contrarian")
    contrarian = MacroRegimeAgent().evaluate_with_data(panic)
    monkeypatch.setattr(settings, "VIX_STANCE", "risk")
    risk = MacroRegimeAgent().evaluate_with_data(panic)
    assert contrarian.score > risk.score
    assert contrarian.raw_data["vix_stance"] == "contrarian"


# ------------------------------------------------------------------ provider


class _FakeTicker:
    """Minimal stand-in for yfinance.Ticker used by the provider tests."""

    info: dict = {}
    earnings: pd.DataFrame | Exception | None = None
    tnx_close: float | Exception = 42.5

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    def earnings_history(self):
        if isinstance(self.earnings, Exception):
            raise self.earnings
        return self.earnings

    def history(self, period: str = "5d"):
        if isinstance(self.tnx_close, Exception):
            raise self.tnx_close
        index = pd.DatetimeIndex(["2026-09-24", "2026-09-25"])
        return pd.DataFrame({"Close": [self.tnx_close, self.tnx_close]}, index=index)


def _fake_ticker(info: dict, earnings=None, tnx_close=42.5):
    return type("FakeTicker", (_FakeTicker,), {"info": info, "earnings": earnings, "tnx_close": tnx_close})


_INFO = {
    "quoteType": "EQUITY",
    "shortName": "Fake Inc.",
    "currency": "USD",
    "financialCurrency": "USD",
    "currentPrice": 100.0,
    "marketCap": 1e11,
    "trailingPE": 25.0,
    "debtToEquity": 78.0,
    "freeCashflow": 5e9,
    "sharesOutstanding": 1e9,
    "totalDebt": 2e10,
    "totalCash": 1e10,
    "netIncomeToCommon": 4e9,
    "heldPercentInsiders": 0.07,
    "companyOfficers": [{"name": "Jane Roe", "title": "Founder & CEO", "age": 51, "totalPay": 3_000_000}, "junk"],
    "priceToBook": "Infinity",
    "mostRecentQuarter": 1782518400,
}


def test_provider_parses_info_earnings_and_treasury(monkeypatch):
    earnings = pd.DataFrame(
        {
            "epsActual": [1.0, 1.1, 1.2, 1.3, 1.5],
            "epsEstimate": [0.9, 1.0, 1.25, 1.2, 1.4],
            "surprisePercent": [0.11, 0.1, -0.04, 0.083, 0.071],
        },
        index=pd.DatetimeIndex(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"], name="quarter"),
    )
    monkeypatch.setattr(yfinance, "Ticker", _fake_ticker(_INFO, earnings=earnings, tnx_close=42.5))
    data = fundamentals.fetch_fundamentals("fake")

    assert data.available and data.ticker == "FAKE"
    assert data.debt_to_equity == pytest.approx(0.78)  # Yahoo percent -> ratio
    assert data.price_to_book is None  # non-finite values are dropped
    assert data.treasury_10y == pytest.approx(4.25)  # ^TNX above 20 is divided by 10
    assert [e.outcome for e in data.earnings] == ["beat", "miss", "beat", "beat"]
    assert data.shares_outstanding == 1e9 and data.total_debt == 2e10 and data.net_income_ttm == 4e9
    assert data.held_percent_insiders == pytest.approx(0.07)
    assert data.officers == [{"name": "Jane Roe", "title": "Founder & CEO", "age": 51.0, "total_pay": 3_000_000.0}]


def test_provider_sub_fetch_failures_are_recorded(monkeypatch):
    fake = _fake_ticker(_INFO, earnings=RuntimeError("earnings down"), tnx_close=RuntimeError("tnx down"))
    monkeypatch.setattr(yfinance, "Ticker", fake)
    data = fundamentals.fetch_fundamentals("FAKE")
    assert data.available
    assert data.earnings == [] and "earnings down" in data.earnings_error
    assert data.treasury_10y is None and "tnx down" in data.treasury_error


def test_provider_unknown_symbol_is_unavailable(monkeypatch):
    monkeypatch.setattr(yfinance, "Ticker", _fake_ticker({"trailingPegRatio": None}))
    data = fundamentals.fetch_fundamentals("NOPE")
    assert data.available is False


def test_agents_unavailable_offline():
    """With the network blocked, both agents degrade to UNAVAILABLE instead of raising."""
    for agent in (FundamentalAgent(), ShareholderAgent()):
        assert agent.evaluate("AAPL", make_ohlcv()).data_quality == UNAVAILABLE


# ------------------------------------------------------------------ reinvestment display (T-07 follow-up)


def _table_value(result, needle: str) -> str:
    return next(r["value"] for r in result.raw_data["metrics_table"] if needle in r["metric"])


def test_capex_below_da_floors_growth_capex_and_says_so():
    st = _statements(**{"Capital Expenditure": [-12.0, -28.0, -26.0, -24.0], "Purchase Of Business": [None, 0.0, 0.0, 0.0]})
    m = compute_owner_metrics(st, _profile(), price=10.0, ohlcv=_ohlcv())
    assert m.growth_capex == 0.0 and m.growth_capex_floored is True
    assert m.reinvestment_rate == pytest.approx(0.0)  # formula unchanged: floor at zero, missing acquisitions as zero
    assert m.acquisitions == 0.0 and m.acquisitions_reported is False
    text = reinvestment_explanation(m.capex, m.d_and_a, m.acquisitions, m.acquisitions_reported, m.reinvestment_rate)
    assert text == "0%: CapEx $12 is below D&A $20, so the proxy counts no growth CapEx; acquisitions not reported, counted as zero"


def test_reported_zero_acquisitions_is_not_flagged_as_missing():
    m = compute_owner_metrics(_statements(**{"Purchase Of Business": [0.0, 0.0, 0.0, 0.0]}), _profile(), price=10.0, ohlcv=_ohlcv())
    assert m.acquisitions == 0.0 and m.acquisitions_reported is True and m.growth_capex_floored is False
    text = reinvestment_explanation(m.capex, m.d_and_a, m.acquisitions, m.acquisitions_reported, m.reinvestment_rate)
    assert "no acquisitions" in text and "not reported" not in text


def test_missing_acquisitions_line_does_not_change_the_score():
    agent = FundamentalAgent()
    with_zero = agent.evaluate_with_data(_profile(), _statements(**{"Purchase Of Business": [0.0, 0.0, 0.0, 0.0]}), _ohlcv())
    missing = agent.evaluate_with_data(_profile(), _statements(**{"Purchase Of Business": [None, None, None, None]}), _ohlcv())
    assert with_zero.score == missing.score
    assert _breakdown(with_zero) == _breakdown(missing)


def test_r_and_d_is_information_only():
    st = _statements()
    st.income.loc["Research And Development"] = [8.0, 7.0, 6.0, 5.0]
    m = compute_owner_metrics(st, _profile(), price=10.0, ohlcv=_ohlcv())
    assert m.r_and_d == 8.0
    base = FundamentalAgent().evaluate_with_data(_profile(), _statements(), _ohlcv())
    with_rd = FundamentalAgent().evaluate_with_data(_profile(), st, _ohlcv())
    assert base.score == with_rd.score


def test_evidence_rows_explain_a_floored_rate_and_label_rd():
    st = _statements(**{"Capital Expenditure": [-12.0, -28.0, -26.0, -24.0], "Purchase Of Business": [None, 0.0, 0.0, 0.0]})
    st.income.loc["Research And Development"] = [8.0, 7.0, 6.0, 5.0]
    result = FundamentalAgent().evaluate_with_data(_profile(), st, _ohlcv())
    assert "not reported" in _table_value(result, "Acquisitions")
    assert "no growth CapEx" in _table_value(result, "Why this reinvestment rate")
    assert "outside the owner's reinvestment formula" in next(r["metric"] for r in result.raw_data["metrics_table"] if "R&D" in r["metric"])
