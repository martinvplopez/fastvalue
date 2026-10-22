"""Competitive Position agent: metrics, peer choice, scoring, margins extract and the agent with fake providers (offline)."""

from __future__ import annotations

import json
import time
import zlib

import pandas as pd
import pytest
import yfinance

import config.settings as settings
import data.peers as peers_mod
import data.sec_edgar as sec_edgar
import data.sec_filings as filings_mod
from agents.base import LIVE, OWNER, PARTIAL, UNAVAILABLE, AgentAssessment
from agents.competitive_agent import CompetitiveAgent
from agents.registry import get_agent_classes
from agents.scoring import aggregate_layers
from analysis.competitive_metrics import (
    SCORING_VERSION,
    Candidate,
    CompanyMetrics,
    card_years,
    choose_peers,
    choose_peers_by_sic,
    company_metrics,
    is_stale,
    peer_medians,
    score_competitive,
    share_change,
    usable_peers,
)
from config.weights import ScoringConfig, load_raw_config, validate_config
from data import fundamentals
from data.financial_statements import FinancialStatements
from data.sec_margins import extract_margin_facts, margins_asof
from tests.conftest import make_ohlcv


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch: pytest.MonkeyPatch):
    def blocked(*_a, **_k):
        raise RuntimeError("Network access is disabled in tests")

    monkeypatch.setattr(yfinance, "Industry", blocked, raising=False)
    monkeypatch.setattr(yfinance, "Sector", blocked, raising=False)
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "cache")
    monkeypatch.delenv(sec_edgar.SEC_USER_AGENT_ENV, raising=False)
    peers_mod._MEMO.clear()
    peers_mod._FAILED.clear()
    yield
    peers_mod._MEMO.clear()
    peers_mod._FAILED.clear()
    peers_mod._INFLIGHT.clear()


# ------------------------------------------------------------------ helpers


def _years(rev=1000.0, gm=0.40, opm=0.20, dso=40.0, dpo=50.0, dio=60.0, roic=0.15, growth=0.05, n=5, end="2025-12-31", gm_step=0.0):
    """Fiscal years newest first; revenue grows ``growth`` a year, gross margin moves ``gm_step`` a year."""
    out = []
    for i in range(n):
        revenue = rev / (1 + growth) ** i
        margin = gm - gm_step * (i % 2)
        cogs = revenue * (1 - margin)
        out.append({
            "fy_end": f"{int(end[:4]) - i}{end[4:]}", "revenue": revenue, "cogs": cogs, "gross_profit": revenue - cogs,
            "operating_income": revenue * opm, "receivables": revenue * dso / 365, "payables": cogs * dpo / 365,
            "inventory": cogs * dio / 365, "roic": roic,
        })
    return out


def _m(ticker="X", **kw) -> CompanyMetrics:
    return company_metrics(_years(**kw), 1.0, None, ticker)


def _stmts(years: list[dict], rows_income=None) -> FinancialStatements:
    """Yahoo-style statements from the same year dicts."""
    cols = [pd.Timestamp(y["fy_end"]) for y in years]
    income = pd.DataFrame({
        "Total Revenue": [y["revenue"] for y in years], "Cost Of Revenue": [y["cogs"] for y in years],
        "Gross Profit": [y["gross_profit"] for y in years], "Operating Income": [y["operating_income"] for y in years],
        "Pretax Income": [y["operating_income"] for y in years], "Net Income": [y["operating_income"] * 0.8 for y in years],
        "Tax Provision": [y["operating_income"] * 0.2 for y in years],
    }, index=cols).T
    balance = pd.DataFrame({
        "Accounts Receivable": [y["receivables"] for y in years], "Accounts Payable": [y["payables"] for y in years],
        "Inventory": [y["inventory"] for y in years],
    }, index=cols).T
    return FinancialStatements(ticker="X", available=True, income=income, balance=balance)


# ------------------------------------------------------------------ metrics


def test_company_metrics_ratios_from_cost_of_revenue():
    m = _m(gm=0.40, opm=0.20, dso=40, dpo=50, dio=60, roic=0.15, growth=0.05)
    assert m.gross_margin == pytest.approx(0.40)
    assert m.operating_margin == pytest.approx(0.20)
    assert m.dso == pytest.approx(40) and m.dpo == pytest.approx(50) and m.dio == pytest.approx(60)
    assert m.roic == 0.15
    assert m.revenue_cagr_3y == pytest.approx(0.05)
    assert m.revenue_usd == pytest.approx(1000.0) and m.revenue_3y_ago_usd == pytest.approx(1000.0 / 1.05 ** 3)
    assert m.has_cost_of_revenue and m.issues == []


def test_company_metrics_gross_profit_only_and_cost_only():
    years = _years()
    for y in years:
        y["cogs"] = None  # only gross profit reported
    assert company_metrics(years).dpo == pytest.approx(50)  # cogs = revenue - gross profit
    for y in years:
        y["cogs"], y["gross_profit"] = y["revenue"] - y["gross_profit"] if y["gross_profit"] else None, None
    only_cost = [dict(y, gross_profit=None) for y in _years()]
    assert company_metrics(only_cost).gross_margin == pytest.approx(0.40)  # gross profit = revenue - cogs


def test_gross_margin_swing_needs_four_years():
    steady = _m(gm_step=0.0)
    assert steady.gross_margin_std == pytest.approx(0.0)
    moving = _m(gm=0.40, gm_step=0.04)  # alternates 40% and 36%
    assert moving.gross_margin_std == pytest.approx(1.96, abs=0.01)
    assert company_metrics(_years(n=3)).gross_margin_std is None
    assert company_metrics(_years(n=3)).revenue_cagr_3y is None  # three years of growth need four fiscal years


def test_tagging_errors_are_left_out_and_listed():
    years = _years()
    years[0]["gross_profit"], years[0]["cogs"] = 5000.0, None  # gross margin of 400%
    years[0]["receivables"] = years[0]["revenue"] * 3  # 1,095 days
    m = company_metrics(years)
    assert m.gross_margin is None and m.dso is None
    assert len(m.issues) >= 2
    zero = _years()
    zero[0]["revenue"] = 0.0
    assert company_metrics(zero).gross_margin is None and "no revenue reported" in company_metrics(zero).issues
    years = _years()
    years[0]["cogs"], years[0]["gross_profit"] = 0.0, None
    assert company_metrics(years).dpo is None and company_metrics(years).dio is None  # cost of revenue of zero: no days


def test_stale_peers_are_dropped():
    subject = _m(end="2025-12-31")
    fresh, old = _m("A", end="2025-09-30"), _m("B", end="2024-06-30")
    assert not is_stale(fresh, subject) and is_stale(old, subject)
    assert [p.ticker for p in usable_peers(subject, [fresh, old])] == ["A"]


def test_peer_median_needs_three_peers_with_the_figure():
    peers = [_m("A", dpo=40), _m("B", dpo=60), _m("C", dpo=80)]
    assert peer_medians(peers)["dpo"] == (60.0, 3)
    assert peer_medians(peers[:2])["dpo"] == (None, 2)
    missing = [_m("A", dpo=40), CompanyMetrics("B"), _m("C", dpo=80), _m("D", dpo=100)]
    assert peer_medians(missing)["dpo"] == (80.0, 3)


def test_share_of_peer_group_sales_uses_only_peers_with_both_years():
    subject = _m("S", rev=1000, growth=0.20)  # grew fast
    peers = [_m("A", rev=1000, growth=0.0), _m("B", rev=1000, growth=0.0), _m("C", rev=1000, growth=0.0)]
    change = share_change(subject, peers)
    assert change is not None and change > 0.2
    short = [_m("A", rev=1000), _m("B", rev=1000), company_metrics(_years(n=3), 1.0, None, "C")]  # C lacks the old year
    assert share_change(subject, short) is None  # only two peers have both years
    assert share_change(company_metrics(_years(), None, None, "S"), peers) is None  # no exchange rate


def test_card_years_from_yahoo_statements_and_sec_margins():
    years = _years(n=5)
    st = _stmts(years)
    out = card_years(st)
    assert len(out) == 5 and out[0]["fy_end"] == "2025-12-31"
    assert out[0]["cogs"] == pytest.approx(years[0]["cogs"]) and out[0]["receivables"] == pytest.approx(years[0]["receivables"])
    # SEC statements have no cost rows: the margins sidecar fills them, matched by fiscal year end within 7 days
    st.income = st.income.drop(index=["Cost Of Revenue", "Gross Profit"])
    margins = {pd.Timestamp("2026-01-02"): {"cogs": 111.0, "gross_profit": 222.0}}
    out = card_years(st, margins)
    assert out[0]["cogs"] == 111.0 and out[0]["gross_profit"] == 222.0 and out[1]["cogs"] is None


# ------------------------------------------------------------------ margins extract


def _fact(concept, start, end, val, filed, form="10-K"):
    return {"concept": concept, "start": pd.Timestamp(start), "end": pd.Timestamp(end), "val": val, "filed": pd.Timestamp(filed), "form": form}


def test_margins_asof_priority_restatements_and_quarters():
    frame = pd.DataFrame([
        _fact("CostOfGoodsSold", "2024-01-01", "2024-12-31", 700.0, "2025-02-10"),
        _fact("CostOfRevenue", "2024-01-01", "2024-12-31", 650.0, "2025-02-10"),  # first in the priority order
        _fact("GrossProfit", "2024-01-01", "2024-12-31", 350.0, "2025-02-10"),
        _fact("CostOfRevenue", "2024-01-01", "2024-12-31", 640.0, "2025-08-01", form="10-K/A"),  # restated later
        _fact("CostOfRevenue", "2024-10-01", "2024-12-31", 170.0, "2025-02-10"),  # a quarter: ignored
        _fact("CostOfRevenue", "2024-01-01", "2024-09-30", 480.0, "2024-11-05", form="10-Q"),
    ])
    before = margins_asof(frame, pd.Timestamp("2025-03-01"))
    assert before == {pd.Timestamp("2024-12-31"): {"cogs": 650.0, "gross_profit": 350.0}}
    after = margins_asof(frame, pd.Timestamp("2025-09-01"))
    assert after[pd.Timestamp("2024-12-31")]["cogs"] == 640.0  # the restatement counts from its filing date
    assert margins_asof(frame, pd.Timestamp("2025-01-01")) == {}
    assert margins_asof(pd.DataFrame(columns=frame.columns), pd.Timestamp("2025-01-01")) == {}


def test_extract_margin_facts_reads_only_the_cost_concepts():
    raw = {"facts": {"us-gaap": {
        "CostOfRevenue": {"units": {"USD": [{"start": "2024-01-01", "end": "2024-12-31", "val": 5, "filed": "2025-02-01", "form": "10-K"}]}},
        "Revenues": {"units": {"USD": [{"start": "2024-01-01", "end": "2024-12-31", "val": 9, "filed": "2025-02-01", "form": "10-K"}]}},
    }}}
    frame = extract_margin_facts(raw)
    assert list(frame["concept"]) == ["CostOfRevenue"] and frame["val"].iloc[0] == 5.0
    assert extract_margin_facts({}).empty


# ------------------------------------------------------------------ peer choice


def _cands(caps: dict[str, float], sic=3674) -> list[Candidate]:
    return [Candidate(t, f"{t} Corp", c, sic, zlib.crc32(t.encode())) for t, c in caps.items()]


def test_choose_peers_band_and_nearest_eight():
    caps = {f"P{i}": 100.0 * (1 + i / 10) for i in range(12)}
    caps.update({"TINY": 5.0, "HUGE": 5000.0})
    choice = choose_peers(110.0, _cands(caps))
    assert len(choice.tickers) == 8 and "TINY" not in choice.tickers and "HUGE" not in choice.tickers
    assert not choice.wide_band
    assert choice.tickers[0] == "P1"  # nearest in log distance first


def test_choose_peers_widens_the_band_below_four():
    caps = {"A": 100.0, "B": 120.0, "C": 3.0, "D": 1500.0, "E": 2000.0, "F": 9000.0, "G": 5.0}
    strict = choose_peers(100.0, _cands(caps), allow_wide=False)
    assert set(strict.tickers) == {"A", "B"} and not strict.wide_band
    wide = choose_peers(100.0, _cands(caps))
    assert wide.wide_band and set(wide.tickers) == {"A", "B", "D", "E", "G"}  # 0.04x to 25x
    assert choose_peers(None, _cands(caps)).tickers == []


def test_choose_peers_drops_self_share_classes_and_nameless_rows():
    rows = [
        Candidate("SELF", "Self Corp", 100.0, 3674, 1), Candidate("SELFB", "Self Corp Class B", 100.0, 3674, 1),
        Candidate("TWIN", "Self Corporation", 90.0, 3674, 7), Candidate("NONAME", None, 100.0, 3674, 8),
        Candidate("NAN", "nan", 100.0, 3674, 9), Candidate("NOCAP", "No Cap Inc", None, 3674, 10),
        Candidate("GOOD1", "Good One Inc", 100.0, 3674, 11), Candidate("GOOD2", "Good Two Inc", 100.0, 3674, 12),
        Candidate("GOOD2B", "Good Two Inc", 80.0, 3674, 12),  # the second class of a peer
    ]
    choice = choose_peers(100.0, rows, subject_ticker="SELF", subject_name="Self Corp", subject_cik=1, allow_wide=False)
    assert sorted(choice.tickers) == ["GOOD1", "GOOD2"]


def test_sic_ladder_goes_from_four_digits_down_and_reports_the_level():
    directory = (_cands({f"A{i}": 100.0 for i in range(4)}, sic=3674) + _cands({f"B{i}": 100.0 for i in range(4)}, sic=3672)
                 + _cands({f"C{i}": 100.0 for i in range(4)}, sic=3610))
    four = choose_peers_by_sic(100.0, 3674, directory)
    assert four.source == "sic4" and all(t.startswith("A") for t in four.tickers)
    sparse = [d for d in directory if not d.ticker.startswith("A")] + [Candidate("A0", "A0 Corp", 100.0, 3674, 99)]
    three = choose_peers_by_sic(100.0, 3674, sparse)  # one company with its code: the 3-digit group has B0..B3
    assert three.source == "sic3" and set(three.tickers) == {"A0", "B0", "B1", "B2", "B3"}
    two = choose_peers_by_sic(100.0, 3674, [d for d in directory if d.ticker[0] == "C"])
    assert two.source == "sic2" and len(two.tickers) == 4
    assert choose_peers_by_sic(100.0, 3674, directory[:2]).tickers == []  # fewer than three anywhere
    assert choose_peers_by_sic(100.0, None, directory).tickers == []


def test_sic_ladder_accepts_exactly_three_peers_as_a_last_resort():
    three = choose_peers_by_sic(100.0, 3610, _cands({"A0": 100.0, "A1": 110.0, "A2": 90.0}, sic=3612))
    assert three.source == "sic2" and sorted(three.tickers) == ["A0", "A1", "A2"] and not three.wide_band
    two = choose_peers_by_sic(100.0, 3610, _cands({"A0": 100.0, "A1": 110.0}, sic=3612))
    assert two.tickers == []


def test_sic_ladder_prefers_a_close_industry_in_a_wide_band_over_a_broad_one():
    close = _cands({"K1": 8000.0, "K2": 7000.0, "K3": 6000.0, "K4": 5000.0}, sic=3572)  # 3-digit match, 0.05x to 0.08x the company
    broad = _cands({f"M{i}": 100000.0 for i in range(4)}, sic=3531)  # same 2 digits, in the strict band
    choice = choose_peers_by_sic(100000.0, 3571, close + broad)
    assert choice.source == "sic3" and choice.wide_band and set(choice.tickers) == {"K1", "K2", "K3", "K4"}


# ------------------------------------------------------------------ scoring


def _scored(subject_kw: dict, peer_kw: dict, n_peers=6):
    subject = _m("S", **subject_kw)
    peers = [_m(f"P{i}", **peer_kw) for i in range(n_peers)]
    return score_competitive(subject, peers)


def _points(score, rule):
    return next(r["points"] for r in score.rules if r["id"] == rule)


@pytest.mark.parametrize("dpo,points", [(70.0, 4), (69.9, 2), (57.5, 2), (57.4, 0), (42.6, 0), (42.5, -2), (35.1, -2), (35.0, -4)])
def test_rule_s1_days_to_pay_suppliers(dpo, points):
    assert _points(_scored({"dpo": dpo}, {"dpo": 50.0}), "S1") == points  # ratio 1.40, 1.15, 0.85, 0.70 of a median of 50


@pytest.mark.parametrize("dso,points", [(30.0, 4), (30.5, 2), (42.5, 2), (42.6, 0), (57.4, 0), (57.5, -2), (69.9, -2), (70.0, -4)])
def test_rule_c1_days_to_get_paid(dso, points):
    assert _points(_scored({"dso": dso}, {"dso": 50.0}), "C1") == points  # ratio 0.60, 0.85, 1.15, 1.40


@pytest.mark.parametrize("gm,points", [(0.50, 6), (0.499, 3), (0.44, 3), (0.439, 0), (0.361, 0), (0.36, -3), (0.301, -3), (0.30, -6)])
def test_rule_c2_gross_margin_gap(gm, points):
    assert _points(_scored({"gm": gm}, {"gm": 0.40}), "C2") == points  # gaps +10, +4, -4, -10 points


@pytest.mark.parametrize("step,points", [(0.0, 3), (0.02, 3), (0.025, 1), (0.04, 0), (0.06, -1), (0.10, -3)])
def test_rule_c3_gross_margin_stability(step, points):
    # peers alternate 40% and 36% (std 1.96 points); the company's swing is compared with that
    subject_std = _m(gm=0.40, gm_step=step).gross_margin_std
    assert _points(_scored({"gm": 0.40, "gm_step": step}, {"gm": 0.40, "gm_step": 0.04}), "C3") == points, subject_std


@pytest.mark.parametrize("roic,points", [(0.25, 6), (0.24, 3), (0.19, 3), (0.18, 0), (0.12, 0), (0.11, -3), (0.06, -3), (0.05, -6)])
def test_rule_p1_roic_gap(roic, points):
    assert _points(_scored({"roic": roic}, {"roic": 0.15}), "P1") == points  # gaps +10, +4, -4, -10 points


@pytest.mark.parametrize("opm,points", [(0.28, 4), (0.27, 2), (0.23, 2), (0.22, 0), (0.18, 0), (0.17, -2), (0.13, -2), (0.12, -4)])
def test_rule_p2_operating_margin_gap(opm, points):
    assert _points(_scored({"opm": opm}, {"opm": 0.20}), "P2") == points  # gaps +8, +3, -3, -8 points


@pytest.mark.parametrize("growth,points", [(0.11, 3), (0.10, 1), (0.07, 1), (0.06, 0), (0.04, 0), (0.03, -1), (-0.009, -1), (-0.01, -3)])
def test_rule_p3_sales_growth_gap(growth, points):
    assert _points(_scored({"growth": growth}, {"growth": 0.05}), "P3") == points  # gaps +6, +2, -2, -6 points


def test_rule_p4_share_of_peer_group_sales():
    up = _scored({"rev": 1000, "growth": 0.30}, {"rev": 1000, "growth": 0.0})
    assert up.share_change > 0.15 and _points(up, "P4") == 3
    flat = _scored({"rev": 1000, "growth": 0.0}, {"rev": 1000, "growth": 0.0})
    assert _points(flat, "P4") == 0
    down = _scored({"rev": 1000, "growth": -0.20}, {"rev": 1000, "growth": 0.0})
    assert down.share_change < -0.15 and _points(down, "P4") == -3


def test_scored_rules_components_total_and_quality():
    score = _scored({"gm": 0.50, "dpo": 70, "dso": 30, "roic": 0.25, "opm": 0.28, "growth": 0.12}, {})
    assert score.rules_scored == 8 and score.quality == LIVE and score.not_scored == []
    assert score.components["suppliers"] == 4
    assert score.components["customers"] == sum(_points(score, r) for r in ("C1", "C2", "C3"))
    assert score.total == pytest.approx(sum(score.components.values()))
    assert [r["factor"] for r in score.rules if r["id"] == "S1"] == ["Pays suppliers in 70 days against a peer median of 50 (DPO)"]
    assert [r["factor"] for r in score.rules if r["id"] == "C2"] == ["Gross margin 50.0% against a peer median of 40.0%"]


def test_confidence_follows_rules_scored_peer_count_and_weak_groups():
    full = _scored({}, {}, n_peers=6)
    assert full.confidence == pytest.approx(80.0)
    assert _scored({}, {}, n_peers=4).confidence == pytest.approx(72.0)
    assert _scored({}, {}, n_peers=3).confidence == pytest.approx(60.0) and _scored({}, {}, n_peers=3).quality == PARTIAL
    weak = score_competitive(_m("S"), [_m(f"P{i}") for i in range(6)], weak_peer_group=True)
    assert weak.confidence == pytest.approx(70.0)
    assert SCORING_VERSION == 1


def test_rules_without_a_median_are_skipped_and_listed():
    subject = company_metrics([{**y, "cogs": None, "gross_profit": None} for y in _years()], 1.0, None, "S")  # no cost of revenue
    score = score_competitive(subject, [_m(f"P{i}") for i in range(5)])
    assert {r["id"] for r in score.rules} == {"C1", "P1", "P2", "P3", "P4"}
    assert score.rules_scored == 5 and score.quality == PARTIAL
    assert any("days to pay suppliers (no cost of revenue reported)" in n for n in score.not_scored)
    two = score_competitive(_m("S"), [_m("A"), _m("B")])  # a median needs three peers
    assert two.rules_scored == 0 and two.quality == UNAVAILABLE and len(two.not_scored) == 8


def test_info_rules_carry_zero_points_and_stay_out_of_the_score():
    notes = {"competitors": [{"name": "Rival Inc", "ticker": "RIV"}], "sourcing": ["a", "b"], "raw_materials": "Steel.",
             "concentration": [{"kind": "single", "n": 1, "pct": 26.0, "sentence": "x"}]}
    plain = score_competitive(_m("S", gm=0.5), [_m(f"P{i}") for i in range(5)])
    rich = score_competitive(_m("S", gm=0.5), [_m(f"P{i}") for i in range(5)], notes)
    info = [r for r in rich.rules if r["info"]]
    assert {r["id"] for r in info} == {"I2", "I3", "I4", "I5"} and all(r["points"] == 0 for r in info)
    assert rich.total == plain.total and rich.rules_scored == 8
    ev = score_competitive(company_metrics(_years(), 1.0, 20.0, "S"), [company_metrics(_years(), 1.0, 15.0, f"P{i}") for i in range(4)])
    assert [r["id"] for r in ev.rules if r["info"]] == ["I1"] and ev.rules[-1]["points"] == 0
    none = score_competitive(_m("S"), [_m(f"P{i}") for i in range(5)], {"no_major_customer": "No customer above 10%."})
    assert [r["factor"] for r in none.rules if r["info"]] == ["No customer above 10% of sales is disclosed in the 10-K"]


# ------------------------------------------------------------------ the agent with fake providers

INFO_AAA = {"quoteType": "EQUITY", "sector": "Technology", "industry": "Software", "industryKey": "software", "sectorKey": "technology",
            "marketCap": 10e9, "shortName": "Alpha Corp", "financialCurrency": "USD", "currency": "USD", "enterpriseValue": 12e9,
            "freeCashflow": 0.6e9}
FORBIDDEN_TOP_KEYS = ("excluded_sector", "is_financial", "ev_to_fcf", "fcf")


def _card(ticker: str, source="SEC", **kw) -> dict:
    return {"version": peers_mod.CARD_VERSION, "ticker": ticker, "name": f"{ticker} Corp", "source": source, "cik": None, "sic": 7372,
            "currency": "USD", "market_cap": 9e9, "ev_to_fcf": 18.0, "years": _years(**kw), "error": None}


def _members(n=7):
    return [{"ticker": f"P{i}", "name": f"Peer {i} Inc", "market_cap": 8e9 + i * 1e9} for i in range(n)]


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch):
    """Fake Yahoo info, industry, company cards and 10-K notes; the calls are recorded."""
    calls: list[str] = []
    monkeypatch.setattr(fundamentals, "_load_info", lambda symbol: dict(INFO_AAA))

    def industry(key):
        calls.append(f"industry:{key}")
        return _members()

    def card(ticker, info=None, max_age_days=7, market_cap=None, new_filings=False):
        calls.append(f"card:{ticker}")
        return _card(ticker, gm=0.55 if ticker == "AAA" else 0.40, dpo=70 if ticker == "AAA" else 50)

    def notes(cik, deadline=None, name=None, ticker=None):
        calls.append("notes")
        return filings_mod.FilingNotes(error=filings_mod.NO_ACCESS)

    monkeypatch.setattr(peers_mod, "industry_members", industry)
    monkeypatch.setattr(peers_mod, "company_card", card)
    monkeypatch.setattr(filings_mod, "tenk_notes", notes)
    monkeypatch.setattr(peers_mod, "sic_directory", lambda: [])
    monkeypatch.setattr(peers_mod, "is_sec_filer", lambda ticker: True)  # the cap on new Yahoo-only cards has its own test
    yield calls


def _run(ticker="AAA", budget=12.0) -> AgentAssessment:
    return CompetitiveAgent(budget=budget).evaluate(ticker, make_ohlcv())


def test_agent_live_path_and_raw_data_contract(providers):
    a = _run()
    assert a.data_quality == LIVE and a.agent_key == "competitive" and a.error is None, a.key_factors
    raw = a.raw_data
    for key in ("score_breakdown", "components", "caption", "lead_table", "filing_notes", "metrics_table", "extra_tables", "links",
                "peers", "peers_pending", "peer_source", "peer_group_label", "wide_band", "metrics", "filing", "data_source_label",
                "sector", "industry", "scoring_version"):
        assert key in raw
    assert not any(k in raw for k in FORBIDDEN_TOP_KEYS)
    assert raw["peer_source"] == "yahoo_industry" and raw["peer_group_label"] == "Software (Yahoo industry)"
    assert raw["caption"].startswith("Against 7 peers in Yahoo's Software industry: ")
    lead = raw["lead_table"]
    assert lead["columns"] == ["Measure", "Company", "Peer median", "Peers"] and len(lead["tips"]) == len(lead["rows"])
    assert lead["rows"][0] == ["Gross margin", "55.0%", "40.0%", "7"]
    assert raw["filing_notes"] == [{"topic": "10-K", "text": filings_mod.NO_ACCESS, "quote": None, "url": None}]
    assert {r["component"] for r in raw["score_breakdown"]} == {"suppliers", "customers", "standing"}
    assert a.score > 0 and a.data_as_of == "2025-12-31"
    assert a.summary.startswith("Alpha Corp against 7 peers in Yahoo's Software industry: gross margin 55.0% vs 40.0%")
    assert not any(link["label"] == "SEC EDGAR filings" for link in raw["links"])  # no CIK known: no EDGAR page that would not match
    assert "Evidence only." in a.summary
    assert a.key_factors[0].endswith("(+6)") or "(+" in a.key_factors[0]
    assert {t for t in providers if t.startswith("card:")} == {"card:AAA"} | {f"card:P{i}" for i in range(7)}


@pytest.mark.parametrize("info,reason", [
    ({"quoteType": "ETF", "sector": None, "industry": None}, "fund"),
    ({"quoteType": "EQUITY", "sector": "Financial Services", "industry": "Banks - Diversified"}, "bank or insurer"),
    ({"quoteType": "EQUITY", "sector": "Utilities", "industry": "Utilities - Regulated Electric"}, "Utilities"),
    ({"quoteType": "EQUITY", "sector": "Real Estate", "industry": "REIT - Retail"}, "Real Estate"),
])
def test_agent_not_applicable_returns_at_once_without_peer_calls(providers, monkeypatch, info, reason):
    monkeypatch.setattr(fundamentals, "_load_info", lambda symbol: dict(INFO_AAA, **info))
    started = time.monotonic()
    a = _run()
    assert a.data_quality == UNAVAILABLE and reason in a.summary
    assert time.monotonic() - started < 1.0 and providers == []


def test_agent_provider_errors_are_unavailable_and_fast(providers, monkeypatch):
    def boom(symbol):
        raise LookupError("no quote")

    monkeypatch.setattr(fundamentals, "_load_info", boom)
    started = time.monotonic()
    assert _run().data_quality == UNAVAILABLE
    monkeypatch.setattr(fundamentals, "_load_info", lambda symbol: dict(INFO_AAA))

    def explode(*_a, **_k):
        raise RuntimeError("card service down")

    monkeypatch.setattr(peers_mod, "company_card", explode)
    peers_mod._INFLIGHT.clear()
    assert _run().data_quality == UNAVAILABLE
    assert time.monotonic() - started < 1.0


def _london_world(monkeypatch, subject_cap_gbp=100e9):
    """A pence-quoted London subject whose Yahoo market cap is in pounds, with big and tiny industry members."""
    info = dict(INFO_AAA, currency="GBp", financialCurrency="GBP", marketCap=subject_cap_gbp)
    monkeypatch.setattr(fundamentals, "_load_info", lambda symbol: dict(info))
    asked = []

    def rate(value, from_ccy, to_ccy):
        asked.append(from_ccy)
        return value * 1.25  # the pound, not the penny, is converted

    monkeypatch.setattr(peers_mod, "convert_latest", rate)
    members = ([{"ticker": f"BIG{i}", "name": f"Big {i} Inc", "market_cap": (60e9 + i * 20e9) * 1.25} for i in range(5)]
               + [{"ticker": f"TINY{i}", "name": f"Tiny {i} Inc", "market_cap": 1.5e9 + i * 1e8} for i in range(5)])
    monkeypatch.setattr(peers_mod, "industry_members", lambda key: members)
    return asked


def test_agent_london_market_cap_is_converted_from_pounds_not_pence(providers, monkeypatch):
    asked = _london_world(monkeypatch)
    a = _run()
    assert a.data_quality == LIVE, a.summary
    assert asked and set(asked) == {"GBP"}  # never the minor unit "GBp"
    chosen = [p["ticker"] for p in a.raw_data["peers"]]
    assert sorted(chosen) == [f"BIG{i}" for i in range(5)]  # a hundredfold smaller cap would have picked the tiny ones


def test_agent_without_a_rate_shows_no_peers_rather_than_wrong_ones(providers, monkeypatch):
    _london_world(monkeypatch)

    def no_rate(*_a):
        raise LookupError("no rate")

    monkeypatch.setattr(peers_mod, "convert_latest", no_rate)
    assert peers_mod.market_cap_usd({"marketCap": 5e9, "currency": "GBp"}) is None
    a = _run()
    assert a.data_quality == UNAVAILABLE and "fewer than 3 comparable companies" in a.summary


def test_card_market_cap_is_stored_in_dollars_and_ev_to_fcf_needs_one_currency(monkeypatch):
    monkeypatch.setattr(peers_mod, "convert_latest", lambda value, f, t: value * 1.25)
    monkeypatch.setattr(peers_mod, "fetch_statements", lambda t: FinancialStatements(ticker=t, available=False, error="no data"))
    same = peers_mod.company_card("UKCO.L", {"currency": "GBp", "financialCurrency": "GBP", "marketCap": 80e9,
                                             "enterpriseValue": 100e9, "freeCashflow": 4e9})
    assert same["market_cap"] == 100e9 and same["ev_to_fcf"] == 25.0
    mixed = peers_mod.company_card("DUAL.L", {"currency": "GBp", "financialCurrency": "USD", "marketCap": 80e9,
                                              "enterpriseValue": 100e9, "freeCashflow": 4e9})
    assert mixed["ev_to_fcf"] is None and mixed["market_cap"] == 100e9
    adr = peers_mod.company_card("ADR", {"currency": "USD", "financialCurrency": "TWD", "marketCap": 80e9,
                                         "enterpriseValue": 100e9, "freeCashflow": 4e9})
    assert adr["ev_to_fcf"] is None


def test_agent_unavailable_with_fewer_than_three_peers(providers, monkeypatch):
    monkeypatch.setattr(peers_mod, "industry_members", lambda key: _members(2))
    a = _run()
    assert a.data_quality == UNAVAILABLE and "fewer than 3 comparable companies" in a.summary


def test_agent_missing_statements_are_explained_in_plain_words(providers, monkeypatch):
    monkeypatch.setattr(peers_mod, "company_card", lambda t, info=None, max_age_days=7, market_cap=None, new_filings=False: dict(
        _card(t), years=[], error=peers_mod._reason(LookupError("LookupError: Yahoo Finance returned no financial statements for AAA"))))
    a = _run()
    assert a.data_quality == UNAVAILABLE
    assert "Yahoo has no annual statements for AAA" in a.summary and "funds and trackers" in a.summary
    assert "LookupError" not in a.summary and a.error == "Yahoo Finance returned no financial statements for AAA"


def test_agent_summary_leaves_out_a_clause_when_the_company_figure_is_missing(providers, monkeypatch):
    def card(ticker, info=None, max_age_days=7, market_cap=None, new_filings=False):
        out = _card(ticker, gm=0.55 if ticker == "AAA" else 0.40)
        if ticker == "AAA":
            for y in out["years"]:
                y["receivables"] = None
        return out

    monkeypatch.setattr(peers_mod, "company_card", card)
    a = _run()
    assert a.data_quality in (LIVE, PARTIAL) and "n/a" not in a.summary and "gets paid" not in a.summary
    assert a.summary.startswith("Alpha Corp against 7 peers in Yahoo's Software industry: gross margin 55.0% vs 40.0%, ROIC")


def test_agent_edgar_link_is_built_from_the_cik(providers, monkeypatch):
    monkeypatch.setattr(peers_mod, "company_card", lambda t, info=None, max_age_days=7, market_cap=None, new_filings=False: dict(_card(t), cik=1045810))
    links = {link["label"]: link["url"] for link in _run().raw_data["links"]}
    assert "CIK=0001045810" in links["SEC EDGAR filings"]


def test_agent_notes_that_raise_are_not_reported_as_still_being_read(providers, monkeypatch):
    def broken(cik, deadline=None, name=None, ticker=None):
        raise KeyError("filings")

    monkeypatch.setattr(filings_mod, "tenk_notes", broken)
    note = _run().raw_data["filing_notes"][0]
    assert note["text"] == "The 10-K could not be read; analyze again later"


def test_agent_slow_peer_fetch_gives_partial_within_the_budget(providers, monkeypatch):
    def slow_card(ticker, info=None, max_age_days=7, market_cap=None, new_filings=False):
        if ticker == "P6":
            time.sleep(0.7)
        return _card(ticker, gm=0.55 if ticker == "AAA" else 0.40)

    monkeypatch.setattr(peers_mod, "company_card", slow_card)
    started = time.monotonic()
    a = _run(budget=0.5)
    assert time.monotonic() - started < 0.8
    assert a.data_quality == PARTIAL and a.raw_data["peers_pending"] == ["P6"]
    assert any("still loading" in f for f in a.key_factors)  # the fetch keeps running and fills the cache for the next run


def test_agent_warm_disk_cache_needs_no_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(fundamentals, "_load_info", lambda symbol: dict(INFO_AAA))
    monkeypatch.setattr(peers_mod, "sic_directory", lambda: [])
    peers_mod.competitive_dir()
    for t in ["AAA"] + [m["ticker"] for m in _members()]:
        peers_mod._card_path(t).write_text(json.dumps(_card(t)))
    (peers_mod.competitive_dir() / "industry_software.json").write_text(json.dumps(_members()))
    started = time.monotonic()
    a = _run()
    assert a.data_quality == LIVE and time.monotonic() - started < 0.3
    assert a.raw_data["data_source_label"] == "SEC EDGAR 10-K figures (company and 7 peers)"


def test_card_cache_ignores_old_versions_and_failures_are_not_written(monkeypatch):
    path = peers_mod._card_path("OLD")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(_card("OLD"), version=0)))
    assert peers_mod.cached_card("OLD") is None
    monkeypatch.setattr(peers_mod, "fetch_statements", lambda t: FinancialStatements(ticker=t, available=False, error="no data"))
    card = peers_mod.company_card("NOPE", {"shortName": "No Pe"})
    assert card["error"] and card["years"] == [] and not peers_mod._card_path("NOPE").exists()
    assert peers_mod.recently_failed("NOPE")
    cards, pending, failed = peers_mod.peer_cards(["NOPE"], time.monotonic() + 1)
    assert cards == {} and pending == [] and failed == ["NOPE"]  # skipped for 30 minutes, no second fetch


def test_yahoo_only_cards_are_limited_per_evaluation(monkeypatch):
    monkeypatch.setattr(peers_mod, "company_card", lambda t, info=None, age=7, cap=None, new_filings=False: _card(t, source="Yahoo"))
    tickers = [f"Y{i}" for i in range(6)]
    cards, pending, failed = peers_mod.peer_cards(tickers, time.monotonic() + 2, yahoo_budget=4)
    assert len(cards) == 4 and len(pending) == 2 and failed == []


# ------------------------------------------------------------------ no existing score changes


def test_registry_has_the_agent_as_owner_evidence_at_weight_zero():
    cls = get_agent_classes()["competitive"]
    assert cls.role == OWNER and cls.default_weight == 0.0 and cls.category == "Fundamental" and cls.score_scale == 20.0


def test_shipped_config_gives_weight_zero_and_a_local_override_without_the_key_does_too():
    defaults = {k: c.default_weight for k, c in get_agent_classes().items()}
    roles = {k: c.role for k, c in get_agent_classes().items()}
    shipped = validate_config(load_raw_config(), defaults, roles)
    assert shipped.owner_weights["competitive"] == 0.0 and "competitive" in shipped.agent_order
    assert all(profile["competitive"] == 0.0 for profile in shipped.regime_weights.values())
    local = {"owner_weights": {"fundamentals": 0.5, "shareholder": 0.5}, "agent_order": ["shareholder", "fundamentals"]}
    cfg = validate_config({**load_raw_config(), **local}, defaults, roles)
    assert cfg.owner_weights["competitive"] == 0.0 and cfg.agent_order.index("competitive") > cfg.agent_order.index("fundamentals")


def _assess(key: str, score: float) -> AgentAssessment:
    return AgentAssessment(key, score, "NEUTRAL", 80.0, "", agent_key=key, data_quality=LIVE, raw_data={})


def test_a_strongly_negative_competitive_score_does_not_move_the_owner_decision():
    cfg = ScoringConfig(regime_weights={}, thresholds={"strong_buy": 45.0, "buy": 15.0, "sell": -15.0, "strong_sell": -45.0},
                        owner_weights={"fundamentals": 0.45, "shareholder": 0.55, "competitive": 0.0},
                        owner_thresholds={"strong_buy": 45.0, "buy": 15.0, "sell": -15.0, "strong_sell": -45.0})
    roles = {"fundamentals": OWNER, "shareholder": OWNER, "competitive": OWNER}
    base = [_assess("fundamentals", 40), _assess("shareholder", 30)]
    without = aggregate_layers(base, cfg, None, {}, roles)
    withit = aggregate_layers(base + [_assess("competitive", -33)], cfg, None, {"competitive": 20.0}, roles)
    assert (withit.decision, withit.owner.composite_score, withit.owner.confidence, withit.owner.coverage) == (
        without.decision, without.owner.composite_score, without.owner.confidence, without.owner.coverage)
    entry = next(c for c in withit.owner.contributions if c.agent_key == "competitive")
    assert not entry.included and "informational" in entry.exclusion_reason


def test_owner_concerns_ignore_a_weight_zero_agent():
    from core.portfolio import _owner_concerns

    class Rec:
        contributions = [{"agent_key": "fundamentals", "layer": "owner", "included": True},
                         {"agent_key": "competitive", "layer": "owner", "included": False}]
        agent_assessments = [
            AgentAssessment("f", 0, "NEUTRAL", 50, "", agent_key="fundamentals", data_quality=LIVE, raw_data={"score_breakdown": []}),
            AgentAssessment("c", 0, "NEUTRAL", 50, "", agent_key="competitive", data_quality=LIVE,
                            raw_data={"score_breakdown": [{"factor": "Gross margin far below peers", "points": -6}]}),
        ]
        warnings: list[str] = []

    assert _owner_concerns(Rec()) == []


def test_filings_and_company_facts_share_one_sec_throttle():
    assert filings_mod._rate_lock is sec_edgar._rate_lock
    assert filings_mod._last_request is sec_edgar._last_request
    assert filings_mod.MIN_REQUEST_INTERVAL == sec_edgar.MIN_REQUEST_INTERVAL


def test_filing_notes_cache_by_accession_and_say_why_when_missing(monkeypatch):
    assert filings_mod.tenk_notes(1234).error == filings_mod.NO_ACCESS  # no contact email set
    monkeypatch.setenv(sec_edgar.SEC_USER_AGENT_ENV, "test test@example.com")
    assert filings_mod.tenk_notes(None).error == filings_mod.NO_TENK
    ref = {"accession": "0000000000-26-000001", "filed": "2026-02-01", "period": "2025-12-31", "primary_doc": "x.htm", "url": "https://example.invalid/x.htm"}
    monkeypatch.setattr(filings_mod, "latest_10k", lambda cik: ref)
    page = ("<p>Item 1. Business</p><p>Competition</p><p>" + "We compete on price and quality with many firms. " * 6 + "</p>"
            "<p>Net sales to our largest customer accounted for approximately 31% of our net sales in 2025.</p>")
    reads = []
    monkeypatch.setattr(filings_mod, "fetch_document", lambda r: reads.append(1) or page)
    first = filings_mod.tenk_notes(1234, name="Test Corp")
    assert first.found and first.concentration[0]["pct"] == 31.0 and first.ref["accession"] == ref["accession"]
    second = filings_mod.tenk_notes(1234, name="Test Corp")
    assert second.concentration == first.concentration and len(reads) == 1  # the second read comes from disk
    stored = json.loads((filings_mod.competitive_dir() / f"tenk_1234_{ref['accession']}.json").read_text())
    assert "error" not in stored and "<p>" not in json.dumps(stored)  # extracts only, never the document
    monkeypatch.setattr(filings_mod, "latest_10k", lambda cik: None)
    assert filings_mod.tenk_notes(99).error == filings_mod.NO_TENK
    monkeypatch.setattr(filings_mod, "latest_10k", lambda cik: {"url": "x"})  # a malformed index entry
    broken = filings_mod.tenk_notes(98)
    assert broken.error == "The 10-K could not be read; analyze again later" and not broken.pending
