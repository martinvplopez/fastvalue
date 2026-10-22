"""Newest statements for live scoring, SEC caches that follow new filings, and the score board (no network)."""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest

import config.settings as settings
import data.financial_statements as fs
import data.sec_edgar as se
from agents.master_agent import display_number
from agents.scoring import normalize_score
from core.score_board import score_board
from data.financial_statements import FinancialStatements
from data.history import _connect, save_watchlist
from data.portfolio import save_portfolio


def _st(year_ends: list[str], fcf: float, source: str = "Yahoo Finance") -> FinancialStatements:
    cols = [pd.Timestamp(y) for y in year_ends]
    income = pd.DataFrame({c: {"Total Revenue": 100.0, "EBIT": 20.0, "Net Income": 10.0, "Pretax Income": 15.0} for c in cols})
    cashflow = pd.DataFrame({c: {"Free Cash Flow": fcf} for c in cols})
    return FinancialStatements(ticker="MU", available=True, income=income, cashflow=cashflow, source=source)


# ------------------------------------------------------------------ newest statements


def test_sec_statements_win_when_they_have_a_newer_fiscal_year(monkeypatch):
    yahoo = _st(["2025-08-31", "2024-08-31"], 1.7)
    yahoo.splits = pd.Series([2.0], index=[pd.Timestamp("2000-05-01")])
    sec = _st(["2026-09-03", "2025-08-28"], 59.0, source=se.SOURCE_LABEL)
    monkeypatch.setattr(fs, "fetch_statements", lambda t: yahoo)
    monkeypatch.setattr(fs, "_sec_statements", lambda symbol, splits: sec)
    out = fs.fetch_current_statements("MU")
    assert out is sec and out.value("cashflow", "Free Cash Flow") == 59.0
    assert out.splits.equals(yahoo.splits)
    assert "2026-09-03" in out.basis_note and "2025-08-31" in out.basis_note


def test_yahoo_is_kept_on_the_same_fiscal_year_or_without_sec(monkeypatch):
    yahoo = _st(["2025-08-31", "2024-08-31"], 1.7)
    monkeypatch.setattr(fs, "fetch_statements", lambda t: yahoo)
    monkeypatch.setattr(fs, "_sec_statements", lambda symbol, splits: _st(["2025-08-28"], 1.6, source=se.SOURCE_LABEL))
    assert fs.fetch_current_statements("MU") is yahoo  # 52/53-week year end three days apart: the same year
    fs.fetch_current_statements.cache_clear()
    monkeypatch.setattr(fs, "_sec_statements", lambda symbol, splits: None)
    out = fs.fetch_current_statements("MU")
    assert out is yahoo and out.basis_note is None


def test_sec_statements_stand_in_when_yahoo_has_none(monkeypatch):
    monkeypatch.setattr(fs, "fetch_statements", lambda t: FinancialStatements(ticker="MU", available=False, error="401"))
    sec = _st(["2026-09-03"], 59.0, source=se.SOURCE_LABEL)
    monkeypatch.setattr(fs, "_sec_statements", lambda symbol, splits: sec)
    out = fs.fetch_current_statements("MU")
    assert out is sec and "Yahoo has none" in out.basis_note


def test_no_sec_access_means_no_sec_lookup(monkeypatch):
    monkeypatch.setattr(se, "has_sec_access", lambda: False)
    assert fs._sec_statements("MU", pd.Series(dtype=float)) is None


# ------------------------------------------------------------------ caches that follow new filings


def _subs(*filings: tuple[str, str]) -> dict:
    return {"filings": {"recent": {"form": [f for f, _ in filings], "acceptanceDateTime": [a for _, a in filings],
                                   "filingDate": [a[:10] for _, a in filings]}}}


def test_latest_periodic_filing_ignores_other_forms():
    subs = _subs(("8-K", "2026-10-09T12:00:00.000Z"), ("10-K", "2026-10-03T16:05:00.000Z"), ("10-Q", "2026-06-26T16:00:00.000Z"))
    assert se.latest_periodic_filing(subs) == pd.Timestamp("2026-10-03T16:05:00Z")
    assert se.latest_periodic_filing(_subs(("8-K", "2026-10-09T12:00:00.000Z"))) is None


def test_filed_since_is_false_when_the_filing_list_fails(monkeypatch):
    monkeypatch.setattr(se, "load_submissions", lambda cik, *a: (_ for _ in ()).throw(se.SecDataError("down")))
    assert se.filed_since(1, 0.0) is False


def test_cached_facts_refresh_after_a_new_10k_only_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(se, "_get_json", lambda url: calls.append(url) or {"entityName": "Test Co", "facts": {}})
    monkeypatch.setattr(se, "load_submissions", lambda cik, *a: _subs(("10-K", "2026-10-03T16:05:00.000Z")) | {"sic": "3674"})
    se._load_one(7, 7)
    assert len(calls) == 1  # first read downloads
    facts_path = se.sec_cache_dir() / f"facts_v{se.CONCEPT_VERSION}_{7:010d}.parquet"
    written_before_filing = pd.Timestamp("2026-10-01T00:00:00Z").timestamp()
    os.utime(facts_path, (written_before_filing, written_before_filing))
    monkeypatch.setattr(time, "time", lambda: written_before_filing + 3 * 86400)  # three days old: fresh by age
    se._load_one(7, 7)
    assert len(calls) == 1  # studies keep the weekly cache
    se._load_one(7, 7, new_filings=True)
    assert len(calls) == 2  # the live agents read the new 10-K now
    se._load_one(7, 7, new_filings=True)
    assert len(calls) == 2  # and only once


# ------------------------------------------------------------------ the score board


def _record(ticker: str, when: str, decision: str, agents: dict[str, tuple[float, float]], owner: float | None = 0.0) -> None:
    scores = {k: {"score": s, "normalized": 0.0, "confidence": c, "quality": "LIVE", "included": True} for k, (s, c) in agents.items()}
    with _connect() as conn:
        conn.execute(
            "INSERT INTO analyses (ticker, generated_at, decision, composite_score, confidence, agent_scores, owner_score)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)", (ticker, when, decision, owner or 0.0, 60.0, json.dumps(scores), owner))


def test_score_board_recomposes_the_latest_rated_analysis(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    recent, old = (now - pd.Timedelta(days=2)).isoformat(), (now - pd.Timedelta(days=200)).isoformat()
    _record("AAA", (now - pd.Timedelta(days=9)).isoformat(), "SELL", {"shareholder": (-60.0, 80.0)}, -50.0)
    _record("AAA", recent, "HOLD", {"shareholder": (20.0, 80.0), "momentum": (90.0, 90.0)}, 1.0)  # stored owner score is stale
    _record("BBB", recent, "BUY", {"shareholder": (40.0, 70.0), "fundamentals": (40.0, 70.0)}, 30.0)
    _record("FUND", recent, "NOT_RATED", {"momentum": (50.0, 80.0)})
    _record("OLD", old, "BUY", {"shareholder": (60.0, 80.0)}, 40.0)
    save_watchlist(["BBB"])
    save_portfolio({"positions": [{"ticker": "AAA", "shares": 1, "avg_cost": 10}], "cash": 0})
    board = score_board()
    rows = {r["ticker"]: r for r in board["rows"]}
    assert set(rows) == {"AAA", "BBB"}  # not rated and older than 90 days are left out
    assert [r["ticker"] for r in board["rows"]] == ["BBB", "AAA"]  # highest first
    # The latest AAA analysis, owner layer only on today's weights: one owner agent with data is the whole score.
    assert rows["AAA"]["score"] == pytest.approx(normalize_score(20.0, 35.0), abs=0.05)
    assert rows["AAA"]["recorded"] == 1.0
    assert rows["AAA"]["held"] and not rows["AAA"]["watched"] and rows["BBB"]["watched"]
    assert board["thresholds"]["buy"] > board["thresholds"]["sell"]


def test_scores_route():
    from web.app import app

    res = app.test_client().get("/api/scores")
    assert res.status_code == 200 and res.get_json()["rows"] == []


# ------------------------------------------------------------------ numbers as the page shows them


def test_display_number_rounds_like_the_page():
    assert display_number(64.5) == "65" and display_number(38.5, 0, True) == "+39"
    assert display_number(-4.65, 1, True) == "-4.7" and display_number(12, 1, True) == "+12.0"
    assert display_number(-0.04, 1, True) == "0.0"


def test_fundamentals_rules_name_their_period_and_source():
    from agents.fundamental_agent import FundamentalAgent
    from tests.test_fundamental_agent import _ohlcv, _profile, _statements

    st = _statements()
    st.basis_note = "Statements from the SEC 10-K for the fiscal year to 2025-12-31: Yahoo's still end at 2024-12-31"
    result = FundamentalAgent().evaluate_with_data(_profile(), st, _ohlcv())
    factors = [r["factor"] for r in result.raw_data["score_breakdown"]]
    assert any(f.startswith("EV/FCF") and "year to Dec 2025" in f for f in factors)
    assert any(f.startswith("PER") and "last twelve months" in f for f in factors)  # PER uses Yahoo's TTM net income
    assert any(f.startswith("ROIC ") and "year to Dec 2025" in f for f in factors)
    assert st.basis_note in result.key_factors and result.raw_data["statements_note"] == st.basis_note
