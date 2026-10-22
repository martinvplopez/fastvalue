"""Typed holdings and the saved review: the opening date, the fix for an unpriced typed holding and the review kept
across reloads (synthetic data only, no network)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core import portfolio as review
from data import portfolio as pf


def test_an_opening_date_in_the_future_is_refused():
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(pf.PortfolioError, match="cannot be in the future"):
        pf.upsert_position({"ticker": "NVDA", "shares": 1, "avg_cost": 10, "opened": tomorrow})
    saved = pf.upsert_position({"ticker": "NVDA", "shares": 1, "avg_cost": 10, "opened": date.today().isoformat()})
    assert saved["positions"][0]["opened"] == date.today().isoformat()


def test_the_position_endpoint_answers_400_for_a_future_date():
    import web.app as web_app

    resp = web_app.app.test_client().post("/api/portfolio/position", json={"ticker": "NVDA", "shares": 1, "avg_cost": 10, "opened": "2999-01-15"})
    assert resp.status_code == 400 and "future" in resp.get_json()["error"]


def _unpriced(source: str, position_source: str | None = None) -> dict:
    u = {"ticker": "ZZZQX", "name": None, "class": "unpriced", "value": 100.0, "error": "No market data", "source": source}
    if position_source:
        u["position_source"] = position_source
    return u


def test_an_unpriced_typed_holding_is_not_sent_to_the_import_card():
    typed = review._unpriced_item(_unpriced("portfolio", "manual"), 1000.0, "EUR", None)
    assert "import card" not in typed["fix"] and "Remove it in Your holdings" in typed["fix"]
    imported = review._unpriced_item(_unpriced("import"), 1000.0, "EUR", None)
    assert "import card" in imported["fix"]
    mapped = review._unpriced_item(_unpriced("portfolio", "generic"), 1000.0, "EUR", None)
    assert "import card" in mapped["fix"]


def test_the_last_review_comes_back_until_the_holdings_change():
    import web.app as web_app

    client = web_app.app.test_client()
    pf.save_portfolio({"cash": 10.0, "currency": "EUR", "positions": [{"ticker": "KO", "shares": 2, "avg_cost": 50}]})
    assert client.get("/api/portfolio/review/last").get_json() == {}
    pf.save_last_review({"reviewed_at": "2026-10-21T17:08:00", "positions": []})
    assert client.get("/api/portfolio/review/last").get_json()["reviewed_at"] == "2026-10-21T17:08:00"
    pf.upsert_position({"ticker": "AAPL", "shares": 1, "avg_cost": 100})
    assert client.get("/api/portfolio/review/last").get_json() == {}


def test_the_price_of_a_candidate_note_has_separators_and_a_symbol():
    assert review._price_text(1676.694, "USD") == "$1,676.69"
    assert review._price_text(12.5, "CHF") == "12.50 CHF"
