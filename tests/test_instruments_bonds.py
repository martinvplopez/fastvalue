"""ISIN validation, bond detection and the /api/suggest shape. No network: Yahoo's search is replaced."""

from __future__ import annotations

import pytest

import data.instruments as ins


@pytest.mark.parametrize("code", ["US0378331005", "DE0001102580", "IE00B4L5Y983", "US912828YK04", "us0378331005"])
def test_valid_isins(code):
    assert ins.is_valid_isin(code)


@pytest.mark.parametrize("code", ["US0378331004", "AAPL", "", "US037833100", "1S0378331005", "US0378331005X", None])
def test_invalid_isins(code):
    assert not ins.is_valid_isin(code)


def test_bond_only_when_known():
    assert ins.is_bond("BOND") and ins.is_bond("bond") and ins.is_bond(quote_type="BOND")
    assert not ins.is_bond() and not ins.is_bond("STOCK", "EQUITY") and not ins.is_bond("", "")


@pytest.fixture
def fake_search(monkeypatch, tmp_path):
    monkeypatch.setattr(ins.settings, "STORE_DIR", tmp_path)
    box = {"quotes": []}
    monkeypatch.setattr(ins, "_quotes", lambda q, limit: box["quotes"])
    return box


def test_isin_without_hits_gets_the_general_note(fake_search):
    r = ins.suggestion_report("DE0001102580")
    assert r["isin"] and not r["bond"] and r["suggestions"] == [] and "Bonds are not supported" in r["note"]


def test_plain_symbol_has_no_note(fake_search):
    fake_search["quotes"] = [{"symbol": "HEIA.AS", "quoteType": "EQUITY", "shortname": "Heineken"}]
    r = ins.suggestion_report("HEIA")
    assert not r["isin"] and not r["bond"] and r["note"] is None and r["suggestions"][0]["ticker"] == "HEIA.AS"


def test_known_bond_from_import_offers_no_listings(fake_search):
    ins.save_map({"DE0001102580": {"isin": "DE0001102580", "asset_class": "BOND"}})
    fake_search["quotes"] = [{"symbol": "IWDA.L", "quoteType": "ETF"}]
    r = ins.suggestion_report("DE0001102580")
    assert r["bond"] and r["suggestions"] == [] and r["note"].startswith("This looks like a bond")


def test_yahoo_bond_quote_type_is_a_bond(fake_search):
    fake_search["quotes"] = [{"symbol": "XYZ", "quoteType": "BOND"}]
    assert ins.suggestion_report("US912828YK04")["bond"]


def test_route_shape(fake_search):
    from web.app import app

    app.config["TESTING"] = True
    body = app.test_client().get("/api/suggest/US912828YK04").get_json()
    assert set(body) == {"query", "suggestions", "isin", "bond", "note"} and body["isin"]


def test_bond_class_is_not_priced_and_skips_the_search(fake_search):
    import core.broker_import as bi

    assert "BOND" in ins.UNPRICED_CLASSES
    ledger = {"positions": [{"isin": "DE0001102580", "name": "Bund", "asset_class": "BOND", "shares": 1, "cost_broker": 100.0}]}
    row = bi._instrument_rows(ledger, {})[0]
    assert row["bond"] and "bonds are not supported" in row["note"]


@pytest.mark.parametrize("answers, calls, sleeps", [
    ([[]], 1, []),                                            # an empty answer is final (yfinance caches it)
    ([[{"symbol": "HEIA.AS"}]], 1, []),                       # quotes at once
    ([RuntimeError("401"), [{"symbol": "HEIA.AS"}]], 2, [1.5]),  # an error is retried after a pause
    ([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")], 3, [1.5, 3.0]),  # no pause after the last try
])
def test_search_retries_only_after_errors(monkeypatch, answers, calls, sleeps):
    import time

    import data.yahoo as yahoo

    seen, slept = [], []

    def fake_call(fn, label="yahoo"):
        answer = answers[len(seen)]
        seen.append(label)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(yahoo, "yahoo_call", fake_call)
    monkeypatch.setattr(time, "sleep", slept.append)
    found = ins._search("NL0000009165")
    assert len(seen) == calls and slept == sleeps
    assert found == (["HEIA.AS"] if isinstance(answers[-1], list) and answers[-1] else [])


def test_index_derivatives_only_when_an_index_was_asked_for(fake_search):
    fake_search["quotes"] = [
        {"symbol": "NESN.SW", "quoteType": "EQUITY", "shortname": "Nestle", "exchDisp": "Zurich"},
        {"symbol": "AARC.Z", "quoteType": "INDEX", "shortname": "Decrement index"},
    ]
    names = [s["ticker"] for s in ins.suggestion_report("NESN")["suggestions"]]
    assert names == ["NESN.SW"]
    assert ins.suggestion_report("NESN")["suggestions"][0]["exchange"] == "Zurich"
    fake_search["quotes"] = [{"symbol": "^GSPC", "quoteType": "INDEX", "shortname": "S&P 500"}]
    assert [s["ticker"] for s in ins.suggestion_report("^GSPC")["suggestions"]] == ["^GSPC"]
