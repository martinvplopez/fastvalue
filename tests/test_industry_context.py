"""Industry context: source selection, the curated source file and the documents payload (all providers mocked)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import data.industry_context as ic
from data import fundamentals, peers

CONFIG = Path(ic.SOURCES_PATH)
YAHOO_SECTORS = {"Technology", "Healthcare", "Financial Services", "Consumer Cyclical", "Consumer Defensive",
                 "Industrials", "Energy", "Basic Materials", "Communication Services", "Utilities", "Real Estate"}


def _filing(kind: str, filed: str, url: str, exhibits: list[dict] | None = None, title: str = "t") -> dict:
    return {"form": "8-K", "kind": kind, "filed": filed, "period": None, "title": title, "items": [], "url": url,
            "index_url": url, "exhibits": exhibits or []}


def _result(name: str, filings: list[dict], status: str = "ok") -> dict:
    return {"ticker": "X", "name": name, "cik": 1, "status": status, "message": None, "fix": None, "filings": filings,
            "latest": {}, "links": []}


@pytest.fixture
def world(monkeypatch):
    """KO with industry members; recent_filings answers come from the ``filings`` dict."""
    info = {"quoteType": "EQUITY", "sector": "Consumer Defensive", "industry": "Beverages - Non-Alcoholic",
            "industryKey": "beverages-non-alcoholic", "shortName": "Coca-Cola", "marketCap": 250e9, "currency": "USD"}
    members = [{"ticker": t, "name": n, "market_cap": c} for t, n, c in [
        ("KO", "Coca-Cola", 250e9), ("PEP", "PepsiCo", 220e9), ("MNST", "Monster Beverage", 55e9),
        ("KDP", "Keurig Dr Pepper", 45e9), ("CELH", "Celsius", 12e9), ("FIZZ", "National Beverage", 4e9),
        ("COKE", "Coca-Cola Consolidated", 9e9)]]
    filings: dict[str, object] = {}
    monkeypatch.setattr(fundamentals, "_load_info", lambda s: dict(info))
    monkeypatch.setattr(peers, "industry_members", lambda key: members)
    monkeypatch.setattr(peers, "market_cap_usd", lambda i: i["marketCap"])
    monkeypatch.setattr(peers, "sic_directory", lambda: [])
    monkeypatch.setattr(peers, "sic_records", lambda: {})

    def fake(ticker: str):
        value = filings.get(ticker, _result(ticker, []))
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(ic, "_recent_filings", fake)
    return info, filings


def test_sources_order_and_dedup():
    got = ic.sources_for("Energy", "Oil & Gas E&P")
    scopes = [s["scope"] for s in got]
    assert scopes == sorted(scopes, key=["industry", "sector", "general"].index)
    urls = [s["url"] for s in got]
    assert len(urls) == len(set(urls))
    assert got[0]["scope"] == "industry" and "rig" in got[0]["url"]


def test_unknown_sector_gets_general_only():
    got = ic.sources_for("Nonsense", None)
    assert got and {s["scope"] for s in got} == {"general"}


def test_latest_documents_takes_filing_or_exhibit_newest_first():
    result = _result("PepsiCo", [
        _filing("earnings", "2026-07-10", "https://sec/e2", [
            {"label": "Earnings press release (Ex. 99.1)", "kind": "earnings", "url": "https://sec/e2x"},
            {"label": "Investor presentation (Ex. 99.2)", "kind": "presentation", "url": "https://sec/p2x"}]),
        _filing("earnings", "2026-04-10", "https://sec/e1"),
        _filing("presentation", "2026-03-01", "https://sec/p0"),
    ])
    docs = {d["kind"]: d for d in ic.latest_documents(result, "PEP", "PepsiCo", True)}
    assert docs["earnings"]["url"] == "https://sec/e2x" and docs["earnings"]["filed"] == "2026-07-10"
    assert docs["presentation"]["label"] == "Investor presentation (Ex. 99.2)"
    assert docs["presentation"]["is_peer"] is True


def test_payload_company_and_peers(world):
    _, filings = world
    filings["KO"] = _result("Coca-Cola", [_filing("earnings", "2026-07-22", "https://sec/ko")])
    filings["PEP"] = _result("PepsiCo", [_filing("earnings", "2026-07-10", "https://sec/pep", [
        {"label": "Investor presentation (Ex. 99.2)", "kind": "presentation", "url": "https://sec/pep-p"}])])
    got = ic.industry_context("KO")
    assert got["status"] == "ok" and got["sector"] == "Consumer Defensive"
    assert got["industry"] == "Beverages - Non-Alcoholic"
    assert got["sources"][0]["scope"] in {"industry", "sector"}
    mine = [d for d in got["documents"] if not d["is_peer"]]
    assert [d["ticker"] for d in mine] == ["KO"]
    assert any(d["ticker"] == "PEP" and d["kind"] == "presentation" for d in got["documents"])
    assert len({d["ticker"] for d in got["documents"] if d["is_peer"]}) <= ic.MAX_PEERS
    assert any("no earnings release or presentation" in n for n in got["notes"])


def test_failing_peer_is_skipped_with_a_note(world):
    _, filings = world
    filings["KO"] = _result("Coca-Cola", [_filing("earnings", "2026-07-22", "https://sec/ko")])
    filings["PEP"] = RuntimeError("SEC refused")
    got = ic.industry_context("KO")
    assert got["status"] == "ok"
    assert any("PepsiCo" in n and "skipped" in n for n in got["notes"])
    assert any(d["ticker"] == "KO" for d in got["documents"])


def test_not_sec_filer_peer_gets_a_note(world):
    _, filings = world
    filings["PEP"] = _result("PepsiCo", [], status="not_sec_filer")
    got = ic.industry_context("KO")
    assert any("PepsiCo" in n and "no 10-K" in n for n in got["notes"])


def test_no_access_blocks_documents_but_keeps_sources(world):
    _, filings = world
    filings["KO"] = _result("Coca-Cola", [], status="no_access") | {"message": "Set the SEC contact", "fix": "Backtest tab"}
    got = ic.industry_context("KO")
    assert got["status"] == "no_access" and got["fix"] == "Backtest tab"
    assert got["documents"] == [] and got["sources"]


def test_missing_connector_is_a_note_not_a_crash(world, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(fundamentals, "_load_info", lambda s: {"quoteType": "EQUITY", "sector": "Energy", "industry": "Uranium",
                                                              "shortName": "Cameco"})
    monkeypatch.setattr(ic, "_recent_filings", lambda t: (_ for _ in ()).throw(RuntimeError(ic.NO_CONNECTOR)))
    got = ic.industry_context("CCJ")
    assert got["sources"] and got["documents"] == []
    assert any(ic.NO_CONNECTOR in n for n in got["notes"])


def test_fund_gets_sources_only(monkeypatch):
    monkeypatch.setattr(fundamentals, "_load_info", lambda s: {"quoteType": "ETF", "shortName": "Fund"})
    got = ic.industry_context("VWCE.DE")
    assert got["documents"] == [] and got["sources"]
    assert got["status"] == "ok"


def test_yahoo_failure_returns_general_sources(monkeypatch):
    def boom(_):
        raise LookupError("nothing")

    monkeypatch.setattr(fundamentals, "_load_info", boom)
    got = ic.industry_context("ZZZZ")
    assert got["status"] == "error" and got["sources"]


def test_route_validates_and_returns_payload(world):
    from web.app import app

    client = app.test_client()
    ok = client.get("/api/industry/KO")
    assert ok.status_code == 200 and ok.get_json()["ticker"] == "KO"
    assert client.get("/api/industry/bad%20ticker!").status_code in (400, 404)


# ------------------------------------------------------------------ the curated file


def _all_sources(data: dict):
    for s in data["general"]:
        yield "general", s
    for sector, block in data["sectors"].items():
        for s in block["sources"]:
            yield sector, s
        for industry, items in block["industries"].items():
            for s in items:
                yield f"{sector}/{industry}", s


def test_sources_file_is_valid():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["version"] == 1 and re.match(r"^\d{4}-\d{2}-\d{2}$", data["checked"])
    assert set(data["sectors"]) <= YAHOO_SECTORS
    count = 0
    for where, s in _all_sources(data):
        count += 1
        for field in ("name", "publisher", "what", "cadence"):
            assert isinstance(s.get(field), str) and s[field].strip(), (where, field)
        assert s["url"].startswith("https://"), (where, s["url"])
        assert s["free"] is True and isinstance(s["conference"], bool), where
        assert not re.search(r"[–—]", json.dumps(s, ensure_ascii=False)), where
    assert count >= 30


def test_sources_file_industry_names_belong_to_their_sector():
    yahoo = {
        "Technology": {"Semiconductors", "Solar"},
        "Healthcare": {"Healthcare Plans"},
        "Consumer Cyclical": {"Auto Manufacturers", "Auto Parts", "Residential Construction", "Apparel Retail"},
        "Consumer Defensive": {"Farm Products", "Packaged Foods", "Grocery Stores"},
        "Industrials": {"Airlines", "Aerospace & Defense", "Airports & Air Services"},
        "Energy": {"Oil & Gas E&P", "Oil & Gas Equipment & Services", "Oil & Gas Drilling"},
        "Basic Materials": {"Gold", "Steel", "Copper"},
        "Utilities": {"Utilities - Renewable"},
        "Real Estate": {"Real Estate - Development"},
    }
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    for sector, block in data["sectors"].items():
        assert set(block["industries"]) <= yahoo.get(sector, set()), sector
