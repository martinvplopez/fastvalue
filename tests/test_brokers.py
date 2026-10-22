"""The broker registry and the generic CSV any broker can be converted to (synthetic rows only, no network)."""

from __future__ import annotations

import io
import time

import pytest

from core import broker_import
from core.ledger import build_ledger
from data import brokers, generic_ledger as gl
from data import portfolio as pf
from data.trade_republic import ExportFormatError


def test_registry_lists_both_parsers_and_rejects_unknown_brokers():
    keys = [b["key"] for b in brokers.describe()]
    assert keys == ["trade_republic", "generic"] and brokers.DEFAULT == "trade_republic"
    assert all("parse" not in b for b in brokers.describe())
    with pytest.raises(ExportFormatError):
        brokers.parse("nope", b"date,type,amount\n")
    assert brokers.label("generic").startswith("Any broker") and brokers.label(None) == "broker"


def test_template_parses_and_the_ledger_reconciles():
    parsed = brokers.parse("generic", gl.template_csv().encode())
    assert parsed["source"] == "generic" and parsed["rows"] == 7 and all(c["ok"] for c in parsed["checks"])
    kinds = [e["kind"] for e in parsed["events"]]
    assert kinds == ["deposit", "buy", "buy", "sell", "dividend", "interest", "withdrawal"]
    plan = [e for e in parsed["events"] if e["kind"] == "buy" and e["savings_plan"]]
    assert len(plan) == 1 and plan[0]["asset_class"] == "FUND"
    ledger = build_ledger(parsed["events"])
    # 10000 - 1001 - 200 + 589 + 4 + 15 - 300
    assert ledger["cash"] == pytest.approx(9107.0)
    held = {p["isin"]: p["shares"] for p in ledger["positions"]}
    assert held == {"US0000000001": 5.0, "IE0000000006": 2.0}
    assert ledger["summary"]["realised_total"]["fifo"] == pytest.approx(100.0)  # 5 x (120 - 100)


def test_signs_are_normalised_and_bad_rows_are_explained():
    rows = "date;type;isin;name;shares;price;amount;fee;tax;currency\n" \
           "2024-01-02;deposit;;;;;10000;;;EUR\n" \
           "2024-01-03;buy;US0000000001;Example;10;100;1000;1;;EUR\n" \
           "2024-01-04;sell;US0000000001;Example;2;110;250;;;EUR\n" \
           "2024-01-05;party;;;;;5;;;EUR\n" \
           "2024-01-06;fee;;;;;3;;;EUR\n" \
           "2024-01-07;tax;;;;;;;2.5;EUR\n"
    parsed = gl.parse_export(rows.encode())
    buy = next(e for e in parsed["events"] if e["kind"] == "buy")
    assert buy["amount"] == -1000.0 and buy["fee"] == -1.0 and buy["shares"] == 10.0
    assert any("does not equal" in e for e in parsed["errors"]) and any("unknown type" in e for e in parsed["errors"])
    fee = next(e for e in parsed["events"] if e["kind"] == "fee")
    assert fee["amount"] == 0.0 and fee["fee"] == -3.0
    tax = next(e for e in parsed["events"] if e["kind"] == "tax")
    assert tax["tax"] == 2.5  # a refund keeps its sign
    assert not all(c["ok"] for c in parsed["checks"])
    with pytest.raises(ExportFormatError):
        gl.parse_export(b"date,amount\n2024-01-01,5\n")
    with pytest.raises(ExportFormatError):
        gl.parse_export(b"date,type,amount,currency\n2024-01-01,deposit,5,USD\n")
    with pytest.raises(ExportFormatError):
        gl.parse_export(b"")


def test_a_second_broker_replaces_the_first_brokers_positions_and_keeps_manual_ones():
    pf.save_portfolio({"cash": 0.0, "currency": "EUR", "positions": [
        {"ticker": "AAA", "shares": 1, "avg_cost": 10, "source": "trade_republic"},
        {"ticker": "BBB", "shares": 1, "avg_cost": 10, "source": "manual"},
    ]})
    stored, _ = pf.replace_imported([{"ticker": "CCC", "shares": 2, "avg_cost": 5}], 100.0, "EUR", "generic")
    assert {(p["ticker"], p["source"]) for p in stored["positions"]} == {("BBB", "manual"), ("CCC", "generic")}
    with pytest.raises(pf.PortfolioError):
        pf.replace_imported([], 0.0, "EUR", "manual")


def test_a_typed_holding_replaces_an_imported_one_only_with_force():
    pf.save_portfolio({"cash": 0.0, "currency": "EUR", "positions": [
        {"ticker": "AAPL", "shares": 30, "avg_cost": 146.73, "source": "generic"}]})
    with pytest.raises(pf.PortfolioError, match="AAPL comes from your Any broker"):
        pf.upsert_position({"ticker": "AAPL", "shares": 1, "avg_cost": 100})
    assert pf.get_position("AAPL")["shares"] == 30 and pf.get_position("AAPL")["source"] == "generic"
    forced = pf.upsert_position({"ticker": "AAPL", "shares": 1, "avg_cost": 100}, force=True)
    assert [(p["shares"], p["source"]) for p in forced["positions"]] == [(1, "manual")]
    assert pf.upsert_position({"ticker": "AAPL", "shares": 2, "avg_cost": 100})["positions"][0]["shares"] == 2  # typed: free


def test_the_portfolio_endpoints_refuse_to_change_an_imported_holding():
    import web.app as web_app

    pf.save_portfolio({"cash": 0.0, "currency": "EUR", "positions": [
        {"ticker": "AAPL", "shares": 30, "avg_cost": 146.73, "source": "generic"}]})
    client = web_app.app.test_client()
    refused = client.post("/api/portfolio/position", json={"ticker": "AAPL", "shares": 1, "avg_cost": 100})
    assert refused.status_code == 400 and "import a newer export" in refused.get_json()["error"]
    stored = client.get("/api/portfolio").get_json()
    changed = {**stored, "positions": [{**stored["positions"][0], "shares": 1}]}
    assert client.put("/api/portfolio", json=changed).status_code == 400
    assert client.put("/api/portfolio", json={**stored, "positions": []}).status_code == 400
    assert client.put("/api/portfolio", json={**stored, "cash": 50}).status_code == 200  # what Save cash sends
    assert pf.get_position("AAPL")["shares"] == 30
    assert client.put("/api/portfolio", json={**changed, "force": True}).status_code == 200
    assert pf.get_position("AAPL")["shares"] == 1


def test_a_new_portfolio_is_in_eur_and_a_stored_one_keeps_its_currency():
    assert pf.load_portfolio()["currency"] == "EUR"
    pf._path().write_text('{"cash": 0, "currency": "USD", "positions": []}')
    assert pf.load_portfolio()["currency"] == "USD"
    pf._path().write_text('{"cash": 0, "positions": []}')  # files from before the currency field read as USD
    assert pf.load_portfolio()["currency"] == "USD"


def test_an_import_in_another_currency_names_the_typed_costs_to_check():
    pf.save_portfolio({"cash": 0.0, "currency": "USD", "positions": [
        {"ticker": "ZZZQX", "shares": 5, "avg_cost": 10}, {"ticker": "KEEP", "shares": 1, "avg_cost": 3}]})
    stored, notes = pf.replace_imported([{"ticker": "CCC", "shares": 2, "avg_cost": 5}], 0.0, "EUR", "generic")
    assert stored["currency"] == "EUR"
    assert notes == ["Check the costs of ZZZQX, KEEP: you typed them in USD and they now count in EUR. Edit them in Your holdings."]


def test_import_endpoint_takes_a_broker_and_serves_the_template(monkeypatch):
    import web.app as web_app
    from data import instruments as ins

    monkeypatch.setattr(ins, "resolve", lambda *a, **k: {})
    monkeypatch.setattr(ins, "load_map", lambda: {})
    client = web_app.app.test_client()
    listed = client.get("/api/brokers").get_json()
    assert [b["key"] for b in listed["brokers"]] == ["trade_republic", "generic"]
    tpl = client.get("/api/brokers/template")
    assert tpl.status_code == 200 and tpl.mimetype == "text/csv" and tpl.data.startswith(b"date,type,isin")
    bad = client.post("/api/portfolio/import", data={"file": (io.BytesIO(tpl.data), "t.csv"), "broker": "nope"},
                      content_type="multipart/form-data")
    assert bad.status_code == 400
    res = client.post("/api/portfolio/import", data={"file": (io.BytesIO(tpl.data), "t.csv"), "broker": "generic"},
                      content_type="multipart/form-data")
    job = res.get_json()["job"]
    for _ in range(200):
        j = client.get(f"/api/lab/jobs/{job}").get_json()
        if j["status"] != "running":
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    status = client.get("/api/portfolio/import").get_json()
    assert status["imported"] and status["source"] == "generic" and status["source_label"].startswith("Any broker")
    assert status["rows"] == 7 and status["summary"]["cash"] == pytest.approx(9107.0)
    assert broker_import.delete_import() == {"imported": False}


def test_import_errors_say_what_to_do():
    """Each failure carries its fix; a file of the other broker is named so the page can offer one click."""
    def msg(broker, data):
        with pytest.raises(ExportFormatError) as exc:
            brokers.parse(broker, data)
        return exc.value

    assert "Download the export from your broker again" in str(msg("generic", b"  \n"))
    assert "This is a PDF" in str(msg("trade_republic", b"%PDF-1.7 synthetic"))
    assert "date, type and amount" in str(msg("generic", b"foo,bar\n1,2\n"))
    unknown = str(msg("generic", b"date,type,amount\n2024-01-02,staking,5\n2024-01-03,crypto_reward,6\n"))
    assert "staking and crypto_reward are not types FastValue knows" in unknown and "transfer_out or perk" in unknown
    usd = str(msg("generic", b"date,type,amount,currency\n2024-01-02,deposit,5,USD\n"))
    assert "EUR accounts only for now" in usd and "USD" in usd
    wrong = msg("trade_republic", gl.template_csv().encode())
    assert wrong.other_broker == "generic" and "This is a FastValue CSV, not a Trade Republic export." == str(wrong)


def test_wrong_broker_error_reaches_the_page():
    import web.app as web_app

    client = web_app.app.test_client()
    res = client.post("/api/portfolio/import", data={"file": (io.BytesIO(gl.template_csv().encode()), "t.csv"), "broker": "trade_republic"},
                      content_type="multipart/form-data")
    body = res.get_json()
    assert res.status_code == 400 and body["other_broker"] == "generic" and body["other_broker_label"].startswith("Any broker")


def test_broker_labels_and_hints_for_the_page():
    by_key = {b["key"]: b for b in brokers.describe()}
    assert by_key["trade_republic"]["button"] == "Import Trade Republic export" and by_key["generic"]["button"] == "Import FastValue CSV"
    for b in by_key.values():
        assert "<from>" not in b["file"] + b["hint"] and "below" not in b["hint"] and "dropped" not in b["hint"]
