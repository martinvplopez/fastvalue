"""Trade Republic import: parsing, privacy, ledger accounting, import flow and the track record's math.

Everything here runs on a small synthetic export built in code (no personal data, no network). The check against
the owner's real export lives in test_trade_republic_real.py and only runs when that file is pointed to.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from core import ledger as ledger_mod
from core import track_record as trk
from data import trade_republic as tr

COLUMNS = ["datetime", "date", "account_type", "category", "type", "asset_class", "name", "symbol", "shares", "price",
           "amount", "fee", "tax", "currency", "original_amount", "original_currency", "fx_rate", "description",
           "transaction_id", "counterparty_name", "counterparty_iban", "payment_reference", "mcc_code"]
PERSONAL = ("DOE JANE", "Jane Doe", "DE89370400440532013000", "NL91ABNA0417164300", "Hotel Example", "COBADEFF")

STOCK, NEW_ISIN, OLD_ISIN, RIGHTS = "US0000000001", "CA0000000002", "CA0000000003", "ES0000000004"
PARENT, ETF, TURBO, PRIV, MOVED = "ES0000000005", "IE0000000006", "DE0000000007", "LU0000000008", "US0000000009"


def row(n: int, ts: str, category: str, type_: str, **kw: object) -> dict[str, str]:
    r = {c: "" for c in COLUMNS}
    r.update({"datetime": ts, "date": ts[:10], "account_type": "DEFAULT", "category": category, "type": type_,
              "currency": "EUR", "transaction_id": f"00000000-0000-0000-0000-{n:012d}"})
    r.update({k: str(v) for k, v in kw.items()})
    return r


def sample_rows() -> list[dict[str, str]]:
    rows = [
        row(1, "2024-01-02T10:00:00.000000Z", "CASH", "CUSTOMER_INBOUND", name="DOE JANE", amount="10000.00",
            description="COBADEFF", counterparty_name="DOE JANE", counterparty_iban="DE89370400440532013000"),
        row(2, "2024-01-03T09:00:00.000Z", "TRADING", "BUY", asset_class="STOCK", name="Example Corp", symbol=STOCK,
            shares="10", price="100", amount="-1000.00", fee="-1.00", description=f"Buy trade {STOCK} EXAMPLE, quantity: 10"),
        row(3, "2024-01-10T09:00:00.000Z", "TRADING", "BUY", asset_class="STOCK", name="Example Corp", symbol=STOCK,
            shares="1", price="50", amount="-50.00", description=f"Savings plan execution {STOCK} EXAMPLE, quantity: 1"),
        row(4, "2024-01-15T09:00:00.000Z", "TRADING", "BUY", asset_class="FUND", name="World ETF", symbol=ETF,
            shares="2", price="100", amount="-200.00", description=f"Savings plan execution {ETF} WORLD, quantity: 2"),
        # A split sale: whole shares carry the fee, the fraction does not; same order.
        row(5, "2024-03-01T10:00:00.000Z", "TRADING", "SELL", asset_class="STOCK", name="Example Corp", symbol=STOCK,
            shares="-5", price="120", amount="600.00", fee="-1.00", tax="-10.00", description=f"Sell trade {STOCK} EXAMPLE, quantity: 5"),
        row(6, "2024-03-01T10:00:00.002Z", "TRADING", "SELL", asset_class="STOCK", name="Example Corp", symbol=STOCK,
            shares="-0.5", price="120", amount="60.00", description=f"Sell trade {STOCK} EXAMPLE, quantity: 0.5"),
        row(7, "2024-03-05T08:00:00.000Z", "CASH", "DIVIDEND", asset_class="STOCK", name="Example Corp", symbol=STOCK,
            shares="5.5", amount="5.00", tax="-1.00", original_amount="5.4", original_currency="USD", fx_rate="0.92"),
        row(8, "2024-03-05T09:00:00.000Z", "CASH", "DIVIDEND", asset_class="STOCK", name="Example Corp", symbol=STOCK,
            shares="5.5", amount="-5.00", tax="1.00"),  # reversal
        row(9, "2024-03-05T10:00:00.000Z", "CASH", "DIVIDEND", asset_class="STOCK", name="Example Corp", symbol=STOCK,
            shares="5.5", amount="5.00", tax="-1.00"),
        row(10, "2024-04-01T01:00:00.000Z", "CASH", "INTEREST_PAYMENT", amount="20.00", tax="-5.00",
            description="Interest payment Booking"),
        # Stock perk followed by the matching buy.
        row(11, "2024-04-02T12:00:00.000Z", "CASH", "STOCKPERK", asset_class="STOCK", name="Perk Co", symbol=MOVED,
            amount="10.00", description="Stockperk"),
        row(12, "2024-04-02T12:00:01.000Z", "TRADING", "BUY", asset_class="STOCK", name="Perk Co", symbol=MOVED,
            shares="0.1", price="100", amount="-10.00", description=f"Buy trade {MOVED} PERK, quantity: 0.1"),
        row(13, "2024-04-03T12:00:00.000Z", "TRADING", "BUY", asset_class="STOCK", name="Perk Co", symbol=MOVED,
            shares="1.9", price="100", amount="-190.00", fee="-1.00", description=f"Buy trade {MOVED} PERK, quantity: 1.9"),
        # ISIN change: bought under the old ISIN, sold under the new one.
        row(14, "2024-04-10T09:00:00.000Z", "TRADING", "BUY", asset_class="STOCK", name="Old Mining", symbol=OLD_ISIN,
            shares="4", price="25", amount="-100.00", fee="-1.00", description=f"Buy trade {OLD_ISIN} OLD, quantity: 4"),
        row(15, "2024-05-01T10:00:00.000Z", "CORPORATE_ACTION", "EXCHANGE", asset_class="STOCK", name="New Mining",
            symbol=NEW_ISIN, shares="4", currency=""),
        row(16, "2024-05-01T10:00:00.000Z", "CORPORATE_ACTION", "EXCHANGE", asset_class="STOCK", name="Old Mining",
            symbol=OLD_ISIN, shares="-4", currency=""),
        # Broker migration: a +/- pair carrying the broker's cost per share (FIFO with fees and taxes).
        row(17, "2024-05-02T03:00:00.000Z", "DELIVERY", "MIGRATION", asset_class="STOCK", name="New Mining",
            symbol=NEW_ISIN, shares="-4", price="25.25"),
        row(18, "2024-05-02T03:00:00.001Z", "DELIVERY", "MIGRATION", asset_class="STOCK", name="New Mining",
            symbol=NEW_ISIN, shares="4", price="25.25"),
        row(19, "2024-06-01T10:00:00.000Z", "TRADING", "SELL", asset_class="STOCK", name="New Mining", symbol=NEW_ISIN,
            shares="-4", price="30", amount="120.00", fee="-1.00", description=f"Sell trade {NEW_ISIN} NEW, quantity: 4"),
        # Rights: a temporary ISIN credited, paid out, removed; the cash belongs to the parent stock.
        row(20, "2024-06-03T09:00:00.000Z", "TRADING", "BUY", asset_class="STOCK", name="Parent", symbol=PARENT,
            shares="3", price="40", amount="-120.00", fee="-1.00", description=f"Buy trade {PARENT} PARENT SA, quantity: 3"),
        row(21, "2024-06-10T09:00:00.000Z", "CORPORATE_ACTION", "INTERMEDIATE_SECURITIES_DISTRIBUTION",
            asset_class="SYNTHETIC", name="PARENT SA -ANR-", symbol=RIGHTS, shares="3", currency=""),
        row(22, "2024-06-20T09:00:00.000Z", "CORPORATE_ACTION", "LIQUIDATION_DIVIDEND", asset_class="SYNTHETIC",
            name="PARENT SA -ANR-", symbol=RIGHTS, shares="-3", currency=""),
        row(23, "2024-06-21T09:00:00.000Z", "CASH", "LIQUIDATION_PROCEEDS", asset_class="SYNTHETIC",
            name="PARENT SA -ANR-", symbol=RIGHTS, amount="1.50"),
        # Shares moved to another broker.
        row(24, "2024-07-01T09:00:00.000Z", "DELIVERY", "FREE_RECEIPT", asset_class="STOCK", name="Perk Co",
            symbol=MOVED, shares="-2"),
        # Private markets: cash legs, bonus, and the trading row with the shares but no amount.
        row(25, "2024-07-02T09:00:00.000Z", "CASH", "PRIVATE_MARKET_BUY", asset_class="PRIVATE_FUND",
            name="Private Equity", symbol=PRIV, amount="-500.00", fee="-1.00", description="Private Markets pre-payment"),
        row(26, "2024-07-02T09:00:01.000Z", "CASH", "BONUS", amount="5.00",
            description="Private Markets bonus for buy order"),
        row(27, "2024-07-20T09:00:00.000Z", "TRADING", "BUY", asset_class="PRIVATE_FUND", name="Private Equity",
            symbol=PRIV, shares="5", price="100", description=f"BUY {PRIV}"),
        # Turbo round trip.
        row(28, "2024-08-01T09:00:00.000Z", "TRADING", "BUY", asset_class="DERIVATIVE", name="Long 120,5000 $",
            symbol=TURBO, shares="100", price="1.5", amount="-150.00", fee="-1.00",
            description=f"Buy trade {TURBO} Best Turbo auf EXAMPLE INC., quantity: 100"),
        row(29, "2024-08-20T09:00:00.000Z", "TRADING", "SELL", asset_class="DERIVATIVE", name="Long 120,5000 $",
            symbol=TURBO, shares="-100", price="0.5", amount="50.00", fee="-1.00",
            description=f"Sell trade {TURBO} Best Turbo auf EXAMPLE INC., quantity: 100"),
        # Tax corrections keep their exported signs.
        row(30, "2024-09-01T09:00:00.000Z", "CASH", "MANUAL", amount="0.000000", tax="0.95",
            description="Capital gains tax"),
        row(31, "2024-09-02T09:00:00.000Z", "CASH", "TAX_OPTIMIZATION", amount="0.000000", tax="-2.00",
            description="Tax Recalculation"),
        # Money out: a transfer (the description repeats a name and IBAN) and card spending.
        row(32, "2024-09-10T09:00:00.000Z", "CASH", "TRANSFER_INSTANT_OUTBOUND", amount="-300.00",
            description="Outgoing transfer for Jane Doe (NL91ABNA0417164300)"),
        row(33, "2024-09-11T09:00:00.000Z", "CASH", "CARD_TRANSACTION", name="Hotel Example", amount="-80.00",
            description="Hotel Example", mcc_code="4722"),
        # Late-evening UTC booking: the date differs from the timestamp's day.
        row(34, "2024-09-12T23:30:00.000Z", "CASH", "TRANSFER_INSTANT_INBOUND", amount="1000.00",
            description="Incoming transfer from Jane Doe (DE89370400440532013000)"),
    ]
    rows[-1]["date"] = "2024-09-13"
    return rows


def to_csv(rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


@pytest.fixture()
def parsed() -> dict:
    return tr.parse_export(to_csv(sample_rows()))


@pytest.fixture()
def built(parsed) -> dict:
    return ledger_mod.build_ledger(parsed["events"])


# ------------------------------------------------------------------ parsing and privacy


def test_parse_counts_checks_and_aliases(parsed):
    assert parsed["rows"] == 34
    assert parsed["type_counts"]["BUY"] == 9 and parsed["type_counts"]["SELL"] == 4
    assert all(c["ok"] for c in parsed["checks"]), parsed["checks"]
    assert parsed["aliases"] == {OLD_ISIN: NEW_ISIN, RIGHTS: PARENT}
    assert parsed["date_differs_from_timestamp"] == 1
    assert not parsed["warnings"] and not parsed["errors"]


def test_personal_fields_never_reach_the_events(parsed):
    blob = json.dumps(parsed)
    for text in PERSONAL:
        assert text not in blob
    assert "counterparty" not in blob and "mcc" not in blob and "original_amount" not in blob
    flows = [e for e in parsed["events"] if e["kind"] in ("deposit", "withdrawal", "card")]
    assert flows and all(e["name"] == "" for e in flows)


def test_split_sale_is_one_order_and_savings_plans_are_flagged(parsed, built):
    sells = [o for o in built["orders"] if o["side"] == "sell" and o["isin"] == STOCK]
    assert len(sells) == 1 and sells[0]["rows"] == 2 and sells[0]["shares"] == pytest.approx(5.5)
    assert sells[0]["fee"] == -1.0 and sells[0]["price"] == pytest.approx(120.0)
    plans = {e["isin"] for e in parsed["events"] if e.get("savings_plan")}
    assert plans == {STOCK, ETF}


def test_turbo_details_come_from_name_and_description(built):
    info = built["instruments"][TURBO]["turbo"]
    assert info == {"direction": "long", "strike": pytest.approx(120.5), "strike_currency": "$",
                    "underlying": "EXAMPLE INC."}


def test_bad_files_are_refused_before_anything_is_stored():
    with pytest.raises(tr.ExportFormatError):
        tr.parse_export(b"")
    with pytest.raises(tr.ExportFormatError):
        tr.parse_export(b'"a","b"\n"1","2"\n')
    assert tr.load_ledger() is None


def test_duplicate_rows_are_dropped_and_flagged():
    rows = sample_rows()
    p = tr.parse_export(to_csv(rows + [dict(rows[1])]))
    assert p["rows"] == 34
    dup = next(c for c in p["checks"] if c["name"] == "Unique transaction ids")
    assert not dup["ok"]


def test_unknown_row_types_count_in_cash_and_are_reported():
    rows = sample_rows() + [row(99, "2024-10-01T09:00:00.000Z", "CASH", "SOMETHING_NEW", amount="3.00")]
    p = tr.parse_export(to_csv(rows))
    assert any("SOMETHING_NEW" in w for w in p["warnings"])
    assert ledger_mod.build_ledger(p["events"])["cash"] == pytest.approx(_cash(rows))


# ------------------------------------------------------------------ ledger


def _cash(rows: list[dict[str, str]]) -> float:
    num = lambda v: float(v) if v else 0.0  # noqa: E731
    return round(sum(num(r["amount"]) + num(r["fee"]) + num(r["tax"]) for r in rows), 6)


def test_cash_reconciles_with_every_row(built):
    assert built["cash"] == pytest.approx(_cash(sample_rows()))
    assert built["summary"]["cash"] == pytest.approx(round(_cash(sample_rows()), 2))


def test_realised_pnl_average_cost_and_fifo(built):
    # Example Corp: 10 @ 100 and 1 @ 50 (average 95.4545...), 5.5 sold @ 120.
    avg = 1050.0 / 11.0
    expected_avg = 660.0 - avg * 5.5
    expected_fifo = 660.0 - 5.5 * 100.0
    r = [x for x in built["realisations"] if x["isin"] == STOCK]
    assert sum(x["pnl_avg"] for x in r) == pytest.approx(expected_avg)
    assert sum(x["pnl_fifo"] for x in r) == pytest.approx(expected_fifo)
    # The ISIN change carries cost and history: 4 @ 25 sold @ 30 under the new ISIN.
    assert sum(x["pnl_avg"] for x in built["realisations"] if x["isin"] == NEW_ISIN) == pytest.approx(20.0)
    year = built["summary"]["realised_by_year"]["2024"]
    assert year["average"] == pytest.approx(expected_avg + 20.0 + (50.0 - 150.0), abs=0.01)
    assert year["fifo"] == pytest.approx(expected_fifo + 20.0 + (50.0 - 150.0), abs=0.01)


def test_broker_cost_check_and_position_cost_with_fees(built):
    mig = built["migration_check"]
    assert len(mig) == 1 and abs(mig[0]["diff_pct"]) < 1e-6  # (100 + 1 fee) / 4 = 25.25
    pos = {p["isin"]: p for p in built["positions"]}
    # Example Corp: 5.5 left; FIFO keeps 4.5 of the first lot and the 1-share savings plan.
    assert pos[STOCK]["shares"] == pytest.approx(5.5)
    assert pos[STOCK]["cost_fifo"] == pytest.approx(4.5 * 100 + 50)
    assert pos[STOCK]["cost_broker"] == pytest.approx(4.5 * 100 + 50 + 1.0 * 4.5 / 10)
    assert pos[STOCK]["opened"] == "2024-01-03"


def test_income_perks_rights_and_reversals(built):
    s = built["summary"]
    assert s["dividends_gross"] == pytest.approx(5.0)  # booked, reversed, booked again
    assert s["dividends_tax"] == pytest.approx(-1.0)
    assert s["liquidation_proceeds"] == pytest.approx(1.5)
    assert s["interest_gross"] == pytest.approx(20.0) and s["interest_tax"] == pytest.approx(-5.0)
    assert s["perks"] == pytest.approx(10.0)
    parent = next(ep for ep in built["episodes"] if ep["isin"] == PARENT)
    assert parent["dividends"] == pytest.approx(1.5)  # the rights' cash belongs to the parent stock
    perk = next(ep for ep in built["episodes"] if ep["isin"] == MOVED)
    assert perk["bought"] == pytest.approx(200.0)  # the perk is income, the buy keeps its full cost


def test_transfers_private_funds_and_external_flows(built):
    s = built["summary"]
    assert s["securities_moved_out_at_cost"] == pytest.approx(200.0)
    moved = next(ep for ep in built["episodes"] if ep["isin"] == MOVED)
    assert moved["end_reason"] == "transferred" and moved["realised_avg"] == 0.0
    assert s["private_funds_moved_out"] == pytest.approx(496.0)  # -500 - 1 fee + 5 bonus
    assert PRIV not in {p["isin"] for p in built["positions"]}
    assert s["deposits"] == pytest.approx(11000.0) and s["withdrawals"] == pytest.approx(-300.0)
    assert s["card_spending"] == pytest.approx(-80.0) and s["net_external_flows"] == pytest.approx(10620.0)
    assert s["taxes_by_type"]["MANUAL"] == pytest.approx(0.95) and s["taxes_by_type"]["TAX_OPTIMIZATION"] == -2.0
    assert s["fees"] == pytest.approx(-9.0)


def test_round_trips(built):
    closed = Counter(ep["end_reason"] for ep in built["episodes"])
    assert closed == Counter({"sold": 2, "transferred": 1, "open": 3})
    turbo = next(ep for ep in built["episodes"] if ep["isin"] == TURBO)
    assert turbo["realised_avg"] == pytest.approx(-100.0) and turbo["fees"] == pytest.approx(-2.0)


def test_merge_keeps_one_copy_per_transaction(parsed):
    events = parsed["events"]
    merged = tr.merge_events(events[:20], events[10:])
    assert len(merged) == len(events)
    # Time order, not text order: "10:00:00.002Z" sorts before "10:00:00Z" as text.
    assert [e["id"] for e in merged] == sorted([e["id"] for e in merged],
                                               key=lambda i: next(tr.event_time(e["ts"]) for e in events if e["id"] == i))
    assert [e["id"][-2:] for e in merged[4:6]] == ["05", "06"]


def test_isin_change_with_a_ratio_rescales_the_holding():
    # Buy 4 OLD at 25, exchange 4 OLD for 2 NEW (1:2 reverse split), sell 2 NEW at 60: P&L +20, nothing left open.
    def row(i, ts, typ, isin, shares, price, amount, category="TRADING", asset="STOCK"):
        return {"datetime": ts, "date": ts[:10], "account_type": "DEFAULT", "category": category, "type": typ,
                "asset_class": asset, "name": "Example", "symbol": isin, "shares": shares, "price": price,
                "amount": amount, "fee": "", "tax": "", "currency": "EUR", "original_amount": "",
                "original_currency": "", "fx_rate": "", "description": "", "transaction_id": f"id-{i}",
                "counterparty_name": "", "counterparty_iban": "", "payment_reference": "", "mcc_code": ""}
    rows = [
        row(1, "2024-01-02T10:00:00Z", "CUSTOMER_INBOUND", "", "", "", "100", category="CASH", asset=""),
        row(2, "2024-01-03T10:00:00Z", "BUY", "DE000000OLD1", "4", "25", "-100"),
        row(3, "2024-02-01T10:00:00Z", "EXCHANGE", "DE000000OLD1", "-4", "", "0", category="CORPORATE_ACTION"),
        row(4, "2024-02-01T10:00:00Z", "EXCHANGE", "DE000000NEW8", "2", "", "0", category="CORPORATE_ACTION"),
        row(5, "2024-03-01T10:00:00Z", "SELL", "DE000000NEW8", "-2", "60", "120"),
    ]
    header = list(rows[0])
    text = ",".join(header) + "\n" + "\n".join(",".join(str(r[c]) for c in header) for r in rows) + "\n"
    p = tr.parse_export(text)
    assert p["aliases"] == {"DE000000OLD1": "DE000000NEW8"}
    led = ledger_mod.build_ledger(p["events"], method="fifo")
    assert not led["positions"]
    assert led["summary"]["realised_by_year"]["2024"]["fifo"] == pytest.approx(20.0)


# ------------------------------------------------------------------ storage and import into the portfolio


def test_save_load_merge_and_delete(parsed):
    stored = tr.save_ledger(parsed)
    assert stored["rows"] == 34 and tr.load_ledger()["events"] == stored["events"]
    assert "report" in stored and stored["report"]["type_counts"]["BUY"] == 9
    assert tr.save_ledger(parsed, merge=True)["rows"] == 34
    assert tr.delete_ledger() and tr.load_ledger() is None


def test_import_replaces_imported_positions_and_keeps_manual_ones(monkeypatch, parsed):
    from core import broker_import
    from data import instruments as ins
    from data.portfolio import load_portfolio, upsert_position

    upsert_position({"ticker": "KEEP", "shares": 1, "avg_cost": 10})
    mapping = {STOCK: {"isin": STOCK, "ticker": "EXA", "status": "ok", "source": "search", "median_gap": 0.0},
               ETF: {"isin": ETF, "ticker": "WRLD.L", "status": "mismatch", "source": "search", "median_gap": 0.5},
               PARENT: {"isin": PARENT, "ticker": "PAR.MC", "status": "ok", "source": "search", "median_gap": 0.0}}
    monkeypatch.setattr(ins, "resolve", lambda *a, **k: mapping)
    monkeypatch.setattr(ins, "load_map", lambda: mapping)
    report = broker_import.import_trade_republic(parsed=parsed)
    pf = load_portfolio()
    assert pf["currency"] == "EUR" and pf["cash"] == pytest.approx(round(_cash(sample_rows()), 2))
    tickers = {p["ticker"]: p for p in pf["positions"]}
    assert set(tickers) == {"KEEP", "EXA", "PAR.MC"}  # the ETF's listing failed the price check: left out
    assert tickers["EXA"]["source"] == "trade_republic" and tickers["EXA"]["isin"] == STOCK
    assert tickers["EXA"]["avg_cost"] == pytest.approx((450 + 50 + 0.45) / 5.5, rel=1e-6)
    assert any("no price source" in n for n in report["notes"])
    assert any(c["name"].startswith("Cost per share") and c["ok"] for c in report["checks"])
    etf_row = next(r for r in report["instruments"] if r["isin"] == ETF)
    assert etf_row["usable"] is False
    assert broker_import.delete_import() == {"imported": False}
    assert {p["ticker"] for p in load_portfolio()["positions"]} == {"KEEP"}


# ------------------------------------------------------------------ prices, FX and the ticker check


def test_fx_minor_units_and_cross_rates(monkeypatch):
    from data import fx

    rates = {"USD": pd.Series([1.10, 1.20], index=pd.to_datetime(["2024-01-02", "2024-01-03"])),
             "GBP": pd.Series([0.80, 0.80], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))}
    monkeypatch.setattr(fx, "per_eur", lambda c: rates[fx.major(c)[0]])
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-06"])
    usd = fx.factor_series("USD", "EUR", days)
    assert usd.tolist() == pytest.approx([1 / 1.10, 1 / 1.20, 1 / 1.20])  # weekend carries the last rate
    pence = fx.factor_series("GBp", "EUR", days)
    assert pence.iloc[0] == pytest.approx(0.01 / 0.80)
    assert fx.convert_latest(100.0, "USD", "GBP") == pytest.approx(100.0 / 1.20 * 0.80)
    assert fx.major("GBX") == ("GBP", 0.01) and fx.quote_currency("AAPL") == "USD"


def test_ticker_check_uses_the_daily_range_in_eur(monkeypatch):
    from data import fx
    from data import instruments as ins

    monkeypatch.setattr(fx, "quote_currency", lambda t: "USD")
    monkeypatch.setattr(fx, "factor_series", lambda f, t, idx: pd.Series(0.5, index=idx))  # 1 USD = 0.5 EUR
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "adj_close": 100.0,
                       "volume": 1.0, "dividends": 0.0, "splits": 0.0}, index=idx)
    good = ins.check_ticker("X", [("2024-01-03", 50.0), ("2024-01-05", 51.5)], df)
    assert good["status"] == "ok" and good["median_gap"] == 0.0
    pence = ins.check_ticker("X", [("2024-01-03", 0.5)], df)  # a price 100 times off: the wrong unit or listing
    assert pence["status"] == "mismatch"


# ------------------------------------------------------------------ track record math


def test_xirr_and_modified_dietz():
    t0 = pd.Timestamp("2024-01-01")
    assert trk._xirr([(t0, -100.0), (t0 + pd.Timedelta(days=365), 110.0)]) == pytest.approx(0.1, abs=2e-3)
    assert trk._xirr([(t0, 100.0)]) is None
    flows = pd.Series([100.0, 0.0], index=pd.to_datetime(["2024-01-01", "2024-12-31"]))
    assert trk._modified_dietz(0.0, 110.0, flows) == pytest.approx(0.1, abs=1e-3)  # 2024 has 366 days


def test_confidence_needs_count_t_and_both_halves():
    assert trk._confidence(5, 3.0, True)[0] == "low"
    assert trk._confidence(40, 2.5, True)[0] == "high"
    assert trk._confidence(40, 2.5, False)[0] == "low"
    assert trk._confidence(20, 1.7, True)[0] == "medium"
    assert trk._confidence(40, 0.4, True)[0] == "low"


def test_clusters_count_one_name_month_once():
    rows = [{"isin": "A", "date": f"2024-01-{d:02d}", "v": 0.1} for d in (1, 8, 15, 22)]
    rows += [{"isin": "B", "date": "2024-01-03", "v": -0.1}]
    cl = trk._clusters(rows, "v")
    assert len(cl) == 2
    p = trk._pattern(rows, "v", pd.Timestamp("2024-06-01"))
    assert p["n"] == 5 and p["n_eff"] == 2


def test_all_in_benchmark_follows_the_same_flows():
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    m = trk._Market(days, days[-1], pd.DataFrame(index=days), {}, pd.Series([10.0, 20.0, 20.0], index=days), "b", {}, [])
    value, notes = trk._all_in(m, pd.Series([100.0, 0.0, -100.0], index=days))
    assert value.tolist() == pytest.approx([100.0, 200.0, 100.0]) and not notes
    value, notes = trk._all_in(m, pd.Series([100.0, -500.0, 0.0], index=days))
    assert value.iloc[-1] == 0.0 and notes  # emptied, never shorted


def test_exit_rules_fire_on_the_first_close_past_the_level():
    days = pd.date_range("2024-01-01", periods=6, freq="D")
    path = pd.Series([95.0, 80.0, 70.0, 160.0, 210.0], index=days[1:])
    m = trk._Market(days, days[-1], pd.DataFrame({"A": [100.0, *path.tolist()]}, index=days), {}, pd.Series(1.0, index=days), "b", {}, [])
    lot = {"isin": "A", "date": "2024-01-01", "entry": 100.0, "actual": 210.0, "exit_day": days[-1], "exit": "open",
           "path": path, "shares": 1.0}
    assert trk._apply_rule(lot, m, 0.15, None, None) == 80.0
    assert trk._apply_rule(lot, m, None, 0.5, None) == 160.0
    assert trk._apply_rule(lot, m, 0.35, 0.5, None) == 160.0  # whichever comes first
    assert trk._apply_rule(lot, m, 0.5, 1.5, None) == 210.0  # never fires: the actual exit
    assert np.isfinite(trk._tstat([1.0, 2.0, 3.0]))


# ------------------------------------------------------------------ web API


def test_import_endpoints_and_sec_contact(monkeypatch):
    import time

    import web.app as web_app
    from data import instruments as ins
    from data.sec_edgar import user_agent

    monkeypatch.setattr(ins, "resolve", lambda *a, **k: {})
    monkeypatch.setattr(ins, "load_map", lambda: {})
    client = web_app.app.test_client()
    assert client.get("/api/portfolio/import").get_json() == {"imported": False}
    bad = client.post("/api/portfolio/import", data={"file": (io.BytesIO(b"x,y\n1,2\n"), "t.csv")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400 and "missing columns" in bad.get_json()["error"]
    res = client.post("/api/portfolio/import", data={"file": (io.BytesIO(to_csv(sample_rows())), "t.csv")},
                      content_type="multipart/form-data")
    job = res.get_json()["job"]
    for _ in range(100):
        j = client.get(f"/api/lab/jobs/{job}").get_json()
        if j["status"] != "running":
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    status = client.get("/api/portfolio/import").get_json()
    assert status["imported"] and status["rows"] == 34
    for text in PERSONAL:
        assert text not in json.dumps(status)
    assert client.put(f"/api/portfolio/import/instrument/{'X' * 3}", json={"ticker": "A"}).status_code == 400

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert client.get("/api/settings/sec").get_json() == {"configured": False, "source": None}
    assert client.put("/api/settings/sec", json={"email": "not-an-email"}).status_code == 400
    ok = client.put("/api/settings/sec", json={"email": "someone@example.com"}).get_json()
    assert ok == {"configured": True, "source": "saved"}
    assert user_agent() == "fastvalue someone@example.com"
    assert client.put("/api/settings/sec", json={"email": ""}).get_json()["configured"] is False


def test_rows_without_isin_never_keep_a_name_and_bad_encodings_are_explained():
    rows = sample_rows()
    extra = dict(rows[-1], category="CARD", type="CARD_REFUND", asset_class="", symbol="", shares="", price="",
                 name="Hotel Example", description="Refund Hotel Example", transaction_id="refund-1", amount="12.5")
    p = tr.parse_export(to_csv(rows + [extra]))
    refund = next(e for e in p["events"] if e["id"] == "refund-1")
    assert refund["name"] == ""
    with pytest.raises(tr.ExportFormatError, match="UTF-8"):
        tr.parse_export(to_csv(rows).replace(b"EUR", b"\xe9UR", 1))


def test_merge_moves_older_buys_onto_an_isin_change_found_later():
    old = [{"id": "b1", "ts": "2024-01-03T10:00:00Z", "isin": "DE000000OLD1", "raw_isin": None}]
    new = [{"id": "x1", "ts": "2024-02-01T10:00:00Z", "isin": "DE000000NEW8", "raw_isin": "DE000000OLD1"}]
    merged = tr.merge_events(old, new, {"DE000000OLD1": "DE000000NEW8"})
    assert [(e["id"], e["isin"], e["raw_isin"]) for e in merged] == [
        ("b1", "DE000000NEW8", "DE000000OLD1"), ("x1", "DE000000NEW8", "DE000000OLD1")]
