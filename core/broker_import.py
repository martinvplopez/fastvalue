"""Import a broker export into the Portfolio tab: ledger, ticker map and positions in one step.

``import_export`` parses and sanitizes the file with the broker's parser
(``data.brokers``: Trade Republic natively, any other broker through the
documented CSV of ``data.generic_ledger``), stores the cleaned events, replays
them (``core.ledger``), maps each open holding's ISIN to a priced ticker
(``data.instruments``) and replaces the imported positions and the cash in the
portfolio store, keeping the positions added by hand. The portfolio currency
becomes EUR, the account's currency.

A position's average cost is what the broker records: FIFO lots with their buy
fees and taxes (the export's own cost figures match it to 0.001%). Holdings
with no usable ticker stay out of the portfolio file and are listed so a ticker can
be set by hand; the review counts them at cost (``unpriced_holdings``), private
funds and temporary rights excepted.
"""

from __future__ import annotations

from typing import Any, Callable

from core.ledger import build_ledger
from data import brokers
from data import instruments as ins
from data import trade_republic as tr
from data.portfolio import last_review_path, load_portfolio, replace_imported, save_portfolio

Progress = Callable[[int, int, str], None]
CURRENCY = "EUR"
MIGRATION_TOLERANCE = 0.001


def _n(count: int, one: str, many: str) -> str:
    """The count with the right noun: "1 case", "2 cases"."""
    return f"{count} {one if count == 1 else many}"


def _ledger_checks(ledger: dict[str, Any], mapping: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    mig = [m for m in ledger["migration_check"] if m.get("diff_pct") is not None]
    if mig:
        bad = [m for m in mig if abs(m["diff_pct"]) > MIGRATION_TOLERANCE]
        checks.append({"name": "Cost per share matches the broker's records", "ok": not bad,
                       "detail": (f"{len(mig)} of {len(mig)} within {MIGRATION_TOLERANCE:.1%}" if not bad
                                  else f"{len(bad)} of {len(mig)} differ by more than {MIGRATION_TOLERANCE:.1%}")})
    checks.append({"name": "Cash is not negative", "ok": ledger["cash"] >= -0.005,
                   "detail": "the replayed cash never ends below zero" if ledger["cash"] >= -0.005
                   else "the replayed cash ends below zero: rows may be missing"})
    held = [p for p in ledger["positions"]]
    priced = [p for p in held if ins.usable(mapping.get(p["isin"]))]
    checks.append({"name": "Open holdings with a price source", "ok": len(priced) == len(held),
                   "detail": f"{len(priced)} of {len(held)}"})
    if ledger["warnings"]:
        checks.append({"name": "Ledger replay", "ok": False, "detail": "; ".join(ledger["warnings"][:3])})
    return checks


def _positions(ledger: dict[str, Any], mapping: dict[str, dict[str, Any]], source: str = tr.SOURCE) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    notes: list[str] = []
    seen: dict[str, str] = {}
    for p in ledger["positions"]:
        entry = mapping.get(p["isin"])
        if not ins.usable(entry):
            continue
        ticker = entry["ticker"]
        if ticker in seen:
            notes.append(f"{p['isin']} maps to {ticker}, already used by {seen[ticker]}: set another ticker for it.")
            continue
        seen[ticker] = p["isin"]
        out.append({"ticker": ticker, "shares": p["shares"], "avg_cost": round(p["avg_price_broker"], 6),
                    "opened": p["opened"], "isin": p["isin"], "name": p["name"], "source": source})
    return out, notes


def _instrument_rows(ledger: dict[str, Any], mapping: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Open holdings with their price source, problems first."""
    rows = []
    for p in ledger["positions"]:
        e = mapping.get(p["isin"]) or {}
        rows.append({"isin": p["isin"], "name": p["name"], "asset_class": p["asset_class"], "shares": p["shares"],
                     "cost": round(p["cost_broker"], 2), "ticker": e.get("ticker"), "currency": e.get("currency"),
                     "status": e.get("status", "unmapped"), "source": e.get("source"),
                     "median_gap": e.get("median_gap"), "usable": ins.usable(e), "reason": e.get("reason"),
                     "candidates": e.get("candidates", []),
                     "bond": ins.is_bond(p["asset_class"]), "note": ins.BOND_FOUND_NOTE if ins.is_bond(p["asset_class"]) else None})
    rows.sort(key=lambda r: (r["usable"], -r["cost"]))
    return rows


def _report(stored: dict[str, Any], ledger: dict[str, Any], mapping: dict[str, dict[str, Any]],
            notes: list[str] | None = None) -> dict[str, Any]:
    report = stored.get("report", {})
    return {
        "imported": True,
        "source": stored.get("source"),
        "source_label": brokers.label(stored.get("source")),
        "imported_at": stored.get("imported_at"),
        "rows": stored.get("rows"),
        "period": stored.get("period"),
        "type_counts": report.get("type_counts", {}),
        "checks": report.get("checks", []) + _ledger_checks(ledger, mapping),
        "warnings": report.get("warnings", []) + ledger["warnings"],
        "summary": ledger["summary"],
        "instruments": _instrument_rows(ledger, mapping),
        "currency": CURRENCY,
        "notes": notes or [],
    }


def _sync_portfolio(ledger: dict[str, Any], mapping: dict[str, dict[str, Any]], source: str = tr.SOURCE) -> list[str]:
    positions, notes = _positions(ledger, mapping, source)
    _, more = replace_imported(positions, round(max(ledger["cash"], 0.0), 2), CURRENCY, source)
    unpriced = [p for p in ledger["positions"] if not ins.usable(mapping.get(p["isin"]))]
    counted = [p for p in unpriced if p["asset_class"] not in ("PRIVATE_FUND", "SYNTHETIC")]
    left_out = [p for p in unpriced if p not in counted]
    if counted:
        notes.append(f"{_n(len(counted), 'holding', 'holdings')} {'has' if len(counted) == 1 else 'have'} no price source yet and {'is' if len(counted) == 1 else 'are'} counted at cost in the review: "
                     + ", ".join(p["name"] or p["isin"] for p in counted) + ". Set a ticker below.")
    if left_out:
        notes.append(f"{_n(len(left_out), 'holding', 'holdings')} {'is' if len(left_out) == 1 else 'are'} left out of the review (private funds and temporary rights): "
                     + ", ".join(p["name"] or p["isin"] for p in left_out) + ".")
    return notes + more


def import_export(broker: str = brokers.DEFAULT, data: bytes | str | None = None, parsed: dict[str, Any] | None = None,
                  merge: bool = False, progress: Progress | None = None) -> dict[str, Any]:
    """Parse (or take an already parsed export) with the broker's parser, store it, map tickers and update the portfolio."""
    progress = progress or (lambda *_: None)
    progress(0, 1, "Reading the export")
    parsed = parsed if parsed is not None else brokers.parse(broker, data or b"")
    parsed.setdefault("source", broker)
    stored = tr.save_ledger(parsed, merge=merge)
    ledger = build_ledger(stored["events"])
    open_isins = [p["isin"] for p in ledger["positions"]]
    mapping = ins.resolve(ledger["instruments"], ins.trade_prices(ledger), isins=open_isins, progress=progress)
    progress(1, 1, "Updating the portfolio")
    notes = _sync_portfolio(ledger, mapping, stored.get("source") or broker)
    return _report(stored, ledger, mapping, notes)


def import_trade_republic(data: bytes | str | None = None, parsed: dict[str, Any] | None = None, merge: bool = False,
                          progress: Progress | None = None) -> dict[str, Any]:
    """The Trade Republic import (kept for the CLI and older callers)."""
    return import_export(tr.SOURCE, data=data, parsed=parsed, merge=merge, progress=progress)


def import_status() -> dict[str, Any]:
    stored = tr.load_ledger()
    if not stored:
        return {"imported": False}
    ledger = build_ledger(stored["events"])
    return _report(stored, ledger, ins.load_map())


def set_ticker(isin: str, ticker: str | None) -> dict[str, Any]:
    """Set the ticker of one imported ISIN by hand (checked against your trade prices), then refresh positions."""
    stored = tr.load_ledger()
    if not stored:
        raise LookupError("Nothing imported yet")
    ledger = build_ledger(stored["events"])
    info = ledger["instruments"].get(isin)
    if info is None:
        raise LookupError(f"{isin} is not in the imported export")
    ins.set_manual(isin, ticker, info, ins.trade_prices(ledger).get(isin, []))
    mapping = ins.load_map()
    notes = _sync_portfolio(ledger, mapping, stored.get("source") or tr.SOURCE)
    return _report(stored, ledger, mapping, notes)


def unpriced_holdings() -> list[dict[str, Any]]:
    """Open holdings of the stored export without a usable ticker, at cost (private funds and temporary rights excepted).

    They are not in ``portfolio.json`` (a position needs a ticker), so the review adds them to the totals at cost
    and lists them under "Needs a price" with the fix.
    """
    stored = tr.load_ledger()
    if not stored:
        return []
    ledger = build_ledger(stored["events"])
    mapping = ins.load_map()
    out = []
    for p in ledger["positions"]:
        entry = mapping.get(p["isin"])
        if ins.usable(entry) or p["asset_class"] in ("PRIVATE_FUND", "SYNTHETIC"):
            continue
        out.append({"isin": p["isin"], "name": p["name"], "asset_class": p["asset_class"], "shares": p["shares"],
                    "cost": round(p["cost_broker"], 2), "status": (entry or {}).get("status", "unmapped"),
                    "reason": (entry or {}).get("reason") or ins.UNPRICED_CLASSES.get(p["asset_class"])})
    return out


def derivatives_activity(year: int) -> dict[str, Any]:
    """Money put into and taken out of derivatives (knock-out products) in a calendar year, and what is open at cost."""
    stored = tr.load_ledger()
    empty = {"bought": 0.0, "sold": 0.0, "open_cost": 0.0, "open_count": 0}
    if not stored:
        return empty
    ledger = build_ledger(stored["events"])
    bought = sold = 0.0
    for o in ledger["orders"]:
        if o["asset_class"] != "DERIVATIVE" or not str(o["date"]).startswith(str(year)):
            continue
        cash = -(o["amount"] + o["fee"] + o["tax"])  # cash put in is positive, like position_histories
        if o["side"] == "buy":
            bought += max(cash, 0.0)
        else:
            sold += max(-cash, 0.0)
    held = [p for p in ledger["positions"] if p["asset_class"] == "DERIVATIVE"]
    return {"bought": round(bought, 2), "sold": round(sold, 2),
            "open_cost": round(sum(p["cost_broker"] for p in held), 2), "open_count": len(held)}


def position_histories(isins: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Per ISIN, from the stored export: trades, net dividends and open lots, in EUR (empty when nothing is imported).

    - trades: (date, share change, cash put in) with buys positive and sales negative, fees and taxes
      included; shares moved between brokers have no cash (None) and are valued at the day's price.
    - dividends: (date, amount after withholding tax).
    - lots: open FIFO lots (buy date, cost with buy fees and taxes, shares), for the currency split.
    """
    stored = tr.load_ledger()
    if not stored:
        return {}
    ledger = build_ledger(stored["events"])
    wanted = set(isins) if isins is not None else None
    out: dict[str, dict[str, Any]] = {}

    def slot(isin: str) -> dict[str, Any] | None:
        if wanted is not None and isin not in wanted:
            return None
        info = ledger["instruments"].get(isin) or {}
        return out.setdefault(isin, {"trades": [], "dividends": [], "lots": [], "asset_class": info.get("asset_class")})

    for o in ledger["orders"]:
        s = slot(o["isin"])
        if s is not None:
            delta = o["shares"] if o["side"] == "buy" else -o["shares"]
            s["trades"].append((o["date"], delta, -(o["amount"] + o["fee"] + o["tax"])))
    for f in ledger["flows"]:
        if f["kind"] in ("securities_out", "securities_in") and f.get("isin"):
            s = slot(f["isin"])
            if s is not None:
                s["trades"].append((f["date"], f["shares"] if f["kind"] == "securities_in" else -f["shares"], None))
    for i in ledger["income"]:
        if i["kind"] == "dividend" and i.get("isin"):
            s = slot(i["isin"])
            if s is not None:
                s["dividends"].append((i["date"], i["amount"] + i["tax"]))
    for lot in ledger["lots"]:
        if lot["exit"] == "open":
            s = slot(lot["isin"])
            if s is not None:
                s["lots"].append((lot["buy_date"], lot["cost"] + lot["buy_costs"], lot["shares"]))
    for s in out.values():
        s["trades"].sort(key=lambda t: t[0])
    return out


def delete_import() -> dict[str, Any]:
    """Forget the stored export and everything derived from it: its positions, the cash it set, the ISIN-to-ticker
    map and the track-record report. Positions added by hand stay."""
    from core.style_report import share_report_path
    from core.track_record import report_path, result_path

    had_ledger = tr.delete_ledger()
    portfolio = load_portfolio()
    portfolio["positions"] = [p for p in portfolio["positions"] if p.get("source") not in brokers.SOURCES]
    if had_ledger:
        portfolio["cash"] = 0.0  # the cash came from the export's bookings
    save_portfolio(portfolio)
    for path in (ins._path(), report_path(), result_path(), share_report_path(), last_review_path()):
        path.unlink(missing_ok=True)
    return {"imported": False}
