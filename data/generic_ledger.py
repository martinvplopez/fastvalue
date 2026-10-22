"""Generic broker CSV: one documented format any broker's export can be converted to.

Trade Republic has its own parser (``data.trade_republic``). Every other broker
goes through this file: convert the export to the columns below (a spreadsheet
does it in minutes) and import it with "Any broker" in the Portfolio tab. The
events it produces are the same the ledger, the portfolio and the track record
already understand.

Columns (header names, any order, extra columns ignored, UTF-8, comma separated):

- ``date``: ``YYYY-MM-DD``, optionally followed by a time (``2024-01-03 09:00``
  or ISO 8601). Rows are replayed in this order.
- ``type``: ``deposit``, ``withdrawal``, ``card`` (card spending), ``buy``,
  ``sell``, ``dividend``, ``interest``, ``fee``, ``tax``, ``transfer_in``,
  ``transfer_out`` (shares moved to or from another broker), ``perk``.
- ``isin``: the instrument, for buys, sells, dividends and transfers.
- ``name``: the instrument's name (optional; kept only on rows with an ISIN).
- ``shares``: for buys, sells and transfers (positive; the type gives the sign).
- ``price``: price per share in the account currency (buys and sells; optional
  for transfers, where it sets the cost basis of shares moved in).
- ``amount``: the cash of the row. The type gives the sign (money in is
  positive, money out negative), so an unsigned column works too.
- ``fee``, ``tax``: the row's fee and withheld tax, as costs (made negative); a
  tax refund is a ``tax`` row with a positive amount in ``tax``.
- ``currency``: the account currency, the same on every row. EUR only for now:
  the ledger, the portfolio import and the track record are in EUR.
- ``id`` (optional): a unique id per row; missing ids are derived from the row.
- ``asset_class`` (optional): ``stock`` (default), ``fund``, ``derivative``,
  ``bond`` or ``crypto``.
- ``plan`` (optional): ``yes``, ``true`` or ``1`` marks a savings-plan buy.

Privacy: the file is read in memory and never stored; only these columns are
read, and a row without an ISIN keeps no name. The cleaned events are stored
under data/store/ (git-ignored) like any other import.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from data.trade_republic import EMPTY_FILE, ISIN_RE, MAX_BYTES, ExportFormatError

SOURCE = "generic"
REQUIRED = ("date", "type", "amount")
OPTIONAL = ("isin", "name", "shares", "price", "fee", "tax", "currency", "id", "asset_class", "plan")
TYPES = {
    "deposit": ("deposit", 1), "withdrawal": ("withdrawal", -1), "card": ("card", -1),
    "buy": ("buy", -1), "sell": ("sell", 1), "dividend": ("dividend", 1), "interest": ("interest", 1),
    "fee": ("fee", -1), "tax": ("tax", 0), "transfer_in": ("transfer_in", 0), "transfer_out": ("transfer_out", 0),
    "perk": ("perk", 1),
}
ASSET_CLASSES = {"stock": "STOCK", "fund": "FUND", "etf": "FUND", "derivative": "DERIVATIVE", "turbo": "DERIVATIVE",
                 "bond": "BOND", "crypto": "CRYPTO"}
WITH_ISIN = ("buy", "sell", "dividend", "transfer_in", "transfer_out")
TRADE_TOLERANCE = 0.005  # relative gap allowed between |amount| and shares x price
SUPPORTED_CURRENCIES = ("EUR",)

TEMPLATE_ROWS = [
    ["date", "type", "isin", "name", "shares", "price", "amount", "fee", "tax", "currency", "id", "asset_class", "plan"],
    ["2024-01-02", "deposit", "", "", "", "", "10000", "", "", "EUR", "", "", ""],
    ["2024-01-03 09:30", "buy", "US0000000001", "Example Corp", "10", "100", "-1000", "-1", "", "EUR", "", "stock", ""],
    ["2024-01-15", "buy", "IE0000000006", "World ETF", "2", "100", "-200", "", "", "EUR", "", "fund", "yes"],
    ["2024-03-01", "sell", "US0000000001", "Example Corp", "5", "120", "600", "-1", "-10", "EUR", "", "stock", ""],
    ["2024-03-05", "dividend", "US0000000001", "Example Corp", "", "", "5", "", "-1", "EUR", "", "", ""],
    ["2024-04-01", "interest", "", "", "", "", "20", "", "-5", "EUR", "", "", ""],
    ["2024-09-10", "withdrawal", "", "", "", "", "-300", "", "", "EUR", "", "", ""],
]


def template_csv() -> str:
    buf = io.StringIO()
    csv.writer(buf).writerows(TEMPLATE_ROWS)
    return buf.getvalue()


def _num(value: Any, label: str, line: int, errors: list[str]) -> float | None:
    text = str(value or "").strip().replace("−", "-")
    if not text:
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")  # 1234,56
    text = text.replace(" ", "")
    try:
        return float(text)
    except ValueError:
        errors.append(f"Line {line}: {label} is not a number ({value!r})")
        return None


def _when(value: str, line: int, errors: list[str]) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        errors.append(f"Line {line}: date {value!r} is not YYYY-MM-DD")
        return None


def parse_export(data: bytes | str) -> dict[str, Any]:
    """Parse the generic CSV into the events the ledger understands. Raises ``ExportFormatError`` on a bad file."""
    if isinstance(data, bytes):
        if len(data) > MAX_BYTES:
            raise ExportFormatError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ExportFormatError("The file is not UTF-8 text; save it as CSV UTF-8.") from exc
    else:
        text = data.lstrip("﻿")
    if not text.strip():
        raise ExportFormatError(EMPTY_FILE)
    sample = text[:4096]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    header = [h.strip().lower() for h in (reader.fieldnames or [])]
    reader.fieldnames = header
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise ExportFormatError("This file does not have FastValue's columns (missing " + ", ".join(missing) + "). It needs at least "
                                "date, type and amount, plus isin, shares and price for trades. Download the CSV template "
                                "and copy your rows into it.")

    errors: list[str] = []
    unknown_types: list[str] = []
    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    currencies: Counter[str] = Counter()
    seen: set[str] = set()
    duplicates = 0
    for line, raw in enumerate(reader, start=2):
        kind_in = str(raw.get("type") or "").strip().lower().replace(" ", "_")
        if kind_in not in TYPES:
            errors.append(f"Line {line}: unknown type {kind_in!r}")
            unknown_types.append(kind_in or "(empty)")
            continue
        kind, sign = TYPES[kind_in]
        when = _when(raw.get("date", ""), line, errors)
        if when is None:
            continue
        isin = str(raw.get("isin") or "").strip().upper()
        if isin and not ISIN_RE.match(isin):
            errors.append(f"Line {line}: {isin!r} is not an ISIN")
            continue
        if kind in WITH_ISIN and not isin:
            errors.append(f"Line {line}: a {kind_in} row needs an ISIN")
            continue
        amount = _num(raw.get("amount"), "amount", line, errors) or 0.0
        fee = _num(raw.get("fee"), "fee", line, errors) or 0.0
        tax = _num(raw.get("tax"), "tax", line, errors) or 0.0
        shares = _num(raw.get("shares"), "shares", line, errors)
        price = _num(raw.get("price"), "price", line, errors)
        if sign:
            amount = sign * abs(amount)
        fee = -abs(fee)
        if kind == "tax":
            tax = tax if tax else amount  # a refund stays positive, a charge negative, as written
            amount = 0.0
        elif kind == "fee":
            fee = fee if fee else -abs(amount)
            amount = 0.0
        else:
            tax = -abs(tax) if kind != "tax" else tax
        if kind in ("buy", "sell"):
            if not shares or shares <= 0 or not price or price <= 0:
                errors.append(f"Line {line}: a {kind} row needs positive shares and price")
                continue
            if abs(abs(amount) - shares * price) > max(TRADE_TOLERANCE * shares * price, 0.02):
                errors.append(f"Line {line}: amount {abs(amount):.2f} does not equal shares x price {shares * price:.2f}")
            shares = shares if kind == "buy" else -shares
        elif kind in ("transfer_in", "transfer_out"):
            if not shares or shares <= 0:
                errors.append(f"Line {line}: a {kind_in} row needs positive shares")
                continue
            shares = shares if kind == "transfer_in" else -shares
            amount = 0.0
        ccy = str(raw.get("currency") or "EUR").strip().upper() or "EUR"
        currencies[ccy] += 1
        asset_class = ASSET_CLASSES.get(str(raw.get("asset_class") or "").strip().lower(), "STOCK") if isin else ""
        plan = str(raw.get("plan") or "").strip().lower() in ("yes", "true", "1", "y")
        row_id = str(raw.get("id") or "").strip()
        if not row_id:
            digest = hashlib.sha1("|".join([when.isoformat(), kind, isin, f"{shares}", f"{amount:.6f}", f"{fee:.6f}",
                                             f"{tax:.6f}", str(line)]).encode()).hexdigest()[:16]
            row_id = f"gen-{digest}"
        if row_id in seen:
            duplicates += 1
            continue
        seen.add(row_id)
        ev: dict[str, Any] = {
            "id": row_id, "ts": when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "date": when.date().isoformat(), "type": kind_in.upper(), "kind": kind, "asset_class": asset_class,
            "isin": isin, "raw_isin": None, "name": str(raw.get("name") or "").strip() if isin else "",
            "shares": shares, "price": price, "amount": round(amount, 6), "fee": round(fee, 6), "tax": round(tax, 6),
            "cash": round(amount + fee + tax, 6),
        }
        if kind in ("buy", "sell"):
            ev["savings_plan"] = plan
            ev["order"] = row_id
        events.append(ev)
    if not events and not errors:
        raise ExportFormatError("No rows found in the file")
    if errors and not events:
        if unknown_types and len(unknown_types) == len(errors):
            names = list(dict.fromkeys(unknown_types))
            listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            raise ExportFormatError(f"No row could be read: {listed} {'is' if len(names) == 1 else 'are'} not "
                                    f"{'a type' if len(names) == 1 else 'types'} FastValue knows. Use "
                                    "deposit, withdrawal, card, buy, sell, dividend, interest, fee, tax, transfer_in, "
                                    "transfer_out or perk.")
        raise ExportFormatError("No row could be read: " + "; ".join(errors[:3]))
    events.sort(key=lambda e: e["ts"])
    unsupported = [c for c in currencies if c not in SUPPORTED_CURRENCIES]
    if unsupported:
        raise ExportFormatError(f"FastValue imports EUR accounts only for now, and this file has {', '.join(sorted(unsupported))} rows. "
                                "You can type your holdings in below instead.")
    if len(currencies) > 1:
        raise ExportFormatError("Every row must be in the same currency")
    types = Counter(e["type"] for e in events)
    checks = [
        {"name": "Every row parsed", "ok": not errors, "detail": "no errors" if not errors else f"{len(errors)} {'row' if len(errors) == 1 else 'rows'} skipped or flagged"},
        {"name": "Ids are unique", "ok": duplicates == 0, "detail": "no duplicates" if not duplicates else f"{duplicates} duplicate {'row' if duplicates == 1 else 'rows'} dropped"},
        {"name": "Trade amounts equal shares x price", "ok": not any("does not equal" in e for e in errors),
         "detail": f"within {TRADE_TOLERANCE:.1%}"},
        {"name": "One currency", "ok": True, "detail": ", ".join(currencies) or "EUR"},
    ]
    if duplicates:
        warnings.append(f"{duplicates} duplicate {'row' if duplicates == 1 else 'rows'} dropped")
    return {
        "source": SOURCE,
        "rows": len(events),
        "type_counts": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "period": {"first": min(e["date"] for e in events), "last": max(e["date"] for e in events)},
        "date_differs_from_timestamp": 0,
        "aliases": {},
        "events": events,
        "checks": checks,
        "warnings": warnings,
        "errors": errors[:50],
    }
