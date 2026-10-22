"""Trade Republic transaction export: parse, validate and sanitize.

The export (``transactions_<from>_<to>.csv``) has one row per booking: trades,
cash movements, dividends, corporate actions and deliveries, all in EUR. This
module turns it into clean events for ``core.ledger``:

- Personal data is dropped while the file is read. The counterparty name and
  IBAN columns go, and so do the name, description, merchant code and original
  currency fields of bank transfers and card payments (they repeat the account
  holder's name, IBANs and where the money was spent). The ledger needs only
  the date, the type and the amount of those rows.
- Instruments are keyed by ISIN, never by name (names repeat and change). An
  ISIN change (an EXCHANGE pair) folds the old ISIN into the new one, and a
  temporary rights ISIN (asset class SYNTHETIC) is attributed to its parent
  stock. Both go through an alias table that is inferred here and can be
  overridden.
- Orders the broker splits into a whole-share row (with the fee) and a
  fractional row (without) share one order id, so round trips count them once.

Nothing here touches the network. ``save_ledger`` stores the sanitized events
under data/store/ (git-ignored); the raw file is never stored.
"""

from __future__ import annotations

import csv
import io
import json
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config.settings as settings

SOURCE = "trade_republic"
EMPTY_FILE = "This file is empty. Download the export from your broker again and choose the new file."
REQUIRED_COLUMNS = ("datetime", "date", "category", "type", "asset_class", "name", "symbol", "shares", "price",
                    "amount", "fee", "tax", "currency", "description", "transaction_id")
# Read only to be thrown away: never kept in memory past the row they arrive in.
PERSONAL_COLUMNS = ("counterparty_name", "counterparty_iban", "payment_reference", "mcc_code",
                    "original_amount", "original_currency", "fx_rate")
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
MAX_BYTES = 8 * 1024 * 1024
TRADE_TOLERANCE = 0.005  # abs(amount) = shares x price within this many EUR
SPLIT_ORDER_SECONDS = 5.0  # whole and fractional rows of one order arrive within milliseconds

# type -> event kind
FLOW_KINDS = {
    "CUSTOMER_INBOUND": "deposit", "CUSTOMER_INPAYMENT": "deposit", "TRANSFER_INBOUND": "deposit",
    "TRANSFER_INSTANT_INBOUND": "deposit",
    "TRANSFER_INSTANT_OUTBOUND": "withdrawal", "TRANSFER_OUTBOUND": "withdrawal",
    "CUSTOMER_OUTBOUND_REQUEST": "withdrawal",
    "CARD_TRANSACTION": "card", "CARD_TRANSACTION_INTERNATIONAL": "card",
}
INCOME_KINDS = {"INTEREST_PAYMENT": "interest", "DIVIDEND": "dividend", "STOCKPERK": "perk",
                "LIQUIDATION_PROCEEDS": "dividend", "BONUS": "bonus"}
TAX_TYPES = ("MANUAL", "TAX_OPTIMIZATION")
CORPORATE_KINDS = {"EXCHANGE": "isin_change", "INTERMEDIATE_SECURITIES_DISTRIBUTION": "rights",
                   "LIQUIDATION_DIVIDEND": "rights"}
KNOWN_TYPES = (set(FLOW_KINDS) | set(INCOME_KINDS) | set(TAX_TYPES) | set(CORPORATE_KINDS)
               | {"BUY", "SELL", "PRIVATE_MARKET_BUY", "MIGRATION", "FREE_RECEIPT"})
# Rows whose free text is personal (transfers, card payments, the account holder's own name).
SCRUB_KINDS = {"deposit", "withdrawal", "card"}

_LOCK = threading.Lock()


class ExportFormatError(ValueError):
    """The file is not a Trade Republic export this importer understands.

    ``other_broker`` names the broker whose export the file looks like, when it is not the one chosen.
    """

    def __init__(self, message: str, other_broker: str | None = None):
        super().__init__(message)
        self.other_broker = other_broker


def _num(value: str, label: str, line: int, errors: list[str]) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        errors.append(f"Line {line}: {label} is not a number")
        return None
    if out != out or out in (float("inf"), float("-inf")):
        errors.append(f"Line {line}: {label} is not a finite number")
        return None
    return out


def _timestamp(value: str, line: int, errors: list[str]) -> datetime | None:
    text = (value or "").strip()
    try:
        ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"Line {line}: datetime is not ISO 8601")
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _turbo(name: str, description: str) -> dict[str, Any] | None:
    """Direction, strike and underlying of a knock-out product, from its name and trade description."""
    m = re.match(r"^(Long|Short)\s+([\d.]+(?:,\d+)?)\s*(\S+)?", name or "")
    under = re.search(r"(?:Turbo|Mini Future \(Turbo\))\s+auf\s+(.+?)(?:,\s*quantity:|$)", description or "")
    if not m and not under:
        return None
    strike = None
    if m:
        try:
            strike = float(m.group(2).replace(".", "").replace(",", "."))
        except ValueError:
            strike = None
    return {
        "direction": m.group(1).lower() if m else None,
        "strike": strike,
        "strike_currency": (m.group(3) if m else None),
        "underlying": re.sub(r"\s+", " ", under.group(1)).strip() if under else None,
    }


def _first_word(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text or "")
    return words[0].lower() if words else ""


def infer_aliases(rows: list[dict[str, Any]], overrides: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Old ISIN -> new ISIN (EXCHANGE pairs) and rights ISIN -> parent ISIN (SYNTHETIC)."""
    aliases: dict[str, str] = {}
    warnings: list[str] = []
    by_ts: dict[datetime, list[dict[str, Any]]] = {}
    for r in rows:
        if r["type"] == "EXCHANGE" and r["isin"]:
            by_ts.setdefault(r["ts"], []).append(r)
    for ts, pair in by_ts.items():
        out = [r for r in pair if (r["shares"] or 0) < 0]
        into = [r for r in pair if (r["shares"] or 0) > 0]
        if len(out) == 1 and len(into) == 1 and out[0]["isin"] != into[0]["isin"]:
            aliases[out[0]["isin"]] = into[0]["isin"]
        else:
            warnings.append(f"ISIN change on {ts.date()} is not a clean pair; left as booked")

    names: dict[str, str] = {}
    for r in rows:
        if r["isin"] and r["asset_class"] not in ("SYNTHETIC", "") and r["category"] == "TRADING":
            names.setdefault(r["isin"], r["name"])
    for isin in sorted({r["isin"] for r in rows if r["asset_class"] == "SYNTHETIC" and r["isin"]}):
        name = next(r["name"] for r in rows if r["isin"] == isin)
        word = _first_word(name)
        matches = [i for i, n in names.items() if word and _first_word(n) == word]
        same_country = [i for i in matches if i[:2] == isin[:2]]
        pick = same_country or matches
        if len(pick) == 1:
            aliases[isin] = pick[0]
        else:
            warnings.append(f"Rights ISIN {isin} could not be matched to its stock; its cash stays under its own ISIN")
    for old, new in (overrides or {}).items():
        if ISIN_RE.match(str(old)) and ISIN_RE.match(str(new)):
            aliases[str(old)] = str(new)
    # Follow chains (A -> B -> C).
    for old in list(aliases):
        seen, target = {old}, aliases[old]
        while target in aliases and target not in seen:
            seen.add(target)
            target = aliases[target]
        aliases[old] = target
    return aliases, warnings


def _kind(row: dict[str, Any]) -> str:
    t, cat = row["type"], row["category"]
    if row["asset_class"] == "PRIVATE_FUND" or t == "PRIVATE_MARKET_BUY":
        return "private"
    if t == "BONUS" and "private markets" in row["_description"].lower():
        return "private"
    if cat == "TRADING" and t in ("BUY", "SELL"):
        return t.lower()
    if t in FLOW_KINDS:
        return FLOW_KINDS[t]
    if t in INCOME_KINDS:
        return INCOME_KINDS[t]
    if t in TAX_TYPES:
        return "tax"
    if t in CORPORATE_KINDS:
        return CORPORATE_KINDS[t]
    if t == "MIGRATION":
        return "migration"
    if t == "FREE_RECEIPT":
        return "transfer_out" if (row["shares"] or 0) < 0 else "transfer_in"
    return "other"


def parse_export(data: bytes | str, aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """Parse and sanitize an export. Returns events, checks, warnings and a summary of what was read.

    Raises ``ExportFormatError`` when the file is not a Trade Republic export (missing columns, empty, too large).
    """
    if isinstance(data, bytes):
        if len(data) > MAX_BYTES:
            raise ExportFormatError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB")
        try:
            text = data.decode("utf-8-sig", errors="strict") if data else ""
        except UnicodeDecodeError as exc:
            raise ExportFormatError("The file is not UTF-8 text. Use the export exactly as downloaded from "
                                    "Trade Republic (re-saving it in Excel changes the encoding).") from exc
    else:
        text = data.lstrip("﻿")
    if not text.strip():
        raise ExportFormatError(EMPTY_FILE)
    if "\x00" in text:
        raise ExportFormatError("The file contains binary data: it is not a CSV export")
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ExportFormatError("Not a Trade Republic transaction export: missing columns " + ", ".join(missing))

    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for line, raw in enumerate(reader, start=2):
        for col in PERSONAL_COLUMNS:  # dropped before anything else reads the row
            raw.pop(col, None)
        ts = _timestamp(raw.get("datetime", ""), line, errors)
        if ts is None:
            continue
        isin = (raw.get("symbol") or "").strip().upper()
        if isin and not ISIN_RE.match(isin):
            errors.append(f"Line {line}: symbol is not an ISIN")
            isin = ""
        row = {
            "line": line,
            "id": (raw.get("transaction_id") or "").strip() or f"line-{line}",
            "ts": ts,
            "date": (raw.get("date") or "").strip()[:10] or ts.date().isoformat(),
            "category": (raw.get("category") or "").strip().upper(),
            "type": (raw.get("type") or "").strip().upper(),
            "asset_class": (raw.get("asset_class") or "").strip().upper(),
            "name": (raw.get("name") or "").strip(),
            "isin": isin,
            "shares": _num(raw.get("shares", ""), "shares", line, errors),
            "price": _num(raw.get("price", ""), "price", line, errors),
            "amount": _num(raw.get("amount", ""), "amount", line, errors) or 0.0,
            "fee": _num(raw.get("fee", ""), "fee", line, errors) or 0.0,
            "tax": _num(raw.get("tax", ""), "tax", line, errors) or 0.0,
            "currency": (raw.get("currency") or "").strip().upper(),
            "_description": (raw.get("description") or "").strip(),
        }
        row["kind"] = _kind(row)
        # Without an ISIN a row is never an instrument, so its name and description can only be personal
        # (counterparties, merchants, references): blank them all, not only the kinds known today.
        if row["kind"] in SCRUB_KINDS or not row["isin"]:
            row["name"] = ""
            row["_description"] = ""
        rows.append(row)
    if not rows:
        raise ExportFormatError("No transactions found in the file")

    # Stable order: by event time, then by position in the file.
    rows.sort(key=lambda r: (r["ts"], r["line"]))
    seen: set[str] = set()
    duplicates = 0
    unique: list[dict[str, Any]] = []
    for r in rows:
        if r["id"] in seen:
            duplicates += 1
            continue
        seen.add(r["id"])
        unique.append(r)
    rows = unique

    alias_map, warnings = infer_aliases(rows, aliases)
    events = _events(rows, alias_map)
    checks = _checks(rows, events, errors, duplicates)
    types = Counter(r["type"] for r in rows)
    unknown = sorted(t for t in types if t not in KNOWN_TYPES)
    if unknown:
        warnings.append("Row types this importer does not model (their cash is counted, positions are not): "
                        + ", ".join(f"{t} ({types[t]})" for t in unknown))
    return {
        "source": SOURCE,
        "rows": len(rows),
        "type_counts": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "period": {"first": min(r["date"] for r in rows), "last": max(r["date"] for r in rows)},
        "date_differs_from_timestamp": sum(1 for r in rows if r["date"] != r["ts"].date().isoformat()),
        "aliases": alias_map,
        "events": events,
        "checks": checks,
        "warnings": warnings,
        "errors": errors[:50],
    }


def _events(rows: list[dict[str, Any]], aliases: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_order: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        isin = aliases.get(r["isin"], r["isin"])
        desc = r["_description"]
        ev: dict[str, Any] = {
            "id": r["id"],
            "ts": r["ts"].isoformat().replace("+00:00", "Z"),
            "date": r["date"],
            "type": r["type"],
            "kind": r["kind"],
            "asset_class": r["asset_class"],
            "isin": isin,
            "raw_isin": r["isin"] if r["isin"] != isin else None,
            "name": r["name"],
            "shares": r["shares"],
            "price": r["price"],
            "amount": round(r["amount"], 6),
            "fee": round(r["fee"], 6),
            "tax": round(r["tax"], 6),
            "cash": round(r["amount"] + r["fee"] + r["tax"], 6),
        }
        if r["kind"] in ("buy", "sell"):
            ev["savings_plan"] = desc.lower().startswith("savings plan execution")
            key = (isin, r["kind"])
            prev = last_order.get(key)
            if (prev is not None and prev["price"] == r["price"] and prev["date"] == r["date"]
                    and abs((r["ts"] - prev["_ts"]).total_seconds()) <= SPLIT_ORDER_SECONDS):
                ev["order"] = prev["order"]
            else:
                ev["order"] = r["id"]
            last_order[key] = {"order": ev["order"], "price": r["price"], "date": r["date"], "_ts": r["ts"]}
            if r["asset_class"] == "DERIVATIVE":
                ev["turbo"] = _turbo(r["name"], desc)
        out.append(ev)
    return out


def _checks(rows: list[dict[str, Any]], events: list[dict[str, Any]], errors: list[str], duplicates: int) -> list[dict[str, Any]]:
    """Consistency checks that need nothing but the file itself."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("Numbers parse", not errors, "all fields read" if not errors else f"{len(errors)} field(s) could not be read")
    add("Unique transaction ids", duplicates == 0,
        "no duplicates" if not duplicates else f"{duplicates} duplicate row(s) dropped")
    other = sorted({r["currency"] for r in rows if r["currency"] not in ("EUR", "")})
    add("Account currency EUR", not other, "all EUR" if not other else "other currencies: " + ", ".join(other))
    trades = [r for r in rows if r["kind"] in ("buy", "sell") and r["shares"] is not None and r["price"] is not None]
    off = [r for r in trades if abs(abs(r["amount"]) - abs(r["shares"]) * r["price"]) > TRADE_TOLERANCE + 1e-9]
    add("Trade amount = shares x price", not off,
        f"{len(trades)} trades within {TRADE_TOLERANCE} EUR" if not off else f"{len(off)} of {len(trades)} trades differ")
    signs = [r for r in trades if (r["kind"] == "buy") != ((r["shares"] or 0) > 0) or (r["kind"] == "buy") != (r["amount"] < 0)]
    add("Buy and sell signs", not signs, "consistent" if not signs else f"{len(signs)} rows with unexpected signs")
    mig = Counter()
    for r in rows:
        if r["kind"] == "migration":
            mig[r["isin"]] += r["shares"] or 0.0
    unbalanced = [i for i, v in mig.items() if abs(v) > 1e-6]
    add("Broker migration rows net to zero", not unbalanced,
        f"{sum(1 for r in rows if r['kind'] == 'migration')} rows, net zero" if not unbalanced
        else f"{len(unbalanced)} instrument(s) do not net to zero")
    # Positions never go below zero.
    pos: dict[str, float] = {}
    short: set[str] = set()
    for e in events:
        if e["kind"] in ("buy", "sell", "transfer_out", "transfer_in") and e["asset_class"] != "PRIVATE_FUND":
            pos[e["isin"]] = pos.get(e["isin"], 0.0) + (e["shares"] or 0.0)
            if pos[e["isin"]] < -1e-6:
                short.add(e["isin"])
    add("No position goes below zero", not short,
        "every sale was covered by earlier buys" if not short else f"{len(short)} instrument(s) sold beyond their holding")
    return checks


# ------------------------------------------------------------------ storage


def _path() -> Path:
    return settings.STORE_DIR / "broker_ledger.json"


def event_time(ts: str) -> datetime:
    """Sort key for stored timestamps: as text, ``10:00:00.4Z`` sorts before ``10:00:00Z``."""
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _resolve(aliases: dict[str, str]) -> dict[str, str]:
    out = dict(aliases)
    for old in list(out):
        seen, target = {old}, out[old]
        while target in out and target not in seen:
            seen.add(target)
            target = out[target]
        out[old] = target
    return out


def merge_events(existing: list[dict[str, Any]], new: list[dict[str, Any]],
                 aliases: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Union of two exports by transaction id (the newer export wins), in event order.

    ``aliases`` (ISIN changes of both imports) are applied again to every event, so a change that only the newer
    file shows also moves the older file's buys onto the new ISIN.
    """
    by_id = {e["id"]: dict(e) for e in existing}
    by_id.update({e["id"]: dict(e) for e in new})
    resolved = _resolve(aliases or {})
    for e in by_id.values():
        raw = e.get("raw_isin") or e["isin"]
        target = resolved.get(raw, raw)
        e["isin"], e["raw_isin"] = target, (raw if raw != target else None)
    return sorted(by_id.values(), key=lambda e: event_time(e["ts"]))


def save_ledger(parsed: dict[str, Any], merge: bool = False) -> dict[str, Any]:
    """Store the sanitized events (git-ignored). With ``merge``, events from an earlier import are kept."""
    events = parsed["events"]
    if merge:
        previous = load_ledger()
        if previous:
            aliases = {**(previous.get("aliases") or {}), **(parsed.get("aliases") or {})}
            events = merge_events(previous.get("events", []), events, aliases)
            parsed = {**parsed, "aliases": _resolve(aliases)}
    events = _scrub([dict(e) for e in events])
    payload = {
        "source": parsed.get("source") or SOURCE,
        "imported_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(events),
        "period": {"first": min(e["date"] for e in events), "last": max(e["date"] for e in events)},
        "aliases": parsed.get("aliases", {}),
        # What the last file looked like (counts and checks only, no row content).
        "report": {k: parsed.get(k) for k in ("type_counts", "checks", "warnings", "errors",
                                               "date_differs_from_timestamp")},
        "events": events,
    }
    with _LOCK:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    return payload


def _scrub(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No name on an event without an ISIN (ledgers stored before that rule covered every such row)."""
    for e in events:
        if not e.get("isin") and e.get("name"):
            e["name"] = ""
    return events


def load_ledger() -> dict[str, Any] | None:
    path = _path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not (isinstance(data, dict) and isinstance(data.get("events"), list)):
        return None
    _scrub(data["events"])
    return data


def delete_ledger() -> bool:
    path = _path()
    if path.exists():
        path.unlink()
        return True
    return False
