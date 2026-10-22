"""Broker registry: which exports FastValue can import, and one entry point to parse any of them.

Each broker is a parser that turns its export into the sanitized event list the
ledger understands (see ``data.trade_republic`` for the native format and
``data.generic_ledger`` for the documented CSV any broker can be converted to).
Adding a broker is adding an entry here with a ``parse(data) -> parsed`` function;
the import flow, the portfolio and the track record need no change.
"""

from __future__ import annotations

from typing import Any, Callable

from data import generic_ledger as generic
from data import trade_republic as tr
from data.trade_republic import ExportFormatError

BROKERS: dict[str, dict[str, Any]] = {
    tr.SOURCE: {
        "label": "Trade Republic",
        "button": "Import Trade Republic export",
        "file": "your Trade Republic transaction export (a CSV file)",
        "hint": "Export your transactions as CSV in the Trade Republic app, save the file on this computer, then choose it here.",
        # The app's menu as of October 2026 (German app: Profil, Kontoauszüge, Transaktionsexport).
        "tip": ("In the Trade Republic app, tap your profile at the top right, then Statements, then Transaction export. "
                "Pick a start date before your first deposit so the whole history comes in, tap Create and save the file "
                "on this computer. It is named transactions_ followed by two dates. No Transaction export? Update the app."),
        "parse": tr.parse_export,
    },
    generic.SOURCE: {
        "label": "Any broker (FastValue CSV)",
        "button": "Import FastValue CSV",
        "file": "a FastValue CSV",
        "hint": "Fill in the FastValue CSV template from your broker's export. EUR accounts for now.",
        "parse": generic.parse_export,
        "template": True,
    },
}
SOURCES: tuple[str, ...] = tuple(BROKERS)
DEFAULT = tr.SOURCE


def describe() -> list[dict[str, Any]]:
    return [{"key": k, **{f: v for f, v in b.items() if f != "parse"}} for k, b in BROKERS.items()]


def label(source: str | None) -> str:
    return BROKERS.get(source or "", {}).get("label", source or "broker")


def parser(broker: str) -> Callable[[bytes | str], dict[str, Any]]:
    if broker not in BROKERS:
        raise ExportFormatError(f"Unknown broker {broker!r}; choose one of: " + ", ".join(BROKERS))
    return BROKERS[broker]["parse"]


_REQUIRED = {tr.SOURCE: tr.REQUIRED_COLUMNS, generic.SOURCE: generic.REQUIRED}


def _other_broker(chosen: str, raw: bytes) -> str | None:
    """The other broker whose required columns the file's first line holds, if any."""
    first = raw[:4096].decode("utf-8-sig", "ignore").splitlines()[:1]
    if not first:
        return None
    header = {h.strip().strip('"').lower() for h in first[0].replace(";", ",").split(",")}
    for key, required in _REQUIRED.items():
        if key != chosen and key in BROKERS and all(c in header for c in required):
            return key
    return None


def _wrong_broker(chosen: str, other: str) -> str:
    if other == generic.SOURCE:
        return f"This is a FastValue CSV, not a {label(chosen)} export."
    return f"This looks like a {label(other)} export."


def parse(broker: str, data: bytes | str) -> dict[str, Any]:
    """Parse ``data`` with the broker's parser. Raises ``ExportFormatError`` on a bad file or broker."""
    raw = data if isinstance(data, bytes) else data.encode("utf-8", "ignore")
    if raw[:5] == b"%PDF-":
        raise ExportFormatError("This is a PDF. FastValue needs the transaction export as a CSV file, not a statement.")
    try:
        parsed = parser(broker)(data)
    except ExportFormatError as exc:
        other = _other_broker(broker, raw)
        if other is None or exc.other_broker:
            raise
        raise ExportFormatError(_wrong_broker(broker, other), other) from exc
    parsed["source"] = broker
    return parsed
