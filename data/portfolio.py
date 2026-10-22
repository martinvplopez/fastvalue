"""Portfolio store: the positions you hold, kept in ``data/store/portfolio.json`` (git-ignored).

A position is a ticker, a share count, an average cost and, optionally, the
date it was opened and free-text notes. When a position is added through the
API or CLI after an analysis, the owner score and decision at that moment are
stored with it so later reviews can show how the thesis has moved.

The portfolio has one currency (``currency``): cash and average costs are in
it, and the review converts each holding's price from its listing currency.
Positions imported from a broker export also carry their ISIN, name and
``source``; a new import replaces the positions of the same source and keeps
the ones you added by hand.

The portfolio also holds your plan: the share you want in core funds and a
yearly budget for derivatives, both optional (the review checks drift and
spending against them). Only what is needed to review risk is stored: no
broker identifiers, no account numbers. The file is personal and stays out of
git.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any

import config.settings as settings

_LOCK = threading.Lock()
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,11}$")
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
from data.brokers import SOURCES as BROKER_SOURCES, label as broker_label  # noqa: E402 - the registry has no dependency on this module

SOURCES = ("manual",) + BROKER_SOURCES
DECISIONS = frozenset({"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "NOT_RATED"})
MAX_POSITIONS = 100
PLAN_KEYS = ("core_target", "derivatives_budget")


class PortfolioError(ValueError):
    pass


def _path() -> Path:
    return settings.STORE_DIR / "portfolio.json"


def _clean_ticker(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not TICKER_RE.match(symbol):
        raise PortfolioError(f"Invalid ticker symbol: {symbol!r}")
    return symbol


def _positive(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise PortfolioError(f"{label} must be a number") from exc
    if not out > 0 or out != out:
        raise PortfolioError(f"{label} must be greater than 0")
    return out


def _iso_date(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    try:
        parsed = date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise PortfolioError(f"{label} must be an ISO date (YYYY-MM-DD)") from exc
    if parsed > date.today():
        raise PortfolioError("The opening date cannot be in the future.")
    return parsed.isoformat()


def clean_position(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate one position dict. Raises PortfolioError."""
    if not isinstance(raw, dict):
        raise PortfolioError("A position must be an object")
    out: dict[str, Any] = {
        "ticker": _clean_ticker(raw.get("ticker")),
        "shares": _positive(raw.get("shares"), "Shares"),
        "avg_cost": _positive(raw.get("avg_cost"), "Average cost"),
        "opened": _iso_date(raw.get("opened"), "Opened"),
        "notes": str(raw.get("notes") or "")[:500],
    }
    for key in ("owner_score_at_entry", "timing_score_at_entry"):
        value = raw.get(key)
        out[key] = float(value) if isinstance(value, (int, float)) and value == value else None
    decision = str(raw.get("owner_decision_at_entry") or "").upper()
    out["owner_decision_at_entry"] = decision if decision in DECISIONS else None
    out["rated_at"] = str(raw["rated_at"])[:32] if raw.get("rated_at") else None
    isin = str(raw.get("isin") or "").strip().upper()
    out["isin"] = isin if ISIN_RE.match(isin) else None
    out["name"] = str(raw.get("name") or "")[:80] or None
    out["source"] = raw.get("source") if raw.get("source") in SOURCES else "manual"
    return out


def clean_plan(raw: Any) -> dict[str, float | None]:
    """Validate the plan: ``core_target`` a share between 0 and 1, ``derivatives_budget`` an amount of 0 or more (both optional)."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise PortfolioError("The plan must be an object")
    out: dict[str, float | None] = {}
    for key, label in (("core_target", "Core share"), ("derivatives_budget", "Derivatives budget")):
        value = raw.get(key)
        if value in (None, ""):
            out[key] = None
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PortfolioError(f"{label} must be a number") from exc
        if number != number or number < 0:
            raise PortfolioError(f"{label} cannot be negative")
        if key == "core_target" and number > 1:
            raise PortfolioError("Core share must be between 0% and 100%")
        out[key] = round(number, 4) if key == "core_target" else round(number, 2)
    return out


def _empty() -> dict[str, Any]:
    # A new portfolio is in EUR, like the imports; a stored file without a currency still reads as USD (clean_portfolio).
    return {"cash": 0.0, "currency": "EUR", "positions": [], "plan": clean_plan(None)}


def clean_portfolio(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a whole portfolio payload (cash plus positions). Raises PortfolioError."""
    if not isinstance(raw, dict):
        raise PortfolioError("Portfolio must be an object")
    cash = raw.get("cash", 0.0)
    try:
        cash = float(cash or 0.0)
    except (TypeError, ValueError) as exc:
        raise PortfolioError("Cash must be a number") from exc
    if cash < 0 or cash != cash:
        raise PortfolioError("Cash cannot be negative")
    positions_raw = raw.get("positions") or []
    if not isinstance(positions_raw, list):
        raise PortfolioError("Positions must be a list")
    if len(positions_raw) > MAX_POSITIONS:
        raise PortfolioError(f"At most {MAX_POSITIONS} positions")
    positions: dict[str, dict[str, Any]] = {}
    for item in positions_raw:
        pos = clean_position(item)
        if pos["ticker"] in positions:
            raise PortfolioError(f"Duplicate position: {pos['ticker']}")
        positions[pos["ticker"]] = pos
    currency = str(raw.get("currency") or "USD").strip().upper()
    if not CURRENCY_RE.match(currency):
        raise PortfolioError("Currency must be a three-letter code such as EUR or USD")
    return {"cash": round(cash, 2), "currency": currency, "positions": list(positions.values()),
            "plan": clean_plan(raw.get("plan"))}


def load_portfolio() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return _empty()
    try:
        return clean_portfolio(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, PortfolioError):
        return _empty()


def save_portfolio(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist the whole portfolio. Returns the stored version (a payload without a plan keeps the stored one)."""
    if isinstance(raw, dict) and "plan" not in raw:
        raw = {**raw, "plan": load_portfolio()["plan"]}
    portfolio = clean_portfolio(raw)
    with _LOCK:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(portfolio, indent=2))
        tmp.replace(path)
    return portfolio


def last_review_path() -> Path:
    return settings.STORE_DIR / "last_review.json"


def _signature(portfolio: dict[str, Any]) -> str:
    body = json.dumps({"positions": portfolio.get("positions"), "cash": portfolio.get("cash")}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def save_last_review(review: dict[str, Any]) -> None:
    """Keep the latest review next to the portfolio (local, git-ignored) so a reload can show it again."""
    payload = {"portfolio_sig": _signature(load_portfolio()), "review": review}
    with _LOCK:
        path = last_review_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(path)


def load_last_review() -> dict[str, Any]:
    """The saved review, only while the holdings and cash are what they were when it ran; otherwise {}."""
    try:
        saved = json.loads(last_review_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(saved, dict) or saved.get("portfolio_sig") != _signature(load_portfolio()):
        return {}
    review = saved.get("review")
    return review if isinstance(review, dict) else {}


def clear_last_review() -> None:
    last_review_path().unlink(missing_ok=True)


def get_position(ticker: str) -> dict[str, Any] | None:
    symbol = _clean_ticker(ticker)
    return next((p for p in load_portfolio()["positions"] if p["ticker"] == symbol), None)


def imported_message(position: dict[str, Any]) -> str:
    """Why a holding from a broker import cannot be typed over, and the way to change it."""
    return (f"{position['ticker']} comes from your {broker_label(position.get('source'))} import. To change it, import a "
            "newer export: a holding typed here would replace the imported one.")


def check_imported_unchanged(raw: dict[str, Any]) -> None:
    """Refuse a whole-portfolio payload that drops or changes an imported holding (a newer import changes those)."""
    incoming = {p["ticker"]: p for p in clean_portfolio(raw)["positions"]}
    for p in load_portfolio()["positions"]:
        if p.get("source") in (None, "manual"):
            continue
        new = incoming.get(p["ticker"])
        if new is None or (new["source"], new["shares"], new["avg_cost"]) != (p["source"], p["shares"], p["avg_cost"]):
            raise PortfolioError(imported_message(p))


def upsert_position(position: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Add or replace one position (by ticker). Returns the stored portfolio.

    A position from a broker import is replaced only with ``force``: typed over by accident, it would silently drop
    the imported shares and cost.
    """
    pos = clean_position(position)
    portfolio = load_portfolio()
    existing = next((p for p in portfolio["positions"] if p["ticker"] == pos["ticker"]), None)
    if existing is not None and existing.get("source") not in (None, "manual") and not force:
        raise PortfolioError(imported_message(existing))
    if existing is not None:
        # Keep the entry snapshot unless a new one is supplied.
        for key in ("owner_score_at_entry", "owner_decision_at_entry", "timing_score_at_entry", "rated_at", "opened"):
            if pos.get(key) is None and existing.get(key) is not None:
                pos[key] = existing[key]
        portfolio["positions"] = [pos if p["ticker"] == pos["ticker"] else p for p in portfolio["positions"]]
    else:
        portfolio["positions"].append(pos)
    return save_portfolio(portfolio)


def remove_position(ticker: str) -> dict[str, Any]:
    symbol = _clean_ticker(ticker)
    portfolio = load_portfolio()
    portfolio["positions"] = [p for p in portfolio["positions"] if p["ticker"] != symbol]
    return save_portfolio(portfolio)


def set_cash(amount: float) -> dict[str, Any]:
    portfolio = load_portfolio()
    portfolio["cash"] = amount
    return save_portfolio(portfolio)


def set_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Set the core share and the derivatives budget (a key left out keeps its value, null clears it). Returns the stored portfolio."""
    if not isinstance(plan, dict):
        raise PortfolioError("The plan must be an object")
    portfolio = load_portfolio()
    portfolio["plan"] = clean_plan({**portfolio["plan"], **{k: v for k, v in plan.items() if k in PLAN_KEYS}})
    return save_portfolio(portfolio)


def set_notes(ticker: str, notes: str) -> dict[str, Any]:
    """Write the notes of one position (at most 500 characters; the rest of the position, its source included, stays)."""
    symbol = _clean_ticker(ticker)
    portfolio = load_portfolio()
    pos = next((p for p in portfolio["positions"] if p["ticker"] == symbol), None)
    if pos is None:
        raise LookupError(f"{symbol} is not in the portfolio")
    pos["notes"] = str(notes or "")[:500]
    return save_portfolio(portfolio)


def replace_imported(positions: list[dict[str, Any]], cash: float, currency: str, source: str) -> tuple[dict[str, Any], list[str]]:
    """Swap in the positions of one broker ``source`` and its cash; keep positions added by hand.

    Returns the stored portfolio and notes about what was replaced.
    """
    if source not in BROKER_SOURCES:
        raise PortfolioError(f"Unknown import source: {source}")
    portfolio = load_portfolio()
    notes: list[str] = []
    incoming = [clean_position({**p, "source": source}) for p in positions]
    tickers = {p["ticker"] for p in incoming}
    # One imported account at a time: positions from any broker import are replaced, whichever broker it was.
    previous = {p["ticker"]: p for p in portfolio["positions"] if p.get("source") in BROKER_SOURCES}
    kept = []
    for p in portfolio["positions"]:
        if p.get("source") in BROKER_SOURCES:
            continue
        if p["ticker"] in tickers:
            notes.append(f"{p['ticker']} was added by hand and is now replaced by the imported position.")
            continue
        kept.append(p)
    if kept and portfolio.get("currency") != currency:
        notes.append(f"Check the costs of {', '.join(p['ticker'] for p in kept)}: you typed them in {portfolio.get('currency')} "
                     f"and they now count in {currency}. Edit them in Your holdings.")
    for p in incoming:  # an entry rating recorded earlier for the same holding survives a re-import
        old = previous.get(p["ticker"])
        if old and old.get("opened") == p.get("opened"):
            for key in ("owner_score_at_entry", "owner_decision_at_entry", "timing_score_at_entry", "rated_at", "notes"):
                if p.get(key) in (None, "") and old.get(key) not in (None, ""):
                    p[key] = old[key]
    stored = save_portfolio({"cash": cash, "currency": currency, "positions": kept + incoming, "plan": portfolio["plan"]})
    return stored, notes
