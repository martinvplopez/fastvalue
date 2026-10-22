"""Instrument map: ISIN -> Yahoo ticker and quote currency, checked against the prices actually paid.

A broker export books every instrument by ISIN, in EUR. Prices come from Yahoo
by ticker, in the listing's own currency. Each ISIN is resolved once:

1. Candidates from Yahoo's search by ISIN, and by name when the ISIN finds
   nothing (Yahoo misses some share classes).
2. Each candidate is checked against the execution prices in the export: the
   gap is how far the price paid lies outside the listing's daily low-high
   range, converted to EUR at that day's ECB rate. The day before and after
   count too, because Trade Republic trades from 7:30 to 23:00 CET, outside
   the US session and after Asian markets close. The median gap over all
   trades decides: up to 3% the listing fits, above 8% it is the wrong one
   (another share class, another fund, pence read as pounds) and the
   instrument is left unpriced until you set its ticker by hand. A hit for the
   ISIN itself wins when it fits; hits found by name go by the best fit.

Knock-out turbos, private funds and temporary rights have no public price and
are not mapped. A ticker you set by hand is checked the same way and never
replaced. The map is stored in data/store/instruments.json (git-ignored); only
ISINs, names and tickers are sent to Yahoo.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import config.settings as settings

logger = logging.getLogger(__name__)

HISTORY_START = "2008-06-01"  # same start as core.lab, so the shared price cache serves both
OK_GAP = 0.03
CHECK_GAP = 0.08
MAX_CANDIDATES = 4
UNPRICED_CLASSES = {"DERIVATIVE": "knock-out product: no public price history",
                    "PRIVATE_FUND": "private-market fund: left out of the analysis",
                    "SYNTHETIC": "temporary rights: counted with their parent stock",
                    "BOND": "bond: no listing with a price history"}
BOND_CLASSES = ("BOND",)
_BOND_WHY = ("FastValue scores companies and prices stocks, funds and ETFs. A bond has no listing with a price "
             "history on its data source and no filings to score.")
ISIN_NOTE = "Looking for a bond? Bonds are not supported. " + _BOND_WHY
BOND_FOUND_NOTE = "This looks like a bond, and bonds are not supported. " + _BOND_WHY
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
QUOTE_TYPES = ("EQUITY", "ETF", "MUTUALFUND", "INDEX")
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,11}$")

_LOCK = threading.Lock()
Progress = Callable[[int, int, str], None]


def _path() -> Path:
    return settings.STORE_DIR / "instruments.json"


def load_map() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_map(mapping: dict[str, dict[str, Any]]) -> None:
    with _LOCK:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(mapping, indent=1, sort_keys=True))
        tmp.replace(path)


def _clean_name(name: str) -> str:
    """'Berkshire Hathaway (B)' -> 'Berkshire Hathaway'; 'Core S&P 500 USD (Acc)' -> 'Core S&P 500'."""
    text = re.sub(r"\([^)]*\)", " ", name or "")
    text = re.sub(r"\b(USD|EUR|Acc|Dist|Inc|plc|SA|AG|NV|SE|Corp|Corporation|Group|Holdings?)\b\.?", " ", text,
                  flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _search(query: str, retries: int = 2) -> list[str]:
    import time

    import yfinance as yf

    from data.yahoo import yahoo_call

    quotes: list[dict[str, Any]] = []
    for attempt in range(retries + 1):
        try:
            quotes = yahoo_call(lambda: yf.Search(query, max_results=8).quotes, label=f"search {query}") or []
        except Exception as exc:  # noqa: BLE001 - a failed search leaves the instrument unmapped
            logger.info("Yahoo search failed for %s: %s", query, exc)
            quotes = []
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            continue
        # An answer, even an empty one, is final: yfinance keeps Search responses in its own in-process cache,
        # so asking again returns the same quotes (retrying an empty answer used to sleep 9 s for nothing).
        break
    out = []
    for q in quotes or []:
        sym = str(q.get("symbol") or "").upper()
        if TICKER_RE.match(sym) and (q.get("quoteType") or "EQUITY") in QUOTE_TYPES and sym not in out:
            out.append(sym)
    return out


def is_valid_isin(text: str) -> bool:
    """Format (2 letters, 9 letters or digits, 1 digit) and the Luhn check digit over the letters read as 10..35."""
    code = (text or "").strip().upper()
    if not ISIN_RE.match(code):
        return False
    digits = "".join(str(int(ch, 36)) for ch in code)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return total % 10 == 0


def is_bond(asset_class: str | None = None, quote_type: str | None = None) -> bool:
    """A bond only when the broker's asset class or Yahoo's quote type says so; unknown is never a bond."""
    return str(asset_class or "").upper() in BOND_CLASSES or str(quote_type or "").upper() == "BOND"


def suggestion_report(query: str, limit: int = 6) -> dict[str, Any]:
    """The /api/suggest answer: listings plus, for an ISIN or a known bond, a plain note.

    A bond is known from the broker import (its asset class for that ISIN) or a Yahoo quote type; then no listings
    are offered, since none would be the same instrument. Any other ISIN gets the general note only.
    """
    isin = is_valid_isin(query)
    known = (load_map().get(query.strip().upper()) or {}).get("asset_class") if isin else None
    if is_bond(known):
        return {"suggestions": [], "isin": True, "bond": True, "note": BOND_FOUND_NOTE}
    found = _quotes(query, limit)
    if any(is_bond(quote_type=q.get("quoteType")) for q in found):
        return {"suggestions": [], "isin": isin, "bond": True, "note": BOND_FOUND_NOTE}
    return {"suggestions": suggest(query, limit, quotes=found), "isin": isin, "bond": False, "note": ISIN_NOTE if isin else None}


def _quotes(query: str, limit: int) -> list[dict[str, Any]]:
    import yfinance as yf

    from data.yahoo import yahoo_call

    try:
        return yahoo_call(lambda: yf.Search(query, max_results=limit + 4).quotes, label=f"suggest {query}") or []
    except Exception as exc:  # noqa: BLE001 - no suggestions is a fine answer
        logger.info("Yahoo search failed for %s: %s", query, exc)
        return []


def suggest(query: str, limit: int = 6, quotes: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Listings Yahoo's search finds for a symbol, a name or an ISIN: [{ticker, name, exchange, type}].

    For the Analyze tab when a ticker has no data. Only the query leaves the machine.
    """
    if quotes is None:
        quotes = _quotes(query, limit)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    types = QUOTE_TYPES if str(query).strip().startswith("^") else tuple(t for t in QUOTE_TYPES if t != "INDEX")  # index derivatives only when an index was asked for
    for q in quotes:
        sym = str(q.get("symbol") or "").upper()
        if not TICKER_RE.match(sym) or sym in seen or (q.get("quoteType") or "EQUITY") not in types:
            continue
        seen.add(sym)
        out.append({"ticker": sym, "name": str(q.get("longname") or q.get("shortname") or "")[:80],
                    "exchange": str(q.get("exchDisp") or q.get("exchange") or ""), "type": str(q.get("quoteType") or "")})
        if len(out) >= limit:
            break
    return out


def candidates(isin: str, name: str) -> list[tuple[str, str]]:
    """[(ticker, 'isin' or 'name')]: hits for the ISIN, else hits for the cleaned name."""
    found = [(t, "isin") for t in _search(isin)]
    if not found and _clean_name(name):
        found = [(t, "name") for t in _search(_clean_name(name))]
    return found[:MAX_CANDIDATES]


def _gaps(ticker: str, df: pd.DataFrame, trades: list[tuple[str, float]]) -> tuple[np.ndarray, str]:
    """For each trade: relative distance of the price paid from the EUR low-high range of the day before,
    the day of, or the day after the trade (0 when it lies inside one of them)."""
    from data.fx import factor_series, quote_currency
    from data.price_history import as_traded

    ccy = quote_currency(ticker)
    bars = as_traded(df)[["low", "high"]].dropna()
    fx = factor_series(ccy, "EUR", bars.index).to_numpy()
    low, high = bars["low"].to_numpy() * fx, bars["high"].to_numpy() * fx
    index = bars.index
    out = []
    for day, paid in trades:
        pos = int(index.searchsorted(pd.Timestamp(day)))
        best = np.inf
        for i in (pos - 1, pos, pos + 1):
            if 0 <= i < len(index) and abs((index[i] - pd.Timestamp(day)).days) <= 4 and paid > 0:
                lo, hi = low[i], high[i]
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    continue
                gap = 0.0 if lo <= paid <= hi else min(abs(lo / paid - 1.0), abs(hi / paid - 1.0))
                best = min(best, gap)
        out.append(best)
    return np.array(out, dtype=float), ccy


def check_ticker(ticker: str, trades: list[tuple[str, float]], df: pd.DataFrame | None) -> dict[str, Any]:
    """Median gap between the prices paid (date, EUR price) and the ticker's EUR daily range."""
    if df is None or df.empty:
        return {"status": "no_prices", "median_gap": None, "trades": len(trades), "currency": None}
    try:
        gaps, ccy = _gaps(ticker, df, trades)
    except Exception as exc:  # noqa: BLE001 - FX outage: report rather than guess
        return {"status": "no_prices", "median_gap": None, "trades": len(trades), "currency": None, "error": str(exc)}
    gaps = gaps[np.isfinite(gaps)]
    if not len(gaps):
        status, median = ("no_trades" if not trades else "no_prices"), None
    else:
        median = float(np.median(gaps))
        status = "ok" if median <= OK_GAP else "check" if median <= CHECK_GAP else "mismatch"
    first = df.index[0].strftime("%Y-%m-%d") if len(df) else None
    return {"status": status, "median_gap": round(median, 4) if median is not None else None,
            "trades": len(trades), "currency": ccy, "history_from": first}


def _entry(isin: str, info: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"isin": isin, "name": info.get("name"), "asset_class": info.get("asset_class"), "ticker": None,
            "currency": None, "source": "none", "status": "unmapped", "median_gap": None, "trades": 0,
            "candidates": [], "updated": datetime.now().isoformat(timespec="seconds"), **fields}


def usable(entry: dict[str, Any] | None) -> bool:
    """Price this instrument? Searched tickers must pass the check; a ticker set by hand is trusted."""
    if not entry or not entry.get("ticker"):
        return False
    return entry.get("source") == "manual" or entry.get("status") in ("ok", "check", "no_trades")


def resolve(instruments: dict[str, dict[str, Any]], trades: dict[str, list[tuple[str, float]]],
            isins: list[str] | None = None, force: bool = False, progress: Progress | None = None) -> dict[str, dict[str, Any]]:
    """Map ``isins`` (default: all) that are not mapped yet. Returns the whole, saved map."""
    from data.price_history import fetch_price_history

    progress = progress or (lambda *_: None)
    mapping = load_map()
    todo = []
    for isin in isins if isins is not None else list(instruments):
        info = instruments.get(isin, {})
        cls = info.get("asset_class")
        if cls in UNPRICED_CLASSES:
            if isin not in mapping or mapping[isin].get("source") != "manual":
                mapping[isin] = _entry(isin, info, status="not_priced", reason=UNPRICED_CLASSES[cls])
            continue
        current = mapping.get(isin)
        if current and not force and (current.get("source") == "manual" or current.get("status") in ("ok", "check")):
            continue
        todo.append(isin)

    found: dict[str, list[tuple[str, str]]] = {}
    for n, isin in enumerate(todo, 1):
        progress(n, len(todo), f"Finding tickers ({n}/{len(todo)})")
        found[isin] = candidates(isin, instruments.get(isin, {}).get("name", ""))
    tickers = sorted({t for c in found.values() for t, _ in c})
    progress(0, 1, f"Loading prices for {len(tickers)} listings")
    prices = fetch_price_history(tickers, HISTORY_START) if tickers else {}
    for n, isin in enumerate(todo, 1):
        progress(n, len(todo), f"Checking listings against your trade prices ({n}/{len(todo)})")
        info = instruments.get(isin, {})
        results = []
        for t, origin in found[isin]:
            res = check_ticker(t, trades.get(isin, []), prices.get(t))
            results.append({"ticker": t, "origin": origin, **res})
        priced = [r for r in results if r["median_gap"] is not None]
        if priced:
            # The first hit for the ISIN (usually the primary listing) wins when it fits; otherwise the best fit.
            best = min(priced, key=lambda r: r["median_gap"])
            primary = priced[0]
            pick = primary if primary["origin"] == "isin" and primary["median_gap"] <= OK_GAP else best
        else:
            pick = next((r for r in results if r["status"] == "no_trades"), None)
        if pick is None:
            mapping[isin] = _entry(isin, info, candidates=[r["ticker"] for r in results],
                                   reason="no listing with prices found" if results else "Yahoo found no listing")
            continue
        mapping[isin] = _entry(isin, info, ticker=pick["ticker"], currency=pick.get("currency"), source="search",
                               status=pick["status"], median_gap=pick["median_gap"], trades=pick["trades"],
                               history_from=pick.get("history_from"),
                               candidates=[{"ticker": r["ticker"], "median_gap": r["median_gap"]} for r in results])
    save_map(mapping)
    return mapping


def set_manual(isin: str, ticker: str | None, info: dict[str, Any], trades: list[tuple[str, float]]) -> dict[str, Any]:
    """Set (or with ``ticker=None`` clear) the ticker of one ISIN by hand; it is checked and then kept."""
    from data.price_history import fetch_price_history

    mapping = load_map()
    if not ticker:
        mapping.pop(isin, None)
        save_map(mapping)
        return _entry(isin, info)
    sym = ticker.strip().upper()
    if not TICKER_RE.match(sym):
        raise ValueError(f"Invalid ticker symbol: {sym!r}")
    df = fetch_price_history([sym], HISTORY_START).get(sym)
    res = check_ticker(sym, trades, df)
    entry = _entry(isin, info, ticker=sym, currency=res.get("currency"), source="manual", status=res["status"],
                   median_gap=res["median_gap"], trades=res["trades"], history_from=res.get("history_from"))
    mapping[isin] = entry
    save_map(mapping)
    return entry


def trade_prices(ledger: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    """Execution prices per ISIN from the ledger's orders: [(date, EUR price)]."""
    out: dict[str, list[tuple[str, float]]] = {}
    for o in ledger.get("orders", []):
        if o.get("price"):
            out.setdefault(o["isin"], []).append((o["date"], float(o["price"])))
    return out
