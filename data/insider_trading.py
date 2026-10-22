"""Insider trading provider: SEC Form 4 transactions (via Yahoo Finance) and analytics.

Only open-market purchases (Form 4 code P) and open-market sales (code S) carry
signal. Grants, option exercises, gifts, tax withholding, and unlabelled rows are
classified and counted, but never scored.

Nothing is estimated when data is missing: a provider error or an empty filing
list yields ``InsiderTradingData(available=False, error=...)``.

``analyze_insider_transactions`` is pure (no network) and point-in-time friendly:
it only uses rows dated on or before the reference date (the last OHLCV bar, or
today when no price history is given), and forward returns for the track record
only use bars that exist in the OHLCV passed in.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import yfinance as yf

import config.settings as settings
from data.cache_utils import ttl_cache
from data.yahoo import yahoo_call

logger = logging.getLogger(__name__)

# Transaction categories (SEC Form 4 transaction codes in brackets).
OPEN_MARKET_BUY = "OPEN_MARKET_BUY"  # P
OPEN_MARKET_SALE = "OPEN_MARKET_SALE"  # S
OPTION_EXERCISE = "OPTION_EXERCISE"  # M (also X, C)
GRANT_AWARD = "GRANT_AWARD"  # A
GIFT = "GIFT"  # G
TAX_WITHHOLDING = "TAX_WITHHOLDING"  # F
OTHER = "OTHER"

OPEN_MARKET_CATEGORIES = (OPEN_MARKET_BUY, OPEN_MARKET_SALE)

CATEGORY_CODES: dict[str, str] = {
    OPEN_MARKET_BUY: "P",
    OPEN_MARKET_SALE: "S",
    OPTION_EXERCISE: "M",
    GRANT_AWARD: "A",
    GIFT: "G",
    TAX_WITHHOLDING: "F",
    OTHER: "",
}

CATEGORY_LABELS: dict[str, str] = {
    OPEN_MARKET_BUY: "Open-market buy (P)",
    OPEN_MARKET_SALE: "Open-market sale (S)",
    OPTION_EXERCISE: "Ignored: option exercise (M)",
    GRANT_AWARD: "Ignored: grant/award (A)",
    GIFT: "Ignored: gift (G)",
    TAX_WITHHOLDING: "Ignored: tax withholding (F)",
    OTHER: "Ignored: unclassified",
}

_CODE_TO_CATEGORY: dict[str, str] = {
    "P": OPEN_MARKET_BUY,
    "S": OPEN_MARKET_SALE,
    "M": OPTION_EXERCISE,
    "X": OPTION_EXERCISE,
    "C": OPTION_EXERCISE,
    "A": GRANT_AWARD,
    "G": GIFT,
    "F": TAX_WITHHOLDING,
}

# Analytics parameters.
DEFAULT_LOOKBACK_DAYS = 180
CLUSTER_WINDOW_DAYS = 30  # 2+ distinct buyers within this many calendar days
SALE_WEIGHT = 0.35  # sales are a weak signal (diversification, taxes, 10b5-1 plans)
DRAWDOWN_THRESHOLD = 0.20  # purchases this far below the trailing high get a boost
DRAWDOWN_BOOST = 1.3
DRAWDOWN_LOOKBACK_BARS = 252
TRACK_RECORD_HORIZON_BARS = 60
ADV_BARS = 20
DOLLAR_UNIT = 10_000.0  # log scale: units = log10(1 + dollars / DOLLAR_UNIT)
MIN_RECENCY_WEIGHT = 0.5  # weight of a trade at the far edge of the lookback window

ROLE_WEIGHT_EXECUTIVE = 1.5
ROLE_WEIGHT_OFFICER = 1.1
ROLE_WEIGHT_DIRECTOR = 1.0
ROLE_WEIGHT_OWNER = 0.8
ROLE_WEIGHT_DEFAULT = 1.0

_PRICE_RE = re.compile(
    r"at\s+price\s+\$?\s*([\d,]*\.?\d+)(?:\s*-\s*\$?\s*([\d,]*\.?\d+))?\s*per\s+share",
    re.IGNORECASE,
)
_VICE_PRESIDENT_RE = re.compile(r"\b(vice[\s-]*president|evp|svp|vp)\b", re.IGNORECASE)
_EXECUTIVE_RE = re.compile(r"\b(ceo|cfo|chief|chair\w*|president)\b", re.IGNORECASE)
_OFFICER_RE = re.compile(
    r"\b(officer|general\s+counsel|secretary|treasurer|controller|executive|evp|svp|vp|vice[\s-]*president)\b",
    re.IGNORECASE,
)
_DIRECTOR_RE = re.compile(r"\b(director|trustee)\b", re.IGNORECASE)
_OWNER_RE = re.compile(r"(10\s*%|beneficial\s+owner|ten\s+percent)", re.IGNORECASE)


# --------------------------------------------------------------------------- data


@dataclass
class InsiderTransaction:
    """One Form 4 row, classified and valued."""

    date: pd.Timestamp
    insider: str
    position: str
    category: str
    shares: int
    value: float | None  # dollar value used for weighting (None when it cannot be derived)
    value_source: str  # "reported", "shares x filing price", "shares x close", "unknown"
    price: float | None  # per-share price from the filing text (midpoint of a range)
    text: str
    url: str
    role: str
    role_weight: float
    recency_weight: float = 1.0
    drawdown: float | None = None  # purchases only: fraction below the trailing 252-bar closing high
    forward_return: float | None = None  # purchases only: 60-bar forward return, when history allows

    @property
    def insider_key(self) -> str:
        return " ".join(self.insider.upper().split())

    @property
    def is_open_market(self) -> bool:
        return self.category in OPEN_MARKET_CATEGORIES

    @property
    def code(self) -> str:
        return CATEGORY_CODES.get(self.category, "")

    @property
    def timing_weight(self) -> float:
        if self.category == OPEN_MARKET_BUY and self.drawdown is not None and self.drawdown >= DRAWDOWN_THRESHOLD:
            return DRAWDOWN_BOOST
        return 1.0

    @property
    def dollars(self) -> float:
        return float(self.value) if self.value is not None and self.value > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (keys match the UI's transactions_table)."""
        return {
            "insider": self.insider,
            "position": self.position,
            "type": CATEGORY_LABELS.get(self.category, self.category),
            "category": self.category,
            "shares": int(self.shares),
            "value": None if self.value is None else round(float(self.value), 2),
            "price": None if self.price is None else round(float(self.price), 4),
            "date": self.date.strftime("%Y-%m-%d"),
            "is_open_market": self.is_open_market,
            "url": self.url or None,
            "value_source": self.value_source,
            "role_weight": self.role_weight,
            "drawdown_at_purchase": None if self.drawdown is None else round(float(self.drawdown), 4),
        }


@dataclass
class InsiderTrackRecord:
    """Forward 60-bar returns after one insider's past open-market purchases."""

    insider: str
    purchases_evaluated: int
    hits: int
    avg_forward_return: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.purchases_evaluated if self.purchases_evaluated else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "insider": self.insider,
            "purchases_evaluated": self.purchases_evaluated,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "avg_forward_return": round(self.avg_forward_return, 4),
        }


@dataclass
class InsiderTradingData:
    """Insider activity analytics for a ticker over a lookback window."""

    ticker: str
    available: bool
    error: str | None = None
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    reference_date: pd.Timestamp | None = None
    window_start: pd.Timestamp | None = None
    has_price_history: bool = False

    # Row accounting.
    total_rows: int = 0
    rows_in_window: int = 0
    rows_after_reference: int = 0
    rows_before_window: int = 0
    undated_rows: int = 0
    coverage_start: pd.Timestamp | None = None
    coverage_end: pd.Timestamp | None = None
    latest_transaction_date: pd.Timestamp | None = None  # newest row on or before the reference date
    latest_open_market_date: pd.Timestamp | None = None  # newest open-market row in the window

    # In-window transactions, newest first.
    transactions: list[InsiderTransaction] = field(default_factory=list)

    # Open-market counts and dollars (unweighted).
    total_buy_transactions: int = 0
    total_sell_transactions: int = 0
    ignored_non_open_market_count: int = 0
    ignored_by_category: dict[str, int] = field(default_factory=dict)
    distinct_buyers: int = 0
    distinct_sellers: int = 0
    buy_value: float = 0.0
    sell_value: float = 0.0
    net_buy_value: float = 0.0
    buy_shares: int = 0
    sell_shares: int = 0
    net_buy_shares: int = 0
    unvalued_open_market_count: int = 0

    # Weighted intensities. Buy units are per-insider log-scaled dollars times the
    # dollar-weighted role x recency (x timing) multiplier, summed over insiders.
    # Sell units log-scale the pooled sale total (breadth of selling is not extra signal).
    buy_units: float = 0.0
    buy_units_no_boost: float = 0.0
    sell_units: float = 0.0
    weighted_buy_dollars: float = 0.0
    weighted_buy_dollars_no_boost: float = 0.0
    weighted_sell_dollars: float = 0.0
    avg_dollar_volume_20d: float | None = None

    # Cluster buying.
    cluster_buy_detected: bool = False
    max_buyers_in_cluster_window: int = 0
    cluster_insiders: list[str] = field(default_factory=list)
    cluster_start: pd.Timestamp | None = None
    cluster_end: pd.Timestamp | None = None

    # Timing context.
    avg_drawdown_at_purchase: float | None = None
    buys_with_drawdown: int = 0
    buys_after_drawdown: int = 0

    # Track record (all purchases on or before the reference date with 60 bars of forward data).
    track_records: list[InsiderTrackRecord] = field(default_factory=list)
    buyer_track_evaluated: int = 0  # pooled over insiders who bought inside the window
    buyer_track_hits: int = 0
    buyer_track_avg_return: float | None = None

    @property
    def buy_to_adv(self) -> float | None:
        if not self.avg_dollar_volume_20d:
            return None
        return self.weighted_buy_dollars / self.avg_dollar_volume_20d

    @property
    def buy_to_adv_no_boost(self) -> float | None:
        if not self.avg_dollar_volume_20d:
            return None
        return self.weighted_buy_dollars_no_boost / self.avg_dollar_volume_20d

    @property
    def sell_to_adv(self) -> float | None:
        if not self.avg_dollar_volume_20d:
            return None
        return self.weighted_sell_dollars / self.avg_dollar_volume_20d

    @property
    def net_to_adv(self) -> float | None:
        if not self.avg_dollar_volume_20d:
            return None
        return self.net_buy_value / self.avg_dollar_volume_20d

    @property
    def open_market_count(self) -> int:
        return self.total_buy_transactions + self.total_sell_transactions


# ------------------------------------------------------------------- parsing helpers


def parse_price(text: str | None) -> float | None:
    """Per-share price from Yahoo's Form 4 text ("... at price 12.10 per share."); midpoint for ranges."""
    match = _PRICE_RE.search(text or "")
    if not match:
        return None
    try:
        low = float(match.group(1).replace(",", ""))
        high = float(match.group(2).replace(",", "")) if match.group(2) else low
    except ValueError:
        return None
    price = (low + high) / 2.0
    return price if price > 0 else None


def classify_transaction(text: str | None, code: str | None = "") -> str:
    """
    Classify a Form 4 row into a transaction category.

    An explicit one-letter transaction code wins. Otherwise Yahoo's text is used:
    "Purchase at price ..." is P, "Sale at price ..." is S, "Conversion of Exercise
    of derivative security" is M, "Stock Award(Grant)" is A, "Stock Gift" is G.
    Anything unrecognised (including empty text) is OTHER and never scored.
    """
    raw_code = (code or "").strip().upper()
    if len(raw_code) == 1 and raw_code in _CODE_TO_CATEGORY:
        return _CODE_TO_CATEGORY[raw_code]

    t = f"{raw_code} {text or ''}".lower()
    if "tax" in t or "withh" in t:
        return TAX_WITHHOLDING
    if any(k in t for k in ("exercise", "derivative", "conversion", "option")):
        return OPTION_EXERCISE
    if "gift" in t:
        return GIFT
    if "grant" in t or "award" in t:
        return GRANT_AWARD
    if re.search(r"\b(purchase|buy|bought)\b", t):
        return OPEN_MARKET_BUY
    if re.search(r"\b(sale|sell|sold)\b", t):
        return OPEN_MARKET_SALE
    return OTHER


def role_weight(position: str | None, insider: str | None = "") -> tuple[float, str]:
    """
    Role weight from the Position (and Insider) text.

    CEO / CFO / Chair / President / Chief x1.5, other officers x1.1, directors x1.0,
    10% / beneficial owners x0.8. "Vice President" counts as an officer, not a President.
    """
    text = f"{position or ''} {insider or ''}"
    if _EXECUTIVE_RE.search(_VICE_PRESIDENT_RE.sub(" ", text)):
        return ROLE_WEIGHT_EXECUTIVE, "Executive"
    if _OFFICER_RE.search(text):
        return ROLE_WEIGHT_OFFICER, "Officer"
    if _DIRECTOR_RE.search(text):
        return ROLE_WEIGHT_DIRECTOR, "Director"
    if _OWNER_RE.search(text):
        return ROLE_WEIGHT_OWNER, "10% owner"
    return ROLE_WEIGHT_DEFAULT, "Insider"


def insider_links(ticker: str) -> list[dict[str, str]]:
    """Source links for a ticker's insider filings."""
    symbol = quote(ticker.upper().strip())
    return [
        {
            "label": "SEC EDGAR Form 4 filings",
            "url": (
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                f"&CIK={symbol}&type=4&dateb=&owner=include&count=40"
            ),
        },
        {"label": "OpenInsider screener", "url": f"http://openinsider.com/screener?s={symbol}"},
        {"label": "Yahoo Finance insider transactions", "url": f"https://finance.yahoo.com/quote/{symbol}/insider-transactions/"},
    ]


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_day(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _row_value(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index:
            return row[name]
    return None


def _prepare_prices(ohlcv: pd.DataFrame | None) -> tuple[pd.Series | None, pd.Series | None]:
    """Close and volume series with a tz-naive, day-normalized, sorted index."""
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return None, None
    frame = ohlcv.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    close = close[close > 0]
    if close.empty:
        return None, None
    volume = None
    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce").reindex(close.index)
    return close, volume


def _close_on_or_before(close: pd.Series | None, day: pd.Timestamp) -> tuple[int, float] | None:
    if close is None:
        return None
    pos = int(close.index.searchsorted(day, side="right")) - 1
    if pos < 0:
        return None
    return pos, float(close.iloc[pos])


def _drawdown_at(close: pd.Series | None, day: pd.Timestamp) -> float | None:
    """Fraction the close on ``day`` sits below the trailing 252-bar closing high (0.25 = 25% below)."""
    located = _close_on_or_before(close, day)
    if located is None or close is None:
        return None
    pos, price = located
    trailing_high = float(close.iloc[max(0, pos - DRAWDOWN_LOOKBACK_BARS + 1) : pos + 1].max())
    if trailing_high <= 0:
        return None
    return max(0.0, 1.0 - price / trailing_high)


def _forward_return(close: pd.Series | None, day: pd.Timestamp) -> float | None:
    """Return over the 60 bars after the first bar on or after ``day``; None without enough history."""
    if close is None:
        return None
    start = int(close.index.searchsorted(day, side="left"))
    end = start + TRACK_RECORD_HORIZON_BARS
    if start >= len(close) or end >= len(close):
        return None
    return float(close.iloc[end] / close.iloc[start] - 1.0)


def _avg_dollar_volume(close: pd.Series | None, volume: pd.Series | None) -> float | None:
    if close is None or volume is None:
        return None
    dollar_volume = (close * volume).dropna().tail(ADV_BARS)
    if dollar_volume.empty:
        return None
    adv = float(dollar_volume.mean())
    return adv if adv > 0 else None


def _side_intensity(
    transactions: list[InsiderTransaction],
    with_timing: bool,
    per_insider: bool = True,
) -> tuple[float, float]:
    """
    (units, weighted dollars) for one side.

    Dollars are summed per insider and log-scaled (so a $5M purchase is ~9x a
    $10K one, not 500x, and splitting one order into many rows does not inflate
    it), then multiplied by that insider's dollar-weighted role x recency
    (x drawdown timing) multiplier and summed across insiders.

    With ``per_insider=False`` all rows are pooled into one log-scaled total. Sales
    use this: many insiders each selling a little is routine, not a stronger signal.
    """
    by_insider: dict[str, list[InsiderTransaction]] = {}
    for tx in transactions:
        by_insider.setdefault(tx.insider_key if per_insider else "*", []).append(tx)

    units = 0.0
    weighted_dollars = 0.0
    for items in by_insider.values():
        dollars = sum(t.dollars for t in items)
        if dollars <= 0:
            continue
        multiplier = (
            sum(t.dollars * t.role_weight * t.recency_weight * (t.timing_weight if with_timing else 1.0) for t in items)
            / dollars
        )
        units += multiplier * math.log10(1.0 + dollars / DOLLAR_UNIT)
        weighted_dollars += multiplier * dollars
    return units, weighted_dollars


def _detect_cluster(buys: list[InsiderTransaction]) -> tuple[int, list[str], pd.Timestamp | None, pd.Timestamp | None]:
    """Largest number of distinct buyers inside any rolling 30-calendar-day window."""
    ordered = sorted(buys, key=lambda t: t.date)
    best: tuple[int, list[str], pd.Timestamp | None, pd.Timestamp | None] = (0, [], None, None)
    horizon = pd.Timedelta(days=CLUSTER_WINDOW_DAYS)
    for i, anchor in enumerate(ordered):
        members = [t for t in ordered[i:] if t.date - anchor.date <= horizon]
        names: dict[str, str] = {}
        for t in members:
            names.setdefault(t.insider_key, t.insider)
        if len(names) > best[0]:
            best = (len(names), list(names.values()), anchor.date, max(t.date for t in members))
    return best


# --------------------------------------------------------------------------- public


@ttl_cache(settings.CACHE_TTL_INSIDER)
def fetch_insider_transactions(ticker: str) -> pd.DataFrame | None:
    """
    Raw Form 4 rows from Yahoo Finance (network only, cached per ticker).

    Returns None or an empty frame when Yahoo has no insider data. Provider errors
    propagate so that a transient failure is not cached for hours.
    """
    symbol = ticker.upper().strip()
    frame = yahoo_call(lambda: yf.Ticker(symbol).insider_transactions, f"{symbol} insider")
    if frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"unexpected insider_transactions type {type(frame).__name__}")
    return frame


def analyze_insider_transactions(
    df: pd.DataFrame | None,
    ohlcv: pd.DataFrame | None,
    ticker: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> InsiderTradingData:
    """
    Classify, window, value, and weight insider transactions (pure, no network).

    Args:
        df: Frame shaped like ``yfinance.Ticker(...).insider_transactions``
            (Shares, Value, URL, Text, Insider, Position, Transaction, Start Date).
        ohlcv: Daily OHLCV, oldest first. Its last bar is the reference date. Used
            for dollar volume, drawdown at purchase, close-based valuation, and
            forward returns. May be None (reference date falls back to today).
        ticker: Symbol.
        lookback_days: Calendar days before the reference date to include.
    """
    symbol = ticker.upper().strip()
    lookback_days = max(1, int(lookback_days))
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return InsiderTradingData(ticker=symbol, available=False, error="no insider transactions reported", lookback_days=lookback_days)

    close, volume = _prepare_prices(ohlcv)
    reference = close.index[-1] if close is not None else pd.Timestamp.today().normalize()
    window_start = reference - pd.Timedelta(days=lookback_days)

    data = InsiderTradingData(
        ticker=symbol,
        available=True,
        lookback_days=lookback_days,
        reference_date=reference,
        window_start=window_start,
        has_price_history=close is not None,
        total_rows=int(len(df)),
        avg_dollar_volume_20d=_avg_dollar_volume(close, volume),
    )

    known_rows: list[InsiderTransaction] = []  # every dated row on or before the reference date
    for _, row in df.iterrows():
        day = _to_day(_row_value(row, "Start Date", "Date", "startDate"))
        if day is None:
            data.undated_rows += 1
            continue
        if day > reference:
            data.rows_after_reference += 1
            continue

        text = _clean_str(_row_value(row, "Text"))
        category = classify_transaction(text, _clean_str(_row_value(row, "Transaction")))
        insider = _clean_str(_row_value(row, "Insider")) or "Unknown insider"
        position = _clean_str(_row_value(row, "Position"))
        weight, role = role_weight(position, insider)
        shares_raw = _clean_float(_row_value(row, "Shares"))
        shares = int(abs(shares_raw)) if shares_raw is not None else 0
        price = parse_price(text)

        reported = _clean_float(_row_value(row, "Value"))
        if reported is not None and reported > 0:
            value, source = abs(reported), "reported"
        elif shares > 0 and price is not None:
            value, source = shares * price, "shares x filing price"
        else:
            # Close-based valuation only for trades that were actually priced in the market.
            located = _close_on_or_before(close, day) if shares > 0 and category in OPEN_MARKET_CATEGORIES else None
            if located is not None:
                value, source = shares * located[1], "shares x close"
            else:
                value, source = None, "unknown"
        if price is None and value is not None and shares > 0 and source == "reported":
            price = value / shares

        tx = InsiderTransaction(
            date=day,
            insider=insider,
            position=position,
            category=category,
            shares=shares,
            value=value,
            value_source=source,
            price=price,
            text=text,
            url=_clean_str(_row_value(row, "URL")),
            role=role,
            role_weight=weight,
        )
        if category == OPEN_MARKET_BUY:
            tx.drawdown = _drawdown_at(close, day)
            tx.forward_return = _forward_return(close, day)
        known_rows.append(tx)

    if not known_rows:
        data.available = False
        if data.rows_after_reference:
            data.error = "insider data starts after the reference date"
        else:
            data.error = "insider rows have no usable dates"
        return data

    dates = [t.date for t in known_rows]
    data.coverage_start = min(dates)
    data.coverage_end = max(dates)
    data.latest_transaction_date = max(dates)

    in_window = [t for t in known_rows if t.date >= window_start]
    data.rows_before_window = len(known_rows) - len(in_window)
    data.rows_in_window = len(in_window)
    for tx in in_window:
        age = min(max((reference - tx.date).days, 0), lookback_days)
        tx.recency_weight = 1.0 - (1.0 - MIN_RECENCY_WEIGHT) * age / lookback_days
    in_window.sort(key=lambda t: t.date, reverse=True)
    data.transactions = in_window

    buys = [t for t in in_window if t.category == OPEN_MARKET_BUY]
    sells = [t for t in in_window if t.category == OPEN_MARKET_SALE]
    ignored = [t for t in in_window if not t.is_open_market]

    data.total_buy_transactions = len(buys)
    data.total_sell_transactions = len(sells)
    data.ignored_non_open_market_count = len(ignored)
    counts: dict[str, int] = {}
    for tx in ignored:
        counts[tx.category] = counts.get(tx.category, 0) + 1
    data.ignored_by_category = counts
    data.distinct_buyers = len({t.insider_key for t in buys})
    data.distinct_sellers = len({t.insider_key for t in sells})
    data.buy_value = float(sum(t.dollars for t in buys))
    data.sell_value = float(sum(t.dollars for t in sells))
    data.net_buy_value = data.buy_value - data.sell_value
    data.buy_shares = int(sum(t.shares for t in buys))
    data.sell_shares = int(sum(t.shares for t in sells))
    data.net_buy_shares = data.buy_shares - data.sell_shares
    data.unvalued_open_market_count = sum(1 for t in buys + sells if t.value is None)
    open_market = buys + sells
    if open_market:
        data.latest_open_market_date = max(t.date for t in open_market)

    data.buy_units, data.weighted_buy_dollars = _side_intensity(buys, with_timing=True)
    data.buy_units_no_boost, data.weighted_buy_dollars_no_boost = _side_intensity(buys, with_timing=False)
    data.sell_units, data.weighted_sell_dollars = _side_intensity(sells, with_timing=False, per_insider=False)

    count, members, start, end = _detect_cluster(buys)
    data.max_buyers_in_cluster_window = count
    data.cluster_buy_detected = count >= 2
    if data.cluster_buy_detected:
        data.cluster_insiders, data.cluster_start, data.cluster_end = members, start, end

    drawdowns = [t.drawdown for t in buys if t.drawdown is not None]
    data.buys_with_drawdown = len(drawdowns)
    data.buys_after_drawdown = sum(1 for d in drawdowns if d >= DRAWDOWN_THRESHOLD)
    data.avg_drawdown_at_purchase = float(np.mean(drawdowns)) if drawdowns else None

    # Track record over every purchase on or before the reference date with enough forward bars.
    records: dict[str, list[InsiderTransaction]] = {}
    for tx in known_rows:
        if tx.category == OPEN_MARKET_BUY and tx.forward_return is not None:
            records.setdefault(tx.insider_key, []).append(tx)
    track: list[InsiderTrackRecord] = []
    for items in records.values():
        returns = [float(t.forward_return) for t in items if t.forward_return is not None]
        track.append(
            InsiderTrackRecord(
                insider=items[0].insider,
                purchases_evaluated=len(returns),
                hits=sum(1 for r in returns if r > 0),
                avg_forward_return=float(np.mean(returns)),
            )
        )
    track.sort(key=lambda r: r.purchases_evaluated, reverse=True)
    data.track_records = track
    current_buyers = {t.insider_key for t in buys}
    pooled = [t.forward_return for key in current_buyers for t in records.get(key, []) if t.forward_return is not None]
    data.buyer_track_evaluated = len(pooled)
    data.buyer_track_hits = sum(1 for r in pooled if r > 0)
    data.buyer_track_avg_return = float(np.mean(pooled)) if pooled else None
    return data


def fetch_insider_trading(
    ticker: str,
    ohlcv: pd.DataFrame | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> InsiderTradingData:
    """Fetch Form 4 rows for ``ticker`` and analyze them against ``ohlcv``. Never raises."""
    symbol = ticker.upper().strip()
    try:
        frame = fetch_insider_transactions(symbol)
    except Exception as exc:  # noqa: BLE001 - any provider failure means "no data", never a guess
        logger.warning("Could not fetch insider transactions for %s: %s", symbol, exc)
        detail = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__
        return InsiderTradingData(
            ticker=symbol,
            available=False,
            error=f"insider data provider error: {detail}",
            lookback_days=int(lookback_days),
        )
    try:
        return analyze_insider_transactions(frame, ohlcv, symbol, lookback_days)
    except Exception as exc:  # noqa: BLE001 - malformed provider data must not crash the scan
        logger.exception("Could not analyze insider transactions for %s", symbol)
        return InsiderTradingData(
            ticker=symbol,
            available=False,
            error=f"insider data could not be parsed: {type(exc).__name__}",
            lookback_days=int(lookback_days),
        )
