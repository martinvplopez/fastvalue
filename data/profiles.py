"""Instrument profiles for the portfolio breakdowns: asset type, sector, industry, country and region.

Yahoo's quote profile gives a company's sector, industry and headquarters
country. Funds have no country: their sector mix comes from Yahoo's fund data
(look-through: an MSCI World ETF counts as 30% technology, 16% financials, ...),
and their region is read from the fund's name (S&P 500, MSCI World, Emerging
Markets, ...), labelled as such. Commodity trackers (physical gold or silver,
ETCs) are classed by name, because Yahoo lists some of them as equities.

Sectors are grouped the way Morningstar groups them by sensitivity to the
economic cycle: cyclical (materials, consumer cyclical, financials, real
estate), sensitive (communication, energy, industrials, technology) and
defensive (consumer defensive, healthcare, utilities).

Profiles change rarely, so a successful lookup is cached for 30 days in
data/cache/profiles.json. Only tickers are sent to Yahoo; a failed lookup
falls back to what the name, the ISIN and the broker's asset class tell.
Funds also carry their yearly cost (Yahoo's net expense ratio, a share), when
Yahoo publishes one.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import config.settings as settings

logger = logging.getLogger(__name__)

MAX_AGE_SECONDS = 30 * 24 * 3600
ASSET_TYPES = ("Stock", "ETF", "Fund", "Commodity", "Bond", "Crypto", "Other")
FUND_SECTOR_KEYS = {
    "realestate": "Real Estate", "consumer_cyclical": "Consumer Cyclical", "basic_materials": "Basic Materials",
    "consumer_defensive": "Consumer Defensive", "technology": "Technology",
    "communication_services": "Communication Services", "financial_services": "Financial Services",
    "utilities": "Utilities", "industrials": "Industrials", "energy": "Energy", "healthcare": "Healthcare",
}
SUPER_SECTORS = {
    "Basic Materials": "Cyclical", "Consumer Cyclical": "Cyclical", "Financial Services": "Cyclical",
    "Real Estate": "Cyclical", "Communication Services": "Sensitive", "Energy": "Sensitive",
    "Industrials": "Sensitive", "Technology": "Sensitive", "Consumer Defensive": "Defensive",
    "Healthcare": "Defensive", "Utilities": "Defensive",
}
SUPER_SECTOR_TIPS = {
    "Cyclical": "Materials, consumer cyclical, financials and real estate: profits rise and fall most with the economy.",
    "Sensitive": "Communication, energy, industrials and technology: moderately tied to the economic cycle.",
    "Defensive": "Consumer defensive, healthcare and utilities: demand holds up in recessions.",
}

# MSCI-style regions. Countries not listed fall into "Other".
REGIONS = {
    "North America": ("United States", "Canada"),
    "Europe": ("United Kingdom", "Germany", "France", "Netherlands", "Switzerland", "Denmark", "Sweden", "Norway",
               "Finland", "Spain", "Italy", "Belgium", "Austria", "Portugal", "Ireland", "Luxembourg", "Israel",
               "Jersey", "Guernsey", "Isle of Man"),
    "Developed Asia-Pacific": ("Japan", "Australia", "Hong Kong", "Singapore", "New Zealand", "Macau"),
    "Emerging markets": ("China", "Taiwan", "India", "South Korea", "Brazil", "Mexico", "South Africa", "Indonesia",
                         "Thailand", "Malaysia", "Philippines", "Chile", "Peru", "Colombia", "Poland", "Turkey",
                         "Greece", "Czech Republic", "Hungary", "Saudi Arabia", "United Arab Emirates", "Qatar",
                         "Kuwait", "Egypt", "Argentina", "Uruguay", "Kazakhstan"),
}
_COUNTRY_REGION = {c: region for region, countries in REGIONS.items() for c in countries}
# Fallback when Yahoo has no country: the ISIN prefix is where the security is registered (ADRs show as US).
ISIN_COUNTRY = {
    "US": "United States", "CA": "Canada", "GB": "United Kingdom", "DE": "Germany", "FR": "France",
    "NL": "Netherlands", "CH": "Switzerland", "DK": "Denmark", "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "ES": "Spain", "IT": "Italy", "BE": "Belgium", "AT": "Austria", "PT": "Portugal", "IE": "Ireland",
    "LU": "Luxembourg", "IL": "Israel", "JE": "Jersey", "JP": "Japan", "AU": "Australia", "HK": "Hong Kong",
    "SG": "Singapore", "NZ": "New Zealand", "CN": "China", "TW": "Taiwan", "IN": "India", "KR": "South Korea",
    "BR": "Brazil", "MX": "Mexico", "ZA": "South Africa",
}
# Region of a fund from its name, first match wins: (pattern, region, country or None for several).
FUND_REGION_RULES: tuple[tuple[re.Pattern[str], str, str | None], ...] = (
    (re.compile(r"emerging|\bEM\b|frontier", re.I), "Emerging markets", None),
    (re.compile(r"china|csi ?300", re.I), "Emerging markets", "China"),
    (re.compile(r"\bindia|nifty", re.I), "Emerging markets", "India"),
    (re.compile(r"brazil|taiwan|korea|latin", re.I), "Emerging markets", None),
    (re.compile(r"s&p ?500|nasdaq|russell|dow jones|msci usa|\bu\.?s\.?a?\b|united states|america", re.I),
     "North America", "United States"),
    (re.compile(r"japan|nikkei|topix", re.I), "Developed Asia-Pacific", "Japan"),
    (re.compile(r"pacific|asia", re.I), "Developed Asia-Pacific", None),
    (re.compile(r"\bdax\b|germany", re.I), "Europe", "Germany"),
    (re.compile(r"ftse 100|\buk\b|united kingdom", re.I), "Europe", "United Kingdom"),
    (re.compile(r"stoxx|europe|euro\b|eurozone|\bemu\b|cac 40|aex|ibex", re.I), "Europe", None),
)
GLOBAL_FUND = "Global (funds)"
COMMODITY_RE = re.compile(r"\b(gold|silver|platinum|palladium|copper|commodit\w*|brent|wti|crude)\b", re.I)
# Names that are commodity trackers whatever Yahoo calls them (Xetra-Gold is listed as an equity).
TRACKER_RE = re.compile(r"\b(physical|bullion|etc)\b|xetra-gold|euwax gold", re.I)
MINERS_RE = re.compile(r"\b(miner\w*|mining|producer\w*|royalt\w*)\b", re.I)
CRYPTO_RE = re.compile(r"\b(bitcoin|ethereum|crypto\w*|solana)\b", re.I)
BOND_RE = re.compile(r"\b(bond|bonds|treasury|treasuries|govt|government|aggregate|corporate|gilt|bund)\b", re.I)

_LOCK = threading.Lock()


def _path() -> Path:
    return settings.DATA_DIR / "profiles.json"


def _load_cache() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    with _LOCK:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
        tmp.replace(path)


def region_of(country: str | None) -> str:
    if not country:
        return "Unknown"
    return _COUNTRY_REGION.get(country, "Other")


def fund_region(name: str | None) -> tuple[str, str | None]:
    """(region, country or None) of a fund from its name; global when nothing matches."""
    for pattern, region, country in FUND_REGION_RULES:
        if pattern.search(name or ""):
            return region, country
    return GLOBAL_FUND, None


def _asset_type(quote_type: str | None, name: str, broker_class: str | None, bond_share: float | None) -> str:
    if quote_type == "CRYPTOCURRENCY" or broker_class == "CRYPTO" or (quote_type != "EQUITY" and CRYPTO_RE.search(name)):
        return "Crypto"
    fund_like = quote_type in ("ETF", "MUTUALFUND") or (quote_type is None and broker_class == "FUND")
    if not MINERS_RE.search(name) and (TRACKER_RE.search(name) or (fund_like and COMMODITY_RE.search(name))):
        return "Commodity"
    if broker_class == "BOND" or (bond_share is not None and bond_share >= 0.5) or (
            quote_type in ("ETF", "MUTUALFUND") and BOND_RE.search(name)):
        return "Bond"
    if quote_type == "ETF" or (quote_type is None and broker_class == "FUND"):
        return "ETF"
    if quote_type == "MUTUALFUND":
        return "Fund"
    if quote_type == "EQUITY" or broker_class == "STOCK":
        return "Stock"
    return "Other"


def _expense_ratio(info: dict[str, Any]) -> float | None:
    """Yearly fund cost as a share (Yahoo gives its net expense ratio in percent); None when not published."""
    try:
        value = float(info.get("netExpenseRatio"))
    except (TypeError, ValueError):
        return None
    return round(value / 100.0, 6) if value == value and value >= 0 else None


def build_profile(ticker: str, info: dict[str, Any] | None = None, fund_sectors: dict[str, float] | None = None,
                  bond_share: float | None = None, name: str | None = None, isin: str | None = None,
                  broker_class: str | None = None) -> dict[str, Any]:
    """Profile from Yahoo's quote info (may be empty) plus the hints the portfolio already has."""
    info = info or {}
    quote_type = str(info.get("quoteType") or "").upper() or None
    label = str(info.get("longName") or info.get("shortName") or name or ticker)
    asset_type = _asset_type(quote_type, f"{label} {name or ''}", broker_class, bond_share)
    sector = info.get("sector") or None
    industry = info.get("industry") or None
    country = info.get("country") or None
    if asset_type in ("ETF", "Fund") or (asset_type == "Bond" and quote_type in ("ETF", "MUTUALFUND", None)):
        region, country = fund_region(label)
        country = country or "Several countries (fund)"
    elif asset_type == "Stock":
        if not country and isin:
            country = ISIN_COUNTRY.get(isin[:2])
        region = region_of(country)
    else:
        country = region = f"No country ({asset_type.lower()})"
    sectors = None
    if fund_sectors:
        total = sum(v for v in fund_sectors.values() if v and v > 0)
        if total > 0:
            sectors = {FUND_SECTOR_KEYS.get(k, k.replace("_", " ").title()): round(v / total, 4)
                       for k, v in fund_sectors.items() if v and v > 0}
    return {
        "ticker": ticker,
        "name": label,
        "quote_type": quote_type,
        "asset_type": asset_type,
        "sector": sector if asset_type == "Stock" else None,
        "industry": industry if asset_type == "Stock" else None,
        "country": country or "Unknown",
        "region": region,
        "fund_sectors": sectors,
        "category": info.get("category") or None,
        "expense_ratio": _expense_ratio(info) if asset_type in ("ETF", "Fund", "Bond", "Commodity") else None,
        "source": "yahoo" if info else "name",
    }


def _fetch(ticker: str) -> tuple[dict[str, Any], dict[str, float] | None, float | None]:
    import yfinance as yf

    from data.yahoo import yahoo_call

    tk = yf.Ticker(ticker)
    info = yahoo_call(lambda: tk.info, f"{ticker} profile")
    if not isinstance(info, dict) or not info.get("quoteType"):
        raise LookupError(f"No profile for {ticker}")
    sectors = bond = None
    if str(info.get("quoteType")).upper() in ("ETF", "MUTUALFUND"):
        try:
            fd = yahoo_call(lambda: tk.funds_data, f"{ticker} fund data")
            sectors = dict(fd.sector_weightings or {}) or None
            bond = float((fd.asset_classes or {}).get("bondPosition") or 0.0)
        except Exception as exc:  # noqa: BLE001 - look-through is optional
            logger.info("No fund data for %s: %s", ticker, exc)
    keep = {k: info.get(k) for k in ("quoteType", "longName", "shortName", "sector", "industry", "country", "category",
                                     "netExpenseRatio")}
    return keep, sectors, bond


def _lacks_cost(hit: dict[str, Any]) -> bool:
    """A cached fund from before the cost was kept: looked up once more so the cost shows without waiting 30 days."""
    info = hit.get("info") or {}
    return str(info.get("quoteType")).upper() in ("ETF", "MUTUALFUND") and "netExpenseRatio" not in info


def get_profiles(items: list[dict[str, Any]], max_parallel: int = 4) -> dict[str, dict[str, Any]]:
    """Profiles for ``items`` (dicts with ``ticker`` and optionally ``name``, ``isin``, ``asset_class``)."""
    cache = _load_cache()
    now = time.time()
    out: dict[str, dict[str, Any]] = {}
    todo: list[dict[str, Any]] = []
    for it in items:
        t = it["ticker"]
        hit = cache.get(t)
        if hit and now - float(hit.get("_at", 0)) < MAX_AGE_SECONDS and not _lacks_cost(hit):
            out[t] = build_profile(t, hit.get("info"), hit.get("fund_sectors"), hit.get("bond_share"),
                                   it.get("name"), it.get("isin"), it.get("asset_class"))
        else:
            todo.append(it)

    def work(it: dict[str, Any]) -> tuple[dict[str, Any], tuple[dict[str, Any], dict[str, float] | None, float | None] | None]:
        try:
            return it, _fetch(it["ticker"])
        except Exception as exc:  # noqa: BLE001 - fall back to the name and ISIN
            logger.info("Profile lookup failed for %s: %s", it["ticker"], exc)
            return it, None

    if todo:
        with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="profile") as pool:
            results = list(pool.map(work, todo))
        changed = False
        for it, fetched in results:
            t = it["ticker"]
            if fetched is None:  # an expired profile beats a guess from the name while Yahoo is down
                old = cache.get(t) or {}
                out[t] = build_profile(t, old.get("info"), old.get("fund_sectors"), old.get("bond_share"),
                                       it.get("name"), it.get("isin"), it.get("asset_class"))
                continue
            info, sectors, bond = fetched
            cache[t] = {"info": info, "fund_sectors": sectors, "bond_share": bond, "_at": now}
            changed = True
            out[t] = build_profile(t, info, sectors, bond, it.get("name"), it.get("isin"), it.get("asset_class"))
        if changed:
            try:
                _save_cache(cache)
            except OSError as exc:
                logger.warning("Profile cache not written: %s", exc)
    return out
