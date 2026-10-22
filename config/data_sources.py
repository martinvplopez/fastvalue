"""Where FastValue looks things up, and what each lookup sends.

The "Your data and costs" panel in the footer is built from this list, so the copy cannot drift from the code:
``tests/test_data_sources.py`` fails when a fetcher under ``data/`` or ``core/`` calls a host that is not listed here.
Only instrument identifiers (tickers, ISINs, company or fund names) are ever sent; holdings, amounts, cash, notes and
the broker file never leave the computer. ``optional`` sources are off until the owner sets a key or a switch.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DATA_SOURCES: list[dict[str, Any]] = [
    {"name": "Yahoo Finance", "host": "finance.yahoo.com", "optional": False,
     "sends": "tickers, ISINs and company names",
     "why": "prices, fundamentals, options, news and listings"},
    {"name": "SEC EDGAR", "host": "data.sec.gov", "optional": False,
     "sends": "company identifiers, and your SEC contact email in the request header if you set one",
     "why": "filings, financial statements and insider trades"},
    {"name": "SEC EDGAR (documents)", "host": "www.sec.gov", "optional": False,
     "sends": "company identifiers, and your SEC contact email in the request header if you set one",
     "why": "the text of annual reports and the company index"},
    {"name": "FRED", "host": "fred.stlouisfed.org", "optional": False,
     "sends": "series names such as the VIX and interest rates, nothing about you",
     "why": "the market regime"},
    {"name": "European Central Bank", "host": "data-api.ecb.europa.eu", "optional": False,
     "sends": "currency codes",
     "why": "exchange rates into euros"},
    {"name": "S&P 500 constituents list (GitHub)", "host": "raw.githubusercontent.com", "optional": False,
     "sends": "nothing but the request for a public file",
     "why": "which companies were in the S&P 500 for the studies"},
    {"name": "Company announcements", "host": "mfn.se", "optional": False,
     "sends": "company names", "why": "regulatory news for Swedish companies"},
    {"name": "Cision", "host": "news.cision.com", "optional": False,
     "sends": "company names", "why": "press releases for Swedish companies"},
    {"name": "ASX", "host": "www.asx.com.au", "optional": False,
     "sends": "ticker codes", "why": "announcements for Australian companies"},
    {"name": "Investegate", "host": "www.investegate.co.uk", "optional": False,
     "sends": "ticker codes", "why": "announcements for UK companies"},
    {"name": "London Stock Exchange", "host": "www.londonstockexchange.com", "optional": False,
     "sends": "nothing: a link you can open yourself", "why": "announcements for UK companies"},
    {"name": "SEDAR+", "host": "www.sedarplus.ca", "optional": False,
     "sends": "nothing: a link you can open yourself", "why": "filings for Canadian companies"},
    {"name": "Alpha Vantage", "host": "www.alphavantage.co", "optional": True,
     "sends": "tickers, and your free key", "why": "more news sources"},
    {"name": "Finnhub", "host": "finnhub.io", "optional": True,
     "sends": "tickers, and your free key", "why": "more news sources"},
    {"name": "StockTwits", "host": "api.stocktwits.com", "optional": True,
     "sends": "tickers", "why": "social tone, off unless you switch it on"},
    {"name": "Reddit", "host": "www.reddit.com", "optional": True,
     "sends": "tickers", "why": "social tone, off unless you switch it on"},
    {"name": "Reddit (sign-in)", "host": "oauth.reddit.com", "optional": True,
     "sends": "tickers, and your Reddit app key", "why": "social tone, off unless you switch it on"},
    {"name": "Anthropic", "host": "api.anthropic.com", "optional": True,
     "sends": "public social posts, and your key", "why": "scoring social posts, off unless you switch it on"},
]

HOST_RE = re.compile(r"https://([A-Za-z0-9.-]+)")


def _version() -> str:
    """The version in pyproject.toml, so the panel does not carry a second copy of it."""
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return found.group(1) if found else ""


VERSION = _version()


def hosts() -> set[str]:
    return {s["host"] for s in DATA_SOURCES}
