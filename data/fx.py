"""Exchange rates and quote currencies, to show holdings from many markets in one currency.

Prices stay in each listing's own currency everywhere in FastValue (the owner's
call: source currency). Only where several holdings are added up, such as the
Portfolio tab's values and weights or a broker account kept in EUR, are they
converted, with these rates:

- ECB euro reference rates (daily, free, no key), as units of a currency per
  1 EUR. Yahoo's ``EUR<CCY>=X`` series is the fallback for a currency the ECB
  does not publish.
- Minor units are handled: Yahoo quotes London in pence (GBp or GBX),
  Johannesburg in cents (ZAc) and Tel Aviv in agorot (ILA).

Rates are cached as Parquet under data/cache/fx/ and refreshed once a day.
Only currency codes leave the machine.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
import urllib.request
from pathlib import Path

import pandas as pd

import config.settings as settings

logger = logging.getLogger(__name__)

ECB_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.{ccy}.EUR.SP00.A?startPeriod=2000-01-01&format=csvdata"
MAX_AGE_SECONDS = 24 * 3600
FFILL_DAYS = 7  # weekends and holidays carry the last rate
# Yahoo minor-unit codes -> (major currency, factor to the major unit)
MINOR_UNITS = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ZAC": ("ZAR", 0.01),
               "ILA": ("ILS", 0.01)}
# When Yahoo does not say: the listing suffix decides.
SUFFIX_CURRENCY = {
    "L": "GBp", "IL": "USD", "DE": "EUR", "F": "EUR", "SG": "EUR", "MU": "EUR", "BE": "EUR", "DU": "EUR",
    "HM": "EUR", "PA": "EUR", "AS": "EUR", "BR": "EUR", "MC": "EUR", "MI": "EUR", "LS": "EUR", "HE": "EUR",
    "VI": "EUR", "IR": "EUR", "TO": "CAD", "V": "CAD", "CN": "CAD", "NE": "CAD", "AX": "AUD", "ST": "SEK",
    "CO": "DKK", "OL": "NOK", "SW": "CHF", "HK": "HKD", "T": "JPY", "SI": "SGD", "NZ": "NZD", "JO": "ZAc",
    "TA": "ILA", "SA": "BRL", "MX": "MXN", "KS": "KRW", "TW": "TWD", "WA": "PLN", "PR": "CZK",
}

_LOCK = threading.Lock()
_MEMO: dict[str, tuple[float, pd.Series]] = {}
# Suffix guesses made while Yahoo did not answer: kept in memory for an hour, never written to the disk cache,
# so one failed lookup cannot pin a wrong currency (a USD-quoted London ETF read as pence) for good.
_GUESSES: dict[str, tuple[float, str]] = {}
GUESS_TTL = 3600.0
# ISO codes accepted after a dash in Yahoo pairs such as BTC-EUR (a one- or two-letter class like BRK-B is not one).
PAIR_CODES = frozenset({"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "HKD", "SGD",
                        "CNY", "INR", "BRL", "KRW", "MXN", "ZAR", "PLN", "CZK", "HUF", "TRY", "ILS"})


class FxError(LookupError):
    pass


def _dir() -> Path:
    path = settings.DATA_DIR / "fx"
    path.mkdir(parents=True, exist_ok=True)
    return path


def major(ccy: str | None) -> tuple[str, float]:
    """('GBp') -> ('GBP', 0.01); ('usd') -> ('USD', 1.0)."""
    code = str(ccy or "").strip()
    if code in MINOR_UNITS:
        return MINOR_UNITS[code]
    return code.upper(), 1.0


def amount_text(value: float, currency: str = "") -> str:
    """A whole amount the way the interface writes it: EUR €1,818, USD $1,818, GBP £1,818, any other code 1,818 CHF."""
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency)
    sign = "-" if round(value) < 0 else ""
    body = f"{abs(value):,.0f}"
    return f"{sign}{symbol}{body}" if symbol else f"{sign}{body} {currency}".strip()


def _fetch_ecb(ccy: str) -> pd.Series:
    req = urllib.request.Request(ECB_URL.format(ccy=ccy), headers={"User-Agent": "fastvalue", "Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        frame = pd.read_csv(io.BytesIO(resp.read()), usecols=["TIME_PERIOD", "OBS_VALUE"])
    s = pd.Series(pd.to_numeric(frame["OBS_VALUE"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(frame["TIME_PERIOD"]), name=ccy).dropna()
    return s[s > 0].sort_index()


def _fetch_yahoo(ccy: str) -> pd.Series:
    from data.price_history import fetch_price_history

    sym = f"EUR{ccy}=X"
    df = fetch_price_history([sym], "2000-01-01").get(sym)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df["close"].rename(ccy)


def per_eur(ccy: str) -> pd.Series:
    """Daily units of ``ccy`` per 1 EUR (major unit). EUR is 1.0. Raises FxError when no source has it."""
    code, _ = major(ccy)
    if code == "EUR":
        return pd.Series(dtype=float, name="EUR")
    now = time.time()
    with _LOCK:
        hit = _MEMO.get(code)
        if hit and now - hit[0] < MAX_AGE_SECONDS:
            return hit[1]
    path = _dir() / f"{code}.parquet"
    series: pd.Series | None = None
    if path.exists() and now - path.stat().st_mtime < MAX_AGE_SECONDS:
        series = pd.read_parquet(path).iloc[:, 0]
    if series is None:
        for fetch in (_fetch_ecb, _fetch_yahoo):
            try:
                series = fetch(code)
            except Exception as exc:  # noqa: BLE001 - try the next source
                logger.info("FX %s from %s failed: %s", code, fetch.__name__, exc)
                series = None
            if series is not None and not series.empty:
                series.to_frame(code).to_parquet(path)
                break
        else:
            if path.exists():  # stale beats nothing
                series = pd.read_parquet(path).iloc[:, 0]
    if series is None or series.empty:
        raise FxError(f"No exchange rate available for {code}")
    with _LOCK:
        _MEMO[code] = (now, series)
    return series


def factor_series(from_ccy: str, to_ccy: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Multiply a price in ``from_ccy`` by this to get ``to_ccy``, for each date (rates carried over gaps)."""
    f_code, f_unit = major(from_ccy)
    t_code, t_unit = major(to_ccy)
    dates = pd.DatetimeIndex(dates)
    if f_code == t_code:
        return pd.Series(f_unit / t_unit, index=dates)

    def aligned(code: str) -> pd.Series:
        if code == "EUR":
            return pd.Series(1.0, index=dates)
        s = per_eur(code)
        # ``dates`` may repeat (two lots bought the same day): build on the unique days, then expand.
        full = s.reindex(s.index.union(dates.unique())).ffill(limit=FFILL_DAYS * 3).bfill(limit=FFILL_DAYS)
        return full.reindex(dates)

    out = aligned(t_code) / aligned(f_code) * (f_unit / t_unit)
    return out


def convert_latest(value: float, from_ccy: str, to_ccy: str) -> float:
    """``value`` in ``from_ccy`` at the latest available rate, in ``to_ccy``."""
    f_code, f_unit = major(from_ccy)
    t_code, t_unit = major(to_ccy)
    if f_code == t_code:
        return value * f_unit / t_unit
    rate_f = 1.0 if f_code == "EUR" else float(per_eur(f_code).iloc[-1])
    rate_t = 1.0 if t_code == "EUR" else float(per_eur(t_code).iloc[-1])
    return value * f_unit / rate_f * rate_t / t_unit


# ------------------------------------------------------------------ quote currency


def _currency_cache() -> Path:
    return _dir() / "quote_currency.json"


def _read_currency_cache() -> dict[str, str]:
    try:
        return json.loads(_currency_cache().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _pair_currency(t: str) -> str | None:
    """``BTC-EUR`` -> EUR (Yahoo's crypto and cross pairs quote in the code after the dash)."""
    if "-" in t and "." not in t:
        code = t.rsplit("-", 1)[1]
        return code if code in PAIR_CODES else None
    return None


def suffix_currency(ticker: str) -> str:
    t = ticker.upper()
    if "." in t:
        return SUFFIX_CURRENCY.get(t.rsplit(".", 1)[1], "USD")
    if t.endswith("=X"):
        return t[3:6] if len(t) >= 8 else "USD"
    return _pair_currency(t) or "USD"


def quote_currency(ticker: str) -> str:
    """Currency Yahoo quotes ``ticker`` in (minor units kept, e.g. GBp).

    US listings (no suffix) are USD and ``XXX-EUR`` pairs are read from the symbol; everything else (suffixes,
    FX pairs, indexes such as ^GDAXI) asks Yahoo once and caches the answer on disk. When Yahoo does not
    answer, the suffix rule is used for an hour without being stored.
    """
    t = ticker.upper()
    pair = _pair_currency(t)
    if pair:
        return pair
    if "." not in t and not t.endswith("=X") and not t.startswith("^"):
        return "USD"  # Yahoo's convention: no suffix is a US listing
    with _LOCK:
        cache = _read_currency_cache()
        guess = _GUESSES.get(t)
    if t in cache:
        return cache[t]
    if guess and time.time() - guess[0] < GUESS_TTL:
        return guess[1]
    ccy = None
    try:
        import yfinance as yf

        from data.yahoo import yahoo_call

        ccy = yahoo_call(lambda: yf.Ticker(t).fast_info.get("currency"), label=f"currency {t}")
    except Exception as exc:  # noqa: BLE001 - the suffix rule is a good fallback
        logger.info("Quote currency lookup failed for %s: %s", t, exc)
    if not ccy:
        guess_ccy = suffix_currency(t)
        with _LOCK:
            _GUESSES[t] = (time.time(), guess_ccy)
        return guess_ccy
    ccy = str(ccy)
    with _LOCK:
        cache = _read_currency_cache()
        cache[t] = ccy
        _GUESSES.pop(t, None)
        _currency_cache().write_text(json.dumps(cache, indent=1, sort_keys=True))
    return ccy


def to_usd(amount: float | None, ticker: str) -> float | None:
    """``amount`` in ``ticker``'s quote currency, in US dollars at the latest ECB rate.

    Liquidity thresholds are in dollars, while a London close is in pence and a Tokyo close in yen. When no
    rate is available the minor unit is still undone (pence to pounds), which is the 100x part of the error.
    """
    if amount is None:
        return None
    ccy = quote_currency(ticker)
    code, unit = major(ccy)
    if code == "USD":
        return amount * unit
    try:
        return convert_latest(amount, ccy, "USD")
    except Exception as exc:  # noqa: BLE001 - FxError, or no rates cached while offline
        logger.info("No %s rate for %s: %s", code, ticker, exc)
        return amount * unit


def set_quote_currency(ticker: str, ccy: str) -> None:
    with _LOCK:
        cache = _read_currency_cache()
        cache[ticker.upper()] = ccy
        _currency_cache().write_text(json.dumps(cache, indent=1, sort_keys=True))

