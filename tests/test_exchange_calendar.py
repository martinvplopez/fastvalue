"""Exchange calendars: holiday rules on known dates, and freshness that uses the listing's own exchange."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.exchange_calendar import EXCHANGES, NYSE, OTHER, easter, exchange_for
from data.fetcher import CLOSE, LIVE, STALE, assess_freshness, is_forming


def _bars(last: str, n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range(end=last, periods=n)
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": np.linspace(1, 2, n), "volume": 1e6}, index=idx)


def _at(tz: str, *args: int) -> datetime:
    return datetime(*args, tzinfo=ZoneInfo(tz))


def test_easter_dates():
    assert easter(2024) == date(2024, 3, 31)
    assert easter(2025) == date(2025, 4, 20)
    assert easter(2026) == date(2026, 4, 5)


def test_nyse_2026_holidays_and_special_closures():
    expected = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
                date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)}
    assert NYSE.holidays(2026) == expected
    assert not NYSE.is_session(date(2025, 1, 9))  # national day of mourning
    assert NYSE.is_session(date(2021, 12, 31))  # New Year's Day 2022 fell on a Saturday: no Friday closure
    assert NYSE.is_session(date(2021, 6, 18))  # Juneteenth only from 2022


def test_european_and_canadian_rules():
    london, xetra, tsx, cph = EXCHANGES["XLON"], EXCHANGES["XETR"], EXCHANGES["XTSE"], EXCHANGES["XCSE"]
    # 2021: Christmas on a Saturday and Boxing Day on a Sunday move to Monday 27 and Tuesday 28.
    assert not london.is_session(date(2021, 12, 27)) and not london.is_session(date(2021, 12, 28))
    assert london.is_session(date(2021, 12, 29))
    assert not london.is_session(date(2022, 9, 19))  # state funeral
    assert not xetra.is_session(date(2026, 12, 24)) and not xetra.is_session(date(2026, 12, 31))
    assert xetra.is_session(date(2026, 10, 2)) and xetra.is_session(date(2026, 5, 25))  # Whit Monday trades
    assert not tsx.is_session(date(2026, 5, 18))  # Victoria Day: the Monday on or before 24 May
    assert not tsx.is_session(date(2026, 10, 12))  # Canadian Thanksgiving
    assert not cph.is_session(date(2023, 5, 5)) and cph.is_session(date(2024, 4, 26))  # Great Prayer Day abolished


def test_suffix_mapping():
    assert exchange_for("AAPL") is NYSE and exchange_for("^VIX") is NYSE
    assert exchange_for("SAP.DE").code == "XETR"
    assert exchange_for("HEIA.AS").code == "XAMS"
    assert exchange_for("IWDA.L").code == "XLON"
    assert exchange_for("0700.HK").code == "XHKG" and not exchange_for("0700.HK").holidays_modeled
    assert exchange_for("EURUSD=X") is OTHER and exchange_for("FOO.ZZ") is OTHER


def test_holiday_is_not_stale():
    # Good Friday 2026: Thursday's bar is the latest session in New York and in Frankfurt.
    df = _bars("2026-04-02")
    assert assess_freshness("AAPL", df, _at("America/New_York", 2026, 4, 3, 18, 0)).status == CLOSE
    assert assess_freshness("SAP.DE", df, _at("Europe/Berlin", 2026, 4, 6, 20, 0)).status == CLOSE  # Easter Monday
    # A normal missing day is still reported.
    stale = assess_freshness("AAPL", _bars("2026-03-31"), _at("America/New_York", 2026, 4, 2, 18, 0))
    assert stale.status == STALE and stale.sessions_behind == 2 and "New York" in stale.note


def test_european_session_uses_its_own_hours():
    df = _bars("2026-09-29")
    # 9:00 in New York is 15:00 in Frankfurt: Xetra is open, so today's bar is forming (never cached).
    ny_morning = _at("America/New_York", 2026, 9, 29, 9, 0)
    assert is_forming("SAP.DE", df, ny_morning)
    assert assess_freshness("SAP.DE", df, ny_morning).status == LIVE
    assert not is_forming("AAPL", df, ny_morning)  # the US bar for today cannot exist before the open
    # 12:00 in New York is 18:00 in Frankfurt: Xetra has closed, the bar is complete.
    ny_noon = _at("America/New_York", 2026, 9, 29, 12, 0)
    assert not is_forming("SAP.DE", df, ny_noon)
    fresh = assess_freshness("SAP.DE", df, ny_noon)
    assert fresh.status == CLOSE and fresh.exchange == "Xetra"


def test_unmodeled_exchange_says_so():
    stale = assess_freshness("0700.HK", _bars("2026-09-25"), _at("Asia/Hong_Kong", 2026, 9, 30, 18, 0))
    assert stale.status == STALE and "not modeled" in stale.note
