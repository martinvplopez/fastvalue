"""Trading calendars for the exchanges FastValue prices from: session hours, time zone and full-day holidays.

Freshness (``data.fetcher.assess_freshness``) uses them to tell a missing
session from an exchange holiday, and the live refresh uses the session hours
to decide whether today's bar is still forming (it is never cached while it is).

The exchange comes from the Yahoo suffix (``SAP.DE`` -> Xetra, no suffix -> NYSE).
Holidays are rules (fixed dates, nth weekdays, Easter offsets, weekend
substitutes) plus the one-off closures since 2011. Half days count as sessions.
Exchanges whose holidays follow a lunar calendar (Hong Kong, Tokyo, ...) and
unknown suffixes have hours but no holiday list (``holidays_modeled`` is
False); there a missing weekday may still be a holiday.

Checked against cached Yahoo bars, 2012 to September 2026 (October 2026):
New York (60 listings), Toronto, Zurich and Lisbon match on every weekday;
the other European rule sets differ only where Yahoo has no bar on a
Christmas Eve or New Year's Eve half day, or a single listing has a stray
bar. Xetra closed on some Whit Mondays and on 3 October before 2022 and has
traded on both since; the rules follow today's practice, which is what
freshness needs. ``tests/test_exchange_calendar.py`` has the fixed dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Callable
from zoneinfo import ZoneInfo

Rule = Callable[[int], set[date]]


def easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Monday 0) of the month; ``n = -1`` is the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = (date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _us_observed(day: date) -> date:
    """Saturday holidays move to Friday, Sunday ones to Monday."""
    return day - timedelta(days=1) if day.weekday() == 5 else day + timedelta(days=1) if day.weekday() == 6 else day


def _substitute_forward(days: list[date]) -> set[date]:
    """UK and Canadian style: a holiday on a weekend moves to the next weekday not already a holiday."""
    out: set[date] = set()
    for d in sorted(days):
        while d.weekday() >= 5 or d in out:
            d += timedelta(days=1)
        out.add(d)
    return out


def _nyse(year: int) -> set[date]:
    days = {
        nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        easter(year) - timedelta(days=2),  # Good Friday
        nth_weekday(year, 5, 0, -1),  # Memorial Day
        _us_observed(date(year, 7, 4)),
        nth_weekday(year, 9, 0, 1),  # Labor Day
        nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _us_observed(date(year, 12, 25)),
    }
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # NYSE does not close on the Friday before a Saturday New Year's Day
        days.add(_us_observed(new_year))
    if year >= 2022:
        days.add(_us_observed(date(year, 6, 19)))  # Juneteenth
    return days


_NYSE_SPECIAL = {date(2012, 10, 29), date(2012, 10, 30), date(2018, 12, 5), date(2025, 1, 9)}  # Sandy, two state funerals


def _xetra(year: int) -> set[date]:
    e = easter(year)
    return {date(year, 1, 1), e - timedelta(days=2), e + timedelta(days=1), date(year, 5, 1),
            date(year, 12, 24), date(year, 12, 25), date(year, 12, 26), date(year, 12, 31)}


def _euronext(year: int) -> set[date]:
    e = easter(year)
    return {date(year, 1, 1), e - timedelta(days=2), e + timedelta(days=1), date(year, 5, 1),
            date(year, 12, 25), date(year, 12, 26)}


def _milan(year: int) -> set[date]:
    return _euronext(year) | {date(year, 8, 15), date(year, 12, 24), date(year, 12, 31)}


def _madrid(year: int) -> set[date]:
    return _euronext(year)


def _london(year: int) -> set[date]:
    e = easter(year)
    days = {e - timedelta(days=2), e + timedelta(days=1), nth_weekday(year, 8, 0, -1)}
    days |= _substitute_forward([date(year, 1, 1)])
    days |= _substitute_forward([date(year, 12, 25), date(year, 12, 26)])
    early_may = {2020: date(2020, 5, 8)}.get(year, nth_weekday(year, 5, 0, 1))  # 2020 moved to VE Day
    spring = {2012: date(2012, 6, 4), 2022: date(2022, 6, 2)}.get(year, nth_weekday(year, 5, 0, -1))  # jubilees
    days |= {early_may, spring}
    return days


_LONDON_SPECIAL = {date(2011, 4, 29), date(2012, 6, 5), date(2022, 6, 3), date(2022, 9, 19), date(2023, 5, 8)}


def _toronto(year: int) -> set[date]:
    e = easter(year)
    victoria = nth_weekday(year, 5, 0, -1)
    if victoria.day > 24:
        victoria -= timedelta(days=7)  # the Monday on or before May 24
    days = {
        nth_weekday(year, 2, 0, 3),  # Family Day
        e - timedelta(days=2),
        victoria,
        nth_weekday(year, 8, 0, 1),  # Civic Holiday
        nth_weekday(year, 9, 0, 1),  # Labour Day
        nth_weekday(year, 10, 0, 2),  # Thanksgiving
    }
    days |= _substitute_forward([date(year, 1, 1)])
    days |= _substitute_forward([date(year, 7, 1)])
    days |= _substitute_forward([date(year, 12, 25), date(year, 12, 26)])
    return days


def _copenhagen(year: int) -> set[date]:
    e = easter(year)
    days = {date(year, 1, 1), e - timedelta(days=3), e - timedelta(days=2), e + timedelta(days=1),
            e + timedelta(days=39), e + timedelta(days=40), e + timedelta(days=50), date(year, 6, 5),
            date(year, 12, 24), date(year, 12, 25), date(year, 12, 26), date(year, 12, 31)}
    if year <= 2023:
        days.add(e + timedelta(days=26))  # Great Prayer Day, abolished from 2024
    return days


def _stockholm(year: int) -> set[date]:
    e = easter(year)
    midsummer_eve = date(year, 6, 19) + timedelta(days=(4 - date(year, 6, 19).weekday()) % 7)  # Friday 19-25 June
    return {date(year, 1, 1), date(year, 1, 6), e - timedelta(days=2), e + timedelta(days=1), date(year, 5, 1),
            e + timedelta(days=39), date(year, 6, 6), midsummer_eve, date(year, 12, 24), date(year, 12, 25),
            date(year, 12, 26), date(year, 12, 31)}


def _helsinki(year: int) -> set[date]:
    e = easter(year)
    midsummer_eve = date(year, 6, 19) + timedelta(days=(4 - date(year, 6, 19).weekday()) % 7)
    return {date(year, 1, 1), date(year, 1, 6), e - timedelta(days=2), e + timedelta(days=1), date(year, 5, 1),
            e + timedelta(days=39), midsummer_eve, date(year, 12, 6), date(year, 12, 24), date(year, 12, 25),
            date(year, 12, 26), date(year, 12, 31)}


def _zurich(year: int) -> set[date]:
    e = easter(year)
    return {date(year, 1, 1), date(year, 1, 2), e - timedelta(days=2), e + timedelta(days=1), date(year, 5, 1),
            e + timedelta(days=39), e + timedelta(days=50), date(year, 8, 1), date(year, 12, 24),
            date(year, 12, 25), date(year, 12, 26), date(year, 12, 31)}


@dataclass(frozen=True)
class Exchange:
    code: str
    name: str
    tz: ZoneInfo
    open: time
    close: time
    rule: Rule | None = None  # None: holidays not modeled
    special: frozenset[date] = frozenset()

    @property
    def holidays_modeled(self) -> bool:
        return self.rule is not None

    def holidays(self, year: int) -> set[date]:
        return _holidays(self.code, year)

    def is_session(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays(day.year)

    def local(self, now: datetime | None = None) -> datetime:
        if now is None:
            return datetime.now(self.tz)
        return now.astimezone(self.tz) if now.tzinfo else now.replace(tzinfo=self.tz)

    def is_open(self, now: datetime | None = None) -> bool:
        t = self.local(now)
        return self.is_session(t.date()) and self.open <= t.time() < self.close

    def last_completed_session(self, now: datetime | None = None) -> date:
        """Most recent session whose regular hours have ended."""
        t = self.local(now)
        day = t.date()
        if self.is_session(day) and t.time() >= self.close:
            return day
        day -= timedelta(days=1)
        while not self.is_session(day):
            day -= timedelta(days=1)
        return day

    def sessions_between(self, start: date, end: date) -> int:
        """Sessions in (start, end]."""
        count, day = 0, start + timedelta(days=1)
        while day <= end:
            count += self.is_session(day)
            day += timedelta(days=1)
        return count


def _ex(code: str, name: str, tz: str, open_: tuple[int, int], close: tuple[int, int], rule: Rule | None = None,
        special: set[date] | None = None) -> Exchange:
    return Exchange(code, name, ZoneInfo(tz), time(*open_), time(*close), rule, frozenset(special or ()))


NYSE = _ex("XNYS", "New York", "America/New_York", (9, 30), (16, 0), _nyse, _NYSE_SPECIAL)
EXCHANGES: dict[str, Exchange] = {e.code: e for e in (
    NYSE,
    _ex("XETR", "Xetra", "Europe/Berlin", (9, 0), (17, 30), _xetra),
    _ex("XFRA", "Frankfurt", "Europe/Berlin", (8, 0), (22, 0), _xetra),
    _ex("XPAR", "Euronext Paris", "Europe/Paris", (9, 0), (17, 30), _euronext),
    _ex("XAMS", "Euronext Amsterdam", "Europe/Amsterdam", (9, 0), (17, 30), _euronext),
    _ex("XBRU", "Euronext Brussels", "Europe/Brussels", (9, 0), (17, 30), _euronext),
    _ex("XLIS", "Euronext Lisbon", "Europe/Lisbon", (8, 0), (16, 30), _euronext),
    _ex("XMIL", "Borsa Italiana", "Europe/Rome", (9, 0), (17, 30), _milan),
    _ex("XMAD", "Madrid", "Europe/Madrid", (9, 0), (17, 30), _madrid),
    _ex("XLON", "London", "Europe/London", (8, 0), (16, 30), _london, _LONDON_SPECIAL),
    _ex("XSWX", "SIX Swiss", "Europe/Zurich", (9, 0), (17, 30), _zurich),
    _ex("XCSE", "Copenhagen", "Europe/Copenhagen", (9, 0), (17, 0), _copenhagen),
    _ex("XSTO", "Stockholm", "Europe/Stockholm", (9, 0), (17, 30), _stockholm),
    _ex("XHEL", "Helsinki", "Europe/Helsinki", (10, 0), (18, 30), _helsinki),
    _ex("XTSE", "Toronto", "America/Toronto", (9, 30), (16, 0), _toronto),
    _ex("XHKG", "Hong Kong", "Asia/Hong_Kong", (9, 30), (16, 0)),
    _ex("XTKS", "Tokyo", "Asia/Tokyo", (9, 0), (15, 30)),
    _ex("XASX", "Sydney", "Australia/Sydney", (10, 0), (16, 0)),
)}
# Unknown listings: weekdays, no holidays, and a session that only ends at midnight UTC, so today's bar is
# never mistaken for a completed one.
OTHER = _ex("OTHER", "Other exchange", "UTC", (0, 0), (23, 59))

SUFFIXES: dict[str, str] = {
    "DE": "XETR", "F": "XFRA", "PA": "XPAR", "AS": "XAMS", "BR": "XBRU", "LS": "XLIS", "MI": "XMIL",
    "MC": "XMAD", "L": "XLON", "IL": "XLON", "SW": "XSWX", "CO": "XCSE", "ST": "XSTO", "HE": "XHEL",
    "TO": "XTSE", "V": "XTSE", "HK": "XHKG", "T": "XTKS", "AX": "XASX",
}


@lru_cache(maxsize=512)
def _holidays(code: str, year: int) -> set[date]:
    ex = EXCHANGES.get(code)
    if ex is None or ex.rule is None:
        return set()
    return ex.rule(year) | {d for d in ex.special if d.year == year}


def exchange_for(ticker: str) -> Exchange:
    """The exchange a Yahoo symbol trades on: by suffix, US without one (FX and futures trade on weekdays)."""
    symbol = (ticker or "").strip().upper()
    if symbol.endswith("=X") or symbol.endswith("=F"):
        return OTHER
    if symbol.startswith("^") or "." not in symbol:
        return NYSE
    return EXCHANGES.get(SUFFIXES.get(symbol.rsplit(".", 1)[1], ""), OTHER)
