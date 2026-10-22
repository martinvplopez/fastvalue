"""Official announcements and press releases of a listed company, from the exchange or regulator of its market.

One interface for every market: ``announcements(ticker, limit)`` returns the latest items as
``{date, title, kind, url, source}`` plus the status of each source that was tried. Only the list is read (date,
headline, link to the official document). Figures are never extracted (decision_log.md D-37, D-38).

Sources, in the order they are tried (decision_log.md D-38):

- US: SEC EDGAR (through ``data.sec_reports``: 10-K and 10-Q as reports, 8-K as regulatory, and the EX-99 exhibits of an
  8-K as press releases). Backup: the company's investor relations site (link only).
- Sweden (.ST): MFN (mfn.se), found by the issuer's ticker through MFN's own company search. MFN does not mark which
  items are regulatory, so its items are ``announcement``. Backup: Cision (news.cision.com) for the same issuer, then
  the issuer's own site.
- Australia (.AX): ASX market announcements by ASX code. Backup: the ASX announcements page (link).
- Canada (.TO, .V, .CN, .NE): SEDAR+ cannot be read automatically (a session and a captcha stand in front of the
  search), so it is not worked around: the status says so and links to the search. The issuer's own site is the backup
  (link only), because no newswire can be matched to an issuer reliably from a ticker.
- United Kingdom (.L): RNS as republished by Investegate (the London Stock Exchange news pages are closed to
  automated readers by their robots.txt). Backup: the London Stock Exchange news search (link).

Rules:

- Never fill a field that was not read. An issuer is matched only by an exact ticker (MFN, Investegate, ASX); two
  candidates give ``ambiguous``, none gives ``issuer_not_found``. Nothing is matched by name.
- A source's robots.txt is honoured, there are no logins and no bot-wall workarounds. The User-Agent names the app and
  never carries the SEC contact email (that goes to sec.gov only).
- One request per source per view. A successful read is cached for 15 minutes; a failure is never cached.

Statuses of a source: ``ok``, ``issuer_not_found``, ``ambiguous``, ``unreachable``, ``not_readable``, ``no_access``
(SEC contact email missing) and, for a market without an adapter, ``no_adapter``.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from data.cache_utils import ttl_cache

logger = logging.getLogger(__name__)

USER_AGENT = "FastValue/1.0 (personal research tool; reads public announcement lists)"
TIMEOUT = 15
MAX_BYTES = 3_000_000
CACHE_SECONDS = 900
DEFAULT_LIMIT = 20

MARKETS: dict[str, dict[str, Any]] = {
    "US": {"name": "United States", "suffixes": (), "chain": ("sec",)},
    "SE": {"name": "Sweden", "suffixes": (".ST",), "chain": ("mfn", "cision")},
    "AU": {"name": "Australia", "suffixes": (".AX",), "chain": ("asx",)},
    "CA": {"name": "Canada", "suffixes": (".TO", ".V", ".CN", ".NE"), "chain": ("sedar",)},
    "UK": {"name": "United Kingdom", "suffixes": (".L",), "chain": ("investegate", "lse")},
}
COVERED = "United States (SEC EDGAR), Sweden (MFN), Australia (ASX), Canada (SEDAR+, link only) and the United Kingdom (RNS)"
NO_ADAPTER = f"FastValue reads official announcements for {COVERED}. This company is listed elsewhere."

REPORT_WORDS = re.compile(
    r"interim report|year.end report|annual report|quarterly report|quarterly activities|half.?year|full.?year|financial statements|"
    r"delårsrapport|bokslutskommuniké|årsredovisning|kvartalsrapport|halvårsrapport|\bappendix 4[cde]\b|"
    r"\bq[1-4]\b.{0,20}(results|report)|(final|interim|preliminary) results",
    re.I,
)


class SourceError(Exception):
    """A source could not be read (network, a server error, or a protection we do not work around)."""


class NotFound(SourceError):
    """The server answered that the page does not exist."""


# ------------------------------------------------------------------ results


@dataclass
class Result:
    source: str
    status: str = "ok"
    message: str | None = None
    fix: str | None = None
    items: list[dict[str, Any]] = field(default_factory=list)
    issuer: str | None = None
    url: str | None = None  # the issuer's page at this source, or its search page
    notes: list[str] = field(default_factory=list)

    def attempt(self) -> dict[str, Any]:
        return {"source": self.source, "status": self.status, "message": self.message}


def _item(date: str | None, title: str, kind: str, url: str, source: str, **extra: Any) -> dict[str, Any]:
    return {"date": date, "title": title, "kind": kind, "url": url, "source": source, **extra}


NOT_A_REPORT = re.compile(r"invitation|inbjudan|notice of|conference|webcast|presentation|\bcall\b|briefing", re.I)


def _kind_from_title(title: str, default: str) -> str:
    return "report" if REPORT_WORDS.search(title) and not NOT_A_REPORT.search(title) else default


# ------------------------------------------------------------------ markets


def market_of(ticker: str) -> str | None:
    """The market code of a Yahoo ticker from its suffix: US when there is none, None when no adapter exists."""
    symbol = ticker.strip().upper()
    if "." not in symbol:
        return "US"
    suffix = "." + symbol.rsplit(".", 1)[1]
    for code, info in MARKETS.items():
        if suffix in info["suffixes"]:
            return code
    return "US" if len(suffix) == 2 else None  # BRK.B style share classes are US tickers


def _base_symbol(ticker: str) -> str:
    return ticker.strip().upper().rsplit(".", 1)[0]


# ------------------------------------------------------------------ http


_robots: dict[str, tuple[float, urllib.robotparser.RobotFileParser | None]] = {}


def _raw_get(url: str, accept: str = "text/html,application/json") -> tuple[str, str]:
    """(final url, text) of one GET. Raises NotFound for 404 and SourceError for anything else that fails."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    last: Exception | None = None
    for attempt in range(2):  # one retry, only for a server error
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - fixed https hosts
                data = response.read(MAX_BYTES + 1)
                return response.geturl(), data[:MAX_BYTES].decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NotFound(f"HTTP 404 from {urllib.parse.urlsplit(url).netloc}") from exc
            last = exc
            if exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            break
        if attempt == 0:
            time.sleep(1.0)
    raise SourceError(f"{urllib.parse.urlsplit(url).netloc} did not answer ({type(last).__name__}: {str(last)[:80]})")


def _robots_allow(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    host = f"{parts.scheme}://{parts.netloc}"
    hit = _robots.get(host)
    if hit is None or time.monotonic() - hit[0] > 86400:
        parser: urllib.robotparser.RobotFileParser | None = urllib.robotparser.RobotFileParser()
        try:
            _, text = _raw_get(f"{host}/robots.txt", "text/plain")
            parser.parse(text.splitlines())
        except NotFound:
            parser = None  # no robots.txt: nothing is disallowed
        except SourceError:
            parser = None
        _robots[host] = hit = (time.monotonic(), parser)
    parser = hit[1]
    return parser is None or parser.can_fetch(USER_AGENT, url)


def _fetch(url: str, accept: str = "text/html,application/json") -> tuple[str, str]:
    """``_raw_get`` after a robots.txt check. Tests replace this function."""
    if not _robots_allow(url):
        raise SourceError(f"{urllib.parse.urlsplit(url).netloc} asks automated readers not to open this page")
    return _raw_get(url, accept)


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _unreachable(source: str, exc: SourceError, url: str | None = None) -> Result:
    return Result(source, "unreachable", f"{source} could not be reached ({exc}).", "Try again in a minute.", url=url)


# ------------------------------------------------------------------ Sweden: MFN, then Cision

MFN = "MFN"
MFN_HOME = "https://mfn.se/all/s/nordic"
MFN_MICS = ("XSTO", "XNGM", "FNSE")  # Nasdaq Stockholm, Nasdaq First North, other Swedish markets


def mfn_ticker(ticker: str) -> str:
    """MFN's form of a Yahoo ticker: VOLV-B.ST becomes VOLV B."""
    return _base_symbol(ticker).replace("-", " ").replace(".", " ").strip()


def match_mfn_issuer(companies: list[dict[str, Any]], ticker: str) -> tuple[str, dict[str, Any] | list[dict[str, Any]] | None]:
    """('ok', company), ('ambiguous', candidates) or ('none', None). Only an exact exchange ticker counts."""
    wanted = mfn_ticker(ticker).upper()
    hits = []
    for company in companies:
        refs = {str(t).upper() for t in company.get("tickers") or []}
        if any(f"{mic}:{wanted}" in refs for mic in MFN_MICS):
            hits.append(company)
    if len(hits) == 1:
        return "ok", hits[0]
    return ("ambiguous", hits) if hits else ("none", None)


@ttl_cache(CACHE_SECONDS)
def _mfn_companies(query: str) -> list[dict[str, Any]]:
    import json

    url = "https://mfn.se/search/companies?" + urllib.parse.urlencode({"limit": 10, "query": query})
    _, text = _fetch(url, "application/json")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise SourceError("MFN's company search returned something unexpected") from exc
    return [c for c in data if isinstance(c, dict)] if isinstance(data, list) else []


def parse_mfn_feed(html: str, limit: int) -> list[dict[str, Any]]:
    """Items of an MFN issuer page: date, headline and the link to the release on MFN."""
    items: list[dict[str, Any]] = []
    for block in html.split('class="short-item')[1:]:
        date = re.search(r'class="compressed-date">\s*(\d{4}-\d{2}-\d{2})', block)
        link = re.search(r'<a class="title-link item-link" href="([^"]+)"[^>]*title="([^"]*)"', block)
        if not link:
            continue
        href = html_lib.unescape(link.group(1))
        url = href if href.startswith("http") else f"https://mfn.se{href}"
        title = _text(link.group(2))
        items.append(_item(date.group(1) if date else None, title, _kind_from_title(title, "announcement"), url, MFN))
        if len(items) >= limit:
            break
    return items


@ttl_cache(CACHE_SECONDS)
def _mfn_feed(slug: str, limit: int) -> list[dict[str, Any]]:
    _, text = _fetch(f"https://mfn.se/all/a/{urllib.parse.quote(slug)}")
    return parse_mfn_feed(text, limit)


def source_mfn(ticker: str, limit: int, ctx: dict[str, Any]) -> Result:
    query = mfn_ticker(ticker).split(" ")[0]
    try:
        companies = _mfn_companies(query)
    except SourceError as exc:
        return _unreachable(MFN, exc, MFN_HOME)
    state, found = match_mfn_issuer(companies, ticker)
    if state == "none":
        return Result(MFN, "issuer_not_found", f"MFN has no issuer with the ticker {mfn_ticker(ticker)}.",
                      "Search the company on MFN by name.", url=MFN_HOME)
    if state == "ambiguous":
        names = ", ".join(str(c.get("name")) for c in found)  # type: ignore[union-attr]
        return Result(MFN, "ambiguous", f"More than one MFN issuer has the ticker {mfn_ticker(ticker)} ({names}), so none was chosen.",
                      "Search the company on MFN by name.", url=MFN_HOME)
    slug, name = str(found["slug"]), str(found.get("name") or found["slug"])  # type: ignore[index]
    ctx.update(slug=slug, name=name)
    page = f"https://mfn.se/all/a/{slug}"
    try:
        items = _mfn_feed(slug, limit)
    except SourceError as exc:
        return _unreachable(MFN, exc, page)
    result = Result(MFN, issuer=name, url=page, items=items,
                    notes=["MFN does not mark which releases are regulatory, so they are listed as announcements."])
    if not items:
        result.message = "MFN lists no releases for this issuer."
    return result


CISION = "Cision"


def parse_cision_feed(html: str, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in re.findall(r"<article>(.*?)</article>", html, flags=re.S | re.I):
        link = re.search(r'<a href="([^"]+)"[^>]*>\s*<h2>(.*?)</h2>', block, flags=re.S)
        when = re.search(r'datetime="(\d{4}-\d{2}-\d{2})', block)
        if not link:
            continue
        title = _text(link.group(2))
        items.append(_item(when.group(1) if when else None, title, _kind_from_title(title, "announcement"),
                           html_lib.unescape(link.group(1)), CISION))
        if len(items) >= limit:
            break
    return items


@ttl_cache(CACHE_SECONDS)
def _cision_page(slug: str) -> tuple[str, str]:
    """(page title, html) of an issuer's Cision newsroom."""
    _, text = _fetch(f"https://news.cision.com/se/{urllib.parse.quote(slug)}")
    title = re.search(r"<title>([^<]*)", text)
    return _text(title.group(1)) if title else "", text


def source_cision(ticker: str, limit: int, ctx: dict[str, Any]) -> Result:
    slug, name = ctx.get("slug"), ctx.get("name")
    if not slug:
        return Result(CISION, "issuer_not_found", "The issuer could not be identified, so no Cision newsroom was tried.",
                      "Search the company on Cision.", url="https://news.cision.com/se/")
    page = f"https://news.cision.com/se/{slug}"
    try:
        title, text = _cision_page(slug)
    except NotFound:
        return Result(CISION, "issuer_not_found", f"Cision has no newsroom at the address of {name}.", "Search the company on Cision.", url="https://news.cision.com/se/")
    except SourceError as exc:
        return _unreachable(CISION, exc, page)
    if title.casefold() != str(name).casefold():  # same address, different company: do not guess
        return Result(CISION, "issuer_not_found", f"The Cision newsroom at that address is called {title or 'unnamed'}, not {name}, so it was not used.",
                      "Search the company on Cision.", url="https://news.cision.com/se/")
    return Result(CISION, issuer=name, url=page, items=parse_cision_feed(text, limit))


# ------------------------------------------------------------------ Australia: ASX

ASX = "ASX"


def asx_page(code: str) -> str:
    return f"https://www.asx.com.au/markets/trade-our-cash-market/announcements.{code.lower()}"


def parse_asx_announcements(html: str, limit: int) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """(code, company name, items) from the ASX announcements page. Dates are day/month/year on the page."""
    head = re.search(r"Company announcements for\s*(?:<br>)?\s*(.*?)\s*\(([A-Z0-9]+)\)", html, flags=re.S)
    name, code = (_text(head.group(1)), head.group(2)) if head else (None, None)
    items: list[dict[str, Any]] = []
    for row in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        date = re.search(r"(\d{2})/(\d{2})/(\d{4})", row)
        link = re.search(r'<a[^>]*href="([^"]*displayAnnouncement[^"]*)"[^>]*>\s*(.*?)<br>', row, flags=re.S)
        if not date or not link:
            continue
        title = _text(link.group(2))
        url = "https://www.asx.com.au" + html_lib.unescape(link.group(1))
        sensitive = "icon-price-sensitive" in row
        items.append(_item(f"{date.group(3)}-{date.group(2)}-{date.group(1)}", title, _kind_from_title(title, "regulatory"),
                           url, ASX, price_sensitive=sensitive))
        if len(items) >= limit:
            break
    return code, name, items


@ttl_cache(CACHE_SECONDS)
def _asx_page(code: str) -> str:
    query = urllib.parse.urlencode({"by": "asxCode", "asxCode": code, "timeframe": "D", "period": "M6"})
    return _fetch(f"https://www.asx.com.au/asx/v2/statistics/announcements.do?{query}")[1]


def source_asx(ticker: str, limit: int, ctx: dict[str, Any]) -> Result:
    code = _base_symbol(ticker)
    link = asx_page(code)
    try:
        text = _asx_page(code)
    except SourceError as exc:
        return _unreachable(ASX, exc, link)
    found, name, items = parse_asx_announcements(text, limit)
    if found is None:
        return Result(ASX, "issuer_not_found", f"ASX has no company with the code {code}.", "Check the code on the ASX announcements page.", url=link)
    if found != code:
        return Result(ASX, "ambiguous", f"ASX answered for the code {found} when asked for {code}, so nothing was used.",
                      "Check the code on the ASX announcements page.", url=link)
    result = Result(ASX, issuer=name, url=link, items=items,
                    notes=["The ASX list covers the last six months. All items are continuous disclosure announcements."])
    if not items:
        result.message = "ASX lists no announcements for this company in the last six months."
    return result


# ------------------------------------------------------------------ Canada: SEDAR+ (not readable)

SEDAR = "SEDAR+"
SEDAR_URL = "https://www.sedarplus.ca/landingpage/"


def source_sedar(ticker: str, limit: int, ctx: dict[str, Any]) -> Result:
    return Result(SEDAR, "not_readable", "SEDAR+ cannot be read automatically: its search runs behind a session and a captcha, and FastValue does not work around that.",
                  "Search the issuer on SEDAR+ yourself. Press releases are usually also on the company's own investor page.", url=SEDAR_URL)


# ------------------------------------------------------------------ United Kingdom: RNS via Investegate

INVESTEGATE = "Investegate (RNS)"
LSE = "London Stock Exchange news"


def epic_of(ticker: str) -> str:
    """The LSE code of a Yahoo ticker: BT-A.L becomes BT.A."""
    return _base_symbol(ticker).replace("-", ".")


def parse_investegate(html: str, limit: int) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """(epic, company name, items) of an Investegate company page."""
    head = re.search(r"<title>\s*(.*?)\s+RNS Announcements\s*\|\s*([A-Z0-9.]+)\s+RNS Announcements", html, flags=re.S)
    name, epic = (_text(head.group(1)), head.group(2)) if head else (None, None)
    items: list[dict[str, Any]] = []
    for row in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        link = re.search(r'<a class="announcement-link" href="([^"]+)">(.*?)</a>', row, flags=re.S)
        date = re.search(r"<td>\s*(\d{1,2} [A-Z][a-z]{2} \d{4})\s*</td>", row)
        if not link or not date:
            continue
        try:
            day = datetime.strptime(date.group(1), "%d %b %Y").date().isoformat()
        except ValueError:
            continue
        supplier = re.search(r'title="supplier: ([^"]*)"', row)
        label = supplier.group(1) if supplier else ""
        default = "press release" if "non-regulatory" in label.lower() else "regulatory" if "regulatory" in label.lower() else "announcement"
        title = _text(link.group(2))
        items.append(_item(day, title, _kind_from_title(title, default), html_lib.unescape(link.group(1)), INVESTEGATE))
        if len(items) >= limit:
            break
    return epic, name, items


@ttl_cache(CACHE_SECONDS)
def _investegate_page(epic: str) -> tuple[str, str]:
    return _fetch(f"https://www.investegate.co.uk/company/{urllib.parse.quote(epic)}")


def source_investegate(ticker: str, limit: int, ctx: dict[str, Any]) -> Result:
    epic = epic_of(ticker)
    link = f"https://www.investegate.co.uk/company/{epic}"
    try:
        _, text = _investegate_page(epic)
    except NotFound:
        return Result(INVESTEGATE, "issuer_not_found", f"Investegate has no company with the code {epic}.", "Search the company on the London Stock Exchange.", url=lse_search(epic))
    except SourceError as exc:
        return _unreachable(INVESTEGATE, exc, link)
    found, name, items = parse_investegate(text, limit)
    if found is None or found.upper() != epic:  # an unknown code redirects to the home page
        return Result(INVESTEGATE, "issuer_not_found", f"Investegate has no company with the code {epic}.", "Search the company on the London Stock Exchange.", url=lse_search(epic))
    result = Result(INVESTEGATE, issuer=name, url=link, items=items,
                    notes=["Investegate republishes the Regulatory News Service (RNS) of the London Stock Exchange."])
    if not items:
        result.message = "Investegate lists no announcements for this company."
    return result


def lse_search(epic: str) -> str:
    return "https://www.londonstockexchange.com/search?" + urllib.parse.urlencode({"searchtype": "all", "q": epic})


def source_lse(ticker: str, limit: int, ctx: dict[str, Any]) -> Result:
    return Result(LSE, "not_readable", "The London Stock Exchange news pages ask automated readers to stay out (robots.txt), so they are not read.",
                  "Open the news search for this company.", url=lse_search(epic_of(ticker)))


# ------------------------------------------------------------------ United States: SEC EDGAR

SEC = "SEC EDGAR"


def sec_items(filings: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Announcement items from the filing list of ``data.sec_reports.recent_filings``.

    An 8-K with exhibit 99 gives one press release item per exhibit; other 8-Ks are regulatory; annual, quarterly and
    foreign annual reports are reports. Titles come from the filing and exhibit labels the SEC index gives.
    """
    out: list[dict[str, Any]] = []
    for f in filings:
        form, filed = str(f.get("form") or ""), f.get("filed")
        exhibits = [e for e in f.get("exhibits") or [] if e.get("url")]
        if form.startswith("8-K") and exhibits:
            for e in exhibits:
                label = str(e.get("label") or "Exhibit")
                number = re.search(r"\(Ex\. ([\d.]+)\)", label)
                if e.get("kind") in ("presentation", "transcript"):
                    kind = e["kind"]
                elif e.get("kind") == "earnings" or (number and number.group(1) == "99.1"):
                    kind = "press release"  # exhibit 99.1 is where companies put the press release
                else:
                    kind = "exhibit"
                what = f"Exhibit {number.group(1)}" if label.startswith("Other exhibit") and number else re.sub(r"\s*\(Ex\. [\d.]+\)", "", label)
                out.append(_item(filed, f"{what}: {f.get('title') or form}", kind, e["url"], SEC, form=form))
            continue
        kind = "regulatory" if form.startswith(("8-K", "6-K")) else "report" if f.get("kind") in ("annual", "quarterly", "foreign_annual") else "regulatory"
        url = f.get("url") or f.get("index_url")
        if url:
            out.append(_item(filed, str(f.get("title") or form), kind, url, SEC, form=form))
    return out[:limit]


def source_sec(ticker: str, limit: int, ctx: dict[str, Any]) -> Result:
    from data.sec_reports import recent_filings

    data = recent_filings(ticker, 80)
    link = f"https://www.sec.gov/edgar/search/#/q={urllib.parse.quote(ticker)}"
    status = data.get("status")
    if status == "no_access":
        return Result(SEC, "no_access", data.get("message"), data.get("fix"), url=link)
    if status == "not_sec_filer":
        return Result(SEC, "issuer_not_found", data.get("message"), "Open the company's investor relations site.", url=link)
    if status != "ok":
        return Result(SEC, "unreachable", data.get("message"), data.get("fix"), url=link)
    return Result(SEC, issuer=data.get("name"), url=(data.get("links") or [{}])[0].get("url") or link,
                  items=sec_items(data.get("filings") or [], limit), notes=list(data.get("notes") or []))


# ------------------------------------------------------------------ interface

ADAPTERS: dict[str, Callable[[str, int, dict[str, Any]], Result]] = {
    "sec": source_sec, "mfn": source_mfn, "cision": source_cision, "asx": source_asx,
    "sedar": source_sedar, "investegate": source_investegate, "lse": source_lse,
}


def _website(ticker: str) -> str | None:
    """The company's own site from its Yahoo profile (the backup of every market), or None."""
    try:
        from data import fundamentals as _fundamentals

        site = str(_fundamentals._load_info(ticker).get("website") or "").strip()
    except Exception as exc:  # noqa: BLE001 - the link is optional
        logger.info("No website for %s: %s", ticker, exc)
        return None
    if not site:
        return None
    site = site if site.startswith("http") else f"https://{site}"
    return site if site.startswith("https://") or site.startswith("http://") else None


def _links(market: str, symbol: str, results: list[Result]) -> list[dict[str, Any]]:
    """Pages to open by hand: each source's issuer page (when it answered) or search page (when it did not)."""
    out: list[dict[str, Any]] = []
    for r in results:
        if r.url and all(r.url != x["url"] for x in out):
            out.append({"label": r.source if r.status == "ok" else f"Search at {r.source}", "url": r.url,
                        "note": "The issuer's page at the source." if r.status == "ok" else None})
    if market == "AU" and all("announcements." not in x["url"] for x in out):
        out.append({"label": "ASX announcements page", "url": asx_page(_base_symbol(symbol)), "note": None})
    return out


def announcements(ticker: str, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """The /api/announcements payload. Never raises for expected conditions.

    ``source`` names the source that answered (None when none did), ``attempts`` lists every source tried with its
    status, ``links`` holds the pages to open by hand (an issuer page, a search, the company's own site).
    """
    symbol = ticker.strip().upper()
    limit = max(1, min(int(limit), 100))
    market = market_of(symbol)
    payload: dict[str, Any] = {
        "announcements": True, "ticker": symbol, "market": market, "market_name": MARKETS[market]["name"] if market else None,
        "status": "ok", "message": None, "fix": None, "source": None, "name": None, "items": [], "attempts": [],
        "links": [], "notes": [], "covered": COVERED,
    }
    if market is None:
        payload.update(status="no_adapter", message=NO_ADAPTER, fix="Open the exchange's own announcements page for this company.")
        return payload
    ctx: dict[str, Any] = {}
    results: list[Result] = []
    for key in MARKETS[market]["chain"]:
        try:
            result = ADAPTERS[key](symbol, limit, ctx)
        except Exception as exc:  # noqa: BLE001 - one broken adapter must not hide the others
            logger.warning("Announcement source %s failed for %s: %s", key, symbol, exc, exc_info=True)
            result = Result(key, "unreachable", f"{key} failed unexpectedly ({type(exc).__name__}).", "Try again later.")
        results.append(result)
        if result.status == "ok":
            break
    answered = results[-1] if results[-1].status == "ok" else None
    payload["attempts"] = [r.attempt() for r in results]
    payload["links"] = _links(market, symbol, results)
    if answered:
        payload.update(source=answered.source, name=answered.issuer, items=answered.items, message=answered.message, notes=list(answered.notes))
        if len(results) > 1:
            first = results[0]
            payload["notes"].append(f"{first.source} did not answer ({first.message}) so {answered.source} was used instead.")
    else:
        first = results[0]
        payload.update(status=first.status, message=first.message, fix=first.fix)
        if len(results) > 1:
            payload["notes"] += [f"{r.source}: {r.message}" for r in results[1:] if r.message]
    if market != "US":
        site = _website(symbol)
        if site:
            payload["links"].append({"label": "Company website", "url": site,
                                     "note": "Look for an Investors or Media section. Press releases are usually there too."})
    return payload
