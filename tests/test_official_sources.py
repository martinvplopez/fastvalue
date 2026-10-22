"""Official announcement sources: parsing, issuer matching, fallback and status messages. No network."""

import json

import pytest

from data import official_sources as o

MFN_PAGE = """
<h1 class="heading-1"><a href="/all/a/volvo">Volvo</a></h1>
<div class="short-item-wrapper"><div class="short-item compressible" id="a" onclick="x">
<span class="compressed-date">2026-10-09</span><span class="compressed-time">11:25:56</span>
<a class="title-link item-link" href="/cis/a/volvo/ab-volvo-one-1" title="AB Volvo: Interim report &amp; more" >AB Volvo</a></div></div>
<div class="short-item-wrapper"><div class="short-item compressible" id="b" onclick="x">
<span class="compressed-date">2026-10-08</span>
<a class="title-link item-link" href="/a/volvo/two-2" title="AB Volvo: Invitation to a conference" >x</a></div></div>
"""
MFN_SEARCH = json.dumps([
    {"slug": "volvo", "name": "Volvo", "tickers": ["XSTO:VOLV A", "XSTO:VOLV B", "XLON:0MHW"]},
    {"slug": "volvo-car", "name": "Volvo Car", "tickers": ["XSTO:VOLCAR B"]},
])
CISION_PAGE = """<title>Volvo</title><article> <a href="https://news.cision.com/se/volvo/r/one,c1" class="bodytext content">
<h2>Delårsrapport &#229;ret</h2> <time datetime="2026-10-05 06:30:00Z">x</time> <p>text</p> </a> </article>"""
ASX_PAGE = """<h2>Search results: Company announcements for <br>
   BHP GROUP LIMITED (BHP)<br></h2>
<table><tr><th>Date</th></tr>
<tr><td>08/10/2026<br><span class="dates-time">9:23 am</span></td>
<td class="pricesens"><img src="/asx/v2/markets/image/icon-price-sensitive.svg" class="pricesens" title="price sensitive"></td>
<td><a href="/asx/v2/statistics/displayAnnouncement.do?display=pdf&amp;idsId=03149953">
   Update - Dividend/Distribution - BHP<br><img src="x"></a></td></tr>
<tr><td>07/10/2026<br></td><td class="pricesens"> </td>
<td><a href="/asx/v2/statistics/displayAnnouncement.do?display=pdf&amp;idsId=1">Half Year Report<br></a></td></tr></table>"""
ASX_UNKNOWN = '<h2>Search for recent and past announcements</h2><p class="error">The ASX code you entered does not match any companies code.</p>'
INVESTEGATE = """<title>AstraZeneca RNS Announcements | AZN RNS Announcements | Investegate</title>
<table class="table-investegate"><tbody>
<tr><td>05 Oct 2026</td><td>12:00 PM</td><td><a class="regulatory source-RNS" title="supplier: Regulatory News Service (Regulatory)" href="x">RNS</a></td>
<td><a class="announcement-link" href="https://www.investegate.co.uk/announcement/rns/a/b/1">AstraZeneca completes investment</a></td></tr>
<tr><td>02 Oct 2026</td><td>07:30 AM</td><td><a title="supplier: Globenewswire (Non-Regulatory)" href="x">GNW</a></td>
<td><a class="announcement-link" href="https://www.investegate.co.uk/announcement/gnw/a/b/2">Half-year Report 2026</a></td></tr>
</tbody></table>"""
INVESTEGATE_HOME = "<title>Investegate | Latest RNS</title><body>home</body>"


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    from data.cache_utils import clear_all_caches

    clear_all_caches()
    monkeypatch.setattr(o, "_website", lambda ticker: None)
    yield
    clear_all_caches()


def route(monkeypatch, pages: dict[str, object]):
    """Replace the fetcher: the first key contained in the url answers (an exception is raised)."""
    calls: list[str] = []

    def fake(url, accept="text/html"):
        calls.append(url)
        for key, value in pages.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value if isinstance(value, tuple) else (url, value)
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(o, "_fetch", fake)
    return calls


def test_market_from_suffix():
    assert [o.market_of(t) for t in ("AAPL", "BRK.B", "VOLV-B.ST", "BHP.AX", "SHOP.TO", "XYZ.V", "AZN.L", "SAP.DE")] == [
        "US", "US", "SE", "AU", "CA", "CA", "UK", None]


def test_no_adapter_lists_covered_markets():
    data = o.announcements("SAP.DE")
    assert data["status"] == "no_adapter" and data["items"] == []
    assert "Sweden" in data["message"] and "Australia" in data["message"]


def test_mfn_items_and_issuer_match_by_ticker(monkeypatch):
    calls = route(monkeypatch, {"search/companies": MFN_SEARCH, "/all/a/volvo": MFN_PAGE})
    data = o.announcements("VOLV-B.ST", 10)
    assert data["status"] == "ok" and data["source"] == "MFN" and data["name"] == "Volvo"
    first, second = data["items"]
    assert first == {"date": "2026-10-09", "title": "AB Volvo: Interim report & more", "kind": "report",
                     "url": "https://mfn.se/cis/a/volvo/ab-volvo-one-1", "source": "MFN"}
    assert second["kind"] == "announcement" and second["url"] == "https://mfn.se/a/volvo/two-2"
    assert len(calls) == 2  # one issuer search, one feed


def test_mfn_does_not_guess_when_ticker_unknown_or_ambiguous(monkeypatch):
    route(monkeypatch, {"search/companies": MFN_SEARCH})
    data = o.announcements("NOPE.ST")
    assert data["status"] == "issuer_not_found" and data["items"] == [] and data["source"] is None
    two = json.dumps([{"slug": "a", "name": "A", "tickers": ["XSTO:AAA"]}, {"slug": "b", "name": "B", "tickers": ["XNGM:AAA"]}])
    o._mfn_companies.cache_clear()
    route(monkeypatch, {"search/companies": two})
    data = o.announcements("AAA.ST")
    assert data["status"] == "ambiguous" and "A, B" in data["message"]


def test_mfn_down_falls_back_to_cision(monkeypatch):
    route(monkeypatch, {"search/companies": MFN_SEARCH, "mfn.se/all/a/volvo": o.SourceError("mfn.se did not answer"),
                        "news.cision.com/se/volvo": CISION_PAGE})
    data = o.announcements("VOLV-B.ST")
    assert data["status"] == "ok" and data["source"] == "Cision"
    assert data["items"][0]["title"] == "Delårsrapport året" and data["items"][0]["date"] == "2026-10-05"
    assert data["items"][0]["kind"] == "report"
    assert [a["status"] for a in data["attempts"]] == ["unreachable", "ok"]
    assert any("MFN did not answer" in n and "Cision was used" in n for n in data["notes"])


def test_cision_page_for_another_company_is_not_used(monkeypatch):
    route(monkeypatch, {"search/companies": MFN_SEARCH, "mfn.se/all/a/volvo": o.SourceError("boom"),
                        "news.cision.com/se/volvo": CISION_PAGE.replace("<title>Volvo", "<title>Volvo Car")})
    data = o.announcements("VOLV-B.ST")
    assert data["status"] == "unreachable" and data["items"] == [] and data["source"] is None
    assert data["attempts"][1]["status"] == "issuer_not_found"


def test_mfn_search_down_says_not_reachable_and_is_not_cached(monkeypatch):
    route(monkeypatch, {"search/companies": o.SourceError("timeout")})
    assert o.announcements("VOLV-B.ST")["status"] == "unreachable"
    route(monkeypatch, {"search/companies": MFN_SEARCH, "/all/a/volvo": MFN_PAGE})
    assert o.announcements("VOLV-B.ST")["status"] == "ok"


def test_asx_parsing_and_price_sensitive(monkeypatch):
    route(monkeypatch, {"announcements.do": ASX_PAGE})
    data = o.announcements("BHP.AX")
    assert data["status"] == "ok" and data["source"] == "ASX" and data["name"] == "BHP GROUP LIMITED"
    a, b = data["items"]
    assert a["date"] == "2026-10-08" and a["title"] == "Update - Dividend/Distribution - BHP" and a["kind"] == "regulatory"
    assert a["price_sensitive"] is True and a["url"].startswith("https://www.asx.com.au/asx/v2/statistics/displayAnnouncement.do?display=pdf&idsId=")
    assert b["price_sensitive"] is False and b["kind"] == "report"


def test_asx_unknown_code_and_wrong_code(monkeypatch):
    route(monkeypatch, {"announcements.do": ASX_UNKNOWN})
    data = o.announcements("ZZZ.AX")
    assert data["status"] == "issuer_not_found" and any("announcements.zzz" in link["url"] for link in data["links"])
    o._asx_page.cache_clear()
    route(monkeypatch, {"announcements.do": ASX_PAGE})
    assert o.announcements("BHPX.AX")["status"] == "ambiguous"


def test_sedar_is_not_read_and_gives_link():
    data = o.announcements("SHOP.TO")
    assert data["status"] == "not_readable" and data["items"] == []
    assert "SEDAR+ cannot be read automatically" in data["message"]
    assert any("sedarplus.ca" in link["url"] for link in data["links"])


def test_investegate_items_and_kinds(monkeypatch):
    calls = route(monkeypatch, {"investegate.co.uk/company/AZN": ("u", INVESTEGATE)})
    data = o.announcements("AZN.L")
    assert data["status"] == "ok" and data["source"] == "Investegate (RNS)" and data["name"] == "AstraZeneca"
    a, b = data["items"]
    assert a["date"] == "2026-10-05" and a["kind"] == "regulatory"
    assert b["date"] == "2026-10-02" and b["kind"] == "report"
    assert len(calls) == 1


def test_investegate_epic_and_unknown(monkeypatch):
    assert o.epic_of("BT-A.L") == "BT.A"
    route(monkeypatch, {"company/ZZZ": ("https://www.investegate.co.uk/", INVESTEGATE_HOME)})
    data = o.announcements("ZZZ.L")
    assert data["status"] == "issuer_not_found"
    assert [a["status"] for a in data["attempts"]] == ["issuer_not_found", "not_readable"]
    o._investegate_page.cache_clear()
    route(monkeypatch, {"company/ZZZ": o.NotFound("HTTP 404")})
    assert o.announcements("ZZZ.L")["status"] == "issuer_not_found"


def test_investegate_down_leaves_lse_link(monkeypatch):
    route(monkeypatch, {"investegate.co.uk": o.SourceError("down")})
    data = o.announcements("AZN.L")
    assert data["status"] == "unreachable" and data["items"] == []
    assert [a["status"] for a in data["attempts"]] == ["unreachable", "not_readable"]
    assert any("londonstockexchange.com" in link["url"] for link in data["links"])


def test_company_website_is_added_for_non_us(monkeypatch):
    monkeypatch.setattr(o, "_website", lambda ticker: "https://www.example.com")
    data = o.announcements("SHOP.TO")
    assert data["links"][-1]["label"] == "Company website"


def test_sec_items_press_releases_and_reports():
    filings = [
        {"form": "8-K", "kind": "earnings", "filed": "2026-07-30", "title": "Quarterly results (Item 2.02)",
         "exhibits": [{"label": "Earnings press release (Ex. 99.1)", "kind": "earnings", "url": "https://sec/ex1"},
                      {"label": "Investor presentation or slides (Ex. 99.2)", "kind": "presentation", "url": "https://sec/ex2"},
                      {"label": "Other exhibit (Ex. 99.3)", "kind": "other", "url": "https://sec/ex3"}]},
        {"form": "8-K", "kind": "current", "filed": "2026-06-01", "title": "Current report (8-K)", "exhibits": [],
         "url": "https://sec/doc"},
        {"form": "10-Q", "kind": "quarterly", "filed": "2026-05-01", "title": "Quarterly report (10-Q)", "exhibits": [],
         "url": "https://sec/q", "index_url": "https://sec/qi"},
    ]
    items = o.sec_items(filings, 10)
    assert [i["kind"] for i in items] == ["press release", "presentation", "exhibit", "regulatory", "report"]
    assert items[0]["title"] == "Earnings press release: Quarterly results (Item 2.02)" and items[0]["url"] == "https://sec/ex1"
    assert items[2]["title"].startswith("Exhibit 99.3:")
    assert len(o.sec_items(filings, 2)) == 2


def test_sec_statuses(monkeypatch):
    import data.sec_reports as sr

    monkeypatch.setattr(sr, "recent_filings", lambda t, n: {"status": "no_access", "message": "needs email", "fix": "set it"})
    data = o.announcements("AAPL")
    assert data["status"] == "no_access" and data["fix"] == "set it"
    monkeypatch.setattr(sr, "recent_filings", lambda t, n: {"status": "not_sec_filer", "message": "Not an SEC filer"})
    assert o.announcements("ZZZ")["status"] == "issuer_not_found"
    monkeypatch.setattr(sr, "recent_filings", lambda t, n: {"status": "error", "message": "SEC down"})
    assert o.announcements("AAPL")["status"] == "unreachable"
    monkeypatch.setattr(sr, "recent_filings", lambda t, n: {"status": "ok", "name": "Apple Inc.", "links": [{"url": "https://sec/all"}],
                                                            "filings": [{"form": "10-Q", "kind": "quarterly", "filed": "2026-05-01", "title": "Q", "url": "https://sec/q"}], "notes": []})
    data = o.announcements("AAPL")
    assert data["status"] == "ok" and data["source"] == "SEC EDGAR" and data["items"][0]["kind"] == "report"


def test_robots_txt_is_honoured(monkeypatch):
    texts = {"https://blocked.example/robots.txt": "User-agent: *\nDisallow: /secret\n", "https://blocked.example/open": "hello"}

    def raw(url, accept="text/html"):
        if url in texts:
            return url, texts[url]
        raise o.NotFound("404")

    monkeypatch.setattr(o, "_raw_get", raw)
    o._robots.clear()
    with pytest.raises(o.SourceError, match="not to open"):
        o._fetch("https://blocked.example/secret/page")
    assert o._fetch("https://blocked.example/open")[0] == "https://blocked.example/open"
    o._robots.clear()


def test_endpoint_returns_payload(monkeypatch):
    from web.app import app

    monkeypatch.setattr(o, "announcements", lambda t, n: {"announcements": True, "ticker": t, "limit": n, "status": "ok"})
    res = app.test_client().get("/api/announcements/VOLV-B.ST?limit=5")
    assert res.status_code == 200 and res.get_json()["ticker"] == "VOLV-B.ST" and res.get_json()["limit"] == 5
    assert app.test_client().get("/api/announcements/AAPL?limit=x").status_code == 400


def test_script_uses_announcements_for_foreign_listings():
    from web.app import app

    js = app.test_client().get("/static/app.js").data.decode()
    assert "/api/announcements/" in js and "coAnnouncementsHtml" in js and 'data-co-kind="press"' in js
