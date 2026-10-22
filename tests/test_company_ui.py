"""The Company card in the Analyze tab: its markup and script hooks ship with the page."""

import pytest


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_page_has_company_card_with_four_views(client):
    page = client.get("/").data.decode()
    assert 'id="company-card"' in page and 'id="company-body"' in page
    for view in ("financials", "checklist", "filings", "industry"):
        assert f'data-co-view="{view}"' in page
    assert 'data-co-period="annual"' in page and 'data-co-period="quarterly"' in page
    # The card sits before the decision flow, after the charts.
    assert page.index('id="history-card"') < page.index('id="company-card"') < page.index('id="flow-diagram"')


def test_script_loads_company_lazily_and_opens_links_safely(client):
    js = client.get("/static/app.js").data.decode()
    assert "loadCompany(d.ticker)" in js and "setupCompany()" in js
    for route in ("/api/financials/", "/api/filings/", "/api/industry/"):
        assert route in js
    assert 'rel="noopener noreferrer"' in js
