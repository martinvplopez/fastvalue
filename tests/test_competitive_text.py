"""10-K text parsers of the Competitive Position agent, on hand-written snippets (offline)."""

from __future__ import annotations

import pytest
import yfinance

import config.settings as settings
from analysis.filing_text import (
    customer_concentration,
    excerpt,
    html_to_text,
    named_competitors,
    no_major_customer,
    normalize_company_name,
    sourcing_sentences,
    split_items,
    subsections,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch: pytest.MonkeyPatch):
    def blocked(*_a, **_k):
        raise RuntimeError("Network access is disabled in tests")

    monkeypatch.setattr(yfinance, "Industry", blocked, raising=False)
    monkeypatch.setattr(yfinance, "Sector", blocked, raising=False)
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "cache")


def test_html_to_text_drops_hidden_blocks_and_breaks_lines():
    html = (
        "<html><head><style>p {color: red}</style><script>var x = 1;</script></head><body>"
        "<ix:header><ix:hidden>0000320193 FY</ix:hidden></ix:header>"
        "<div>Item 1. Business</div><p>We make &amp; sell devices.</p><table><tr><td>A</td><td>B</td></tr></table>"
        "</body></html>"
    )
    text = html_to_text(html)
    assert "color" not in text and "var x" not in text and "0000320193" not in text
    assert "We make & sell devices." in text
    assert text.split("\n")[0] == "Item 1. Business"
    assert text.endswith("A\nB")  # table cells are lines too


def test_split_items_prefers_body_over_table_of_contents():
    toc = "\n".join(f"Item {n}. Heading {n}" for n in ("1", "1A", "1B", "2"))
    body = (
        "Item 1. Business\n" + "Business text. " * 400 + "\n"
        "Item 1A. Risk Factors\n" + "Risk text. " * 400 + "\n"
        "Item 1B. Unresolved Staff Comments\nNone.\nItem 2. Properties\nWe own buildings."
    )
    text = "Table of Contents\n" + toc + "\n" + body
    spans = split_items(text)
    start, end = spans["1"]
    assert end - start > 4000 and "Business text." in text[start:end]
    start, end = spans["1A"]
    assert "Risk text." in text[start:end] and "Business text." not in text[start:end]


ITEM_ONE = """Item 1. Business
General
The Company designs and sells devices to many customers around the world, from many places and in many ways.
Competition
The Company competes with Advanced Micro Devices, Inc., Intel Corporation and Qualcomm Incorporated in several markets.
Pricing, features and service decide the winner of most competitive bids in each of these markets today.
Table of Contents
12
Raw Materials and Component Products
The Company relies on a limited number of suppliers for chips, glass and special metals used in each of its main products.
Sales and Marketing
Most sales go through resellers, carriers and the Company's own online and retail stores in every region it serves.
Human Capital
Employees are the backbone of the Company, and it invests in them.
"""


def test_subsections_find_competition_suppliers_and_customers():
    found = subsections(ITEM_ONE)
    assert found["competition"].startswith("The Company competes with Advanced Micro Devices")
    assert "Pricing, features and service" in found["competition"]  # a Table of Contents line and a page number do not end it
    assert "Raw Materials" not in found["competition"] and "limited number of suppliers" not in found["competition"]
    assert found["suppliers"].startswith("The Company relies on a limited number of suppliers")
    assert found["customers"].startswith("Most sales go through resellers")
    assert "Employees are the backbone" not in found["customers"]


def test_subsections_competitive_environment_and_run_in_heading():
    item = (
        "Competitive Environment\nThe markets for our products are highly competitive and change quickly with each new "
        "release from a rival, so we must keep investing in new products every year.\nGeneral\nOther things.\n"
    )
    assert subsections(item)["competition"].startswith("The markets for our products")
    run_in = (
        "Competition. We compete with large retailers, online sellers and wholesale clubs on price, assortment and "
        "service, and that competition is intense in all of the countries where we operate our stores.\nSeasonality\nNone."
    )
    assert subsections(run_in)["competition"].startswith("Competition. We compete with large retailers")


CLX = ("Net sales to the Company's largest customer, Walmart Stores, Inc. and its affiliates, were 26%, 27% and 25% "
       "of consolidated net sales for each of the fiscal years ended June 30, 2026, 2025 and 2024, respectively.")


def test_customer_concentration_single_top_n_and_government():
    single = customer_concentration(CLX)
    assert [(c.kind, c.n, c.pct) for c in single] == [("single", 1, 26.0)]
    top = customer_concentration(
        "During fiscal 2026, sales to our ten largest customers accounted for approximately 73% of our total consolidated revenues.")
    assert [(c.kind, c.n, c.pct) for c in top] == [("top_n", 10, 73.0)]
    government = customer_concentration(
        "We derive our sales mainly from the U.S. Government, from which we derived 72% of our sales in 2025, including 63% from one agency.")
    assert [(c.kind, c.pct) for c in government] == [("government", 72.0)]


def test_customer_concentration_rejects_segments_geography_and_negations():
    assert customer_concentration(
        "Our Walmart U.S. and Sam's Club U.S. operating segments generated approximately 82% of our net sales.") == []
    assert customer_concentration(
        "In fiscal 2026 we derived approximately 12.3% of our net sales from customers located outside of the United States.") == []
    assert customer_concentration("No customer accounted for 10% or more of our net sales in fiscal 2026.") == []


def test_no_major_customer_finds_the_negation():
    text = "Revenue is spread widely. No customer accounted for 10% or more of net sales in 2026, 2025 or 2024."
    assert no_major_customer(text).startswith("No customer accounted for 10%")
    assert no_major_customer(CLX) is None


def test_sourcing_sentences_keep_supply_risk_and_drop_look_alikes():
    text = (
        "The Company relies on single-source partners, including some outside of the U.S., to supply and manufacture many components. "
        "Sales of single-source drugs and biologics are subject to government price negotiation. "
        "We do not obtain a significant portion of our merchandise from any one supplier. "
        "Some key parts come from a limited number of suppliers, which could raise our costs."
    )
    found = sourcing_sentences(text)
    assert len(found) == 2
    assert found[0].startswith("The Company relies on single-source partners")
    assert found[1].startswith("Some key parts come from a limited number of suppliers")


def test_named_competitors_maps_names_and_skips_the_company_itself():
    titles = {normalize_company_name("Advanced Micro Devices, Inc."): "AMD", normalize_company_name("Intel Corporation"): "INTC"}
    section = ("The Company competes with Advanced Micro Devices, Inc., Intel Corporation and Qualcomm Incorporated. "
               "Apple Inc. also sells devices, and The Company faces pressure from them.")
    found = named_competitors(section, titles, own_names=("Apple Inc.",))
    assert found == [
        {"name": "Advanced Micro Devices, Inc", "ticker": "AMD"},
        {"name": "Intel Corporation", "ticker": "INTC"},
        {"name": "Qualcomm Incorporated", "ticker": None},
    ]


def test_excerpt_normalizes_dashes_and_cuts_at_a_word():
    assert excerpt("Sales – net of returns — grew.") == "Sales - net of returns - grew."
    long = "word " * 100
    out = excerpt(long)
    assert len(out) <= 240 and out.endswith("...") and not out[:-3].endswith(" wor")
    assert excerpt("  short   text ") == "short text"
