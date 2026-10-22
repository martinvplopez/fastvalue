"""One money, percent and plural format in the text the server writes (synthetic numbers only)."""
from __future__ import annotations

from core import broker_import, monitor, portfolio as pf, style_profile, style_report, track_record
from data.fx import amount_text


def test_amount_text_uses_the_symbol_for_eur_usd_gbp_and_the_code_for_the_rest():
    assert amount_text(1818.4, "EUR") == "€1,818"
    assert amount_text(1818, "USD") == "$1,818"
    assert amount_text(1818, "GBP") == "£1,818"
    assert amount_text(1818, "CHF") == "1,818 CHF"
    assert amount_text(-33, "EUR") == "-€33"
    assert amount_text(500) == "500"


def test_review_reasons_use_the_row_percentage_and_the_symbol_amount():
    assert pf._share(0.249) == "24.9%" and pf._share(0.25) == "25.0%"
    assert pf._amount(1818, "EUR") == "€1,818"


def test_record_money_is_signed_for_differences_and_plain_for_totals():
    assert track_record._eur(2873.2) == "+€2,873" and track_record._eur(-33) == "-€33"
    assert track_record._eur_abs(13367) == "€13,367" and track_record._eur_abs(-33) == "-€33"
    assert track_record._eur(None) == "n/a" and track_record._eur_abs(None) == "n/a"


def test_the_share_page_still_drops_the_new_euro_figures():
    strip = style_report._strip_amounts
    assert strip("Idle cash is the strongest rule by its weaker half (+€4,997); it does not pass.") == \
        "Idle cash is the strongest rule by its weaker half; it does not pass."
    assert strip("Fees came to €1,234; the rest stays.") == "The rest stays."


def test_counts_get_real_plurals():
    for mod in (track_record, style_report, style_profile, monitor, broker_import):
        assert mod._n(1, "case", "cases") == "1 case" and mod._n(0, "case", "cases") == "0 cases"
        assert mod._n(2, "closed round trip", "closed round trips") == "2 closed round trips"
    assert track_record._confidence(1, None, None)[1] == "only 1 case; too few to call it a habit"
