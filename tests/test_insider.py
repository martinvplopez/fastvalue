"""Tests for the insider trading provider and InsiderAgent (no network, synthetic Form 4 rows)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import data.insider_trading as insider_trading
from agents.base import LIVE, NEUTRAL, PARTIAL, UNAVAILABLE
from agents.insider_agent import SALE_POINTS_CAP, InsiderAgent
from data.insider_trading import (
    GIFT,
    GRANT_AWARD,
    OPEN_MARKET_BUY,
    OPEN_MARKET_SALE,
    OPTION_EXERCISE,
    OTHER,
    TAX_WITHHOLDING,
    analyze_insider_transactions,
    classify_transaction,
    parse_price,
    role_weight,
)
from tests.conftest import make_ohlcv

CEO = ("DOE JANE", "Chief Executive Officer")
CFO = ("ROE RICHARD", "Chief Financial Officer")
DIRECTOR = ("SMITH ALEX", "Director")
DIRECTOR_2 = ("JONES KIM", "Director")


def _row(
    ref: pd.Timestamp,
    days_before: int,
    who: tuple[str, str],
    text: str,
    shares: int,
    value: float = np.nan,
    url: str = "",
) -> dict:
    insider, position = who
    return {
        "Shares": shares,
        "Value": value,
        "URL": url,
        "Text": text,
        "Insider": insider,
        "Position": position,
        "Transaction": "",
        "Start Date": ref - pd.Timedelta(days=days_before),
        "Ownership": "D",
    }


def _buy(ref, days_before, who, dollars, price=100.0, **kw) -> dict:
    return _row(ref, days_before, who, f"Purchase at price {price:.2f} per share.", int(dollars / price), dollars, **kw)


def _sale(ref, days_before, who, dollars, price=100.0, **kw) -> dict:
    return _row(ref, days_before, who, f"Sale at price {price:.2f} per share.", int(dollars / price), dollars, **kw)


def _frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _evaluate(monkeypatch, frame, ohlcv, ticker: str = "TEST"):
    monkeypatch.setattr(insider_trading, "fetch_insider_transactions", lambda _ticker: frame)
    return InsiderAgent().evaluate(ticker, ohlcv)


def _points(assessment, prefix: str) -> float:
    for item in assessment.raw_data["score_breakdown"]:
        if item["factor"].startswith(prefix):
            return item["points"]
    raise AssertionError(f"no score_breakdown factor starting with {prefix!r}")


@pytest.fixture
def uptrend() -> pd.DataFrame:
    return make_ohlcv(periods=300, start=100.0, end=140.0)


@pytest.fixture
def ref(uptrend) -> pd.Timestamp:
    return uptrend.index[-1]


# ------------------------------------------------------------------ registration


def test_registered_under_insider_key():
    from agents.registry import get_agent_classes

    assert get_agent_classes()["insider"] is InsiderAgent
    meta = InsiderAgent.metadata()
    assert meta["name"] == "Insider Trading Agent"
    assert meta["category"] == "Flow"
    assert meta["score_scale"] == 35.0
    # Owner-layer evidence with no vote: no signal in the S&P 500 point-in-time test (decision_log.md T-10).
    assert meta["default_weight"] == 0.0
    assert meta["role"] == "owner"
    assert meta["point_in_time"] is False


# ------------------------------------------------------------------ availability


def test_provider_error_is_unavailable_without_fabricated_rows(uptrend):
    # conftest makes yfinance.Ticker raise, so this exercises the real fetch path.
    result = InsiderAgent().evaluate("AAPL", uptrend)
    assert result.data_quality == UNAVAILABLE
    assert result.confidence == 0.0
    assert result.score == 0.0
    assert "provider error" in (result.error or "")
    assert "transactions_table" not in result.raw_data
    assert "total_buy_transactions" not in result.raw_data
    assert any("type=4" in link["url"] for link in result.raw_data["links"])


@pytest.mark.parametrize("frame", [None, pd.DataFrame(), pd.DataFrame(columns=["Shares", "Text", "Start Date"])])
def test_empty_or_missing_frame_is_unavailable(monkeypatch, uptrend, frame):
    result = _evaluate(monkeypatch, frame, uptrend)
    assert result.data_quality == UNAVAILABLE
    assert result.confidence == 0.0
    assert "transactions_table" not in result.raw_data
    data = analyze_insider_transactions(frame, uptrend, "TEST")
    assert data.available is False
    assert data.transactions == []
    assert data.total_buy_transactions == 0


def test_price_drop_without_rows_never_invents_cluster_buying(monkeypatch):
    falling = make_ohlcv(periods=300, start=150.0, end=80.0)
    result = _evaluate(monkeypatch, pd.DataFrame(), falling)
    assert result.data_quality == UNAVAILABLE
    assert result.raw_data.get("cluster_buy_detected") is None


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    ("text", "code", "expected"),
    [
        ("Purchase at price 12.10 per share.", "", OPEN_MARKET_BUY),
        ("Sale at price 340.06 per share.", "", OPEN_MARKET_SALE),
        ("Sale at price 284.57 - 285.04 per share.", "", OPEN_MARKET_SALE),
        ("Stock Award(Grant) at price 0.00 per share.", "", GRANT_AWARD),
        ("Stock Gift at price 0.00 per share.", "", GIFT),
        ("Conversion of Exercise of derivative security at price 5.00 per share.", "", OPTION_EXERCISE),
        ("Payment of exercise price or tax liability", "", TAX_WITHHOLDING),
        ("", "", OTHER),
        ("", "P", OPEN_MARKET_BUY),
        ("", "F", TAX_WITHHOLDING),
    ],
)
def test_classification(text, code, expected):
    assert classify_transaction(text, code) == expected


def test_parse_price_handles_single_range_and_zero():
    assert parse_price("Purchase at price 12.10 per share.") == pytest.approx(12.10)
    assert parse_price("Sale at price 284.57 - 285.04 per share.") == pytest.approx((284.57 + 285.04) / 2)
    assert parse_price("Sale at price 1,234.50 per share.") == pytest.approx(1234.50)
    assert parse_price("Stock Gift at price 0.00 per share.") is None
    assert parse_price("") is None


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("Chief Executive Officer", 1.5),
        ("Chief Financial Officer", 1.5),
        ("Chairman of the Board", 1.5),
        ("President", 1.5),
        ("Vice President, Sales", 1.1),
        ("General Counsel", 1.1),
        ("Officer", 1.1),
        ("Officer and Director", 1.1),
        ("Director", 1.0),
        ("Director and Beneficial Owner of more than 10% of a Class of Security", 1.0),
        ("Beneficial Owner of more than 10% of a Class of Security", 0.8),
    ],
)
def test_role_weights(position, expected):
    assert role_weight(position)[0] == expected


def test_value_falls_back_to_filing_price_then_close(uptrend, ref):
    frame = _frame(
        _row(ref, 10, CEO, "Purchase at price 50.00 per share.", 1_000),  # Value NaN, price in text
        _row(ref, 12, DIRECTOR, "Purchase", 2_000),  # no Value, no price: shares x close
    )
    data = analyze_insider_transactions(frame, uptrend, "TEST")
    by_insider = {t.insider: t for t in data.transactions}
    ceo = by_insider[CEO[0]]
    assert ceo.value == pytest.approx(50_000.0)
    assert ceo.value_source == "shares x filing price"
    director = by_insider[DIRECTOR[0]]
    day = ref - pd.Timedelta(days=12)
    expected_close = float(uptrend["close"][uptrend.index <= day].iloc[-1])
    assert director.value == pytest.approx(2_000 * expected_close)
    assert director.value_source == "shares x close"


# ------------------------------------------------------------------ scoring


def test_big_ceo_purchase_outscores_small_director_purchase(monkeypatch, uptrend, ref):
    big = _evaluate(monkeypatch, _frame(_buy(ref, 10, CEO, 5_000_000)), uptrend)
    small = _evaluate(monkeypatch, _frame(_buy(ref, 10, DIRECTOR, 10_000)), uptrend)
    assert big.data_quality == LIVE and small.data_quality == LIVE
    assert small.score > 0
    assert big.score > small.score + 20
    assert big.score >= 20  # BULLISH on its own


def test_role_weighting_with_equal_dollars(monkeypatch, uptrend, ref):
    ceo = _evaluate(monkeypatch, _frame(_buy(ref, 10, CEO, 500_000)), uptrend)
    director = _evaluate(monkeypatch, _frame(_buy(ref, 10, DIRECTOR, 500_000)), uptrend)
    assert ceo.score > director.score


def test_cluster_requires_two_distinct_insiders_within_30_days(monkeypatch, uptrend, ref):
    close_together = _frame(_buy(ref, 20, DIRECTOR, 200_000), _buy(ref, 10, DIRECTOR_2, 200_000))
    far_apart = _frame(_buy(ref, 60, DIRECTOR, 200_000), _buy(ref, 15, DIRECTOR_2, 200_000))  # 45 days apart
    same_insider = _frame(_buy(ref, 20, DIRECTOR, 200_000), _buy(ref, 10, DIRECTOR, 200_000))

    together = analyze_insider_transactions(close_together, uptrend, "TEST")
    assert together.cluster_buy_detected is True
    assert together.max_buyers_in_cluster_window == 2
    assert analyze_insider_transactions(far_apart, uptrend, "TEST").cluster_buy_detected is False
    assert analyze_insider_transactions(same_insider, uptrend, "TEST").cluster_buy_detected is False

    assert _points(_evaluate(monkeypatch, close_together, uptrend), "Cluster buying") > 0
    far = _evaluate(monkeypatch, far_apart, uptrend)
    assert _points(far, "Cluster buying") == 0
    assert far.raw_data["cluster_buy_detected"] is False


def test_rows_outside_window_or_after_reference_are_ignored(monkeypatch, uptrend, ref):
    frame = _frame(
        _buy(ref, 200, CEO, 5_000_000),  # older than the 180-day window
        _buy(ref, -3, CFO, 5_000_000),  # after the reference date (look-ahead)
        _sale(ref, 20, DIRECTOR, 50_000),
    )
    data = analyze_insider_transactions(frame, uptrend, "TEST")
    assert data.reference_date == ref
    assert data.total_buy_transactions == 0
    assert data.total_sell_transactions == 1
    assert data.rows_before_window == 1
    assert data.rows_after_reference == 1
    assert [t.category for t in data.transactions] == [OPEN_MARKET_SALE]

    result = _evaluate(monkeypatch, frame, uptrend)
    assert result.score <= 0
    assert result.raw_data["total_buy_transactions"] == 0
    assert all(tx["insider"] != CFO[0] for tx in result.raw_data["transactions_table"])

    # A longer lookback brings the old purchase back in, the future one never.
    longer = analyze_insider_transactions(frame, uptrend, "TEST", lookback_days=365)
    assert longer.total_buy_transactions == 1
    assert longer.transactions[-1].insider == CEO[0]


def test_reference_date_is_last_bar_of_truncated_history(uptrend, ref):
    frame = _frame(_buy(ref, 0, CEO, 1_000_000))
    truncated = uptrend.iloc[:-50]
    data = analyze_insider_transactions(frame, truncated, "TEST")
    assert data.reference_date == truncated.index[-1]
    assert data.available is False  # every row is after the reference date
    assert data.rows_after_reference == 1


def test_reference_date_falls_back_to_today_without_prices(monkeypatch):
    today = pd.Timestamp.today().normalize()
    frame = _frame(_buy(today, 5, CEO, 1_000_000))
    data = analyze_insider_transactions(frame, None, "TEST")
    assert data.reference_date == today
    assert data.has_price_history is False
    assert data.total_buy_transactions == 1
    result = _evaluate(monkeypatch, frame, None)
    assert result.data_quality == PARTIAL
    assert result.score > 0


def test_sales_only_is_mildly_negative(monkeypatch, uptrend, ref):
    sale = _evaluate(monkeypatch, _frame(_sale(ref, 10, CEO, 5_000_000)), uptrend)
    buy = _evaluate(monkeypatch, _frame(_buy(ref, 10, CEO, 5_000_000)), uptrend)
    assert sale.score < 0
    assert sale.score >= -SALE_POINTS_CAP
    assert abs(sale.score) < 0.5 * buy.score
    assert sale.signal == NEUTRAL
    assert any("weak signal" in f for f in sale.key_factors)


def test_many_small_sellers_stay_mild(monkeypatch, uptrend, ref):
    sellers = [(f"SELLER {i}", "Officer") for i in range(8)]
    frame = _frame(*[_sale(ref, 5 + i, who, 300_000) for i, who in enumerate(sellers)])
    result = _evaluate(monkeypatch, frame, uptrend)
    assert -SALE_POINTS_CAP <= result.score < 0


def test_grants_exercises_and_gifts_are_excluded(monkeypatch, uptrend, ref):
    noise = [
        _row(ref, 5, CEO, "Stock Award(Grant) at price 0.00 per share.", 100_000, 0.0),
        _row(ref, 6, CFO, "Conversion of Exercise of derivative security at price 10.00 per share.", 50_000, 500_000.0),
        _row(ref, 7, DIRECTOR, "Stock Gift at price 0.00 per share.", 20_000, 0.0),
        _row(ref, 8, DIRECTOR_2, "", 30_000),
    ]
    data = analyze_insider_transactions(_frame(*noise), uptrend, "TEST")
    assert data.total_buy_transactions == 0
    assert data.total_sell_transactions == 0
    assert data.ignored_non_open_market_count == 4

    only_noise = _evaluate(monkeypatch, _frame(*noise), uptrend)
    assert only_noise.data_quality == LIVE  # real rows, just nothing open-market
    assert only_noise.score == 0.0
    assert only_noise.confidence <= 20
    assert only_noise.raw_data["ignored_non_open_market_count"] == 4
    assert any("absence of buying" in f.lower() for f in only_noise.key_factors)

    buy_row = _buy(ref, 10, DIRECTOR, 250_000)
    alone = _evaluate(monkeypatch, _frame(buy_row), uptrend)
    mixed = _evaluate(monkeypatch, _frame(buy_row, *noise), uptrend)
    assert mixed.score == alone.score


def test_buy_after_drawdown_gets_boost(monkeypatch, ref):
    falling = make_ohlcv(periods=300, start=140.0, end=90.0)
    rising = make_ohlcv(periods=300, start=100.0, end=140.0)
    frame = _frame(_buy(falling.index[-1], 5, CEO, 1_000_000))

    boosted = _evaluate(monkeypatch, frame, falling)
    plain = _evaluate(monkeypatch, frame, rising)
    assert _points(boosted, "Buys after") > 0
    assert _points(plain, "Buys after") == 0
    avg_dd = next(m["value"] for m in boosted.raw_data["metrics_table"] if m["metric"] == "Avg drawdown at purchase")
    assert "below 252-day high" in avg_dd
    assert boosted.raw_data["avg_drawdown_at_purchase"] >= 0.20
    assert any("drawdown" in f for f in boosted.key_factors)


def test_track_record_uses_history_before_the_window(monkeypatch, uptrend, ref):
    frame = _frame(
        _buy(ref, 250, CEO, 200_000),
        _buy(ref, 200, CEO, 200_000),
        _buy(ref, 10, CEO, 200_000),
    )
    data = analyze_insider_transactions(frame, uptrend, "TEST")
    assert data.total_buy_transactions == 1  # only the recent one is in the window
    assert data.buyer_track_evaluated == 2  # the recent one has no 60 bars of forward data yet
    assert data.buyer_track_hits == 2
    result = _evaluate(monkeypatch, frame, uptrend)
    assert _points(result, "Buyer track record") > 0


def test_track_record_skips_gracefully_with_short_history(monkeypatch):
    short = make_ohlcv(periods=40)
    frame = _frame(_buy(short.index[-1], 20, CEO, 200_000))
    data = analyze_insider_transactions(frame, short, "TEST")
    assert data.buyer_track_evaluated == 0
    assert data.track_records == []
    result = _evaluate(monkeypatch, frame, short)
    assert _points(result, "Buyer track record") == 0
    track = next(m["value"] for m in result.raw_data["metrics_table"] if m["metric"].startswith("Buyer track record"))
    assert "Not enough history" in track


def test_confidence_grows_with_count_and_recency(monkeypatch, uptrend, ref):
    one_old = _evaluate(monkeypatch, _frame(_buy(ref, 150, DIRECTOR, 100_000)), uptrend)
    several_recent = _evaluate(
        monkeypatch,
        _frame(_buy(ref, 3, DIRECTOR, 100_000), _buy(ref, 5, DIRECTOR_2, 100_000), _buy(ref, 8, CFO, 100_000)),
        uptrend,
    )
    assert several_recent.confidence > one_old.confidence


# ------------------------------------------------------------------ output shape


def test_raw_data_shape_links_and_table(monkeypatch, uptrend, ref):
    frame = _frame(
        _buy(ref, 4, CEO, 1_000_000, url="https://www.sec.gov/Archives/edgar/data/1/form4.xml"),
        _sale(ref, 9, DIRECTOR, 80_000),
        _row(ref, 12, CFO, "Stock Award(Grant) at price 0.00 per share.", 5_000, 0.0),
    )
    result = _evaluate(monkeypatch, frame, uptrend, ticker="nvda")
    raw = result.raw_data

    for key in (
        "transactions_table",
        "net_buy_value",
        "net_buy_shares",
        "cluster_buy_detected",
        "total_buy_transactions",
        "total_sell_transactions",
        "ignored_non_open_market_count",
        "lookback_days",
        "metrics_table",
        "score_breakdown",
        "links",
        "table",
    ):
        assert key in raw, key

    tx_keys = {"insider", "position", "type", "category", "shares", "value", "price", "date", "is_open_market", "url"}
    assert all(tx_keys <= set(tx) for tx in raw["transactions_table"])
    assert [tx["date"] for tx in raw["transactions_table"]] == sorted((tx["date"] for tx in raw["transactions_table"]), reverse=True)
    assert raw["transactions_table"][0]["url"].startswith("https://www.sec.gov/")

    edgar = next(link["url"] for link in raw["links"] if "sec.gov" in link["url"])
    assert "type=4" in edgar
    assert "CIK=NVDA" in edgar
    assert any("openinsider.com" in link["url"] and "NVDA" in link["url"] for link in raw["links"])
    assert any("finance.yahoo.com/quote/NVDA/insider-transactions" in link["url"] for link in raw["links"])

    table = raw["table"]
    assert "Filing" in table["columns"]
    assert all(len(row) == len(table["columns"]) for row in table["rows"])
    assert table["rows"][0][-1].startswith("https://www.sec.gov/")

    assert {"factor", "points"} <= set(raw["score_breakdown"][0])
    assert result.score == pytest.approx(sum(b["points"] for b in raw["score_breakdown"]), abs=0.15)
    assert {"metric", "value"} <= set(raw["metrics_table"][0])
    assert result.data_as_of == (ref - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    json.dumps(raw)  # must be JSON serializable for the web API

    text = " ".join([result.summary, *result.key_factors, *(m["value"] for m in raw["metrics_table"])])
    assert "—" not in text and "–" not in text
