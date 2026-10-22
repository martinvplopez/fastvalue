"""Unit tests for the News Sentiment Agent and its headline provider (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import yfinance

import agents.sentiment_agent as sentiment_agent
import data.news_sentiment as news
from agents.base import LIVE, UNAVAILABLE
from agents.registry import get_agent_classes
from agents.sentiment_agent import SentimentAgent
from data.news_sentiment import Headline, YahooNewsSource, parse_yahoo_news_item, score_text, summarize_headlines
from tests.conftest import make_ohlcv

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)

POSITIVE_TITLES = [
    "Acme beats estimates and raises guidance",
    "Analyst upgrades Acme to outperform",
    "Acme shares soar after FDA approval",
    "Acme announces $5 billion buyback",
    "Acme signs partnership with major cloud provider",
]
NEGATIVE_TITLES = [
    "Acme misses estimates and cuts guidance",
    "Analyst downgrades Acme to underperform",
    "Acme shares plunge as SEC opens probe",
    "Acme faces class action lawsuit over recall",
    "Acme announces layoffs amid fraud investigation",
]


def _headline(title: str, hours_ago: float = 2.0, url: str | None = None, now: datetime = NOW) -> Headline:
    return Headline(
        title=title,
        source="Test",
        publisher="Test Wire",
        url=url,
        published=now - timedelta(hours=hours_ago),
    )


def _patch_source(monkeypatch: pytest.MonkeyPatch, titles: list[str]) -> None:
    now = datetime.now(timezone.utc)

    def fake_fetch(self, ticker):
        return [_headline(t, hours_ago=2 + i, url=f"https://example.com/{i}", now=now) for i, t in enumerate(titles)]

    monkeypatch.setattr(YahooNewsSource, "fetch", fake_fetch)


def test_registered_with_metadata():
    classes = get_agent_classes()
    assert classes["sentiment"] is SentimentAgent
    meta = SentimentAgent.metadata()
    assert meta["name"] == "News Sentiment Agent"
    assert meta["category"] == "Sentiment"
    assert meta["score_scale"] == 40.0
    assert meta["default_weight"] == 0.0
    assert meta["point_in_time"] is False


def test_lexicon_polarity_and_phrases():
    assert score_text("Acme beats estimates")[0] > 0
    assert score_text("Acme misses estimates")[0] < 0
    score, terms = score_text("Acme cuts guidance")
    assert score == pytest.approx(-0.9)
    assert terms == ("cuts guidance",)
    # Longest phrase wins: "record loss" is negative even though "record" is positive.
    assert score_text("Acme posts record loss")[0] < 0
    assert score_text("Acme posts record revenue")[0] > 0
    assert score_text("Acme holds annual meeting") == (0.0, ())
    # Scores stay within -1..1.
    assert score_text("Acme soars, surges, skyrockets on upgrade and record buyback")[0] == 1.0


def test_negation_flips_polarity():
    assert score_text("Acme does not beat estimates")[0] < 0
    assert score_text("Acme fails to beat estimates")[0] < 0
    assert score_text("Acme doesn't expect layoffs")[0] > 0
    assert score_text("No recall planned, Acme says")[0] > 0
    _, terms = score_text("Acme fails to beat estimates")
    assert terms == ("not beat estimates",)
    # Negation only reaches a few tokens forward.
    assert score_text("Not a word about the plan, but Acme beats estimates")[0] > 0


def test_question_headlines_are_damped():
    assert score_text("Is Acme a buy after it beats estimates?")[0] == pytest.approx(0.35)


def test_positive_vs_negative_headline_sets(monkeypatch):
    _patch_source(monkeypatch, POSITIVE_TITLES)
    positive = SentimentAgent().evaluate("ACME", make_ohlcv())
    news._fetch_source.cache_clear()
    _patch_source(monkeypatch, NEGATIVE_TITLES)
    negative = SentimentAgent().evaluate("ACME", make_ohlcv())

    assert positive.score > 20 and positive.signal == "BULLISH"
    assert negative.score < -20 and negative.signal == "BEARISH"
    for result in (positive, negative):
        assert result.data_quality == LIVE
        assert 0 < result.confidence <= 60.0
        assert result.data_as_of is not None


def test_raw_data_shape_and_breakdown(monkeypatch):
    _patch_source(monkeypatch, POSITIVE_TITLES + ["Acme holds annual meeting"])
    result = SentimentAgent().evaluate("ACME", make_ohlcv())
    raw = result.raw_data

    table = raw["table"]
    assert table["columns"] == ["Date", "Source", "Headline", "Score"]
    assert len(table["rows"]) == 6
    assert all(isinstance(cell, str) for row in table["rows"] for cell in row)

    first = raw["headlines"][0]
    assert {"title", "publisher", "url", "published", "score"} <= set(first)
    datetime.fromisoformat(first["published"])

    assert raw["links"] == [{"label": "Yahoo Finance news", "url": "https://finance.yahoo.com/quote/ACME/news/"}]
    assert all(isinstance(row["value"], str) for row in raw["metrics_table"])
    total = sum(row["points"] for row in raw["score_breakdown"])
    assert total == pytest.approx(result.score, abs=0.3)

    text = json.dumps([result.summary, result.key_factors, raw["score_breakdown"], raw["metrics_table"]])
    assert "\u2013" not in text and "\u2014" not in text


def test_shrinkage_and_confidence_scale_with_count():
    one = summarize_headlines("ACME", [_headline(POSITIVE_TITLES[0])], now=NOW)
    many = summarize_headlines("ACME", [_headline(t, hours_ago=i + 1) for i, t in enumerate(POSITIVE_TITLES)], now=NOW)
    agent = SentimentAgent()
    single, several = agent.evaluate_with_data(one), agent.evaluate_with_data(many)
    # Even though one headline alone is strongly positive, it is shrunk toward 0.
    assert one.weighted_score > 0.5
    assert single.score < one.weighted_score * 100 * 0.5
    assert single.confidence < several.confidence <= 60.0


def test_recency_weighting():
    fresh_positive = _headline("Analyst upgrades Acme to outperform", hours_ago=6)
    old_negative = _headline("Analyst downgrades Acme to underperform", hours_ago=10 * 24)
    data = summarize_headlines("ACME", [old_negative, fresh_positive], now=NOW)

    assert data.weighted_score > 0
    weights = {h.title: h.weight for h in data.headlines}
    assert weights[old_negative.title] < weights[fresh_positive.title] / 5
    assert news.recency_weight(3.0) == pytest.approx(0.5)

    # Items older than 14 days are ignored entirely.
    stale = _headline("Acme files for bankruptcy", hours_ago=15 * 24)
    data = summarize_headlines("ACME", [stale, fresh_positive], now=NOW)
    assert [h.title for h in data.headlines] == [fresh_positive.title]
    assert data.dropped_stale == 1


def test_near_duplicates_and_undated_are_dropped():
    items = [
        _headline("Acme beats estimates, raises guidance", hours_ago=1),
        _headline("Acme Beats Estimates; Raises Guidance!", hours_ago=3),
        _headline("Totally different story about Acme", hours_ago=2, url="https://example.com/a"),
        _headline("Another different story entirely", hours_ago=4, url="https://example.com/a"),
        Headline(title="Undated item", source="Test"),
    ]
    data = summarize_headlines("ACME", items, now=NOW)
    assert len(data.headlines) == 2
    assert data.dropped_duplicates == 2
    assert data.dropped_undated == 1
    assert data.headlines[0].published == NOW - timedelta(hours=1)  # newest copy kept


NEW_SHAPE = {
    "id": "abc",
    "content": {
        "title": "Acme beats estimates",
        "summary": "Quarterly results topped forecasts.",
        "pubDate": "2026-09-27T10:00:00Z",
        "provider": {"displayName": "Wire Service"},
        "canonicalUrl": {"url": "https://publisher.example.com/story"},
        "clickThroughUrl": {"url": "https://finance.yahoo.com/story"},
    },
}
OLD_SHAPE = {
    "uuid": "def",
    "title": "Acme misses estimates",
    "publisher": "Old Wire",
    "link": "https://old.example.com/story",
    "providerPublishTime": int(datetime(2026, 9, 26, 8, 0, tzinfo=timezone.utc).timestamp()),
}


def test_parse_both_yahoo_news_shapes():
    new = parse_yahoo_news_item(NEW_SHAPE)
    assert new.title == "Acme beats estimates"
    assert new.publisher == "Wire Service"
    assert new.url == "https://publisher.example.com/story"
    assert new.published == datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc)
    assert new.summary == "Quarterly results topped forecasts."

    old = parse_yahoo_news_item(OLD_SHAPE)
    assert old.title == "Acme misses estimates"
    assert old.publisher == "Old Wire"
    assert old.url == "https://old.example.com/story"
    assert old.published == datetime(2026, 9, 26, 8, 0, tzinfo=timezone.utc)

    # Click-through URL is the fallback; unsafe URL schemes and empty titles are rejected.
    no_canonical = {"content": {**NEW_SHAPE["content"], "canonicalUrl": None}}
    assert parse_yahoo_news_item(no_canonical).url == "https://finance.yahoo.com/story"
    unsafe = {**OLD_SHAPE, "link": "javascript:alert(1)"}
    assert parse_yahoo_news_item(unsafe).url is None
    assert parse_yahoo_news_item({"content": {"title": "  "}}) is None
    assert parse_yahoo_news_item("not a dict") is None


def test_yahoo_source_fetch_handles_mixed_payload(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            self.news = [NEW_SHAPE, OLD_SHAPE, {"junk": True}]

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    headlines = YahooNewsSource().fetch("ACME")
    assert [h.title for h in headlines] == ["Acme beats estimates", "Acme misses estimates"]
    assert all(h.source == "Yahoo Finance" for h in headlines)


def test_provider_exception_is_unavailable():
    # conftest makes yfinance.Ticker raise: the provider must report, not raise.
    data = news.fetch_news_sentiment("ACME")
    assert data.available is False
    assert "Yahoo Finance" in data.source_errors
    result = SentimentAgent().evaluate("ACME", make_ohlcv())
    assert result.data_quality == UNAVAILABLE
    assert result.confidence == 0.0


def test_source_fetch_raising_is_unavailable(monkeypatch):
    def broken(self, ticker):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(YahooNewsSource, "fetch", broken)
    result = SentimentAgent().evaluate("ACME", make_ohlcv())
    assert result.data_quality == UNAVAILABLE
    assert "rate limited" in (result.error or "")


def test_agent_survives_provider_raising(monkeypatch):
    def boom(ticker):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(sentiment_agent, "fetch_news_sentiment", boom)
    assert SentimentAgent().evaluate("ACME", make_ohlcv()).data_quality == UNAVAILABLE


def test_no_headlines_is_unavailable(monkeypatch):
    _patch_source(monkeypatch, [])
    result = SentimentAgent().evaluate("ACME", make_ohlcv())
    assert result.data_quality == UNAVAILABLE
    assert "No dated news headlines" in result.summary
