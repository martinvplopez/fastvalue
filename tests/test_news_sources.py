"""News API sources (Alpha Vantage, Finnhub), provider sentiment blending and daily snapshots (no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import config.settings as settings
import data.news_sentiment as news
from agents.sentiment_agent import SentimentAgent
from data.news_sentiment import (
    AlphaVantageNewsSource,
    FinnhubNewsSource,
    Headline,
    YahooNewsSource,
    all_sources,
    extra_sources,
    load_sentiment_history,
    parse_alpha_vantage_item,
    parse_finnhub_item,
    record_sentiment_snapshot,
    summarize_headlines,
)
from tests.conftest import make_ohlcv

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
AV_ITEM = {
    "title": "Acme beats estimates and raises guidance",
    "url": "https://example.com/acme",
    "time_published": "20260927T090000",
    "source": "Example Wire",
    "summary": "Quarterly results.",
    "overall_sentiment_score": 0.12,
    "ticker_sentiment": [
        {"ticker": "OTHER", "relevance_score": "0.9", "ticker_sentiment_score": "-0.5"},
        {"ticker": "ACME", "relevance_score": "0.65", "ticker_sentiment_score": "0.41", "ticker_sentiment_label": "Bullish"},
    ],
}


def test_parse_alpha_vantage_item_uses_the_tickers_own_sentiment():
    h = parse_alpha_vantage_item(AV_ITEM, "acme")
    assert h.provider_score == pytest.approx(0.41) and h.relevance == pytest.approx(0.65)
    assert h.published == datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc)
    assert h.publisher == "Example Wire" and h.url == "https://example.com/acme" and h.source == "Alpha Vantage"
    other = parse_alpha_vantage_item({**AV_ITEM, "ticker_sentiment": []}, "ACME")
    assert other.provider_score == pytest.approx(0.12) and other.relevance is None
    assert parse_alpha_vantage_item({"title": ""}, "ACME") is None
    assert parse_alpha_vantage_item("junk", "ACME") is None


def test_parse_finnhub_item():
    h = parse_finnhub_item({"headline": "Acme announces buyback", "datetime": 1790000000, "source": "Wire", "url": "https://x.y/z", "summary": "s"})
    assert h.title == "Acme announces buyback" and h.provider_score is None and h.source == "Finnhub"
    assert h.published == datetime.fromtimestamp(1790000000, tz=timezone.utc)
    assert parse_finnhub_item({"headline": None}) is None


def test_sources_switch_on_with_environment_keys(monkeypatch):
    monkeypatch.delenv(news.ALPHA_VANTAGE_KEY_ENV, raising=False)
    monkeypatch.delenv(news.FINNHUB_KEY_ENV, raising=False)
    assert extra_sources() == [] and [s.name for s in all_sources()] == ["Yahoo Finance"]
    monkeypatch.setenv(news.ALPHA_VANTAGE_KEY_ENV, "av-key")
    monkeypatch.setenv(news.FINNHUB_KEY_ENV, "fh-key")
    names = [s.name for s in all_sources()]
    assert names == ["Yahoo Finance", "Alpha Vantage", "Finnhub"]
    assert isinstance(all_sources()[1], AlphaVantageNewsSource) and isinstance(all_sources()[2], FinnhubNewsSource)


def test_provider_scores_and_relevance_drive_the_weighting():
    lexicon_only = Headline("Acme misses estimates", "Yahoo Finance", published=NOW - timedelta(hours=1))
    provider_pos = Headline("Acme holds annual meeting", "Alpha Vantage", published=NOW - timedelta(hours=1),
                            provider_score=0.6, relevance=1.0, url="https://a/1")
    provider_irrelevant = Headline("Sector roundup mentions Acme", "Alpha Vantage", published=NOW - timedelta(hours=1),
                                   provider_score=-0.9, relevance=0.05, url="https://a/2")
    data = summarize_headlines("ACME", [lexicon_only, provider_pos, provider_irrelevant], now=NOW)
    by_title = {h.title: h for h in data.headlines}
    assert by_title["Acme misses estimates"].score_source == "lexicon" and by_title["Acme misses estimates"].score < 0
    assert by_title["Acme holds annual meeting"].score_source == "provider" and by_title["Acme holds annual meeting"].score == pytest.approx(0.6)
    # Relevance floors at NEWS_MIN_RELEVANCE, so the barely related item weighs a fifth of a fully relevant one.
    assert by_title["Sector roundup mentions Acme"].weight == pytest.approx(by_title["Acme holds annual meeting"].weight * settings.NEWS_MIN_RELEVANCE, rel=1e-3)
    assert data.provider_scored == 2
    # Lexicon -0.7 and provider +0.6 at full weight, provider -0.9 at a fifth: the mean lands below zero.
    expected = (-0.7 + 0.6 - 0.9 * settings.NEWS_MIN_RELEVANCE) / (2 + settings.NEWS_MIN_RELEVANCE)
    assert data.weighted_score == pytest.approx(expected, abs=0.01)
    # Without the lexicon item the relevant positive provider score outweighs the irrelevant negative one.
    providers_only = summarize_headlines("ACME", [provider_pos, provider_irrelevant], now=NOW)
    assert providers_only.weighted_score > 0


def test_api_failure_is_isolated_and_never_leaks_the_key(monkeypatch):
    monkeypatch.setenv(news.ALPHA_VANTAGE_KEY_ENV, "SECRET-KEY-123")
    monkeypatch.setattr(YahooNewsSource, "fetch", lambda self, ticker: [Headline("Acme beats estimates", "Yahoo Finance", published=datetime.now(timezone.utc))])
    news._fetch_source.cache_clear()
    data = news.fetch_news_sentiment("ACME")
    assert data.available and data.sources == ["Yahoo Finance"]
    assert "Alpha Vantage" in data.source_errors
    assert "SECRET-KEY-123" not in " ".join(data.source_errors.values())
    result = SentimentAgent().evaluate_with_data(data)
    assert result.data_quality == "PARTIAL" and any("Alpha Vantage news could not be loaded" in f for f in result.key_factors)


def test_alpha_vantage_rejects_payload_without_feed(monkeypatch):
    monkeypatch.setattr(news, "_get_json", lambda url, label, timeout=20.0: {"Information": "rate limit reached"})
    with pytest.raises(RuntimeError) as exc:
        AlphaVantageNewsSource("k").fetch("ACME")
    assert "rate limit" in str(exc.value)
    monkeypatch.setattr(news, "_get_json", lambda url, label, timeout=20.0: {"feed": [AV_ITEM]})
    assert AlphaVantageNewsSource("k").fetch("ACME")[0].provider_score == pytest.approx(0.41)
    monkeypatch.setattr(news, "_get_json", lambda url, label, timeout=20.0: [{"headline": "Acme news", "datetime": 1790000000}])
    assert FinnhubNewsSource("k").fetch("ACME")[0].title == "Acme news"


def test_daily_snapshot_is_recorded_once_and_loaded(monkeypatch):
    data = summarize_headlines("ACME", [Headline("Acme beats estimates", "Yahoo Finance", published=NOW - timedelta(hours=1))], now=NOW)
    assert record_sentiment_snapshot(data, today=NOW.date()) is True
    assert record_sentiment_snapshot(data, today=NOW.date()) is False  # one line a day
    assert record_sentiment_snapshot(data, today=(NOW + timedelta(days=1)).date()) is True
    history = load_sentiment_history("acme")
    assert [h["date"] for h in history] == ["2026-09-27", "2026-09-28"] and history[0]["score"] > 0
    empty = summarize_headlines("ACME", [], now=NOW)
    assert record_sentiment_snapshot(empty) is False

    monkeypatch.setattr(YahooNewsSource, "fetch", lambda self, ticker: [Headline("Acme beats estimates", "Yahoo Finance", published=datetime.now(timezone.utc))])
    news._fetch_source.cache_clear()
    result = SentimentAgent().evaluate("NEWCO", make_ohlcv())
    assert result.data_quality == "LIVE"
    assert load_sentiment_history("NEWCO")[-1]["headlines"] == 1
    assert result.raw_data["sentiment_history"][-1]["headlines"] == 1
