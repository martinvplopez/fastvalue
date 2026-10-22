"""Social sentiment (Reddit, StockTwits): parsing, scorer fallback chain, minimum count, snapshot fields (no network)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

import config.settings as settings
import data.social_sentiment as soc
from agents.sentiment_agent import SentimentAgent
from data.news_sentiment import NewsSentimentData, load_sentiment_history, record_sentiment_snapshot, summarize_headlines, Headline

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def post(text, hours=1, tag=None, source="StockTwits"):
    return soc.SocialPost(text=text, source=source, created=NOW - timedelta(hours=hours), tag=tag)


def many(n, tag=None, prefix="post", source="StockTwits"):
    return [post(f"{prefix} number {i} about $ACME", hours=i % 24, tag=tag, source=source) for i in range(n)]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (soc.SOURCES_ENV, soc.REDDIT_ID_ENV, soc.REDDIT_SECRET_ENV, soc.OLLAMA_MODEL_ENV, soc.API_SCORER_ENV,
                 soc.ANTHROPIC_KEY_ENV, soc.OLLAMA_HOST_ENV):
        monkeypatch.delenv(name, raising=False)
    soc._fetch_posts.cache_clear()
    yield
    soc._fetch_posts.cache_clear()


def test_parse_stocktwits_tags_and_junk():
    payload = {"messages": [
        {"body": "Loading up $ACME", "created_at": "2026-10-09T10:00:00Z", "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "meh", "created_at": "2026-10-09T09:00:00Z", "entities": {"sentiment": None}},
        {"body": "out", "created_at": "2026-10-09T08:00:00Z", "entities": {"sentiment": {"basic": "Bearish"}}},
        {"body": "", "created_at": "2026-10-09T08:00:00Z"}, {"body": "no date"}, "junk", None,
    ]}
    posts = soc.parse_stocktwits(payload)
    assert [p.tag for p in posts] == [1, None, -1]
    assert soc.parse_stocktwits({"messages": "x"}) == [] and soc.parse_stocktwits(None) == []


def test_parse_reddit_listing():
    payload = {"data": {"children": [
        {"data": {"title": "ACME earnings", "selftext": "great quarter", "created_utc": 1791540000}},
        {"data": {"title": "", "selftext": "", "created_utc": 1791540000}},
        {"data": {"title": "no time"}}, {"kind": "t3"}, "x",
    ]}}
    posts = soc.parse_reddit_listing(payload)
    assert len(posts) == 1 and posts[0].text == "ACME earnings. great quarter" and posts[0].tag is None


def test_mentions_ticker():
    assert soc.mentions_ticker("buying $acme today", "ACME")
    assert soc.mentions_ticker("ACME to the moon", "ACME")
    assert not soc.mentions_ticker("acmeish things", "ACME")
    assert not soc.mentions_ticker("a day in F1", "F")  # short tickers need a cashtag
    assert soc.mentions_ticker("$F is cheap", "F")


def test_below_minimum_post_count_does_not_count():
    reading = soc.summarize_posts("ACME", "StockTwits", many(soc.MIN_POSTS - 1, tag=1), now=NOW)
    assert reading.posts == soc.MIN_POSTS - 1 and not reading.counts and reading.score is None
    assert "need" in reading.status


def test_minimum_met_uses_author_tags_and_weights():
    posts = many(soc.MIN_POSTS, tag=1)
    reading = soc.summarize_posts("ACME", "StockTwits", posts, now=NOW)
    assert reading.counts and reading.score == pytest.approx(1.0)
    assert reading.scorers == {"author tag": soc.MIN_POSTS} and reading.tagged == soc.MIN_POSTS


def test_duplicates_stale_and_reddit_mentions_dropped():
    posts = [post("same text $ACME")] * 5 + [post("old $ACME", hours=24 * 30)] + [post("unrelated chatter", source="Reddit")]
    reading = soc.summarize_posts("ACME", "Reddit", posts, now=NOW)
    assert reading.posts == 1  # one distinct fresh mention (the unrelated Reddit post and the old one are dropped)


def test_fallback_to_lexicon_when_nothing_else_configured():
    scores, used = soc.score_posts([post("ACME beats estimates, bullish"), post("ACME plunges, bagholder"), post("nothing here")], "ACME")
    assert used == ["lexicon"] * 3
    assert scores[0] > 0 > scores[1] and scores[2] == 0


def test_ollama_used_when_server_answers(monkeypatch):
    monkeypatch.setenv(soc.OLLAMA_MODEL_ENV, "tiny:1b")
    calls = []

    def fake_http(request, label, timeout):
        url = request.full_url
        calls.append(url)
        if url.endswith("/api/tags"):
            return {"models": [{"name": "tiny:1b"}]}
        n = len(json.loads(request.data)["messages"][0]["content"].strip().splitlines()[-1:]) and 0
        body = json.loads(request.data)
        count = sum(1 for line in body["messages"][0]["content"].splitlines() if line[:1].isdigit() and ". " in line[:4])
        return {"message": {"content": json.dumps({"scores": [0.5] * count})}}

    monkeypatch.setattr(soc, "_http", fake_http)
    posts = [post(f"text {i}") for i in range(soc.OLLAMA_BATCH + 3)]  # two batches
    scores, used = soc.score_posts(posts, "ACME")
    assert set(used) == {"ollama:tiny:1b"} and scores == [0.5] * len(posts)
    assert sum(u.endswith("/api/chat") for u in calls) == 2


def test_ollama_bad_json_falls_back_to_lexicon(monkeypatch):
    monkeypatch.setenv(soc.OLLAMA_MODEL_ENV, "tiny:1b")

    def fake_http(request, label, timeout):
        if request.full_url.endswith("/api/tags"):
            return {"models": [{"name": "tiny:1b"}]}
        return {"message": {"content": "sorry, I cannot do that"}}

    monkeypatch.setattr(soc, "_http", fake_http)
    scores, used = soc.score_posts([post("ACME beats estimates")], "ACME")
    assert used == ["lexicon"] and scores[0] > 0


def test_ollama_wrong_count_or_text_scores_rejected():
    with pytest.raises(ValueError):
        soc._parse_scores({"scores": [1, 0]}, 3)
    with pytest.raises(ValueError):
        soc._parse_scores({"scores": [1, "bullish"]}, 2)
    assert soc._parse_scores('noise {"scores": [5, -9]} noise', 2) == [1.0, -1.0]  # clamped


def test_ollama_server_down_reports_status(monkeypatch):
    monkeypatch.setenv(soc.OLLAMA_MODEL_ENV, "tiny:1b")
    monkeypatch.setattr(soc, "_http", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Ollama: refused")))
    model, status = soc.ollama_status()
    assert model is None and "no local server" in status


def test_api_scorer_needs_flag_and_key(monkeypatch):
    monkeypatch.setenv(soc.ANTHROPIC_KEY_ENV, "k")
    assert not soc.api_scorer_enabled()  # a key alone does not enable it
    monkeypatch.setenv(soc.API_SCORER_ENV, "1")
    assert soc.api_scorer_enabled()

    def fake_http(request, label, timeout):
        assert request.get_header("X-api-key") == "k" and json.loads(request.data)["model"] == "claude-haiku-4-5"
        return {"content": [{"type": "text", "text": '{"scores": [-0.5]}'}]}

    monkeypatch.setattr(soc, "_http", fake_http)
    scores, used = soc.score_posts([post("text")], "ACME")
    assert scores == [-0.5] and used == ["claude-haiku-4-5"]


def test_api_failure_falls_back_to_lexicon(monkeypatch):
    monkeypatch.setenv(soc.ANTHROPIC_KEY_ENV, "k")
    monkeypatch.setenv(soc.API_SCORER_ENV, "1")
    monkeypatch.setattr(soc, "_http", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Anthropic API: HTTP 529")))
    _, used = soc.score_posts([post("ACME surges")], "ACME")
    assert used == ["lexicon"]


def test_sources_are_opt_in_and_reddit_needs_credentials(monkeypatch):
    assert soc.configured_sources() == ([], [])
    monkeypatch.setenv(soc.SOURCES_ENV, "reddit, stocktwits, bogus")
    sources, notes = soc.configured_sources()
    assert [s.name for s in sources] == ["StockTwits"] and "REDDIT_CLIENT_ID" in notes[0]
    monkeypatch.setenv(soc.REDDIT_ID_ENV, "id")
    monkeypatch.setenv(soc.REDDIT_SECRET_ENV, "secret")
    assert [s.name for s in soc.configured_sources()[0]] == ["Reddit", "StockTwits"]


def test_reddit_source_oauth_then_search(monkeypatch):
    seen = []

    def fake_http(request, label, timeout):
        seen.append(request.full_url)
        if "access_token" in request.full_url:
            assert request.get_header("Authorization").startswith("Basic ")
            return {"access_token": "tok"}
        assert request.get_header("Authorization") == "bearer tok" and "wallstreetbets" in request.full_url
        return {"data": {"children": [{"data": {"title": "$ACME up", "created_utc": 1791540000}}]}}

    monkeypatch.setattr(soc, "_http", fake_http)
    assert len(soc.RedditSource("id", "secret").fetch("ACME")) == 1 and len(seen) == 2


def test_fetch_social_sentiment_off_by_default_and_error_status(monkeypatch):
    assert soc.fetch_social_sentiment("ACME").enabled is False
    monkeypatch.setenv(soc.SOURCES_ENV, "stocktwits")
    monkeypatch.setattr(soc, "_http", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("StockTwits: HTTP 403")))
    result = soc.fetch_social_sentiment("ACME")
    assert result.enabled and result.readings == [] and any("HTTP 403" in n for n in result.notes)
    assert result.snapshot_fields() is None


def test_snapshot_fields_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORE_DIR", tmp_path)
    reading = soc.summarize_posts("ACME", "StockTwits", many(25, tag=-1), now=NOW)
    data = soc.SocialSentimentData(ticker="ACME", enabled=True, readings=[reading])
    fields = data.snapshot_fields()
    assert fields["posts"] == 25 and fields["sources"]["StockTwits"]["posts"] == 25 and fields["scorers"] == {"author tag": 25}
    assert fields["score"] == pytest.approx(-1.0)

    news = summarize_headlines("ACME", [Headline(title="Acme beats estimates", source="Yahoo Finance", published=NOW - timedelta(hours=2))], now=NOW)
    news.sources = ["Yahoo Finance"]
    assert record_sentiment_snapshot(news, today=date(2026, 10, 9), social=fields)
    row = load_sentiment_history("ACME")[-1]
    assert row["social"]["posts"] == 25 and row["score"] == pytest.approx(news.weighted_score, abs=1e-3)


def test_agent_shows_social_evidence_without_changing_score():
    news = summarize_headlines("ACME", [Headline(title="Acme beats estimates", source="Yahoo Finance", published=datetime.now(timezone.utc) - timedelta(hours=2))])
    news.sources = ["Yahoo Finance"]
    agent = SentimentAgent()
    base = agent.evaluate_with_data(news)
    reading = soc.summarize_posts("ACME", "StockTwits", many(25, tag=-1), now=NOW)
    reading.sent_off_machine = "the ticker symbol went to StockTwits"
    social = soc.SocialSentimentData(ticker="ACME", enabled=True, readings=[reading], notes=[reading.status])
    out = agent.evaluate_with_data(news, social)
    assert out.score == base.score and out.confidence == base.confidence
    metrics = {m["metric"]: m["value"] for m in out.raw_data["metrics_table"]}
    assert "25 posts" in metrics["Social: StockTwits"] and "author tag 25" in metrics["Social scorer (StockTwits)"]
    assert "ticker symbol" in metrics["Left this machine (StockTwits)"]
    assert agent.default_weight == 0.0
