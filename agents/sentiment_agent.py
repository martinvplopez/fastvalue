"""News Sentiment Domain Agent (timing panel, informational by default).

Turns recency-weighted headline sentiment from :mod:`data.news_sentiment` into an
assessment. Sources: Yahoo Finance (always), Alpha Vantage NEWS_SENTIMENT and
Finnhub company news when their API keys are set. A provider's own sentiment
score is used when present (Alpha Vantage), else the lexicon. Headline tone is
noisy and has no replayable history yet, so the agent ships with
``default_weight = 0``: it is shown in the UI and records one snapshot per ticker
per day so a history builds up for a later validation. Confidence is capped.
"""

from __future__ import annotations

import logging
from collections import Counter
from urllib.parse import quote

import pandas as pd

from agents.base import LIVE, PARTIAL, TIMING, AgentAssessment, BaseAgent
from agents.registry import register_agent
from data.social_sentiment import SocialSentimentData, fetch_social_sentiment
from data.news_sentiment import (
    MAX_AGE_DAYS,
    NewsSentimentData,
    ScoredHeadline,
    fetch_news_sentiment,
    load_sentiment_history,
    record_sentiment_snapshot,
)

logger = logging.getLogger(__name__)

SHRINKAGE_K = 4.0  # score is multiplied by n / (n + K): few headlines pull it toward 0
MAX_CONFIDENCE = 60.0
BASE_CONFIDENCE = 10.0
CONFIDENCE_HALF_N = 6.0  # headline count at which half of the confidence range is reached
NEUTRAL_BAND = 0.05  # |headline score| below this counts as neutral
MAX_TERMS_SHOWN = 5


def _tone_label(mean: float) -> str:
    if mean >= 0.15:
        return "positive"
    if mean >= 0.05:
        return "slightly positive"
    if mean > -0.05:
        return "neutral"
    if mean > -0.15:
        return "slightly negative"
    return "negative"


def _top_terms(headlines: list[ScoredHeadline], positive: bool) -> list[str]:
    counts: Counter[str] = Counter()
    for h in headlines:
        if (h.score >= NEUTRAL_BAND) if positive else (h.score <= -NEUTRAL_BAND):
            counts.update(h.terms)
    return [term for term, _ in counts.most_common(MAX_TERMS_SHOWN)]


def _headline_rows(headlines: list[ScoredHeadline]) -> list[list[str]]:
    return [
        [
            h.published.strftime("%Y-%m-%d %H:%M"),
            h.publisher or h.source,
            h.title,
            f"{h.score:+.2f}",
        ]
        for h in headlines
    ]


def _social_evidence(social: SocialSentimentData | None) -> tuple[list[str], list[dict[str, str]]]:
    """Evidence rows for the social reading: which source answered, how many posts, which scorer, what left the machine.

    Shown for transparency only. It never changes the agent's score or confidence.
    """
    if social is None or not social.enabled:
        return [], []
    factors = [f"Social (informational, not in the score): {note}" for note in social.notes]
    rows: list[dict[str, str]] = []
    for r in social.readings:
        tone = f"{r.score:+.2f}" if r.counts and r.score is not None else "below the minimum post count"
        rows.append({"metric": f"Social: {r.source}", "value": f"{r.posts} posts, tone {tone}"})
        if r.scorers:
            rows.append({"metric": f"Social scorer ({r.source})", "value": ", ".join(f"{k} {n}" for k, n in sorted(r.scorers.items()))})
        rows.append({"metric": f"Left this machine ({r.source})", "value": r.sent_off_machine})
    if not social.readings:
        rows.append({"metric": "Social", "value": "no source answered"})
    return factors, rows


@register_agent
class SentimentAgent(BaseAgent):
    """Recency-weighted news headline tone from a finance keyword lexicon."""

    key = "sentiment"
    name = "News Sentiment Agent"
    description = (
        "Recent news tone: Yahoo Finance headlines, plus Alpha Vantage (sentiment and relevance attached per "
        "ticker) and Finnhub company news when ALPHAVANTAGE_API_KEY or FINNHUB_API_KEY is set. Provider "
        "scores are used when present, else a finance lexicon with phrases and negation. Recency-weighted "
        "(3-day half-life, 14-day cutoff), duplicates dropped, small samples shrunk toward neutral. One "
        "snapshot a day is stored per ticker to build a history. Entry timing; weight 0 until validated."
    )
    category = "Sentiment"
    score_scale = 40.0
    default_weight = 0.0
    point_in_time = False
    role = TIMING

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        try:
            data = fetch_news_sentiment(ticker)
        except Exception as exc:  # noqa: BLE001 - the provider should not raise, but never crash the run
            logger.warning("News sentiment provider raised for %s: %s", ticker, exc)
            return self.unavailable("news provider error", error=str(exc))
        try:
            social = fetch_social_sentiment(ticker)
        except Exception as exc:  # noqa: BLE001 - opt-in and informational: never break the news reading
            logger.warning("Social sentiment raised for %s: %s", ticker, exc)
            social = None
        try:
            record_sentiment_snapshot(data, social=social.snapshot_fields() if social else None)
        except OSError as exc:
            logger.warning("Could not store the sentiment snapshot for %s: %s", ticker, exc)
        return self.evaluate_with_data(data, social)

    def evaluate_with_data(self, data: NewsSentimentData, social: SocialSentimentData | None = None) -> AgentAssessment:
        """Score an already fetched :class:`NewsSentimentData`. ``social`` adds evidence rows only (weight 0, no vote)."""
        if not data.available:
            return self.unavailable(data.error or "news unavailable", error=data.error)
        headlines = data.headlines
        if not headlines or data.weighted_score is None:
            return self.unavailable(f"No dated news headlines in the last {MAX_AGE_DAYS:.0f} days")

        n = len(headlines)
        mean = data.weighted_score
        shrink = n / (n + SHRINKAGE_K)
        raw = mean * 100.0
        score = raw * shrink
        confidence = min(MAX_CONFIDENCE, BASE_CONFIDENCE + (MAX_CONFIDENCE - BASE_CONFIDENCE) * n / (n + CONFIDENCE_HALF_N))

        positives = [h for h in headlines if h.score >= NEUTRAL_BAND]
        negatives = [h for h in headlines if h.score <= -NEUTRAL_BAND]
        neutrals = n - len(positives) - len(negatives)
        total_weight = sum(h.weight for h in headlines)
        pos_points = sum(h.weight * h.score for h in positives) / total_weight * 100.0
        neg_points = sum(h.weight * h.score for h in negatives) / total_weight * 100.0
        # Headlines inside the neutral band still carry a tiny score; fold it into the neutral row
        # so the breakdown adds up to the final score.
        neutral_points = raw - pos_points - neg_points
        score_breakdown = [
            {"factor": f"Positive headlines ({len(positives)})", "points": round(pos_points, 1)},
            {"factor": f"Negative headlines ({len(negatives)})", "points": round(neg_points, 1)},
            {"factor": f"Neutral headlines ({neutrals}, dilute the average)", "points": round(neutral_points, 1)},
            {"factor": f"Small-sample shrinkage ({n} headlines, x{shrink:.2f})", "points": round(raw * (shrink - 1.0), 1)},
        ]

        tone = _tone_label(mean)
        sources = ", ".join(data.sources) or "news sources"
        key_factors = [
            f"{n} headlines from {sources} in the last {MAX_AGE_DAYS:.0f} days "
            f"({len(positives)} positive, {len(negatives)} negative, {neutrals} neutral)",
            f"Recency-weighted tone {mean:+.2f} on a -1 to +1 scale ({tone}), 3-day half-life",
        ]
        pos_terms = _top_terms(headlines, positive=True)
        neg_terms = _top_terms(headlines, positive=False)
        if pos_terms:
            key_factors.append(f"Positive terms: {', '.join(pos_terms)}")
        if neg_terms:
            key_factors.append(f"Negative terms: {', '.join(neg_terms)}")
        skipped = data.dropped_duplicates + data.dropped_stale + data.dropped_undated
        if skipped:
            key_factors.append(
                f"Skipped {data.dropped_duplicates} near duplicates, {data.dropped_stale} older than "
                f"{MAX_AGE_DAYS:.0f} days and {data.dropped_undated} undated items"
            )
        if shrink < 0.75:
            key_factors.append(f"Few headlines: score shrunk toward neutral (x{shrink:.2f})")
        if data.provider_scored:
            key_factors.append(f"Provider sentiment scores used for {data.provider_scored} of {n} headlines (relevance-weighted)")
        for name in data.source_errors:
            key_factors.append(f"{name} news could not be loaded")
        history = load_sentiment_history(data.ticker)
        if len(history) >= 5:
            recent = sum(h["score"] for h in history[-5:]) / 5
            key_factors.append(f"Stored history: {len(history)} daily snapshots, 5-day average tone {recent:+.2f}")
        social_factors, social_rows = _social_evidence(social)
        key_factors.extend(social_factors)
        key_factors.append("Informational: weight 0 in entry timing unless you raise it")

        summary = (
            f"News tone is {tone} across {n} recent headlines (recency-weighted {mean:+.2f}); "
            f"score {score:+.0f} after small-sample shrinkage."
        )

        metrics_table = [
            {"metric": "Headlines scored", "value": str(n)},
            {"metric": "Positive / negative / neutral", "value": f"{len(positives)} / {len(negatives)} / {neutrals}"},
            {"metric": "Recency-weighted tone (-1 to +1)", "value": f"{mean:+.3f}"},
            {"metric": "Small-sample shrinkage", "value": f"x{shrink:.2f}"},
            {"metric": "Newest headline", "value": data.as_of or "n/a"},
            {"metric": "Items fetched", "value": str(data.fetched_count)},
            {"metric": "Near duplicates skipped", "value": str(data.dropped_duplicates)},
            {"metric": f"Older than {MAX_AGE_DAYS:.0f} days skipped", "value": str(data.dropped_stale)},
            {"metric": "Undated items skipped", "value": str(data.dropped_undated)},
            {"metric": "Headlines with a provider sentiment score", "value": str(data.provider_scored)},
            {"metric": "Sources", "value": sources},
            {"metric": "Stored daily snapshots", "value": str(len(history))},
            *social_rows,
        ]

        raw_data = {
            "metrics_table": metrics_table,
            "score_breakdown": score_breakdown,
            "table": {
                "title": "Recent headlines",
                "columns": ["Date", "Source", "Headline", "Score"],
                "rows": _headline_rows(headlines),
            },
            "headlines": [
                {
                    "title": h.title,
                    "publisher": h.publisher,
                    "url": h.url,
                    "published": h.published_iso,
                    "score": h.score,
                    "weight": h.weight,
                    "terms": list(h.terms),
                    "score_source": h.score_source,
                    "relevance": h.relevance,
                }
                for h in headlines
            ],
            "weighted_score": round(mean, 4),
            "provider_scored": data.provider_scored,
            "sentiment_history": history,
            "links": [
                {"label": "Yahoo Finance news", "url": f"https://finance.yahoo.com/quote/{quote(data.ticker, safe='')}/news/"}
            ],
            "data_source_label": f"{sources} news headlines",
        }

        return self.assessment(
            score=score,
            confidence=confidence,
            summary=summary,
            key_factors=key_factors,
            raw_data=raw_data,
            data_quality=PARTIAL if data.source_errors else LIVE,
            data_as_of=data.as_of,
        )
