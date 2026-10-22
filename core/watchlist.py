"""Watchlist scanner: score many tickers, rank them, flag decision changes."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents.master_agent import MasterDecisionAgent, TradeRecommendation
from data.fetcher import fetch_live_ohlcv
from data.history import get_previous_many, record_analyses, record_analysis

logger = logging.getLogger(__name__)

DECISION_RANK = {"STRONG_BUY": 2, "BUY": 1, "HOLD": 0, "NOT_RATED": 0, "SELL": -1, "STRONG_SELL": -2}


def analyze_ticker(
    ticker: str,
    master: MasterDecisionAgent | None = None,
    record: bool = True,
) -> tuple[TradeRecommendation, Any]:
    """Fetch fresh data for one ticker, analyze it and (optionally) record it."""
    symbol = ticker.upper()
    data = fetch_live_ohlcv([symbol])
    if symbol not in data:
        raise LookupError(f"No market data available for {symbol}")
    ohlcv, freshness = data[symbol]
    master = master or MasterDecisionAgent()
    rec = master.analyze(symbol, ohlcv, freshness.to_dict())
    if record:
        record_analysis(rec)
    return rec, ohlcv


def top_points(rec: TradeRecommendation, limit: int = 2) -> list[dict[str, Any]]:
    """The largest scored rules behind the owner score (rule, points, agent), positive or negative. Display only."""
    owner_keys = {c["agent_key"] for c in rec.contributions if c.get("layer") == "owner" and c.get("included")}
    rules: list[dict[str, Any]] = []
    for a in rec.agent_assessments:
        if a.agent_key not in owner_keys:
            continue
        for b in (a.raw_data or {}).get("score_breakdown") or []:
            points = b.get("points")
            if isinstance(points, (int, float)) and points != 0:
                rules.append({"factor": str(b.get("factor", "")), "points": points, "agent": a.agent_name.removesuffix(" Agent")})
    rules.sort(key=lambda r: abs(r["points"]), reverse=True)
    return rules[:limit]


def _row(rec: TradeRecommendation, ohlcv, previous: dict[str, Any] | None) -> dict[str, Any]:
    close = ohlcv["close"]
    change_1d = float(close.iloc[-1] / close.iloc[-2] - 1.0) if len(close) > 1 else 0.0
    prev_decision = previous["decision"] if previous else None
    prev_score = previous["composite_score"] if previous else None
    return {
        "ticker": rec.ticker,
        "price": rec.suggested_entry,
        "change_1d": round(change_1d, 4),
        "decision": rec.decision,
        "composite_score": rec.composite_score,
        "confidence": rec.overall_confidence,
        "owner_applicable": rec.owner.get("applicable", True),
        "top_points": top_points(rec),
        "timing_score": rec.timing.get("score"),
        "timing_label": rec.timing.get("label"),
        "trend": rec.timing.get("trend"),
        "setup": rec.timing.get("setup"),
        "regime": rec.market_regime,
        "risk_rating": rec.risk_rating,
        "conflict": rec.conflict,
        "coverage": rec.coverage,
        "warnings": rec.warnings,
        "freshness": rec.freshness,
        "agents": {
            c["agent_key"]: {
                "score": c["normalized_score"],
                "quality": c["data_quality"],
                "included": c["included"],
                "layer": c.get("layer", "timing"),
            }
            for c in rec.contributions
        },
        "previous_decision": prev_decision,
        "previous_score": prev_score,
        "score_change": round(rec.composite_score - prev_score, 1) if prev_score is not None else None,
        "decision_changed": prev_decision is not None and prev_decision != rec.decision,
        "upgraded": prev_decision is not None and DECISION_RANK.get(rec.decision, 0) > DECISION_RANK.get(prev_decision, 0),
    }


def scan_watchlist(tickers: list[str], max_parallel: int = 4, record: bool = True) -> dict[str, Any]:
    """
    Analyze every ticker (a few in parallel; each analysis already runs its
    agents concurrently) and return rows ranked by composite score.
    """
    symbols = [t.upper() for t in tickers if t.strip()]
    # One batched price refresh up front; per-ticker analyses then hit warm caches.
    fetched = fetch_live_ohlcv(symbols)
    master = MasterDecisionAgent()

    def work(symbol: str) -> dict[str, Any] | tuple[TradeRecommendation, Any]:
        if symbol not in fetched:
            return {"ticker": symbol, "error": "No market data"}
        ohlcv, freshness = fetched[symbol]
        try:
            return master.analyze(symbol, ohlcv, freshness.to_dict()), ohlcv
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not sink the scan
            logger.exception("Watchlist analysis failed for %s", symbol)
            return {"ticker": symbol, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="watch") as pool:
        results = list(pool.map(work, symbols))

    # The history reads and writes once for the whole scan (it was two connections per ticker); each ticker's
    # previous analysis is still read before its new one is stored.
    done = [(symbol, r[0]) for symbol, r in zip(symbols, results) if isinstance(r, tuple)]
    previous = iter(get_previous_many([(symbol, rec.generated_at) for symbol, rec in done]))
    if record:
        record_analyses([rec for _, rec in done])
    rows = [_row(r[0], r[1], next(previous)) if isinstance(r, tuple) else r for r in results]

    # Rated companies first (by owner score), then not-rated tickers (funds, indexes) by timing score.
    ok = sorted(
        (r for r in rows if "error" not in r),
        key=lambda r: (r["owner_applicable"], r["composite_score"] if r["owner_applicable"] else (r["timing_score"] or 0.0)),
        reverse=True,
    )
    failed = [r for r in rows if "error" in r]
    return {"rows": ok, "errors": failed}
