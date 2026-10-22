"""The latest owner score of every company analysed, on today's weights, so companies can be compared.

Each analysis stores its agents' raw scores and confidences (data/history.py). Scores recorded under
older weights are not comparable with today's, so the owner layer is recomposed here with the saved
weights (the same ``aggregate_layers`` the live analysis uses; the owner layer has no regime profile).
Analyses that were not rated (funds, indexes, excluded sectors) are left out. Display only: nothing
here changes a score, a threshold or a stored row.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from agents.base import AgentAssessment
from agents.registry import get_agent_classes
from agents.scoring import NOT_RATED, aggregate_layers
from config.weights import load_scoring_config
from data.history import get_latest_all, load_watchlist
from data.portfolio import load_portfolio

MAX_AGE_DAYS = 90  # older analyses describe figures that have since changed too much to compare


def _assessments(agent_scores: dict[str, Any]) -> list[AgentAssessment]:
    return [
        AgentAssessment(agent_name=key, agent_key=key, score=float(v.get("score") or 0.0), signal="NEUTRAL",
                        confidence=float(v.get("confidence") or 0.0), summary="", data_quality=str(v.get("quality") or "LIVE"))
        for key, v in agent_scores.items() if isinstance(v, dict)
    ]


def score_board(max_age_days: int = MAX_AGE_DAYS) -> dict[str, Any]:
    """Rated companies with their latest owner score recomposed on the saved weights, highest first."""
    since = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat(timespec="seconds")
    cfg = load_scoring_config()
    classes = get_agent_classes()
    roles = {k: c.role for k, c in classes.items()}
    scales = {k: c.score_scale for k, c in classes.items()}
    try:
        held = {str(p.get("ticker") or "").upper() for p in load_portfolio().get("positions") or []}
    except Exception:  # noqa: BLE001 - the flags are a convenience
        held = set()
    watched = set(load_watchlist())
    rows = []
    for rec in get_latest_all(since):
        if rec.get("decision") == NOT_RATED:
            continue
        layered = aggregate_layers(_assessments(rec["agent_scores"]), cfg, None, scales, roles)
        if not layered.owner_applicable:
            continue
        ticker = rec["ticker"]
        rows.append({
            "ticker": ticker,
            "score": round(layered.owner.composite_score, 1),
            "recorded": None if rec.get("owner_score") is None else round(float(rec["owner_score"]), 1),
            "analysed": str(rec["generated_at"])[:10],
            "held": ticker in held,
            "watched": ticker in watched,
        })
    rows.sort(key=lambda r: -r["score"])
    return {"rows": rows, "max_age_days": max_age_days, "thresholds": dict(cfg.owner_thresholds)}
