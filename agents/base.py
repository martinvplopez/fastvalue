"""Base agent definition and data structures for the FastValue multi-agent framework.

Adding a new agent
------------------
1. Create ``agents/<something>_agent.py``.
2. Subclass :class:`BaseAgent`, set the class attributes (``key``, ``name``,
   ``description``, ``score_scale``, ``default_weight``, ``point_in_time``).
3. Decorate the class with ``@register_agent`` (from ``agents.registry``).
4. Implement ``evaluate(ticker, ohlcv)`` and return ``self.assessment(...)``,
   or ``self.unavailable(reason)`` when the underlying data could not be loaded.

The agent is discovered automatically and appears in the weights editor.
Agents missing from a regime weight profile use ``default_weight``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

import pandas as pd

# Data quality levels reported by every agent.
LIVE = "LIVE"  # Real data from the provider, fresh enough to act on.
PARTIAL = "PARTIAL"  # Real data, but some inputs were missing or stale.
UNAVAILABLE = "UNAVAILABLE"  # No usable data. The agent is excluded from the composite.

DATA_QUALITY_LEVELS = (LIVE, PARTIAL, UNAVAILABLE)

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

# Decision layers (see agents/scoring.py). OWNER agents judge the business and
# decide BUY / HOLD / SELL on a 12-month horizon; TIMING agents describe the
# trend, positioning and market backdrop for entry timing and sizing, without a vote.
# INFO agents are shown with their evidence but scored in neither layer. The class
# role is the shipped default; users can reassign roles in the Weights tab
# (``ScoringConfig.roles``).
OWNER = "owner"
TIMING = "timing"
INFO = "info"
ROLES = (OWNER, TIMING, INFO)


@dataclass
class AgentAssessment:
    """Standard output structure returned by every agent."""

    agent_name: str
    score: float  # Scale from -100.0 (extreme sell/bearish) to +100.0 (extreme buy/bullish)
    signal: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float  # 0.0 to 100.0 (%)
    summary: str
    key_factors: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)
    agent_key: str = ""
    data_quality: str = LIVE
    data_as_of: str | None = None  # ISO date/time of the newest data point used
    error: str | None = None

    @property
    def is_available(self) -> bool:
        return self.data_quality != UNAVAILABLE


def signal_from_score(score: float, threshold: float = 20.0) -> str:
    """Map a -100..+100 score to BULLISH / BEARISH / NEUTRAL."""
    if score >= threshold:
        return BULLISH
    if score <= -threshold:
        return BEARISH
    return NEUTRAL


class BaseAgent(ABC):
    """Abstract base class for all domain expert agents."""

    # Stable identifier used in weight profiles and the API. Never rename casually.
    key: ClassVar[str] = ""
    # Human-readable name shown in the UI.
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    # Short grouping label for the UI (e.g. "Technical", "Flow", "Fundamental", "Macro").
    category: ClassVar[str] = "Other"
    # tanh normalization scale: the raw score at which the normalized score reaches ~76.
    # Agents whose raw scores cluster in a narrow band use a smaller scale.
    score_scale: ClassVar[float] = 50.0
    # Weight used when the agent is missing from a regime profile.
    default_weight: ClassVar[float] = 0.0
    # True when evaluate() only depends on the OHLCV passed in, so it can be
    # replayed historically without look-ahead bias (used by the score backtester).
    point_in_time: ClassVar[bool] = False
    # True for the agent whose raw_data["regime"] selects the regime weight profile.
    provides_regime: ClassVar[bool] = False
    # Decision layer: OWNER (fundamentals, shareholders; decides) or TIMING (no vote).
    role: ClassVar[str] = TIMING
    # Set for an agent that makes reviews and scans noticeably slower: a phrase that follows the
    # agent's name and gives the remedy. The UI shows it in the Weights tab and when a review or scan runs long.
    slow_note: ClassVar[str] = ""

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name  # type: ignore[misc]

    @abstractmethod
    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        """
        Evaluate the ticker and return an AgentAssessment.

        Args:
            ticker: Symbol (e.g. AAPL, SPY).
            ohlcv: DataFrame of OHLCV daily data, oldest first. The last row is
                the most recent (possibly intraday, still-forming) bar.

        Returns:
            AgentAssessment object.
        """
        ...

    # ------------------------------------------------------------------ helpers

    def assessment(
        self,
        score: float,
        confidence: float,
        summary: str,
        key_factors: list[str] | None = None,
        raw_data: dict[str, Any] | None = None,
        data_quality: str = LIVE,
        data_as_of: str | None = None,
        signal: str | None = None,
    ) -> AgentAssessment:
        """Build an AgentAssessment with clamped values and this agent's identity."""
        score = max(-100.0, min(100.0, float(score)))
        confidence = max(0.0, min(100.0, float(confidence)))
        return AgentAssessment(
            agent_name=self.name,
            agent_key=self.key,
            score=round(score, 1),
            signal=signal or signal_from_score(score),
            confidence=round(confidence, 1),
            summary=summary,
            key_factors=list(key_factors or []),
            raw_data=dict(raw_data or {}),
            data_quality=data_quality,
            data_as_of=data_as_of,
        )

    def unavailable(
        self,
        reason: str,
        raw_data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AgentAssessment:
        """
        Return a neutral, zero-confidence assessment that the master agent excludes.

        Use this instead of estimating or inventing data: a missing signal must
        never be presented as a real one.
        """
        return AgentAssessment(
            agent_name=self.name,
            agent_key=self.key,
            score=0.0,
            signal=NEUTRAL,
            confidence=0.0,
            summary=f"No usable data: {reason}",
            key_factors=[f"Excluded from composite: {reason}"],
            raw_data=dict(raw_data or {}),
            data_quality=UNAVAILABLE,
            error=error,
        )

    @classmethod
    def metadata(cls) -> dict[str, Any]:
        """Describe the agent for the API / weights editor."""
        return {
            "key": cls.key,
            "name": cls.name,
            "description": cls.description,
            "category": cls.category,
            "score_scale": cls.score_scale,
            "default_weight": cls.default_weight,
            "point_in_time": cls.point_in_time,
            "provides_regime": cls.provides_regime,
            "role": cls.role,
            "slow_note": cls.slow_note,
        }


def iso_date(ts: Any) -> str | None:
    """Best-effort conversion of a timestamp-like value to an ISO date string."""
    if ts is None:
        return None
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
