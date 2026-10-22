"""Composite scoring: turns agent assessments + a ScoringConfig into one decision.

Pure functions with no I/O. The master agent, the ``/api/rescore`` endpoint
(live what-if when weights change), the watchlist scanner and the score
backtester all call the same functions so they can never drift apart.

Two layers (``aggregate_layers``):

- **Owner score** (agents with ``role == OWNER``: fundamentals, shareholders,
  insiders as evidence). Weighted with ``config.owner_weights`` regardless of
  the market regime, and it alone decides BUY / HOLD / SELL on a 12-month
  horizon. When no owner agent has data (an ETF, an index, a company without
  statements), or the company is in a sector outside the framework (utilities
  and REITs, owner's call T-12), the decision is NOT_RATED.
- **Timing panel** (agents with ``role == TIMING``: trend, volume, options,
  macro). Weighted with the regime profile, summarized as FAVORABLE / NEUTRAL /
  UNFAVORABLE, and used for entry timing, staging and sizing. It has no vote:
  the point-in-time value study found that letting trading signals decide cost
  return (see decision_log.md, T-01).
- **Informative** (``role == INFO``): shown with their evidence, scored in
  neither layer.

Roles come from the agent classes (the shipped defaults) and are overridden by
``config.roles``, which the user sets in the Weights tab. A user who moves a
trading agent into the owner layer gets a warning with the study's result.

Pipeline per agent inside one layer (``aggregate``):
1. Skip if disabled, weight 0, or data UNAVAILABLE (reported as excluded).
2. Normalize the raw score with ``tanh(score / scale) * 100`` so agents with
   narrow natural ranges are comparable to agents with wide ones.
3. Effective weight = base weight * (floor + (1 - floor) * confidence).
4. Composite = sum(normalized * effective) / sum(effective).

Then: dispersion across agents lowers confidence, strongly opposed agents
raise a conflict warning, and thin data coverage lowers confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from agents.base import INFO, OWNER, TIMING, UNAVAILABLE, AgentAssessment
from config.weights import ScoringConfig

DECISIONS: tuple[str, ...] = ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")
NOT_RATED = "NOT_RATED"  # no owner agent had data: the framework does not apply
ALL_DECISIONS: tuple[str, ...] = DECISIONS + (NOT_RATED,)
TIMING_LABELS: tuple[str, ...] = ("FAVORABLE", "NEUTRAL", "UNFAVORABLE")

# Agents whose normalized scores are at least this far apart in opposite
# directions (each with decent confidence) trigger a conflict warning.
CONFLICT_LEVEL = 40.0
CONFLICT_MIN_CONFIDENCE = 55.0
INFO_REASON = "Informative: shown as evidence, not scored"


@dataclass
class AgentContribution:
    """How one agent moved the composite score."""

    agent_key: str
    agent_name: str
    raw_score: float
    normalized_score: float
    confidence: float
    data_quality: str
    base_weight: float
    effective_weight: float
    weight_share: float  # effective_weight / total effective weight (0..1)
    contribution: float  # normalized_score * weight_share (points of composite)
    included: bool
    exclusion_reason: str | None = None
    layer: str = TIMING  # OWNER, TIMING or INFO


@dataclass
class CompositeResult:
    composite_score: float
    confidence: float
    decision: str
    regime: str
    contributions: list[AgentContribution]
    weights_used: dict[str, float]
    dispersion: float
    conflict: bool
    coverage: float  # share of configured weight that had data (0..1)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayeredResult:
    """Owner score (decides) plus timing panel (informs), see the module docstring."""

    decision: str  # owner decision, or NOT_RATED
    owner: CompositeResult
    timing: CompositeResult
    timing_label: str
    regime: str
    owner_applicable: bool
    warnings: list[str] = field(default_factory=list)
    informative: list[AgentContribution] = field(default_factory=list)  # role INFO, not scored

    @property
    def contributions(self) -> list[AgentContribution]:
        return list(self.owner.contributions) + list(self.timing.contributions) + list(self.informative)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "owner": self.owner.to_dict(),
            "timing": self.timing.to_dict(),
            "timing_label": self.timing_label,
            "regime": self.regime,
            "owner_applicable": self.owner_applicable,
            "warnings": list(self.warnings),
            "informative": [asdict(c) for c in self.informative],
            # Compatibility with callers that read one composite.
            "composite_score": self.owner.composite_score,
            "confidence": self.owner.confidence,
        }


def normalize_score(raw_score: float, scale: float) -> float:
    """Map a raw -100..+100 score through tanh so every agent saturates similarly."""
    return float(np.tanh(raw_score / max(scale, 1e-6)) * 100.0)


def confidence_weight(base_weight: float, confidence: float, floor: float = 0.4) -> float:
    """
    Modulate a base weight by agent confidence.

    ``effective = base * (floor + (1 - floor) * confidence / 100)``; with the
    default floor of 0.4, a 0% confidence agent keeps 40% of its weight.
    """
    c = max(0.0, min(float(confidence), 100.0)) / 100.0
    return base_weight * (floor + (1.0 - floor) * c)


def decide(composite: float, thresholds: dict[str, float]) -> str:
    if composite >= thresholds["strong_buy"]:
        return "STRONG_BUY"
    if composite >= thresholds["buy"]:
        return "BUY"
    if composite <= thresholds["strong_sell"]:
        return "STRONG_SELL"
    if composite <= thresholds["sell"]:
        return "SELL"
    return "HOLD"


def timing_label(composite: float, thresholds: dict[str, float]) -> str:
    """FAVORABLE / NEUTRAL / UNFAVORABLE from the timing composite (buy and sell thresholds)."""
    if composite >= thresholds["buy"]:
        return "FAVORABLE"
    if composite <= thresholds["sell"]:
        return "UNFAVORABLE"
    return "NEUTRAL"


def aggregate(
    assessments: list[AgentAssessment],
    config: ScoringConfig,
    regime: str | None,
    scales: dict[str, float] | None = None,
    weight_overrides: dict[str, float] | None = None,
    agent_keys: Iterable[str] | None = None,
    thresholds: dict[str, float] | None = None,
    layer: str = TIMING,
) -> CompositeResult:
    """
    Combine agent assessments into a composite score and decision.

    Args:
        assessments: Output of each agent that ran.
        config: Scoring configuration (weights, thresholds, enabled flags).
        regime: Market regime used to pick the weight profile (None -> fallback).
        scales: agent_key -> tanh scale (agent class defaults); config.score_scales wins.
        weight_overrides: Explicit agent_key -> weight map that bypasses the regime profile.
        agent_keys: Only these agents take part (others are left out entirely).
        thresholds: Decision thresholds (default: ``config.thresholds``).
        layer: Label stored on each contribution (OWNER or TIMING).
    """
    effective_regime = regime if regime in config.regime_weights else config.fallback_regime
    base_weights = dict(weight_overrides) if weight_overrides else config.weights_for(effective_regime)
    scales = {**(scales or {}), **config.score_scales}
    thresholds = thresholds or config.thresholds
    if agent_keys is not None:
        allowed = set(agent_keys)
        assessments = [a for a in assessments if (a.agent_key or a.agent_name) in allowed]

    contributions: list[AgentContribution] = []
    included: list[tuple[AgentContribution, float]] = []
    configured_weight = 0.0
    available_weight = 0.0

    for a in assessments:
        key = a.agent_key or a.agent_name
        base_w = float(base_weights.get(key, 0.0))
        enabled = config.is_enabled(key)
        normalized = normalize_score(a.score, scales.get(key, 50.0))

        reason = None
        if not enabled:
            reason = "Disabled in weights config"
        elif base_w <= 0:
            reason = "Weight is 0 (informational only)"
        elif a.data_quality == UNAVAILABLE:
            reason = "No usable data"

        if enabled and base_w > 0:
            configured_weight += base_w
            if a.data_quality != UNAVAILABLE:
                available_weight += base_w

        effective_w = confidence_weight(base_w, a.confidence, config.confidence_floor) if reason is None else 0.0
        contrib = AgentContribution(
            agent_key=key,
            agent_name=a.agent_name,
            raw_score=round(float(a.score), 1),
            normalized_score=round(normalized, 1),
            confidence=round(float(a.confidence), 1),
            data_quality=a.data_quality,
            base_weight=round(base_w, 4),
            effective_weight=round(effective_w, 4),
            weight_share=0.0,
            contribution=0.0,
            included=reason is None,
            exclusion_reason=reason,
            layer=layer,
        )
        contributions.append(contrib)
        if reason is None:
            included.append((contrib, normalized))

    total_w = sum(c.effective_weight for c, _ in included)
    warnings: list[str] = []

    if total_w <= 0:
        return CompositeResult(
            composite_score=0.0,
            confidence=0.0,
            decision="HOLD",
            regime=effective_regime,
            contributions=contributions,
            weights_used=base_weights,
            dispersion=0.0,
            conflict=False,
            coverage=0.0,
            warnings=["No agent produced usable data; defaulting to HOLD."],
        )

    composite = 0.0
    confidence = 0.0
    for c, normalized in included:
        share = c.effective_weight / total_w
        c.weight_share = round(share, 4)
        c.contribution = round(normalized * share, 2)
        composite += normalized * share
        confidence += c.confidence * share

    normalized_scores = np.array([n for _, n in included])
    shares = np.array([c.weight_share for c, _ in included])
    dispersion = float(np.sqrt(np.sum(shares * (normalized_scores - composite) ** 2)))

    strong_bulls = [c for c, n in included if n >= CONFLICT_LEVEL and c.confidence >= CONFLICT_MIN_CONFIDENCE]
    strong_bears = [c for c, n in included if n <= -CONFLICT_LEVEL and c.confidence >= CONFLICT_MIN_CONFIDENCE]
    conflict = bool(strong_bulls and strong_bears)
    if conflict:
        up = ", ".join(c.agent_name.replace(" Agent", "") for c in strong_bulls)
        down = ", ".join(c.agent_name.replace(" Agent", "") for c in strong_bears)
        reads_up = "reads" if len(strong_bulls) == 1 else "read"
        reads_down = "reads" if len(strong_bears) == 1 else "read"
        if layer == OWNER:
            warnings.append(
                f"Owner agents disagree: {up} {reads_up} up, {down} {reads_down} down. "
                "Confidence is reduced; open both agent cards below before acting."
            )
        else:
            warnings.append(
                f"Timing agents disagree: {up} {reads_up} up, {down} {reads_down} down. "
                "Treat the timing read as weak and stage any entry."
            )

    # Dispersion penalty: up to 25 confidence points when agents are far apart.
    confidence -= min(25.0, dispersion * 0.3)

    coverage = available_weight / configured_weight if configured_weight > 0 else 0.0
    if coverage < 0.7:
        missing = [c.agent_name for c in contributions if c.exclusion_reason == "No usable data"]
        warnings.append(
            f"Only {coverage:.0%} of the configured weight had data"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + "; confidence reduced."
        )
        confidence *= 0.5 + 0.5 * coverage

    if regime not in config.regime_weights and not weight_overrides:
        warnings.append(f"Market regime unknown; used the {effective_regime} weight profile.")

    composite = float(np.clip(composite, -100.0, 100.0))
    return CompositeResult(
        composite_score=round(composite, 1),
        confidence=round(max(0.0, min(100.0, confidence)), 1),
        decision=decide(composite, thresholds),
        regime=effective_regime,
        contributions=contributions,
        weights_used=base_weights,
        dispersion=round(dispersion, 1),
        conflict=conflict,
        coverage=round(coverage, 3),
        warnings=warnings,
    )


def margin_of_safety_cap(assessments: list[AgentAssessment], config: ScoringConfig) -> str | None:
    """Reason to cap an owner BUY at HOLD on price alone, or None.

    The owner's anchor is 16-17x EV/FCF. Above ``config.owner_max_ev_to_fcf``
    (default 35x, where the fundamentals agent already gives its harshest
    valuation penalty), or with negative FCF, a business can be well run and
    still be too expensive to buy: the shareholder side must not turn that into
    a BUY (decision_log.md T-08).
    """
    cap = config.owner_max_ev_to_fcf
    if cap is None:
        return None
    for a in assessments:
        if (a.agent_key or a.agent_name) != "fundamentals" or a.data_quality == UNAVAILABLE:
            continue
        if a.raw_data.get("is_financial"):
            return None
        ev_to_fcf = a.raw_data.get("ev_to_fcf")
        fcf = a.raw_data.get("fcf")
        if isinstance(ev_to_fcf, (int, float)) and ev_to_fcf >= cap:
            return (f"Margin of safety: EV/FCF {ev_to_fcf:.0f}x is at or above the {cap:g}x cap: no new money "
                    "(owner anchor 16-17x). Well run, but too expensive to buy here. The score is shown as computed.")
        if ev_to_fcf is None and isinstance(fcf, (int, float)) and fcf <= 0:
            return ("Margin of safety: free cash flow is negative, so there is no EV/FCF to anchor on: "
                    "no new money. The score is shown as computed.")
    return None


def framework_exclusion(assessments: list[AgentAssessment]) -> str | None:
    """Reason the value framework does not rate this company at all, or None.

    The fundamentals and shareholder agents record the company's sector;
    utilities and REITs are outside the framework (decision_log.md T-12): the
    debt and ROIC rules exclude them by construction and EV/FCF on REIT
    depreciation is not meaningful. Their evidence is still shown, but the
    decision is NOT_RATED. This is about the company, so it applies whatever
    roles the user gave those agents.
    """
    for a in assessments:
        if a.raw_data.get("excluded_sector"):
            sector = a.raw_data.get("sector") or "This sector"
            return (
                f"Not rated: {sector} is outside the value framework (owner's call, decision_log.md T-12). "
                "The debt and ROIC rules exclude utilities and REITs by construction, and EV/FCF on REIT "
                "depreciation is not meaningful. The owner agents' evidence and the entry timing signals are shown for information only."
            )
    return None


def split_by_role(keys: Iterable[str], roles: dict[str, str] | None) -> tuple[set[str], set[str], set[str]]:
    """(owner keys, timing keys, informative keys). Agents with an unknown role are timing agents."""
    roles = roles or {}
    keys = set(keys)
    owner = {k for k in keys if roles.get(k) == OWNER}
    info = {k for k in keys if roles.get(k) == INFO}
    return owner, keys - owner - info, info


# Study result quoted when a user lets a trading agent decide (research_results.md, all-agents study, third pass).
TRADING_VOTE_EVIDENCE = (
    "In the 2011-2026 S&P 500 test, the design with trading agents voting made 12.2% a year "
    "against 15.6% for the owner score alone (decision_log.md D-11). Check your setup in the backtests."
)


def aggregate_layers(
    assessments: list[AgentAssessment],
    config: ScoringConfig,
    regime: str | None,
    scales: dict[str, float] | None = None,
    roles: dict[str, str] | None = None,
    weight_overrides: dict[str, float] | None = None,
) -> LayeredResult:
    """
    Owner score decides, timing panel informs.

    Args:
        assessments: Output of every agent that ran (both layers).
        config: Scoring configuration; ``owner_weights`` and ``owner_thresholds``
            drive the owner layer, ``regime_weights`` and ``thresholds`` the timing layer.
        regime: Market regime from the macro agent (timing profile only).
        scales: agent_key -> tanh scale.
        roles: agent_key -> OWNER / TIMING (from the registry or the agent
            instances). ``config.roles`` (the user's roles) overrides it.
        weight_overrides: Explicit timing weights that bypass the regime profile.
    """
    keys = [a.agent_key or a.agent_name for a in assessments]
    base_roles = dict(roles or {})
    owner_keys, timing_keys, info_keys = split_by_role(keys, {**base_roles, **config.roles})

    owner = aggregate(
        assessments, config, None, scales,
        weight_overrides=config.owner_weights, agent_keys=owner_keys,
        thresholds=config.owner_thresholds, layer=OWNER,
    )
    timing = aggregate(
        assessments, config, regime, scales,
        weight_overrides=weight_overrides, agent_keys=timing_keys, layer=TIMING,
    )

    all_scales = {**(scales or {}), **config.score_scales}
    informative = [
        AgentContribution(
            agent_key=a.agent_key or a.agent_name,
            agent_name=a.agent_name,
            raw_score=round(float(a.score), 1),
            normalized_score=round(normalize_score(a.score, all_scales.get(a.agent_key or a.agent_name, 50.0)), 1),
            confidence=round(float(a.confidence), 1),
            data_quality=a.data_quality,
            base_weight=0.0,
            effective_weight=0.0,
            weight_share=0.0,
            contribution=0.0,
            included=False,
            exclusion_reason=INFO_REASON,
            layer=INFO,
        )
        for a in assessments if (a.agent_key or a.agent_name) in info_keys
    ]

    exclusion = framework_exclusion(assessments)
    applicable = exclusion is None and owner.coverage > 0 and any(c.included for c in owner.contributions)
    decision = owner.decision if applicable else NOT_RATED
    label = timing_label(timing.composite_score, config.thresholds) if any(c.included for c in timing.contributions) else "NEUTRAL"

    warnings: list[str] = []
    trading_deciders = [c.agent_name.removesuffix(" Agent") for c in owner.contributions
                        if c.included and base_roles.get(c.agent_key, TIMING) != OWNER]
    if trading_deciders:
        warnings.append(f"Your weights let {', '.join(trading_deciders)} decide. {TRADING_VOTE_EVIDENCE}")
    cap_note = margin_of_safety_cap(assessments, config) if applicable and decision in ("BUY", "STRONG_BUY") else None
    if cap_note:
        decision = "HOLD"
        warnings.append(cap_note)
    if exclusion:
        warnings.append(exclusion)
    elif not applicable:
        warnings.append(
            "Not rated: no owner-layer data (fundamentals, shareholders). The value framework does not apply "
            "to funds and indexes, or the statements could not be loaded. The entry timing signals are shown for information only."
        )
    else:
        warnings += [w for w in owner.warnings if not w.startswith("No agent produced")]
    warnings += [w for w in timing.warnings if not w.startswith("No agent produced")]

    return LayeredResult(
        decision=decision,
        owner=owner,
        timing=timing,
        timing_label=label,
        regime=timing.regime,
        owner_applicable=applicable,
        warnings=warnings,
        informative=informative,
    )
