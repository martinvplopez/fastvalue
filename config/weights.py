"""Load, validate and persist the multi-agent scoring configuration.

``config/agent_weights.json`` holds the shipped defaults (tracked in git).
User edits from the web UI are written to ``data/store/agent_weights.json``
(git-ignored, with the rest of the local state) and layered on top, so
"reset to defaults" is just deleting that file.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config.settings as settings

logger = logging.getLogger(__name__)

REGIMES: tuple[str, ...] = ("BULL", "CORRECTION", "BEAR", "CRISIS")
THRESHOLD_KEYS: tuple[str, ...] = ("strong_buy", "buy", "sell", "strong_sell")
DEFAULT_THRESHOLDS: tuple[float, ...] = (45.0, 15.0, -15.0, -45.0)
OWNER_ROLE = "owner"
# Layers an agent can belong to: decides (owner), timing (no vote), informative (not scored).
ROLE_VALUES: tuple[str, ...] = ("owner", "timing", "info")

_LOCK = threading.Lock()


def _defaults_path() -> Path:
    return settings.CONFIG_DIR / "agent_weights.json"


def _local_path() -> Path:
    return settings.STORE_DIR / "agent_weights.json"


@dataclass
class ScoringConfig:
    """Everything the aggregation step needs, independent of which agents ran.

    Two layers: ``owner_weights`` and ``owner_thresholds`` drive the owner score
    (fundamentals and shareholder agents; decides BUY/HOLD/SELL and does not
    depend on the market regime). ``regime_weights`` and ``thresholds`` drive
    the timing panel (trend, volume, options, macro), which has no vote.
    """

    regime_weights: dict[str, dict[str, float]]
    thresholds: dict[str, float]
    enabled: dict[str, bool] = field(default_factory=dict)
    score_scales: dict[str, float] = field(default_factory=dict)
    confidence_floor: float = 0.4
    fallback_regime: str = "CORRECTION"
    owner_weights: dict[str, float] = field(default_factory=dict)
    owner_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(zip(THRESHOLD_KEYS, DEFAULT_THRESHOLDS)))
    # Margin of safety: no owner BUY when EV/FCF is at or above this multiple (None disables the cap).
    owner_max_ev_to_fcf: float | None = 35.0
    # agent_key -> "owner" / "timing" / "info": the user's roles, which override the
    # agent classes' roles. Empty means every agent keeps its class role.
    roles: dict[str, str] = field(default_factory=dict)
    # Display order only (no effect on scores): agents inside each layer, and the layers.
    agent_order: list[str] = field(default_factory=list)
    layer_order: list[str] = field(default_factory=lambda: list(ROLE_VALUES))

    def weights_for(self, regime: str | None) -> dict[str, float]:
        profile = self.regime_weights.get(regime or "") or self.regime_weights.get(self.fallback_regime, {})
        return dict(profile)

    def is_enabled(self, agent_key: str) -> bool:
        return bool(self.enabled.get(agent_key, True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_weights": copy.deepcopy(self.regime_weights),
            "thresholds": dict(self.thresholds),
            "owner_weights": dict(self.owner_weights),
            "owner_thresholds": dict(self.owner_thresholds),
            "owner_max_ev_to_fcf": self.owner_max_ev_to_fcf,
            "roles": dict(self.roles),
            "agent_order": list(self.agent_order),
            "layer_order": list(self.layer_order),
            "enabled": dict(self.enabled),
            "score_scales": dict(self.score_scales),
            "confidence_floor": self.confidence_floor,
            "fallback_regime": self.fallback_regime,
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable config %s: %s", path, exc)
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _thresholds(raw: Any, label: str) -> dict[str, float]:
    raw = raw or {}
    out = {k: float(raw.get(k, d)) for k, d in zip(THRESHOLD_KEYS, DEFAULT_THRESHOLDS)}
    if not (out["strong_buy"] >= out["buy"] > out["sell"] >= out["strong_sell"]):
        raise ValueError(f"{label} must run from high to low: strong entry, entry, exit, strong exit.")
    if not all(-100.0 <= v <= 100.0 for v in out.values()):
        raise ValueError(f"{label} must be between -100 and +100.")
    return out


def validate_config(
    raw: dict[str, Any],
    agent_defaults: dict[str, float] | None = None,
    agent_roles: dict[str, str] | None = None,
) -> ScoringConfig:
    """
    Validate a raw config dict and fill gaps.

    Args:
        raw: Parsed JSON-like dict (defaults merged with local overrides).
        agent_defaults: agent_key -> default_weight for registered agents. A
            timing agent missing from a regime profile gets its default weight
            and an owner agent missing from ``owner_weights`` gets its default
            weight, so a new agent file works without touching the config.
        agent_roles: agent_key -> the agent class's role, "owner" or "timing"
            (unknown agents are timing agents). ``raw["roles"]`` overrides it.
            An agent moved out of its class role starts at weight 0 in its new
            layer unless the config gives it a weight, so a role change never
            silently changes a score.

    Raises:
        ValueError: On malformed weights, thresholds, roles or order.
    """
    agent_defaults = agent_defaults or {}
    agent_roles = agent_roles or {}
    raw_roles = {str(k): str(v) for k, v in (raw.get("roles") or {}).items()}
    bad = sorted({v for v in raw_roles.values() if v not in ROLE_VALUES})
    if bad:
        raise ValueError(f"Unknown role {bad[0]!r}: use one of {', '.join(ROLE_VALUES)}")
    roles = {**agent_roles, **raw_roles}
    owner_keys = {k for k, r in roles.items() if r == OWNER_ROLE}
    moved = {k for k, r in roles.items() if k in agent_roles and agent_roles[k] != r}
    if agent_roles and not any(roles.get(k) == OWNER_ROLE for k in agent_roles):
        raise ValueError("At least one agent must decide, or every company comes back not rated.")

    thresholds = _thresholds(raw.get("thresholds"), "Entry timing levels")
    owner_thresholds = _thresholds(raw.get("owner_thresholds") or raw.get("thresholds"), "Owner score levels")

    regime_weights: dict[str, dict[str, float]] = {}
    raw_profiles = raw.get("regime_weights", {})
    for regime in REGIMES:
        profile = {str(k): float(v) for k, v in (raw_profiles.get(regime) or {}).items()}
        for key, default in agent_defaults.items():
            # Owner agents do not vote in the timing panel: 0 unless the profile says otherwise.
            profile.setdefault(key, 0.0 if key in owner_keys or key in moved else float(default))
        if any(v < 0 for v in profile.values()):
            raise ValueError(f"Timing weights in the {regime.title()} profile cannot be negative.")
        regime_weights[regime] = profile

    owner_weights = {str(k): float(v) for k, v in (raw.get("owner_weights") or {}).items()}
    for key in owner_keys:
        owner_weights.setdefault(key, 0.0 if key in moved else float(agent_defaults.get(key, 0.0)))
    if any(v < 0 for v in owner_weights.values()):
        raise ValueError("Owner score weights cannot be negative.")

    cap_raw = raw.get("owner_max_ev_to_fcf", 35.0)
    owner_cap: float | None
    if cap_raw in (None, "", 0, "0"):
        owner_cap = None
    else:
        try:
            owner_cap = float(cap_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("The safety cap is a number above 0, or blank for no cap.") from exc
        if owner_cap <= 0:
            raise ValueError("The safety cap is a number above 0, or blank for no cap.")

    enabled = {str(k): bool(v) for k, v in (raw.get("enabled") or {}).items()}
    for key in agent_defaults:
        enabled.setdefault(key, True)

    score_scales = {str(k): float(v) for k, v in (raw.get("score_scales") or {}).items()}
    if any(v <= 0 for v in score_scales.values()):
        raise ValueError("Score scales must be positive")

    floor = float(raw.get("confidence_floor", 0.4))
    if not 0.0 <= floor <= 1.0:
        raise ValueError("The confidence floor must be between 0% and 100%.")

    fallback = str(raw.get("fallback_regime", "CORRECTION"))
    if fallback not in REGIMES:
        raise ValueError(f"fallback_regime must be one of {REGIMES}")

    order = [str(k) for k in (raw.get("agent_order") or [])]
    if agent_defaults:
        order = [k for k in order if k in agent_defaults]
    order = list(dict.fromkeys(order)) + [k for k in agent_defaults if k not in order]

    layer_order = list(dict.fromkeys(str(x) for x in (raw.get("layer_order") or [])))
    if any(x not in ROLE_VALUES for x in layer_order):
        raise ValueError(f"layer_order may only contain {', '.join(ROLE_VALUES)}")
    layer_order += [x for x in ROLE_VALUES if x not in layer_order]

    return ScoringConfig(
        regime_weights=regime_weights,
        thresholds=thresholds,
        enabled=enabled,
        score_scales=score_scales,
        confidence_floor=floor,
        fallback_regime=fallback,
        owner_weights=owner_weights,
        owner_thresholds=owner_thresholds,
        owner_max_ev_to_fcf=owner_cap,
        roles=roles,
        agent_order=order,
        layer_order=layer_order,
    )


def _registered_defaults() -> dict[str, float]:
    # Imported lazily: config must stay importable without the agents package.
    from agents.registry import get_agent_classes

    return {key: cls.default_weight for key, cls in get_agent_classes().items()}


def registered_roles() -> dict[str, str]:
    """agent_key -> class role ("owner" or "timing") for every registered agent (before user roles)."""
    from agents.registry import get_agent_classes

    return {key: cls.role for key, cls in get_agent_classes().items()}


def load_raw_config() -> dict[str, Any]:
    return _deep_merge(_read_json(_defaults_path()), _read_json(_local_path()))


def load_scoring_config() -> ScoringConfig:
    """Load defaults + local overrides, filled with registered agents' defaults."""
    return validate_config(load_raw_config(), _registered_defaults(), registered_roles())


def save_scoring_config(payload: dict[str, Any]) -> ScoringConfig:
    """Validate and persist user overrides. Returns the effective config."""
    merged = _deep_merge(_read_json(_defaults_path()), payload)
    cfg = validate_config(merged, _registered_defaults(), registered_roles())
    # Only on save: load_scoring_config must still open a file written before this rule existed.
    if not any(r == OWNER_ROLE and cfg.enabled.get(k, True) and cfg.owner_weights.get(k, 0) > 0
               for k, r in cfg.roles.items()):
        raise ValueError("At least one deciding agent must stay on with a weight above 0, or every company comes back not rated.")
    with _LOCK:
        path = _local_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)
    return cfg


def reset_scoring_config() -> ScoringConfig:
    with _LOCK:
        _local_path().unlink(missing_ok=True)
    return load_scoring_config()


def has_local_overrides() -> bool:
    return _local_path().exists()
