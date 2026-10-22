"""Tests for the multi-agent orchestration: master agent, scoring and weights config."""

from __future__ import annotations

import time

import pytest

from agents.base import LIVE, OWNER, TIMING, UNAVAILABLE, AgentAssessment, BaseAgent
from agents.master_agent import MasterDecisionAgent
from agents.registry import get_agent_classes
from agents.scoring import aggregate, confidence_weight, decide, normalize_score
from config.weights import (
    REGIMES,
    ScoringConfig,
    load_scoring_config,
    reset_scoring_config,
    save_scoring_config,
    validate_config,
)
from tests.conftest import make_ohlcv

THRESHOLDS = {"strong_buy": 45.0, "buy": 15.0, "sell": -15.0, "strong_sell": -45.0}


def _config(weights: dict[str, float], owner: dict[str, float] | None = None, **kw) -> ScoringConfig:
    return ScoringConfig(
        regime_weights={r: dict(weights) for r in REGIMES},
        thresholds=dict(THRESHOLDS),
        owner_weights=dict(owner or {}),
        owner_thresholds=dict(THRESHOLDS),
        **kw,
    )


def _a(key: str, score: float, confidence: float = 80.0, quality: str = LIVE, regime: str | None = None) -> AgentAssessment:
    raw = {"regime": regime} if regime else {}
    return AgentAssessment(key.title(), score, "NEUTRAL", confidence, "", agent_key=key, data_quality=quality, raw_data=raw)


class StubAgent(BaseAgent):
    category = "Test"

    def __init__(self, key: str, score: float, confidence: float = 80.0, regime: str | None = None,
                 delay: float = 0.0, fail: bool = False, role: str = TIMING) -> None:
        super().__init__()
        self.key = key  # type: ignore[misc]
        self.name = f"Stub {key}"  # type: ignore[misc]
        self.role = role  # type: ignore[misc]
        self._score, self._conf, self._regime, self._delay, self._fail = score, confidence, regime, delay, fail

    def evaluate(self, ticker, ohlcv):
        if self._delay:
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("boom")
        raw = {"regime": self._regime} if self._regime else {}
        return self.assessment(self._score, self._conf, "stub", raw_data=raw)


# --------------------------------------------------------------- normalization


def test_score_normalization_equalizes_scales():
    # Insider (scale 35) amplifies more than Momentum (scale 55) for the same raw score.
    assert MasterDecisionAgent._normalize_score(35.0, "insider") > MasterDecisionAgent._normalize_score(35.0, "momentum")
    # Display names still work for lookups.
    assert MasterDecisionAgent._normalize_score(35.0, "Insider Trading Agent") == MasterDecisionAgent._normalize_score(35.0, "insider")
    assert abs(normalize_score(0.0, 50.0)) < 1e-9
    assert normalize_score(100.0, 55.0) > 85.0
    assert normalize_score(-100.0, 45.0) < -85.0


def test_confidence_weighting():
    assert confidence_weight(0.4, 95.0) > confidence_weight(0.4, 50.0) > confidence_weight(0.4, 10.0)
    assert confidence_weight(0.4, 100.0) == pytest.approx(0.4)
    assert confidence_weight(0.4, 0.0) == pytest.approx(0.16)  # 40% floor
    assert confidence_weight(0.4, 0.0, floor=0.0) == 0.0


def test_decide_thresholds():
    assert decide(50, THRESHOLDS) == "STRONG_BUY"
    assert decide(15, THRESHOLDS) == "BUY"
    assert decide(0, THRESHOLDS) == "HOLD"
    assert decide(-15, THRESHOLDS) == "SELL"
    assert decide(-60, THRESHOLDS) == "STRONG_SELL"


# ------------------------------------------------------------------ aggregate


def test_aggregate_excludes_unavailable_and_zero_weight():
    cfg = _config({"a": 0.5, "b": 0.5, "c": 0.0})
    res = aggregate(
        [_a("a", 60), _a("b", -80, quality=UNAVAILABLE), _a("c", -90)],
        cfg, "BULL", {"a": 50, "b": 50, "c": 50},
    )
    by_key = {c.agent_key: c for c in res.contributions}
    assert by_key["a"].included and by_key["a"].weight_share == pytest.approx(1.0)
    assert not by_key["b"].included and by_key["b"].exclusion_reason == "No usable data"
    assert not by_key["c"].included and "informational" in by_key["c"].exclusion_reason
    assert res.composite_score == pytest.approx(normalize_score(60, 50), abs=0.1)
    # Half the configured weight had no data: flagged and confidence reduced.
    assert res.coverage == pytest.approx(0.5)
    assert any("configured weight" in w for w in res.warnings)
    assert res.confidence < 80


def test_aggregate_disabled_agent_is_excluded():
    cfg = _config({"a": 0.5, "b": 0.5}, enabled={"a": True, "b": False})
    res = aggregate([_a("a", 40), _a("b", -90)], cfg, "BULL", {})
    assert res.composite_score > 0
    assert {c.agent_key: c.exclusion_reason for c in res.contributions}["b"] == "Disabled in weights config"


def test_aggregate_contributions_sum_to_composite():
    cfg = _config({"a": 0.3, "b": 0.5, "c": 0.2})
    res = aggregate([_a("a", 50, 90), _a("b", -20, 40), _a("c", 70, 70)], cfg, "BULL", {})
    assert sum(c.contribution for c in res.contributions) == pytest.approx(res.composite_score, abs=0.2)
    assert sum(c.weight_share for c in res.contributions if c.included) == pytest.approx(1.0, abs=1e-3)


def test_aggregate_confidence_changes_relative_influence():
    cfg = _config({"a": 0.5, "b": 0.5})
    confident_bull = aggregate([_a("a", 60, 95), _a("b", -60, 20)], cfg, "BULL", {})
    confident_bear = aggregate([_a("a", 60, 20), _a("b", -60, 95)], cfg, "BULL", {})
    assert confident_bull.composite_score > 0 > confident_bear.composite_score


def test_aggregate_conflict_detection():
    cfg = _config({"a": 0.5, "b": 0.5})
    res = aggregate([_a("a", 90, 80), _a("b", -90, 80)], cfg, "BULL", {})
    assert res.conflict
    assert any("Timing agents disagree" in w and "reads up" in w and "stage any entry" in w for w in res.warnings)
    assert not any("bullish" in w or "bearish" in w or " Agent" in w for w in res.warnings)
    calm = aggregate([_a("a", 90, 80), _a("b", 70, 80)], cfg, "BULL", {})
    assert not calm.conflict
    assert calm.confidence > res.confidence


def test_aggregate_no_data_defaults_to_hold():
    cfg = _config({"a": 1.0})
    res = aggregate([_a("a", 90, quality=UNAVAILABLE)], cfg, "BULL", {})
    assert res.decision == "HOLD" and res.composite_score == 0.0 and res.confidence == 0.0


def test_aggregate_unknown_regime_uses_fallback():
    cfg = _config({"a": 1.0})
    res = aggregate([_a("a", 30)], cfg, None, {})
    assert res.regime == "CORRECTION"
    assert any("unknown" in w.lower() for w in res.warnings)


# --------------------------------------------------------------------- config


def test_default_profiles_sum_to_one_and_cover_registered_agents():
    cfg = load_scoring_config()
    classes = get_agent_classes()
    registered = set(classes)
    owner_agents = {k for k, c in classes.items() if c.role == OWNER}
    for regime in REGIMES:
        profile = cfg.regime_weights[regime]
        assert sum(profile.values()) == pytest.approx(1.0, abs=1e-6), regime
        assert registered <= set(profile), f"{regime} missing {registered - set(profile)}"
        # Owner agents never vote in the timing panel.
        assert all(profile[k] == 0.0 for k in owner_agents), regime
    assert sum(cfg.owner_weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert owner_agents <= set(cfg.owner_weights)
    assert cfg.owner_weights["shareholder"] > cfg.owner_weights["fundamentals"] > cfg.owner_weights["insider"] == 0.0


def test_dynamic_weight_selection():
    master = MasterDecisionAgent(agents=[])
    bull, bear = master._select_weights("BULL"), master._select_weights("BEAR")
    assert bull["momentum"] > bull["insider"]
    assert bear["momentum"] < bull["momentum"]
    assert bear["macro"] > bull["macro"]


def test_weight_overrides_bypass_regime():
    custom = {"momentum": 0.5, "options": 0.2, "insider": 0.2, "macro": 0.1}
    master = MasterDecisionAgent(weight_overrides=custom, agents=[])
    assert master._select_weights("BEAR") == custom


def test_validate_config_fills_new_agents_with_default_weight():
    cfg = validate_config({"regime_weights": {"BULL": {"momentum": 0.3}}}, {"momentum": 0.25, "brand_new": 0.07})
    assert cfg.regime_weights["BULL"]["momentum"] == 0.3
    assert cfg.regime_weights["BULL"]["brand_new"] == 0.07
    assert cfg.regime_weights["CRISIS"]["brand_new"] == 0.07
    assert cfg.enabled["brand_new"] is True


@pytest.mark.parametrize("bad", [
    {"thresholds": {"strong_buy": 10, "buy": 20, "sell": -15, "strong_sell": -45}},
    {"regime_weights": {"BULL": {"momentum": -0.1}}},
    {"confidence_floor": 1.5},
    {"fallback_regime": "SIDEWAYS"},
])
def test_validate_config_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_config(bad, {})


def test_save_and_reset_config_roundtrip(_isolated_store):
    cfg = load_scoring_config()
    payload = cfg.to_dict()
    payload["regime_weights"]["BULL"]["momentum"] = 0.9
    payload["thresholds"]["buy"] = 20.0
    save_scoring_config(payload)
    assert (_isolated_store / "agent_weights.json").exists()
    reloaded = load_scoring_config()
    assert reloaded.regime_weights["BULL"]["momentum"] == 0.9
    assert reloaded.thresholds["buy"] == 20.0
    restored = reset_scoring_config()
    assert restored.regime_weights["BULL"]["momentum"] == cfg.regime_weights["BULL"]["momentum"]


def test_save_refuses_a_setup_with_no_deciding_agent(_isolated_store):
    payload = load_scoring_config().to_dict()
    owners = [k for k, r in payload["roles"].items() if r == "owner"]
    assert owners
    for k in owners:
        payload["enabled"][k] = False
    with pytest.raises(ValueError, match="At least one deciding agent must stay on"):
        save_scoring_config(payload)
    assert not (_isolated_store / "agent_weights.json").exists()
    payload["enabled"][owners[0]] = True
    payload["owner_weights"] = {**payload["owner_weights"], **{k: 0 for k in owners}}
    with pytest.raises(ValueError, match="weight above 0"):
        save_scoring_config(payload)


# --------------------------------------------------------------- orchestration


def test_master_uses_regime_agent_profile_for_timing_and_owner_decides():
    cfg = ScoringConfig(
        regime_weights={
            "BULL": {"b": 1.0, "c": 0.0, "m": 0.0},
            "CORRECTION": {"b": 0.0, "c": 1.0, "m": 0.0},
            "BEAR": {"b": 0.0, "c": 1.0, "m": 0.0},
            "CRISIS": {"b": 0.0, "c": 1.0, "m": 0.0},
        },
        thresholds=dict(THRESHOLDS),
        owner_weights={"a": 1.0},
        owner_thresholds=dict(THRESHOLDS),
    )
    agents = [StubAgent("a", 80, role=OWNER), StubAgent("b", -80), StubAgent("c", 80), StubAgent("m", 0, regime="BULL")]
    rec = MasterDecisionAgent(config=cfg, agents=agents).analyze("test", make_ohlcv())
    # The legacy regime lookup picks up the stub's raw_data regime, which selects the timing profile.
    assert rec.market_regime == "BULL"
    assert rec.ticker == "TEST"
    # Owner layer decides: a is bullish, so STRONG_BUY even though the BULL timing profile (b) is very bearish.
    assert rec.decision == "STRONG_BUY" and rec.composite_score > 0
    assert rec.timing["score"] < 0 and rec.timing["label"] == "UNFAVORABLE"
    layers = {c["agent_key"]: c["layer"] for c in rec.contributions}
    assert layers == {"a": OWNER, "b": TIMING, "c": TIMING, "m": TIMING}


def test_master_isolates_failing_and_slow_agents():
    cfg = _config({}, owner={"ok": 0.4, "bad": 0.3, "slow": 0.3})
    agents = [StubAgent("ok", 50, role=OWNER), StubAgent("bad", 90, fail=True, role=OWNER),
              StubAgent("slow", 90, delay=2.0, role=OWNER)]
    rec = MasterDecisionAgent(config=cfg, agents=agents, agent_timeout=0.5).analyze("TEST", make_ohlcv())
    quality = {a.agent_key: a.data_quality for a in rec.agent_assessments}
    assert quality == {"ok": LIVE, "bad": UNAVAILABLE, "slow": UNAVAILABLE}
    bad = next(a for a in rec.agent_assessments if a.agent_key == "bad")
    assert bad.error == "boom"
    assert rec.coverage == pytest.approx(0.4)
    assert rec.warnings


def test_master_trade_plan_is_long_only_risk_reference():
    cfg = _config({}, owner={"a": 1.0})
    ohlcv = make_ohlcv(noise=0.01)
    buy = MasterDecisionAgent(config=cfg, agents=[StubAgent("a", 100, 95, role=OWNER)]).analyze("T", ohlcv)
    assert buy.decision == "STRONG_BUY"
    assert buy.suggested_stop_loss < buy.suggested_entry < buy.suggested_target_price
    assert buy.reward_risk_ratio == pytest.approx(2.0, abs=0.05)
    assert "not an automatic exit" in buy.stop_loss_method
    sell = MasterDecisionAgent(config=cfg, agents=[StubAgent("a", -100, 95, role=OWNER)]).analyze("T", ohlcv)
    # SELL means do not own it; there is no short setup, only reference levels.
    assert sell.decision == "STRONG_SELL"
    assert sell.suggested_stop_loss < sell.suggested_entry
    assert "Reference levels only" in sell.stop_loss_method


def test_master_not_rated_without_owner_data():
    cfg = _config({"t": 1.0}, owner={"o": 1.0})

    class Missing(StubAgent):
        def evaluate(self, ticker, ohlcv):
            return self.unavailable("no statements")

    rec = MasterDecisionAgent(config=cfg, agents=[Missing("o", 0, role=OWNER), StubAgent("t", 60)]).analyze("SPY", make_ohlcv())
    assert rec.decision == "NOT_RATED" and rec.composite_score == 0.0
    assert rec.owner["applicable"] is False
    assert rec.timing["label"] == "FAVORABLE" and rec.timing["score"] > 0
    assert any("Not rated" in w for w in rec.warnings)
    assert "not rated" in rec.executive_summary


def test_master_collects_key_levels():
    class LevelAgent(StubAgent):
        def evaluate(self, ticker, ohlcv):
            a = super().evaluate(ticker, ohlcv)
            a.raw_data["levels"] = [{"label": "Wall", "price": 150.0, "kind": "resistance"}, {"label": "bad", "price": "x"}]
            return a

    cfg = _config({"a": 1.0}, owner={})
    rec = MasterDecisionAgent(config=cfg, agents=[LevelAgent("a", 10)]).analyze("T", make_ohlcv())
    assert [lvl["label"] for lvl in rec.key_levels] == ["Wall"]
    assert rec.key_levels[0]["distance_pct"] == pytest.approx((150 / 140 - 1) * 100, abs=0.01)


def test_all_registered_agents_run_without_network():
    """Every real agent must degrade to UNAVAILABLE/PARTIAL instead of raising offline."""
    rec = MasterDecisionAgent().analyze("TEST", make_ohlcv(noise=0.01))
    keys = {a.agent_key for a in rec.agent_assessments}
    assert {"momentum", "volume", "options", "insider", "macro", "pullback", "fundamentals", "shareholder"} <= keys
    assert rec.decision == "NOT_RATED"  # no fundamentals offline
    for a in rec.agent_assessments:
        assert a.error is None or a.data_quality == UNAVAILABLE
    offline = {a.agent_key for a in rec.agent_assessments if a.data_quality == UNAVAILABLE}
    assert {"options", "insider", "macro"} <= offline
    assert "momentum" not in offline
