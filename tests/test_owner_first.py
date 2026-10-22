"""Owner-first decision flow: layered scoring, pullback agent, VIX rule, portfolio review and candidate sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config.settings as settings
from agents.base import INFO, LIVE, OWNER, TIMING, UNAVAILABLE, AgentAssessment
from agents.master_agent import MasterDecisionAgent
from agents.pullback_agent import PullbackAgent
from agents.scoring import NOT_RATED, aggregate_layers, timing_label
from config.weights import REGIMES, ScoringConfig, validate_config
from core import portfolio as pf
from data import portfolio as store
from data.macro_data import contrarian_vix_points
from tests.conftest import make_ohlcv
from tests.test_multi_agent import THRESHOLDS, StubAgent

ROLES = {"fund": OWNER, "share": OWNER, "ins": OWNER, "mom": TIMING, "macro": TIMING}


def _cfg(owner: dict[str, float] | None = None, timing: dict[str, float] | None = None) -> ScoringConfig:
    return ScoringConfig(
        regime_weights={r: dict(timing or {"mom": 0.6, "macro": 0.4}) for r in REGIMES},
        thresholds=dict(THRESHOLDS),
        owner_weights=dict(owner or {"fund": 0.45, "share": 0.55, "ins": 0.0}),
        owner_thresholds=dict(THRESHOLDS),
    )


def _a(key: str, score: float, confidence: float = 80.0, quality: str = LIVE, regime: str | None = None) -> AgentAssessment:
    raw = {"regime": regime} if regime else {}
    return AgentAssessment(key, score, "NEUTRAL", confidence, "", agent_key=key, data_quality=quality, raw_data=raw)


# ------------------------------------------------------------------ scoring


def test_owner_layer_decides_and_timing_has_no_vote():
    res = aggregate_layers([_a("fund", -70), _a("share", -60), _a("mom", 90), _a("macro", 90, regime="BULL")],
                           _cfg(), "BULL", {}, ROLES)
    assert res.decision in ("SELL", "STRONG_SELL")
    assert res.owner.composite_score < 0 < res.timing.composite_score
    assert res.timing_label == "FAVORABLE"
    assert res.owner_applicable
    assert {c.agent_key for c in res.owner.contributions} == {"fund", "share"}
    assert {c.agent_key for c in res.timing.contributions} == {"mom", "macro"}
    assert all(c.layer == OWNER for c in res.owner.contributions)
    assert res.to_dict()["composite_score"] == res.owner.composite_score


def test_insider_is_owner_evidence_without_weight():
    res = aggregate_layers([_a("fund", 50), _a("share", 50), _a("ins", -100), _a("mom", 0)], _cfg(), "BULL", {}, ROLES)
    ins = next(c for c in res.owner.contributions if c.agent_key == "ins")
    assert not ins.included and "informational" in ins.exclusion_reason
    assert res.decision.endswith("BUY")


def test_not_rated_when_owner_layer_has_no_data():
    res = aggregate_layers([_a("fund", 0, quality=UNAVAILABLE), _a("share", 0, quality=UNAVAILABLE), _a("mom", 40)],
                           _cfg(), "BULL", {}, ROLES)
    assert res.decision == NOT_RATED and not res.owner_applicable
    assert res.owner.composite_score == 0.0
    assert res.timing.composite_score > 0
    assert any("Not rated" in w for w in res.warnings)


def test_utilities_and_reits_are_not_rated_even_with_owner_data():
    # T-12 (owner's call): the owner agents still score, the layered decision is NOT_RATED with the reason
    fund = AgentAssessment("fund", 60.0, "BUY", 80.0, "", agent_key="fund", data_quality=LIVE,
                           raw_data={"sector": "Utilities", "excluded_sector": True})
    res = aggregate_layers([fund, _a("share", 50), _a("mom", 20)], _cfg(), "BULL", {}, ROLES)
    assert res.decision == NOT_RATED and not res.owner_applicable
    assert any(w.startswith("Not rated: Utilities") and "T-12" in w for w in res.warnings)
    assert res.timing_label in ("FAVORABLE", "NEUTRAL", "UNFAVORABLE")
    normal = aggregate_layers([_a("fund", 60), _a("share", 50), _a("mom", 20)], _cfg(), "BULL", {}, ROLES)
    assert normal.decision.endswith("BUY")


def test_not_rated_reason_reaches_the_portfolio_actions():
    from types import SimpleNamespace

    reason = "Not rated: Real Estate is outside the value framework (owner's call, decision_log.md T-12)."
    rec = SimpleNamespace(decision=NOT_RATED, owner={"score": 0.0, "not_rated_reason": reason, "coverage": 1.0},
                          warnings=[reason], timing={}, composite_score=0.0)
    action, reasons = pf._action(rec, 0.05, {"from_high": -0.02, "basis": ""}, None, 1000.0)
    # Outside the framework by the owner's call (T-12): a standing REVIEW would repeat itself every time.
    assert action == "HOLD" and reasons[0].startswith(reason) and "Judge the dividend cover" in reasons[0]
    no_data = SimpleNamespace(decision=NOT_RATED, owner={"score": 0.0, "not_rated_reason": "Not rated: no owner-layer data.", "coverage": 0.0},
                              warnings=[], timing={}, composite_score=0.0)
    action, reasons = pf._action(no_data, 0.05, {"from_high": -0.02, "basis": ""}, None, 1000.0)
    assert action == "REVIEW" and "statements could not be loaded" in reasons[0]
    verdict, reasons, _ = pf._verdict(rec)
    assert verdict == "NOT_RATED" and reasons == [reason]


def test_owner_score_ignores_the_regime():
    for regime in REGIMES:
        res = aggregate_layers([_a("fund", 30), _a("share", 30), _a("mom", -90)], _cfg(), regime, {}, ROLES)
        assert res.owner.composite_score == pytest.approx(aggregate_layers(
            [_a("fund", 30), _a("share", 30), _a("mom", -90)], _cfg(), "BULL", {}, ROLES).owner.composite_score)


def test_margin_of_safety_caps_expensive_buy_at_hold():
    cfg = _cfg(owner={"fundamentals": 0.45, "shareholder": 0.55})
    roles = {"fundamentals": OWNER, "shareholder": OWNER, "mom": TIMING}
    pricey = AgentAssessment("F", -5, "NEUTRAL", 70, "", agent_key="fundamentals",
                             raw_data={"ev_to_fcf": 60.0, "fcf": 1.0, "is_financial": False})
    cannibal = _a("shareholder", 30)
    res = aggregate_layers([pricey, cannibal, _a("mom", 0)], cfg, "BULL", {}, roles)
    assert res.owner.decision == "BUY" and res.decision == "HOLD"
    assert any("Margin of safety" in w and "60x" in w for w in res.warnings)
    # Cheap enough: the BUY stands.
    cheap = AgentAssessment("F", -5, "NEUTRAL", 70, "", agent_key="fundamentals",
                            raw_data={"ev_to_fcf": 20.0, "fcf": 1.0, "is_financial": False})
    assert aggregate_layers([cheap, cannibal, _a("mom", 0)], cfg, "BULL", {}, roles).decision == "BUY"
    # Negative FCF has no multiple to anchor on: capped as well.
    burning = AgentAssessment("F", -5, "NEUTRAL", 70, "", agent_key="fundamentals",
                              raw_data={"ev_to_fcf": None, "fcf": -3.0, "is_financial": False})
    assert aggregate_layers([burning, cannibal, _a("mom", 0)], cfg, "BULL", {}, roles).decision == "HOLD"
    # Financials are not judged on EV/FCF; the cap can be switched off.
    bank = AgentAssessment("F", -5, "NEUTRAL", 70, "", agent_key="fundamentals",
                           raw_data={"ev_to_fcf": None, "fcf": None, "is_financial": True})
    assert aggregate_layers([bank, cannibal, _a("mom", 0)], cfg, "BULL", {}, roles).decision == "BUY"
    cfg.owner_max_ev_to_fcf = None
    assert aggregate_layers([pricey, cannibal, _a("mom", 0)], cfg, "BULL", {}, roles).decision == "BUY"
    with pytest.raises(ValueError):
        validate_config({"owner_max_ev_to_fcf": -5}, {})
    assert validate_config({"owner_max_ev_to_fcf": None}, {}).owner_max_ev_to_fcf is None
    assert validate_config({}, {}).owner_max_ev_to_fcf == 35.0


def test_timing_label_thresholds():
    assert timing_label(20, THRESHOLDS) == "FAVORABLE"
    assert timing_label(0, THRESHOLDS) == "NEUTRAL"
    assert timing_label(-20, THRESHOLDS) == "UNFAVORABLE"


def test_validate_config_owner_agents_default_to_zero_in_timing_profiles():
    cfg = validate_config({"regime_weights": {"BULL": {"momentum": 0.3}}},
                          {"momentum": 0.25, "fundamentals": 0.45, "shareholder": 0.55},
                          {"momentum": TIMING, "fundamentals": OWNER, "shareholder": OWNER})
    assert cfg.regime_weights["BULL"]["fundamentals"] == 0.0 and cfg.regime_weights["BEAR"]["shareholder"] == 0.0
    assert cfg.owner_weights == {"fundamentals": 0.45, "shareholder": 0.55}
    assert cfg.owner_thresholds == cfg.thresholds
    with pytest.raises(ValueError):
        validate_config({"owner_thresholds": {"strong_buy": 0, "buy": 10, "sell": -10, "strong_sell": -45}}, {})
    with pytest.raises(ValueError):
        validate_config({"owner_weights": {"fundamentals": -1}}, {})


def test_validate_config_user_roles_and_display_order():
    defaults = {"momentum": 0.25, "fundamentals": 0.45, "shareholder": 0.55}
    classes = {"momentum": TIMING, "fundamentals": OWNER, "shareholder": OWNER}
    cfg = validate_config({"roles": {"momentum": "owner", "fundamentals": "timing"},
                           "agent_order": ["shareholder", "ghost", "shareholder"], "layer_order": ["timing"]},
                          defaults, classes)
    assert cfg.roles == {"momentum": OWNER, "fundamentals": TIMING, "shareholder": OWNER}
    # A moved agent starts at weight 0 in its new layer, so a role change never silently moves a score.
    assert cfg.owner_weights["momentum"] == 0.0
    assert all(cfg.regime_weights[r]["fundamentals"] == 0.0 for r in REGIMES)
    assert cfg.agent_order == ["shareholder", "momentum", "fundamentals"]  # unknown and repeated keys dropped
    assert cfg.layer_order == ["timing", "owner", "info"]
    assert validate_config({}, defaults, classes).to_dict()["roles"] == classes
    with pytest.raises(ValueError):
        validate_config({"roles": {"momentum": "boss"}}, defaults, classes)
    with pytest.raises(ValueError):  # nothing left to decide
        validate_config({"roles": {"fundamentals": "info", "shareholder": "timing"}}, defaults, classes)
    with pytest.raises(ValueError):
        validate_config({"layer_order": ["owner", "later"]}, defaults, classes)


def test_user_roles_move_agents_between_layers_with_a_warning():
    cfg = _cfg(owner={"fund": 0.5, "share": 0.0, "mom": 0.5})
    cfg.roles = {"mom": OWNER, "share": INFO}
    assessments = [_a("fund", 40), _a("share", -90), _a("mom", 90), _a("macro", 10, regime="BULL")]
    res = aggregate_layers(assessments, cfg, "BULL", {}, ROLES)
    assert {c.agent_key for c in res.owner.contributions} == {"fund", "mom"}
    assert {c.agent_key for c in res.timing.contributions} == {"macro"}
    assert [(c.agent_key, c.layer, c.included) for c in res.informative] == [("share", INFO, False)]
    assert "share" in {c.agent_key for c in res.contributions}
    assert any(w.startswith("Your weights let mom decide") and "12.2%" in w for w in res.warnings)
    shipped = aggregate_layers(assessments, _cfg(), "BULL", {}, ROLES)
    assert not any(w.startswith("Your weights let") for w in shipped.warnings)
    assert shipped.informative == []


def test_excluded_sector_is_not_rated_whatever_the_roles():
    fund = AgentAssessment("fund", 60.0, "BUY", 80.0, "", agent_key="fund", data_quality=LIVE,
                           raw_data={"sector": "Real Estate", "excluded_sector": True})
    cfg = _cfg(owner={"share": 1.0})
    cfg.roles = {"fund": INFO}
    res = aggregate_layers([fund, _a("share", 50), _a("mom", 20)], cfg, "BULL", {}, ROLES)
    assert res.decision == NOT_RATED


# ------------------------------------------------------------------ pullback agent


def _pullback_frame(kind: str) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=320)
    trend = np.linspace(100, 160, 320)
    if kind == "pullback":
        trend[-6:] = trend[-6:] - np.linspace(2, 9, 6)  # a dip of about 2 ATR below the 20-day mean
    elif kind == "extended":
        trend[-5:] = trend[-5:] + np.linspace(3, 14, 5)
    elif kind == "broken":
        trend = np.linspace(160, 100, 320)
    close = pd.Series(trend, index=idx)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


def test_pullback_agent_readings():
    agent = PullbackAgent()
    assert agent.role == TIMING and agent.point_in_time and agent.default_weight == 0.15
    pull = agent.evaluate("T", _pullback_frame("pullback"))
    assert pull.raw_data["setup"] == "pullback_in_uptrend" and pull.score > 0
    ext = agent.evaluate("T", _pullback_frame("extended"))
    assert ext.raw_data["setup"] == "extended" and ext.score < 0
    broken = agent.evaluate("T", _pullback_frame("broken"))
    assert broken.raw_data["setup"] == "trend_broken" and broken.score == 0
    assert agent.evaluate("T", make_ohlcv(periods=100)).data_quality == UNAVAILABLE


def test_master_timing_panel_reports_setup_and_notes():
    cfg = _cfg(owner={"o": 1.0}, timing={"pullback": 1.0})
    master = MasterDecisionAgent(config=cfg, agents=[StubAgent("o", 60, role=OWNER), PullbackAgent()])
    rec = master.analyze("T", _pullback_frame("pullback"))
    assert rec.decision.endswith("BUY")
    assert rec.timing["setup"] == "pullback_in_uptrend" and rec.timing["trend"] == "UP"
    assert any("Pullback inside an uptrend" in n for n in rec.timing["notes"])
    assert rec.timing["momentum_12_1"] is not None and rec.timing["momentum_12_1"] > 0
    broken = master.analyze("T", _pullback_frame("broken"))
    assert broken.timing["trend"] == "DOWN" and broken.timing["value_trap_check"]
    assert any("Trend is down" in n for n in broken.timing["notes"])
    assert any("value-trap check" in n for n in broken.timing["notes"])


# ------------------------------------------------------------------ VIX rule (T-09)


def test_contrarian_vix_rewards_only_extreme_fear():
    assert contrarian_vix_points(45.0)[0] == 6.0
    assert contrarian_vix_points(33.0)[0] == 0.0
    assert contrarian_vix_points(22.0)[0] == 0.5
    assert contrarian_vix_points(11.0)[0] == -1.0


# ------------------------------------------------------------------ portfolio store


def test_portfolio_store_roundtrip_and_validation():
    assert store.load_portfolio() == {"cash": 0.0, "currency": "EUR", "positions": [],
                                      "plan": {"core_target": None, "derivatives_budget": None}}
    p = store.upsert_position({"ticker": "aapl", "shares": 10, "avg_cost": 150, "opened": "2026-01-15", "notes": "core"})
    assert p["positions"][0]["ticker"] == "AAPL" and p["positions"][0]["opened"] == "2026-01-15"
    p = store.upsert_position({"ticker": "AAPL", "shares": 12, "avg_cost": 152, "owner_score_at_entry": 20.5, "owner_decision_at_entry": "BUY"})
    assert len(p["positions"]) == 1 and p["positions"][0]["shares"] == 12
    assert p["positions"][0]["opened"] == "2026-01-15"  # kept from the earlier entry
    p = store.upsert_position({"ticker": "AAPL", "shares": 12, "avg_cost": 152})
    assert p["positions"][0]["owner_score_at_entry"] == 20.5  # snapshot kept
    p = store.set_cash(5000)
    assert p["cash"] == 5000.0
    assert store.get_position("AAPL")["avg_cost"] == 152.0
    p = store.remove_position("AAPL")
    assert p["positions"] == [] and p["cash"] == 5000.0
    for bad in ({"ticker": "<x>", "shares": 1, "avg_cost": 1}, {"ticker": "A", "shares": 0, "avg_cost": 1},
                {"ticker": "A", "shares": 1, "avg_cost": 1, "opened": "yesterday"}):
        with pytest.raises(store.PortfolioError):
            store.clean_position(bad)
    with pytest.raises(store.PortfolioError):
        store.clean_portfolio({"cash": -1, "positions": []})
    with pytest.raises(store.PortfolioError):
        store.clean_portfolio({"positions": [{"ticker": "A", "shares": 1, "avg_cost": 1}, {"ticker": "A", "shares": 2, "avg_cost": 1}]})


# ------------------------------------------------------------------ portfolio review


class _OwnerByTicker(StubAgent):
    """Owner stub whose score depends on the ticker, so one master can review a mixed portfolio."""

    def __init__(self, scores: dict[str, float]) -> None:
        super().__init__("owner", 0, role=OWNER)
        self._scores = scores

    def evaluate(self, ticker, ohlcv):
        score = self._scores.get(ticker)
        if score is None:
            return self.unavailable("no statements")
        return self.assessment(score, 80, "stub", raw_data={"score_breakdown": [{"factor": "Test rule", "points": min(score, -10.0)}]})


def _fetched(prices: dict[str, tuple[float, float]], volume: float = 2e6, noise: float = 0.01) -> dict:
    out = {}
    for i, (ticker, (start, end)) in enumerate(prices.items()):
        ohlcv = make_ohlcv(periods=300, start=start, end=end, volume=volume, noise=noise, seed=i)
        out[ticker] = (ohlcv, {"status": "CLOSE", "as_of": ohlcv.index[-1].strftime("%Y-%m-%d")})
    return out


def test_review_portfolio_actions_and_summary():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}),
                                 agents=[_OwnerByTicker({"GOOD": 70.0, "BAD": -80.0, "MEH": 5.0}), StubAgent("mom", 30)])
    fetched = _fetched({"GOOD": (100, 130), "BAD": (50, 40), "MEH": (20, 21), "SPY": (400, 450)})
    portfolio = {
        "cash": 10_000.0,
        "positions": [
            {"ticker": "GOOD", "shares": 100, "avg_cost": 110.0, "opened": "2024-06-03", "owner_score_at_entry": 40.0, "owner_decision_at_entry": "BUY"},
            {"ticker": "BAD", "shares": 200, "avg_cost": 55.0, "opened": "2024-03-01"},
            {"ticker": "MEH", "shares": 100, "avg_cost": 20.0},
            {"ticker": "SPY", "shares": 10, "avg_cost": 420.0},
        ],
    }
    profiles = {"SPY": {"asset_type": "ETF", "name": "SPDR S&P 500 ETF Trust"}, "GOOD": {"asset_type": "Stock"},
                "BAD": {"asset_type": "Stock"}, "MEH": {"asset_type": "Stock"}}
    result = pf.review_portfolio(portfolio, master=master, record=False, fetched=fetched, profiles=profiles, histories={})
    rows = {r["ticker"]: r for r in result["positions"]}
    assert result["errors"] == []
    assert rows["BAD"]["action"] == "EXIT" and rows["BAD"]["decision"] == "STRONG_SELL"
    # A broad index fund is not rated by design: it is a core fund and keeps HOLD, with its class as the reason.
    assert rows["SPY"]["action"] == "HOLD" and rows["SPY"]["decision"] == NOT_RATED and rows["SPY"]["class"] == "core"
    assert rows["SPY"]["rule"] == "core_fund" and not rows["SPY"]["rated"]
    assert rows["MEH"]["action"] == "HOLD" and "between the entry and exit levels" in rows["MEH"]["reasons"][0]
    assert rows["GOOD"]["action"] in ("HOLD", "TRIM")
    assert rows["GOOD"]["owner_change"] == pytest.approx(rows["GOOD"]["owner_score"] - 40.0, abs=0.1)
    assert rows["GOOD"]["pnl_pct"] > 0 > rows["BAD"]["pnl_pct"]
    assert rows["BAD"]["concerns"] and "Test rule" in rows["BAD"]["concerns"][0]
    assert result["positions"][0]["action"] == "EXIT"  # sorted by action severity

    s = result["summary"]
    total = 10_000 + sum(r["value"] for r in rows.values())
    assert s["total_value"] == pytest.approx(total, rel=1e-6)
    assert sum(r["weight"] for r in rows.values()) + s["cash_weight"] == pytest.approx(1.0, abs=1e-3)
    assert s["positions"] == 4 and s["actions"]["EXIT"] == 1
    assert s["volatility"] is not None and s["max_drawdown_1y"] is not None
    assert s["avg_correlation"] is not None and s["max_correlation"]["value"] <= 1.0
    assert s["owner_sell_weight"] == pytest.approx(rows["BAD"]["weight"], abs=1e-3)
    assert s["not_rated_weight"] == 0.0  # the fund is not a company the framework failed to rate
    assert not any("rates SELL" in n for n in result["notes"])  # the Sell group says it


def test_review_flags_concentration_and_drawdown():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}),
                                 agents=[_OwnerByTicker({"BIG": 40.0, "DOWN": 5.0, "FALL": 40.0}), StubAgent("mom", 0)])
    fetched = _fetched({"BIG": (100, 100), "DOWN": (100, 60), "FALL": (100, 60)})
    portfolio = {"cash": 0.0, "positions": [
        {"ticker": "BIG", "shares": 1000, "avg_cost": 90.0},
        {"ticker": "DOWN", "shares": 100, "avg_cost": 100.0, "opened": "2024-01-02"},
        {"ticker": "FALL", "shares": 100, "avg_cost": 100.0, "opened": "2024-01-02"},
    ]}
    result = pf.review_portfolio(portfolio, master=master, record=False, fetched=fetched, profiles={}, histories={})
    rows = {r["ticker"]: r for r in result["positions"]}
    assert rows["BIG"]["action"] == "TRIM" and "concentration" in rows["BIG"]["flags"]
    # A fall asks for a look when the owner score is in the HOLD band; for a BUY the fall is where the best returns were.
    assert rows["DOWN"]["action"] == "REVIEW" and rows["DOWN"]["rule"] == "drawdown" and "drawdown" in rows["DOWN"]["flags"]
    assert rows["DOWN"]["drawdown"]["from_high"] < -settings.DRAWDOWN_REVIEW
    assert rows["FALL"]["action"] == "HOLD" and rows["FALL"]["rule"] == "thesis_intact" and "2011-2026" in rows["FALL"]["why"]
    assert any("above the" in n and "cap" in n for n in result["notes"])


def test_review_empty_portfolio():
    result = pf.review_portfolio({"cash": 500.0, "positions": []}, record=False, fetched={})
    assert result["positions"] == [] and result["summary"]["total_value"] == 500.0


# ------------------------------------------------------------------ candidate


def test_candidate_enter_with_sizing():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}),
                                 agents=[_OwnerByTicker({"NEW": 60.0, "HELD": 30.0}), StubAgent("mom", 20)])
    fetched = _fetched({"NEW": (50, 70), "HELD": (100, 110)}, noise=0.002)  # clean uptrends: nothing to stage for
    portfolio = {"cash": 20_000.0, "positions": [{"ticker": "HELD", "shares": 100, "avg_cost": 100.0}]}
    result = pf.assess_candidate("new", portfolio, master=master, record=False, fetched=fetched)
    assert result["verdict"] == "ENTER" and not result["held"], result["reasons"]
    total = 20_000 + 100 * float(fetched["HELD"][0]["close"].iloc[-1])
    assert result["portfolio_value"] == pytest.approx(total, rel=1e-6)
    sz = result["sizing"]
    assert sz["shares"] > 0 and sz["weight"] <= settings.NEW_POSITION_WEIGHT + 1e-6
    assert sz["binding"] in sz["constraints"] and sz["shares"] == min(sz["constraints"].values())
    assert sz["risk_amount"] <= settings.RISK_PER_POSITION * total + 1.0
    assert result["entry_snapshot"]["owner_decision_at_entry"].endswith("BUY")
    assert result["staging"]["tranches"] == 1


def test_candidate_watch_avoid_not_rated_and_held():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}),
                                 agents=[_OwnerByTicker({"MEH": 5.0, "BAD": -60.0, "HELD": 30.0}), StubAgent("mom", 0)])
    fetched = _fetched({"MEH": (20, 21), "BAD": (20, 15), "ETF": (300, 320), "HELD": (100, 110)})
    portfolio = {"cash": 1_000.0, "positions": [{"ticker": "HELD", "shares": 10, "avg_cost": 100.0}]}
    assert pf.assess_candidate("MEH", portfolio, master=master, record=False, fetched=fetched)["verdict"] == "WATCH"
    assert pf.assess_candidate("BAD", portfolio, master=master, record=False, fetched=fetched)["verdict"] == "AVOID"
    etf = pf.assess_candidate("ETF", portfolio, master=master, record=False, fetched=fetched)
    assert etf["verdict"] == "NOT_RATED" and etf["sizing"] is None
    held = pf.assess_candidate("HELD", portfolio, master=master, record=False, fetched=fetched)
    assert held["held"] and held["position"]["ticker"] == "HELD" and held["verdict"] == held["position"]["action"]
    with pytest.raises(LookupError):
        pf.assess_candidate("NONE", portfolio, master=master, record=False, fetched=fetched)


def test_candidate_staged_when_trend_is_broken():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}),
                                 agents=[_OwnerByTicker({"FALL": 60.0}), StubAgent("mom", -40)])
    fetched = _fetched({"FALL": (100, 60)})
    result = pf.assess_candidate("FALL", {"cash": 10_000.0, "positions": []}, master=master, record=False, fetched=fetched)
    assert result["verdict"] == "ENTER_STAGED" and result["staging"]["tranches"] >= 2
    assert any("Stage the entry" in r for r in result["reasons"])


# ------------------------------------------------------------------ API


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_api_portfolio_crud(client):
    assert client.get("/api/portfolio").get_json()["positions"] == []
    res = client.post("/api/portfolio/position", json={"ticker": "msft", "shares": 5, "avg_cost": 400, "opened": "2026-02-02"})
    assert res.status_code == 200 and res.get_json()["positions"][0]["ticker"] == "MSFT"
    assert client.post("/api/portfolio/position", json={"ticker": "msft", "shares": 0, "avg_cost": 400}).status_code == 400
    res = client.put("/api/portfolio", json={"cash": 2500, "positions": [{"ticker": "MSFT", "shares": 5, "avg_cost": 400}]})
    assert res.get_json()["cash"] == 2500.0 and "limits" in res.get_json()
    res = client.delete("/api/portfolio/position/MSFT")
    assert res.get_json()["positions"] == []
    assert client.get("/api/agents").get_json()["portfolio_limits"]["max_position_weight"] == settings.MAX_POSITION_WEIGHT


def test_api_portfolio_review_and_candidate(client, monkeypatch):
    import web.app as web_app

    monkeypatch.setattr(web_app, "review_portfolio", lambda record=True: {"positions": [], "summary": {"total_value": 1.0}, "errors": [], "notes": []})
    assert client.post("/api/portfolio/review", json={}).get_json()["summary"]["total_value"] == 1.0

    def fake_candidate(ticker, record=True):
        return {"ticker": ticker, "verdict": "WATCH", "held": False, "reasons": [], "analysis": {"decision": "HOLD"}}

    monkeypatch.setattr(web_app, "assess_candidate", fake_candidate)
    monkeypatch.setattr(web_app, "fetch_ohlcv", lambda tickers: {})
    data = client.post("/api/portfolio/candidate", json={"ticker": "ko"}).get_json()
    assert data["ticker"] == "KO" and data["verdict"] == "WATCH"
    assert client.post("/api/portfolio/candidate", json={"ticker": "<bad>"}).status_code == 400


# ------------------------------------------------------------------ exit-rule study


def test_exit_rule_book_enters_on_buy_and_exits_at_the_level():
    from core.exit_rule_study import held_book

    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    # A: BUY, then HOLD, then SELL at -16 (exits under -15, not under -30), then BUY again (re-entered).
    # B: BUY throughout. C: never BUY. D: BUY, then leaves the index.
    scores = {"A": [30, 5, -16, -16, 20, 20], "B": [40, 40, 40, 40, 40, 40], "C": [0, 0, 0, 0, 0, 0], "D": [30, 30, None, None, None, None]}
    rows = []
    for i, d in enumerate(dates):
        for t, s in scores.items():
            if s[i] is None:
                continue
            rows.append({"date": d, "ticker": t, "composite": float(s[i]),
                         "owner_decision_capped": "BUY" if s[i] >= 15 else "SELL" if s[i] <= -15 else "HOLD"})
    panel = pd.DataFrame(rows)
    fwd1 = pd.DataFrame({t: [0.01, 0.02, -0.01, 0.03, 0.01, 0.0] for t in scores}, index=dates)
    fwd1["A"] = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10]  # A's return is only earned while held

    import core.exit_rule_study as ers
    ers_min = ers.MIN_NAMES
    ers.MIN_NAMES = 1
    try:
        book15 = held_book(panel, fwd1, -15.0, 1, cost_bps=0)
        book30 = held_book(panel, fwd1, -30.0, 1, cost_bps=0)
        confirm2 = held_book(panel, fwd1, -15.0, 2, cost_bps=0)
    finally:
        ers.MIN_NAMES = ers_min
    exits15 = book15["exits"]
    assert list(exits15[exits15["why"] == "rule"]["ticker"]) == ["A"] and exits15[exits15["why"] == "rule"]["date"].iloc[0] == dates[2]
    assert list(exits15[exits15["why"] == "left"]["ticker"]) == ["D"]
    assert book30["exits"][book30["exits"]["why"] == "rule"].empty  # -16 never reaches -30
    # With confirmation, A is sold one month later (second month-end at or below -15)
    r2 = confirm2["exits"][confirm2["exits"]["why"] == "rule"]
    assert list(r2["ticker"]) == ["A"] and r2["date"].iloc[0] == dates[3]
    # Month 1 (index 0) holds A, B, D: mean of 0.10, 0.01, 0.01
    first = book15["returns"].iloc[0]
    assert first == pytest.approx((0.10 + 0.01 + 0.01) / 3)
    # Month 3 (index 2): A sold at the month-end, so the month's return excludes it (B only)
    assert book15["returns"].iloc[2] == pytest.approx(-0.01)
    assert book30["returns"].iloc[2] == pytest.approx((0.10 - 0.01) / 2)
