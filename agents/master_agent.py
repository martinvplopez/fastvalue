"""Master Decision Agent: runs every enabled agent and synthesizes one recommendation.

Two layers (see ``agents/scoring.py``):

- The **owner score** (fundamentals and shareholder agents, insider trades as
  evidence) is the verdict, a score from -100 to +100 on a 12-month horizon. It does not
  depend on the market regime. When it has no data (funds, indexes, missing
  statements, or a utility or REIT, which the owner does not rate) the ticker is NOT_RATED.
- The **timing panel** (trend, volume, options, macro, pullback) is summarized
  as a timing score with a trend reading, a 12-1 month
  momentum value-trap check, the pullback setup, event risk from options and
  the VIX. It informs entry timing, staging and sizing. It has no vote.

Agents come from the registry (``agents/*_agent.py``), weights and thresholds
from ``config/weights.py``, and the composite math from ``agents/scoring.py``.
Adding an agent or changing weights needs no change here.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from decimal import ROUND_HALF_UP, Decimal
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from agents.base import OWNER, TIMING, UNAVAILABLE, AgentAssessment, BaseAgent, utc_now_iso
from agents.registry import create_agents, get_agent_classes
from agents.scoring import (
    NOT_RATED,
    CompositeResult,
    LayeredResult,
    aggregate_layers,
    confidence_weight,
    normalize_score,
)
from config.weights import ScoringConfig, load_scoring_config
from data.fx import to_usd

logger = logging.getLogger(__name__)

OWNER_HORIZON = "12 months"


@dataclass
class TradeRecommendation:
    """Recommendation output of the multi-agent system.

    ``decision``, ``composite_score`` and ``overall_confidence`` are the owner
    layer's (the call that matters). ``owner`` and ``timing`` carry each layer
    in full; ``contributions`` lists both layers (each row has a ``layer``).
    """

    ticker: str
    decision: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL or NOT_RATED (owner layer)
    composite_score: float  # owner score, -100.0 to +100.0
    overall_confidence: float  # owner confidence, 0.0 to 100.0 (%)
    risk_rating: str  # LOW, MODERATE, HIGH, EXTREME
    risk_score: float  # 1.0 to 10.0
    suggested_entry: float
    suggested_stop_loss: float
    suggested_target_price: float
    reward_risk_ratio: float
    atr_14: float
    atr_multiplier: float
    stop_loss_method: str
    executive_summary: str
    market_regime: str  # BULL, CORRECTION, BEAR, CRISIS
    regime_weights_used: dict[str, float] = field(default_factory=dict)  # timing panel weights
    agent_assessments: list[AgentAssessment] = field(default_factory=list)
    contributions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    key_levels: list[dict[str, Any]] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)  # owner thresholds
    dispersion: float = 0.0
    conflict: bool = False
    coverage: float = 1.0  # owner-layer coverage
    freshness: dict[str, Any] | None = None
    owner: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    horizon: str = OWNER_HORIZON
    generated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Below this average daily dollar volume a position can be hard to exit (forced-selling risk).
LOW_LIQUIDITY_DOLLARS = 5_000_000.0
DEEP_NEGATIVE_MOMENTUM = -0.20  # 12-1 month return below this is a value-trap check
EARNINGS_WINDOW_DAYS = 14  # earnings this close count as event risk


def average_dollar_volume(ohlcv: pd.DataFrame, window: int = 20) -> float | None:
    """Mean of close x volume over the last completed sessions."""
    if ohlcv is None or ohlcv.empty or "volume" not in ohlcv.columns:
        return None
    bars = ohlcv.iloc[:-1] if ohlcv.attrs.get("intraday_partial") else ohlcv
    tail = bars.tail(window)
    if len(tail) < 5:
        return None
    return float((tail["close"] * tail["volume"]).mean())


def average_share_volume(ohlcv: pd.DataFrame, window: int = 20) -> float | None:
    if ohlcv is None or ohlcv.empty or "volume" not in ohlcv.columns:
        return None
    bars = ohlcv.iloc[:-1] if ohlcv.attrs.get("intraday_partial") else ohlcv
    tail = bars.tail(window)
    if len(tail) < 5:
        return None
    return float(tail["volume"].mean())


def wilder_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    if len(tr.dropna()) < period:
        return float(close.iloc[-1]) * 0.02
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def momentum_12_1(ohlcv: pd.DataFrame) -> float | None:
    """12-1 month price momentum: the return from 12 months ago to 1 month ago."""
    close = ohlcv["close"]
    if len(close) < 252:
        return None
    start, end = float(close.iloc[-252]), float(close.iloc[-21])
    return end / start - 1.0 if start > 0 else None


def display_number(value: float, places: int = 0, sign: bool = False) -> str:
    """A number as the page shows it: halves rounded away from zero (JavaScript's toFixed), never "-0"."""
    rounded = Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    text = str(abs(rounded) if rounded == 0 else rounded)
    return f"+{text}" if sign and rounded > 0 else text


class MasterDecisionAgent:
    """
    Multi-agent synthesizer and orchestrator.

    - Runs registered, enabled agents concurrently (one failing or slow agent
      is excluded instead of breaking the analysis).
    - Owner agents decide; timing agents inform (agents/scoring.py).
    - The regime-providing agent (Macro) selects the timing weight profile.
    - Outputs a decision, confidence, per-agent contributions, warnings, the
      timing panel and an ATR-based risk reference for sizing.
    """

    def __init__(
        self,
        weight_overrides: dict[str, float] | None = None,
        config: ScoringConfig | None = None,
        agent_keys: list[str] | None = None,
        agents: list[BaseAgent] | None = None,
        agent_timeout: float | None = None,
    ) -> None:
        self.weight_overrides = weight_overrides
        self.config = config or load_scoring_config()
        if agents is None:
            keys = agent_keys if agent_keys is not None else [
                k for k in get_agent_classes() if self.config.is_enabled(k)
            ]
            agents = create_agents(keys)
        self.agents = agents
        self.agent_timeout = agent_timeout if agent_timeout is not None else settings.AGENT_TIMEOUT_SECONDS

    # ------------------------------------------------------------- scoring API

    @staticmethod
    def _scale_for(agent: str) -> float:
        for key, cls in get_agent_classes().items():
            if agent in (key, cls.name):
                return cls.score_scale
        return 50.0

    @staticmethod
    def _normalize_score(raw_score: float, agent: str) -> float:
        """Normalize a raw agent score (agent given by key or display name)."""
        return normalize_score(raw_score, MasterDecisionAgent._scale_for(agent))

    @staticmethod
    def _confidence_weight(base_weight: float, confidence: float) -> float:
        return confidence_weight(base_weight, confidence)

    def _select_weights(self, regime: str) -> dict[str, float]:
        """Timing-panel weights by agent key for ``regime`` (explicit overrides win)."""
        if self.weight_overrides:
            return dict(self.weight_overrides)
        return self.config.weights_for(regime)

    def scales(self) -> dict[str, float]:
        out = {key: cls.score_scale for key, cls in get_agent_classes().items()}
        for agent in self.agents:
            out.setdefault(agent.key, agent.score_scale)
        return out

    def roles(self) -> dict[str, str]:
        """agent_key -> OWNER / TIMING: registry roles, overridden by the instances that ran."""
        out = {key: cls.role for key, cls in get_agent_classes().items()}
        for agent in self.agents:
            out[agent.key] = getattr(agent, "role", TIMING)
        return out

    def score(self, assessments: list[AgentAssessment], regime: str | None) -> LayeredResult:
        return aggregate_layers(assessments, self.config, regime, self.scales(), self.roles(), self.weight_overrides)

    # --------------------------------------------------------------- execution

    def run_agents(self, ticker: str, ohlcv: pd.DataFrame) -> list[AgentAssessment]:
        """Evaluate all agents concurrently with a per-analysis timeout."""
        pool = ThreadPoolExecutor(max_workers=max(1, len(self.agents)), thread_name_prefix="agent")
        futures = {pool.submit(agent.evaluate, ticker, ohlcv): agent for agent in self.agents}
        done, _ = wait(futures, timeout=self.agent_timeout)
        results: list[AgentAssessment] = []
        for future, agent in futures.items():
            if future not in done:
                logger.warning("%s timed out for %s", agent.name, ticker)
                results.append(agent.unavailable(f"timed out after {self.agent_timeout:.0f}s"))
                continue
            try:
                result = future.result()
                if not result.agent_key:
                    result.agent_key = agent.key
                results.append(result)
            except Exception as exc:  # noqa: BLE001 - isolate agent failures
                logger.exception("%s failed for %s", agent.name, ticker)
                results.append(agent.unavailable("agent raised an error", error=str(exc)))
        pool.shutdown(wait=False, cancel_futures=True)
        return results

    def _regime_from(self, assessments: list[AgentAssessment]) -> str | None:
        classes = get_agent_classes()
        for a in assessments:
            cls = classes.get(a.agent_key)
            if cls is not None and cls.provides_regime and a.data_quality != UNAVAILABLE:
                return a.raw_data.get("regime")
        # Legacy/mocked assessments without a key.
        for a in assessments:
            if a.raw_data.get("regime") and a.data_quality != UNAVAILABLE:
                return a.raw_data["regime"]
        return None

    def analyze(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        freshness: dict[str, Any] | None = None,
    ) -> TradeRecommendation:
        """Run all agents and synthesize the recommendation for ``ticker``."""
        symbol = ticker.upper()
        assessments = self.run_agents(symbol, ohlcv)
        return self.synthesize(symbol, ohlcv, assessments, freshness)

    def synthesize(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        assessments: list[AgentAssessment],
        freshness: dict[str, Any] | None = None,
    ) -> TradeRecommendation:
        regime = self._regime_from(assessments)
        layered = self.score(assessments, regime)
        owner, timing = layered.owner, layered.timing
        plan = self._trade_plan(layered.decision, ohlcv, assessments)
        panel = self._timing_panel(ohlcv, assessments, layered)

        warnings = list(layered.warnings)
        if freshness and freshness.get("status") == "STALE":
            warnings.append(f"Price data is stale ({freshness.get('note')}).")

        adv = to_usd(average_dollar_volume(ohlcv), symbol)  # pence and yen would read 100-150x too liquid
        if adv is not None and adv < LOW_LIQUIDITY_DOLLARS:
            warnings.append(
                f"Low trading liquidity: about ${adv / 1e6:.1f}M traded per day. "
                "Hard to exit in a sell-off (forced-selling risk); size positions accordingly."
            )
        for a in assessments:
            if a.agent_key == "shareholder" and any("reverse split" in f.lower() and "red flag" in f.lower() for f in a.key_factors):
                warnings.append("Red flag: reverse split in the last 10 years (see Shareholders & Management).")
                break

        risk_score = self._risk_score(ohlcv, layered, adv, panel)
        risk_rating = (
            "LOW" if risk_score <= 3.5 else "MODERATE" if risk_score <= 6.5 else "HIGH" if risk_score <= 8.5 else "EXTREME"
        )

        owner_dict = {
            "score": owner.composite_score,
            "decision": layered.decision,
            "confidence": owner.confidence,
            "coverage": owner.coverage,
            "dispersion": owner.dispersion,
            "conflict": owner.conflict,
            "applicable": layered.owner_applicable,
            "weights": owner.weights_used,
            "thresholds": dict(self.config.owner_thresholds),
            "max_ev_to_fcf": self.config.owner_max_ev_to_fcf,
            "capped": layered.decision != owner.decision and layered.owner_applicable,
            "not_rated_reason": next((w for w in layered.warnings if w.startswith("Not rated")), None),
            "horizon": OWNER_HORIZON,
            "warnings": list(owner.warnings),
        }
        timing_dict = {
            "score": timing.composite_score,
            "label": layered.timing_label,
            "confidence": timing.confidence,
            "coverage": timing.coverage,
            "dispersion": timing.dispersion,
            "conflict": timing.conflict,
            "regime": layered.regime,
            "weights": timing.weights_used,
            "thresholds": dict(self.config.thresholds),
            "warnings": list(timing.warnings),
            **panel,
        }

        return TradeRecommendation(
            ticker=symbol,
            decision=layered.decision,
            composite_score=owner.composite_score,
            overall_confidence=owner.confidence,
            risk_rating=risk_rating,
            risk_score=risk_score,
            suggested_entry=plan["entry"],
            suggested_stop_loss=plan["stop"],
            suggested_target_price=plan["target"],
            reward_risk_ratio=plan["rr"],
            atr_14=plan["atr"],
            atr_multiplier=plan["atr_multiplier"],
            stop_loss_method=plan["method"],
            executive_summary=self._summary(symbol, layered, panel, risk_rating),
            market_regime=regime or "UNKNOWN",
            regime_weights_used=timing.weights_used,
            agent_assessments=assessments,
            contributions=[asdict(c) for c in layered.contributions],
            warnings=warnings,
            key_levels=self._key_levels(ohlcv, assessments),
            thresholds=dict(self.config.owner_thresholds),
            dispersion=owner.dispersion,
            conflict=owner.conflict,
            coverage=owner.coverage,
            freshness=freshness,
            owner=owner_dict,
            timing=timing_dict,
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _summary(symbol: str, layered: LayeredResult, panel: dict[str, Any], risk_rating: str) -> str:
        # The same precision and rounding as the page (one decimal for the owner score and the agents' points),
        # so the summary never reads 64% next to a 65% tile.
        num = display_number
        owner, timing = layered.owner, layered.timing
        if layered.owner_applicable:
            included = [c for c in owner.contributions if c.included]
            drivers = sorted(included, key=lambda c: abs(c.contribution), reverse=True)[:2]
            driver_text = ", ".join(f"{c.agent_name.replace(' Agent', '')} ({num(c.contribution, 1, True)})" for c in drivers)
            text = (
                f"{symbol}: owner score {num(owner.composite_score, 1, True)} out of 100 "
                f"({num(owner.confidence)}% confidence, {OWNER_HORIZON} horizon). "
            )
            if driver_text:
                text += f"Drivers: {driver_text}. "
        else:
            text = f"{symbol}: not rated (no fundamentals: the value framework does not apply to funds and indexes). "
        text += (
            f"Entry timing score {num(timing.composite_score, 0, True)} in a {layered.regime.lower()} regime, "
            f"trend {panel.get('trend', 'UNKNOWN').lower()}. Setup risk: {risk_rating.lower()}."
        )
        return text

    @staticmethod
    def _risk_score(ohlcv: pd.DataFrame, layered: LayeredResult, adv: float | None, panel: dict[str, Any]) -> float:
        ann_vol = float(ohlcv["close"].pct_change().tail(30).std() * np.sqrt(252))
        volatility_risk = min(5.0, ann_vol * 10.0) if ann_vol == ann_vol else 2.5
        owner = layered.owner
        disagreement_risk = min(2.5, owner.dispersion / 20.0) if layered.owner_applicable else 0.0
        coverage_risk = (1.0 - owner.coverage) * 2.0 if layered.owner_applicable else 2.0
        liquidity_risk = 0.0 if adv is None else 3.0 if adv < 1e6 else 1.5 if adv < LOW_LIQUIDITY_DOLLARS else 0.0
        event_risk = 1.0 if (panel.get("event_risk") or {}).get("elevated") else 0.0
        total = 2.5 + volatility_risk + disagreement_risk + coverage_risk + liquidity_risk + event_risk
        return round(min(10.0, max(1.0, total)), 1)

    @staticmethod
    def _key_levels(ohlcv: pd.DataFrame, assessments: list[AgentAssessment]) -> list[dict[str, Any]]:
        price = float(ohlcv["close"].iloc[-1])
        levels: list[dict[str, Any]] = []
        for a in assessments:
            if a.data_quality == UNAVAILABLE:
                continue
            for lvl in a.raw_data.get("levels", []) or []:
                try:
                    p = float(lvl["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(p) or p <= 0:
                    continue
                levels.append({
                    "label": str(lvl.get("label", "")),
                    "price": round(p, 2),
                    "kind": lvl.get("kind", "pivot"),
                    "source": a.agent_name,
                    "distance_pct": round((p / price - 1.0) * 100.0, 2),
                })
        return sorted(levels, key=lambda lvl: lvl["price"], reverse=True)

    @staticmethod
    def _timing_panel(ohlcv: pd.DataFrame, assessments: list[AgentAssessment], layered: LayeredResult) -> dict[str, Any]:
        """Trend, momentum value-trap check, pullback setup, event risk and VIX, with plain-language notes."""
        by_key = {a.agent_key: a for a in assessments if a.data_quality != UNAVAILABLE}
        close = ohlcv["close"]
        price = float(close.iloc[-1])

        trend = "UNKNOWN"
        mom = by_key.get("momentum")
        sma_50 = sma_200 = None
        if mom is not None:
            sma_50 = mom.raw_data.get("sma_50")
            sma_200 = mom.raw_data.get("sma_200")
        if sma_200 is None and len(close) >= 200:
            sma_200 = float(close.tail(200).mean())
        if sma_50 is None and len(close) >= 50:
            sma_50 = float(close.tail(50).mean())
        # Trend = the 50/200-day structure with price on the right side of the 200-day SMA, so a pullback
        # below the 50-day mean inside a rising structure still reads UP.
        if sma_200 is not None and sma_50 is not None:
            if price > sma_200 and sma_50 > sma_200:
                trend = "UP"
            elif price < sma_200 and sma_50 < sma_200:
                trend = "DOWN"
            else:
                trend = "MIXED"
        elif sma_50 is not None:
            trend = "UP" if price > sma_50 else "DOWN"

        mom_12_1 = momentum_12_1(ohlcv)
        pullback = by_key.get("pullback")
        setup = pullback.raw_data.get("setup") if pullback is not None else None

        options = by_key.get("options")
        event_risk: dict[str, Any] = {"elevated": False}
        if options is not None:
            iv_rank = options.raw_data.get("iv_rank") if options.raw_data.get("iv_rank_source") == "history" else None
            iv_hv = options.raw_data.get("iv_hv_ratio")
            earnings = options.raw_data.get("earnings_date")
            days_to_earnings = None
            if earnings:
                try:
                    days_to_earnings = int((pd.Timestamp(earnings) - pd.Timestamp(ohlcv.index[-1])).days)
                except (ValueError, TypeError):
                    days_to_earnings = None
            elevated = bool((iv_rank is not None and iv_rank >= 75) or (iv_hv is not None and iv_hv >= 1.4)
                            or (days_to_earnings is not None and 0 <= days_to_earnings <= EARNINGS_WINDOW_DAYS))
            event_risk = {"elevated": elevated, "iv_rank": iv_rank, "iv_hv_ratio": iv_hv,
                          "earnings_date": earnings, "days_to_earnings": days_to_earnings}

        macro = by_key.get("macro")
        vix: dict[str, Any] = {}
        if macro is not None:
            level = macro.raw_data.get("vix_level")
            level = float(level) if isinstance(level, (int, float)) and level == level else None
            vix = {"level": level, "extreme": bool(level is not None and level >= 40.0),
                   "stance": macro.raw_data.get("vix_stance")}

        notes = ["Entry timing does not vote: the owner score is the verdict. Use the timing score for when to enter and how much."]
        if setup == "pullback_in_uptrend":
            notes.append("Pullback inside an uptrend: a reasonable entry point (in-sample evidence only, see the Pullback agent).")
        elif setup == "extended":
            notes.append("Price is extended above its 20-day mean: consider half the size now and half on a pullback.")
        if trend == "DOWN":
            notes.append(
                "Trend is down (below the 200-day SMA). The value study found a momentum filter on value picks cost "
                "0.9% a year, so this is not a reason to skip a company whose owner score is at or above the entry level, but stage the entry."
            )
        if mom_12_1 is not None and mom_12_1 <= DEEP_NEGATIVE_MOMENTUM:
            notes.append(
                f"12-1 month momentum {mom_12_1:+.0%}: value-trap check. Confirm the owner thesis (share count, net dilution, "
                "FCF) before adding. In the 2011-2026 S&P 500 test, owner scores at or above the entry level in this bucket had the weakest next month "
                "but the best next 12 months (+22.7%), so an intact thesis argues for a staged entry, not for skipping it."
            )
        from_high = None
        if len(ohlcv) >= 252:
            high_252 = float(ohlcv["high"].tail(252).max())
            from_high = price / high_252 - 1.0 if high_252 > 0 else None
        if layered.owner_applicable and layered.decision in ("BUY", "STRONG_BUY") and from_high is not None and from_high <= -0.30:
            notes.append(
                f"Price is {abs(from_high):.0%} below its 52-week high and an owner score at or above the entry level: in the 2011-2026 test this was the "
                "best bucket for those scores (+20% over the next 12 months). Being far below the high is not a reason to wait."
            )
        if layered.regime in ("BEAR", "CRISIS") and layered.owner_applicable and layered.decision in ("BUY", "STRONG_BUY"):
            notes.append(
                f"{layered.regime} regime: in the 2011-2026 S&P 500 test, stress months were followed by better 12-month returns "
                "for the whole market (about 7 points more than calm months, not significant, six episodes), and high owner scores did no "
                "better than other stocks then. A reason to stage the entry, not to wait for calm or to buy more."
            )
        if event_risk.get("elevated"):
            when = event_risk.get("days_to_earnings")
            soon = when is not None and 0 <= when <= EARNINGS_WINDOW_DAYS
            notes.append(
                (f"Earnings in {when} days" if soon else "Options price a larger move than usual")
                + ": expect a gap; size for it or wait until after the event."
            )
        if vix.get("extreme"):
            notes.append(
                "VIX at or above 40: historically among the best 12-month entry points (SPY +22% on average after the first "
                "close above 40, 1993-2026). Deploy in 2-3 tranches over 2-4 weeks: the market fell a further 24-32% after "
                "the signal in 2008 and 2020."
            )
        if not layered.owner_applicable:
            notes.append("No owner rating (fund, index, missing statements, or a sector outside the framework): the entry timing signals are shown for information only.")

        return {
            "trend": trend,
            "sma_50": round(sma_50, 2) if sma_50 is not None else None,
            "sma_200": round(sma_200, 2) if sma_200 is not None else None,
            "momentum_12_1": round(mom_12_1, 4) if mom_12_1 is not None else None,
            "value_trap_check": bool(mom_12_1 is not None and mom_12_1 <= DEEP_NEGATIVE_MOMENTUM),
            "setup": setup,
            "event_risk": event_risk,
            "vix": vix,
            "notes": notes,
        }

    @staticmethod
    def _trade_plan(decision: str, ohlcv: pd.DataFrame, assessments: list[AgentAssessment]) -> dict[str, Any]:
        """Long-only risk reference: ATR stop for sizing (not an automatic exit) and a 2R target for BUYs."""
        latest = float(ohlcv["close"].iloc[-1])
        atr = wilder_atr(ohlcv)

        options = next((a for a in assessments if a.agent_key == "options" and a.data_quality != UNAVAILABLE), None)
        iv_rank = iv_hv = None
        if options is not None:
            # Until enough daily IV snapshots exist, iv_rank is an HV-based proxy: do not treat it as a true rank.
            if options.raw_data.get("iv_rank_source") == "history":
                iv_rank = options.raw_data.get("iv_rank")
            iv_hv = options.raw_data.get("iv_hv_ratio")
        iv_hv = float(iv_hv) if iv_hv is not None else 1.0
        iv_rank = float(iv_rank) if iv_rank is not None else 50.0

        # Widen the reference when options price in bigger-than-usual moves (events, earnings).
        expansion = max(1.2, min(1.8, iv_hv)) if (iv_rank >= 75.0 or iv_hv >= 1.4) else 1.0
        mult = round(2.0 * expansion, 2)

        swing_low = float(ohlcv["low"].tail(20).min())
        stop = round(min(latest - mult * atr, swing_low * 0.99), 2)
        risk = abs(latest - stop)
        is_buy = decision in ("STRONG_BUY", "BUY")
        reward_mult = 2.0 if is_buy else 1.5
        target = round(latest + reward_mult * risk, 2)
        method = (
            f"Risk reference for sizing: {mult:.1f}x ATR14 or below the 20D swing low ${swing_low:.2f} "
            "(not an automatic exit; the owner score is re-rated yearly)"
            if is_buy
            else f"Reference levels only (no new money at this score): {mult:.1f}x ATR14 buffer"
        )

        return {
            "entry": round(latest, 2),
            "stop": stop,
            "target": target,
            "rr": round(abs(target - latest) / max(risk, 0.01), 2),
            "atr": round(atr, 2),
            "atr_multiplier": mult,
            "method": method,
        }
