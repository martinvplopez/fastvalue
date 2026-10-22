"""Macro Regime Detection Agent.

Classifies the market environment (BULL, CORRECTION, BEAR, CRISIS) from broad
market trend, VIX level and term structure, the Treasury curve (FRED), breadth,
credit spreads and 10Y yield momentum. The regime selects the weight profile
the master agent uses for every other agent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config.settings as settings
from agents.base import LIVE, PARTIAL, AgentAssessment, BaseAgent
from agents.registry import register_agent
from data.macro_data import MAX_REGIME_SCORE, MacroRegimeData, fetch_macro_data


def _fmt(value: float | None, pattern: str, missing: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return missing
    return pattern.format(value)


@register_agent
class MacroRegimeAgent(BaseAgent):
    """Expert agent evaluating macro market conditions and regime state."""

    key = "macro"
    name = "Macro Regime Agent"
    description = (
        "Market-wide backdrop: SPY trend, VIX, VIX term structure, Treasury curve, breadth, credit "
        "spreads and 10Y yield momentum. Picks the regime weight profile. With the contrarian VIX "
        "stance (default), high fear raises the score: long-run VIX average 20, above 40 is extreme fear."
    )
    category = "Macro"
    score_scale = 50.0
    default_weight = 0.18
    # Snapshot is computed as of the last bar of the OHLCV passed in, so it replays without look-ahead.
    point_in_time = True
    provides_regime = True

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        """Evaluate macro conditions as of the last bar in ``ohlcv`` (ticker is not used)."""
        asof = ohlcv.index[-1] if ohlcv is not None and len(ohlcv) else None
        macro = fetch_macro_data(asof)
        if macro.data_quality == "ESTIMATED":
            return self.unavailable(
                "SPY and VIX data could not be loaded",
                raw_data={"regime": None, "missing": macro.missing},
            )
        return self._build_assessment(macro)

    def evaluate_with_data(self, macro: MacroRegimeData) -> AgentAssessment:
        """Build an assessment from pre-fetched macro data (tests, batch jobs)."""
        return self._build_assessment(macro)

    def _build_assessment(self, macro: MacroRegimeData) -> AgentAssessment:
        contrarian = settings.VIX_STANCE == "contrarian" and macro.opportunity_score is not None
        raw_points = macro.opportunity_score if contrarian else macro.regime_score
        components = macro.opportunity_components if contrarian else macro.components
        health_score = float(np.clip((raw_points / MAX_REGIME_SCORE) * 100.0, -100.0, 100.0))

        base_confidence = {"FULL": 80.0, "PARTIAL": 60.0}.get(macro.data_quality, 40.0)
        signal_strength = min(abs(raw_points) / MAX_REGIME_SCORE, 1.0)
        confidence = min(95.0, base_confidence + signal_strength * 15.0)

        factors: list[str] = []
        if macro.spy_price == macro.spy_price:  # not NaN
            if macro.spy_price > macro.spy_sma_200:
                if macro.spy_price > macro.spy_sma_50:
                    factors.append(
                        f"SPY in full uptrend ({macro.spy_price:.0f} above 50 SMA {macro.spy_sma_50:.0f} "
                        f"and 200 SMA {macro.spy_sma_200:.0f})"
                    )
                else:
                    factors.append(f"SPY above 200 SMA but below 50 SMA: short-term weakness ({macro.spy_price:.0f})")
            else:
                factors.append(f"SPY below 200-day SMA ({macro.spy_price:.0f} < {macro.spy_sma_200:.0f}): bear market signal")

        if macro.vix_level == macro.vix_level and contrarian:
            pct = f"{macro.vix_percentile:.0f}th percentile of 1Y"
            if macro.vix_level >= 40:
                factors.append(f"VIX {macro.vix_level:.1f} ({pct}): extreme fear, historically one of the best times to buy")
            elif macro.vix_level >= 30:
                factors.append(f"VIX {macro.vix_level:.1f} ({pct}): high fear, bargains start to appear")
            elif macro.vix_level >= 20:
                factors.append(f"VIX {macro.vix_level:.1f} ({pct}): above the long-run average of 20")
            else:
                factors.append(f"VIX {macro.vix_level:.1f} ({pct}): below the long-run average of 20, no fear discount")
        elif macro.vix_level == macro.vix_level:
            pct = f"{macro.vix_percentile:.0f}th percentile of 1Y"
            if macro.vix_level < 18:
                factors.append(f"VIX low at {macro.vix_level:.1f} ({pct}): calm, possibly complacent")
            elif macro.vix_level < 25:
                factors.append(f"VIX elevated at {macro.vix_level:.1f} ({pct}): heightened uncertainty")
            elif macro.vix_level < 35:
                factors.append(f"VIX high at {macro.vix_level:.1f} ({pct}): significant fear")
            else:
                factors.append(f"VIX extreme at {macro.vix_level:.1f} ({pct}): panic conditions")

        if macro.vix_term_ratio is not None:
            if macro.vix_term_ratio > 1.0:
                factors.append(f"VIX term structure in backwardation (VIX/VIX3M {macro.vix_term_ratio:.2f}): near-term stress")
            else:
                factors.append(f"VIX term structure in contango (VIX/VIX3M {macro.vix_term_ratio:.2f})")

        if macro.yield_curve_status == "INVERTED":
            factors.append(f"Yield curve inverted (10Y-2Y {macro.yield_spread_10y_2y:+.2f} pp): recession warning")
        elif macro.yield_curve_status == "FLAT":
            factors.append(f"Yield curve flat (10Y-2Y {macro.yield_spread_10y_2y:+.2f} pp): late cycle")
        elif macro.yield_curve_status == "NORMAL":
            factors.append(f"Yield curve normal (10Y-2Y {macro.yield_spread_10y_2y:+.2f} pp)")
        if macro.curve_uninverted:
            factors.append("Curve recently un-inverted after a deep inversion, which has often preceded recessions")

        if macro.breadth_trend == "IMPROVING":
            factors.append("Market breadth improving (equal weight outperforming): broad participation")
        elif macro.breadth_trend == "DETERIORATING":
            factors.append("Market breadth deteriorating (mega caps carrying the index): narrow rally")

        if macro.credit_trend == "RISK_OFF":
            factors.append("Credit spreads widening (HYG underperforming LQD): risk-off")
        elif macro.credit_trend == "RISK_ON":
            factors.append("Credit spreads tightening (HYG outperforming LQD): risk-on")

        if macro.ten_year_yield is not None and macro.ten_year_change_60d is not None:
            factors.append(f"10Y Treasury {macro.ten_year_yield:.2f}% ({macro.ten_year_change_60d:+.2f} pp over 60 days)")

        if macro.missing:
            factors.append(f"Missing inputs skipped: {', '.join(macro.missing)}")

        stance = "buying conditions (contrarian VIX)" if contrarian else "market health"
        summary = (
            f"Market regime is {macro.regime.lower()}; {stance} {health_score:+.0f}/100. "
            f"VIX {_fmt(macro.vix_level, '{:.1f}')}, yield curve {macro.yield_curve_status.lower()}, "
            f"breadth {macro.breadth_trend.lower()}, credit {macro.credit_trend.lower().replace('_', '-')}."
        )

        metrics_table = [
            {"metric": "Market Regime", "value": macro.regime.capitalize()},
            {"metric": "Regime Score (classification)", "value": f"{macro.regime_score:+.1f}"},
            {"metric": "Buying-conditions score (contrarian VIX)", "value": "n/a" if macro.opportunity_score is None else f"{macro.opportunity_score:+.1f}"},
            {"metric": "VIX stance", "value": "Contrarian: high VIX is an opportunity" if contrarian else "Risk: high VIX is a warning"},
            {"metric": "SPY Price", "value": _fmt(macro.spy_price, "${:.2f}")},
            {"metric": "SPY 50-Day SMA", "value": _fmt(macro.spy_sma_50, "${:.2f}")},
            {"metric": "SPY 200-Day SMA", "value": _fmt(macro.spy_sma_200, "${:.2f}")},
            {"metric": "VIX Level", "value": _fmt(macro.vix_level, "{:.2f}")},
            {"metric": "VIX 1Y Percentile", "value": _fmt(macro.vix_percentile, "{:.1f}%")},
            {"metric": "VIX / VIX3M", "value": _fmt(macro.vix_term_ratio, "{:.2f}")},
            {"metric": "10Y-2Y Yield Spread (FRED)", "value": _fmt(macro.yield_spread_10y_2y, "{:+.2f} pp")},
            {"metric": "10Y-3M Yield Spread (FRED)", "value": _fmt(macro.yield_spread_10y_3m, "{:+.2f} pp")},
            {"metric": "Yield Curve Status", "value": macro.yield_curve_status},
            {"metric": "10Y Treasury Yield", "value": _fmt(macro.ten_year_yield, "{:.2f}%")},
            {"metric": "Breadth Ratio (RSP/SPY)", "value": _fmt(macro.breadth_ratio, "{:.4f}")},
            {"metric": "Breadth Trend", "value": macro.breadth_trend},
            {"metric": "Credit Ratio (HYG/LQD)", "value": _fmt(macro.credit_ratio, "{:.4f}")},
            {"metric": "Credit Trend", "value": macro.credit_trend},
            {"metric": "Data Quality", "value": macro.data_quality},
        ]

        score_breakdown = [
            {"factor": f"{c['component']}: {c['detail']}", "points": round(c["points"] / MAX_REGIME_SCORE * 100.0, 1)}
            for c in components
        ]

        return self.assessment(
            score=health_score,
            confidence=confidence,
            summary=summary,
            key_factors=factors,
            data_quality=LIVE if macro.data_quality == "FULL" else PARTIAL,
            data_as_of=macro.as_of,
            raw_data={
                "source_url": "https://fred.stlouisfed.org/series/T10Y2Y",
                "alt_source_url": "https://www.cboe.com/tradable_products/vix/term_structure/",
                "data_source_label": "FRED (Federal Reserve) / Yahoo Finance",
                "links": [
                    {"label": "FRED 10Y-2Y spread", "url": "https://fred.stlouisfed.org/series/T10Y2Y"},
                    {"label": "CBOE VIX term structure", "url": "https://www.cboe.com/tradable_products/vix/term_structure/"},
                ],
                "regime": macro.regime,
                "regime_score": macro.regime_score,
                "opportunity_score": macro.opportunity_score,
                "vix_stance": "contrarian" if contrarian else "risk",
                "vix_level": macro.vix_level,
                "vix_percentile": macro.vix_percentile,
                "vix_term_ratio": macro.vix_term_ratio,
                "yield_spread_10y_2y": macro.yield_spread_10y_2y,
                "yield_spread_10y_3m": macro.yield_spread_10y_3m,
                "yield_curve_status": macro.yield_curve_status,
                "ten_year_yield": macro.ten_year_yield,
                "spy_price": macro.spy_price,
                "spy_sma_50": macro.spy_sma_50,
                "spy_sma_200": macro.spy_sma_200,
                "breadth_ratio": macro.breadth_ratio,
                "breadth_trend": macro.breadth_trend,
                "credit_ratio": macro.credit_ratio,
                "credit_trend": macro.credit_trend,
                "data_quality": macro.data_quality,
                "missing": macro.missing,
                "metrics_table": metrics_table,
                "score_breakdown": score_breakdown,
            },
        )
