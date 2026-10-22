"""Pullback / entry-timing agent (timing panel).

Where it comes from: the composite score backtest found the trend composite
contrarian as an entry timer (STRONG_BUY readings were followed by worse
20-60 day returns), while a "pullback inside an uptrend" reading (price below
its 20-day mean in ATR units, RSI(3) washed out, only above the 200-day SMA)
had positive IC in 10 of 12 ticker/horizon cases in-sample.

Validated out of sample in September 2026 (``core/timing_validation.py``,
32 tickers the idea was not tuned on, weekly, 10 years): IC +0.018 at 20
sessions (t 1.7) and +0.019 at 60 sessions (t 2.4), positive on 66% of
tickers, while the momentum and volume agents were negative at every horizon.
On the S&P 500 point-in-time panel (``core/all_agents_study.py``) owner BUYs
with this reading returned 20.7% annualized over the next month against 8.8%
when extended. Small but consistent, so it carries 15% of the timing panel.
It never votes on the owner decision.

Readings (``raw_data["setup"]``):
- ``trend_broken``: below the 200-day SMA. No pullback setup; score 0.
- ``pullback_in_uptrend``: 1 to 3 ATR below the 20-day mean in an uptrend.
- ``extended``: more than 2 ATR above the 20-day mean or RSI(3) above 90.
- ``neutral``: everything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.base import TIMING, AgentAssessment, BaseAgent, iso_date
from agents.registry import register_agent

MIN_BARS = 210
PULLBACK_MIN_ATR = 1.0
PULLBACK_MAX_ATR = 3.0
EXTENDED_ATR = 2.0


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    value = float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    return value if value > 0 else float(close.iloc[-1]) * 0.02


@register_agent
class PullbackAgent(BaseAgent):
    """Pullback inside an uptrend vs extended: where in the swing the price is."""

    key = "pullback"
    name = "Pullback & Entry Timing Agent"
    description = (
        "Entry timing inside an uptrend: how far price sits below its 20-day mean in ATR units and whether "
        "RSI(3) is washed out, only above the 200-day SMA. Validated out of sample (IC +0.02 at 20-60 sessions, "
        "t 2.4; owner BUYs on a pullback made 20.7% annualized the next month vs 8.8% when extended)."
    )
    category = "Technical"
    score_scale = 40.0
    default_weight = 0.15
    point_in_time = True
    role = TIMING

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        if ohlcv is None or len(ohlcv) < MIN_BARS:
            return self.unavailable(f"Need at least {MIN_BARS} daily bars for the pullback reading")
        close = ohlcv["close"]
        latest = float(close.iloc[-1])
        sma20 = float(close.tail(20).mean())
        sma200 = float(close.tail(200).mean())
        a = atr(ohlcv)
        depth = (sma20 - latest) / a  # positive = below the 20-day mean
        rsi3 = float(wilder_rsi(close, 3).iloc[-1])
        uptrend = latest > sma200

        breakdown: list[dict] = []

        def add(factor: str, points: float) -> None:
            breakdown.append({"factor": factor, "points": float(points)})

        if not uptrend:
            setup = "trend_broken"
            add(f"Below the 200-day SMA ({sma200:.2f}): no pullback setup, trend broken", 0.0)
        elif PULLBACK_MIN_ATR <= depth <= PULLBACK_MAX_ATR:
            setup = "pullback_in_uptrend"
            add(f"Uptrend pullback: {depth:.1f} ATR below the 20-day mean ({sma20:.2f})", 25.0)
            if rsi3 < 20:
                add(f"RSI(3) washed out at {rsi3:.0f}", 15.0)
            elif rsi3 < 35:
                add(f"RSI(3) low at {rsi3:.0f}", 6.0)
        elif depth > PULLBACK_MAX_ATR:
            setup = "neutral"
            add(f"{depth:.1f} ATR below the 20-day mean: deeper than a pullback, wait for stabilization", 5.0)
        elif depth <= -EXTENDED_ATR or rsi3 > 90:
            setup = "extended"
            add(f"Extended: {-depth:.1f} ATR above the 20-day mean, RSI(3) {rsi3:.0f}", -25.0)
        else:
            setup = "neutral"
            add(f"Near the 20-day mean ({depth:+.1f} ATR), RSI(3) {rsi3:.0f}: no edge either way", 0.0)

        score = float(np.clip(sum(b["points"] for b in breakdown), -100.0, 100.0))
        confidence = 55.0 if setup in ("pullback_in_uptrend", "extended") else 35.0
        labels = {
            "trend_broken": "trend broken (below the 200-day SMA)",
            "pullback_in_uptrend": "pullback inside an uptrend",
            "extended": "extended above the 20-day mean",
            "neutral": "no setup",
        }
        summary = (
            f"Entry timing: {labels[setup]}. Price {depth:+.1f} ATR from the 20-day mean, RSI(3) {rsi3:.0f}, "
            f"{'above' if uptrend else 'below'} the 200-day SMA."
        )
        return self.assessment(
            score=score,
            confidence=confidence,
            summary=summary,
            key_factors=[b["factor"] for b in breakdown],
            data_as_of=iso_date(ohlcv.index[-1]),
            raw_data={
                "setup": setup,
                "depth_atr": round(depth, 2),
                "rsi_3": round(rsi3, 1),
                "sma_20": round(sma20, 2),
                "sma_200": round(sma200, 2),
                "atr_14": round(a, 4),
                "uptrend": uptrend,
                "data_source_label": "Computed from daily OHLCV (validated out of sample, September 2026)",
                "metrics_table": [
                    {"metric": "Setup", "value": labels[setup]},
                    {"metric": "Distance from 20-day mean (ATR)", "value": f"{depth:+.2f}"},
                    {"metric": "RSI(3)", "value": f"{rsi3:.1f}"},
                    {"metric": "SMA 20 / 200", "value": f"{sma20:.2f} / {sma200:.2f}"},
                    {"metric": "ATR 14", "value": f"{a:.2f}"},
                ],
                "score_breakdown": breakdown,
            },
        )
