"""Technical and Momentum Domain Agent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.base import AgentAssessment, BaseAgent, iso_date
from agents.registry import register_agent

MIN_BARS = 60
LOOKBACK_BARS = 300  # Everything below needs at most ~260 bars.


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def average_directional_index(ohlcv: pd.DataFrame, period: int = 14) -> tuple[float, float, float]:
    """Return (ADX, +DI, -DI) using Wilder smoothing."""
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=ohlcv.index).ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=ohlcv.index).ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return float(adx.iloc[-1] or 0.0), float(plus_di.iloc[-1] or 0.0), float(minus_di.iloc[-1] or 0.0)


@register_agent
class MomentumAgent(BaseAgent):
    """
    Expert agent evaluating trend alignment (SMA 50/200, EMA 9/21, weekly trend),
    trend strength (ADX), momentum (ROC, MACD histogram), RSI conditions and divergences.
    """

    key = "momentum"
    name = "Momentum & Technicals Agent"
    description = (
        "Trend and momentum: SMA 50/200 and EMA 9/21 alignment, weekly trend, ADX strength, "
        "ROC, MACD histogram shifts, RSI levels and divergences, distance from the 52-week high."
    )
    category = "Technical"
    score_scale = 55.0
    default_weight = 0.25
    point_in_time = True

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        if ohlcv is None or len(ohlcv) < MIN_BARS:
            return self.unavailable(f"Need at least {MIN_BARS} daily bars for momentum analysis")

        df = ohlcv.tail(LOOKBACK_BARS)
        close = df["close"]
        n = len(close)
        latest = float(close.iloc[-1])

        sma_20 = float(close.tail(20).mean())
        sma_50 = float(close.tail(50).mean())
        has_200 = n >= 200
        sma_200 = float(close.tail(200).mean()) if has_200 else float("nan")
        ema_9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        ema_21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

        roc_20 = latest / float(close.iloc[-21]) - 1.0
        roc_60 = latest / float(close.iloc[-61]) - 1.0 if n > 60 else float("nan")

        rsi = wilder_rsi(close)
        rsi_14 = float(rsi.iloc[-1])
        rsi_window = rsi.tail(14)
        rsi_range = float(rsi_window.max() - rsi_window.min())
        stoch_rsi = float((rsi_14 - rsi_window.min()) / rsi_range) if rsi_range > 1e-9 else 0.5

        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        hist = macd - macd.ewm(span=9, adjust=False).mean()
        hist_now = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-4])
        # Sign change within the last 3 bars.
        hist_crossed_up = hist_now > 0 and hist_prev <= 0
        hist_crossed_down = hist_now < 0 and hist_prev >= 0

        adx, plus_di, minus_di = average_directional_index(df)

        high_252 = float(df["high"].tail(252).max())
        from_high = latest / high_252 - 1.0 if high_252 > 0 else 0.0

        breakdown: list[dict[str, float | str]] = []
        factors: list[str] = []

        def add(points: float, factor: str) -> None:
            breakdown.append({"factor": factor, "points": points})
            factors.append(factor)

        # 1. Trend alignment
        if has_200 and latest > sma_50 > sma_200:
            add(25.0, f"Uptrend alignment: price > 50 SMA ({sma_50:.2f}) > 200 SMA ({sma_200:.2f})")
        elif has_200 and latest < sma_50 < sma_200:
            add(-25.0, f"Downtrend alignment: price < 50 SMA ({sma_50:.2f}) < 200 SMA ({sma_200:.2f})")
        elif latest > sma_50:
            add(10.0, f"Price above the 50-day SMA ({sma_50:.2f})")
        else:
            add(-10.0, f"Price below the 50-day SMA ({sma_50:.2f})")

        add(8.0 if ema_9 > ema_21 else -8.0,
            f"EMA 9 {'above' if ema_9 > ema_21 else 'below'} EMA 21 (short-term trend {'up' if ema_9 > ema_21 else 'down'})")

        # 2. Weekly timeframe alignment
        weekly = close.resample("W-FRI").last().dropna()
        if len(weekly) >= 40:
            w_sma10 = float(weekly.tail(10).mean())
            w_sma40 = float(weekly.tail(40).mean())
            if latest > w_sma10 > w_sma40:
                add(10.0, "Weekly trend up: price > 10-week > 40-week SMA")
            elif latest < w_sma10 < w_sma40:
                add(-10.0, "Weekly trend down: price < 10-week < 40-week SMA")

        # 3. Trend strength
        if adx >= 25:
            direction = 1.0 if plus_di > minus_di else -1.0
            add(10.0 * direction, f"Strong trend (ADX {adx:.0f}), {'+DI' if direction > 0 else '-DI'} leading")
        elif adx < 18:
            factors.append(f"No clear trend (ADX {adx:.0f}): signals less reliable")

        # 4. Rate of change
        if roc_20 > 0.05:
            add(15.0, f"Strong 20-day momentum ({roc_20:+.1%})")
        elif roc_20 < -0.05:
            add(-15.0, f"Negative 20-day momentum ({roc_20:+.1%})")
        if roc_60 == roc_60:
            if roc_60 > 0.10:
                add(8.0, f"Positive 60-day momentum ({roc_60:+.1%})")
            elif roc_60 < -0.10:
                add(-8.0, f"Negative 60-day momentum ({roc_60:+.1%})")

        # 5. MACD histogram
        if hist_crossed_up:
            add(10.0, "MACD histogram just turned positive (bullish momentum shift)")
        elif hist_crossed_down:
            add(-10.0, "MACD histogram just turned negative (bearish momentum shift)")
        elif hist_now > 0 and hist_now > hist_prev:
            add(6.0, "MACD histogram positive and rising")
        elif hist_now < 0 and hist_now < hist_prev:
            add(-6.0, "MACD histogram negative and falling")

        # 6. RSI extremes
        if rsi_14 < 30:
            pts = 14.0 if stoch_rsi > 0.2 else 8.0
            add(pts, f"RSI oversold ({rsi_14:.1f}){', Stoch RSI turning up' if stoch_rsi > 0.2 else ''}: bounce potential")
        elif rsi_14 < 35:
            add(6.0, f"RSI near oversold ({rsi_14:.1f})")
        elif rsi_14 > 75:
            add(-12.0, f"RSI overbought ({rsi_14:.1f}): extended")
        elif rsi_14 > 70:
            add(-6.0, f"RSI elevated ({rsi_14:.1f})")
        else:
            factors.append(f"RSI neutral ({rsi_14:.1f})")

        # 7. RSI divergence over the last 20 bars vs the 20 before
        if n >= 45:
            recent, prior = slice(-20, None), slice(-40, -20)
            price_recent_low, price_prior_low = float(close.iloc[recent].min()), float(close.iloc[prior].min())
            rsi_recent_low, rsi_prior_low = float(rsi.iloc[recent].min()), float(rsi.iloc[prior].min())
            price_recent_high, price_prior_high = float(close.iloc[recent].max()), float(close.iloc[prior].max())
            rsi_recent_high, rsi_prior_high = float(rsi.iloc[recent].max()), float(rsi.iloc[prior].max())
            if price_recent_low < price_prior_low and rsi_recent_low > rsi_prior_low + 3:
                add(12.0, "Bullish RSI divergence: lower price low with a higher RSI low")
            elif price_recent_high > price_prior_high and rsi_recent_high < rsi_prior_high - 3:
                add(-12.0, "Bearish RSI divergence: higher price high with a lower RSI high")

        # 8. 52-week high context
        if from_high > -0.03 and latest > sma_50:
            add(5.0, f"Trading within 3% of the 52-week high ({high_252:.2f}): breakout zone")
        elif from_high < -0.30:
            add(-5.0, f"{abs(from_high):.0%} below the 52-week high: long-term damage")

        raw = float(sum(float(b["points"]) for b in breakdown))
        score = float(np.clip(raw, -100.0, 100.0))

        agreeing = sum(abs(float(b["points"])) for b in breakdown if np.sign(float(b["points"])) == np.sign(score))
        total = sum(abs(float(b["points"])) for b in breakdown) or 1.0
        confidence = 40.0 + 35.0 * (agreeing / total) + min(abs(score), 60.0) * 0.25
        if not has_200:
            confidence -= 10.0
        if adx < 18:
            confidence -= 5.0
        confidence = float(np.clip(confidence, 20.0, 92.0))

        swing_low = float(df["low"].tail(20).min())
        swing_high = float(df["high"].tail(20).max())
        levels = [
            {"label": "SMA 50", "price": round(sma_50, 2), "kind": "pivot"},
            {"label": "20D swing low", "price": round(swing_low, 2), "kind": "support"},
            {"label": "20D swing high", "price": round(swing_high, 2), "kind": "resistance"},
        ]
        if has_200:
            levels.append({"label": "SMA 200", "price": round(sma_200, 2), "kind": "pivot"})

        summary = (
            f"Technicals score {score:+.0f}/100: 20-day return {roc_20:+.1%}, RSI {rsi_14:.1f}, "
            f"ADX {adx:.0f}, price {abs(from_high):.0%} {'below' if from_high < 0 else 'at'} the 52-week high."
        )

        def fmt(x: float, pattern: str) -> str:
            return pattern.format(x) if x == x else "n/a"

        metrics_table = [
            {"metric": "Latest Close", "value": f"${latest:.2f}"},
            {"metric": "SMA 20 / 50 / 200", "value": f"{sma_20:.2f} / {sma_50:.2f} / {fmt(sma_200, '{:.2f}')}"},
            {"metric": "EMA 9 / 21", "value": f"{ema_9:.2f} / {ema_21:.2f}"},
            {"metric": "RSI 14 (Wilder)", "value": f"{rsi_14:.1f}"},
            {"metric": "Stochastic RSI", "value": f"{stoch_rsi:.2f}"},
            {"metric": "ADX 14 (+DI / -DI)", "value": f"{adx:.1f} ({plus_di:.1f} / {minus_di:.1f})"},
            {"metric": "MACD Histogram", "value": f"{hist_now:+.3f}"},
            {"metric": "ROC 20d / 60d", "value": f"{roc_20:+.1%} / {fmt(roc_60, '{:+.1%}')}"},
            {"metric": "From 52-Week High", "value": f"{from_high:+.1%}"},
        ]

        return self.assessment(
            score=score,
            confidence=confidence,
            summary=summary,
            key_factors=factors,
            data_as_of=iso_date(ohlcv.index[-1]),
            raw_data={
                "source_url": f"https://finviz.com/quote.ashx?t={ticker}",
                "alt_source_url": f"https://www.tradingview.com/symbols/{ticker}/",
                "data_source_label": "Computed from daily OHLCV",
                "links": [
                    {"label": "Finviz", "url": f"https://finviz.com/quote.ashx?t={ticker}"},
                    {"label": "TradingView chart", "url": f"https://www.tradingview.com/symbols/{ticker}/"},
                ],
                "rsi_14": round(rsi_14, 2),
                "roc_20d": round(roc_20, 4),
                "sma_50": round(sma_50, 2),
                "sma_200": round(sma_200, 2) if has_200 else None,
                "adx": round(adx, 1),
                "macd_hist": round(hist_now, 4),
                "levels": levels,
                "metrics_table": metrics_table,
                "score_breakdown": breakdown,
            },
        )
