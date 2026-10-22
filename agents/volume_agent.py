"""Volume and Accumulation Agent.

Reads buying and selling pressure from price and volume alone:

- On-Balance Volume (OBV) trend and OBV/price divergence
- Chaikin Money Flow (20) and the Accumulation/Distribution line trend
- Rolling 20-day VWAP and the price deviation from it (percent and ATR units)
- Volume profile over the last 120 bars: point of control and value area
- Relative volume (RVOL) of the last completed bar vs 5/20/60-day averages
- Selling and buying volume climaxes
- Up-volume vs down-volume ratio

Everything is computed from the OHLCV passed in (at most the last 260 bars) with
vectorized numpy, so the agent is point-in-time safe and fast enough for the
score backtester. When ``ohlcv.attrs["intraday_partial"]`` is set, RVOL and
climax detection use the last completed bar; cumulative flow measures (OBV, CMF,
VWAP, profile) keep the partial bar because its volume has really traded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from agents.base import LIVE, PARTIAL, AgentAssessment, BaseAgent, iso_date
from agents.registry import register_agent

LOOKBACK = 260  # Only the most recent bars are used.
MIN_BARS = 60
FLOW_WINDOW = 20  # OBV slope/divergence, CMF, VWAP, up/down volume, climax baseline
AD_WINDOW = 60
ATR_WINDOW = 14
VWAP_SLOPE_LAG = 5  # VWAP is "rising" when above its value this many bars ago
PROFILE_WINDOW = 120
PROFILE_BINS = 30
VALUE_AREA_SHARE = 0.70
RVOL_WINDOWS = (5, 20, 60)
CLIMAX_VOLUME_MULT = 2.5
CLIMAX_RANGE_MULT = 1.5  # true range vs 20-bar average true range
CLIMAX_EXTREME_ATR = 0.5  # "near the 20-day low/high" tolerance in ATRs
CLIMAX_LOOKBACK = 5  # completed bars scanned for a recent climax
VWAP_EXTENDED_ATR = 2.5

REQUIRED_COLUMNS = ("high", "low", "close", "volume")


def _slope(y: np.ndarray) -> float:
    """Least-squares slope of ``y`` per bar."""
    x = np.arange(y.size, dtype=float)
    x -= x.mean()
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


def _ratio(num: float, den: float) -> float:
    """``num / den`` or NaN when the denominator is not positive."""
    return float(num / den) if den > 0 else float("nan")


def _value_area(tp: np.ndarray, vol: np.ndarray, lo: float, hi: float) -> tuple[float, float, float]:
    """Return (POC, VAH, VAL) of a volume-weighted typical-price histogram."""
    hist, edges = np.histogram(tp, bins=PROFILE_BINS, range=(lo, hi), weights=vol)
    poc_i = int(hist.argmax())
    lo_i = hi_i = poc_i
    covered = float(hist[poc_i])
    target = VALUE_AREA_SHARE * float(hist.sum())
    last = hist.size - 1
    # Grow the value area one bin at a time toward the heavier neighbour.
    while covered < target and (lo_i > 0 or hi_i < last):
        below = hist[lo_i - 1] if lo_i > 0 else -1.0
        above = hist[hi_i + 1] if hi_i < last else -1.0
        if above >= below:
            hi_i += 1
            covered += float(above)
        else:
            lo_i -= 1
            covered += float(below)
    poc = float((edges[poc_i] + edges[poc_i + 1]) / 2.0)
    return poc, float(edges[hi_i + 1]), float(edges[lo_i])


def _fmt_volume(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.2f}{suffix}"
    return f"{x:.0f}"


def _fmt_price(x: float) -> str:
    return f"${x:,.2f}" if abs(x) >= 1 else f"${x:,.4f}"


def _fmt_rvol(x: float) -> str:
    return f"{x:.2f}x" if np.isfinite(x) else "n/a (not enough history)"


def _num(x: float, digits: int = 4) -> float | None:
    """JSON-safe rounded float (None for NaN/inf)."""
    return round(float(x), digits) if np.isfinite(x) else None


@register_agent
class VolumeAgent(BaseAgent):
    """Scores accumulation vs distribution from OBV, money flow, VWAP, volume profile and climaxes."""

    key = "volume"
    name = "Volume & Accumulation Agent"
    description = (
        "Reads accumulation and distribution from OBV, Chaikin Money Flow, VWAP, "
        "the volume profile, relative volume and volume climaxes."
    )
    category = "Technical"
    score_scale = 45.0
    default_weight = 0.15
    point_in_time = True

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in ohlcv.columns]
        if missing_cols:
            return self.unavailable(f"OHLCV is missing columns: {', '.join(missing_cols)}")

        partial = bool(ohlcv.attrs.get("intraday_partial", False))
        frame = ohlcv.iloc[-LOOKBACK:]
        index = frame.index
        h = frame["high"].to_numpy(dtype=float)
        l = frame["low"].to_numpy(dtype=float)  # noqa: E741
        c = frame["close"].to_numpy(dtype=float)
        v = frame["volume"].to_numpy(dtype=float)

        valid = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v)
        if not valid.all():
            if valid.size and not valid[-1]:
                partial = False  # the forming bar itself was dropped
            h, l, c, v, index = h[valid], l[valid], c[valid], v[valid], index[valid]

        n = c.size
        if n < MIN_BARS:
            return self.unavailable(f"need at least {MIN_BARS} daily bars with volume, got {n}")
        w = FLOW_WINDOW
        avg_vol = float(v[-w:].mean())
        if avg_vol <= 0:
            return self.unavailable("no traded volume reported in the last 20 bars")

        vi = n - 2 if partial else n - 1  # bar used for volume ratios
        last_close = float(c[-1])

        # ------------------------------------------------------------ base series
        prev_c = np.concatenate((c[:1], c[:-1]))
        dc = c - prev_c
        obv = np.cumsum(np.sign(dc) * v)
        bar_range = h - l
        safe_range = np.where(bar_range > 0, bar_range, 1.0)
        mfv = np.where(bar_range > 0, ((c - l) - (h - c)) / safe_range, 0.0) * v
        ad = np.cumsum(mfv)
        tr = np.maximum(bar_range, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        tp = (h + l + c) / 3.0
        pv = tp * v
        atr = float(tr[-ATR_WINDOW:].mean())

        # 1. OBV trend and divergence
        obv_slope = _slope(obv[-w:]) / avg_vol
        half = w // 2
        c_prior, c_recent = c[-w:-half], c[-half:]
        o_prior, o_recent = obv[-w:-half], obv[-half:]
        tol = 0.25 * avg_vol
        bull_div = bool(c_recent.min() < c_prior.min() and o_recent.min() > o_prior.min() + tol)
        bear_div = bool(c_recent.max() > c_prior.max() and o_recent.max() < o_prior.max() - tol)

        # 2. Chaikin Money Flow and A/D line
        cmf = _ratio(float(mfv[-w:].sum()), float(v[-w:].sum()))
        ad_slope = _slope(ad[-AD_WINDOW:]) / max(float(v[-AD_WINDOW:].mean()), 1e-12)

        # 3. Rolling 20-day VWAP
        vwap = _ratio(float(pv[-w:].sum()), float(v[-w:].sum()))
        lag = VWAP_SLOPE_LAG
        vwap_prev = _ratio(float(pv[-w - lag:-lag].sum()), float(v[-w - lag:-lag].sum()))
        vwap_rising = bool(np.isfinite(vwap_prev) and vwap > vwap_prev)
        vwap_dev_pct = last_close / vwap - 1.0
        vwap_dev_atr = (last_close - vwap) / atr if atr > 0 else float("nan")

        # 4. Volume profile
        pw = min(PROFILE_WINDOW, n)
        poc, vah, val = _value_area(tp[-pw:], v[-pw:], float(l[-pw:].min()), float(h[-pw:].max()))

        # 5. Relative volume of the last completed bar
        rvol: dict[int, float] = {}
        rvol_avg: dict[int, float] = {}
        for win in RVOL_WINDOWS:
            if vi >= win:
                rvol_avg[win] = float(v[vi - win:vi].mean())
                rvol[win] = _ratio(float(v[vi]), rvol_avg[win])
            else:
                rvol_avg[win] = rvol[win] = float("nan")

        # 6. Volume climax over the last few completed bars
        climax = self._detect_climax(h, l, c, v, dc, tr, bar_range, vi)

        # 7. Up vs down volume
        seg_dc, seg_v = dc[-w:], v[-w:]
        up_vol = float(seg_v[seg_dc > 0].sum())
        down_vol = float(seg_v[seg_dc < 0].sum())
        if down_vol > 0:
            ud_ratio = up_vol / down_vol
        else:
            ud_ratio = float("inf") if up_vol > 0 else float("nan")

        # ---------------------------------------------------------------- scoring
        breakdown: list[dict[str, Any]] = []
        factors: list[str] = []

        def add(label: str, points: float, text: str) -> None:
            if points:
                breakdown.append({"factor": label, "points": float(points)})
            factors.append(text)

        obv_text = f"over 20 bars ({obv_slope:+.2f}x avg volume per bar)"
        if obv_slope >= 0.30:
            add("OBV trend rising strongly", 12, f"OBV rising strongly {obv_text}")
        elif obv_slope >= 0.10:
            add("OBV trend rising", 6, f"OBV rising {obv_text}")
        elif obv_slope <= -0.30:
            add("OBV trend falling strongly", -12, f"OBV falling strongly {obv_text}")
        elif obv_slope <= -0.10:
            add("OBV trend falling", -6, f"OBV falling {obv_text}")
        else:
            add("OBV trend flat", 0, f"OBV roughly flat {obv_text}")

        if bull_div:
            add("Bullish OBV divergence", 10, "Bullish OBV divergence: price made a lower low but OBV made a higher low")
        elif bear_div:
            add("Bearish OBV divergence", -10, "Bearish OBV divergence: price made a higher high but OBV made a lower high")

        if cmf >= 0.15:
            add("CMF strong inflow", 10, f"Chaikin Money Flow (20) at {cmf:+.2f} shows strong accumulation")
        elif cmf >= 0.05:
            add("CMF inflow", 5, f"Chaikin Money Flow (20) at {cmf:+.2f} shows mild accumulation")
        elif cmf <= -0.15:
            add("CMF strong outflow", -10, f"Chaikin Money Flow (20) at {cmf:+.2f} shows strong distribution")
        elif cmf <= -0.05:
            add("CMF outflow", -5, f"Chaikin Money Flow (20) at {cmf:+.2f} shows mild distribution")
        else:
            add("CMF neutral", 0, f"Chaikin Money Flow (20) neutral at {cmf:+.2f}")

        if ad_slope >= 0.10:
            add("A/D line rising (60d)", 5, "Accumulation/Distribution line has trended up over 60 bars")
        elif ad_slope <= -0.10:
            add("A/D line falling (60d)", -5, "Accumulation/Distribution line has trended down over 60 bars")

        vwap_dir = "rising" if vwap_rising else "falling"
        if np.isfinite(vwap_dev_atr):
            dev_text = f"{vwap_dev_pct:+.1%}, {vwap_dev_atr:+.1f} ATR"
            if vwap_dev_atr >= VWAP_EXTENDED_ATR:
                add("Extended above VWAP", -5, f"Price extended far above the 20-day VWAP ({dev_text}); pullback risk")
            elif vwap_dev_atr >= 0 and vwap_rising:
                add("Above rising VWAP", 8, f"Price modestly above a rising 20-day VWAP ({dev_text})")
            elif vwap_dev_atr >= 0:
                add("Above flat or falling VWAP", 3, f"Price above a flat or falling 20-day VWAP ({dev_text})")
            elif not vwap_rising:
                add("Below falling VWAP", -8, f"Price below a falling 20-day VWAP ({dev_text}), a distribution sign")
            else:
                add("Below rising VWAP", -4, f"Price below a rising 20-day VWAP ({dev_text})")

        if last_close > vah:
            va_state = "Above value area"
            add("Above value area high", 8, f"Price above the {pw}-bar value area high ({_fmt_price(vah)}): acceptance higher")
        elif last_close < val:
            va_state = "Below value area"
            add("Below value area low", -8, f"Price below the {pw}-bar value area low ({_fmt_price(val)}): acceptance lower")
        elif last_close >= poc:
            va_state = "Inside value area, above POC"
            add("Inside value area above POC", 2, f"Price inside the {pw}-bar value area, above the point of control (balance)")
        else:
            va_state = "Inside value area, below POC"
            add("Inside value area below POC", -2, f"Price inside the {pw}-bar value area, below the point of control (balance)")

        rvol20 = rvol[20]
        if climax is not None and climax["bars_ago"] == 0:
            pass  # the climax rule interprets this bar's volume
        elif np.isfinite(rvol20) and rvol20 >= 1.5 and dc[vi] > 0:
            add("Heavy volume up day", 6, f"Last completed bar rose on {rvol20:.1f}x its 20-day average volume")
        elif np.isfinite(rvol20) and rvol20 >= 1.5 and dc[vi] < 0:
            add("Heavy volume down day", -6, f"Last completed bar fell on {rvol20:.1f}x its 20-day average volume")
        elif np.isfinite(rvol20) and rvol20 < 0.5:
            factors.append(f"Light volume on the last completed bar ({rvol20:.2f}x the 20-day average)")

        if climax is not None:
            when = "on the last completed bar" if climax["bars_ago"] == 0 else f"{climax['bars_ago']} bars ago"
            if climax["kind"] == "selling":
                add("Selling climax", 12, (
                    f"Selling climax {when} ({climax['volume_mult']:.1f}x avg volume near the 20-day low, "
                    "closed in the upper half): potential capitulation bottom"
                ))
            else:
                add("Buying climax", -12, (
                    f"Buying climax {when} ({climax['volume_mult']:.1f}x avg volume at the 20-day high, "
                    "closed in the lower half): potential exhaustion"
                ))

        if ud_ratio == float("inf"):
            ud_text = "All volume over the last 20 bars came on up days"
        else:
            ud_text = f"Up-day volume is {self._fmt_ratio(ud_ratio)} down-day volume over 20 bars"
        if ud_ratio >= 1.5:  # NaN compares False
            add("Up volume dominates", 8, ud_text)
        elif ud_ratio >= 1.2:
            add("Up volume leads", 4, ud_text)
        elif ud_ratio <= 1 / 1.5:
            add("Down volume dominates", -8, ud_text)
        elif ud_ratio <= 1 / 1.2:
            add("Down volume leads", -4, ud_text)

        vol_bar_date = iso_date(index[vi])
        if partial:
            factors.append(f"Volume ratios use the last completed bar ({vol_bar_date}) because today's bar is still forming")

        limited: list[str] = []
        if not np.isfinite(rvol[60]):
            limited.append("60-day RVOL")
        if pw < PROFILE_WINDOW:
            limited.append(f"volume profile ({pw} of {PROFILE_WINDOW} bars)")
        if limited:
            factors.append(f"Limited history for: {', '.join(limited)}")

        points = [b["points"] for b in breakdown]
        raw_score = float(np.clip(sum(points), -100.0, 100.0))
        confidence = self._confidence(points, n, bool(limited))
        data_quality = PARTIAL if limited else LIVE

        # ---------------------------------------------------------------- output
        vol_bar_label = vol_bar_date or "n/a"
        if partial:
            vol_bar_label += " (today still forming)"
        climax_label = "None"
        if climax is not None:
            climax_label = (
                f"{climax['kind'].capitalize()} climax on {iso_date(index[climax['bar']])} "
                f"({climax['volume_mult']:.1f}x avg volume)"
            )
        div_label = "Bullish" if bull_div else "Bearish" if bear_div else "None"

        metrics_table = [
            {"metric": "Bars analyzed", "value": str(n)},
            {"metric": "Volume bar used for ratios", "value": vol_bar_label},
            {"metric": "Average volume (20d)", "value": _fmt_volume(avg_vol)},
            {"metric": "OBV slope (20d, per bar vs avg volume)", "value": f"{obv_slope:+.2f}x"},
            {"metric": "OBV divergence (20d)", "value": div_label},
            {"metric": "Chaikin Money Flow (20)", "value": f"{cmf:+.3f}"},
            {"metric": "A/D line slope (60d, per bar vs avg volume)", "value": f"{ad_slope:+.2f}x"},
            {"metric": "VWAP (20d)", "value": f"{_fmt_price(vwap)} ({vwap_dir})"},
            {"metric": "Price vs VWAP", "value": (
                f"{vwap_dev_pct:+.2%} ({vwap_dev_atr:+.2f} ATR)" if np.isfinite(vwap_dev_atr) else f"{vwap_dev_pct:+.2%}"
            )},
            {"metric": "ATR (14)", "value": _fmt_price(atr)},
            {"metric": f"Point of control ({pw} bars)", "value": _fmt_price(poc)},
            {"metric": "Value area high", "value": _fmt_price(vah)},
            {"metric": "Value area low", "value": _fmt_price(val)},
            {"metric": "Price vs value area", "value": va_state},
            {"metric": "RVOL vs 5d avg", "value": _fmt_rvol(rvol[5])},
            {"metric": "RVOL vs 20d avg", "value": _fmt_rvol(rvol[20])},
            {"metric": "RVOL vs 60d avg", "value": _fmt_rvol(rvol[60])},
            {"metric": f"Volume climax (last {CLIMAX_LOOKBACK} bars)", "value": climax_label},
            {"metric": "Up/down volume ratio (20d)", "value": self._fmt_ratio(ud_ratio)},
        ]
        rvol_table = {
            "title": f"Relative volume ({vol_bar_label})",
            "columns": ["Window", "Average volume", "Bar volume", "RVOL"],
            "rows": [
                [f"{win} day", _fmt_volume(rvol_avg[win]), _fmt_volume(float(v[vi])), _fmt_rvol(rvol[win])]
                for win in RVOL_WINDOWS
            ],
        }
        levels = [
            {"label": "VWAP 20", "price": round(vwap, 4), "kind": "pivot"},
            {"label": "Point of control", "price": round(poc, 4), "kind": "pivot"},
            {"label": "Value area high", "price": round(vah, 4), "kind": "resistance"},
            {"label": "Value area low", "price": round(val, 4), "kind": "support"},
        ]

        raw_data: dict[str, Any] = {
            "metrics_table": metrics_table,
            "score_breakdown": breakdown,
            "levels": levels,
            "links": [{"label": "TradingView chart", "url": f"https://www.tradingview.com/symbols/{ticker}/"}],
            "table": rvol_table,
            "raw_score": round(raw_score, 1),
            "bars_used": n,
            "volume_bar_date": vol_bar_date,
            "intraday_partial": partial,
            "obv_slope_norm": _num(obv_slope),
            "obv_divergence": div_label.lower(),
            "cmf_20": _num(cmf),
            "ad_slope_norm": _num(ad_slope),
            "vwap_20": _num(vwap),
            "vwap_rising": vwap_rising,
            "vwap_dev_pct": _num(vwap_dev_pct),
            "vwap_dev_atr": _num(vwap_dev_atr),
            "atr_14": _num(atr),
            "poc": _num(poc),
            "vah": _num(vah),
            "val": _num(val),
            "profile_bars": pw,
            "rvol_5": _num(rvol[5]),
            "rvol_20": _num(rvol[20]),
            "rvol_60": _num(rvol[60]),
            "climax": climax["kind"] if climax else None,
            "climax_date": iso_date(index[climax["bar"]]) if climax else None,
            "up_down_volume_ratio": _num(ud_ratio),
        }

        return self.assessment(
            score=raw_score,
            confidence=confidence,
            summary=self._summary(raw_score, breakdown),
            key_factors=factors,
            raw_data=raw_data,
            data_quality=data_quality,
            data_as_of=iso_date(index[-1]),
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _detect_climax(
        h: np.ndarray,
        l: np.ndarray,  # noqa: E741
        c: np.ndarray,
        v: np.ndarray,
        dc: np.ndarray,
        tr: np.ndarray,
        bar_range: np.ndarray,
        vi: int,
    ) -> dict[str, Any] | None:
        """Most recent valid selling/buying climax among the last completed bars."""
        w = FLOW_WINDOW
        start = max(w, vi - CLIMAX_LOOKBACK + 1)
        idx = np.arange(start, vi + 1)
        if idx.size == 0:
            return None
        cs_v = np.concatenate(([0.0], np.cumsum(v)))
        cs_tr = np.concatenate(([0.0], np.cumsum(tr)))
        avg_v = (cs_v[idx] - cs_v[idx - w]) / w  # prior 20 bars, excluding the bar
        avg_tr = (cs_tr[idx] - cs_tr[idx - w]) / w
        low20 = sliding_window_view(l, w)[idx - w + 1].min(axis=1)  # 20 bars ending at the bar
        high20 = sliding_window_view(h, w)[idx - w + 1].max(axis=1)
        vol_mult = np.divide(v[idx], avg_v, out=np.zeros(idx.size), where=avg_v > 0)
        rng = bar_range[idx]
        pos = np.divide(c[idx] - l[idx], rng, out=np.full(idx.size, 0.5), where=rng > 0)

        climactic = (vol_mult >= CLIMAX_VOLUME_MULT) & (tr[idx] >= CLIMAX_RANGE_MULT * avg_tr)
        selling = climactic & (dc[idx] < 0) & (l[idx] - low20 <= CLIMAX_EXTREME_ATR * avg_tr) & (pos >= 0.5)
        buying = climactic & (dc[idx] > 0) & (high20 - h[idx] <= CLIMAX_EXTREME_ATR * avg_tr) & (pos < 0.5)
        if not (selling.any() or buying.any()):
            return None

        # A later close beyond the climax extreme invalidates it.
        tail = c[start:]
        later_min = np.append(np.minimum.accumulate(tail[::-1])[::-1][1:], np.inf)[idx - start]
        later_max = np.append(np.maximum.accumulate(tail[::-1])[::-1][1:], -np.inf)[idx - start]
        selling &= later_min >= l[idx]
        buying &= later_max <= h[idx]
        hits = np.flatnonzero(selling | buying)
        if hits.size == 0:
            return None
        k = int(hits[-1])
        return {
            "kind": "selling" if selling[k] else "buying",
            "bar": int(idx[k]),
            "bars_ago": int(vi - idx[k]),
            "volume_mult": float(vol_mult[k]),
        }

    @staticmethod
    def _confidence(points: list[float], n_bars: int, limited: bool) -> float:
        """Base 50, up to +30 when rules agree, minus up to 15 for short history."""
        bull = sum(p for p in points if p > 0)
        bear = -sum(p for p in points if p < 0)
        gross = bull + bear
        agreement = abs(bull - bear) / gross if gross else 0.0
        confidence = 50.0 + 30.0 * agreement * min(1.0, gross / 40.0)
        confidence -= 15.0 * min(1.0, max(0.0, (LOOKBACK - n_bars) / (LOOKBACK - MIN_BARS)))
        if limited:
            confidence -= 5.0
        return float(np.clip(confidence, 10.0, 90.0))

    @staticmethod
    def _fmt_ratio(ratio: float) -> str:
        if ratio == float("inf"):
            return "all up volume"
        return f"{ratio:.2f}x" if np.isfinite(ratio) else "n/a"

    @staticmethod
    def _summary(score: float, breakdown: list[dict[str, Any]]) -> str:
        tone = "accumulation" if score >= 20 else "distribution" if score <= -20 else "a balanced tape"
        text = f"Volume and money flow point to {tone} (raw score {score:+.0f})."
        top = sorted(breakdown, key=lambda b: abs(b["points"]), reverse=True)[:2]
        if top:
            drivers = " and ".join(f"{b['factor']} ({b['points']:+.0f})" for b in top)
            text += f" Main driver{'s' if len(top) > 1 else ''}: {drivers}."
        return text
