"""Options Flow Agent: positioning, volatility surface and open interest levels.

Turns the chain analytics from :mod:`data.options_flow` into a rules-based
score with a transparent ``score_breakdown``. A missing chain is reported as
UNAVAILABLE. Stale quotes or thin coverage are reported as PARTIAL with lower
confidence. Nothing is estimated or back-filled.

yfinance has no trade side (bid vs ask prints), so volume can not tell buying
from selling. Flow factors are therefore kept moderate, and the call overwriting
check is explicitly labeled as a heuristic.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from agents.base import LIVE, PARTIAL, AgentAssessment, BaseAgent
from agents.registry import register_agent
from data.options_flow import OptionsFlowData, fetch_options_flow

logger = logging.getLogger(__name__)

# Products whose put open interest structurally dwarfs calls (index ETFs used for
# hedging). Their P/C and skew are judged against their own baseline.
HEDGE_HEAVY_PC_OI = 1.5
SINGLE_NAME_PC_BASELINE = 0.70
MIN_FLOW_PREMIUM = 25_000.0
MAX_PAIN_DISTANCE = 0.03


def _links(ticker: str) -> list[dict[str, str]]:
    t = ticker.upper()
    return [
        {"label": "Yahoo Finance options", "url": f"https://finance.yahoo.com/quote/{t}/options/"},
        {"label": "Barchart options", "url": f"https://www.barchart.com/stocks/quotes/{t}/options"},
    ]


def _usd(value: float) -> str:
    v = abs(value)
    sign = "-" if value < 0 else ""
    if v >= 1e9:
        return f"{sign}${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{sign}${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{sign}${v / 1e3:.0f}K"
    return f"{sign}${v:.0f}"


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: float | None, fmt: str = ".2f", suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{fmt}}{suffix}"


def _strike(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}".rstrip("0").rstrip(".")


@register_agent
class OptionsFlowAgent(BaseAgent):
    """Scores options positioning (P/C, premium flow, unusual activity) and the IV surface."""

    key = "options"
    name = "Options Flow Agent"
    description = (
        "Listed options positioning and volatility: put/call volume and open interest, premium-weighted "
        "flow, unusual activity, ATM IV, skew, term structure, IV vs realized volatility, max pain and "
        "open interest walls (Yahoo Finance chains, 7 to 180 DTE)."
    )
    category = "Flow"
    score_scale = 45.0
    default_weight = 0.15
    point_in_time = False

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        return self.evaluate_with_data(fetch_options_flow(ticker, ohlcv))

    def evaluate_with_data(self, data: OptionsFlowData) -> AgentAssessment:
        """Build the assessment from already analyzed chain data (used by tests)."""
        links = _links(data.ticker)
        if not data.available:
            reason = data.error or "no option chain available"
            return self.unavailable(reason, raw_data={"links": links}, error=data.error)

        breakdown, factors = self._score(data)
        score = sum(item["points"] for item in breakdown)
        confidence, liquidity_note = self._confidence(data)
        factors.extend(self._context_factors(data, liquidity_note))

        levels = self._levels(data)
        table = self._unusual_table(data)
        raw_data: dict[str, Any] = {
            "iv_hv_ratio": data.iv_hv_ratio,
            "iv_rank": data.iv_rank,
            "iv_percentile": data.iv_percentile,
            "iv_rank_source": data.iv_rank_source,
            "iv_history_days": data.iv_history_days,
            "atm_iv": data.atm_iv,
            "atm_expiry": data.atm_expiry,
            "atm_dte": data.atm_dte,
            "atm_iv_far": data.atm_iv_far,
            "far_expiry": data.far_expiry,
            "far_dte": data.far_dte,
            "hv20": data.hv20,
            "skew": data.skew,
            "skew_normalized": data.skew_normalized,
            "term_structure_ratio": data.term_structure_ratio,
            "term_structure_state": data.term_structure_state,
            "max_pain": data.max_pain,
            "max_pain_expiry": data.max_pain_expiry,
            "call_wall": data.call_wall,
            "put_wall": data.put_wall,
            "put_call_volume_ratio": data.put_call_volume_ratio,
            "put_call_oi_ratio": data.put_call_oi_ratio,
            "total_call_volume": data.total_call_volume,
            "total_put_volume": data.total_put_volume,
            "total_call_oi": data.total_call_oi,
            "total_put_oi": data.total_put_oi,
            "call_premium": data.call_premium,
            "put_premium": data.put_premium,
            "otm_call_premium": data.otm_call_premium,
            "otm_put_premium": data.otm_put_premium,
            "iv_crush_warning": data.iv_crush_warning,
            "call_overwriting_heuristic": data.call_overwriting_heuristic,
            "earnings_date": data.earnings_date,
            "spot": data.spot,
            "quoted_share": data.quoted_share,
            "quality_notes": list(data.quality_notes),
            "expirations": list(data.expirations),
            "levels": levels,
            "unusual_contracts": list(data.unusual_contracts),
            "table": table,
            "metrics_table": self._metrics_table(data),
            "score_breakdown": breakdown,
            "links": links,
            "source_url": links[0]["url"],
            "data_source_label": "Yahoo Finance option chains",
        }

        return self.assessment(
            score=score,
            confidence=confidence,
            summary=self._summary(data, score),
            key_factors=factors,
            raw_data=raw_data,
            data_quality=PARTIAL if data.quality == PARTIAL else LIVE,
            data_as_of=data.data_as_of,
        )

    # ------------------------------------------------------------------ scoring

    @staticmethod
    def _score(d: OptionsFlowData) -> tuple[list[dict[str, Any]], list[str]]:
        breakdown: list[dict[str, Any]] = []
        factors: list[str] = []

        def add(label: str, points: float, note: str | None = None) -> None:
            breakdown.append({"factor": label, "points": float(points)})
            if note and points:
                factors.append(note)

        hedge_heavy = d.put_call_oi_ratio is not None and d.put_call_oi_ratio >= HEDGE_HEAVY_PC_OI

        # 1. Put/call volume ratio, relative to the name's baseline.
        pcv = d.put_call_volume_ratio
        if pcv is None:
            add("P/C volume ratio (n/a)", 0.0)
        else:
            baseline = d.put_call_oi_ratio if hedge_heavy else SINGLE_NAME_PC_BASELINE
            rel = pcv / baseline
            basis = "own OI ratio, hedge-heavy product" if hedge_heavy else "single-name norm"
            label = f"P/C volume ratio {pcv:.2f} vs baseline {baseline:.2f} ({basis})"
            if rel <= 0.60:
                add(label, 10.0, f"Call-heavy option volume: P/C {pcv:.2f} vs a {baseline:.2f} baseline")
            elif rel <= 0.85:
                add(label, 5.0, f"Calls outpace puts: P/C volume {pcv:.2f} vs a {baseline:.2f} baseline")
            elif rel >= 1.80:
                add(label, -10.0, f"Put-heavy option volume: P/C {pcv:.2f} vs a {baseline:.2f} baseline")
            elif rel >= 1.35:
                add(label, -5.0, f"Puts outpace calls: P/C volume {pcv:.2f} vs a {baseline:.2f} baseline")
            else:
                add(label, 0.0)

        # 2. Premium-weighted OTM flow (directional bets).
        otm_total = d.otm_call_premium + d.otm_put_premium
        if otm_total < MIN_FLOW_PREMIUM:
            add("OTM premium flow (too thin to read)", 0.0)
        else:
            share = d.otm_call_premium / otm_total
            label = f"OTM premium flow {share:.0%} calls of {_usd(otm_total)}"
            scale = 0.5 if hedge_heavy else 1.0
            if share >= 0.70:
                pts, note = 10.0, f"OTM call premium dominates: {share:.0%} of {_usd(otm_total)} OTM premium"
            elif share >= 0.60:
                pts, note = 5.0, f"OTM call premium leads: {share:.0%} of {_usd(otm_total)} OTM premium"
            elif share <= 0.30:
                pts, note = -10.0, f"OTM put premium dominates: {1 - share:.0%} of {_usd(otm_total)} OTM premium"
            elif share <= 0.40:
                pts, note = -5.0, f"OTM put premium leads: {1 - share:.0%} of {_usd(otm_total)} OTM premium"
            else:
                pts, note = 0.0, None
            if hedge_heavy and pts:
                label += " (halved: hedge-heavy product)"
            add(label, pts * scale, note)

        # 3. Put skew, normalized by ATM IV so high and low vol names compare.
        if d.skew is None or d.skew_normalized is None:
            add("Put skew (n/a)", 0.0)
        else:
            steep, elevated = (0.50, 0.40) if hedge_heavy else (0.20, 0.10)
            label = f"Put skew {d.skew:+.1f} vol pts ({d.skew_normalized:+.2f}x ATM IV)"
            if d.skew_normalized >= steep:
                add(label, -8.0, f"Steep put skew: 95% puts {d.skew:.1f} vol pts over 105% calls, heavy hedging demand")
            elif d.skew_normalized >= elevated:
                add(label, -4.0, f"Elevated put skew: 95% puts {d.skew:.1f} vol pts over 105% calls")
            elif d.skew_normalized <= -0.05:
                add(label, 5.0, f"Call skew: 105% calls priced {abs(d.skew):.1f} vol pts over 95% puts, upside demand")
            else:
                add(label, 0.0)

        # 4. Term structure (near ~30 DTE vs 90 to 180 DTE).
        ratio = d.term_structure_ratio
        if ratio is None:
            add("IV term structure (n/a)", 0.0)
        else:
            label = f"IV term structure {ratio:.2f} (near / far)"
            event = d.earnings_before_atm_expiry
            if ratio >= 1.10:
                pts = -4.0 if event else -8.0
                note = (
                    f"Inverted IV term structure ({ratio:.2f}) into earnings on {d.earnings_date}: event premium"
                    if event
                    else f"Inverted IV term structure ({ratio:.2f}): near-term stress or event risk"
                )
                add(label + (" (halved: earnings)" if event else ""), pts, note)
            elif ratio > 1.05:
                pts = -2.0 if event else -4.0
                note = f"Mild IV backwardation ({ratio:.2f})" + (f" ahead of earnings on {d.earnings_date}" if event else "")
                add(label + (" (halved: earnings)" if event else ""), pts, note)
            elif ratio < 0.95:
                add(label, 3.0, f"IV term structure in contango ({ratio:.2f}): calm near-term pricing")
            else:
                add(label, 0.0)

        # 5. IV vs realized volatility (event premium, crush risk).
        ivhv = d.iv_hv_ratio
        if ivhv is None:
            add("IV / HV20 (n/a)", 0.0)
        elif ivhv >= 1.5:
            add(f"IV / HV20 {ivhv:.2f}x", -6.0, f"IV is {ivhv:.2f}x 20-day realized volatility: rich event premium, IV crush risk")
        elif ivhv >= 1.25:
            add(f"IV / HV20 {ivhv:.2f}x", -3.0, f"IV above realized volatility ({ivhv:.2f}x HV20)")
        else:
            add(f"IV / HV20 {ivhv:.2f}x", 0.0)

        # 6. Unusual activity bias (volume > 2x OI).
        unusual_total = d.unusual_call_premium + d.unusual_put_premium
        if d.unusual_count == 0 or unusual_total <= 0:
            add("Unusual activity (none)", 0.0)
        else:
            net = (d.unusual_call_premium - d.unusual_put_premium) / unusual_total
            label = f"Unusual activity {d.unusual_count} contracts, net {net:+.0%} calls"
            detail = f"{_usd(d.unusual_call_premium)} calls vs {_usd(d.unusual_put_premium)} puts"
            if net >= 0.5:
                add(label, 8.0, f"Unusual call activity: {d.unusual_count} contracts over 2x OI, {detail}")
            elif net >= 0.2:
                add(label, 4.0, f"Unusual activity leans to calls: {detail}")
            elif net <= -0.5:
                add(label, -8.0, f"Unusual put activity: {d.unusual_count} contracts over 2x OI, {detail}")
            elif net <= -0.2:
                add(label, -4.0, f"Unusual activity leans to puts: {detail}")
            else:
                add(label, 0.0)

        # 7. Spot vs max pain (weak, informational).
        if d.max_pain is None or not d.spot:
            add("Spot vs max pain (n/a)", 0.0)
        else:
            dist = d.spot / d.max_pain - 1.0
            label = f"Spot {dist:+.1%} vs max pain {_strike(d.max_pain)} (weak)"
            if dist >= MAX_PAIN_DISTANCE:
                add(label, -2.0, f"Spot {dist:+.1%} above max pain {_strike(d.max_pain)} ({d.max_pain_expiry}), weak pull lower")
            elif dist <= -MAX_PAIN_DISTANCE:
                add(label, 2.0, f"Spot {dist:+.1%} below max pain {_strike(d.max_pain)} ({d.max_pain_expiry}), weak pull higher")
            else:
                add(label, 0.0)

        # 8. Call overwriting heuristic (no trade side in yfinance).
        if d.call_overwriting_heuristic:
            add(
                "Call overwriting heuristic (no trade side data)",
                -3.0,
                "Heuristic only: heavy OTM call volume with a flat or falling price and rich IV may be "
                "call overwriting rather than call buying",
            )

        return breakdown, factors

    @staticmethod
    def _confidence(d: OptionsFlowData) -> tuple[float, str | None]:
        confidence = 75.0
        note = None
        volume = d.total_call_volume + d.total_put_volume
        oi = d.total_call_oi + d.total_put_oi
        if volume >= 50_000 and oi >= 250_000:
            confidence += 15.0
        elif volume >= 10_000 and oi >= 50_000:
            confidence += 8.0
        elif volume < 1_000 or oi < 5_000:
            confidence -= 20.0
            note = f"Thin options market ({volume:,} contracts traded, {oi:,} open interest): signals are noisy"
        confidence -= 10.0 * len(d.quality_notes)
        if d.stale_quotes:
            confidence -= 10.0
        return max(10.0, min(90.0, confidence)), note

    @staticmethod
    def _context_factors(d: OptionsFlowData, liquidity_note: str | None) -> list[str]:
        factors: list[str] = []
        days = f"{d.iv_history_days} day{'s' if d.iv_history_days != 1 else ''}"
        if d.iv_rank_source == "history" and d.iv_rank is not None:
            factors.append(
                f"IV rank {d.iv_rank:.0f}, IV percentile {d.iv_percentile:.0f} over {days} of history"
            )
        elif d.iv_rank_source == "hv_proxy" and d.iv_rank is not None:
            factors.append(
                f"IV history still building ({days}); IV vs 1Y HV range (proxy) is {d.iv_rank:.0f}"
            )
        elif d.atm_iv is not None:
            factors.append(f"IV history still building ({days}); no IV rank yet")
        if d.iv_crush_warning:
            factors.append("IV crush warning: options price an event or IV is rich versus realized volatility")
        if d.earnings_date and not d.earnings_before_atm_expiry:
            factors.append(f"Next earnings {d.earnings_date}, after the {d.atm_expiry} expiry")
        if liquidity_note:
            factors.append(liquidity_note)
        if d.market_state and d.market_state.upper() != "REGULAR" and d.data_as_of:
            factors.append(f"Market not in regular session: quotes and volume are from the last session ({d.data_as_of})")
        factors.extend(d.quality_notes)
        return factors

    # ------------------------------------------------------------------ presentation

    @staticmethod
    def _levels(d: OptionsFlowData) -> list[dict[str, Any]]:
        levels: list[dict[str, Any]] = []
        if d.max_pain is not None:
            levels.append({"label": f"Max pain ({d.max_pain_expiry})", "price": d.max_pain, "kind": "pivot"})
        if d.call_wall is not None:
            levels.append({"label": f"Call OI wall ({d.call_wall_oi:,} OI)", "price": d.call_wall, "kind": "resistance"})
        if d.put_wall is not None:
            levels.append({"label": f"Put OI wall ({d.put_wall_oi:,} OI)", "price": d.put_wall, "kind": "support"})
        return levels

    @staticmethod
    def _unusual_table(d: OptionsFlowData) -> dict[str, Any]:
        rows = [
            [
                c["type"],
                c["expiry"],
                _strike(c["strike"]),
                f"{c['moneyness_pct']:+.1f}",
                f"{c['volume']:,}",
                f"{c['open_interest']:,}",
                f"{c['premium']:,.0f}",
            ]
            for c in d.unusual_contracts
        ]
        return {
            "title": "Unusual options activity (volume over 2x open interest)",
            "columns": ["Type", "Expiry", "Strike", "Moneyness %", "Volume", "OI", "Premium $"],
            "rows": rows,
        }

    @staticmethod
    def _metrics_table(d: OptionsFlowData) -> list[dict[str, str]]:
        proxy = d.iv_rank_source == "hv_proxy"
        rank_label = "IV vs 1Y HV range (proxy)" if proxy else "IV rank (1Y history)"
        pct_label = "IV percentile vs 1Y HV (proxy)" if proxy else "IV percentile (1Y history)"
        atm_where = f" ({d.atm_expiry}, {d.atm_dte} DTE)" if d.atm_expiry else ""
        far_where = f" ({d.far_expiry}, {d.far_dte} DTE)" if d.far_expiry else ""
        term = _num(d.term_structure_ratio)
        if d.term_structure_state:
            term += f" ({d.term_structure_state.lower()})"
        rows = [
            ("Spot price", _strike(d.spot)),
            (f"ATM IV{atm_where}", _pct(d.atm_iv)),
            (f"ATM IV far{far_where}", _pct(d.atm_iv_far)),
            ("IV term structure (near / far)", term),
            ("Put skew (95% put IV minus 105% call IV)", _num(d.skew, ".1f", " vol pts")),
            ("Historical volatility (20d)", _pct(d.hv20)),
            ("IV / HV20", _num(d.iv_hv_ratio, ".2f", "x")),
            (rank_label, _num(d.iv_rank, ".0f")),
            (pct_label, _num(d.iv_percentile, ".0f")),
            ("IV history snapshots", str(d.iv_history_days)),
            ("P/C volume ratio (7-60 DTE)", _num(d.put_call_volume_ratio)),
            ("P/C open interest ratio (7-60 DTE)", _num(d.put_call_oi_ratio)),
            ("Call / put volume", f"{d.total_call_volume:,} / {d.total_put_volume:,}"),
            ("Call / put open interest", f"{d.total_call_oi:,} / {d.total_put_oi:,}"),
            ("Call / put premium", f"{_usd(d.call_premium)} / {_usd(d.put_premium)}"),
            ("OTM call / OTM put premium", f"{_usd(d.otm_call_premium)} / {_usd(d.otm_put_premium)}"),
            ("Unusual contracts (volume over 2x OI)", str(d.unusual_count)),
            (f"Max pain ({d.max_pain_expiry})" if d.max_pain_expiry else "Max pain", _strike(d.max_pain)),
            ("Call OI wall (resistance)", _strike(d.call_wall)),
            ("Put OI wall (support)", _strike(d.put_wall)),
            ("Next earnings", d.earnings_date or "n/a"),
            ("IV crush warning", "Active" if d.iv_crush_warning else "Clear"),
            ("Quotes with live bid and ask", _pct(d.quoted_share, 0)),
            ("Expirations analyzed", ", ".join(f"{e['expiry']} ({e['dte']}d)" for e in d.expirations) or "n/a"),
        ]
        return [{"metric": m, "value": v} for m, v in rows]

    @staticmethod
    def _summary(d: OptionsFlowData, score: float) -> str:
        lean = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
        parts = [f"Options positioning leans {lean} (raw score {score:+.0f})."]
        if d.put_call_volume_ratio is not None:
            parts.append(f"P/C volume {d.put_call_volume_ratio:.2f} over 7-60 DTE.")
        if d.atm_iv is not None:
            iv_text = f"ATM IV {d.atm_iv:.1%} at {d.atm_dte} DTE"
            if d.iv_hv_ratio is not None:
                iv_text += f" ({d.iv_hv_ratio:.2f}x HV20)"
            parts.append(iv_text + ".")
        if d.term_structure_ratio is not None:
            parts.append(f"Term structure {d.term_structure_ratio:.2f} ({(d.term_structure_state or '').lower()}).")
        if d.skew is not None:
            parts.append(f"Put skew {d.skew:+.1f} vol pts.")
        if d.iv_crush_warning:
            parts.append("IV crush warning active.")
        return " ".join(parts)
