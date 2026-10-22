"""Insider Trading Agent: scores SEC Form 4 open-market purchases and sales.

Only open-market purchases (code P) and sales (code S) count. Purchases are a
strong signal and are weighted by dollar size (log scaled and relative to the
stock's dollar volume), insider role, recency, and timing (buying after a 20%+
drawdown). Sales are a weak signal (0.35x, capped) because insiders sell for
diversification, taxes, and 10b5-1 plans. Grants, option exercises, gifts, and
tax withholding are shown but never scored.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from agents.base import AgentAssessment, BaseAgent, LIVE, OWNER, PARTIAL, iso_date
from agents.registry import register_agent
from data.insider_trading import (
    CATEGORY_LABELS,
    DEFAULT_LOOKBACK_DAYS,
    DRAWDOWN_BOOST,
    DRAWDOWN_THRESHOLD,
    GIFT,
    GRANT_AWARD,
    OPEN_MARKET_BUY,
    OPTION_EXERCISE,
    OTHER,
    SALE_WEIGHT,
    TAX_WITHHOLDING,
    InsiderTradingData,
    fetch_insider_trading,
    insider_links,
)

logger = logging.getLogger(__name__)

# Scoring rules (raw points, before the master agent's tanh normalization).
POINTS_PER_UNIT = 8.0  # per log10 dollar unit (role, recency, timing weighted)
SIZE_POINTS_PER_LOG = 6.0  # 6 * log10(1 + 100 * dollars / 20d avg dollar volume)
SIZE_POINTS_CAP = 12.0
PURCHASE_POINTS_CAP = 60.0  # before the drawdown boost
CLUSTER_BASE_POINTS = 10.0  # 2 distinct buyers within 30 days
CLUSTER_EXTRA_POINTS = 4.0  # per additional buyer
CLUSTER_POINTS_CAP = 18.0
SALE_POINTS_CAP = 12.0  # sales can never pull the score below -12 on their own
TRACK_RECORD_MAX_POINTS = 5.0
TRACK_RECORD_MIN_SAMPLES = 2

MAX_TRANSACTION_ROWS = 40
MAX_TABLE_ROWS = 25

_IGNORED_LABELS = {
    GRANT_AWARD: "grant/award",
    OPTION_EXERCISE: "option exercise",
    GIFT: "gift",
    TAX_WITHHOLDING: "tax withholding",
    OTHER: "unclassified",
}


def _money(value: float | None, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    sign = ("+" if value > 0 else "-" if value < 0 else "") if signed else ("-" if value < 0 else "")
    v = abs(value)
    if v >= 1e9:
        body = f"${v / 1e9:.2f}B"
    elif v >= 1e6:
        body = f"${v / 1e6:.2f}M"
    elif v >= 1e3:
        body = f"${v / 1e3:.1f}K"
    else:
        body = f"${v:,.0f}"
    return sign + body


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _size_points(ratio: float | None) -> float:
    if ratio is None or ratio <= 0:
        return 0.0
    return min(SIZE_POINTS_CAP, SIZE_POINTS_PER_LOG * math.log10(1.0 + 100.0 * ratio))


@register_agent
class InsiderAgent(BaseAgent):
    """Scores insider open-market buying (strong) and selling (weak) from Form 4 filings."""

    key = "insider"
    name = "Insider Trading Agent"
    description = (
        "SEC Form 4 open-market purchases and sales over the last 180 days, weighted by dollar size, "
        "role, recency, and buying after drawdowns. Grants, exercises, and gifts are ignored."
    )
    category = "Flow"
    score_scale = 35.0
    # Owner-layer evidence with no weight: the point-in-time test found no signal in the S&P 500
    # (decision_log.md T-10). Raise it for small caps after a retest.
    default_weight = 0.0
    point_in_time = False
    role = OWNER

    def __init__(self, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
        super().__init__()
        self.lookback_days = lookback_days

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        symbol = ticker.upper().strip()
        data = fetch_insider_trading(symbol, ohlcv, self.lookback_days)
        return self.evaluate_with_data(data)

    # ------------------------------------------------------------------ scoring

    @staticmethod
    def score_components(data: InsiderTradingData) -> list[dict[str, Any]]:
        """Point contributions from each rule (always the same five factors)."""
        base = min(
            PURCHASE_POINTS_CAP,
            POINTS_PER_UNIT * data.buy_units_no_boost + _size_points(data.buy_to_adv_no_boost),
        )
        boosted = min(
            PURCHASE_POINTS_CAP * DRAWDOWN_BOOST,
            POINTS_PER_UNIT * data.buy_units + _size_points(data.buy_to_adv),
        )
        drawdown_boost = max(0.0, boosted - base)

        cluster = 0.0
        if data.cluster_buy_detected:
            extra = max(0, data.max_buyers_in_cluster_window - 2)
            cluster = min(CLUSTER_POINTS_CAP, CLUSTER_BASE_POINTS + CLUSTER_EXTRA_POINTS * extra)

        sale = 0.0 - min(  # 0.0 - x avoids a "-0.0" in the UI when there are no sales
            SALE_POINTS_CAP,
            SALE_WEIGHT * (POINTS_PER_UNIT * data.sell_units + _size_points(data.sell_to_adv)),
        )

        track = 0.0
        n = data.buyer_track_evaluated
        if data.total_buy_transactions and n >= TRACK_RECORD_MIN_SAMPLES:
            edge = max(-1.0, min(1.0, (data.buyer_track_hits / n - 0.5) * 2.0))
            track = edge * TRACK_RECORD_MAX_POINTS * min(1.0, n / 4.0)

        return [
            {"factor": "Net weighted purchase intensity (size, role, recency)", "points": round(base, 1)},
            {"factor": "Cluster buying bonus (2+ insiders within 30 days)", "points": round(cluster, 1)},
            {"factor": f"Buys after {DRAWDOWN_THRESHOLD:.0%}+ drawdown ({DRAWDOWN_BOOST:.1f}x boost)", "points": round(drawdown_boost, 1)},
            {"factor": f"Weak sale pressure ({SALE_WEIGHT:.2f}x, capped at -{SALE_POINTS_CAP:.0f})", "points": round(sale, 1)},
            {"factor": "Buyer track record (60-day forward returns)", "points": round(track, 1)},
        ]

    @staticmethod
    def confidence_for(data: InsiderTradingData) -> float:
        """Grows with the number of open-market trades and how recent the latest one is."""
        if data.open_market_count == 0:
            return 15.0
        # Purchases are informative; sales add a little confidence, capped, because they are a weak signal.
        activity = 7.0 * data.total_buy_transactions + min(15.0, 2.0 * data.total_sell_transactions)
        confidence = 25.0 + min(35.0, activity)
        if data.latest_open_market_date is not None and data.reference_date is not None:
            age = (data.reference_date - data.latest_open_market_date).days
            confidence += 20.0 if age <= 30 else 10.0 if age <= 90 else 0.0
        if data.cluster_buy_detected:
            confidence += 5.0
        if not data.has_price_history:
            confidence -= 10.0
        return max(10.0, min(90.0, confidence))

    # --------------------------------------------------------------- assessment

    def evaluate_with_data(self, data: InsiderTradingData) -> AgentAssessment:
        """Build the assessment from pre-fetched (or synthetic) insider analytics."""
        links = insider_links(data.ticker)
        if not data.available:
            return self.unavailable(
                data.error or "no insider transactions reported",
                raw_data={"links": links, "lookback_days": data.lookback_days},
                error=data.error,
            )

        breakdown = self.score_components(data)
        score = sum(item["points"] for item in breakdown)
        confidence = self.confidence_for(data)
        data_quality = LIVE if data.has_price_history and data.unvalued_open_market_count == 0 else PARTIAL

        return self.assessment(
            score=score,
            confidence=confidence,
            summary=self._summary(data, score),
            key_factors=self._factors(data, breakdown),
            raw_data=self._raw_data(data, breakdown, links),
            data_quality=data_quality,
            data_as_of=iso_date(data.latest_transaction_date),
        )

    # ------------------------------------------------------------------ text

    @staticmethod
    def _summary(data: InsiderTradingData, score: float) -> str:
        window = f"the last {data.lookback_days} days"
        if data.open_market_count == 0:
            return (
                f"No open-market insider purchases or sales in {window} "
                f"({data.ignored_non_open_market_count} non-open-market rows ignored). "
                "Absence of buying is weak information, so the read is neutral."
            )
        parts: list[str] = []
        if data.total_buy_transactions:
            buy = f"{_plural(data.distinct_buyers, 'insider')} bought {_money(data.buy_value)} on the open market in {window}"
            if data.cluster_buy_detected:
                buy += ", including cluster buying"
            parts.append(buy)
        else:
            parts.append(f"No open-market purchases in {window}")
        if data.total_sell_transactions:
            parts.append(f"{_plural(data.distinct_sellers, 'insider')} sold {_money(data.sell_value)}")
        text = "; ".join(parts) + "."
        if not data.total_buy_transactions:
            text += " Sales are a weak signal, so the read is only mildly negative."
        return f"{text} Insider score {score:+.0f}."

    @staticmethod
    def _factors(data: InsiderTradingData, breakdown: list[dict[str, Any]]) -> list[str]:
        factors: list[str] = []
        buys = [t for t in data.transactions if t.category == OPEN_MARKET_BUY]

        if buys:
            top = max(buys, key=lambda t: t.dollars)
            who = f"{top.insider} ({top.position})" if top.position else top.insider
            factors.append(f"Largest purchase: {who} bought {_money(top.dollars)} on {top.date:%Y-%m-%d}")
            if len(buys) > 1:
                factors.append(
                    f"{_plural(len(buys), 'open-market purchase')} by {_plural(data.distinct_buyers, 'insider')} "
                    f"totaling {_money(data.buy_value)}"
                )
        if data.cluster_buy_detected and data.cluster_start is not None and data.cluster_end is not None:
            if data.cluster_start == data.cluster_end:
                span = f"all on {data.cluster_start:%Y-%m-%d}"
            else:
                span = f"{data.cluster_start:%Y-%m-%d} to {data.cluster_end:%Y-%m-%d}"
            factors.append(
                f"Cluster buying: {data.max_buyers_in_cluster_window} insiders bought within 30 days ({span})"
            )
        if data.buys_after_drawdown:
            factors.append(
                f"{data.buys_after_drawdown} of {data.buys_with_drawdown} purchases came after a "
                f"{DRAWDOWN_THRESHOLD:.0%}+ drawdown from the 252-day high "
                f"(average {data.avg_drawdown_at_purchase:.0%} below the high)"
            )
        elif buys and data.avg_drawdown_at_purchase is not None:
            factors.append(f"Purchases came {data.avg_drawdown_at_purchase:.0%} below the 252-day high on average")
        if data.open_market_count and data.net_to_adv is not None:
            factors.append(
                f"Net insider dollars {_money(data.net_buy_value, signed=True)} equal "
                f"{data.net_to_adv:+.2%} of 20-day average dollar volume"
            )
        if data.total_sell_transactions and data.sell_value > data.buy_value:
            factors.append(
                f"Sales dominate ({_money(data.sell_value)} sold vs {_money(data.buy_value)} bought), but sales are a "
                "weak signal: insiders sell for diversification, taxes, and 10b5-1 plans"
            )
        if data.total_buy_transactions and data.buyer_track_evaluated:
            factors.append(
                f"These buyers' past purchases were higher 60 trading days later in "
                f"{data.buyer_track_hits} of {data.buyer_track_evaluated} cases "
                f"(average {data.buyer_track_avg_return:+.1%})"
            )
        if data.open_market_count == 0:
            factors.append(
                f"No open-market purchases or sales in the last {data.lookback_days} days; "
                "absence of buying is weak information"
            )
        if data.ignored_non_open_market_count:
            kinds = ", ".join(
                f"{n} {_IGNORED_LABELS.get(cat, cat.lower())}"
                for cat, n in sorted(data.ignored_by_category.items(), key=lambda kv: -kv[1])
            )
            factors.append(f"Ignored {data.ignored_non_open_market_count} non-open-market rows ({kinds})")
        if not data.has_price_history:
            factors.append("No price history: dollar volume sizing, drawdown timing, and track record were skipped")
        if data.unvalued_open_market_count:
            factors.append(
                f"{_plural(data.unvalued_open_market_count, 'open-market trade')} had no value or price and were not weighted"
            )
        return factors

    @staticmethod
    def _metrics(data: InsiderTradingData) -> list[dict[str, str]]:
        def day(ts: pd.Timestamp | None) -> str:
            return iso_date(ts) or "n/a"

        if data.cluster_buy_detected:
            cluster = (
                f"Yes: {', '.join(data.cluster_insiders)} "
                f"({day(data.cluster_start)} to {day(data.cluster_end)})"
            )
        else:
            cluster = f"No (max {data.max_buyers_in_cluster_window} distinct buyers in any 30-day window)"

        if data.avg_drawdown_at_purchase is not None:
            drawdown = f"{data.avg_drawdown_at_purchase:.1%} below 252-day high"
        elif data.total_buy_transactions:
            drawdown = "n/a (no price history at purchase dates)"
        else:
            drawdown = "n/a (no purchases)"

        if data.buyer_track_evaluated:
            track = (
                f"{data.buyer_track_hits} of {data.buyer_track_evaluated} positive, "
                f"average {data.buyer_track_avg_return:+.1%}"
            )
        elif data.total_buy_transactions:
            track = "Not enough history (needs 60 trading days after a purchase)"
        else:
            track = "n/a (no purchases)"

        ignored = str(data.ignored_non_open_market_count)
        if data.ignored_by_category:
            ignored += " (" + ", ".join(
                f"{n} {_IGNORED_LABELS.get(cat, cat.lower())}" for cat, n in sorted(data.ignored_by_category.items())
            ) + ")"

        return [
            {"metric": "Lookback window", "value": f"{data.lookback_days} days ({day(data.window_start)} to {day(data.reference_date)})"},
            {
                "metric": "Open-market buys (P)",
                "value": f"{data.total_buy_transactions} by {_plural(data.distinct_buyers, 'insider')}, {_money(data.buy_value)}",
            },
            {
                "metric": "Open-market sales (S)",
                "value": f"{data.total_sell_transactions} by {_plural(data.distinct_sellers, 'insider')}, {_money(data.sell_value)}",
            },
            {"metric": "Net insider dollars", "value": _money(data.net_buy_value, signed=True)},
            {"metric": "Net insider shares", "value": f"{data.net_buy_shares:+,}"},
            {"metric": "20-day avg dollar volume", "value": _money(data.avg_dollar_volume_20d)},
            {
                "metric": "Net dollars vs 20-day dollar volume",
                "value": "n/a" if data.net_to_adv is None else f"{data.net_to_adv:+.2%}",
            },
            {"metric": "Cluster buying", "value": cluster},
            {"metric": "Avg drawdown at purchase", "value": drawdown},
            {
                "metric": f"Purchases after {DRAWDOWN_THRESHOLD:.0%}+ drawdown",
                "value": f"{data.buys_after_drawdown} of {data.total_buy_transactions}",
            },
            {"metric": "Buyer track record (60-day forward)", "value": track},
            {"metric": "Non-open-market rows ignored", "value": ignored},
            {"metric": "Most recent open-market trade", "value": day(data.latest_open_market_date) if data.latest_open_market_date is not None else "None in window"},
            {
                "metric": "Data coverage",
                "value": f"{day(data.coverage_start)} to {day(data.coverage_end)} ({data.total_rows} rows)",
            },
        ]

    @staticmethod
    def _table(data: InsiderTradingData) -> dict[str, Any]:
        rows_src = data.transactions[:MAX_TABLE_ROWS]
        has_urls = any(t.url for t in rows_src)
        columns = ["Date", "Insider", "Position", "Type", "Shares", "Value", "Price", "Scored"]
        if has_urls:
            columns.append("Filing")
        rows: list[list[Any]] = []
        for t in rows_src:
            row: list[Any] = [
                f"{t.date:%Y-%m-%d}",
                t.insider,
                t.position or t.role,
                CATEGORY_LABELS.get(t.category, t.category),
                f"{t.shares:,}",
                _money(t.value),
                "n/a" if t.price is None else f"${t.price:,.2f}",
                "Yes" if t.is_open_market else "No",
            ]
            if has_urls:
                row.append(t.url or "")
            rows.append(row)
        return {
            "title": f"Insider transactions, last {data.lookback_days} days (newest first)",
            "columns": columns,
            "rows": rows,
        }

    def _raw_data(
        self,
        data: InsiderTradingData,
        breakdown: list[dict[str, Any]],
        links: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "data_source_label": "SEC Form 4 filings via Yahoo Finance",
            "source_url": links[2]["url"],
            "sec_url": links[0]["url"],
            "links": links,
            "transactions_table": [t.to_dict() for t in data.transactions[:MAX_TRANSACTION_ROWS]],
            "table": self._table(data),
            "metrics_table": self._metrics(data),
            "score_breakdown": breakdown,
            "net_buy_value": round(data.net_buy_value, 2),
            "net_buy_shares": int(data.net_buy_shares),
            "buy_value": round(data.buy_value, 2),
            "sell_value": round(data.sell_value, 2),
            "cluster_buy_detected": bool(data.cluster_buy_detected),
            "cluster_insiders": list(data.cluster_insiders),
            "total_buy_transactions": int(data.total_buy_transactions),
            "total_sell_transactions": int(data.total_sell_transactions),
            "ignored_non_open_market_count": int(data.ignored_non_open_market_count),
            "lookback_days": int(data.lookback_days),
            "reference_date": iso_date(data.reference_date),
            "avg_dollar_volume_20d": None if data.avg_dollar_volume_20d is None else round(data.avg_dollar_volume_20d, 2),
            "avg_drawdown_at_purchase": None if data.avg_drawdown_at_purchase is None else round(data.avg_drawdown_at_purchase, 4),
            "track_record": [r.to_dict() for r in data.track_records],
        }
