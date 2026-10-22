"""Shareholders & Management Domain Agent.

Checks how a company treats its owners and how safely the stock can be traded:

- Share count trend: buying back shares (cannibal companies) vs diluting. The
  strongest metric in the 2011-2026 point-in-time test (IC +0.057, t 3.5).
- Stock pay net of buybacks: (stock-based compensation - buybacks) / market
  cap is scored (t 4.3 in the same test); gross stock pay / revenue is shown
  as information because it scored backwards in large caps (decision_log.md T-04).
  The options/RSU pool (diluted vs basic shares) is still checked.
- Reverse splits: a red flag.
- Management: insider ownership (skin in the game), founder-led, CEO pay
  (shown, not scored: peer comparison needs the DEF 14A), and expectation
  management through the EPS beat record.
- Trading liquidity: low dollar volume means forced-selling risk.

The qualitative parts of the management checklist (track record, values,
interviews, letters) cannot be automated; they are listed with links so they
can be reviewed by hand.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote

import pandas as pd

from agents.base import LIVE, OWNER, PARTIAL, AgentAssessment, BaseAgent
from agents.registry import register_agent
from analysis.fundamental_metrics import OwnerMetrics, compute_owner_metrics, is_excluded_sector
from data.financial_statements import FinancialStatements, fetch_current_statements
from data.fundamentals import FundamentalsData, fetch_fundamentals
from data.fx import to_usd

logger = logging.getLogger(__name__)

FOUNDER_RE = re.compile(r"\bfounder\b", re.IGNORECASE)
TOP_ROLE_RE = re.compile(r"\b(ceo|chief executive|chair(man|woman|person)?|president)\b", re.IGNORECASE)
CEO_RE = re.compile(r"\b(ceo|chief executive)\b", re.IGNORECASE)

REVERSE_SPLIT_RED_FLAG_YEARS = 10
MAX_CONFIDENCE = 85.0


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    size = abs(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= divisor:
            return f"{sign}${size / divisor:.2f}{suffix}"
    return f"{sign}${size:,.0f}"


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f}%" if signed else f"{value * 100:.2f}%"


def _founder_status(officers: list[dict]) -> tuple[str | None, bool]:
    """(founder name and title, founder holds a top role)."""
    for o in officers:
        title = o.get("title") or ""
        if FOUNDER_RE.search(title):
            return f"{o.get('name') or 'Unnamed'} ({title})", bool(TOP_ROLE_RE.search(title))
    return None, False


def _ceo(officers: list[dict]) -> dict | None:
    return next((o for o in officers if CEO_RE.search(o.get("title") or "")), None)


def _last_close(ohlcv: pd.DataFrame | None) -> float | None:
    try:
        if ohlcv is not None and not ohlcv.empty:
            value = float(ohlcv["close"].iloc[-1])
            return value if value > 0 else None
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return None


@register_agent
class ShareholderAgent(BaseAgent):
    """Capital allocation toward shareholders, management alignment and trading liquidity."""

    key = "shareholder"
    name = "Shareholders & Management Agent"
    description = (
        "Buybacks vs dilution (cannibal companies), stock pay net of buybacks and the options/RSU pool, "
        "reverse splits (red flag), insider ownership, founder-led management, EPS beat record, "
        "and trading liquidity (low volume means forced-selling risk). Qualitative management "
        "checks are listed with links for manual review. Owner layer. Not point-in-time."
    )
    category = "Fundamental"
    score_scale = 35.0
    default_weight = 0.55  # share of the owner score (config owner_weights): the strongest signals in the value study
    point_in_time = False
    role = OWNER

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        try:
            data = fetch_fundamentals(ticker)
            if not data.available or (data.quote_type and data.quote_type != "EQUITY"):
                return self.evaluate_with_data(data, FinancialStatements(ticker=ticker, available=False), ohlcv)
            statements = fetch_current_statements(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Shareholder providers raised for %s: %s", ticker, exc)
            return self.unavailable("fundamentals provider error", error=str(exc))
        return self.evaluate_with_data(data, statements, ohlcv)

    def evaluate_with_data(
        self,
        data: FundamentalsData,
        statements: FinancialStatements,
        ohlcv: pd.DataFrame | None = None,
        asof: pd.Timestamp | None = None,
    ) -> AgentAssessment:
        """Score already fetched data; ``asof`` is the evaluation date (default today)."""
        if not data.available:
            return self.unavailable(data.error or "fundamentals unavailable", error=data.error)
        if data.quote_type and data.quote_type != "EQUITY":
            return self.unavailable(f"Shareholder analysis does not apply to {data.quote_type}")

        m = compute_owner_metrics(statements, data, _last_close(ohlcv), ohlcv, today=asof)
        # The liquidity bands are in dollars; the volume is priced in the listing's currency (pence in London).
        m.avg_dollar_volume = to_usd(m.avg_dollar_volume, data.ticker)
        breakdown: list[dict] = []
        notes: list[str] = []

        def add(factor: str, points: float) -> None:
            breakdown.append({"factor": factor, "points": float(points)})

        # 1. Share count: cannibal vs diluter
        if m.share_cagr is not None:
            g = m.share_cagr
            if g <= -0.03:
                pts, desc = 10, "cannibal company, shrinking the share count fast"
            elif g <= -0.01:
                pts, desc = 6, "steady buybacks reduce the share count"
            elif g <= -0.0025:
                pts, desc = 2, "share count drifting lower"
            elif g < 0.01:
                pts, desc = 0, "share count flat"
            elif g < 0.03:
                pts, desc = -5, "diluting shareholders"
            elif g < 0.08:
                pts, desc = -10, "heavy dilution"
            else:
                pts, desc = -15, "serial diluter"
            add(f"Diluted shares {g:+.1%} per year over {m.share_cagr_years:g} years: {desc}", pts)
        elif statements.available:
            notes.append("Share count history not available")

        # 2. Pay in shares, net of buybacks (T-04)
        if m.net_dilution_cost is not None:
            n = m.net_dilution_cost  # (stock pay - buybacks) / market cap; negative = buybacks exceed stock pay
            if n <= -0.03:
                pts, desc = 6, "buybacks exceed stock pay by 3%+ of market cap: real capital returned"
            elif n <= -0.01:
                pts, desc = 4, "buybacks well above stock pay"
            elif n <= 0.0:
                pts, desc = 1, "buybacks cover stock pay"
            elif n < 0.005:
                pts, desc = 0, "stock pay barely offset by buybacks"
            elif n < 0.015:
                pts, desc = -3, "stock pay not offset by buybacks"
            elif n < 0.03:
                pts, desc = -6, "material net dilution cost"
            else:
                pts, desc = -10, "heavy net dilution cost"
            add(f"Stock pay net of buybacks {n:+.1%} of market cap ({_money(m.sbc)} pay vs {_money(m.buybacks or 0.0)} buybacks): {desc}", pts)
        if m.sbc_to_revenue is not None:
            s = m.sbc_to_revenue
            extra = f", {m.sbc_to_fcf:.0%} of FCF" if m.sbc_to_fcf is not None else ""
            if s >= 0.06:
                notes.append(
                    f"Stock-based compensation {s:.1%} of revenue{extra}: high. Shown, not scored: gross stock pay "
                    "scored backwards in the 2011-2026 S&P 500 test; the net-of-buybacks rule is scored instead"
                )
            else:
                notes.append(f"Stock-based compensation {s:.1%} of revenue{extra} (information, not scored)")
        if m.dilutive_gap is not None:
            gap = m.dilutive_gap
            if gap >= 0.05:
                add(f"Diluted shares {gap:.1%} above basic: large pool of options and RSUs", -4)
            elif gap >= 0.02:
                add(f"Diluted shares {gap:.1%} above basic: meaningful options and RSUs outstanding", -1)

        # 3. Reverse splits
        for rs in m.reverse_splits:
            recent = rs["years_ago"] <= REVERSE_SPLIT_RED_FLAG_YEARS
            add(f"Reverse split {rs['label']} on {rs['date']} ({rs['years_ago']:g} years ago){': red flag' if recent else ''}",
                -20 if recent else -5)
        if statements.splits_error:
            notes.append("Split history could not be loaded; reverse splits not checked")

        # 4. Management alignment
        insiders = data.held_percent_insiders
        if insiders is not None:
            pts = 6 if insiders >= 0.15 else 3 if insiders >= 0.05 else 1 if insiders >= 0.01 else 0
            add(f"Insiders own {insiders:.1%} of the shares", pts)
        founder, founder_top = _founder_status(data.officers)
        if founder:
            add(f"Founder-led: {founder}" if founder_top else f"Founder still involved: {founder}", 5 if founder_top else 2)
        ceo = _ceo(data.officers)
        if ceo and ceo.get("total_pay"):
            ratio = ceo["total_pay"] / m.net_income if m.net_income and m.net_income > 0 else None
            notes.append(
                f"CEO {ceo.get('name') or ''} total pay {_money(ceo['total_pay'])}"
                + (f" ({ratio:.3%} of net income)" if ratio is not None else "")
                + ". Compare with peers in the DEF 14A proxy statement."
            )

        graded = [e for e in data.earnings if e.outcome is not None]
        if len(graded) >= 3:
            beats = sum(1 for e in graded if e.outcome == "beat")
            share = beats / len(graded)
            pts = 3 if share >= 0.75 else -3 if share <= 0.25 else 0
            add(f"Beat EPS expectations in {beats} of the last {len(graded)} quarters (expectation management)", pts)

        # 5. Trading liquidity
        if m.avg_dollar_volume is not None:
            adv = m.avg_dollar_volume
            if adv < 1e6:
                pts, desc = -15, "illiquid: hard to exit, forced-selling risk"
            elif adv < 5e6:
                pts, desc = -8, "thin trading: forced-selling risk in a sell-off"
            elif adv < 20e6:
                pts, desc = -3, "moderate liquidity"
            else:
                pts, desc = 0, "liquid"
            add(f"Average daily volume {_money(adv)} ({desc})", pts)
            if m.daily_turnover is not None and m.daily_turnover < 0.0005:
                notes.append(f"Daily turnover {m.daily_turnover:.3%} of the float is very low")

        if not breakdown:
            return self.unavailable("No share count, compensation, ownership or liquidity data")

        score = sum(b["points"] for b in breakdown)
        checks = len(breakdown)
        confidence = min(MAX_CONFIDENCE, 30.0 + 7.0 * checks)
        quality = LIVE if (m.share_cagr is not None and m.avg_dollar_volume is not None and insiders is not None) else PARTIAL

        key_factors = [f"{b['factor']} ({b['points']:+.0f})" for b in sorted(breakdown, key=lambda b: -abs(b["points"])) if b["points"]][:6]
        key_factors += notes
        if statements.basis_note:
            key_factors.append(statements.basis_note)
        if any(rs["years_ago"] <= REVERSE_SPLIT_RED_FLAG_YEARS for rs in m.reverse_splits):
            key_factors.insert(0, "Red flag: reverse split in the last 10 years")

        label = data.short_name or data.ticker
        summary = (
            f"{label}: shares {_pct(m.share_cagr, signed=True)} per year, stock pay {_pct(m.sbc_to_revenue)} of revenue, "
            f"insiders own {_pct(insiders)}, average daily volume {_money(m.avg_dollar_volume)}; score {score:+.0f}."
        )

        t = quote(data.ticker, safe="")
        links = [
            {"label": "DEF 14A proxy (executive pay)", "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type=DEF+14A&dateb=&owner=include&count=40"},
            {"label": "Company profile and officers", "url": f"https://finance.yahoo.com/quote/{t}/profile/"},
            {"label": "Insider transactions (OpenInsider)", "url": f"http://openinsider.com/screener?s={t}"},
            {"label": "10-K (shareholder letter, capital allocation)", "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type=10-K&dateb=&owner=include&count=40"},
        ]

        metrics_table = [
            {"metric": "Diluted shares, change per year", "value": _pct(m.share_cagr, signed=True)},
            {"metric": "Diluted vs basic shares (options/RSU pool)", "value": _pct(m.dilutive_gap)},
            {"metric": "Stock-based compensation", "value": _money(m.sbc)},
            {"metric": "SBC / revenue", "value": _pct(m.sbc_to_revenue)},
            {"metric": "SBC / FCF", "value": _pct(m.sbc_to_fcf)},
            {"metric": "Buybacks (last fiscal year)", "value": _money(m.buybacks)},
            {"metric": "Buyback yield (buybacks / market cap)", "value": _pct(m.buyback_yield)},
            {"metric": "Shares issued (last fiscal year)", "value": _money(m.stock_issuance)},
            {"metric": "Net buyback yield = (buybacks - issuance) / market cap", "value": _pct(m.net_buyback_yield, signed=True)},
            {"metric": "Net dilution cost = (stock pay - buybacks) / market cap (scored)", "value": _pct(m.net_dilution_cost, signed=True)},
            {"metric": "Reverse splits (last 20 years)", "value": ", ".join(f"{r['label']} on {r['date']}" for r in m.reverse_splits) or "None"},
            {"metric": "Insider ownership", "value": _pct(insiders)},
            {"metric": "Founder", "value": founder or "Not listed among officers"},
            {"metric": "CEO total pay", "value": _money(ceo.get("total_pay")) if ceo else "n/a"},
            {"metric": "Average daily dollar volume (20 sessions)", "value": _money(m.avg_dollar_volume)},
            {"metric": "Average daily volume (shares)", "value": "n/a" if m.avg_volume is None else f"{m.avg_volume:,.0f}"},
            {"metric": "Daily turnover (% of float)", "value": _pct(m.daily_turnover)},
        ]
        share_rows = [
            [y.fiscal_year_end, "n/a" if y.basic_shares is None else f"{y.basic_shares / 1e6:,.1f}M",
             "n/a" if y.diluted_shares is None else f"{y.diluted_shares / 1e6:,.1f}M"]
            for y in m.years if y.diluted_shares or y.basic_shares
        ]

        def status(points: float | None) -> str:
            if points is None:
                return "Check manually"
            return "Positive" if points > 0 else "Negative" if points < 0 else "Neutral"

        def points_for(prefix: str) -> float | None:
            hits = [b["points"] for b in breakdown if b["factor"].startswith(prefix)]
            return sum(hits) if hits else None

        checklist = {
            "title": "Management checklist",
            "columns": ["Checklist item", "Automated evidence", "Status"],
            "rows": [
                ["Founder-led", founder or "No founder among listed officers", status(points_for("Founder"))],
                ["Who are they: expertise, track record, values", "Not automatable: read the 10-K, letters and interviews", "Check manually"],
                ["Cash management and value creation", "See ROIC and reinvestment (Fundamentals agent), buybacks and dilution above", status(points_for("Diluted shares"))],
                ["Expectation management (guidance)", next((b["factor"] for b in breakdown if b["factor"].startswith("Beat EPS")), "EPS history unavailable"), status(points_for("Beat EPS"))],
                ["Incentives: salary and stock pay (DEF 14A)", f"SBC {_pct(m.sbc_to_revenue)} of revenue, net of buybacks {_pct(m.net_dilution_cost, signed=True)} of market cap; CEO pay {_money(ceo.get('total_pay')) if ceo else 'n/a'}", status(points_for("Stock pay net"))],
                ["Are they shareholders? Recent moves", f"Insiders own {_pct(insiders)}; recent trades in the Insider agent", status(points_for("Insiders own"))],
                ["News, interviews, calls and letters (avoid promoters)", "Headline tone in the News Sentiment agent", "Check manually"],
            ],
        }

        return self.assessment(
            score=score,
            confidence=confidence,
            summary=summary,
            key_factors=key_factors,
            data_quality=quality,
            data_as_of=m.fiscal_year_end,
            raw_data={
                "metrics_table": metrics_table,
                "score_breakdown": breakdown,
                "table": checklist,
                "extra_tables": [{"title": "Share count by fiscal year", "columns": ["Fiscal year end", "Basic", "Diluted"], "rows": share_rows}],
                "links": links,
                "data_source_label": f"{statements.source} statements, profile and split history",
                "avg_dollar_volume": m.avg_dollar_volume,
                "avg_volume": m.avg_volume,
                "reverse_splits": m.reverse_splits,
                "share_cagr": m.share_cagr,
                "net_dilution_cost": m.net_dilution_cost,
                "sector": data.sector,
                "excluded_sector": is_excluded_sector(data.sector),
                "insider_ownership": insiders,
            },
        )
