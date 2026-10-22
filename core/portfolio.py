"""Portfolio review and candidate assessment, owner-first, by class of holding.

Every holding first gets a class (``classify``): core funds, other funds,
stocks, metals and commodities, derivatives, or no price yet. The owner score
is a framework for companies, so it decides only for stocks; funds, trackers and
derivatives are checked as what they are (cost, overlap, the cap, the plan).

For a stock the review answers three questions:

1. Is the owner thesis intact? The owner score and decision now versus at
   entry (only when that rating was taken within a month of the purchase), the
   owner agents' biggest concerns, and the red flags.
2. How big is the risk? Weight in the portfolio against the concentration
   cap, drawdown from the high since entry against the holding's own usual
   yearly swing, distance to the ATR stop reference, days needed to exit at a
   tenth of the daily volume, and the whole portfolio's volatility, drawdown
   and pairwise correlation.
3. What does the timing panel say? Trend, pullback setup, event risk, VIX.

It then proposes one action per holding (``decide_action``): EXIT (owner SELL or
a red flag), TRIM (above the position cap; broad index funds have none), REVIEW
(thesis moved, thin data, a deep fall, no statements), ADD (owner BUY, room to
grow, pullback setup) or HOLD. A HOLD-band owner score keeps a position
(buy/hold hysteresis: exit only on SELL), which the trading-cost literature
recommends for low-turnover value. The result is one list, "what to do now",
grouped by action, with a plain reason and a fix for each item; what needs
nothing is folded away (``_todo``). Your plan (a core share and a derivatives
budget, ``data.portfolio``) adds its own checks (``_plan_checks``).

For a ticker you do not hold, ``assess_candidate`` gives a verdict (ENTER,
ENTER_STAGED, WATCH, AVOID, NOT_RATED) and a position size: the smallest of a
risk budget at the ATR stop reference, the new-position cap, available cash
and what the stock's liquidity allows, with staging when the VIX is extreme,
the price is extended, the trend is broken or an event is near. For a stock you
hold and rate BUY it sizes a top-up the same way.

Currencies: prices stay in each listing's own currency (the stop reference
too); values, weights, cash and P&L are in the portfolio's currency, converted
at the latest ECB rate (``data.fx``), so a London price in pence or a Toronto
price in CAD adds up correctly with the rest.

Holdings without market data stay in the totals at cost, so the weights of
the others are not overstated; so do imported holdings without a price source
(knock-out products among them). Allocation, risk, correlation, performance,
winners and losers, currency effects and the cycle view of the holdings
together come from ``core.portfolio_analytics``.

Limits live in ``config/settings.py``; the rule constants of the review sit
below. Nothing here is an order: it is a review to act on by hand.
"""

from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from agents.base import UNAVAILABLE
from agents.master_agent import MasterDecisionAgent, TradeRecommendation, average_share_volume
from agents.scoring import NOT_RATED
from core import portfolio_analytics as pa
from core.style_profile import CORE_RE
from data.fetcher import fetch_live_ohlcv
from data.fx import FxError, amount_text, convert_latest, factor_series, quote_currency
from data.history import get_first_after, get_latest, get_previous, record_analyses, record_analysis
from data.portfolio import load_portfolio
from data.price_history import fetch_price_history
from data.profiles import get_profiles

logger = logging.getLogger(__name__)

ACTIONS: tuple[str, ...] = ("EXIT", "TRIM", "REVIEW", "ADD", "HOLD")
VERDICTS: tuple[str, ...] = ("ENTER", "ENTER_STAGED", "WATCH", "AVOID", "NOT_RATED")
BUYS = ("BUY", "STRONG_BUY")
SELLS = ("SELL", "STRONG_SELL")
CONCERN_POINTS = -8.0  # an owner-agent rule at or below this many points is listed as a concern
LIQUIDITY_SHARE_OF_VOLUME = 0.10
THESIS_DROP_POINTS = 20.0
RETURN_WINDOW = 252
REVISION_DAYS = (30, 90)
ENTRY_REF_MAX_DAYS = 31  # a rating counts as "at entry" when taken within a month of the purchase
CORE_TER_LOOK = 0.0030  # core fund cost worth a note (broad world and S&P 500 index ETFs cost about 0.05% to 0.25%)
FUND_OVERLAP_CORRELATION = 0.95  # two core funds this correlated (weekly, two years) hold the same market
DRIFT_BAND_POINTS = 0.05  # core share this far from the plan (or a fifth of the target, if larger) is drift
DRIFT_BAND_RELATIVE = 0.20
MIN_DAYS_TO_EXIT_SHOWN = 1.0
DEEP_FALL = 0.30  # owner BUYs this far below their high had the best next 12 months (2011-2026 panel, all_agents_study)
RULES_VERSION = "D-35"
CLASSES: dict[str, dict[str, str]] = {
    "core": {"label": "Core funds", "bucket": "Core"},
    "fund": {"label": "Other funds", "bucket": "Satellite"},
    "stock": {"label": "Stocks", "bucket": "Satellite"},
    "commodity": {"label": "Metals and commodities", "bucket": "Metals"},
    "derivative": {"label": "Derivatives", "bucket": "Speculative"},
    "unpriced": {"label": "No price yet", "bucket": "Satellite"},
}
BUCKETS = ("Core", "Satellite", "Metals", "Speculative")
CORE_TOP_SECTOR_MAX = 0.60  # a fund whose largest look-through sector is above this is a sector bet, whatever its name
THEME_RE = re.compile(r"\b(clean energy|solar|wind|hydrogen|batter\w*|lithium|uranium|semiconductor\w*|chip\w*|robot\w*|"
                      r"automation|artificial intelligence|ai|cyber\w*|cloud|fintech|blockchain|bitcoin|crypto\w*|gaming|"
                      r"esports|cannabis|biotech\w*|genomic\w*|water|infrastructure|defen[cs]e|aerospace|space|"
                      r"electric vehicle\w*|metaverse|internet|e-commerce|digital)\b", re.I)


def fx_factors(tickers: list[str], currency: str) -> dict[str, tuple[str, float | None]]:
    """Quote currency of each ticker and the factor that converts its price into ``currency`` (None: no rate)."""
    out: dict[str, tuple[str, float | None]] = {}
    for t in tickers:
        ccy = quote_currency(t)
        try:
            out[t] = (ccy, convert_latest(1.0, ccy, currency))
        except FxError:
            out[t] = (ccy, None)
    return out


def limits() -> dict[str, Any]:
    return {
        "max_position_weight": settings.MAX_POSITION_WEIGHT,
        "new_position_weight": settings.NEW_POSITION_WEIGHT,
        "risk_per_position": settings.RISK_PER_POSITION,
        "drawdown_review": settings.DRAWDOWN_REVIEW,
        "max_liquidation_days": settings.MAX_LIQUIDATION_DAYS,
        "cap_exempt": ["core"],  # classes the position cap does not apply to (a broad fund is not one company)
    }


def entry_snapshot(rec: TradeRecommendation) -> dict[str, Any]:
    """What to store with a position when it is opened after an analysis."""
    return {
        "owner_score_at_entry": rec.owner.get("score", rec.composite_score),
        "owner_decision_at_entry": rec.decision,
        "timing_score_at_entry": rec.timing.get("score"),
        "rated_at": rec.generated_at,
    }


# ------------------------------------------------------------------ pieces


def _owner_concerns(rec: TradeRecommendation) -> list[str]:
    """Rules of weighted owner agents that cost many points, plus red-flag warnings."""
    concerns: list[str] = []
    # Only owner agents that carried weight: evidence-only agents (insiders) did not move the decision.
    owner_keys = {c["agent_key"] for c in rec.contributions if c.get("layer") == "owner" and c.get("included")}
    for a in rec.agent_assessments:
        if a.agent_key not in owner_keys or a.data_quality == "UNAVAILABLE":
            continue
        for rule in a.raw_data.get("score_breakdown") or []:
            try:
                points = float(rule.get("points", 0.0))
            except (TypeError, ValueError):
                continue
            if points <= CONCERN_POINTS:
                concerns.append(f"{rule.get('factor')} ({points:+.0f})")
    for w in rec.warnings:
        if w.startswith("Red flag") or w.startswith("Low trading liquidity"):
            concerns.append(w)
    return concerns


def _drawdown_since(ohlcv: pd.DataFrame, opened: str | None) -> dict[str, Any]:
    close = ohlcv["close"]
    if opened:
        since = close.loc[close.index >= pd.Timestamp(opened)]
        window = since if len(since) >= 2 else close.tail(RETURN_WINDOW)
        basis = "since entry" if len(since) >= 2 else "last year (entry date not in history)"
    else:
        window = close.tail(RETURN_WINDOW)
        basis = "last year (no entry date)"
    high = float(window.max())
    latest = float(close.iloc[-1])
    running = window / window.cummax() - 1.0
    return {
        "from_high": round(latest / high - 1.0, 4) if high > 0 else None,
        "high": round(high, 2),
        "max_drawdown": round(float(running.min()), 4),
        "basis": basis,
    }


def _top_sector_share(profile: dict[str, Any]) -> float | None:
    sectors = profile.get("fund_sectors") or {}
    return max(sectors.values()) if sectors else None


def classify(profile: dict[str, Any] | None, asset_class: str | None = None, name: str | None = None,
             priced: bool = True) -> str:
    """Class of a holding (a key of ``CLASSES``): what it is decides which rules apply, first match wins.

    ``profile`` is the instrument profile (``data.profiles``), ``asset_class`` the broker's class from the ledger
    (STOCK, FUND, DERIVATIVE ...), ``priced`` whether a price source exists. A fund is a core fund when its name
    reads as a broad index (the style view's Core bucket), no theme word is in it and no sector dominates its
    look-through; every other fund, however it is named, is a sector or theme bet.
    """
    profile = profile or {}
    if asset_class == "DERIVATIVE":
        return "derivative"
    if not priced:
        return "unpriced"
    kind = profile.get("asset_type")
    if kind in ("Commodity", "Crypto"):
        return "commodity"
    if kind == "Bond":
        return "core"
    if kind in ("ETF", "Fund") or (kind in (None, "Other") and asset_class == "FUND"):
        label = f"{profile.get('name') or ''} {name or ''}"
        top = _top_sector_share(profile)
        broad = CORE_RE.search(label) and not THEME_RE.search(label) and (top is None or top <= CORE_TOP_SECTOR_MAX)
        return "core" if broad else "fund"
    return "stock"


def bucket_of_class(cls: str, profile: dict[str, Any] | None = None) -> str:
    """Core, Satellite, Metals or Speculative (the style view's names); a crypto tracker is Speculative."""
    if cls == "commodity" and (profile or {}).get("asset_type") == "Crypto":
        return "Speculative"
    return CLASSES[cls]["bucket"]


def _own_volatility(ohlcv: pd.DataFrame) -> float | None:
    """Annualized volatility of weekly log returns over the last two years in the listing's currency (None under 26 weeks)."""
    try:
        weekly = ohlcv["close"].dropna().resample("W-FRI").last().dropna()
        returns = np.log(weekly).diff().replace([np.inf, -np.inf], np.nan).dropna().tail(104)
    except (KeyError, TypeError, ValueError):
        return None
    if len(returns) < 26:
        return None
    vol = float(returns.std(ddof=1) * math.sqrt(52.0))
    return vol if math.isfinite(vol) else None


def _drawdown_trigger(volatility: float | None) -> float:
    """Fall from the high that asks for a look: 25%, or one typical year's move of this holding if that is larger."""
    return settings.DRAWDOWN_REVIEW if not volatility else max(settings.DRAWDOWN_REVIEW, 1.0 - math.exp(-volatility))


def _ref_day(at: str | None) -> date | None:
    found = re.search(r"\d{4}-\d{2}-\d{2}", at or "")
    try:
        return date.fromisoformat(found.group()) if found else None
    except ValueError:
        return None


def _month_year(at: str | None) -> str | None:
    day = _ref_day(at)
    return day.strftime("%B %Y") if day else None


def _entry_ref(pos: dict[str, Any]) -> dict[str, Any]:
    """The owner rating at entry and where it comes from: {score, decision, at, kind, days_after_open, valid}.

    ``kind`` is ``stored`` (snapshot saved with the position), ``replay`` (rated_at starts with "replay": the month-end
    before the purchase), ``first_after`` (the first analysis recorded after the opening date, which can be months
    after the purchase) or ``none``. ``valid`` says the rating can stand for the thesis at entry: it was taken within
    ``ENTRY_REF_MAX_DAYS`` of the purchase (a replay always qualifies). A position without an opening date uses a
    stored or replayed snapshot as is and has no ``first_after`` reference.
    """
    opened = pos.get("opened")
    score, decision, at, kind = None, None, None, "none"
    if pos.get("owner_score_at_entry") is not None:
        score, decision, at = pos["owner_score_at_entry"], pos.get("owner_decision_at_entry"), pos.get("rated_at")
        kind = "replay" if str(at or "").startswith("replay") else "stored"
    elif opened:
        first = get_first_after(pos["ticker"], opened)
        if first is not None:
            score = first.get("owner_score") if first.get("owner_score") is not None else first.get("composite_score")
            decision, at, kind = first.get("decision"), first.get("generated_at"), "first_after"
    day = _ref_day(at)
    days = (day - date.fromisoformat(opened)).days if day and opened else None
    if kind == "none":
        valid = False
    elif kind == "replay" or not opened:
        valid = True
    else:
        valid = days is None or abs(days) <= ENTRY_REF_MAX_DAYS
    return {"score": score, "decision": decision, "at": at, "kind": kind, "days_after_open": days, "valid": valid}


def _entry_rating(pos: dict[str, Any]) -> tuple[float | None, str | None, str | None]:
    """Owner score, decision and date of the entry reference (see ``_entry_ref``)."""
    ref = _entry_ref(pos)
    return ref["score"], ref["decision"], ref["at"]


@dataclass(frozen=True)
class Verdict:
    action: str  # EXIT / TRIM / REVIEW / ADD / HOLD
    rule: str  # the rule that fired: owner_sell, red_flag, above_cap, thin_data, thesis_drop, no_statements, drawdown,
    # pullback_add, hold_band, thesis_intact, outside_framework, core_fund, other_fund, tracker
    reasons: list[str]  # reasons[0] is the one plain sentence shown in the list
    fix: str  # what to do, one sentence


def _label(decision: str) -> str:
    return decision.replace("_", " ")


def _share(weight: float) -> str:
    return f"{weight:.1%}"


def _amount(value: float, currency: str = "") -> str:
    return amount_text(value, currency)


def _price_text(value: float, currency: str) -> str:
    """A share price with thousands separators and its currency symbol ($1,676.69)."""
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency)
    return f"{symbol}{value:,.2f}" if symbol else f"{value:,.2f} {currency}".strip()


def _score_text(owner: dict[str, Any], score: float) -> str:
    """The owner score for a sentence: whole points, one decimal when it sits within half a point of a BUY or SELL line."""
    lines = [v for v in (owner.get("thresholds") or {}).values() if isinstance(v, (int, float))]
    return f"{score:+.1f}" if any(abs(score - v) < 0.5 for v in lines) else f"{score:+.0f}"


def _level(owner: dict[str, Any], key: str) -> str:
    """An owner level for a sentence: ``+15``, or empty when the thresholds are not on the record."""
    v = (owner.get("thresholds") or {}).get(key)
    return f"{v:+g}" if isinstance(v, (int, float)) else ""


def _vs_exit(owner: dict[str, Any], shown: str) -> str:
    lvl = _level(owner, "sell")
    return f"Owner score {shown}, at or below the exit level" + (f" ({lvl})" if lvl else "")


def _vs_entry(owner: dict[str, Any], shown: str) -> str:
    lvl = _level(owner, "buy")
    return f"Owner score {shown}, at or above the entry level" + (f" ({lvl})" if lvl else "")


def _excluded_sector(rec: TradeRecommendation) -> bool:
    """Utilities and REITs are outside the framework by the owner's call (T-12): not rated, on purpose."""
    if any((a.raw_data or {}).get("excluded_sector") for a in getattr(rec, "agent_assessments", None) or []):
        return True
    return "T-12" in str((rec.owner or {}).get("not_rated_reason") or "")


def _missing_owner(rec: TradeRecommendation) -> list[str]:
    return [str(c.get("agent_name", "")).removesuffix(" Agent") for c in getattr(rec, "contributions", None) or []
            if c.get("layer") == "owner" and c.get("exclusion_reason") == "No usable data"]


def decide_action(rec: TradeRecommendation, cls: str, weight: float, drawdown: dict[str, Any], entry: dict[str, Any],
                  cash: float, volatility: float | None, expense_ratio: float | None = None, total_value: float = 0.0,
                  currency: str = "") -> Verdict:
    """The review's action for one holding, with the rule behind it, one plain reason and what to do.

    Funds, trackers and unrated stocks are not judged on the owner score (it is a framework for companies): core
    funds have no cap and no owner REVIEW, other funds and commodities keep the cap, and a stock's rules run in this
    order: owner SELL, red flag (EXIT), above the cap (TRIM), thin data, thesis drop, deep fall (REVIEW), a pullback
    in an uptrend (ADD), else HOLD. ``entry`` is ``_entry_ref``; ``total_value`` and ``currency`` only size the fix.
    """
    lim = limits()
    cap, start = lim["max_position_weight"], lim["new_position_weight"]
    decision = rec.decision
    score = rec.owner.get("score", rec.composite_score)
    over = weight > cap
    sell_down = (f"Sell about {_amount((weight - cap) * total_value, currency)} to bring it back to {cap:.0%}."
                 if total_value > 0 else f"Sell down to {cap:.0%}.")
    nothing = "Nothing to do."
    shown = _score_text(rec.owner, score) if isinstance(score, (int, float)) else ""

    if cls == "derivative":  # a knock-out product with a price source is still judged on the budget, not on the owner score
        return Verdict("HOLD", "derivative", ["Knock-out product: judged against your derivatives budget, not the owner score."],
                       "Set a derivatives budget under Your plan.")
    if cls == "core":
        cost = (f"Costs {expense_ratio:.2%} a year." if expense_ratio is not None
                else "Cost not published; check the fund's key information document.")
        return Verdict("HOLD", "core_fund", [f"Broad index fund: no owner score by design. {cost}"], nothing)
    if cls in ("fund", "commodity"):
        if over:
            what = "one commodity" if cls == "commodity" else "a single holding"
            return Verdict("TRIM", "above_cap", [f"{_share(weight)} of the portfolio, above the {cap:.0%} cap for {what}."], sell_down)
        if cls == "fund":
            return Verdict("HOLD", "other_fund", ["Sector or theme fund: judged on cost and fit, not on the owner score."], nothing)
        return Verdict("HOLD", "tracker", ["Tracks one commodity: there is no company to rate, so no owner score. "
                                           f"Kept unless it grows above the {cap:.0%} cap."], nothing)

    # A stock: the owner framework applies when it is rated.
    rated = decision != NOT_RATED
    flags = [w.strip() for w in rec.warnings if w.startswith("Red flag")]
    if rated and decision in SELLS:
        text = f"{_vs_exit(rec.owner, shown)}, so the exit rule says sell."
        if entry.get("valid") and entry.get("decision") in SELLS:
            when = _month_year(entry.get("at"))
            text += " It was already at or below the exit level when you bought" + (f" ({when})." if when else ".")
        return Verdict("EXIT", "owner_sell", [text] + flags,
                       "Sell it, or write down under Your notes why you hold it outside the framework.")
    if flags:
        first = flags[0] if flags[0].endswith((".", ")")) else flags[0] + "."
        return Verdict("EXIT", "red_flag", [f"{first} A red flag is an exit on its own."] + flags[1:],
                       "Read the filing behind it in Full analysis, then sell.")
    if over:
        return Verdict("TRIM", "above_cap", [f"{_share(weight)} of the portfolio, above the {cap:.0%} cap for a single holding."], sell_down)

    dd = drawdown.get("from_high")
    trigger = _drawdown_trigger(volatility)
    since = "since you bought" if drawdown.get("basis") == "since entry" else "over the last year"
    swing = (f"more than its usual yearly swing ({trigger:.0%})" if trigger > settings.DRAWDOWN_REVIEW + 1e-9
             else f"past the {trigger:.0%} look level")
    missing_fix = ("Open Full analysis to see what is missing; a non-US company may need its own listing's statements.")
    recheck = "Re-check the thesis rather than stop out; tight stops hurt value holdings (D-12)."

    if not rated:
        if _excluded_sector(rec):
            if dd is not None and dd <= -trigger:
                return Verdict("REVIEW", "drawdown", [f"Down {abs(dd):.0%} from its high {since}, {swing}, and outside the framework by your choice (T-12)."],
                               "Judge the dividend cover and the debt yourself; a fall alone is not a sell signal.")
            reason = str(rec.owner.get("not_rated_reason") or "")
            head = reason.split(". ")[0].rstrip(".")
            lead = f"{head}." if head else "Outside the framework by your choice (T-12): no owner score."
            return Verdict("HOLD", "outside_framework", [f"{lead} Judge the dividend cover and the debt yourself."], nothing)
        return Verdict("REVIEW", "no_statements", ["No owner rating: the company's statements could not be loaded."], missing_fix)

    coverage = rec.owner.get("coverage", 1.0)
    if coverage < 0.7:
        missing = _missing_owner(rec)
        return Verdict("REVIEW", "thin_data", [f"Only {coverage:.0%} of the owner score had data"
                                               + (f" (missing: {', '.join(missing)})" if missing else "")
                                               + ", so the rating is not reliable yet."], missing_fix)
    change = round(score - entry["score"], 1) if entry.get("valid") and entry.get("score") is not None else None
    if change is not None and change <= -THESIS_DROP_POINTS and decision == "HOLD":
        return Verdict("REVIEW", "thesis_drop", [f"The owner score fell {abs(change):.0f} points since you bought ({entry['score']:+.0f} then, "
                                                 f"{shown} now) and is now between the entry and exit levels."],
                       "Re-check share count, net dilution, FCF and ROIC in Full analysis; this is not a sell signal.")
    if decision == "HOLD" and dd is not None and dd <= -trigger:
        return Verdict("REVIEW", "drawdown", [f"Down {abs(dd):.0%} from its high {since}, {swing}, and the owner score is between the entry and exit levels."], recheck)
    if (decision in BUYS and weight < start and cash > 0 and rec.timing.get("setup") == "pullback_in_uptrend"):
        return Verdict("ADD", "pullback_add", [f"{_vs_entry(rec.owner, shown)}, at {_share(weight)} of the portfolio, with a pullback inside an "
                                               f"uptrend: room to add up to the {start:.0%} starting size."],
                       "Run Should I enter? for the size and staging.")
    if decision == "HOLD":
        if rec.owner.get("capped"):
            ev = rec.owner.get("max_ev_to_fcf")
            cap_text = f"EV/FCF is at or above the {ev:g}x cap" if isinstance(ev, (int, float)) else "its price is above the margin-of-safety cap"
            return Verdict("HOLD", "hold_band", [f"Owner score {shown} reaches the entry level, but {cap_text}: no new money (margin of safety). "
                                                 "Keep it; the exit comes only at or below the exit level."], nothing)
        return Verdict("HOLD", "hold_band", [f"Owner score {shown}, between the entry and exit levels: keep it; the exit comes only at or below the exit level."], nothing)
    if dd is not None and dd <= -max(trigger, DEEP_FALL):
        return Verdict("HOLD", "thesis_intact", [f"{_vs_entry(rec.owner, shown)}, and down {abs(dd):.0%} from its high: in the 2011-2026 test, "
                                                 "stocks at or above the entry level 30% or more below their high had the best next 12 months. Re-read the thesis before adding."], nothing)
    return Verdict("HOLD", "thesis_intact", [f"{_vs_entry(rec.owner, shown)}: the thesis is intact. Re-rate after the next annual report."], nothing)


def _action(
    rec: TradeRecommendation,
    weight: float,
    drawdown: dict[str, Any],
    owner_change: float | None,
    cash: float,
) -> tuple[str, list[str]]:
    """Action and reasons of a stock (kept for callers and tests); ``owner_change`` stands for a valid entry rating."""
    score = rec.owner.get("score", rec.composite_score)
    entry = {"valid": owner_change is not None, "score": None if owner_change is None else score - owner_change}
    verdict = decide_action(rec, "stock", weight, drawdown, entry, cash, None)
    return verdict.action, verdict.reasons


def _room_to_add(rec: TradeRecommendation, cls: str, action: str, weight: float, cash: float) -> dict[str, Any] | None:
    """A rated BUY below the starting size with cash available and no pullback now: information, not an action."""
    start = settings.NEW_POSITION_WEIGHT
    if cls == "stock" and action == "HOLD" and rec.decision in BUYS and weight < start and cash > 0:
        return {"weight": round(weight, 4), "start_weight": start, "cash": round(cash, 2)}
    return None


def _rating_note(cls: str, rec: TradeRecommendation) -> str | None:
    """Why a holding has no owner rating, in plain words (None when the owner score decides)."""
    if cls == "core":
        return "Broad index fund: no owner score by design."
    if cls == "fund":
        return "Sector or theme fund: no owner score by design."
    if cls == "commodity":
        return "Tracks one commodity: no company to rate."
    if rec.decision == NOT_RATED:
        if _excluded_sector(rec):
            head = str(rec.owner.get("not_rated_reason") or "").split(". ")[0].rstrip(".")
            return f"{head}." if head else "Outside the framework by your choice (T-12): no owner score."
        return "No owner rating: the company's statements could not be loaded."
    return None


def _notices(rec: TradeRecommendation, cls: str) -> list[str]:
    """The warnings that matter for a stock's action (``warnings`` stays raw): red flags, price, thin or missing data."""
    if cls != "stock":
        return []
    keep = ("Red flag", "Margin of safety", "Not rated", "Only ", "Low trading liquidity")
    return [w for w in rec.warnings if w.startswith(keep)]


def _position_row(pos: dict[str, Any], rec: TradeRecommendation, ohlcv: pd.DataFrame, total_value: float, cash: float,
                  price_currency: str | None = None, fx: float = 1.0, profile: dict[str, Any] | None = None,
                  asset_class: str | None = None, currency: str = "") -> dict[str, Any]:
    price = float(ohlcv["close"].iloc[-1])  # listing currency
    shares = float(pos["shares"])
    value = shares * price * fx  # portfolio currency, like the average cost
    weight = value / total_value if total_value > 0 else 0.0
    cost_basis = float(pos["avg_cost"]) * shares
    pnl = value - cost_basis
    pnl_pct = value / cost_basis - 1.0 if cost_basis > 0 else 0.0
    opened = pos.get("opened")
    days_held = (date.today() - date.fromisoformat(opened)).days if opened else None
    drawdown = _drawdown_since(ohlcv, opened)
    stop = rec.suggested_stop_loss
    adv_shares = average_share_volume(ohlcv)
    days_to_exit = shares / (LIQUIDITY_SHARE_OF_VOLUME * adv_shares) if adv_shares else None
    entry = _entry_ref(pos)
    owner_score = rec.owner.get("score", rec.composite_score)
    owner_change = round(owner_score - entry["score"], 1) if entry["valid"] and entry["score"] is not None else None
    profile = profile or {}
    cls = classify(profile, asset_class, pos.get("name") or profile.get("name"))
    volatility = _own_volatility(ohlcv)
    expense_ratio = profile.get("expense_ratio") if cls != "stock" else None
    verdict = decide_action(rec, cls, weight, drawdown, entry, cash, volatility, expense_ratio, total_value, currency)
    concerns = _owner_concerns(rec) if cls == "stock" else []

    flags: list[str] = []
    if weight > settings.MAX_POSITION_WEIGHT:
        flags.append("concentration")
    if days_to_exit is not None and days_to_exit > settings.MAX_LIQUIDATION_DAYS:
        flags.append("illiquid")
    if drawdown.get("from_high") is not None and drawdown["from_high"] <= -settings.DRAWDOWN_REVIEW:
        flags.append("drawdown")
    if (rec.timing.get("event_risk") or {}).get("elevated"):
        flags.append("event")
    if rec.timing.get("value_trap_check"):
        flags.append("value trap check")
    if entry["decision"] and entry["decision"] != rec.decision:
        flags.append(f"{entry['decision']} to {rec.decision}")
    if (rec.freshness or {}).get("status") == "STALE":
        flags.append("stale")

    return {
        "ticker": rec.ticker,
        "isin": pos.get("isin"),
        "name": pos.get("name"),
        "source": pos.get("source", "manual"),
        "price_currency": price_currency,
        "fx": round(fx, 6),
        "shares": shares,
        "avg_cost": float(pos["avg_cost"]),
        "opened": opened,
        "days_held": days_held,
        "notes": pos.get("notes") or "",
        "price": round(price, 2),
        "value": round(value, 2),
        "weight": round(weight, 4),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 4),
        "class": cls,
        "class_label": CLASSES[cls]["label"],
        "bucket": bucket_of_class(cls, profile),
        "decision": rec.decision,
        "rated": cls == "stock" and rec.decision != NOT_RATED,
        "rating_note": _rating_note(cls, rec),
        "owner_score": owner_score,
        "owner_levels": {k: v for k, v in (rec.owner.get("thresholds") or {}).items() if k in ("buy", "sell")},
        "owner_confidence": rec.owner.get("confidence"),
        "owner_applicable": rec.owner.get("applicable", True),
        "owner_score_at_entry": entry["score"],
        "owner_decision_at_entry": entry["decision"],
        "rated_at_entry": entry["at"],
        "entry_ref": {k: entry[k] for k in ("kind", "at", "days_after_open", "valid")},
        "owner_change": owner_change,
        "timing_score": rec.timing.get("score"),
        "timing_label": rec.timing.get("label"),
        "trend": rec.timing.get("trend"),
        "setup": rec.timing.get("setup"),
        "momentum_12_1": rec.timing.get("momentum_12_1"),
        "regime": rec.market_regime,
        "risk_rating": rec.risk_rating,
        "risk_score": rec.risk_score,
        "stop_reference": stop,
        "stop_distance_pct": round((price - stop) / price, 4) if price > 0 else None,
        "atr_14": rec.atr_14,
        "drawdown": drawdown,
        "own_volatility": round(volatility, 4) if volatility is not None else None,
        "drawdown_trigger": round(_drawdown_trigger(volatility), 4),
        "avg_daily_shares": round(adv_shares) if adv_shares else None,
        "days_to_exit": round(days_to_exit, 1) if days_to_exit is not None else None,
        "concerns": concerns,
        "warnings": rec.warnings,
        "notices": _notices(rec, cls),
        "timing_notes": rec.timing.get("notes", []),
        "action": verdict.action,
        "rule": verdict.rule,
        "reasons": verdict.reasons,
        "why": verdict.reasons[0],
        "fix": verdict.fix,
        "room_to_add": _room_to_add(rec, cls, verdict.action, weight, cash),
        "checks": [],
        "expense_ratio": expense_ratio,
        "flags": flags,
        "sector": _sector(rec),
        "freshness": rec.freshness,
        "generated_at": rec.generated_at,
    }


def _sector(rec: TradeRecommendation) -> str | None:
    for a in rec.agent_assessments:
        if a.agent_key == "fundamentals" and a.raw_data.get("sector"):
            return str(a.raw_data["sector"])
    return None


def _portfolio_stats(rows: list[dict[str, Any]], cash: float, unpriced_value: float = 0.0,
                     analytics: dict[str, Any] | None = None) -> dict[str, Any]:
    invested = sum(r["value"] for r in rows) + unpriced_value
    total = invested + cash
    weights = {r["ticker"]: (r["value"] / total if total > 0 else 0.0) for r in rows}
    # A broad fund is not one bet and has no cap: concentration is read over single names and non-core funds.
    single = [r for r in rows if r.get("class") != "core"]
    single_invested = sum(r["value"] for r in single) + unpriced_value
    hhi = sum((r["value"] / single_invested) ** 2 for r in single) if single_invested > 0 else 0.0
    rated = lambda pred: round(sum(weights[r["ticker"]] for r in rows if r.get("rated", True) and pred(r["decision"])), 4)  # noqa: E731
    largest = {r["ticker"]: weights[r["ticker"]] for r in single}
    stats: dict[str, Any] = {
        "total_value": round(total, 2),
        "invested_value": round(invested, 2),
        "unpriced_value": round(unpriced_value, 2),
        "cash": round(cash, 2),
        "cash_weight": round(cash / total, 4) if total > 0 else None,
        "positions": len(rows),
        "single_positions": len(single),
        "effective_positions": round(1.0 / hhi, 1) if hhi > 0 else None,
        "effective_positions_basis": "single names and non-core funds",
        "largest_weight": round(max(largest.values()), 4) if largest else None,
        "largest_ticker": max(largest, key=largest.get) if largest else None,
        "unrealized_pnl": round(sum(r["pnl"] for r in rows), 2),
        "cost_basis": round(sum(r["shares"] * r["avg_cost"] for r in rows), 2),
        "owner_buy_weight": rated(lambda d: d in BUYS),
        "owner_sell_weight": rated(lambda d: d in SELLS),
        # Single stocks the framework could not rate; funds and trackers are not rated by design (see "classes").
        "not_rated_weight": round(sum(weights[r["ticker"]] for r in rows
                                      if r.get("class", "stock") == "stock" and r["decision"] == NOT_RATED), 4),
        "actions": {a: sum(1 for r in rows if r["action"] == a) for a in ACTIONS},
        "volatility": None,
        "max_drawdown_1y": None,
        "return_1y": None,
        "avg_correlation": None,
        "max_correlation": None,
        "beta": None,
        "sectors": {},
    }
    if stats["cost_basis"]:
        stats["unrealized_pnl_pct"] = round(stats["unrealized_pnl"] / stats["cost_basis"], 4)
    if analytics:
        risk = analytics.get("risk") or {}
        corr = risk.get("correlation") or {}
        stats.update({
            "volatility": risk.get("volatility"),
            "max_drawdown_1y": (risk.get("max_drawdown_1y") or {}).get("value"),
            "return_1y": risk.get("return_1y"),
            "avg_correlation": corr.get("average"),
            "max_correlation": corr.get("max"),
            "beta": risk.get("beta"),
            "diversification_ratio": risk.get("diversification_ratio"),
            "bad_month": risk.get("bad_month"),
        })
        movers = (analytics.get("movers") or {}).get("totals") or {}
        stats["dividends"] = movers.get("dividends")
        stats["fx_effect"] = movers.get("fx_effect")
        # Sector weights with funds looked through (share of the whole portfolio).
        stats["sectors"] = {g["label"]: g["weight"] for g in (analytics.get("exposures") or {}).get("sector", [])
                            if g["label"] != pa.CASH and g.get("weight")}
    return stats


def _summary_alerts(stats: dict[str, Any], room: list[str] | None = None) -> list[dict[str, Any]]:
    """Portfolio-level alerts, each with what to do about it and where the page shows it ("todo", "plan" or "risk").

    What the to-do list already says is not repeated here: no "rated SELL" (the Sell group has it), no "outside the
    framework" (the class split has it), no "counted at cost" (the Needs a price group has it).
    """
    alerts: list[dict[str, Any]] = []
    lim = limits()
    total = stats.get("total_value") or 0.0
    if stats.get("largest_weight") and stats["largest_weight"] > lim["max_position_weight"]:
        excess = (stats["largest_weight"] - lim["max_position_weight"]) * total
        alerts.append({"kind": "risk", "where": "todo", "ticker": stats["largest_ticker"], "title": "Above the position cap",
                       "text": f"{stats['largest_ticker']} is {stats['largest_weight']:.0%} of the portfolio, above the {lim['max_position_weight']:.0%} cap.",
                       "fix": f"Trim about {excess:,.0f} to bring it back to {lim['max_position_weight']:.0%} (its action says Trim), "
                              "or raise the cap in config/settings.py if you accept the concentration."})
    eff = stats.get("effective_positions")
    singles = stats.get("single_positions", stats["positions"])
    if eff is not None and singles >= 3 and eff < singles * 0.5:
        fix = (f"Send new money to the stocks at or above the entry level and below their starting size: {', '.join(room[:4])}." if room
               else "Send new money to the core fund: it spreads the risk without picking.")
        alerts.append({"kind": "risk", "where": "todo", "ticker": None, "title": "Few real bets",
                       "text": f"Concentration: {singles} single holdings behave like {eff:g} equal-sized ones (core funds not counted).",
                       "fix": fix})
    if stats.get("avg_correlation") is not None and stats["avg_correlation"] >= 0.6:
        pair = stats.get("max_correlation") or {}
        alerts.append({"kind": "risk", "where": "risk", "ticker": None, "title": "Holdings move together",
                       "text": f"Positions move together (average pairwise correlation {stats['avg_correlation']:.2f}"
                               + (f", highest {pair['pair'][0]}/{pair['pair'][1]} at {pair['value']:.2f}" if pair else "")
                               + "): diversification is thinner than the position count suggests.",
                       "fix": "Risk > Correlation shows the pairs; size a correlated pair as one position."})
    for sector, w in list(stats.get("sectors", {}).items())[:1]:
        if w >= 0.5 and stats["positions"] >= 2:
            alerts.append({"kind": "info", "where": "risk", "ticker": None, "title": "One sector dominates",
                           "text": f"{sector} is {w:.0%} of the portfolio (funds looked through).",
                           "fix": "Check that it is intended; Risk > Exposure compares your sectors with the benchmark."})
    if stats.get("cash_weight") is not None and stats["cash_weight"] >= 0.5 and stats["positions"]:
        alerts.append({"kind": "info", "where": "plan", "ticker": None, "title": "Mostly cash",
                       "text": f"Cash is {stats['cash_weight']:.0%} of the portfolio.",
                       "fix": "Your plan's core fund and the stocks with room to add are where new money goes; "
                              "Should I enter? sizes each."})
    return alerts


def _summary_notes(stats: dict[str, Any]) -> list[str]:
    return [a["text"] for a in _summary_alerts(stats)]


# ------------------------------------------------------------ classes, plan and to-do


def _unpriced_bucket(cls: str, asset_class: str | None, name: str | None) -> str:
    """Bucket of a holding without a price: by its broker class (a fund named like an index is Core), else Satellite."""
    if cls == "derivative":
        return "Speculative"
    if asset_class == "FUND" and CORE_RE.search(name or "") and not THEME_RE.search(name or ""):
        return "Core"
    return "Satellite"


def _cost_check(r: dict[str, Any], kind: str) -> dict[str, Any]:
    ter = r.get("expense_ratio")
    if ter is None:
        text = ("Cost not published; check the product's terms." if kind == "derivative"
                else "Cost not published; check the fund's key information document.")
        return {"key": "cost", "label": "Cost", "status": "na", "text": text, "fix": ""}
    if kind == "core" and ter > CORE_TER_LOOK:
        return {"key": "cost", "label": "Cost", "status": "look",
                "text": f"{r['ticker']} costs {ter:.2%} a year; broad world and S&P 500 index ETFs cost about 0.05% to 0.25%.",
                "fix": "For new money, consider a cheaper fund that tracks the same index; switching what you hold pays only "
                       "if the saving beats the trading cost and the tax."}
    note = {"core": "within the usual range for a broad index fund", "fund": "themed funds cost more by design"}.get(kind)
    return {"key": "cost", "label": "Cost", "status": "ok", "text": f"Costs {ter:.2%} a year" + (f"; {note}." if note else "."), "fix": ""}


def _class_checks(rows: list[dict[str, Any]], analytics: dict[str, Any] | None, profiles: dict[str, dict[str, Any]] | None,
                  instruments: dict[str, dict[str, Any]] | None = None, plan: dict[str, Any] | None = None) -> None:
    """Fill each fund, tracker and derivative row's ``checks``: ok, look or n/a, each with a text and a fix."""
    profiles = profiles or {}
    risk = (analytics or {}).get("risk") or {}
    pairs = (risk.get("correlation") or {}).get("high_pairs") or []
    short = set(risk.get("excluded") or [])
    by_ticker = {r["ticker"]: r for r in rows}
    invested = sum(r["value"] for r in rows)
    drift = next((c for c in (plan or {}).get("checks", []) if c["key"] == "drift"), None)
    for r in rows:
        cls = r["class"]
        if cls == "stock":
            continue
        checks = [_cost_check(r, cls)]
        if cls == "core":
            mine = [p for p in pairs if r["ticker"] in p["pair"] and p["value"] is not None
                    and p["value"] >= FUND_OVERLAP_CORRELATION
                    and by_ticker.get(next(t for t in p["pair"] if t != r["ticker"]), {}).get("class") == "core"]
            if r["ticker"] in short or not analytics:
                overlap = {"status": "na", "text": "Under 26 weeks of prices: overlap with your other funds is not checked.", "fix": ""}
            elif mine:
                p = max(mine, key=lambda p: p["value"])
                other = next(t for t in p["pair"] if t != r["ticker"])
                a, b = r, by_ticker[other]
                known = [x for x in (a, b) if x.get("expense_ratio") is not None]
                cheaper = min(known, key=lambda x: x["expense_ratio"]) if len(known) == 2 else None
                overlap = {"status": "look",
                           "text": f"{a['ticker']} and {b['ticker']} moved together at {p['value']:.2f} over two years: the same market twice.",
                           "fix": (f"Send new money to the cheaper one ({cheaper['ticker']}, {cheaper['expense_ratio']:.2%} a year)."
                                   if cheaper else "Send new money to one of them; the costs of both are not published.")}
            else:
                overlap = {"status": "ok", "text": f"No other core fund moved with it (correlation under {FUND_OVERLAP_CORRELATION:.2f}).", "fix": ""}
            checks.append({"key": "overlap", "label": "Overlap", **overlap})
            if drift:
                checks.append({"key": "plan", "label": "Your plan", "status": drift["status"], "text": drift["text"], "fix": drift["fix"]})
            entry = (instruments or {}).get(r.get("isin") or "") if r.get("source") != "manual" else None
            if entry is not None:
                if entry.get("source") == "manual":
                    fit = {"status": "ok", "text": "Price source set by you.", "fix": ""}
                elif entry.get("status") == "check":
                    fit = {"status": "look", "text": "The price source is a weak match for your trade prices.",
                           "fix": "Check the ticker in the import card (Price source of every holding)."}
                else:
                    fit = {"status": "ok", "text": "The price source matches your trade prices.", "fix": ""}
                checks.append({"key": "source", "label": "Price source", **fit})
        elif cls == "fund":
            sectors = (profiles.get(r["ticker"]) or {}).get("fund_sectors") or {}
            if sectors:
                label, share = max(sectors.items(), key=lambda kv: kv[1])
                checks.append({"key": "sector", "label": "Largest sector", "status": "ok", "text": f"{label} is {share:.0%} of the fund.", "fix": ""})
            else:
                checks.append({"key": "sector", "label": "Largest sector", "status": "na", "text": "No sector breakdown for this fund.", "fix": ""})
        elif cls == "commodity":
            share = sum(x["value"] for x in rows if x["class"] == "commodity") / invested if invested > 0 else 0.0
            checks.append({"key": "bucket", "label": "Metals share", "status": "ok",
                           "text": f"Metals and commodities are {share:.0%} of what you have invested.", "fix": ""})
        r["checks"] = checks


def plan_checks(core_share: float | None, invested: float, derivatives: dict[str, Any] | None,
                plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The plan's checks: core share drift (needs a ``core_target``) and the derivatives budget, each with its fix.

    Drift is flagged beyond five points or a fifth of the target, whichever is larger (the usual rebalancing band).
    A pure function of the numbers, so the page can refresh the checks after the plan is saved without a new review.
    """
    plan = plan or {}
    target, budget = plan.get("core_target"), plan.get("derivatives_budget")
    checks: list[dict[str, Any]] = []
    if invested > 0 and core_share is not None:
        drift = core_share - target if target is not None else None
        if target is None:
            checks.append({"key": "drift", "status": "na", "text": "Your plan has no core share yet, so drift is not checked.",
                           "fix": "Set the share you want in core funds under Your plan."})
        else:
            band = max(DRIFT_BAND_POINTS, DRIFT_BAND_RELATIVE * target)
            text = f"Core funds are {core_share:.0%} of what you have invested, against your plan of {target:.0%}."
            if abs(drift) <= band:
                checks.append({"key": "drift", "status": "ok", "text": text.replace("against", "within", 1), "fix": ""})
            elif drift < 0:
                checks.append({"key": "drift", "status": "look", "text": text,
                               "fix": f"Send new money and savings plans to the core fund until it is within {band * 100:.0f} points of the plan; "
                                      "there is no need to sell satellites for it."})
            else:
                checks.append({"key": "drift", "status": "look", "text": text,
                               "fix": "Send new money to stocks at or above the entry level, or pause the core savings plan for a while."})
    if derivatives and (derivatives.get("bought") or derivatives.get("open_count")):
        bought = derivatives.get("bought") or 0.0
        if budget is None:
            checks.append({"key": "budget", "status": "na",
                           "text": "You hold or bought derivatives and have no yearly budget set.",
                           "fix": "Set a derivatives budget under Your plan."})
        elif bought > budget:
            checks.append({"key": "budget", "status": "look",
                           "text": f"You put {bought:,.0f} into derivatives this year, above your budget of {budget:,.0f}.",
                           "fix": "No new derivatives until January, or raise the budget under Your plan."})
        else:
            checks.append({"key": "budget", "status": "ok",
                           "text": f"You put {bought:,.0f} into derivatives this year, within your budget of {budget:,.0f}.", "fix": ""})
    return checks


def refreshed_plan(core_share: float | None, invested: float, plan: dict[str, Any] | None) -> dict[str, Any]:
    """Checks and drift of a just-saved plan against the last review's core share (the derivatives activity is read now)."""
    plan = plan or {}
    target = plan.get("core_target")
    drift = round(core_share - target, 4) if target is not None and core_share is not None and invested > 0 else None
    return {"core_target": target, "derivatives_budget": plan.get("derivatives_budget"), "drift": drift,
            "checks": plan_checks(core_share, invested, _derivatives_activity(date.today().year), plan)}


def _plan_checks(rows: list[dict[str, Any]], unpriced: list[dict[str, Any]], cash: float, total: float,
                 plan: dict[str, Any] | None, derivatives: dict[str, Any] | None) -> dict[str, Any]:
    """Your plan against what you hold: the split by bucket, core share drift and the derivatives budget.

    The split is shares of the invested value (priced at market, unpriced at cost); cash is stated apart.
    """
    plan = plan or {}
    target, budget = plan.get("core_target"), plan.get("derivatives_budget")
    values = dict.fromkeys(BUCKETS, 0.0)
    for r in rows:
        values[r.get("bucket") or "Satellite"] += r["value"]
    for u in unpriced:
        values[u.get("bucket") or "Satellite"] += u["value"]
    invested = sum(values.values())
    split = {b: round(v / invested, 4) if invested > 0 else 0.0 for b, v in values.items()}
    core_share = split["Core"] if invested > 0 else None
    drift = round(core_share - target, 4) if target is not None and core_share is not None else None
    return {"core_target": target, "derivatives_budget": budget, "split": split, "invested_value": round(invested, 2),
            "cash_weight": round(cash / total, 4) if total > 0 else None, "core_share": core_share, "drift": drift,
            "derivatives": derivatives, "checks": plan_checks(core_share, invested, derivatives, plan)}


def _class_summary(rows: list[dict[str, Any]], unpriced: list[dict[str, Any]], total: float) -> list[dict[str, Any]]:
    invested = sum(r["value"] for r in rows) + sum(u["value"] for u in unpriced)
    out = []
    for key, info in CLASSES.items():
        items = [r for r in rows if r["class"] == key] + [u for u in unpriced if u.get("class") == key]
        if not items:
            continue
        value = sum(i["value"] for i in items)
        out.append({"key": key, "label": info["label"], "bucket": info["bucket"], "count": len(items), "value": round(value, 2),
                    "weight": round(value / total, 4) if total > 0 else 0.0,
                    "invested_share": round(value / invested, 4) if invested > 0 else 0.0,
                    "tickers": [i["ticker"] for i in items]})
    return out


def _item(r: dict[str, Any]) -> dict[str, Any]:
    return {"ticker": r["ticker"], "name": r.get("name"), "class": r["class"], "class_label": r["class_label"],
            "action": r["action"], "rule": r["rule"], "why": r["why"], "fix": r["fix"], "weight": r["weight"],
            "value": r["value"], "risk_note": None}


def _unpriced_item(u: dict[str, Any], total: float, currency: str, budget_check: dict[str, Any] | None) -> dict[str, Any]:
    cost = _amount(u["value"], currency)
    if u.get("class") == "derivative":
        why = f"Knock-out product without a public price: counted at cost ({cost})."
        fix = (budget_check or {}).get("fix") or "Set a derivatives budget under Your plan."
        rule = "derivative"
    else:
        why = f"No price source: counted at cost ({cost})." + (f" {u['error']}." if u.get("error") and u.get("source") == "portfolio" else "")
        typed = u.get("source") == "portfolio" and u.get("position_source", "manual") == "manual"
        fix = ("Remove it in Your holdings, or type the ticker Yahoo uses." if typed
               else "Set its ticker in the import card (Price source of every holding), or fix the symbol.")
        rule = "unpriced"
    return {"ticker": u["ticker"], "name": u.get("name"), "class": u.get("class", "unpriced"),
            "class_label": CLASSES[u.get("class", "unpriced")]["label"], "action": None, "rule": rule, "why": why, "fix": fix,
            "weight": round(u["value"] / total, 4) if total > 0 else 0.0, "value": u["value"], "risk_note": None}


def _vs_entry_row(r: dict[str, Any]) -> str:
    score = r.get("owner_score")
    lvl = (r.get("owner_levels") or {}).get("buy")
    shown = f"{score:+.0f}" if isinstance(score, (int, float)) else ""
    return (f"Owner score {shown}, " if shown else "Owner score ") + "at or above the entry level" + (f" ({lvl:+g})" if isinstance(lvl, (int, float)) else "")


def _todo(rows: list[dict[str, Any]], unpriced: list[dict[str, Any]], alerts: list[dict[str, Any]], plan: dict[str, Any],
          total: float = 0.0, currency: str = "") -> list[dict[str, Any]]:
    """What to do now: groups in this order (sell, trim, look, add, price, room, rest), each only when it has items.

    A risk alert about a holding that has an action is attached to its item (``risk_note``) and marked ``attached``.
    """
    start = settings.NEW_POSITION_WEIGHT
    by_weight = lambda items: sorted(items, key=lambda i: -i["weight"])  # noqa: E731
    groups: list[dict[str, Any]] = []

    def add(key: str, label: str, action: str | None, items: list[dict[str, Any]]) -> None:
        if items:
            groups.append({"key": key, "label": label, "action": action, "count": len(items),
                           "weight": round(sum(i["weight"] for i in items), 4), "items": items})

    for key, label, action in (("sell", "Exit", "EXIT"), ("trim", "Trim", "TRIM"), ("look", "Look at", "REVIEW"), ("add", "Add", "ADD")):
        add(key, label, action, by_weight([_item(r) for r in rows if r["action"] == action]))
    budget_check = next((c for c in plan.get("checks", []) if c["key"] == "budget"), None)
    add("price", "Needs a price", None, by_weight([_unpriced_item(u, total, currency, budget_check) for u in unpriced]))
    room = []
    for r in rows:
        if r.get("room_to_add"):
            item = _item(r)
            item.update({"action": None, "rule": "room_to_add",
                         "why": f"{_vs_entry_row(r)} at {_share(r['weight'])}, below the {start:.0%} starting size, with cash available. "
                                "No pullback now, so no Add yet.",
                         "fix": "Should I enter? sizes a top-up if you want to add anyway."})
            room.append(item)
    add("room", "Room to add", None, by_weight(room))
    in_room = {i["ticker"] for i in room}
    resting = [r for r in rows if r["action"] == "HOLD" and r["ticker"] not in in_room]
    order = {k: i for i, k in enumerate(CLASSES)}
    rest = [_item(r) for r in sorted(resting, key=lambda r: (order[r["class"]], -r["weight"]))]
    add("rest", "Nothing to do", None, rest)
    if rest:
        groups[-1]["counts"] = {
            "core": sum(1 for r in resting if r["class"] == "core"),
            "hold_band": sum(1 for r in resting if r["class"] == "stock" and r["decision"] == "HOLD"),
            "buy": sum(1 for r in resting if r["class"] == "stock" and r["decision"] in BUYS),
            "other": sum(1 for r in resting if not (r["class"] == "core" or (r["class"] == "stock" and (r["decision"] == "HOLD" or r["decision"] in BUYS)))),
        }
    # Attach risk alerts to the item they are about.
    items = {i["ticker"]: i for g in groups if g["action"] for i in g["items"]}
    for a in alerts:
        item = items.get(a.get("ticker") or "")
        if item is None or a.get("where") != "todo":
            continue
        if a["title"].endswith("drives the swings"):
            item["risk_note"] = f"Also most of the portfolio's swings: {a['text']}"
            a["attached"] = True
        elif a["title"] == "Above the position cap" and item["action"] == "TRIM":
            a["attached"] = True
    return groups


def _settings_used(master: MasterDecisionAgent) -> dict[str, Any]:
    """The owner settings that decided: weights of the agents that decide (as shares), thresholds, the EV/FCF cap."""
    from agents.base import OWNER
    from agents.registry import get_agent_classes
    from config.weights import has_local_overrides

    cfg = master.config
    roles = {**master.roles(), **cfg.roles}
    deciding = {k: float(w) for k, w in cfg.owner_weights.items() if roles.get(k) == OWNER and w > 0 and cfg.is_enabled(k)}
    total = sum(deciding.values())
    classes = get_agent_classes()
    labels = {k: (classes[k].name if k in classes else k).removesuffix(" Agent").split(" &")[0] for k in deciding}
    return {"owner_weights": {k: round(w / total, 4) for k, w in deciding.items()} if total > 0 else {}, "labels": labels,
            "buy": cfg.owner_thresholds.get("buy"), "sell": cfg.owner_thresholds.get("sell"),
            "cap": cfg.owner_max_ev_to_fcf, "saved": has_local_overrides(), "limits": limits()}


def _freshness(rows: list[dict[str, Any]], reviewed_at: str) -> dict[str, Any]:
    """How old the prices behind the review are: live (today's session), at the last close, or stale."""
    status = lambda r: (r.get("freshness") or {}).get("status")  # noqa: E731
    days = sorted(str((r.get("freshness") or {}).get("as_of"))[:10] for r in rows if (r.get("freshness") or {}).get("as_of"))
    close_days: dict[str, int] = {}
    for r in rows:  # the close day of the holdings priced at a close: live rows are dated later and would misname it
        if status(r) == "CLOSE" and (r.get("freshness") or {}).get("as_of"):
            day = str(r["freshness"]["as_of"])[:10]
            close_days[day] = close_days.get(day, 0) + 1
    return {"reviewed_at": reviewed_at, "oldest_price": days[0] if days else None, "newest_price": days[-1] if days else None,
            "close_days": dict(sorted(close_days.items())),
            "live": sum(1 for r in rows if status(r) == "LIVE"), "close": sum(1 for r in rows if status(r) == "CLOSE"),
            "stale": [r["ticker"] for r in rows if status(r) == "STALE"]}


def _news(rec: TradeRecommendation) -> dict[str, Any]:
    """News tone and the headlines that set it, from the sentiment agent of the analysis."""
    a = next((x for x in rec.agent_assessments if x.agent_key == "sentiment"), None)
    if a is None or a.data_quality == UNAVAILABLE:
        return {"available": False, "reason": (a.summary if a else "") or "No news sentiment for this ticker"}
    heads = [h for h in a.raw_data.get("headlines") or [] if isinstance(h.get("score"), (int, float))]
    pick = lambda h: {k: h.get(k) for k in ("title", "publisher", "url", "published", "score")}  # noqa: E731
    worst = min(heads, key=lambda h: h["score"], default=None)
    best = max(heads, key=lambda h: h["score"], default=None)
    latest = max(heads, key=lambda h: h.get("published") or "", default=None)
    return {
        "available": True, "score": round(a.score, 1), "signal": a.signal, "confidence": round(a.confidence),
        "headlines": len(heads),
        "worst": pick(worst) if worst and worst["score"] < 0 else None,
        "best": pick(best) if best and best["score"] > 0 else None,
        "latest": pick(latest) if latest else None,
    }


def _revisions(ticker: str, score: float | None, now: datetime) -> dict[str, Any]:
    """The owner score now against the last recorded analysis 30 and 90 days ago."""
    out: dict[str, Any] = {}
    for days in REVISION_DAYS:
        prev = get_previous(ticker, (now - pd.Timedelta(days=days)).isoformat(timespec="seconds"))
        if prev is None:
            out[f"d{days}"] = None
            continue
        old = prev.get("owner_score") if prev.get("owner_score") is not None else prev.get("composite_score")
        out[f"d{days}"] = {"score": old, "decision": prev.get("decision"), "at": prev.get("generated_at"),
                           "change": round(score - old, 1) if isinstance(score, (int, float)) and isinstance(old, (int, float)) else None}
    return out


def _macro(recs: list[TradeRecommendation]) -> dict[str, Any] | None:
    for rec in recs:
        a = next((x for x in rec.agent_assessments if x.agent_key == "macro" and x.data_quality != UNAVAILABLE), None)
        if a is not None:
            return {**a.raw_data, "as_of": a.data_as_of}
    return None


def _histories(positions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Trades, dividends and open lots of imported positions, keyed by ticker (empty when nothing is imported)."""
    from data.brokers import SOURCES as BROKER_SOURCES

    imported = {p["isin"]: p["ticker"] for p in positions if p.get("source") in BROKER_SOURCES and p.get("isin")}
    if not imported:
        return {}
    from core.broker_import import position_histories

    try:
        by_isin = position_histories(list(imported))
    except Exception:  # noqa: BLE001 - the review stands without the trade history
        logger.exception("Could not read the imported trade history")
        return {}
    return {imported[isin]: h for isin, h in by_isin.items()}


def _fx_fn(currency: str):
    return lambda ccy, dates: factor_series(ccy, currency, dates)


# ------------------------------------------------------------------ review


def _import_unpriced() -> list[dict[str, Any]]:
    """Imported holdings without a price source (empty when nothing is imported)."""
    from core.broker_import import unpriced_holdings

    try:
        return unpriced_holdings()
    except Exception:  # noqa: BLE001 - the review stands without them
        logger.exception("Could not read the imported holdings without a price")
        return []


def _derivatives_activity(year: int) -> dict[str, Any] | None:
    """Money put into derivatives this calendar year and what is open, from the stored export (None when no activity)."""
    from core.broker_import import derivatives_activity

    try:
        out = derivatives_activity(year)
    except Exception:  # noqa: BLE001 - the plan checks stand without it
        logger.exception("Could not read the derivatives activity")
        return None
    return {**out, "year": year} if out.get("bought") or out.get("sold") or out.get("open_count") else None


def _instrument_map() -> dict[str, dict[str, Any]]:
    from data import instruments

    try:
        return instruments.load_map()
    except Exception:  # noqa: BLE001
        return {}


def _load_profiles(positions: list[dict[str, Any]], histories: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Profiles of the holdings and the benchmark (cached 30 days; a failed lookup falls back to the name)."""
    items = [{"ticker": p["ticker"], "name": p.get("name"), "isin": p.get("isin"),
              "asset_class": (histories.get(p["ticker"]) or {}).get("asset_class")} for p in positions]
    try:
        return get_profiles(items + [{"ticker": settings.PORTFOLIO_BENCHMARK["ticker"]}])
    except Exception:  # noqa: BLE001 - classes fall back to the stock rules
        logger.exception("Could not load the holding profiles")
        return {}


def _unpriced_items(positions: list[dict[str, Any]], good_tickers: set[str], errors: list[dict[str, Any]], currency: str,
                    extra: list[dict[str, Any]], profiles: dict[str, dict[str, Any]],
                    histories: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Holdings without prices, counted at cost: positions with no market data, and imported ones with no price source."""
    out = []
    for p in positions:
        if p["ticker"] in good_tickers:
            continue
        ac = (histories.get(p["ticker"]) or {}).get("asset_class")
        cls = classify(profiles.get(p["ticker"]), ac, p.get("name"), priced=False)
        out.append({"ticker": p["ticker"], "name": p.get("name"), "isin": p.get("isin"), "shares": float(p["shares"]),
                    "avg_cost": float(p["avg_cost"]), "value": round(float(p["shares"]) * float(p["avg_cost"]), 2),
                    "price_currency": currency, "error": next((e["error"] for e in errors if e["ticker"] == p["ticker"]), None),
                    "class": cls, "asset_class": ac, "source": "portfolio", "position_source": p.get("source", "manual"), "bucket": _unpriced_bucket(cls, ac, p.get("name"))})
    for e in extra:
        cls = classify(None, e.get("asset_class"), e.get("name"), priced=False)
        shares = float(e.get("shares") or 0.0)
        out.append({"ticker": e["isin"], "name": e.get("name"), "isin": e["isin"], "shares": shares,
                    "avg_cost": round(float(e["cost"]) / shares, 6) if shares else 0.0, "value": round(float(e["cost"]), 2),
                    "price_currency": currency, "error": e.get("reason"), "class": cls, "asset_class": e.get("asset_class"),
                    "source": "import", "bucket": _unpriced_bucket(cls, e.get("asset_class"), e.get("name"))})
    return out


def _assemble(rows: list[dict[str, Any]], unpriced: list[dict[str, Any]], cash: float, total: float, currency: str,
              analytics: dict[str, Any] | None, plan: dict[str, Any], master: MasterDecisionAgent, reviewed_at: str,
              now: datetime, profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Everything the review adds on top of the rows: checks, stats, alerts, the to-do list, classes, plan, settings."""
    unpriced_value = sum(u["value"] for u in unpriced)
    plan_info = _plan_checks(rows, unpriced, cash, total, plan, _derivatives_activity(now.year))
    instruments = _instrument_map() if any(r.get("isin") and r.get("source") != "manual" for r in rows) else {}
    _class_checks(rows, analytics, profiles, instruments, plan_info)
    order = {a: i for i, a in enumerate(ACTIONS)}
    rows.sort(key=lambda r: (order.get(r["action"], 9), -r["weight"]))
    stats = _portfolio_stats(rows, cash, unpriced_value, analytics)
    room = [r["ticker"] for r in rows if r.get("room_to_add")]
    extra = (analytics or {}).get("alerts") or []
    # The benchmark comparison says more than "one sector dominates"; keep one alert per sector.
    compared = {a["title"].removesuffix(" overweight") for a in extra if a["title"].endswith(" overweight")}
    summary_alerts = _summary_alerts(stats, room)
    alerts = [a for a in summary_alerts if not (a["title"] == "One sector dominates"
                                                 and any(a["text"].startswith(f"{c} ") for c in compared))] + extra
    todo = _todo(rows, unpriced, alerts, plan_info, total, currency)
    return {"rules_version": RULES_VERSION, "settings": _settings_used(master), "freshness": _freshness(rows, reviewed_at),
            "positions": rows, "unpriced": unpriced, "summary": stats, "notes": [a["text"] for a in summary_alerts],
            "alerts": alerts, "analytics": analytics, "todo": todo, "classes": _class_summary(rows, unpriced, total),
            "plan": plan_info}


def review_portfolio(
    portfolio: dict[str, Any] | None = None,
    master: MasterDecisionAgent | None = None,
    record: bool = True,
    fetched: dict[str, tuple[pd.DataFrame, Any]] | None = None,
    max_parallel: int = 4,
    profiles: dict[str, dict[str, Any]] | None = None,
    histories: dict[str, dict[str, Any]] | None = None,
    extra_unpriced: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze every held position with the owner-first system and assess the portfolio as a whole.

    ``fetched``, ``profiles``, ``histories`` and ``extra_unpriced`` can be passed in (tests, or a caller that
    already has them); otherwise prices come from the live fetcher, profiles from ``data.profiles``, the trade
    history of imported positions from the stored export and the imported holdings without a price source
    from ``core.broker_import.unpriced_holdings`` (they join the totals at cost).
    """
    portfolio = portfolio or load_portfolio()
    positions = list(portfolio.get("positions") or [])
    cash = float(portfolio.get("cash") or 0.0)
    currency = portfolio.get("currency", "USD")
    plan = portfolio.get("plan") or {}
    now = datetime.now()
    reviewed_at = now.isoformat(timespec="seconds")
    master = master or MasterDecisionAgent()
    extra_items = extra_unpriced if extra_unpriced is not None else _import_unpriced()
    base = {"reviewed_at": reviewed_at, "cash": cash, "currency": currency, "limits": limits()}
    histories = histories if histories is not None else _histories(positions)
    profiles = profiles if profiles is not None else _load_profiles(positions, histories)
    if not positions:
        unpriced = _unpriced_items([], set(), [], currency, extra_items, profiles, histories)
        total = cash + sum(u["value"] for u in unpriced)
        return {**base, "errors": [], **_assemble([], unpriced, cash, total, currency, None, plan, master, reviewed_at, now, profiles)}

    tickers = [p["ticker"] for p in positions]
    bench_tickers = [b["ticker"] for b in settings.PORTFOLIO_BENCHMARKS]
    if fetched is None:
        fetched = fetch_live_ohlcv(sorted({*tickers, *bench_tickers}))

    def work(pos: dict[str, Any]) -> tuple[dict[str, Any], TradeRecommendation | None, pd.DataFrame | None, str | None]:
        symbol = pos["ticker"]
        if symbol not in fetched:
            return pos, None, None, "No market data"
        ohlcv, freshness = fetched[symbol]
        try:
            rec = master.analyze(symbol, ohlcv, freshness.to_dict() if hasattr(freshness, "to_dict") else freshness)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not sink the review
            logger.exception("Portfolio analysis failed for %s", symbol)
            return pos, None, None, str(exc)
        return pos, rec, ohlcv, None

    with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="portfolio") as pool:
        results = list(pool.map(work, positions))
    if record:  # one transaction for the review instead of one connection and commit per position
        record_analyses([rec for _, rec, _, err in results if not err])

    fx = fx_factors([pos["ticker"] for pos, rec, _, err in results if not err], currency)
    errors = [{"ticker": pos["ticker"], "error": err} for pos, _, _, err in results if err]
    errors += [{"ticker": t, "error": f"No exchange rate from {ccy} to {currency}"} for t, (ccy, f) in fx.items() if f is None]
    good = [(pos, rec, ohlcv) for pos, rec, ohlcv, err in results if not err and fx[pos["ticker"]][1] is not None]
    good_tickers = {pos["ticker"] for pos, _, _ in good}
    # Positions without prices stay in the totals at cost: dropping them would overstate every other weight.
    unpriced = _unpriced_items(positions, good_tickers, errors, currency, extra_items, profiles, histories)
    unpriced_value = sum(u["value"] for u in unpriced)
    invested = sum(float(pos["shares"]) * float(ohlcv["close"].iloc[-1]) * fx[pos["ticker"]][1] for pos, _, ohlcv in good)
    total = invested + unpriced_value + cash
    rows = [_position_row(pos, rec, ohlcv, total, cash, *fx[pos["ticker"]], profile=profiles.get(pos["ticker"]),
                          asset_class=(histories.get(pos["ticker"]) or {}).get("asset_class"), currency=currency)
            for pos, rec, ohlcv in good]
    recs = {pos["ticker"]: rec for pos, rec, _ in good}
    for r in rows:
        r["news"] = _news(recs[r["ticker"]])
        r["revisions"] = _revisions(r["ticker"], r["owner_score"], now)

    analytics = None
    try:
        analytics = _analytics(rows, unpriced, positions, cash, total, currency, fetched, profiles, histories,
                               _macro(list(recs.values())))
    except Exception:  # noqa: BLE001 - the per-position review stands on its own
        logger.exception("Portfolio analytics failed")
    if analytics:
        _attach(rows, analytics)

    return {**base, "errors": errors, **_assemble(rows, unpriced, cash, total, currency, analytics, plan, master, reviewed_at,
                                                  now, profiles)}


def _analytics(rows: list[dict[str, Any]], unpriced: list[dict[str, Any]], positions: list[dict[str, Any]], cash: float,
               total: float, currency: str, fetched: dict[str, tuple[pd.DataFrame, Any]],
               profiles: dict[str, dict[str, Any]] | None, histories: dict[str, dict[str, Any]] | None,
               macro: dict[str, Any] | None) -> dict[str, Any]:
    bench = settings.PORTFOLIO_BENCHMARK
    bench_entry = settings.PORTFOLIO_BENCHMARKS[0]
    histories = histories if histories is not None else _histories(positions)
    if profiles is None:
        by_ticker = {p["ticker"]: p for p in positions}
        items = [{"ticker": t, "name": by_ticker[t].get("name"), "isin": by_ticker[t].get("isin"),
                  "asset_class": (histories.get(t) or {}).get("asset_class")} for t in by_ticker]
        profiles = get_profiles(items + [{"ticker": bench["ticker"]}])
    bench_profile = profiles.get(bench["ticker"]) or {}
    benchmark = None
    if bench["ticker"] in fetched:
        benchmark = {"ticker": bench["ticker"], "label": bench["label"], "close": fetched[bench["ticker"]][0]["close"],
                     "currency": bench_entry.get("currency") or quote_currency(bench["ticker"]),
                     "sectors": bench_profile.get("fund_sectors")}
    benchmarks = [{"id": b["id"], "ticker": b["ticker"], "label": b["label"],
                   "close": fetched[b["ticker"]][0]["close"] if b["ticker"] in fetched else None,
                   "currency": b.get("currency") or quote_currency(b["ticker"])} for b in settings.PORTFOLIO_BENCHMARKS]
    closes = {r["ticker"]: fetched[r["ticker"]][0]["close"] for r in rows}
    factors = dividend_factors([t for t in closes if (histories.get(t) or {}).get("dividends")])
    return pa.analyze(rows, closes, cash, currency, {t: p for t, p in profiles.items() if t != bench["ticker"] or
                                                     t in closes}, _fx_fn(currency), total=total, unpriced=unpriced,
                      benchmark=benchmark, histories=histories, macro=macro, cap=settings.MAX_POSITION_WEIGHT,
                      dividend_factors=factors, benchmarks=benchmarks)


def dividend_factors(tickers: list[str]) -> dict[str, pd.Series]:
    """Split-adjusted close / dividend-adjusted close per ticker over the last 13 months (empty when unavailable).

    The live closes are total-return adjusted; this factor turns them back into the prices that traded, so the
    period P&L of imported holdings can add the dividends actually received without counting them twice.
    """
    if not tickers:
        return {}
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(months=13)).strftime("%Y-%m-%d")
    try:
        frames = fetch_price_history(tickers, start)
    except Exception as exc:  # noqa: BLE001 - without it the dividends stay in the price
        logger.warning("No unadjusted prices for the period P&L: %s", exc)
        return {}
    out: dict[str, pd.Series] = {}
    for t, df in frames.items():
        if {"close", "adj_close"} <= set(df.columns):
            ratio = (df["close"] / df["adj_close"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio):
                out[t] = ratio
    return out


def _attach(rows: list[dict[str, Any]], analytics: dict[str, Any]) -> None:
    """Copy each holding's risk, profile, dividends and currency split onto its review row."""
    risk = (analytics.get("risk") or {}).get("positions") or {}
    since = {m["ticker"]: m for m in (analytics.get("movers") or {}).get("since_entry", [])}
    profiles = analytics.get("profiles") or {}
    for r in rows:
        r["risk"] = risk.get(r["ticker"])
        m = since.get(r["ticker"]) or {}
        r["dividends"] = m.get("dividends", 0.0)
        r["total_pnl"] = m.get("total_pnl", r["pnl"])
        r["currency_split"] = m.get("currency_split")
        prof = profiles.get(r["ticker"]) or {}
        for key in ("asset_type", "industry", "country", "region"):
            r[key] = prof.get(key)
        if not r.get("sector"):
            r["sector"] = prof.get("sector")


# --------------------------------------------------------------- candidate


def _size_position(rec: TradeRecommendation, total: float, cash: float, ohlcv: pd.DataFrame,
                   price_currency: str = "USD", fx: float = 1.0, held_value: float = 0.0) -> dict[str, Any]:
    """Shares to buy. ``total``, ``cash`` and ``held_value`` (what you already hold of it, for a top-up) are in the
    portfolio currency; prices in the listing's (``fx`` converts)."""
    price = rec.suggested_entry
    stop = rec.suggested_stop_loss
    lim = limits()
    total, cash, held_value = total / fx, cash / fx, held_value / fx  # into the listing currency for the share arithmetic
    if total <= 0 or price <= 0:
        return {"shares": None, "value": None, "weight": None, "binding": None,
                "note": "Add cash or positions to the portfolio to size a position."}
    risk_per_share = max(price - stop, 0.01)
    candidates = {
        "risk budget": (lim["risk_per_position"] * total) / risk_per_share,
        "position cap": max(lim["new_position_weight"] * total - held_value, 0.0) / price,
    }
    if cash > 0:
        candidates["available cash"] = cash / price
    adv = average_share_volume(ohlcv)
    if adv:
        candidates["liquidity"] = lim["max_liquidation_days"] * LIQUIDITY_SHARE_OF_VOLUME * adv
    binding = min(candidates, key=candidates.get)
    shares = math.floor(candidates[binding])
    value = shares * price
    detail = {k: math.floor(v) for k, v in candidates.items()}
    note = (
        f"Risk {lim['risk_per_position']:.0%} of the portfolio between entry {_price_text(price, price_currency)} and the ATR stop reference {_price_text(stop, price_currency)} "
        f"allows {detail['risk budget']:,} shares; the {lim['new_position_weight']:.0%} new-position cap allows {detail['position cap']:,}"
        + (f"; cash allows {detail['available cash']:,}" if "available cash" in detail else "; no cash recorded")
        + (f"; exiting within {lim['max_liquidation_days']:g} days at {LIQUIDITY_SHARE_OF_VOLUME:.0%} of daily volume allows {detail['liquidity']:,}" if "liquidity" in detail else "")
        + f". Binding constraint: {binding}."
    )
    return {
        "shares": shares,
        "value": round(value * fx, 2),  # portfolio currency
        "weight": round(value / total, 4),
        "risk_amount": round(shares * risk_per_share * fx, 2),
        "binding": binding,
        "constraints": detail,
        "cash_short": round(max(value - cash, 0.0) * fx, 2) if cash >= 0 else None,
        "price_currency": price_currency,
        "note": note,
    }


def _verdict(rec: TradeRecommendation) -> tuple[str, list[str], dict[str, Any]]:
    decision = rec.decision
    score = rec.owner.get("score", rec.composite_score)
    timing = rec.timing
    reasons: list[str] = []
    staging: dict[str, Any] = {"tranches": 1, "over_weeks": 0, "why": []}

    if decision == NOT_RATED:
        reason = rec.owner.get("not_rated_reason") or (
            "No owner rating: the value framework needs company statements (funds and indexes are not rated)."
        )
        return "NOT_RATED", [reason], staging
    if decision in SELLS:
        return "AVOID", [f"{_vs_exit(rec.owner, f'{score:+.0f}')}. Not a business to own on this framework."], staging
    if decision == "HOLD":
        if rec.owner.get("capped"):
            return "WATCH", [
                f"Owner score {_score_text(rec.owner, score)} reaches the entry level, but the price is above the margin-of-safety cap: no new money. "
                "Keep it on the watchlist and re-rate after the next 10-K."
            ], staging
        return "WATCH", [
            f"Owner score {_score_text(rec.owner, score)}, between the entry and exit levels: not cheap or well-run enough to buy now. Keep it on the watchlist and re-rate after the next 10-K."
        ], staging

    reasons.append(f"{_vs_entry(rec.owner, f'{score:+.0f}')}, on a 12-month view ({rec.owner.get('confidence', 0):.0f}% confidence).")
    if (timing.get("vix") or {}).get("extreme"):
        staging["why"].append("VIX at or above 40: best entries historically, but the market fell a further 24-32% after the signal in 2008 and 2020")
    if timing.get("setup") == "extended":
        staging["why"].append("price extended above its 20-day mean")
    if timing.get("trend") == "DOWN":
        staging["why"].append("trend broken (below the 200-day SMA)")
    if (timing.get("event_risk") or {}).get("elevated"):
        staging["why"].append("earnings or an options-priced event is near")
    if timing.get("value_trap_check"):
        staging["why"].append("12-1 month momentum deeply negative (value-trap check: weak next month, strong next year in the test)")
    if staging["why"]:
        staging.update({"tranches": 3 if (timing.get("vix") or {}).get("extreme") or timing.get("trend") == "DOWN" else 2,
                        "over_weeks": 4 if (timing.get("vix") or {}).get("extreme") else 3})
        reasons.append("Stage the entry: " + "; ".join(staging["why"]) + ".")
        return "ENTER_STAGED", reasons, staging
    if timing.get("setup") == "pullback_in_uptrend":
        reasons.append("Timing: pullback inside an uptrend, a reasonable entry point.")
    else:
        ts = timing.get("score")
        reasons.append("Entry timing" + (f" {ts:+.0f}" if isinstance(ts, (int, float)) else "") + ": nothing argues for waiting; the owner score decides.")
    return "ENTER", reasons, staging


def assess_candidate(
    ticker: str,
    portfolio: dict[str, Any] | None = None,
    master: MasterDecisionAgent | None = None,
    record: bool = True,
    fetched: dict[str, tuple[pd.DataFrame, Any]] | None = None,
    profiles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Should I enter this ticker, or if I hold it, what is the risk of the position?"""
    symbol = ticker.strip().upper()
    portfolio = portfolio or load_portfolio()
    cash = float(portfolio.get("cash") or 0.0)
    positions = list(portfolio.get("positions") or [])
    held = next((p for p in positions if p["ticker"] == symbol), None)
    tickers = sorted({symbol, *[p["ticker"] for p in positions]})
    fetched = fetched if fetched is not None else fetch_live_ohlcv(tickers)
    if symbol not in fetched:
        raise LookupError(f"No market data available for {symbol}")
    master = master or MasterDecisionAgent()

    ohlcv, freshness = fetched[symbol]
    rec = master.analyze(symbol, ohlcv, freshness.to_dict() if hasattr(freshness, "to_dict") else freshness)
    if record:
        record_analysis(rec)

    currency = portfolio.get("currency", "USD")
    fx = fx_factors(sorted({symbol, *[p["ticker"] for p in positions if p["ticker"] in fetched]}), currency)
    invested = 0.0
    for pos in positions:
        factor = fx.get(pos["ticker"], (None, None))[1]
        if pos["ticker"] in fetched and factor is not None:
            invested += float(pos["shares"]) * float(fetched[pos["ticker"]][0]["close"].iloc[-1]) * factor
        else:
            invested += float(pos["shares"]) * float(pos["avg_cost"])
    total = invested + cash
    price_currency, factor = fx[symbol]
    if factor is None:
        raise LookupError(f"No exchange rate from {price_currency} to {currency} for {symbol}")

    histories = _histories(positions)
    if profiles is None:
        items = [{"ticker": symbol}] + [{"ticker": p["ticker"], "name": p.get("name"), "isin": p.get("isin"),
                                         "asset_class": (histories.get(p["ticker"]) or {}).get("asset_class")} for p in positions]
        try:
            profiles = get_profiles(items)
        except Exception:  # noqa: BLE001 - the verdict stands without profiles
            logger.exception("Could not load the profiles for %s", symbol)
            profiles = {}
    profile = profiles.get(symbol) or {}
    cls = classify(profile, (histories.get(symbol) or {}).get("asset_class"), (held or {}).get("name") or profile.get("name"))
    base = {
        "ticker": symbol,
        "currency": currency,
        "price_currency": price_currency,
        "held": held is not None,
        "assessed_at": datetime.now().isoformat(timespec="seconds"),
        "portfolio_value": round(total, 2),
        "cash": cash,
        "limits": limits(),
        "class": cls,
        "class_label": CLASSES[cls]["label"],
        "analysis": rec.to_dict(),
        "current_price": rec.suggested_entry,
        "entry_snapshot": entry_snapshot(rec),
    }
    if held is not None:
        row = _position_row(held, rec, ohlcv, total, cash, price_currency, factor, profile=profile,
                            asset_class=(histories.get(symbol) or {}).get("asset_class"), currency=currency)
        if row["class"] != "stock":
            row["checks"] = [_cost_check(row, row["class"])]
        top_up = None
        if row["class"] == "stock" and rec.decision in BUYS and (row["room_to_add"] or row["action"] == "ADD"):
            top_up = _size_position(rec, total, cash, ohlcv, price_currency, factor, held_value=row["value"])
        return {**base, "position": row, "verdict": row["action"], "reasons": row["reasons"], "checks": row["checks"],
                "top_up": top_up}

    if cls != "stock":
        fit = _candidate_fit(symbol, positions, fetched, fx, cash, currency, price_currency, None, profiles)
        reasons, checks = _fund_candidate(symbol, cls, profile, fit, positions, profiles, histories)
        return {**base, "position": None, "verdict": "NOT_RATED", "reasons": reasons, "staging": {"tranches": 1, "over_weeks": 0, "why": []},
                "sizing": None, "concerns": [], "checks": checks, "top_up": None, "timing_notes": rec.timing.get("notes", []),
                "previous": get_previous(symbol, rec.generated_at), "fit": fit}

    verdict, reasons, staging = _verdict(rec)
    sizing = (_size_position(rec, total, cash, ohlcv, price_currency, factor)
              if verdict in ("ENTER", "ENTER_STAGED") else None)
    concerns = _owner_concerns(rec)
    return {
        **base,
        "position": None,
        "verdict": verdict,
        "reasons": reasons,
        "staging": staging,
        "sizing": sizing,
        "concerns": concerns,
        "checks": [],
        "top_up": None,
        "timing_notes": rec.timing.get("notes", []),
        "previous": get_previous(symbol, rec.generated_at),
        "fit": _candidate_fit(symbol, positions, fetched, fx, cash, currency, price_currency, sizing, profiles),
    }


def _fund_candidate(symbol: str, cls: str, profile: dict[str, Any], fit: dict[str, Any] | None, positions: list[dict[str, Any]],
                    profiles: dict[str, dict[str, Any]], histories: dict[str, dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """The class sentence and checks of a fund or tracker that is not held: cost, and the held core fund it moves with."""
    cost = _cost_check({"ticker": symbol, "expense_ratio": profile.get("expense_ratio")}, cls)
    checks = [cost]
    if cls == "core":
        held_core = {p["ticker"] for p in positions
                     if classify(profiles.get(p["ticker"]), (histories.get(p["ticker"]) or {}).get("asset_class"), p.get("name")) == "core"}
        twin = next((m for m in (fit or {}).get("most_similar") or [] if m["ticker"] in held_core and (m.get("correlation") or 0) >= pa.HIGH_CORRELATION), None)
        text = "Broad index fund: no owner score by design."
        if twin:
            text += f" It moves with your {twin['ticker']} at {twin['correlation']:.2f}: the same market you already hold."
            checks.append({"key": "overlap", "label": "Overlap", "status": "look" if twin["correlation"] >= FUND_OVERLAP_CORRELATION else "ok",
                           "text": f"It moves with {twin['ticker']} at {twin['correlation']:.2f} over two years.",
                           "fix": "New money buys more of the same market; compare costs before adding." if twin["correlation"] >= FUND_OVERLAP_CORRELATION else ""})
        elif held_core:
            checks.append({"key": "overlap", "label": "Overlap", "status": "ok", "text": "No core fund you hold moves with it as one.", "fix": ""})
        return [text], checks
    if cls == "fund":
        return ["Sector or theme fund: judged on cost and fit, not on the owner score."], checks
    return ["Tracks one commodity: there is no company to rate, so no owner score."], checks


def _candidate_fit(symbol: str, positions: list[dict[str, Any]], fetched: dict[str, tuple[pd.DataFrame, Any]],
                   fx: dict[str, tuple[str, float | None]], cash: float, currency: str, price_currency: str,
                   sizing: dict[str, Any] | None, profiles: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    """Correlation with the portfolio, overlap, and volatility before and after adding the suggested size."""
    try:
        holdings = [{"ticker": p["ticker"], "price_currency": fx[p["ticker"]][0],
                     "value": float(p["shares"]) * float(fetched[p["ticker"]][0]["close"].iloc[-1]) * fx[p["ticker"]][1]}
                    for p in positions if p["ticker"] in fetched and (fx.get(p["ticker"]) or (None, None))[1] is not None]
        closes = {t: fetched[t][0]["close"] for t in [h["ticker"] for h in holdings] + [symbol]}
        if profiles is None:
            profiles = get_profiles([{"ticker": symbol}] + [{"ticker": p["ticker"], "name": p.get("name"), "isin": p.get("isin")}
                                                            for p in positions])
        return pa.candidate_fit(symbol, holdings, closes, cash, currency, profiles, _fx_fn(currency),
                                (sizing or {}).get("value"), price_currency)
    except Exception:  # noqa: BLE001 - the verdict stands without it
        logger.exception("Candidate fit failed for %s", symbol)
        return None
