"""Your history against the weights: do the conditions that made your entries good match what the agents reward?

Built on the tagged entries of ``core.style_profile`` (nothing new is measured about your buys except one tag, the
market regime at entry, computed point in time here). Five parts:

- **Map**: each entry tag is matched to the agent that reads the same condition, with the direction that agent
  rewards (read from the agent's code, listed in ``LINES`` below). Volatility of the name, order size, build phase and
  positions open are style, not agents.
- **Agents**: for each agent its live weight (the owner layer, or each regime's timing profile), what it rewards, what
  your history says about that direction (rewarded buckets against penalised buckets, with the track record's
  statistics), what the S&P 500 panel says (``config/panel_evidence.json``), and a verdict.
- **By regime**: your buys grouped by the market regime the day before, and inside each regime the buckets that did
  best and worst where the sample allows.
- **Now**: recommendations for today's regime, each labelled by its evidence. They are about how to act within the
  current rules; they never change a weight. A history that argues for a weight change names the replay that would
  test it (the protocol of decision_log D-30 and D-32).
- **Panel evidence** is market numbers only and is read generically, so a rerun of a study only needs the JSON updated.

Nothing here changes FastValue's scoring (the evidence rule: one investor's history is a hypothesis generator).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import config.settings as settings
from config.weights import REGIMES
from core.style_profile import _median, _rank_tags
from core.track_record import MAIN_HORIZON, MIN_CASES, _compare, _confidence, _pattern, _same_sign

logger = logging.getLogger(__name__)

PANEL_PATH = Path(__file__).resolve().parent.parent / "config" / "panel_evidence.json"
REGIME_TAG = "market regime at entry"
VALUE = f"excess_{MAIN_HORIZON}"
MIN_REGIME_BUYS = 10  # fewer buys than this in a regime and its inside view says "too few"
MAX_RECOMMENDATIONS = 6
SIGNIFICANT_T = 2.0  # the protocol's bar for a weight change (D-30, D-32)
OWNER = "owner"

# What the owner's tags are called when they are style and not an agent's reading.
STYLE_TAGS = {
    "volatility of the name": "Style: how steady the name is. No agent reads it; position sizing does (volatility-based risk, 1% per position).",
    "kind of buy": "Style: whether you add to a winner or a loser, or start a position. No agent reads it.",
    "savings plan or one-off": "Style: how the order came about. No agent reads it.",
    "size of the order": "Style: how big the order is against your median. Sizing is a portfolio rule, not an agent.",
    "when in the build": "Style: how long the position had been building. No agent reads it.",
    "positions open at the time": "Style: how many positions you held. The portfolio limits cover it, no agent reads it.",
    "market drawdown": "The macro agent reads the market's trend and the VIX, not its distance from the high, so no agent reads this directly.",
}

# (agent key, tag, what the agent reads, {bucket: +1 rewarded / -1 penalised}, factor sign for the panel's IC or None).
# Directions come from the agents' code (agents/pullback_agent.py, momentum_agent.py, regime_agent.py,
# data/macro_data.py); buckets not listed are neither rewarded nor penalised.
LINES: list[dict[str, Any]] = [
    {"agent": OWNER, "tag": "FastValue owner rating at entry",
     "reads": "The owner score rates the business (fundamentals and shareholder returns): entry level at +15, exit level at -15.",
     "rewards": {"BUY": 1, "SELL": -1}},
    {"agent": "pullback", "tag": "distance from the 52-week high",
     "reads": "Price below its 20-day mean (1 to 3 ATR) inside an uptrend scores +25; a name stretched above that mean scores -25. "
              "A name at its high is stretched, one a little below it is not.",
     "rewards": {"5 to 20% below the high": 1, "within 5% of the 52-week high": -1}},
    {"agent": "momentum", "tag": "distance from the 52-week high",
     "reads": "Within 3% of the 52-week high above the 50-day mean scores +5; more than 30% below it scores -5.",
     "rewards": {"within 5% of the 52-week high": 1, "40%+ below the high": -1}},
    {"agent": "pullback", "tag": "move in the month before",
     "reads": "A fall inside an uptrend is the setup it rewards (+5 to +25); a sharp rise leaves the name stretched (-25).",
     "rewards": {"after a 10%+ drop": 1, "after a 10%+ rise": -1}, "factor_sign": -1},
    {"agent": "momentum", "tag": "move in the month before",
     "reads": "A 20-day gain above 5% scores +15, a fall below -5% scores -15 (it follows the trend).",
     "rewards": {"after a 10%+ rise": 1, "after a 10%+ drop": -1}, "factor_sign": 1},
    {"agent": "momentum", "tag": "RSI(14)",
     "reads": "RSI(14) under 30 scores +8 to +14 (bounce potential), above 75 scores -12 (extended): it is contrarian here.",
     "rewards": {"oversold (under 35)": 1, "overbought (over 65)": -1}},
    {"agent": "pullback", "tag": "RSI(14)",
     "reads": "It reads RSI(3), a faster cousin: washed out under 20 adds +15, above 90 marks the name extended (-25).",
     "rewards": {"oversold (under 35)": 1, "overbought (over 65)": -1}},
    {"agent": "momentum", "tag": "trend",
     "reads": "Price above the 50-day above the 200-day mean scores +25; below both scores -25.",
     "rewards": {"above its 200-day mean": 1, "below its 200-day mean": -1}},
    {"agent": "macro", "tag": "market trend",
     "reads": "The S&P 500 above its 200-day mean scores +2 (below, -2) and above its 50-day mean +1 (below, -1).",
     "rewards": {"market above its 200-day mean": 1, "market below its 200-day mean": -1}},
    {"agent": "macro", "tag": "VIX", "vix": True,
     "reads": "With the contrarian VIX setting, fear is a discount: under 13 scores -1, 20 to 40 about 0 to +0.5, 40 or more +6.",
     "rewards": {"VIX under 20": -1, "VIX 30 or more": 1}},
    {"agent": "macro", "tag": REGIME_TAG, "derived": True,
     "reads": "The regime picks which timing profile applies. The agent's own score by regime, measured month by month, says which regimes it favours.",
     "rewards": {}},
]

# How to act on a tag, by the kind of condition. The wording names the buckets; it never touches a weight.
_BUCKET_WORDS = {"BUY": "at or above the entry level", "HOLD": "between the levels", "SELL": "at or below the exit level"}
FAMILY_ACTION = {
    "entry": "Use it as a filter before you buy: prefer names tagged '{best}', and treat '{worst}' as a reason to stage the buy in tranches rather than all at once.",
    "pace": "Use it to pace buys: lean in when the market is '{best}', go slower and stage the buy when it is '{worst}'.",
    "size": "Use it to size: keep orders in the '{best}' band rather than the '{worst}' one.",
    "gate": "Use it as a gate: buys with an owner score '{best}' did better than those '{worst}'; check the score in the Analyze tab first.",
}
TAG_FAMILY = {
    "distance from the 52-week high": "entry", "trend": "entry", "move in the month before": "entry", "RSI(14)": "entry",
    "market trend": "pace", REGIME_TAG: "pace", "VIX": "pace", "market drawdown": "pace",
    "size of the order": "size", "when in the build": "size", "positions open at the time": "size", "volatility of the name": "size",
    "kind of buy": "size", "savings plan or one-off": "size", "FastValue owner rating at entry": "gate",
}

# ================================================================== panel evidence


def validate_panel_evidence(doc: dict[str, Any]) -> dict[str, Any]:
    """Check the shape of config/panel_evidence.json. Raises ValueError with the first problem."""
    if not isinstance(doc, dict) or not isinstance(doc.get("version"), int):
        raise ValueError("panel evidence needs an integer version")
    studies = doc.get("studies")
    if not isinstance(studies, dict) or not studies:
        raise ValueError("panel evidence needs a studies block")
    for key, s in studies.items():
        for field in ("name", "period", "date"):
            if not isinstance(s.get(field), str) or not s[field]:
                raise ValueError(f"study {key!r} needs {field}")

    def check_block(name: str, block: dict[str, Any]) -> None:
        if block.get("study") not in studies:
            raise ValueError(f"{name}: unknown study {block.get('study')!r}")
        if block.get("kind") in ("factor", "note"):
            if block["kind"] == "factor" and not isinstance(block.get("ic"), (int, float)):
                raise ValueError(f"{name}: a factor needs a numeric ic")
            return
        buckets = block.get("buckets")
        if not isinstance(buckets, list) or not buckets:
            raise ValueError(f"{name}: needs buckets")
        for b in buckets:
            if not isinstance(b.get("label"), str) or not isinstance(b.get("tag_buckets"), list):
                raise ValueError(f"{name}: each bucket needs a label and tag_buckets")
            if not isinstance(b.get("value"), (int, float)):
                raise ValueError(f"{name}: bucket {b.get('label')!r} needs a numeric value")
            if b.get("alt") is not None and not isinstance(b["alt"], (int, float)):
                raise ValueError(f"{name}: bucket {b.get('label')!r} alt must be a number")

    for tag, block in (doc.get("conditions") or {}).items():
        check_block(f"condition {tag!r}", block)
    if "by_regime" in doc:
        check_block("by_regime", doc["by_regime"])
        for regime, conds in (doc["by_regime"].get("inside_regime") or {}).items():
            if regime not in REGIMES:
                raise ValueError(f"inside_regime: unknown regime {regime!r}")
            for tag, block in conds.items():
                check_block(f"inside_regime {regime} {tag!r}", {**block, "study": block.get("study", doc["by_regime"]["study"])})
    for key, a in (doc.get("agents") or {}).items():
        if a.get("study") not in studies:
            raise ValueError(f"agent {key!r}: unknown study {a.get('study')!r}")
    return doc


def load_panel_evidence(path: Path | None = None) -> dict[str, Any]:
    return validate_panel_evidence(json.loads((path or PANEL_PATH).read_text()))


# ================================================================== macro regime at each buy


def _load_history(years_for: pd.Timestamp):
    from data.macro_data import history_years_needed, load_macro_history

    return load_macro_history(history_years_needed(years_for))


def _usable(snap: Any) -> bool:
    return snap.data_quality != "ESTIMATED" and "spy_trend" not in snap.missing and "vix" not in snap.missing


def load_macro(buy_days: list[str], loader: Callable[[pd.Timestamp], Any] | None = None) -> dict[str, Any] | None:
    """The market regime the trading day before each buy, today's snapshot and the macro agent's own score by regime.

    One history is loaded and reused; every snapshot is computed as of the day before the buy, so a buy never sees its
    own day. Returns None when the macro series cannot be loaded (the regime tag is then simply absent).
    """
    from agents.regime_agent import MacroRegimeAgent
    from data.macro_data import compute_macro_snapshot

    if not buy_days:
        return None
    first = pd.Timestamp(min(buy_days))
    try:
        history = (loader or _load_history)(first)
    except Exception as exc:  # noqa: BLE001 - the regime tag is optional
        logger.info("Macro history unavailable: %s", exc)
        return None
    if history.series.get("spy") is None or history.errors.get("core"):
        return None
    by_date: dict[str, str] = {}
    for day in sorted(set(buy_days)):
        snap = compute_macro_snapshot(history, pd.Timestamp(day) - pd.Timedelta(days=1))
        if _usable(snap):
            by_date[day] = snap.regime
    # The macro agent's own score by regime, month by month over the same stretch (what it favours, not what we assume).
    agent = MacroRegimeAgent()
    scores: dict[str, list[float]] = defaultdict(list)
    for me in pd.date_range(first, pd.Timestamp.today().normalize(), freq="ME"):
        snap = compute_macro_snapshot(history, me)
        if _usable(snap):
            scores[snap.regime].append(agent.evaluate_with_data(snap).score)
    live = compute_macro_snapshot(history, None)
    spy = history.series.get("spy")
    drawdown = None
    if spy is not None and len(spy) >= 250:
        drawdown = float(spy.iloc[-1] / spy.tail(252).max() - 1.0)
    now = None
    if _usable(live):
        now = {"regime": live.regime, "vix": live.vix_level, "spy_vs_200d": (live.spy_price / live.spy_sma_200 - 1.0)
               if live.spy_sma_200 == live.spy_sma_200 and live.spy_sma_200 else None,
               "market_from_high": drawdown, "as_of": live.as_of, "stance": settings.VIX_STANCE}
    return {"by_date": by_date, "agent_scores": {r: {"n": len(v), "mean": float(np.mean(v))} for r, v in scores.items()}, "now": now}


# ================================================================== history numbers


def _rows_in(rows: list[dict[str, Any]], tag: str, buckets: list[str]) -> list[dict[str, Any]]:
    return [r for r in rows if r["tags"].get(tag) in buckets]


def _gap(rows: list[dict[str, Any]], tag: str, plus: list[str], minus: list[str], split: pd.Timestamp) -> dict[str, Any] | None:
    """Mean excess return of the buys in ``plus`` buckets minus those in ``minus`` buckets, with the track record's tests."""
    a, b = _pattern(_rows_in(rows, tag, plus), VALUE, split), _pattern(_rows_in(rows, tag, minus), VALUE, split)
    if not a["n"] or not b["n"]:
        return None
    c = _compare(a, b, split)
    conf, why = c["confidence"]
    return {"plus_n": a["n"], "minus_n": b["n"], "plus_mean": a["mean"], "minus_mean": b["mean"], "difference": c["difference"],
            "t": c["t"], "first_half": c["first_half"], "second_half": c["second_half"], "n": c["n"], "confidence": conf, "why": why}


def _verdict(gap: dict[str, Any] | None) -> tuple[str, str]:
    """('in line' | 'your history disagrees' | 'too few cases', 'clear' | 'leaning' | '')."""
    if gap is None or gap["difference"] is None or gap["n"] < MIN_CASES or gap["plus_n"] < 3 or gap["minus_n"] < 3:
        return "too few cases", ""
    strength = "clear" if gap["confidence"] in ("medium", "high") else "leaning"
    return ("in line" if gap["difference"] > 0 else "your history disagrees"), strength


def _split(rewards: dict[str, int]) -> tuple[list[str], list[str]]:
    return [b for b, v in rewards.items() if v > 0], [b for b, v in rewards.items() if v < 0]


def _derived_regime_rewards(agent_scores: dict[str, dict[str, Any]] | None) -> dict[str, int]:
    """Regimes where the macro agent's average score is above (+1) or below (-1) its average over all months."""
    usable = {r: s for r, s in (agent_scores or {}).items() if s["n"] >= 2}
    if len(usable) < 2:
        return {}
    allmean = sum(s["mean"] * s["n"] for s in usable.values()) / sum(s["n"] for s in usable.values())
    return {r: (1 if s["mean"] > allmean else -1) for r, s in usable.items() if abs(s["mean"] - allmean) > 1e-9}


def _vix_rewards(line: dict[str, Any]) -> dict[str, int]:
    if settings.VIX_STANCE == "contrarian":
        return dict(line["rewards"])
    return {b: -v for b, v in line["rewards"].items()}


# ================================================================== panel numbers


def _panel_blocks(evidence: dict[str, Any], tag: str, regime: str | None = None) -> dict[str, Any] | None:
    if regime is not None:
        return ((evidence.get("by_regime") or {}).get("inside_regime") or {}).get(regime, {}).get(tag)
    if tag == REGIME_TAG:
        return evidence.get("by_regime")
    return (evidence.get("conditions") or {}).get(tag)


def _study_text(evidence: dict[str, Any], block: dict[str, Any]) -> str:
    s = evidence["studies"].get(block.get("study", "")) or {}
    return f"{s.get('name', 'panel')}, {s.get('period', '')}"


def _weighted(buckets: list[dict[str, Any]], key: str) -> float | None:
    vals = [(b.get(key), b.get("rows") or 1) for b in buckets if b.get(key) is not None]
    return sum(v * w for v, w in vals) / sum(w for _, w in vals) if vals else None


def _pct(v: float | None, d: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{d}%}"


def panel_against(evidence: dict[str, Any], tag: str, plus: list[str], minus: list[str], block: dict[str, Any] | None = None,
                  factor_sign: int | None = None) -> dict[str, Any]:
    """What the S&P 500 panel says about the direction ``plus`` against ``minus`` for this tag.

    direction: 'agrees' (the plus buckets did better on both horizons), 'disagrees' (worse on both), 'mixed' (the two
    horizons differ) or 'none' (the panel has no comparable bucket).
    """
    block = block if block is not None else _panel_blocks(evidence, tag)
    if not block or block.get("kind") == "note":
        return {"direction": "none", "text": (block or {}).get("note") or "The panel has no bucket to compare with this tag.", "study": _study_text(evidence, block) if block else None}
    if block.get("supported") is False:  # the study tested this condition and found no reliable difference
        return {"direction": "none", "text": block.get("note") or "The panel found no reliable difference.", "study": _study_text(evidence, block)}
    if block.get("kind") == "factor":
        ic, t = block["ic"], block.get("ic_t")
        if t is None or abs(t) < SIGNIFICANT_T or factor_sign is None:
            return {"direction": "none", "text": f"rank IC {ic:+.3f} (t {t:+.1f}): no effect the panel can measure",
                    "study": _study_text(evidence, block)}
        return {"direction": "agrees" if ic * factor_sign > 0 else "disagrees", "text": f"rank IC {ic:+.3f} (t {t:+.1f})", "study": _study_text(evidence, block)}
    pb = [b for b in block["buckets"] if set(b["tag_buckets"]) & set(plus)]
    mb = [b for b in block["buckets"] if set(b["tag_buckets"]) & set(minus)]
    if not pb or not mb:
        return {"direction": "none", "text": "The panel has no bucket that matches both sides of this comparison.", "study": _study_text(evidence, block)}
    d1 = (_weighted(pb, "value") or 0.0) - (_weighted(mb, "value") or 0.0)
    a_p, a_m = _weighted(pb, "alt"), _weighted(mb, "alt")
    d2 = (a_p - a_m) if a_p is not None and a_m is not None else None
    if d2 is None or (d1 > 0) == (d2 > 0):
        direction = "agrees" if d1 > 0 else "disagrees"
    else:
        direction = "mixed"
    sides = lambda bs: ", ".join(_BUCKET_WORDS.get(b["label"], b["label"]) for b in bs)  # noqa: E731  (display only)
    text = (f"{sides(pb)}: {_pct(_weighted(pb, 'value'))} a year next month"
            + (f", {_pct(a_p)} over 12 months" if a_p is not None else "")
            + f"; {sides(mb)}: {_pct(_weighted(mb, 'value'))}" + (f", {_pct(a_m)} over 12 months" if a_m is not None else ""))
    return {"direction": direction, "text": text, "study": _study_text(evidence, block), "diff": d1, "diff_alt": d2}


# ================================================================== the review


def _weights_by_unit(cfg: Any) -> dict[str, dict[str, Any]]:
    """Share of its layer each agent has now: the owner layer, and each regime's timing profile."""
    from agents.registry import get_agent_classes

    classes = get_agent_classes()
    role = {k: cfg.roles.get(k, getattr(c, "role", "timing")) for k, c in classes.items()}
    out: dict[str, dict[str, Any]] = {}
    owner = {k: v for k, v in cfg.owner_weights.items() if role.get(k) == "owner" and cfg.is_enabled(k)}
    total = sum(owner.values())
    for k, c in classes.items():
        entry: dict[str, Any] = {"label": c.name, "role": role[k], "enabled": cfg.is_enabled(k), "owner": None, "by_regime": {}}
        if role[k] == "owner":
            entry["owner"] = (owner.get(k, 0.0) / total) if total else 0.0
        else:
            for regime in REGIMES:
                prof = {a: w for a, w in cfg.regime_weights.get(regime, {}).items() if role.get(a) == "timing" and cfg.is_enabled(a)}
                s = sum(prof.values())
                entry["by_regime"][regime] = (prof.get(k, 0.0) / s) if s and cfg.is_enabled(k) else 0.0
        out[k] = entry
    return out


def _tag_index(ranked: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {t["tag"]: t for t in ranked}


def _history_of(rank: dict[str, Any] | None) -> dict[str, Any] | None:
    if not rank:
        return None
    return {"best": rank["best"], "worst": rank["worst"], "spread": rank["spread"], "n": rank["n"], "t": rank["t"],
            "first_half": rank["first_half"], "second_half": rank["second_half"], "confidence": rank["confidence"], "why": rank["why"],
            "buckets": [{"bucket": b["bucket"], "n": b["n"], "mean": b["mean"]} for b in rank["buckets"]]}


def _agent_verdict(lines: list[dict[str, Any]]) -> tuple[str, str]:
    usable = [x for x in lines if x["verdict"] in ("in line", "your history disagrees")]
    if not usable:
        return "too few cases", ""
    clear = [x for x in usable if x["strength"] == "clear"]
    dis_c = [x for x in clear if x["verdict"] == "your history disagrees"]
    in_c = [x for x in clear if x["verdict"] == "in line"]
    if dis_c and in_c:
        return "mixed", "clear"
    if dis_c:
        return "your history disagrees", "clear"
    if in_c and not any(x["verdict"] == "your history disagrees" for x in usable):
        return "in line", "clear"
    if in_c:
        return "in line", "clear"
    if all(x["verdict"] == "in line" for x in usable):
        return "in line", "leaning"
    if all(x["verdict"] == "your history disagrees" for x in usable):
        return "your history disagrees", "leaning"
    return "mixed", "leaning"


def build_agent_rows(rows: list[dict[str, Any]], ranked: list[dict[str, Any]], split: pd.Timestamp, cfg: Any,
                     evidence: dict[str, Any], agent_scores: dict[str, dict[str, Any]] | None, today_regime: str | None) -> list[dict[str, Any]]:
    weights = _weights_by_unit(cfg)
    index = _tag_index(ranked)
    present = {t for r in rows for t in r["tags"]}
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in LINES:
        if line["tag"] not in present:
            continue
        rewards = _derived_regime_rewards(agent_scores) if line.get("derived") else _vix_rewards(line) if line.get("vix") else dict(line["rewards"])
        plus, minus = _split(rewards)
        gap = _gap(rows, line["tag"], plus, minus, split) if plus and minus else None
        verdict, strength = _verdict(gap)
        panel = panel_against(evidence, line["tag"], plus, minus, factor_sign=line.get("factor_sign")) if plus and minus else \
            {"direction": "none", "text": "No direction to compare.", "study": None}
        by_agent[line["agent"]].append({
            "tag": line["tag"], "reads": line["reads"], "rewarded": plus, "penalised": minus, "history": _history_of(index.get(line["tag"])),
            "gap": gap, "verdict": verdict, "strength": strength, "panel": panel,
        })
    out = []
    order = [OWNER, *[k for k in (cfg.agent_order or []) if k != OWNER]]
    for key in order:
        if key == OWNER:
            w = {"label": "Owner score (fundamentals and shareholder)", "role": "owner", "enabled": True,
                 "owner": None, "by_regime": {}, "parts": [{"agent": k, "label": v["label"], "weight": v["owner"]}
                                                           for k, v in weights.items() if v["role"] == "owner" and v["enabled"]]}
        elif key in weights and weights[key]["role"] != "owner":
            w = {**weights[key], "parts": None}
        else:
            continue
        lines = by_agent.get(key, [])
        verdict, strength = _agent_verdict(lines) if lines else ("not testable", "")
        panel_agent = (evidence.get("agents") or {}).get(key)
        dirs = {x["panel"]["direction"] for x in lines} - {"none"}
        pv = "none" if not dirs else dirs.pop() if len(dirs) == 1 else "mixed"
        if panel_agent and panel_agent.get("ic_t") is not None and abs(panel_agent["ic_t"]) >= SIGNIFICANT_T:
            pv = "agrees" if panel_agent["ic"] > 0 else "disagrees"  # the agent's own score, tested out of sample, outranks the tag lines
        note = None
        if not lines:
            note = ("Not replayed: tick 'Rate each buy with FastValue' when you run the track record." if key == OWNER
                    else "None of your entry tags reads this agent's conditions, so your history cannot test it.")
        out.append({
            "key": key, "label": w["label"], "layer": "owner" if w["role"] == "owner" else "timing" if w["role"] == "timing" else "info",
            "enabled": w["enabled"], "owner_parts": w.get("parts"), "by_regime": w["by_regime"],
            "weight_today": (w["by_regime"].get(today_regime) if today_regime and w["by_regime"] else None),
            "lines": lines, "verdict": verdict, "strength": strength, "panel_verdict": pv, "note": note,
            "panel": ({"text": panel_agent["text"], "ic": panel_agent.get("ic"), "ic_t": panel_agent.get("ic_t"), "horizon": panel_agent.get("horizon"),
                       "study": _study_text(evidence, panel_agent)} if panel_agent else None),
        })
    return out


def build_by_regime(rows: list[dict[str, Any]], split: pd.Timestamp, evidence: dict[str, Any], cfg: Any) -> list[dict[str, Any]]:
    tagged = [r for r in rows if REGIME_TAG in r["tags"]]
    block = evidence.get("by_regime") or {}
    panel = {b["label"]: b for b in (block.get("buckets") or [])}
    weights = _weights_by_unit(cfg)
    out = []
    skip_inside = {REGIME_TAG, "market trend", "VIX", "market drawdown"}  # these build the regime, so inside it they say nothing new
    for regime in REGIMES:
        rs = [r for r in tagged if r["tags"][REGIME_TAG] == regime]
        rest = [r for r in tagged if r["tags"][REGIME_TAG] != regime]
        vals = [r[VALUE] for r in rs if r.get(VALUE) is not None]
        p = _pattern(rs, VALUE, split)
        c = _compare(p, _pattern(rest, VALUE, split), split) if rest and rs else None
        profile = sorted(((k, v["by_regime"].get(regime, 0.0)) for k, v in weights.items() if v["by_regime"]), key=lambda kv: -kv[1])
        inside: list[dict[str, Any]] = []
        if len(rs) >= MIN_REGIME_BUYS:
            for t in _rank_tags([{**r, "tags": {k: v for k, v in r["tags"].items() if k not in skip_inside}} for r in rs], split)[:3]:
                inside.append({"tag": t["tag"], "best": t["best"], "worst": t["worst"], "spread": t["spread"], "n": t["n"], "t": t["t"],
                               "first_half": t["first_half"], "second_half": t["second_half"], "confidence": t["confidence"], "why": t["why"],
                               "panel": _inside_panel(evidence, regime, t)})
        pn = panel.get(regime)
        out.append({
            "regime": regime, "n": len(vals), "n_eff": p["n_eff"], "mean": p["mean"], "median": _median(vals),
            "hit_rate": (sum(1 for v in vals if v > 0) / len(vals)) if vals else None,
            "first_half": p["first_half"], "second_half": p["second_half"],
            "n_first": sum(1 for r in rs if r.get(VALUE) is not None and pd.Timestamp(r["date"]) < split),
            "n_second": sum(1 for r in rs if r.get(VALUE) is not None and pd.Timestamp(r["date"]) >= split),
            "t": p["t"], "against_rest": ({"difference": c["difference"], "t": c["t"], "first_half": c["first_half"], "second_half": c["second_half"],
                                           "confidence": c["confidence"][0], "why": c["confidence"][1]} if c else None),
            "heaviest_agents": [{"agent": k, "label": weights[k]["label"], "weight": w} for k, w in profile[:3] if w > 0],
            "panel": ({"value": pn.get("value"), "alt": pn.get("alt"), "t": pn.get("t"), "rows": pn.get("rows"),
                       "supported": block.get("supported", True), "note": block.get("note")} if pn else None),
            "inside": inside, "too_few": len(rs) < MIN_REGIME_BUYS,
        })
    return out


def _inside_panel(evidence: dict[str, Any], regime: str, t: dict[str, Any]) -> dict[str, Any] | None:
    block = _panel_blocks(evidence, t["tag"], regime)
    if not block:
        return None
    return panel_against(evidence, t["tag"], [t["best"]], [t["worst"]], block=block)


def _tag_view(rows: list[dict[str, Any]], tag: str, split: pd.Timestamp, evidence: dict[str, Any],
              ranked: dict[str, dict[str, Any]], regime: str | None = None) -> dict[str, Any] | None:
    """Evidence for one tag: the panel's best and worst mapped buckets, and what your history did with the same two."""
    block = _panel_blocks(evidence, tag, regime)
    rank = ranked.get(tag)
    if block and block.get("kind") not in ("factor", "note"):
        mapped = [b for b in block["buckets"] if b["tag_buckets"]]
        if len(mapped) >= 2:
            hi = max(mapped, key=lambda b: b["value"])
            lo = min(mapped, key=lambda b: b["value"])
            panel = panel_against(evidence, tag, hi["tag_buckets"], lo["tag_buckets"], block=block)
            gap = _gap(rows, tag, hi["tag_buckets"], lo["tag_buckets"], split)
            return {"tag": tag, "better": " or ".join(hi["tag_buckets"]), "worse": " or ".join(lo["tag_buckets"]), "panel": panel, "gap": gap}
    if rank:
        gap = _gap(rows, tag, [rank["best"]], [rank["worst"]], split)
        return {"tag": tag, "better": rank["best"], "worse": rank["worst"], "panel": {"direction": "none", "text": "The panel has no comparable bucket.", "study": None}, "gap": gap}
    return None


def label_evidence(view: dict[str, Any]) -> tuple[str, int] | None:
    """The label and its rank (0 strongest) for one comparison, or None when nothing is strong enough to act on."""
    verdict, strength = _verdict(view["gap"])
    p = view["panel"]["direction"]
    mine = verdict if strength == "clear" else None  # a leaning or thin history never counts as evidence
    if mine == "in line" and p == "agrees":
        return "Your history and the S&P 500 panel agree", 0
    if p == "agrees":
        return "S&P 500 panel only", 1
    if mine and p in ("disagrees", "mixed", "none"):
        # An opposing or silent panel leaves your history as a hypothesis, in the direction your history shows.
        return "Your history only: a hypothesis to test forward", 2
    return None


def _flip(view: dict[str, Any]) -> dict[str, Any]:
    """Swap the sides of a comparison (used when your history shows the 'worse' panel bucket doing better)."""
    g = view["gap"]
    if g and g["difference"] is not None and g["difference"] < 0:
        neg = lambda v: -v if v is not None else None  # noqa: E731
        t, first, second = neg(g["t"]), neg(g["first_half"]), neg(g["second_half"])
        conf, why = _confidence(g["n"], t, _same_sign(first, second))
        g2 = {**g, "difference": -g["difference"], "t": t, "first_half": first, "second_half": second, "confidence": conf, "why": why,
              "plus_n": g["minus_n"], "minus_n": g["plus_n"], "plus_mean": g["minus_mean"], "minus_mean": g["plus_mean"]}
        p = view["panel"]
        return {**view, "better": view["worse"], "worse": view["better"], "gap": g2,
                "panel": {**p, "direction": {"agrees": "disagrees", "disagrees": "agrees"}.get(p["direction"], p["direction"])}}
    return view


def _regime_rec(mine: dict[str, Any], regimes: list[dict[str, Any]], today: str) -> dict[str, Any]:
    base = {"kind": "regime", "tag": REGIME_TAG, "bucket": today, "n": mine["n"], "first_half": mine["first_half"],
            "second_half": mine["second_half"], "t": None, "confidence": None, "link": None}
    ar = mine["against_rest"]
    if mine["too_few"] or not ar:
        return {**base, "label": "Keep what you do", "rank": 3,
                "text": f"Today's regime is {today} and {mine['n']} of your buys were made in it, too few to say anything. "
                        "Keep what you do and watch this table as buys accumulate."}
    base.update({"t": ar["t"], "first_half": ar["first_half"], "second_half": ar["second_half"], "confidence": ar["confidence"]})
    pn = mine.get("panel")
    others = [r["panel"]["value"] for r in regimes if r["regime"] != today and r.get("panel") and r["panel"].get("value") is not None]
    panel_up = None
    if pn and pn.get("value") is not None and others and pn.get("supported", True):
        panel_up = pn["value"] > float(np.mean(others))
    better = (ar["difference"] or 0) > 0
    if ar["confidence"] in ("medium", "high"):
        agree = panel_up is not None and panel_up == better
        label, rank = ("Your history and the S&P 500 panel agree", 0) if agree else ("Your history only: a hypothesis to test forward", 2)
        if better:
            text = (f"Today's regime is {today}. Your buys in it beat your other buys by {ar['difference']:+.1%} after {MAIN_HORIZON} days"
                    f"{' and the S&P 500 panel points the same way' if agree else ''}. Act on it by staging the buy in tranches instead of waiting "
                    "for a calmer market. Rules and weights stay as they are.")
        else:
            text = (f"Today's regime is {today}. Your buys in it did {abs(ar['difference']):.1%} worse than your other buys after {MAIN_HORIZON} days"
                    f"{' and the S&P 500 panel also ranks it low' if agree else ''}. Act on it by waiting for a pullback inside the name and keeping "
                    "the order near your usual size. Rules and weights stay as they are.")
        return {**base, "label": label, "rank": rank, "link": "weights", "text": text}
    if panel_up is not None:
        return {**base, "label": "S&P 500 panel only", "rank": 1, "link": "weights",
                "text": f"Today's regime is {today}. The S&P 500 panel ranks it {'above' if panel_up else 'below'} the other regimes for owner scores at or above the entry level "
                        f"({_pct(pn['value'])} a year next month), but your {mine['n']} buys in it show no clear difference ({ar['why']}). "
                        "Keep what you do. The panel's lean is a reason to stage entries, not to skip them."}
    unsupported = f" {pn['note']}" if pn and not pn.get("supported", True) and pn.get("note") else ""
    return {**base, "label": "Keep what you do", "rank": 3,
            "text": f"Today's regime is {today}. Your {mine['n']} buys in it show no clear difference from your other buys ({ar['why']}).{unsupported} Keep what you do."}


def _tag_rec(view: dict[str, Any], label: str, rank: int) -> dict[str, Any]:
    g = view["gap"] or {}
    tag, best, worst = view["tag"], view["better"], view["worse"]
    best, worst = _BUCKET_WORDS.get(best, best), _BUCKET_WORDS.get(worst, worst)  # display only: no BUY / HOLD / SELL in the text
    head = tag[0].upper() + tag[1:]
    p = view["panel"]
    diff = f"{g['difference']:+.1%}" if g.get("difference") is not None else "n/a"
    if rank == 0:
        body = (f"{head}: your buys tagged '{best}' beat those tagged '{worst}' by {diff} after {MAIN_HORIZON} days, and the S&P 500 panel agrees "
                f"({p['text']}).")
    elif rank == 1:
        why = g.get("why") or "too few buys in these buckets"
        body = f"{head}: in the S&P 500 panel, stocks tagged '{best}' did better than '{worst}' ({p['text']}). Your history cannot confirm it ({why})."
    else:
        extra = {"disagrees": " The S&P 500 panel points the other way.", "mixed": " The S&P 500 panel is mixed on it."}.get(p["direction"], " The panel has nothing comparable.")
        body = f"{head}: your buys tagged '{best}' beat those tagged '{worst}' by {diff} after {MAIN_HORIZON} days ({g.get('why', '')}).{extra}"
    family = TAG_FAMILY.get(tag, "entry")
    return {"kind": "tag", "label": label, "rank": rank, "tag": tag, "bucket": best, "n": g.get("n"), "t": g.get("t"),
            "first_half": g.get("first_half"), "second_half": g.get("second_half"), "confidence": g.get("confidence"),
            "link": None if family == "size" else "weights", "text": f"{body} {FAMILY_ACTION[family].format(best=best, worst=worst)}"}


def build_recommendations(rows: list[dict[str, Any]], ranked: list[dict[str, Any]], regimes: list[dict[str, Any]], split: pd.Timestamp,
                          evidence: dict[str, Any], now: dict[str, Any] | None, agents: list[dict[str, Any]]) -> dict[str, Any]:
    index = _tag_index(ranked)
    recs: list[dict[str, Any]] = []
    today = (now or {}).get("regime")
    mine = next((r for r in regimes if r["regime"] == today), None)
    if today and mine:
        recs.append(_regime_rec(mine, regimes, today))
    tags = [t["tag"] for t in ranked if t["tag"] != REGIME_TAG]
    tags += [t for t, b in (evidence.get("conditions") or {}).items() if t not in tags and b.get("kind") not in ("factor", "note")]
    for tag in tags:
        view = _tag_view(rows, tag, split, evidence, index)
        if not view:
            continue
        verdict, strength = _verdict(view["gap"])
        if strength == "clear" and verdict == "your history disagrees":
            view = _flip(view)  # your history shows the panel's worse bucket doing better: say it that way round
        lab = label_evidence(view)
        if lab:
            recs.append(_tag_rec(view, *lab))
    lead = [r for r in recs if r["kind"] == "regime"]  # today's regime always comes first
    rest = sorted((r for r in recs if r["kind"] != "regime"), key=lambda r: (r["rank"], -(abs(r["t"]) if r.get("t") is not None else 0.0)))
    recs = (lead + rest)[:MAX_RECOMMENDATIONS]
    tests = []
    for a in agents:
        if a["verdict"] == "your history disagrees" and a["strength"] == "clear":
            w = a["weight_today"] if a["weight_today"] is not None else max(a["by_regime"].values(), default=None)
            tags_txt = ", ".join(x["tag"] for x in a["lines"] if x["verdict"] == "your history disagrees" and x["strength"] == "clear")
            panel_agrees = a["panel_verdict"] == "disagrees"  # the panel also finds the agent's direction wrong
            ptext = f" and the S&P 500 panel also finds it backwards ({a['panel']['text'].rstrip('.')})" if panel_agrees and a["panel"] else ""
            weight = f" (now {w:.0%} of its layer)" if w else ""
            tests.append({
                "agent": a["key"], "label": a["label"], "tags": tags_txt,
                "evidence": "Your history and the S&P 500 panel agree" if panel_agrees else "Your history only: a hypothesis to test forward",
                "text": f"Your history disagrees with the {a['label']} on {tags_txt}{ptext}. That is one investor's history, a hypothesis and not a "
                        f"reason to change its weight{weight}. To test a lower weight, replay it on the S&P 500 panel in the Backtest lab: whole period, both "
                        f"halves, |t| of at least {SIGNIFICANT_T:.0f}, and keep the change only if it also holds out of sample."})
    if not any(r["rank"] in (0, 2) for r in recs):
        recs.append({"kind": "none", "label": "Keep what you do", "rank": 3, "tag": None, "bucket": None, "n": len(rows), "t": None,
                     "first_half": None, "second_half": None, "confidence": None, "link": None,
                     "text": f"Nothing in your {len(rows)} tagged buys is strong enough to act on yet. Keep what you do and watch the tables as buys accumulate."})
    return {"now": now, "items": recs, "weight_tests": tests}


def build_review(style: dict[str, Any], macro: dict[str, Any] | None, split: pd.Timestamp, cfg: Any | None = None,
                 evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """The whole 'Your history against the weights' block for the track record's Entries & exits section."""
    from config.weights import load_scoring_config

    cfg = cfg or load_scoring_config()
    evidence = evidence or load_panel_evidence()
    entries = style["entries"]
    rows, ranked = entries["rows"], entries["tags"]
    now = (macro or {}).get("now")
    agents = build_agent_rows(rows, ranked, split, cfg, evidence, (macro or {}).get("agent_scores"), (now or {}).get("regime"))
    regimes = build_by_regime(rows, split, evidence, cfg) if any(REGIME_TAG in r["tags"] for r in rows) else []
    mapped = {line["tag"] for line in LINES}
    unread = [{"tag": t["tag"], "note": STYLE_TAGS.get(t["tag"], "No agent reads this."), "best": t["best"], "worst": t["worst"],
               "spread": t["spread"], "n": t["n"], "confidence": t["confidence"]}
              for t in ranked if t["tag"] not in mapped]
    return {
        "regime_available": bool(regimes),
        "agents": agents,
        "unread": unread,
        "regimes": regimes,
        "now": build_recommendations(rows, ranked, regimes, split, evidence, now, agents),
        "agent_scores_by_regime": (macro or {}).get("agent_scores"),
        "panel": {"studies": {k: {f: s.get(f) for f in ("name", "universe", "period", "date")} for k, s in evidence["studies"].items()}},
        "protocol": f"Directions come from the agents' code. A verdict compares your buys in the buckets an agent rewards with those it penalises, "
                    f"{MAIN_HORIZON} days later against MSCI World, with the track record's statistics: orders in one name and month count once, "
                    f"Welch's t, both halves. 'In line' or 'disagrees' is clear only with medium or high confidence (at least {MIN_CASES} cases, |t| of 1.5 or more, "
                    f"the same sign in both halves); otherwise it is shown as low confidence and is not acted on. The regime is the one the app's macro "
                    f"classifier gave the trading day before each buy. Nothing here changes a weight.",
    }
