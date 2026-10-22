"""Your investing style in numbers, and what made your entries and exits good or bad.

Built on the track record (``core.track_record``): the ledger of the imported
account, daily EUR prices and the per-order forward paths. Nothing here needs
the network beyond the prices the track record already loads (plus the VIX
close, when it is cached or reachable).

Three parts:

- **Style**: holding periods, the number and size of bets, pyramiding (how
  positions were built), hit rate, the size of wins against losses, where the
  P&L came from (how few positions carry it), concentration today, cash, how
  much of the gain is still unrealised, recurring against one-off dividend
  income, and the core / satellite / metals / speculative mix.
- **Entries**: every buy is tagged with the conditions at the time (distance
  from the 52-week high, trend against the 200-day mean, the move of the last
  month, RSI, the name's volatility class, the market's own drawdown and trend,
  the VIX, the size of the order, whether it added to a winner or a loser, how
  long the position had been building, how many positions were open) and the
  tags are ranked by how well they separated good entries from bad ones, on
  the return against MSCI World 90 days later and on what the money made.
- **Exits**: sales into weakness against strength, quick exits of steady names,
  losers cut fast, and what each did against the index afterwards.

Statistics follow the track record's protocol (decision_log.md D-23): orders
in one name and month count once, group differences use Welch's t on those
clusters, both halves of the period must agree for a pattern to count as more
than chance, and a thin sample is said to be thin. With a few dozen round
trips most tags are low confidence; the tables still show the numbers, ranked,
so the pattern can be watched as more trades come in. Nothing here changes
FastValue's scoring.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from core.ledger import EPS
from core.track_record import (
    COMMODITY_WORDS,
    MAIN_HORIZON,
    MIN_CASES,
    _Market,
    _compare,
    _confidence,
    _pattern,
    _same_sign,
    _tstat,
)

logger = logging.getLogger(__name__)

QUICK_DAYS = 45  # a round trip closed within this many days counts as a quick trade
SHORT_HOLD_DAYS = 180  # "held less than six months"
HOLD_BUCKETS = ((0, 45, "under 45 days"), (45, 180, "45 to 180 days"), (180, 365, "180 days to a year"),
                (365, None, "over a year"))
BUY_COUNT_BUCKETS = ((1, 1, "one buy"), (2, 5, "2 to 5 buys"), (6, None, "6 or more buys"))
BUILD_BUCKETS = ((0, 30, "built within a month"), (30, 180, "built over 1 to 6 months"), (180, None, "built over 6+ months"))
WEAKNESS = 0.15  # sold at least this far below the position's high since entry = sold into weakness
PARETO = 0.8  # share of gross gains carried by the fewest positions
VIX_TICKER = "^VIX"
CORE_RE = re.compile(r"\b(msci|world|s&p ?500|all[- ]?world|global|developed|europe|stoxx|euro ?stoxx|nasdaq[- ]?100|"
                     r"ftse|emerging|em\b|bond|treasur|govern|money market|overnight|\bestr\b|eonia|aggregate|"
                     r"dividend aristocrat|value|quality|small cap|russell)\b", re.I)
RECURRING_PAYMENTS = 3  # a name that paid at least this often counts as recurring dividend income


# ================================================================== helpers


def _n(count: int, one: str, many: str) -> str:
    """The count with the right noun: "1 case", "2 cases"."""
    return f"{count} {one if count == 1 else many}"


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _bucket(value: float | None, buckets: tuple) -> str | None:
    if value is None:
        return None
    for lo, hi, label in buckets:
        if value >= lo and (hi is None or value < hi):
            return label
    return None


def _count_bucket(value: int, buckets: tuple) -> str:
    for lo, hi, label in buckets:
        if value >= lo and (hi is None or value <= hi):
            return label
    return buckets[-1][2]


def bucket_of(name: str | None, asset_class: str | None, group: str) -> str:
    """Core (broad index, bond and money-market funds), Metals, Speculative (turbos) or Satellite (the rest)."""
    if group == "Turbos":
        return "Speculative"
    if group == "Metal and commodity ETCs":
        return "Metals"
    if asset_class == "FUND" and CORE_RE.search(name or ""):
        return "Core"
    return "Satellite"


def _rsi(closes: pd.Series, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    up = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    down = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    d = float(down.iloc[-1])
    if d <= 0:
        return 100.0
    rs = float(up.iloc[-1]) / d
    return 100.0 - 100.0 / (1.0 + rs)


def _annual_vol(closes: pd.Series) -> float | None:
    r = np.log(closes).diff().dropna()
    if len(r) < 60:
        return None
    return float(r.std(ddof=1) * math.sqrt(252))


def _vix_series(m: _Market) -> pd.Series | None:
    """VIX close on calendar days (carried over weekends); None when it cannot be loaded."""
    try:
        from data import instruments as ins
        from data.price_history import fetch_price_history

        frames = fetch_price_history([VIX_TICKER], ins.HISTORY_START)
    except Exception as exc:  # noqa: BLE001 - the VIX tag is optional
        logger.info("VIX history unavailable: %s", exc)
        return None
    df = frames.get(VIX_TICKER)
    if df is None or df.empty:
        return None
    s = df["close"].dropna()
    return s.reindex(s.index.union(m.days)).ffill().reindex(m.days)


def _trading_days(s: pd.Series) -> pd.Series:
    return s[s.index.weekday < 5].dropna()


# ================================================================== volatility classes


def volatility_classes(ledger: dict[str, Any], m: _Market) -> dict[str, dict[str, Any]]:
    """Per priced stock or fund: annualised volatility in the year before its first buy, and its tercile."""
    rows: dict[str, dict[str, Any]] = {}
    for isin, info in ledger["instruments"].items():
        if info["asset_class"] not in ("STOCK", "FUND") or isin not in m.full:
            continue
        first = pd.Timestamp(info["first_date"])
        hist = m.full[isin]
        window = hist[(hist.index < first) & (hist.index >= first - pd.Timedelta(days=400))]
        vol = _annual_vol(window) if len(window) >= 60 else _annual_vol(hist.head(260))
        if vol is None:
            continue
        rows[isin] = {"vol": vol, "class": None}
    vols = sorted(r["vol"] for r in rows.values())
    if len(vols) >= 3:
        lo, hi = float(np.quantile(vols, 1 / 3)), float(np.quantile(vols, 2 / 3))
        for r in rows.values():
            r["class"] = "low" if r["vol"] <= lo else "high" if r["vol"] > hi else "mid"
        bounds = {"low_max": lo, "high_min": hi}
    else:
        bounds = {}
    return {"classes": rows, "bounds": bounds}


# ================================================================== style numbers


def _episode_buys(ledger: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Buy orders per (isin, episode start date)."""
    starts: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for ep in ledger["episodes"]:
        starts[ep["isin"]].append((ep["start"], ep["end"]))
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for o in ledger["orders"]:
        if o["side"] != "buy":
            continue
        for start, end in starts.get(o["isin"], []):
            if start <= o["date"] and (end is None or o["date"] <= end):
                out[(o["isin"], start)].append(o)
                break
    return out


def _trip_extras(ledger: dict[str, Any], trips: list[dict[str, Any]], m: _Market,
                 vol: dict[str, dict[str, Any]], orders: list[dict[str, Any]]) -> None:
    """Add build time, buy count, volatility class, bucket and entry drawdown to each round trip (in place)."""
    buys = _episode_buys(ledger)
    first_buy = {o["order"]: o for o in orders if o["side"] == "buy"}
    for t in trips:
        bs = sorted(buys.get((t["isin"], t["start"]), []), key=lambda o: o["date"])
        t["buys"] = len(bs)
        t["build_days"] = (pd.Timestamp(bs[-1]["date"]) - pd.Timestamp(bs[0]["date"])).days if bs else 0
        t["build_bucket"] = _bucket(t["build_days"], BUILD_BUCKETS)
        t["buys_bucket"] = _count_bucket(max(t["buys"], 1), BUY_COUNT_BUCKETS)
        t["hold_bucket"] = _bucket(t["days"], HOLD_BUCKETS)
        t["vol_class"] = (vol.get(t["isin"]) or {}).get("class")
        info = ledger["instruments"].get(t["isin"], {})
        t["bucket"] = bucket_of(info.get("name"), info.get("asset_class"), t["group"])
        o = first_buy.get(bs[0]["order"]) if bs else None
        t["entry_from_high"] = o.get("from_high_52w") if o else None
        t["entry_drawdown_bucket"] = _from_high_bucket(t["entry_from_high"])
        t["is_closed"] = t["outcome"] == "sold"


def _from_high_bucket(v: float | None) -> str | None:
    if v is None:
        return None
    d = -v
    return ("within 5% of the 52-week high" if d <= 0.05 else "5 to 20% below the high" if d <= 0.20
            else "20 to 40% below the high" if d <= 0.40 else "40%+ below the high")


def _pareto(trips: list[dict[str, Any]]) -> dict[str, Any]:
    gains = sorted((t["total_pnl"] for t in trips if t["total_pnl"] > 0), reverse=True)
    total_gain = sum(gains)
    n80, running = 0, 0.0
    for g in gains:
        running += g
        n80 += 1
        if running >= PARETO * total_gain - 1e-9:
            break
    total = sum(t["total_pnl"] for t in trips)
    top8 = sorted(trips, key=lambda t: -t["total_pnl"])[:8]
    return {"positions_with_gains": len(gains), "positions_for_80pct_of_gains": n80 if gains else 0,
            "gross_gains": total_gain, "gross_losses": sum(t["total_pnl"] for t in trips if t["total_pnl"] < 0),
            "top8_pnl": sum(t["total_pnl"] for t in top8), "total_pnl": total,
            "top8_share": (sum(t["total_pnl"] for t in top8) / total) if total > 0 else None}


def _dividend_mix(ledger: dict[str, Any]) -> dict[str, Any]:
    by_name: dict[str, list[float]] = defaultdict(list)
    for i in ledger["income"]:
        if i["kind"] == "dividend" and i.get("isin"):
            by_name[i["isin"]].append(i["amount"] + i["tax"])
    recurring = sum(sum(v) for v in by_name.values() if len(v) >= RECURRING_PAYMENTS)
    one_off = sum(sum(v) for v in by_name.values() if len(v) < RECURRING_PAYMENTS)
    total = recurring + one_off
    return {"recurring_net": recurring, "one_off_net": one_off, "names_recurring": sum(1 for v in by_name.values() if len(v) >= RECURRING_PAYMENTS),
            "names_one_off": sum(1 for v in by_name.values() if len(v) < RECURRING_PAYMENTS),
            "one_off_share": one_off / total if total > 0 else None}


def _buckets(instruments: list[dict[str, Any]], ledger: dict[str, Any], a: dict[str, Any], m: _Market) -> list[dict[str, Any]]:
    """Core / Satellite / Metals / Speculative: money put in, P&L, and the share of today's invested value."""
    by_isin = {r["isin"]: r for r in instruments}
    last = a["values"].iloc[-1] if len(a["values"]) else pd.Series(dtype=float)
    invested_now = float(last.sum()) if len(last) else 0.0
    rows: dict[str, dict[str, Any]] = {}
    for isin, r in by_isin.items():
        info = ledger["instruments"].get(isin, {})
        b = bucket_of(info.get("name"), info.get("asset_class"), r["group"])
        g = rows.setdefault(b, {"bucket": b, "instruments": 0, "bought": 0.0, "pnl": 0.0, "value_now": 0.0, "winners": 0})
        g["instruments"] += 1
        g["bought"] += r["bought"]
        g["pnl"] += r["total_pnl"]
        g["winners"] += 1 if r["total_pnl"] > 0 else 0
        g["value_now"] += float(last.get(isin, 0.0)) if isin in last.index else 0.0
    for g in rows.values():
        g["return_pct"] = g["pnl"] / g["bought"] if g["bought"] > 0 else None
        g["weight_now"] = g["value_now"] / invested_now if invested_now > 0 else None
        g["hit_rate"] = g["winners"] / g["instruments"] if g["instruments"] else None
    order = {"Core": 0, "Satellite": 1, "Metals": 2, "Speculative": 3}
    return sorted(rows.values(), key=lambda g: order.get(g["bucket"], 9))


def style_numbers(ledger: dict[str, Any], m: _Market, a: dict[str, Any], perf: dict[str, Any],
                  orders: list[dict[str, Any]], trips: list[dict[str, Any]], instruments: list[dict[str, Any]],
                  beh: dict[str, Any]) -> dict[str, Any]:
    closed = [t for t in trips if t["outcome"] == "sold" and t["group"] != "Turbos"]
    open_ = [t for t in trips if t["outcome"] == "open"]
    wins = [t for t in closed if t["total_pnl"] > 0]
    losses = [t for t in closed if t["total_pnl"] <= 0]
    disc = [o for o in orders if o["side"] == "buy" and not o["savings_plan"] and o["group"] != "Turbos"]
    plans = [o for o in orders if o["side"] == "buy" and o["savings_plan"]]
    gross_win = sum(t["total_pnl"] for t in wins)
    gross_loss = -sum(t["total_pnl"] for t in losses)
    all_trips = [t for t in trips if t["group"] != "Turbos"]
    pyramided = [t for t in all_trips if t.get("buys", 0) >= 5]
    total_pnl = sum(t["total_pnl"] for t in all_trips)
    top5 = sorted(all_trips, key=lambda t: -t["total_pnl"])[:5]
    # Today's weights among securities.
    last = a["values"].iloc[-1] if len(a["values"]) else pd.Series(dtype=float)
    last = last[last > 0].sort_values(ascending=False)
    invested_now = float(last.sum()) if len(last) else 0.0
    weights = (last / invested_now) if invested_now > 0 else last
    names = {r["isin"]: r["name"] for r in instruments}
    open_pnl = sum(r["unrealised"] for r in instruments if r["status"] == "open")
    total_all = sum(r["total_pnl"] for r in instruments)
    return {
        "round_trips": {
            "closed": len(closed), "open": len(open_),
            "median_days_closed": _median([t["days"] for t in closed]),
            "share_closed_quickly": (sum(1 for t in closed if t["days"] <= QUICK_DAYS) / len(closed)) if closed else None,
            "quick_days": QUICK_DAYS,
            "median_age_open": _median([t["days"] for t in open_]),
            "closed_within_6m": sum(1 for t in closed if t["days"] <= SHORT_HOLD_DAYS),
        },
        "bets": {
            "discretionary_buys": len(disc), "discretionary_total": sum(o["amount"] for o in disc),
            "discretionary_median": _median([o["amount"] for o in disc]),
            "discretionary_names": len({o["isin"] for o in disc}),
            "plan_buys": len(plans), "plan_total": sum(o["amount"] for o in plans),
            "plan_names": len({o["isin"] for o in plans}),
        },
        "pyramiding": {
            "median_buys_per_trip": _median([t.get("buys", 1) for t in all_trips]),
            "trips_with_5plus_buys": len(pyramided),
            "pnl_share_5plus_buys": (sum(t["total_pnl"] for t in pyramided) / total_pnl) if total_pnl > 0 else None,
            "top5": [{"name": t["name"], "buys": t.get("buys"), "days": t["days"], "build_days": t.get("build_days"),
                      "pnl": t["total_pnl"], "return_pct": t["return_pct"], "outcome": t["outcome"]} for t in top5],
        },
        "hit_rate": {
            "wins": len(wins), "losses": len(losses), "rate": (len(wins) / len(closed)) if closed else None,
            "avg_win": (gross_win / len(wins)) if wins else None, "avg_loss": (-gross_loss / len(losses)) if losses else None,
            "payoff": (gross_win / len(wins)) / (gross_loss / len(losses)) if wins and losses and gross_loss > 0 else None,
            "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
            "expectancy": (sum(t["total_pnl"] for t in closed) / len(closed)) if closed else None,
            "median_return_win": _median([t["return_pct"] for t in wins]),
            "median_return_loss": _median([t["return_pct"] for t in losses]),
        },
        "concentration_of_pnl": _pareto(all_trips),
        "today": {
            "top1_weight": float(weights.iloc[0]) if len(weights) else None,
            "top1_name": names.get(weights.index[0]) if len(weights) else None,
            "top3_weight": float(weights.iloc[:3].sum()) if len(weights) else None,
            "top3_names": [names.get(i) for i in weights.index[:3]],
            "positions": int(len(weights)),
            "effective_positions": (beh.get("concentration") or {}).get("current_effective_positions"),
            "cash_share": (perf["cash"] / perf["end_value"]) if perf.get("end_value") else None,
            "avg_cash_share": perf.get("avg_cash_share"),
            "unrealised_share_of_gain": (open_pnl / total_all) if total_all > 0 else None,
            "unrealised": open_pnl, "total_pnl": total_all,
        },
        "dividends": _dividend_mix(ledger),
        "costs": beh.get("costs"),
        "buckets": _buckets(instruments, ledger, a, m),
    }


# ================================================================== entry conditions


def _tag_entries(orders: list[dict[str, Any]], ledger: dict[str, Any], m: _Market, a: dict[str, Any],
                 vol: dict[str, dict[str, Any]], lots: list[dict[str, Any]], vix: pd.Series | None,
                 fastvalue: dict[str, Any] | None, regimes: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Buys of priced stocks and funds, each with its conditions at the time and its outcomes.

    ``regimes``: buy date (YYYY-MM-DD) -> the market regime the trading day before, from the macro classifier
    (``core.weights_review.load_macro``); a buy without an entry has no regime tag.
    """
    bench = _trading_days(m.bench_full if m.bench_full is not None else m.bench)
    bench_sma = bench.rolling(200, min_periods=150).mean()
    bench_high = bench.rolling(250, min_periods=120).max()
    disc_sizes = [o["amount"] for o in orders if o["side"] == "buy" and not o["savings_plan"] and o["group"] != "Turbos"]
    median_size = _median(disc_sizes) or 0.0
    episode_start: dict[tuple[str, str], str] = {}
    for ep in ledger["episodes"]:
        episode_start[(ep["isin"], ep["start"])] = ep["start"]
    starts_by_isin: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for ep in ledger["episodes"]:
        starts_by_isin[ep["isin"]].append((ep["start"], ep["end"]))
    lot_by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lot in lots:
        lot_by_order[lot["buy_order"]].append(lot)
    rated: dict[str, dict[str, Any]] = {}
    if fastvalue:
        for r in fastvalue.get("rated_rows") or []:
            rated[r["order"]] = r
    shares = a["shares"]
    out = []
    for o in orders:
        if o["side"] != "buy" or not o["priced"] or o["group"] == "Turbos":
            continue
        d = pd.Timestamp(o["date"])
        p = o["price"]
        hist = m.full.get(o["isin"])
        w = hist[hist.index < d] if hist is not None else pd.Series(dtype=float)
        row: dict[str, Any] = {k: o.get(k) for k in ("order", "date", "isin", "name", "ticker", "group", "amount", "price",
                                                      "savings_plan", "kind", "clustered", "range_52w", "from_high_52w")}
        for h in (30, 90, 180):
            row[f"ret_{h}"] = o.get(f"ret_{h}")
            row[f"excess_{h}"] = o.get(f"excess_{h}")
            row[f"eur_excess_{h}"] = o.get(f"eur_excess_{h}")
        mine = lot_by_order.get(o["order"], [])
        paid = sum(x["paid"] for x in mine)
        row["money_return"] = (sum(x["pnl"] for x in mine) / paid) if paid > 0 else None
        row["money_pnl"] = sum(x["pnl"] for x in mine) if mine else None
        # Tags.
        tags: dict[str, str | None] = {}
        tags["distance from the 52-week high"] = _from_high_bucket(o.get("from_high_52w"))
        if len(w) >= 150 and p:
            sma200 = float(w.tail(200).mean())
            tags["trend"] = "above its 200-day mean" if p > sma200 else "below its 200-day mean"
        if len(w) >= 21 and p:
            r20 = p / float(w.iloc[-21]) - 1.0
            row["ret_20_before"] = r20
            tags["move in the month before"] = ("after a 10%+ drop" if r20 <= -0.10 else "after a 10%+ rise" if r20 >= 0.10
                                                else "after a flat month")
        rsi = _rsi(w.tail(80)) if len(w) >= 20 else None
        row["rsi14"] = rsi
        if rsi is not None:
            tags["RSI(14)"] = "oversold (under 35)" if rsi < 35 else "overbought (over 65)" if rsi > 65 else "neutral RSI"
        vc = (vol.get(o["isin"]) or {}).get("class")
        row["vol_class"] = vc
        if vc:
            tags["volatility of the name"] = {"low": "steady name", "mid": "average volatility", "high": "volatile name"}[vc]
        tags["kind of buy"] = o.get("kind")
        tags["savings plan or one-off"] = "savings plan" if o["savings_plan"] else "one-off buy"
        if median_size > 0 and not o["savings_plan"]:
            ratio = o["amount"] / median_size
            tags["size of the order"] = ("small (under half your median)" if ratio < 0.5 else "large (over twice your median)"
                                         if ratio > 2.0 else "about your median size")
        # Build phase: days since the episode's first buy.
        start = next((s for s, e in starts_by_isin.get(o["isin"], []) if s <= o["date"] and (e is None or o["date"] <= e)), None)
        if start:
            age = (d - pd.Timestamp(start)).days
            row["days_since_first_buy"] = age
            tags["when in the build"] = ("first buy" if age == 0 else "added within a month" if age <= 30
                                         else "added within 6 months" if age <= 180 else "added after 6+ months")
        # Market at the time.
        bd = bench[bench.index < d]
        if len(bd) and not np.isnan(bench_sma.loc[bd.index[-1]] if bd.index[-1] in bench_sma.index else np.nan):
            tags["market trend"] = ("market above its 200-day mean" if float(bd.iloc[-1]) > float(bench_sma.loc[bd.index[-1]])
                                    else "market below its 200-day mean")
        if len(bd) and bd.index[-1] in bench_high.index and not np.isnan(bench_high.loc[bd.index[-1]]):
            dd = float(bd.iloc[-1]) / float(bench_high.loc[bd.index[-1]]) - 1.0
            row["market_from_high"] = dd
            tags["market drawdown"] = ("market near its high" if dd > -0.05 else "market 5 to 10% off its high" if dd > -0.10
                                       else "market 10%+ off its high")
        if vix is not None and d in vix.index and not np.isnan(vix.loc[d]):
            v = float(vix.loc[d])
            row["vix"] = v
            tags["VIX"] = "VIX under 20" if v < 20 else "VIX 20 to 30" if v < 30 else "VIX 30 or more"
        prev = d - pd.Timedelta(days=1)
        if prev in shares.index:
            n_open = int((shares.loc[prev] > EPS).sum())
            row["open_positions"] = n_open
            tags["positions open at the time"] = "up to 5 positions" if n_open <= 5 else "6 to 10 positions" if n_open <= 10 else "11+ positions"
        if regimes and regimes.get(o["date"]):
            tags["market regime at entry"] = regimes[o["date"]]
        fv = rated.get(o["order"])
        if fv:
            tags["FastValue owner rating at entry"] = fv.get("owner_decision")
        row["tags"] = {k: v for k, v in tags.items() if v}
        out.append(row)
    return out


def _rank_tags(rows: list[dict[str, Any]], split: pd.Timestamp, value: str = f"excess_{MAIN_HORIZON}") -> list[dict[str, Any]]:
    """For each tag: its buckets' results and the spread between the best and the worst bucket, with confidence."""
    tags: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for tag, b in r["tags"].items():
            tags[tag][b].append(r)
    out = []
    for tag, buckets in tags.items():
        pats = {b: _pattern(rs, value, split) for b, rs in buckets.items()}
        usable = {b: p for b, p in pats.items() if p["mean"] is not None and p["n"] >= 3}
        if len(usable) < 2:
            continue
        best = max(usable, key=lambda b: usable[b]["mean"])
        worst = min(usable, key=lambda b: usable[b]["mean"])
        cmp_ = _compare(pats[best], pats[worst], split)
        conf, why = cmp_["confidence"]
        money = {b: _mean([r.get("money_return") for r in rs]) for b, rs in buckets.items()}
        out.append({
            "tag": tag, "best": best, "worst": worst, "spread": cmp_["difference"], "t": cmp_["t"],
            "first_half": cmp_["first_half"], "second_half": cmp_["second_half"], "n": cmp_["n"],
            "confidence": conf, "why": why,
            "buckets": [{"bucket": b, "n": p["n"], "n_eff": p["n_eff"], "mean": p["mean"], "t": p["t"],
                         "first_half": p["first_half"], "second_half": p["second_half"], "eur": p["eur"],
                         "mean_180": _mean([r.get("excess_180") for r in buckets[b]]), "money_return": money.get(b)}
                        for b, p in sorted(pats.items(), key=lambda kv: -(kv[1]["mean"] if kv[1]["mean"] is not None else -9))],
        })
    tier = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda x: (tier.get(x["confidence"], 3), -abs(x["spread"] or 0.0)))
    return out


# ================================================================== exits


def _tag_exits(orders: list[dict[str, Any]], ledger: dict[str, Any], m: _Market, vol: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    starts_by_isin: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for ep in ledger["episodes"]:
        starts_by_isin[ep["isin"]].append((ep["start"], ep["end"]))
    out = []
    for o in orders:
        if o["side"] != "sell" or not o["priced"] or o["group"] == "Turbos":
            continue
        d = pd.Timestamp(o["date"])
        start = next((s for s, e in starts_by_isin.get(o["isin"], []) if s <= o["date"] and (e is None or o["date"] <= e)), None)
        row = {k: o.get(k) for k in ("order", "date", "isin", "name", "ticker", "group", "amount", "price", "gain_at_sale",
                                     "excess_90", "excess_180", "ret_90", "ret_180", "eur_excess_90", "eur_excess_180")}
        row["days_held"] = (d - pd.Timestamp(start)).days if start else None
        peak = None
        if start and o["isin"] in m.px.columns:
            path = m.px[o["isin"]].loc[pd.Timestamp(start):d].dropna()
            if len(path):
                peak = float(path.max())
        row["from_peak"] = (o["price"] / peak - 1.0) if peak and o["price"] else None
        row["vol_class"] = (vol.get(o["isin"]) or {}).get("class")
        tags: dict[str, str | None] = {}
        if row["from_peak"] is not None:
            tags["sold into"] = "weakness (15%+ below the position's high)" if row["from_peak"] <= -WEAKNESS else "strength (near the position's high)"
        if row["gain_at_sale"] is not None:
            tags["result at sale"] = "a gain" if row["gain_at_sale"] > 0 else "a loss"
        if row["days_held"] is not None:
            tags["held"] = _bucket(row["days_held"], HOLD_BUCKETS)
        if row["vol_class"]:
            tags["volatility of the name"] = {"low": "steady name", "mid": "average volatility", "high": "volatile name"}[row["vol_class"]]
        if row["days_held"] is not None and row["gain_at_sale"] is not None:
            tags["quick loser cut"] = ("cut within 90 days at a loss" if row["days_held"] <= 90 and row["gain_at_sale"] <= 0
                                       else "other sale")
        if row["vol_class"] == "low" and row["days_held"] is not None:
            tags["steady name, holding time"] = ("steady name sold within 6 months" if row["days_held"] <= SHORT_HOLD_DAYS
                                                 else "steady name held longer")
        row["tags"] = {k: v for k, v in tags.items() if v}
        out.append(row)
    return out


def _exit_summary(exits: list[dict[str, Any]], split: pd.Timestamp) -> dict[str, Any]:
    ranked = _rank_tags(exits, split, "excess_180")
    weak = [e for e in exits if e["tags"].get("sold into", "").startswith("weakness")]
    steady_quick = [e for e in exits if e["tags"].get("steady name, holding time") == "steady name sold within 6 months"]
    cut = [e for e in exits if e["tags"].get("quick loser cut") == "cut within 90 days at a loss"]

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        pat = _pattern(rows, "excess_180", split)
        conf, why = _confidence(pat["n_eff"], pat["t"], _same_sign(pat["first_half"], pat["second_half"]))
        return {"n": len(rows), "mean_excess_180": pat["mean"], "eur_excess_180": float(np.nansum([r.get("eur_excess_180") or 0.0 for r in rows])),
                "gain_at_sale_mean": _mean([r["gain_at_sale"] for r in rows]), "confidence": conf, "why": why,
                "rows": sorted(rows, key=lambda r: -abs(r.get("eur_excess_180") or 0.0))[:12]}

    return {"n": len(exits), "tags": ranked, "into_weakness": summary(weak), "steady_quick": summary(steady_quick),
            "losers_cut_fast": summary(cut)}


# ================================================================== round-trip cuts


def _cuts(trips: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = [t for t in trips if t["group"] != "Turbos" or True]
    total = sum(t["total_pnl"] for t in rows) or None

    def table(key: str, order: list[str] | None = None) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in rows:
            if t.get(key):
                groups[t[key]].append(t)
        out = []
        for label, ts in groups.items():
            closed = [t for t in ts if t["outcome"] == "sold"]
            out.append({"label": label, "n": len(ts), "closed": len(closed), "winners": sum(1 for t in ts if t["total_pnl"] > 0),
                        "hit_rate": (sum(1 for t in ts if t["total_pnl"] > 0) / len(ts)) if ts else None,
                        "median_return": _median([t["return_pct"] for t in ts]), "pnl": sum(t["total_pnl"] for t in ts),
                        "pnl_share": (sum(t["total_pnl"] for t in ts) / total) if total else None,
                        "bought": sum(t["bought"] for t in ts), "median_days": _median([t["days"] for t in ts]),
                        "avg_pnl": (sum(t["total_pnl"] for t in ts) / len(ts)) if ts else None})
        if order:
            out.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else 99)
        else:
            out.sort(key=lambda r: -r["pnl"])
        return out

    return {
        "holding period": table("hold_bucket", [b[2] for b in HOLD_BUCKETS]),
        "number of buys": table("buys_bucket", [b[2] for b in BUY_COUNT_BUCKETS]),
        "time to build": table("build_bucket", [b[2] for b in BUILD_BUCKETS]),
        "volatility class": table("vol_class", ["low", "mid", "high"]),
        "kind of instrument": table("group"),
        "bucket": table("bucket", ["Core", "Satellite", "Metals", "Speculative"]),
        "entry drawdown": table("entry_drawdown_bucket", ["within 5% of the 52-week high", "5 to 20% below the high",
                                                         "20 to 40% below the high", "40%+ below the high"]),
    }


# ================================================================== headline lessons


def _lessons(style: dict[str, Any], entries: list[dict[str, Any]], exits: dict[str, Any], cuts: dict[str, Any]) -> list[dict[str, Any]]:
    """Short, honest statements: the number, what it suggests, and how sure we can be."""
    out = []
    rt, hr, pc = style["round_trips"], style["hit_rate"], style["concentration_of_pnl"]
    if rt["closed"]:
        out.append({"key": "speed", "title": "How fast you trade",
                    "text": f"{_n(rt['closed'], 'closed round trip', 'closed round trips')} with a median holding period of {rt['median_days_closed']:.0f} days; "
                            f"{rt['share_closed_quickly']:.0%} were closed within {rt['quick_days']} days.",
                    "confidence": "exact"})
    if hr["wins"] + hr["losses"]:
        out.append({"key": "hits", "title": "Hit rate and payoff",
                    "text": f"{hr['rate']:.0%} of closed round trips made money ({_n(hr['wins'], 'win', 'wins')}, {_n(hr['losses'], 'loss', 'losses')}). "
                            + (f"Average win €{hr['avg_win']:,.0f} against an average loss of €{abs(hr['avg_loss']):,.0f} "
                               f"(payoff {hr['payoff']:.2f})." if hr["avg_win"] is not None and hr["avg_loss"] else ""),
                    "confidence": "exact"})
    if pc["positions_with_gains"]:
        out.append({"key": "pareto", "title": "Where the money came from",
                    "text": f"{pc['positions_for_80pct_of_gains']} of {pc['positions_with_gains']} winning positions made 80% of the gross gains"
                            + (f"; the top 8 round trips are {pc['top8_share']:.0%} of the total P&L." if pc["top8_share"] is not None else "."),
                    "confidence": "exact"})
    py = style["pyramiding"]
    if py["trips_with_5plus_buys"]:
        out.append({"key": "pyramiding", "title": "Positions you kept adding to",
                    "text": f"{py['trips_with_5plus_buys']} positions were built with 5 or more buys"
                            + (f" and carry {py['pnl_share_5plus_buys']:.0%} of the total P&L." if py["pnl_share_5plus_buys"] is not None else "."),
                    "confidence": "exact"})
    for e in entries[:3]:
        if e["confidence"] in ("high", "medium"):
            out.append({"key": f"entry:{e['tag']}", "title": f"Entries: {e['tag']}",
                        "text": f"Buys tagged '{e['best']}' did {e['spread']:+.1%} better against MSCI World after {MAIN_HORIZON} days than "
                                f"buys tagged '{e['worst']}' ({e['why']}).", "confidence": e["confidence"]})
    if not any(x["key"].startswith("entry:") for x in out) and entries:
        e = entries[0]
        out.append({"key": "entry:thin", "title": "Entries: no tag is robust yet",
                    "text": f"The widest gap is '{e['tag']}' ({e['best']} against {e['worst']}, {e['spread']:+.1%} after {MAIN_HORIZON} days), "
                            f"but {e['why']}. Keep what you do and watch this table as trades accumulate.", "confidence": "low"})
    w = exits.get("into_weakness") or {}
    if w.get("n"):
        out.append({"key": "exits:weakness", "title": "Sales into weakness",
                    "text": f"{w['n']} sales came 15%+ below the position's high; 180 days later those names had moved "
                            f"{w['mean_excess_180']:+.1%} against MSCI World on average ({w['why']}).", "confidence": w["confidence"]})
    sq = exits.get("steady_quick") or {}
    if sq.get("n"):
        out.append({"key": "exits:steady", "title": "Steady names sold within six months",
                    "text": f"{_n(sq['n'], 'sale', 'sales')} of low-volatility names came within six months of the first buy, at "
                            f"{sq['gain_at_sale_mean']:+.1%} on average; 180 days later they had moved {sq['mean_excess_180']:+.1%} against MSCI World "
                            f"({sq['why']}).", "confidence": sq["confidence"]})
    return out


# ================================================================== entry point


def build_profile(ledger: dict[str, Any], m: _Market, a: dict[str, Any], perf: dict[str, Any],
                  orders: list[dict[str, Any]], trips: list[dict[str, Any]], instruments: list[dict[str, Any]],
                  lots: list[dict[str, Any]], beh: dict[str, Any], fastvalue: dict[str, Any] | None = None,
                  vix: pd.Series | None = None, regimes: dict[str, str] | None = None) -> dict[str, Any]:
    """Style numbers, tagged entries and exits, round-trip cuts and headline lessons."""
    split = beh["split_date"]
    vol = volatility_classes(ledger, m)
    _trip_extras(ledger, trips, m, vol["classes"], orders)
    style = style_numbers(ledger, m, a, perf, orders, trips, instruments, beh)
    tagged = _tag_entries(orders, ledger, m, a, vol["classes"], lots, vix, fastvalue, regimes)
    entries = _rank_tags(tagged, split)
    exits_rows = _tag_exits(orders, ledger, m, vol["classes"])
    exits = _exit_summary(exits_rows, split)
    cuts = _cuts(trips)
    return {
        "style": style,
        "volatility": {"bounds": vol["bounds"], "classes": {i: r["class"] for i, r in vol["classes"].items()},
                       "vols": {i: r["vol"] for i, r in vol["classes"].items()}},
        "entries": {"n": len(tagged), "tags": entries, "rows": tagged},
        "exits": exits,
        "cuts": cuts,
        "lessons": _lessons(style, entries, exits, cuts),
        "protocol": "Tags are judged on the return against MSCI World 90 days after each buy (180 days after each sale). "
                    "Orders in one name and month count once; the best bucket is compared with the worst with Welch's t; "
                    "a tag is more than chance only with at least 10 cases, |t| of 1.5 or more and the same sign in both "
                    "halves of the period. Nothing here changes FastValue's scoring.",
        "min_cases": MIN_CASES,
    }
