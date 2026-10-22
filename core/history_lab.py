"""Rules lab: your own order history replayed under rules you did not follow, measured against what you did.

The track record's what-ifs change one lot at a time. This replays the whole
account day by day, on the same calendar and prices as the actual account:
deposits, withdrawals, card spending, dividends, interest and taxes as they
happened; your buy and sell orders filtered, scaled or delayed by the rules;
and overlays at the close (trims above a weight cap, stop-loss, take-profit,
idle cash swept into a core index fund). The result is a daily account value
that compares like for like with the actual one: end value, time-weighted
return, drawdown and what the rule changed.

Rules (one scenario is a ``Rules`` record; the presets are the ones an
investor with a trader's habits most often asks about):

- exclude a kind of instrument (turbos), skip one-off buys below a size, or
  keep only savings-plan buys (the cash stays, or goes to the core fund);
- cap any name at a share of the account: later buys are scaled down to the
  cap and month-end trims sell what is above it;
- hold every position at least N days, or only the steady (low-volatility)
  names at least N days: a sale that comes earlier is delayed to that day;
- stop-loss and take-profit on the position's average cost (the whole
  position goes at the close that crosses the level);
- cash sweep: at month-end, cash above a reserve share buys the core fund;
  a later buy that needs cash sells core first.

Protocol, fixed before any run (the lab's and the overlays'): a rule is
"adopted" only if it beats what you did over the whole period and when applied
in each half of the period alone, with a t statistic of at least 2 across the
names it changed (one observation per instrument). Everything here is in
sample on one investor's history, so a winning rule is a hypothesis for the
next year, not proof. Taxes are not re-simulated (the withheld tax of an actual
sale follows its shares); every simulated order pays the account's typical fee;
delayed sales and overlays trade at the day's EUR close.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from core.ledger import EPS
from core.track_record import _Market, _annual, _confidence, _same_sign, _tstat

logger = logging.getLogger(__name__)

MIN_CORE_TRADE = 50.0  # EUR: smaller sweeps are not worth a fee
DEFAULT_FEE = 1.0
SIGNIFICANT_T = 2.0
MAX_CUSTOM = 6
HISTORY_FILE = "rules_lab_history.json"  # under STORE_DIR (git-ignored): each import's verdicts, no amounts
CANDIDATE_RUN = 3  # monthly imports in a row ahead in both halves before a rule is worth writing down


@dataclass(frozen=True)
class Rules:
    key: str
    label: str
    description: str = ""
    exclude_groups: tuple[str, ...] = ()
    skip_discretionary: bool = False
    min_buy: float | None = None
    max_weight: float | None = None
    trim: bool = True
    min_hold_days: int | None = None
    low_vol_hold_days: int | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    cash_reserve: float | None = None  # share of the account kept in cash; the rest buys the core fund at month-end
    preset: bool = True

    @property
    def is_actual(self) -> bool:
        return not (self.exclude_groups or self.skip_discretionary or self.min_buy or self.max_weight is not None
                    or self.min_hold_days or self.low_vol_hold_days or self.stop_loss or self.take_profit
                    or self.cash_reserve is not None)

    def validate(self) -> "Rules":
        if self.max_weight is not None and not 0.01 <= self.max_weight <= 1.0:
            raise ValueError("The weight cap must be between 1% and 100%")
        for name in ("min_hold_days", "low_vol_hold_days"):
            v = getattr(self, name)
            if v is not None and not 1 <= int(v) <= 3650:
                raise ValueError("Holding periods must be between 1 and 3650 days")
        for name in ("stop_loss", "take_profit"):
            v = getattr(self, name)
            if v is not None and not 0.02 <= v <= 5.0:
                raise ValueError("Stop-loss and take-profit levels must be between 2% and 500%")
        if self.cash_reserve is not None and not 0.0 <= self.cash_reserve <= 1.0:
            raise ValueError("The cash reserve must be between 0% and 100%")
        if self.min_buy is not None and not 0.0 < self.min_buy <= 1e7:
            raise ValueError("The minimum buy must be a positive amount")
        return self


def presets(median_discretionary_buy: float | None) -> list[Rules]:
    small = float(round(median_discretionary_buy or 0.0, -1)) or None
    out = [
        Rules("actual", "What you did", "Your orders as they happened, through the same engine (a check: it must match the account)."),
        Rules("no_turbos", "No turbos", "Every knock-out product skipped; the cash stays in the account.", exclude_groups=("Turbos",)),
        Rules("hold_180", "Hold at least 6 months", "A sale within 180 days of the first buy is delayed to day 180.", min_hold_days=180),
        Rules("hold_365", "Hold at least a year", "A sale within a year of the first buy is delayed to the anniversary.", min_hold_days=365),
        Rules("stable_3y", "Steady names held 3 years", "Low-volatility names (the calmest third at entry) are held at least three years; the rest as you did.",
              low_vol_hold_days=1095),
        Rules("cap_10", "No name above 10%", "Buys are scaled down to a 10% share of the account and month-end trims sell what is above it.",
              max_weight=0.10),
        Rules("cash_to_core", "Idle cash into MSCI World", "At month-end, cash above a 10% reserve buys the core fund; a buy that needs cash sells core first.",
              cash_reserve=0.10),
        Rules("core_satellite", "Core and satellite", "No turbos, no name above 10% (trimmed to the core fund), idle cash above 10% in the core fund.",
              exclude_groups=("Turbos",), max_weight=0.10, cash_reserve=0.10),
        Rules("plans_only", "Savings plans only", "Every one-off buy skipped; the money goes to the core fund with the cash sweep.",
              skip_discretionary=True, cash_reserve=0.10),
        Rules("stop_25", "Stop-loss at -25%", "A position is sold at the first close 25% below its average cost.", stop_loss=0.25),
        Rules("take_100", "Take profit at +100%", "A position is sold at the first close 100% above its average cost.", take_profit=1.0),
    ]
    if small:
        out.insert(2, Rules("no_small_bets", "No one-off buys below your median size",
                            f"One-off buys under €{small:,.0f} (your median) are skipped; the money goes to the core fund.",
                            min_buy=small, cash_reserve=0.10))
    return out


def rules_from(body: dict[str, Any], key: str = "custom", label: str | None = None) -> Rules:
    """A ``Rules`` record from the dashboard's form (unknown fields ignored, values checked)."""
    def num(name: str, scale: float = 1.0) -> float | None:
        v = body.get(name)
        if v in (None, "", False):
            return None
        try:
            return float(v) * scale
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc

    def days(name: str) -> int | None:
        v = num(name)
        return int(v) if v is not None else None

    groups = tuple(str(g) for g in (body.get("exclude_groups") or []) if str(g))
    if body.get("no_turbos"):
        groups = tuple(dict.fromkeys(groups + ("Turbos",)))
    r = Rules(key=key, label=label or str(body.get("label") or "Your rule")[:80], description=str(body.get("description") or "")[:300],
              exclude_groups=groups, skip_discretionary=bool(body.get("skip_discretionary")), min_buy=num("min_buy"),
              max_weight=num("max_weight", 0.01), trim=bool(body.get("trim", True)), min_hold_days=days("min_hold_days"),
              low_vol_hold_days=days("low_vol_hold_days"), stop_loss=num("stop_loss", 0.01), take_profit=num("take_profit", 0.01),
              cash_reserve=num("cash_reserve", 0.01), preset=False)
    return r.validate()


# ================================================================== inputs


@dataclass
class _Inputs:
    days: pd.DatetimeIndex
    isins: list[str]
    idx: dict[str, int]
    px: np.ndarray
    bench: np.ndarray
    actual_shares: np.ndarray
    base_cash: np.ndarray
    flows: np.ndarray
    orders: list[list[dict[str, Any]]]
    dividends: list[list[tuple[int, float]]]
    extra: list[list[dict[str, Any]]]
    month_end: np.ndarray
    groups: list[str]
    names: list[str]
    vol_class: list[str | None]
    fee: float
    split: pd.Timestamp
    first_trade: pd.Timestamp
    core_label: str


def prepare(ledger: dict[str, Any], m: _Market, a: dict[str, Any], vol_class: dict[str, str | None] | None,
            split: pd.Timestamp, first_trade: pd.Timestamp) -> _Inputs:
    days = m.days
    n = len(days)
    pos = {d: i for i, d in enumerate(days)}
    isins = list(a["shares"].columns)
    idx = {isin: i for i, isin in enumerate(isins)}
    px = m.px.reindex(columns=isins).to_numpy(dtype=float)
    bench = m.bench.to_numpy(dtype=float)
    actual_shares = a["shares"].to_numpy(dtype=float)
    base_cash = pd.Series(ledger["cash_by_date"]).rename(index=pd.Timestamp).reindex(days, fill_value=0.0).to_numpy(dtype=float).copy()
    orders: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    fees = []
    for o in ledger["orders"]:
        t = pos.get(pd.Timestamp(o["date"]))
        if t is None or o["isin"] not in idx:
            continue
        info = ledger["instruments"][o["isin"]]
        if info["asset_class"] in ("PRIVATE_FUND", "SYNTHETIC"):
            continue
        base_cash[t] -= o["amount"] + o["fee"] + o["tax"]
        amount = abs(o["amount"])
        orders[t].append({"isin": idx[o["isin"]], "side": o["side"], "shares": float(o["shares"]), "amount": amount,
                          "fee": -float(o["fee"]), "tax": -float(o["tax"]), "price": float(o["price"] or 0.0),
                          "savings_plan": bool(o["savings_plan"]), "order": o["order"], "ts": str(o.get("ts") or "")})
        if not o["savings_plan"] and o["fee"]:
            fees.append(-float(o["fee"]))
    dividends: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for inc in ledger["income"]:
        if inc["kind"] == "dividend" and inc.get("isin") in idx:
            t = pos.get(pd.Timestamp(inc["date"]))
            if t is not None:
                dividends[t].append((idx[inc["isin"]], float(inc["amount"] + inc["tax"])))
    extra: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for f in ledger["flows"]:
        if f["kind"] in ("securities_out", "securities_in") and f.get("isin") in idx:
            t = pos.get(pd.Timestamp(f["date"]))
            if t is not None:
                extra[t].append({"isin": idx[f["isin"]], "kind": f["kind"], "shares": float(f["shares"]),
                                 "cost": float(f["amount"]) if f["kind"] == "securities_in" else None})
    for c in ledger["changes"]:
        if c.get("ratio") and c["isin"] in idx:
            t = pos.get(pd.Timestamp(c["date"]))
            if t is not None:
                extra[t].append({"isin": idx[c["isin"]], "kind": "ratio", "ratio": float(c["ratio"])})
    for day_orders in orders:
        day_orders.sort(key=lambda o: o["ts"])  # the account's own order within the day (a buy can fund a sale, and the reverse)
    month_end = np.zeros(n, dtype=bool)
    for t in range(n - 1):
        month_end[t] = days[t + 1].month != days[t].month
    month_end[-1] = True
    from core.track_record import _group

    groups = [_group(ledger["instruments"][i]["name"], ledger["instruments"][i]["asset_class"]) for i in isins]
    names = [ledger["instruments"][i]["name"] or i for i in isins]
    vc = [(vol_class or {}).get(i) for i in isins]
    fee = float(np.median(fees)) if fees else DEFAULT_FEE
    return _Inputs(days, isins, idx, px, bench, actual_shares, base_cash, a["flows"].to_numpy(dtype=float), orders,
                   dividends, extra, month_end, groups, names, vc, fee, split, first_trade, m.bench_label)


# ================================================================== simulation


@dataclass
class _Sim:
    value: np.ndarray
    cash: np.ndarray
    core_value: np.ndarray
    contributions: dict[str, float]
    counters: dict[str, Any] = field(default_factory=dict)
    end_shares: np.ndarray | None = None


def simulate(inp: _Inputs, rules: Rules, active: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> _Sim:
    n, k = inp.px.shape
    cash = 0.0
    shares = np.zeros(k)
    cost = np.zeros(k)
    opened = np.full(k, -1)
    net_cash = np.zeros(k)
    core_units = 0.0
    core_net = 0.0
    pending: list[tuple[int, int, float, float]] = []  # (release day, isin, shares, tax per share)
    value = np.zeros(n)
    cash_s = np.zeros(n)
    core_s = np.zeros(n)
    c = {"skipped_buys": 0, "skipped_eur": 0.0, "scaled_buys": 0, "scaled_eur": 0.0, "deferred_sells": 0,
         "unfilled_sells": 0, "unfunded_buys": 0, "unfunded_eur": 0.0, "funding_scaled": 0, "trims": 0, "stops": 0,
         "takes": 0, "core_buys": 0, "core_sells": 0, "unpriced_deferrals": 0}
    excluded = set(rules.exclude_groups)
    # The rule applies on days in [start, end). searchsorted places a date that is not on the calendar (a weekend,
    # a time of day) at the next calendar day; an exact lookup would miss it and stretch the window to the whole period.
    lo = int(inp.days.searchsorted(active[0], side="left")) if active else 0
    hi = int(inp.days.searchsorted(active[1], side="left")) if active else n

    def priced(t: int, i: int) -> float | None:
        v = inp.px[t, i]
        return float(v) if np.isfinite(v) else None

    def position_values(t: int) -> np.ndarray:
        p = inp.px[t]
        vals = np.where(np.isfinite(p), shares * np.nan_to_num(p), cost)
        return vals

    def total_value(t: int) -> float:
        b = inp.bench[t]
        return cash + float(position_values(t).sum()) + (core_units * b if np.isfinite(b) else 0.0)

    def sell_core(t: int, need: float) -> None:
        nonlocal cash, core_units, core_net
        b = inp.bench[t]
        if core_units <= 0 or not np.isfinite(b) or b <= 0:
            return
        units = min(core_units, (need + inp.fee) / b)
        proceeds = units * b
        core_units -= units
        cash += proceeds - inp.fee
        core_net += proceeds - inp.fee
        c["core_sells"] += 1

    def sell(t: int, i: int, q: float, price: float, fee: float, tax: float) -> None:
        nonlocal cash
        q = min(q, shares[i])
        if q <= EPS:
            return
        proceeds = q * price
        cash += proceeds - fee - tax
        net_cash[i] += proceeds - fee - tax
        cost[i] -= cost[i] * q / shares[i] if shares[i] > EPS else 0.0
        shares[i] -= q
        if shares[i] <= EPS:
            shares[i], cost[i], opened[i] = 0.0, 0.0, -1

    for t in range(n):
        on = lo <= t < hi
        cash += inp.base_cash[t]
        for i, amt in inp.dividends[t]:
            actual = inp.actual_shares[t - 1, i] if t > 0 else 0.0
            if actual > EPS:
                adj = (shares[i] / actual - 1.0) * amt
                if abs(adj) > 1e-9:
                    cash += adj
                    net_cash[i] += adj
        for e in inp.extra[t]:
            i = e["isin"]
            if e["kind"] == "securities_out":
                q = min(e["shares"], shares[i])
                if shares[i] > EPS:
                    cost[i] -= cost[i] * q / shares[i]
                shares[i] -= q
                if shares[i] <= EPS:
                    shares[i], cost[i], opened[i] = 0.0, 0.0, -1
            elif e["kind"] == "securities_in":
                shares[i] += e["shares"]
                cost[i] += e["cost"] or 0.0
                if opened[i] < 0:
                    opened[i] = t
            elif e["kind"] == "ratio" and shares[i] > EPS:
                shares[i] *= e["ratio"]
        # Orders in the account's own time order. A buy may be paid for by a sale later the same day (the
        # account did exactly that), so the day's sale proceeds still to come count as available cash.
        day_orders = inp.orders[t]
        hold_need = (max(rules.min_hold_days or 0, (rules.low_vol_hold_days or 0)) if on else 0)
        upcoming = 0.0
        if day_orders and not rules.is_actual:
            sh, op = shares.copy(), opened.copy()
            for o in day_orders:
                i = o["isin"]
                if o["side"] == "buy":
                    sh[i] += o["shares"]
                    if op[i] < 0:
                        op[i] = t
                    continue
                q = min(o["shares"], sh[i])
                if q <= EPS:
                    continue
                need = max(rules.min_hold_days or 0, (rules.low_vol_hold_days or 0) if inp.vol_class[i] == "low" else 0) if on and op[i] >= 0 else 0
                if need and t - op[i] < need:
                    continue
                upcoming += q * o["price"] - (o["fee"] if q >= o["shares"] - EPS else inp.fee) - o["tax"] * q / o["shares"]
                sh[i] -= q
        for o in day_orders:
            i = o["isin"]
            if o["side"] == "sell":
                q = min(o["shares"], shares[i])
                tax_ps = o["tax"] / o["shares"] if o["shares"] > 0 else 0.0
                net = q * o["price"] - (o["fee"] if q >= o["shares"] - EPS else inp.fee) - tax_ps * q
                if q <= EPS:
                    c["unfilled_sells"] += 1
                    continue
                if on and opened[i] >= 0:
                    need = max(rules.min_hold_days or 0, (rules.low_vol_hold_days or 0) if inp.vol_class[i] == "low" else 0)
                    if need and t - opened[i] < need:
                        pending.append((opened[i] + need, i, q, tax_ps))
                        c["deferred_sells"] += 1
                        continue
                upcoming = max(upcoming - net, 0.0)
                sell(t, i, q, o["price"], o["fee"] if q >= o["shares"] - EPS else inp.fee, tax_ps * q)
                continue
            amount, q = o["amount"], o["shares"]
            if on:
                if inp.groups[i] in excluded or (rules.skip_discretionary and not o["savings_plan"]) \
                        or (rules.min_buy and not o["savings_plan"] and amount < rules.min_buy):
                    c["skipped_buys"] += 1
                    c["skipped_eur"] += amount
                    continue
                if rules.max_weight is not None:
                    total = total_value(t)
                    held = float(position_values(t)[i])
                    allowed = rules.max_weight * total - held
                    if allowed < min(amount, 1.0):
                        c["skipped_buys"] += 1
                        c["skipped_eur"] += amount
                        continue
                    if allowed < amount:
                        s = allowed / amount
                        c["scaled_buys"] += 1
                        c["scaled_eur"] += amount - allowed
                        amount, q = allowed, q * s
            fee = o["fee"] if o["fee"] > 0 or o["savings_plan"] else 0.0
            tax = o["tax"] * (amount / o["amount"]) if o["amount"] > 0 else 0.0
            need = amount + fee + tax
            if cash + upcoming < need - 1e-6 and not rules.is_actual:
                if core_units > 0:
                    sell_core(t, need - cash - upcoming)
                if cash + upcoming < need - 1e-6:
                    avail = max(cash + upcoming - fee - tax, 0.0)
                    if avail < 1.0:
                        c["unfunded_buys"] += 1
                        c["unfunded_eur"] += amount
                        continue
                    s = avail / amount
                    c["funding_scaled"] += 1
                    amount, q = avail, q * s
                    need = amount + fee + tax
            cash -= need
            net_cash[i] -= need
            shares[i] += q
            cost[i] += amount
            if opened[i] < 0:
                opened[i] = t
        # Delayed sales whose day has come.
        if pending:
            keep = []
            for release, i, q, tax_ps in pending:
                if release > t and t < n - 1:
                    keep.append((release, i, q, tax_ps))
                    continue
                p = priced(t, i)
                if p is None:
                    if shares[i] > EPS and cost[i] > 0:
                        p = cost[i] / shares[i]
                        c["unpriced_deferrals"] += 1
                    else:
                        continue
                sell(t, i, q, p, inp.fee, tax_ps * min(q, shares[i]))
            pending = keep
        # Overlays at the close.
        if on and (rules.stop_loss or rules.take_profit):
            for i in np.nonzero(shares > EPS)[0]:
                p = priced(t, int(i))
                if p is None or cost[i] <= 0:
                    continue
                avg = cost[i] / shares[i]
                if rules.stop_loss and p <= avg * (1.0 - rules.stop_loss):
                    sell(t, int(i), shares[i], p, inp.fee, 0.0)
                    c["stops"] += 1
                elif rules.take_profit and p >= avg * (1.0 + rules.take_profit):
                    sell(t, int(i), shares[i], p, inp.fee, 0.0)
                    c["takes"] += 1
        if on and inp.month_end[t]:
            if rules.max_weight is not None and rules.trim:
                total = total_value(t)
                vals = position_values(t)
                for i in np.nonzero(vals > rules.max_weight * total + 1.0)[0]:
                    p = priced(t, int(i))
                    if p is None:
                        continue
                    excess = vals[i] - rules.max_weight * total
                    sell(t, int(i), excess / p, p, inp.fee, 0.0)
                    c["trims"] += 1
            if rules.cash_reserve is not None:
                total = total_value(t)
                b = inp.bench[t]
                spare = cash - rules.cash_reserve * total
                if spare > MIN_CORE_TRADE and np.isfinite(b) and b > 0:
                    core_units += (spare - inp.fee) / b
                    cash -= spare
                    core_net -= spare
                    c["core_buys"] += 1
        b = inp.bench[t]
        core_val = core_units * b if np.isfinite(b) else 0.0
        value[t] = cash + float(position_values(t).sum()) + core_val
        cash_s[t] = cash
        core_s[t] = core_val
    contributions = {inp.isins[i]: float(position_values(n - 1)[i] + net_cash[i]) for i in range(k)}
    contributions["__core__"] = float(core_s[-1] + core_net)
    for key in ("skipped_eur", "scaled_eur", "unfunded_eur"):
        c[key] = round(float(c[key]), 2)
    return _Sim(value, cash_s, core_s, contributions, c, shares.copy())


# ================================================================== metrics and verdicts


def _metrics(value: np.ndarray, inp: _Inputs) -> dict[str, Any]:
    prev = np.concatenate([[0.0], value[:-1]])
    den = prev + inp.flows
    r = np.where(den > 1.0, value / np.where(den > 1.0, den, 1.0) - 1.0, 0.0)
    r = pd.Series(r, index=inp.days)
    r = r[r.index >= inp.first_trade]
    growth = (1.0 + r).cumprod()
    dd = float((growth / growth.cummax() - 1.0).min()) if len(growth) else None
    weekly = growth.resample("W-FRI").last().pct_change().dropna()
    vol = float(weekly.std(ddof=1) * math.sqrt(52)) if len(weekly) > 8 else None
    monthly = growth.resample("ME").last().pct_change().dropna()
    return {"end_value": float(value[-1]), "gain": float(value[-1] - inp.flows.sum()), "twr_annual": _annual(r),
            "twr_total": float(growth.iloc[-1] - 1.0) if len(growth) else None, "max_drawdown": dd, "volatility": vol,
            "worst_month": float(monthly.min()) if len(monthly) else None, "_growth": growth}


def _weekly(s: pd.Series, digits: int = 2) -> list[dict[str, Any]]:
    return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), digits)}
            for d, v in s.resample("W-FRI").last().dropna().items()]


def run_scenarios(inp: _Inputs, scenarios: list[Rules]) -> dict[str, Any]:
    actual = simulate(inp, Rules("actual", "What you did"))
    base = _metrics(actual.value, inp)
    rows = []
    curves = {"actual": _weekly(pd.Series(actual.value, index=inp.days)[inp.days >= inp.first_trade])}
    for rules in scenarios:
        if rules.is_actual and rules.key != "actual":
            continue
        sim = simulate(inp, rules)
        met = _metrics(sim.value, inp)
        first = simulate(inp, rules, (inp.days[0], inp.split)) if not rules.is_actual else sim
        second = simulate(inp, rules, (inp.split, inp.days[-1] + pd.Timedelta(days=1))) if not rules.is_actual else sim
        diff = met["end_value"] - base["end_value"]
        d1 = float(first.value[-1] - actual.value[-1]) if not rules.is_actual else 0.0
        d2 = float(second.value[-1] - actual.value[-1]) if not rules.is_actual else 0.0
        per_name = {k: v - actual.contributions.get(k, 0.0) for k, v in sim.contributions.items()}
        changed = {k: v for k, v in per_name.items() if abs(v) > 0.01}
        t = _tstat(list(changed.values())) if len(changed) >= 3 else None
        agree = _same_sign(d1, d2)
        conf, why = _confidence(len(changed), t, agree) if not rules.is_actual else ("exact", "the account itself")
        adopt = bool(not rules.is_actual and diff > 0 and d1 > 0 and d2 > 0 and t is not None and t >= SIGNIFICANT_T)
        if rules.is_actual:
            verdict = "Reference."
        elif adopt:
            verdict = f"Ahead over the whole period and in both halves, t {t:.1f} across {len(changed)} names: worth testing forward."
        elif diff > 0 and d1 > 0 and d2 > 0:
            verdict = f"Ahead in both halves but not significant ({'t ' + format(t, '.1f') if t is not None else 'too few names'}): a lead to watch, not a rule yet."
        elif diff > 0:
            verdict = "Ahead overall but not in both halves: the gain came from one period."
        else:
            verdict = "Behind what you did."
        rows.append({
            "key": rules.key, "label": rules.label, "description": rules.description, "preset": rules.preset,
            "end_value": met["end_value"], "gain": met["gain"], "difference": diff,
            "difference_pct": diff / base["end_value"] if base["end_value"] else None,
            "twr_annual": met["twr_annual"], "max_drawdown": met["max_drawdown"], "volatility": met["volatility"],
            "worst_month": met["worst_month"], "first_half": d1, "second_half": d2, "weaker_half": min(d1, d2),
            "t": t, "names_changed": len(changed), "confidence": conf, "why": why, "adopt": adopt, "verdict": verdict,
            "changes": sim.counters, "end_cash_share": float(sim.cash[-1] / sim.value[-1]) if sim.value[-1] > 1 else None,
            "end_core_share": float(sim.core_value[-1] / sim.value[-1]) if sim.value[-1] > 1 else None,
            "largest_effects": sorted(({"name": inp.names[inp.idx[k]] if k in inp.idx else inp.core_label, "eur": v}
                                       for k, v in changed.items()), key=lambda x: -abs(x["eur"]))[:6],
            "rules": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(rules).items()},
        })
        curves[rules.key] = _weekly(pd.Series(sim.value, index=inp.days)[inp.days >= inp.first_trade])
    rows.sort(key=lambda r: (r["key"] != "actual", -(r["weaker_half"] if r["key"] != "actual" else 0.0)))
    best = next((r for r in rows if r["key"] != "actual"), None)
    return {
        "actual": {k: v for k, v in base.items() if not k.startswith("_")},
        "scenarios": rows, "curves": curves, "best": best["key"] if best else None,
        "adopted": [r["key"] for r in rows if r["adopt"]],
        "fee_per_order": inp.fee, "core": inp.core_label, "split": inp.split.strftime("%Y-%m-%d"),
        "protocol": ("Every scenario replays your deposits, withdrawals, dividends and taxes as they happened and your orders "
                     "under the rule, on the same calendar and prices in the account's currency; delayed sales and overlays trade at the close and pay "
                     f"the account's typical fee (€{inp.fee:.2f}). A rule is adopted only if it beats what you did over the "
                     "whole period and when applied in each half of the period alone, with t of at least 2 across the names it "
                     "changed. All of it is in sample on one history: a winner here is a hypothesis to test forward."),
        "verdict": (f"{best['label']} is the strongest rule by its weaker half ({'-' if round(best['weaker_half']) < 0 else '+'}€{abs(best['weaker_half']):,.0f}); "
                    + ("it passes the protocol." if best["adopt"] else "it does not pass the protocol, so keep what you do and watch it.")
                    if best else "No scenario ran."),
    }


def run_rules_lab(ledger: dict[str, Any], m: _Market, a: dict[str, Any], split: pd.Timestamp, first_trade: pd.Timestamp,
                  vol_class: dict[str, str | None] | None, median_discretionary_buy: float | None,
                  custom: list[Rules] | None = None, include_presets: bool = True) -> dict[str, Any]:
    inp = prepare(ledger, m, a, vol_class, split, first_trade)
    scenarios = (presets(median_discretionary_buy) if include_presets else [Rules("actual", "What you did")]) + list(custom or [])
    result = run_scenarios(inp, scenarios)
    # Self-check: the engine's replay of the actual orders must match the account valued by the track record.
    actual_curve = a["account"].to_numpy(dtype=float)
    engine_curve = simulate(inp, Rules("actual", "What you did")).value
    gap = float(np.nanmax(np.abs(actual_curve - engine_curve))) if len(actual_curve) else 0.0
    result["engine_check"] = {"max_gap_eur": gap, "ok": gap <= max(1.0, 0.001 * float(np.nanmax(np.abs(actual_curve)) or 1.0))}
    return result


# ================================================================== month to month


def track_runs(lab: dict[str, Any], export_end: str | None) -> None:
    """Record this import's verdicts and give each preset rule its run of monthly imports ahead in both halves.

    The replay is in sample, so one run proves nothing; the plan (next_steps.md, P0) is to re-run it after each
    month's export and to write a rule down only when it stays ahead in both halves month after month. One entry per
    export, keyed by its last booking day (a re-run of the same export replaces it); only the latest export of a month
    counts. Stored: labels, shares of the account and yes/no, never an amount.
    """
    if not export_end or not lab or not lab.get("scenarios"):
        return
    path = settings.STORE_DIR / HISTORY_FILE
    try:
        history = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        history = {}
    history[str(export_end)[:10]] = {
        "recorded": datetime.now().isoformat(timespec="seconds"),
        "rules": {r["key"]: {"label": r["label"], "adopt": bool(r["adopt"]), "difference_pct": r.get("difference_pct"),
                             "ahead_both": bool(r["difference"] > 0 and r["first_half"] > 0 and r["second_half"] > 0)}
                  for r in lab["scenarios"] if r["key"] != "actual" and r.get("preset", True)},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=1, sort_keys=True))
    except OSError as exc:
        logger.warning("Could not store the rules lab history: %s", exc)
    by_month: dict[str, dict[str, Any]] = {}
    for end in sorted(history):
        by_month[end[:7]] = history[end]
    imports = [by_month[k] for k in sorted(by_month, reverse=True)]
    for r in lab["scenarios"]:
        if r["key"] == "actual" or not r.get("preset", True):
            continue
        run = 0
        for snap in imports:
            if not (snap["rules"].get(r["key"]) or {}).get("ahead_both"):
                break
            run += 1
        r["run"] = run
        if run >= CANDIDATE_RUN and not r["adopt"]:
            r["verdict"] += (f" Ahead in both halves in {run} monthly imports in a row: a candidate to write down and "
                             "follow on new trades.")
    lab["imports_recorded"] = len(imports)
