"""Track record of an imported broker account: how it did, what made trades good or bad, and what-ifs.

Built on the ledger of ``core.ledger`` (exact accounting from the export) and
daily prices from Yahoo converted to EUR at ECB rates (``data.instruments``,
``data.fx``). Everything is in EUR, the account's currency.

What it answers:

- Performance: daily account value; the invested part (stocks and funds, cash
  left out) against MSCI World, the S&P 500 and the Nasdaq 100, time-weighted
  and with the same buys and sales in each index (money-weighted); and, apart,
  the whole account with its cash: time-weighted (TWR), money-weighted (XIRR),
  Modified Dietz per year, interest, and every deposit put into each index.
- Per instrument and per round trip (first buy until the position is back at
  zero): money in and out, dividends, fees and taxes, total P&L, holding time,
  and the worst and best point while held (MAE and MFE).
- Each order: the price path 30, 90 and 180 days later against MSCI World, and
  where each buy sat in its 52-week range.
- Behaviour: disposition effect (winners sold faster than losers), adding to
  losers, sizing, concentration, turnover and fees, asset classes, savings
  plans against one-off buys, clustered impulse buys, cash drag.
- What-ifs on the same external cash flows and dates: every buy and sale in
  each index instead, every deposit into each index (Cash section), the sold
  shares kept until today, no turbos, and stop-loss,
  take-profit and minimum-holding rules applied to the actual entries. The
  rules are judged by the lab's protocol, written before any run: the
  improvement must hold in both halves of the period with |t| of at least 2,
  otherwise the verdict is to keep what you do.
- Optionally, FastValue's own owner rating at each entry and its sell rule on
  the actual entries, replayed point in time with the saved weights
  (``core.lab`` score panel; US 10-K filers only).

Learnings are ranked by confidence tier (robust patterns, then accounting
facts, then weaker findings) and by EUR impact within a tier. Accounting facts
are exact; patterns get high, medium or low confidence from their count, a t
statistic clustered by name (orders in one name and month are one case) and
whether both halves of the period agree. With a few dozen trades most patterns
are low confidence, and the report says so instead of calling a winner.
Nothing here changes FastValue's scoring (decision_log.md D-23).
"""

from __future__ import annotations

import html
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import config.settings as settings
from core.ledger import EPS, build_ledger
from data import instruments as ins
from data import trade_republic as tr

logger = logging.getLogger(__name__)

Progress = Callable[[int, int, str], None]
_PRIMARY = settings.PORTFOLIO_BENCHMARKS[0]
BENCHMARK = {"isin": _PRIMARY["isin"], "ticker": _PRIMARY["ticker"],
             "label": f"{_PRIMARY['label']} ({_PRIMARY['fund']}, in EUR)"}  # the primary; kept for callers
RESULT_SCHEMA = 2  # 2: invested-part headline, three indices, a cash section (D-36); 1 or missing: before
HORIZONS = (30, 90, 180)
MAIN_HORIZON = 90
NEAR_HIGH, NEAR_LOW = 0.8, 0.2
CLUSTER_DAYS, CLUSTER_NAMES = 3, 3
MIN_CASES = 10
STOPS = (0.15, 0.25, 0.35)
TAKE_PROFITS = (0.5, 1.0)
MIN_HOLD_DAYS = 90
# Shown in place of the stored bucket keys (BUY / HOLD / SELL stay the keys of the stored result).
_ENTRY_BUCKET_WORDS = {"BUY": "At or above the entry level", "HOLD": "Between the levels", "SELL": "At or below the exit level"}
MIN_ANNUAL_DAYS = 365  # a holding's money-weighted return is annualised only when its money was in for a year
COMMODITY_WORDS = ("physical", "gold", "silver", "palladium", "platinum", "copper", "commodit", "oil ", "etc")
# Ranking: robust patterns first, then accounting facts, then weaker findings; by EUR impact within each tier.
CONFIDENCE_TIER = {"high": 0, "exact": 1, "medium": 2, "low": 3}
REPORT_NAME = "track_record.html"


# ================================================================== helpers


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.strftime("%Y-%m-%d")
    return obj


def _xirr(flows: list[tuple[pd.Timestamp, float]], min_days: int = 0) -> float | None:
    """Annual money-weighted return of dated cash flows (negative = paid in). None when it has no root.

    ``min_days``: None when the flows span fewer days. A few weeks' gain scaled up to a year is not a rate anyone
    earned (the reporting standards do not annualise periods under a year).
    """
    flows = [(pd.Timestamp(d), float(a)) for d, a in flows if abs(a) > 1e-9]
    if len(flows) < 2 or not any(a < 0 for _, a in flows) or not any(a > 0 for _, a in flows):
        return None
    t0 = min(d for d, _ in flows)
    if (max(d for d, _ in flows) - t0).days < min_days:
        return None
    years = np.array([(d - t0).days / 365.25 for d, _ in flows])
    amounts = np.array([a for _, a in flows])

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    lo, hi = -0.9999, 1000.0  # short, strong holdings annualise to very large rates
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            break
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def _tstat(values: list[float] | np.ndarray) -> float | None:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(x) < 3:
        return None
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(len(x))) if sd > 0 else None


def _n(count: int, one: str, many: str) -> str:
    """The count with the right noun: "1 case", "2 cases"."""
    return f"{count} {one if count == 1 else many}"


def _confidence(n: int, t: float | None, halves_agree: bool | None) -> tuple[str, str]:
    """Confidence that a pattern is a habit and not chance (the evidence rule: say when the sample is too small)."""
    if n < MIN_CASES:
        return "low", f"only {_n(n, 'case', 'cases')}; too few to call it a habit"
    t_txt = f"t {t:+.1f}" if t is not None else "no spread to test"
    if halves_agree is False:
        return "low", f"{_n(n, 'case', 'cases')}, {t_txt}, but the two halves of the period disagree"
    if t is not None and abs(t) >= 2.0 and halves_agree and n >= 30:
        return "high", f"{_n(n, 'case', 'cases')}, {t_txt}, same sign in both halves"
    if t is not None and abs(t) >= 1.5 and halves_agree:
        return "medium", f"{_n(n, 'case', 'cases')}, {t_txt}, same sign in both halves"
    return "low", f"{_n(n, 'case', 'cases')}, {t_txt}: could be chance"


def _same_sign(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return None
    return (a > 0) == (b > 0) and a != 0 and b != 0


def _group(name: str, asset_class: str) -> str:
    if asset_class == "DERIVATIVE":
        return "Turbos"
    if asset_class == "FUND":
        n = f" {(name or '').lower()} "
        return "Metal and commodity ETCs" if any(w in n for w in COMMODITY_WORDS) else "ETFs"
    if asset_class == "STOCK":
        return "Stocks"
    return asset_class.title() if asset_class else "Other"


# ================================================================== market data


@dataclass
class _Market:
    days: pd.DatetimeIndex
    end: pd.Timestamp
    px: pd.DataFrame  # EUR close per ISIN on calendar days (carried over weekends); NaN when not priced
    full: dict[str, pd.Series]  # EUR close per ISIN on trading days, whole history (for 52-week ranges)
    bench: pd.Series  # benchmark EUR close on calendar days
    bench_label: str
    tickers: dict[str, str]
    warnings: list[str]
    bench_full: pd.Series | None = None  # benchmark EUR close on trading days, whole history (market trend at entry)
    benches: dict[str, pd.Series] = field(default_factory=dict)  # EUR close on calendar days per benchmark id, primary first
    bench_meta: list[dict[str, Any]] = field(default_factory=list)  # id, label, ticker, isin, fund, primary

    def benchmarks(self) -> list[tuple[dict[str, Any], pd.Series]]:
        """Each benchmark with its calendar-day EUR close, primary first. A market built with one benchmark has the primary only."""
        if self.benches:
            return [(meta, self.benches[meta["id"]]) for meta in self.bench_meta if meta["id"] in self.benches]
        first = settings.PORTFOLIO_BENCHMARKS[0]
        return [({"id": first["id"], "label": first["label"], "ticker": first["ticker"], "isin": first["isin"],
                  "fund": first["fund"], "primary": True}, self.bench)]

    def price(self, isin: str, day: pd.Timestamp) -> float | None:
        if isin not in self.px.columns:
            return None
        day = min(pd.Timestamp(day), self.end)
        if day < self.days[0]:
            return None
        v = self.px.at[day, isin]
        return float(v) if pd.notna(v) else None

    def bench_at(self, day: pd.Timestamp) -> float | None:
        day = min(pd.Timestamp(day), self.end)
        if day < self.days[0] or day not in self.bench.index:
            return None
        v = self.bench.at[day]
        return float(v) if pd.notna(v) else None


def _eur_close(ticker: str, df: pd.DataFrame, ccy: str | None) -> pd.Series:
    from data.fx import factor_series, quote_currency
    from data.price_history import as_traded

    close = as_traded(df)["close"].dropna()
    ccy = ccy or quote_currency(ticker)
    return (close * factor_series(ccy, "EUR", close.index).to_numpy()).dropna()


def _market(ledger: dict[str, Any], mapping: dict[str, dict[str, Any]], progress: Progress) -> _Market:
    from data.price_history import fetch_price_history

    start = pd.Timestamp(min(e["date"] for e in ledger["changes"] + ledger["flows"]))
    end = pd.Timestamp(date.today())
    days = pd.date_range(start, end, freq="D")
    usable = {isin: e for isin, e in mapping.items() if isin in ledger["instruments"] and ins.usable(e)}
    metas: list[dict[str, Any]] = []
    for i, b in enumerate(settings.PORTFOLIO_BENCHMARKS):
        entry = mapping.get(b["isin"])
        listed = i == 0 and ins.usable(entry)  # the map may pick the primary's listing; the others keep the settings' listing
        metas.append({**b, "ticker": entry["ticker"] if listed else b["ticker"],
                      "currency": (entry.get("currency") if listed else None) or b.get("currency")})
    tickers = sorted({e["ticker"] for e in usable.values()} | {b["ticker"] for b in metas})
    progress(0, 1, f"Loading prices for {len(tickers)} listings")
    prices = fetch_price_history(tickers, ins.HISTORY_START)
    warnings: list[str] = []
    cols: dict[str, pd.Series] = {}
    full: dict[str, pd.Series] = {}
    open_isins = {p["isin"] for p in ledger.get("positions") or []}
    for isin, e in usable.items():
        df = prices.get(e["ticker"])
        if df is None or df.empty:
            warnings.append(f"No prices for {e['ticker']} ({ledger['instruments'][isin]['name']}): valued at cost")
            continue
        try:
            s = _eur_close(e["ticker"], df, e.get("currency"))
        except Exception as exc:  # noqa: BLE001 - one bad series must not sink the report
            warnings.append(f"{e['ticker']}: {exc}; valued at cost")
            continue
        info = ledger["instruments"][isin]
        held_from, held_to = pd.Timestamp(info["first_date"]), pd.Timestamp(info["last_date"])
        if isin in open_isins:
            held_to = pd.Timestamp.today()  # still held: a split after the last trade changes today's share count
        splits = df["splits"][(df["splits"] > 0) & (df.index >= held_from) & (df.index <= held_to)]
        if len(splits):
            warnings.append(f"{e['ticker']} split while you held it ({', '.join(d.strftime('%Y-%m-%d') for d in splits.index)}); "
                            "the export has no split row, so check the share count")
        full[isin] = s
        cols[isin] = s.reindex(s.index.union(days)).ffill().reindex(days)
    px = pd.DataFrame(cols, index=days)
    benches: dict[str, pd.Series] = {}
    kept: list[dict[str, Any]] = []
    primary_full: pd.Series | None = None
    for i, meta in enumerate(metas):
        df = prices.get(meta["ticker"])
        if df is None or df.empty:
            if i == 0:
                raise LookupError(f"No prices for the benchmark {meta['ticker']}")
            warnings.append(f"No prices for the {meta['label']} ({meta['ticker']}): left out of the comparison.")
            continue
        s = _eur_close(meta["ticker"], df, meta.get("currency"))
        cal = s.reindex(s.index.union(days)).ffill().bfill(limit=10).reindex(days)
        if i == 0:
            primary_full = s
        elif cal.isna().any():
            day = cal.index[cal.isna().to_numpy().argmax()].strftime("%Y-%m-%d")
            warnings.append(f"The {meta['label']} ({meta['ticker']}) has no price on {day}: left out of the comparison.")
            continue
        benches[meta["id"]] = cal
        kept.append({**meta, "primary": i == 0})
    return _Market(days, end, px, full, benches[metas[0]["id"]], BENCHMARK["label"],
                   {i: e["ticker"] for i, e in usable.items()}, warnings, bench_full=primary_full,
                   benches=benches, bench_meta=kept)


# ================================================================== account series


def _by_day(items: list[dict[str, Any]], days: pd.DatetimeIndex, value: str = "amount") -> pd.Series:
    s = pd.Series(0.0, index=days)
    for it in items:
        d = pd.Timestamp(it["date"])
        if d in s.index:
            s.at[d] += float(it[value])
    return s


def _account(ledger: dict[str, Any], m: _Market) -> dict[str, Any]:
    days = m.days
    ch = pd.DataFrame(ledger["changes"])
    ch["date"] = pd.to_datetime(ch["date"])
    shares = ch.pivot_table(index="date", columns="isin", values="delta", aggfunc="sum", fill_value=0.0) \
        .reindex(days, fill_value=0.0).cumsum()
    cost = ch.pivot_table(index="date", columns="isin", values="cost", aggfunc="sum", fill_value=0.0) \
        .reindex(days, fill_value=0.0).cumsum()
    shares = shares.mask(shares.abs() <= EPS, 0.0)
    cost = cost.where(shares > 0, 0.0)
    px = m.px.reindex(columns=shares.columns)
    priced = px.notna()
    values = (shares * px).where(priced, cost)
    cash = pd.Series(ledger["cash_by_date"]).rename(index=pd.Timestamp).reindex(days, fill_value=0.0).cumsum()
    sleeve = values.sum(axis=1)
    account = cash + sleeve

    flows = _by_day([f for f in ledger["flows"] if f["kind"] in ("deposit", "withdrawal", "card", "private")], days)
    moved_in = pd.Series(0.0, index=days)  # shares delivered in, at market on their day (cost basis when unpriced)
    for f in ledger["flows"]:
        if f["kind"] == "securities_out":
            d = pd.Timestamp(f["date"])
            p = m.price(f["isin"], d)
            flows.at[d] += -(f["shares"] * p) if p is not None else f["amount"]
        elif f["kind"] == "securities_in":
            d = pd.Timestamp(f["date"])
            p = m.price(f["isin"], d)
            v = f["shares"] * p if p is not None else f["amount"]
            flows.at[d] += v
            moved_in.at[d] += v

    # Sleeve flows: cash into positions (buys with costs), out of them (sales net of costs), and moves out.
    buys = pd.Series(0.0, index=days)
    sells = pd.Series(0.0, index=days)
    for o in ledger["orders"]:
        d = pd.Timestamp(o["date"])
        net = o["amount"] + o["fee"] + o["tax"]
        if o["side"] == "buy":
            buys.at[d] += -net
        else:
            sells.at[d] += net
    buys += moved_in
    for f in ledger["flows"]:
        if f["kind"] == "securities_out":
            d = pd.Timestamp(f["date"])
            p = m.price(f["isin"], d)
            sells.at[d] += f["shares"] * p if p is not None else -f["amount"]
    divs = _by_day([{"date": i["date"], "amount": i["amount"] + i["tax"]} for i in ledger["income"] if i["kind"] == "dividend"], days)

    def chain(num: pd.Series, den: pd.Series) -> pd.Series:
        r = pd.Series(0.0, index=days)
        ok = den > 1.0
        r[ok] = num[ok] / den[ok] - 1.0
        return r

    r_account = chain(account, account.shift(1, fill_value=0.0) + flows)
    r_sleeve = chain(sleeve + sells + divs, sleeve.shift(1, fill_value=0.0) + buys)
    return {"shares": shares, "cost": cost, "values": values, "cash": cash, "sleeve": sleeve, "account": account,
            "flows": flows, "r_account": r_account, "r_sleeve": r_sleeve, "buys": buys, "sells": sells, "divs": divs}


def _annual(r: pd.Series) -> float | None:
    if len(r) < 2:
        return None
    growth = float((1.0 + r).prod())
    years = (r.index[-1] - r.index[0]).days / 365.25
    return growth ** (1.0 / years) - 1.0 if years > 0.25 and growth > 0 else growth - 1.0


def _all_in(m: _Market, flows: pd.Series, bench: pd.Series | None = None) -> tuple[pd.Series, list[str]]:
    """Value of putting every external flow into the benchmark on its day (withdrawals sell it). ``bench``: another index."""
    units = 0.0
    out = pd.Series(0.0, index=m.days)
    notes: list[str] = []
    bench = (bench if bench is not None else m.bench).to_numpy(float)
    f = flows.to_numpy(float)
    vals = np.zeros(len(f))
    for i in range(len(f)):
        if f[i] and np.isfinite(bench[i]) and bench[i] > 0:
            units += f[i] / bench[i]
            if units < 0:
                if not notes:
                    notes.append("Some withdrawals were larger than the benchmark holding; it was emptied, not shorted.")
                units = 0.0
        vals[i] = units * bench[i] if np.isfinite(bench[i]) else 0.0
    out[:] = vals
    return out, notes


def _index_mirror(flows: pd.Series, bench: pd.Series) -> tuple[pd.Series, bool, list[str]]:
    """Value of the same money moved in and out of the index on the same days (positive flows buy, negative sell).

    Long and Nickels' public market equivalent. Units may go below zero when sales take out more than the index
    position has grown to: the value is then a yardstick, not a holding, but the difference from the actual holdings
    stays exact, and that difference is what the comparison reports. Returns the value, whether the units ever went
    below zero, and notes (flows on a day without an index price are left out and counted).
    """
    units = 0.0
    dipped = False
    skipped = 0
    px = bench.to_numpy(float)
    f = flows.to_numpy(float)
    vals = np.zeros(len(f))
    for i in range(len(f)):
        ok = bool(np.isfinite(px[i]) and px[i] > 0)
        if f[i] and ok:
            units += f[i] / px[i]
            if units < -1e-9:
                dipped = True
        elif f[i]:
            skipped += 1
        vals[i] = units * px[i] if ok else 0.0
    notes = [f"{skipped} flows fell on days without an index price and were left out of the same-buys comparison."] if skipped else []
    return pd.Series(vals, index=flows.index), dipped, notes


def _year_return(series: pd.Series, sel: pd.DatetimeIndex, start: pd.Timestamp) -> float | None:
    """The index's return over the calendar days ``sel`` from ``start`` on (the first trade, in the first year)."""
    if start > sel[-1]:
        return None  # the year closes before the first trade: nothing was invested in it
    prev = series[series.index < max(sel[0], start)]
    end = series.loc[sel[-1]]
    if len(prev):
        base = float(prev.iloc[-1])
    else:
        inside = series.loc[sel].dropna()
        base = float(inside.iloc[0]) if len(inside) else float("nan")
    return float(end / base - 1.0) if np.isfinite(base) and base > 0 and pd.notna(end) else None


def _modified_dietz(v0: float, v1: float, flows: pd.Series) -> float | None:
    if flows.empty:
        return None
    total_days = max((flows.index[-1] - flows.index[0]).days + 1, 1)
    w = np.array([(flows.index[-1] - d).days / total_days for d in flows.index])
    den = v0 + float((flows.to_numpy() * w).sum())
    return (v1 - v0 - float(flows.sum())) / den if den > 1.0 else None


def _performance(ledger: dict[str, Any], m: _Market, a: dict[str, Any]) -> dict[str, Any]:
    acct, flows = a["account"], a["flows"]
    bench_val, bench_notes = _all_in(m, flows)
    cf = [(d, -float(v)) for d, v in flows.items() if v]
    xirr = _xirr(cf + [(m.days[-1], float(acct.iloc[-1]))], MIN_ANNUAL_DAYS)
    bench_xirr = _xirr(cf + [(m.days[-1], float(bench_val.iloc[-1]))], MIN_ANNUAL_DAYS)
    first_trade = pd.Timestamp(ledger["summary"]["first_trade"] or m.days[0])
    r_acc = a["r_account"][a["r_account"].index >= first_trade]
    r_sl = a["r_sleeve"][a["r_sleeve"].index >= first_trade]
    bench_r = m.bench.pct_change().fillna(0.0)
    bench_r = bench_r[bench_r.index >= first_trade]
    net_in = float(flows.sum())
    invested_days = a["account"] > 1.0
    cash_share = (a["cash"] / a["account"]).where(invested_days).dropna()
    nav = float(a["sleeve"].iloc[-1])
    # The invested part's own flows: money into it is positive; sales, moves out at market and net dividends are negative.
    sleeve_flows = a["buys"] - a["sells"] - a["divs"]
    invested_xirr = _xirr([(d, -float(v)) for d, v in sleeve_flows.items() if v] + [(m.days[-1], nav)], MIN_ANNUAL_DAYS)
    end_value = float(acct.iloc[-1])

    # Monthly invested part, for the "skill or luck" test against each index.
    inv_m = (1.0 + r_sl).resample("ME").prod() - 1.0
    benchmarks: list[dict[str, Any]] = []
    bench_values: dict[str, pd.Series] = {}
    mirror_values: dict[str, pd.Series] = {}
    for meta, series in m.benchmarks():
        primary = bool(meta.get("primary"))
        r_b = series.pct_change().fillna(0.0)
        r_b = r_b[r_b.index >= first_trade]
        total, annual = float((1.0 + r_b).prod() - 1.0), _annual(r_b)
        deposits, dep_notes = (bench_val, bench_notes) if primary else _all_in(m, flows, series)
        mirror, dipped, mirror_notes = _index_mirror(sleeve_flows, series)
        mirror_end = float(mirror.iloc[-1])
        positive = mirror_end > 1e-9
        buys_xirr = _xirr([(d, -float(v)) for d, v in sleeve_flows.items() if v] + [(m.days[-1], mirror_end)],
                          MIN_ANNUAL_DAYS) if positive else None
        dep_end = float(deposits.iloc[-1])
        buys_difference = nav - mirror_end
        deposits_difference = end_value - dep_end
        mon_b = (1.0 + r_b).resample("ME").prod() - 1.0
        act = (inv_m - mon_b).dropna()
        half = len(act) // 2
        benchmarks.append({
            "id": meta["id"], "label": meta["label"], "ticker": meta["ticker"], "fund": meta.get("fund"), "primary": primary,
            "total": total, "annual": annual,
            "excess_annual": (_annual(r_sl) - annual) if annual is not None and _annual(r_sl) is not None else None,
            "buys_end_value": mirror_end if positive else None, "buys_xirr": buys_xirr,
            "buys_difference": buys_difference, "buys_dipped": dipped,
            "deposits_end_value": dep_end,
            "deposits_xirr": bench_xirr if primary else _xirr(cf + [(m.days[-1], dep_end)], MIN_ANNUAL_DAYS),
            "deposits_difference": deposits_difference,
            "cash_part": deposits_difference - buys_difference,
            "active": {"monthly_mean": float(act.mean()) if len(act) else None, "t": _tstat(act.to_numpy()),
                       "months": len(act),
                       "first_half": float(act.iloc[:half].mean()) if half else None,
                       "second_half": float(act.iloc[half:].mean()) if half else None},
            "notes": list(dict.fromkeys(dep_notes + mirror_notes)),
        })
        bench_values[meta["id"]] = deposits
        mirror_values[meta["id"]] = mirror

    years = []
    for y in sorted({d.year for d in m.days}):
        sel = m.days[m.days.year == y]
        prev_end = acct[acct.index < sel[0]]
        v0 = float(prev_end.iloc[-1]) if len(prev_end) else 0.0
        v1 = float(acct.loc[sel[-1]])
        fl = flows.loc[sel]
        real = ledger["summary"]["realised_by_year"].get(str(y), {})
        inc = [i for i in ledger["income"] if i["date"][:4] == str(y)]
        shown = invested_days.loc[sel]
        share = (a["cash"].loc[sel] / acct.loc[sel])[shown]
        years.append({
            "year": y, "start_value": v0, "end_value": v1, "net_flows": float(fl.sum()),
            "gain": v1 - v0 - float(fl.sum()),
            "twr": float((1.0 + a["r_account"].loc[sel]).prod() - 1.0),
            "twr_invested": float((1.0 + a["r_sleeve"].loc[sel]).prod() - 1.0),
            "dietz": _modified_dietz(v0, v1, fl),
            "benchmark": _year_return(m.bench, sel, first_trade),
            "benchmarks": {meta["id"]: _year_return(series, sel, first_trade) for meta, series in m.benchmarks()},
            "avg_cash_share": float(share.mean()) if len(share) else None,
            "cash_end": float(a["cash"].loc[sel[-1]]),
            "realised_average": real.get("average", 0.0), "realised_fifo": real.get("fifo", 0.0),
            "dividends_net": sum(i["amount"] + i["tax"] for i in inc if i["kind"] == "dividend"),
            "interest_net": sum(i["amount"] + i["tax"] for i in inc if i["kind"] == "interest"),
            "fees": sum(o["fee"] for o in ledger["orders"] if o["date"][:4] == str(y)),
            "partial": y == m.days[-1].year or (y == m.days[0].year and m.days[0].month > 1),
        })
    # Monthly active returns against the benchmark: the "skill or luck" test of the whole record.
    acc_m = (1.0 + r_acc).resample("ME").prod() - 1.0
    ben_m = (1.0 + bench_r).resample("ME").prod() - 1.0
    active = (acc_m - ben_m).dropna()
    half = len(active) // 2
    return {
        "start": m.days[0], "end": m.days[-1], "first_trade": first_trade,
        "end_value": end_value, "cash": float(a["cash"].iloc[-1]), "invested_value": nav,
        "net_external_flows": net_in, "gain": end_value - net_in,
        "twr_total": float((1.0 + r_acc).prod() - 1.0), "twr_annual": _annual(r_acc),
        "twr_invested_total": float((1.0 + r_sl).prod() - 1.0), "twr_invested_annual": _annual(r_sl),
        "xirr": xirr, "invested_xirr": invested_xirr,
        "invested_put_in": float(a["buys"].sum()), "invested_taken_out": float(a["sells"].sum()),
        "invested_dividends": float(a["divs"].sum()),
        "benchmark_total": float((1.0 + bench_r).prod() - 1.0), "benchmark_annual": _annual(bench_r),
        "benchmark_end_value": float(bench_val.iloc[-1]), "benchmark_xirr": bench_xirr,
        "benchmark_gain": float(bench_val.iloc[-1]) - net_in, "benchmark_notes": bench_notes,
        "benchmarks": benchmarks,
        "avg_cash_share": float(cash_share.mean()) if len(cash_share) else None,
        "active_monthly_mean": float(active.mean()) if len(active) else None,
        "active_t": _tstat(active.to_numpy()), "active_months": len(active),
        "active_first_half": float(active.iloc[:half].mean()) if half else None,
        "active_second_half": float(active.iloc[half:].mean()) if half else None,
        "years": years,
        "_bench_value": bench_val, "_bench_values": bench_values, "_mirror_values": mirror_values,
    }


# ================================================================== instruments, round trips, orders


def _exit_value(m: _Market, isin: str, shares: float, day: pd.Timestamp, fallback: float) -> tuple[float, bool]:
    p = m.price(isin, day)
    return (shares * p, True) if p is not None else (fallback, False)


def _instruments(ledger: dict[str, Any], m: _Market, a: dict[str, Any], method: str) -> list[dict[str, Any]]:
    flows_by: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for o in ledger["orders"]:
        g = agg[o["isin"]]
        net = o["amount"] + o["fee"] + o["tax"]
        g["bought" if o["side"] == "buy" else "sold"] += abs(o["amount"])
        g["fees"] += o["fee"]
        g["taxes"] += o["tax"]
        g["orders"] += 1
        flows_by[o["isin"]].append((pd.Timestamp(o["date"]), net))
    for i in ledger["income"]:
        if i["kind"] == "dividend" and i["isin"]:
            agg[i["isin"]]["dividends"] += i["amount"] + i["tax"]
            flows_by[i["isin"]].append((pd.Timestamp(i["date"]), i["amount"] + i["tax"]))
    for f in ledger["flows"]:
        if f["kind"] == "securities_out":
            val, _ = _exit_value(m, f["isin"], f["shares"], pd.Timestamp(f["date"]), -f["amount"])
            agg[f["isin"]]["moved_out"] += val
            flows_by[f["isin"]].append((pd.Timestamp(f["date"]), val))
    realised = defaultdict(float)
    for r in ledger["realisations"]:
        realised[r["isin"]] += r["pnl_avg"] if method == "average" else r["pnl_fifo"]
    open_pos = {p["isin"]: p for p in ledger["positions"]}
    rows = []
    for isin, info in ledger["instruments"].items():
        if info["asset_class"] in ("PRIVATE_FUND", "SYNTHETIC") or isin not in agg:
            continue
        g = agg[isin]
        pos = open_pos.get(isin)
        open_value, priced = (0.0, True)
        remaining_cost = 0.0
        if pos:
            remaining_cost = pos["cost_avg"] if method == "average" else pos["cost_fifo"]
            open_value, priced = _exit_value(m, isin, pos["shares"], m.end, pos["cost_avg"])
            flows_by[isin].append((m.end, open_value))
        total = g["sold"] + g["moved_out"] + open_value + g["dividends"] + g["fees"] + g["taxes"] - g["bought"]
        rows.append({
            "isin": isin, "name": info["name"], "ticker": m.tickers.get(isin), "asset_class": info["asset_class"],
            "group": _group(info["name"], info["asset_class"]), "status": "open" if pos else "closed",
            "bought": g["bought"], "sold": g["sold"], "moved_out": g["moved_out"], "open_value": open_value,
            "dividends": g["dividends"], "fees": g["fees"], "taxes": g["taxes"], "realised": realised[isin],
            "unrealised": open_value - remaining_cost if pos else 0.0, "total_pnl": total,
            "return_pct": total / g["bought"] if g["bought"] > 0 else None,
            "xirr": _xirr(flows_by[isin], MIN_ANNUAL_DAYS), "orders": int(g["orders"]), "priced": priced and isin in m.px.columns,
            "first_date": info["first_date"], "last_date": info["last_date"],
            "span_days": (max(d for d, _ in flows_by[isin]) - min(d for d, _ in flows_by[isin])).days if flows_by[isin] else None,
        })
    rows.sort(key=lambda r: -r["total_pnl"])
    return rows


def _round_trips(ledger: dict[str, Any], m: _Market, a: dict[str, Any], method: str) -> list[dict[str, Any]]:
    open_pos = {p["isin"]: p for p in ledger["positions"]}
    out = []
    for ep in ledger["episodes"]:
        isin = ep["isin"]
        info = ledger["instruments"][isin]
        if info["asset_class"] in ("PRIVATE_FUND", "SYNTHETIC"):
            continue
        start = pd.Timestamp(ep["start"])
        end = pd.Timestamp(ep["end"]) if ep["end"] else m.end
        moved, open_value = 0.0, 0.0
        if ep["end_reason"] == "transferred":
            moved, _ = _exit_value(m, isin, ep["transferred_shares"], end, ep["transferred_cost"])
        if ep["end_reason"] == "open" and isin in open_pos:
            open_value, _ = _exit_value(m, isin, open_pos[isin]["shares"], m.end, open_pos[isin]["cost_avg"])
        net_div = ep["dividends"] + ep["dividend_tax"]
        total = ep["sold"] + moved + open_value + net_div + ep["fees"] + ep["trade_taxes"] - ep["bought"]
        mae = mfe = None
        if isin in m.px.columns and ep["avg_buy"]:
            path = m.px[isin].loc[start:end].dropna()
            if len(path):
                mae = float(path.min() / ep["avg_buy"] - 1.0)
                mfe = float(path.max() / ep["avg_buy"] - 1.0)
        out.append({
            "isin": isin, "name": info["name"], "ticker": m.tickers.get(isin), "group": _group(info["name"], info["asset_class"]),
            "start": ep["start"], "end": ep["end"], "outcome": {"sold": "sold", "transferred": "moved out", "open": "open"}[ep["end_reason"]],
            "days": (end - start).days, "buy_orders": ep["buy_orders"], "sell_orders": ep["sell_orders"],
            "bought": ep["bought"], "sold": ep["sold"], "moved_out": moved, "open_value": open_value,
            "avg_buy": ep["avg_buy"], "avg_sell": ep["avg_sell"], "dividends": net_div, "fees": ep["fees"],
            "taxes": ep["trade_taxes"], "realised": ep["realised_avg"] if method == "average" else ep["realised_fifo"],
            "total_pnl": total, "return_pct": total / ep["bought"] if ep["bought"] > 0 else None,
            "peak_cost": ep["peak_cost"], "savings_plan_share": ep["savings_plan_share"], "mae": mae, "mfe": mfe,
        })
    return out


def _orders(ledger: dict[str, Any], m: _Market) -> list[dict[str, Any]]:
    """Every buy and sell order with its forward path and, for buys, the 52-week range and the kind of add."""
    shares: dict[str, float] = defaultdict(float)
    cost: dict[str, float] = defaultdict(float)
    state_before: dict[str, tuple[float, float]] = {}
    for e in sorted(ledger_events_for_state(ledger), key=lambda e: e["ts"]):
        if e["kind"] == "buy" and e["order"] not in state_before:
            state_before[e["order"]] = (shares[e["isin"]], cost[e["isin"]])
        if e["kind"] == "sell" and e["order"] not in state_before:
            state_before[e["order"]] = (shares[e["isin"]], cost[e["isin"]])
        q = e["shares"] or 0.0
        if e["kind"] == "buy":
            shares[e["isin"]] += q
            cost[e["isin"]] += -e["amount"]
        elif e["kind"] in ("sell", "transfer_out"):
            avg = cost[e["isin"]] / shares[e["isin"]] if shares[e["isin"]] > EPS else 0.0
            shares[e["isin"]] += q
            cost[e["isin"]] += avg * q
            if shares[e["isin"]] <= EPS:
                shares[e["isin"]], cost[e["isin"]] = 0.0, 0.0
    discretionary = [o for o in ledger["orders"] if o["side"] == "buy" and not o["savings_plan"]]
    out = []
    for o in ledger["orders"]:
        isin = o["isin"]
        info = ledger["instruments"][isin]
        if info["asset_class"] in ("PRIVATE_FUND", "SYNTHETIC"):
            continue
        d = pd.Timestamp(o["date"])
        p = o["price"]
        held, basis = state_before.get(o["order"], (0.0, 0.0))
        avg_before = basis / held if held > EPS else None
        row = {"order": o["order"], "date": o["date"], "isin": isin, "name": info["name"], "ticker": m.tickers.get(isin),
               "group": _group(info["name"], info["asset_class"]), "side": o["side"], "shares": o["shares"],
               "price": p, "amount": abs(o["amount"]), "fee": o["fee"], "savings_plan": o["savings_plan"],
               "priced": isin in m.px.columns}
        b0 = m.bench_at(d)
        for h in HORIZONS:
            t = d + pd.Timedelta(days=h)
            ret = bench = None
            if t <= m.end and p:
                pt = m.price(isin, t)
                bt = m.bench_at(t)
                ret = pt / p - 1.0 if pt is not None else None
                bench = bt / b0 - 1.0 if bt is not None and b0 else None
            row[f"ret_{h}"] = ret
            row[f"bench_{h}"] = bench
            row[f"excess_{h}"] = ret - bench if ret is not None and bench is not None else None
            row[f"eur_{h}"] = o["shares"] * p * ret if ret is not None else None
            # Relative to the index: what the order gained or lost against MSCI World over the window.
            row[f"eur_excess_{h}"] = abs(o["amount"]) * row[f"excess_{h}"] if row[f"excess_{h}"] is not None else None
        if o["side"] == "buy":
            row["kind"] = "first buy" if held <= EPS else ("add to a loser" if avg_before and p < avg_before else "add to a winner")
            row["avg_cost_before"] = avg_before
            hist = m.full.get(isin)
            if hist is not None:
                window = hist[(hist.index < d) & (hist.index >= d - pd.Timedelta(days=365))]
                if len(window) >= 120:
                    hi, lo = float(window.max()), float(window.min())
                    row["range_52w"] = (p - lo) / (hi - lo) if hi > lo else None
                    row["from_high_52w"] = p / hi - 1.0
            if not o["savings_plan"]:
                near = {x["isin"] for x in discretionary if abs((pd.Timestamp(x["date"]) - d).days) <= CLUSTER_DAYS}
                row["clustered"] = len(near) >= CLUSTER_NAMES
        else:
            row["kind"] = "sale"
            row["avg_cost_before"] = avg_before
            row["gain_at_sale"] = p / avg_before - 1.0 if avg_before else None
        out.append(row)
    return out


def ledger_events_for_state(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Share-moving rows rebuilt from the ledger's orders and transfers (one row per order, in time order)."""
    rows = []
    for o in ledger["orders"]:
        rows.append({"ts": o["ts"], "kind": o["side"], "isin": o["isin"], "order": o["order"],
                     "shares": o["shares"] if o["side"] == "buy" else -o["shares"], "amount": o["amount"]})
    for f in ledger["flows"]:
        if f["kind"] == "securities_out":
            rows.append({"ts": f["date"] + "T23:59:59Z", "kind": "transfer_out", "isin": f["isin"], "order": None,
                         "shares": -f["shares"], "amount": 0.0})
    return rows


# ================================================================== behaviour


def _split_date(m: _Market, first_trade: pd.Timestamp) -> pd.Timestamp:
    """The middle of the traded period, as a calendar day (the second half starts on it).

    Halving an odd number of days lands at noon, which no daily calendar contains; a date keeps the rules lab's
    half-period replays and every ``date < split`` comparison on the same day boundary.
    """
    return (first_trade + (m.end - first_trade) / 2).normalize()


def _clusters(rows: list[dict[str, Any]], value: str) -> list[tuple[float, str]]:
    """Mean of ``value`` per instrument and month, with the cluster's first date.

    Orders in the same name and month share most of their forward window (weekly savings plans above all), so
    they are one observation for the t statistic, not four.
    """
    groups: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for r in rows:
        v = r.get(value)
        if v is not None and np.isfinite(v):
            groups[(r["isin"], r["date"][:7])].append((float(v), r["date"]))
    return [(float(np.mean([v for v, _ in g])), min(d for _, d in g)) for g in groups.values()]


def _pattern(rows: list[dict[str, Any]], value: str, split: pd.Timestamp) -> dict[str, Any]:
    vals = [r[value] for r in rows if r.get(value) is not None]
    cl = _clusters(rows, value)
    first = [v for v, d in cl if pd.Timestamp(d) < split]
    second = [v for v, d in cl if pd.Timestamp(d) >= split]
    return {"n": len(vals), "n_eff": len(cl), "mean": float(np.mean(vals)) if vals else None,
            "t": _tstat([v for v, _ in cl]),
            "first_half": float(np.mean(first)) if first else None, "second_half": float(np.mean(second)) if second else None,
            "eur": float(np.nansum([r.get(f"eur_excess_{MAIN_HORIZON}") or 0.0 for r in rows])),
            "_clusters": cl}


def _welch(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(b) < 3:
        return None
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return float((np.mean(a) - np.mean(b)) / se) if se > 0 else None


def _compare(a: dict[str, Any], b: dict[str, Any], split: pd.Timestamp) -> dict[str, Any]:
    """Group a against group b on clustered values: difference of means, Welch t, and the difference in each half."""
    ca, cb = a.get("_clusters", []), b.get("_clusters", [])
    va, vb = [v for v, _ in ca], [v for v, _ in cb]

    def half(first: bool) -> float | None:
        xa = [v for v, d in ca if (pd.Timestamp(d) < split) == first]
        xb = [v for v, d in cb if (pd.Timestamp(d) < split) == first]
        return float(np.mean(xa) - np.mean(xb)) if xa and xb else None

    first, second = half(True), half(False)
    n = 2 * min(len(va), len(vb))
    return {"difference": float(np.mean(va) - np.mean(vb)) if va and vb else None, "t": _welch(va, vb),
            "first_half": first, "second_half": second, "n": n,
            "confidence": _confidence(n, _welch(va, vb), _same_sign(first, second))}


def _merge(*patterns: dict[str, Any]) -> dict[str, Any]:
    cl = [c for p in patterns for c in p.get("_clusters", [])]
    return {"_clusters": cl, "n": sum(p.get("n", 0) for p in patterns),
            "mean": (sum((p.get("mean") or 0.0) * p.get("n", 0) for p in patterns) / max(sum(p.get("n", 0) for p in patterns), 1))}


def _disposition(m: _Market, a: dict[str, Any], orders: list[dict[str, Any]], split: pd.Timestamp | None = None,
                 half: bool | None = None) -> dict[str, Any]:
    """Odean's PGR and PLR on each day with a sale, with the holdings priced that day.

    ``half``: True for sale days before ``split`` only, False for after, None for all.
    """
    rg = rl = pg = pl = 0
    shares, cost = a["shares"], a["cost"]
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in orders:
        if o["side"] == "sell" and o["avg_cost_before"] and o["group"] != "Turbos":
            by_day[o["date"]].append(o)
    for day, sold in sorted(by_day.items()):
        d = pd.Timestamp(day)
        if half is not None and split is not None and (d < split) != half:
            continue
        sold_isins = {o["isin"] for o in sold}
        for o in sold:
            if o["price"] > o["avg_cost_before"]:
                rg += 1
            else:
                rl += 1
        prev = d - pd.Timedelta(days=1)
        if prev not in shares.index:
            continue
        for isin in shares.columns:
            if isin in sold_isins or shares.at[prev, isin] <= EPS or isin not in m.px.columns:
                continue
            p = m.price(isin, d)
            avg = cost.at[prev, isin] / shares.at[prev, isin]
            if p is None or avg <= 0:
                continue
            if p > avg:
                pg += 1
            else:
                pl += 1
    pgr = rg / (rg + pg) if rg + pg else None
    plr = rl / (rl + pl) if rl + pl else None
    t = None
    if pgr is not None and plr is not None and 0 < pgr < 1 and 0 < plr < 1:
        se = math.sqrt(pgr * (1 - pgr) / (rg + pg) + plr * (1 - plr) / (rl + pl))
        t = (pgr - plr) / se if se > 0 else None
    return {"realised_gains": rg, "realised_losses": rl, "paper_gains": pg, "paper_losses": pl,
            "pgr": pgr, "plr": plr, "difference": (pgr - plr) if pgr is not None and plr is not None else None, "t": t}


def _lots(ledger: dict[str, Any], m: _Market) -> list[dict[str, Any]]:
    """FIFO lots with their outcome in EUR (sold at the price received; moved out or open at market)."""
    out = []
    for lot in ledger["lots"]:
        isin = lot["isin"]
        info = ledger["instruments"][isin]
        if info["asset_class"] in ("PRIVATE_FUND", "SYNTHETIC") or lot["shares"] <= EPS:
            continue
        exit_day = pd.Timestamp(lot["exit_date"]) if lot["exit_date"] else m.end
        if lot["exit"] == "sold":
            value = lot["proceeds"]
        else:
            value, _ = _exit_value(m, isin, lot["shares"], exit_day, lot["cost"])
        paid = lot["cost"] + lot.get("buy_costs", 0.0)
        out.append({**lot, "name": info["name"], "group": _group(info["name"], info["asset_class"]),
                    "date": lot["buy_date"], "exit_day": exit_day, "value": value, "paid": paid,
                    "pnl": value - paid, "return_pct": value / paid - 1.0 if paid > 0 else None})
    return out


def _savings_vs_oneoff(lots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, flag in (("Savings plans", True), ("One-off buys", False)):
        sel = [x for x in lots if x["savings_plan"] == flag]
        if not sel:
            continue
        flows = [(pd.Timestamp(x["buy_date"]), -x["paid"]) for x in sel] + [(x["exit_day"], x["value"]) for x in sel]
        paid = sum(x["paid"] for x in sel)
        rows.append({"group": label, "lots": len(sel), "paid": paid, "pnl": sum(x["pnl"] for x in sel),
                     "return_pct": sum(x["pnl"] for x in sel) / paid if paid else None, "xirr": _xirr(flows, MIN_ANNUAL_DAYS)})
    return rows


def _by_group(instruments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for r in instruments:
        g = groups.setdefault(r["group"], {"group": r["group"], "instruments": 0, "bought": 0.0, "pnl": 0.0,
                                           "fees": 0.0, "winners": 0})
        g["instruments"] += 1
        g["bought"] += r["bought"]
        g["pnl"] += r["total_pnl"]
        g["fees"] += r["fees"]
        g["winners"] += 1 if r["total_pnl"] > 0 else 0
    for g in groups.values():
        g["return_pct"] = g["pnl"] / g["bought"] if g["bought"] else None
        g["hit_rate"] = g["winners"] / g["instruments"] if g["instruments"] else None
    return sorted(groups.values(), key=lambda g: g["pnl"])


def _behaviour(ledger: dict[str, Any], m: _Market, a: dict[str, Any], perf: dict[str, Any],
               orders: list[dict[str, Any]], trips: list[dict[str, Any]], instruments: list[dict[str, Any]],
               lots: list[dict[str, Any]]) -> dict[str, Any]:
    split = _split_date(m, perf["first_trade"])
    excess = f"excess_{MAIN_HORIZON}"
    buys = [o for o in orders if o["side"] == "buy" and o["priced"] and o["group"] != "Turbos"]
    sells = [o for o in orders if o["side"] == "sell" and o["priced"] and o["group"] != "Turbos"]
    closed = [t for t in trips if t["outcome"] == "sold" and t["return_pct"] is not None]
    winners = [t["days"] for t in closed if t["total_pnl"] > 0]
    losers = [t["days"] for t in closed if t["total_pnl"] <= 0]

    sizing = None
    sized = [t for t in trips if t["bought"] > 0 and t["return_pct"] is not None and t["group"] != "Turbos"]

    def spearman(rows: list[dict[str, Any]]) -> float | None:
        if len(rows) < 5:
            return None
        return float(pd.Series([t["peak_cost"] for t in rows]).rank().corr(pd.Series([t["return_pct"] for t in rows]).rank()))

    if len(sized) >= 5:
        rho = spearman(sized)
        top = sorted(sized, key=lambda t: -t["peak_cost"])[:5]
        t_rho = rho * math.sqrt((len(sized) - 2) / (1 - rho * rho)) if rho is not None and abs(rho) < 1 else None
        halves = (spearman([t for t in sized if pd.Timestamp(t["start"]) < split]),
                  spearman([t for t in sized if pd.Timestamp(t["start"]) >= split]))
        sizing = {"n": len(sized), "spearman": rho, "t": t_rho, "first_half": halves[0], "second_half": halves[1],
                  "largest5_pnl": sum(t["total_pnl"] for t in top), "all_pnl": sum(t["total_pnl"] for t in sized),
                  "largest5": [{"name": t["name"], "peak_cost": t["peak_cost"], "pnl": t["total_pnl"]} for t in top]}

    vals = a["values"]
    invested = vals.sum(axis=1)
    weights = vals.div(invested.where(invested > 1.0), axis=0)
    hhi = (weights ** 2).sum(axis=1).where(invested > 1.0).dropna()
    hhi = hhi[hhi.index >= perf["first_trade"]]

    gross_gains = sum(r["pnl_avg"] for r in ledger["realisations"] if r["pnl_avg"] > 0)
    total_fees = sum(o["fee"] for o in ledger["orders"])
    sold_value = sum(abs(o["amount"]) for o in ledger["orders"] if o["side"] == "sell")
    avg_invested = float(invested[invested.index >= perf["first_trade"]].mean()) if len(invested) else 0.0
    years = max((m.end - perf["first_trade"]).days / 365.25, 0.1)

    interest_net = sum(i["amount"] + i["tax"] for i in ledger["income"] if i["kind"] == "interest")
    cash_prev = a["cash"].shift(1, fill_value=0.0).clip(lower=0.0)
    opportunity = float((cash_prev * a["r_sleeve"]).sum())

    kinds = {k: _pattern([o for o in buys if o["kind"] == k], excess, split)
             for k in ("first buy", "add to a winner", "add to a loser")}
    ranges = {k: _pattern(v, excess, split) for k, v in {
        "near the 52-week high": [o for o in buys if (o.get("range_52w") is not None and o["range_52w"] >= NEAR_HIGH)],
        "middle of the range": [o for o in buys if (o.get("range_52w") is not None and NEAR_LOW < o["range_52w"] < NEAR_HIGH)],
        "near the 52-week low": [o for o in buys if (o.get("range_52w") is not None and o["range_52w"] <= NEAR_LOW)],
    }.items()}
    clusters = {k: _pattern(v, excess, split) for k, v in {
        "clustered impulse buys": [o for o in buys if o.get("clustered") is True],
        "isolated one-off buys": [o for o in buys if o.get("clustered") is False],
    }.items()}
    plans = {k: _pattern(v, excess, split) for k, v in {
        "savings-plan buys": [o for o in buys if o["savings_plan"]],
        "one-off buys": [o for o in buys if not o["savings_plan"]],
    }.items()}
    exits = _pattern(sells, excess, split)
    exit_clusters = exits["_clusters"]
    return {
        "split_date": split,
        "exits": {**exits, "rows": sells, "n_eff": len(exit_clusters)},
        "kinds": kinds,
        "kinds_compare": _compare(kinds["add to a loser"], _merge(kinds["first buy"], kinds["add to a winner"]), split),
        "range": ranges,
        "range_compare": _compare(ranges["near the 52-week high"], ranges["near the 52-week low"], split),
        "clusters": clusters,
        "clusters_compare": _compare(clusters["clustered impulse buys"], clusters["isolated one-off buys"], split),
        "plans": plans,
        "plans_compare": _compare(plans["savings-plan buys"], plans["one-off buys"], split),
        "entries_all": _pattern(buys, excess, split),
        "disposition": {**_disposition(m, a, orders),
                        "first_half": _disposition(m, a, orders, split, True)["difference"],
                        "second_half": _disposition(m, a, orders, split, False)["difference"],
                        "median_days_winners": float(np.median(winners)) if winners else None,
                        "median_days_losers": float(np.median(losers)) if losers else None,
                        "winners": len(winners), "losers": len(losers)},
        "sizing": sizing,
        "concentration": {"avg_effective_positions": float((1.0 / hhi).mean()) if len(hhi) else None,
                          "min_effective_positions": float((1.0 / hhi).min()) if len(hhi) else None,
                          "max_weight": float(weights.max(axis=1).max()) if len(hhi) else None,
                          "current_effective_positions": float(1.0 / hhi.iloc[-1]) if len(hhi) else None},
        "costs": {"fees": total_fees, "gross_realised_gains": gross_gains,
                  "fees_share_of_gains": -total_fees / gross_gains if gross_gains > 0 else None,
                  "turnover_per_year": sold_value / avg_invested / years if avg_invested > 0 else None,
                  "taxes": sum(ledger["summary"]["taxes_by_type"].values())},
        "cash": {"avg_cash_share": perf["avg_cash_share"], "interest_net": interest_net,
                 "opportunity_eur": opportunity - interest_net,
                 "drag_pp": (perf["twr_invested_annual"] - perf["twr_annual"])
                 if perf["twr_invested_annual"] is not None and perf["twr_annual"] is not None else None},
        "groups": _by_group(instruments),
        "savings_vs_oneoff": _savings_vs_oneoff(lots),
    }


# ================================================================== what-ifs


def _sold_kept(orders: list[dict[str, Any]], m: _Market, split: pd.Timestamp) -> dict[str, Any]:
    rows = []
    for o in orders:
        if o["side"] != "sell" or not o["priced"] or o["group"] == "Turbos":
            continue
        now = m.price(o["isin"], m.end)
        b0, b1 = m.bench_at(pd.Timestamp(o["date"])), m.bench_at(m.end)
        if now is None or not b0 or not b1:
            continue
        index_value = o["amount"] * b1 / b0  # the proceeds put into MSCI World instead
        rows.append({"date": o["date"], "name": o["name"], "shares": o["shares"], "proceeds": o["amount"],
                     "value_today": o["shares"] * now, "difference": o["shares"] * now - o["amount"],
                     "index_value": index_value, "vs_index": o["shares"] * now - index_value})
    by_name: dict[str, float] = defaultdict(float)  # sales of one name share today's price: one observation
    for r in rows:
        by_name[r["name"]] += r["vs_index"]
    first = sum(r["vs_index"] for r in rows if pd.Timestamp(r["date"]) < split)
    second = sum(r["vs_index"] for r in rows if pd.Timestamp(r["date"]) >= split)
    return {"n": len(rows), "names": len(by_name), "eur": float(sum(r["difference"] for r in rows)),
            "eur_vs_index": float(sum(r["vs_index"] for r in rows)),
            "first_half": first, "second_half": second, "t": _tstat(list(by_name.values())),
            "rows": sorted(rows, key=lambda r: -abs(r["vs_index"]))}


def _overlay_lots(lots: list[dict[str, Any]], m: _Market) -> list[dict[str, Any]]:
    out = []
    for lot in lots:
        if lot["group"] == "Turbos" or lot["isin"] not in m.px.columns or lot["shares"] <= EPS:
            continue
        entry_day = pd.Timestamp(lot["buy_date"])
        entry = lot["cost"] / lot["shares"]
        exit_day = lot["exit_day"]
        actual = lot["value"] / lot["shares"]
        path = m.px[lot["isin"]].loc[entry_day + pd.Timedelta(days=1):exit_day].dropna()
        out.append({"isin": lot["isin"], "name": lot["name"], "date": lot["buy_date"], "shares": lot["shares"],
                    "entry": entry, "actual": actual, "exit_day": exit_day, "exit": lot["exit"], "path": path})
    return out


def _apply_rule(lot: dict[str, Any], m: _Market, stop: float | None, take: float | None, min_hold: int | None) -> float:
    """Exit price per share under the rule (actual exit when the rule never fires)."""
    path = lot["path"]
    entry = lot["entry"]
    if min_hold and lot["exit"] == "sold":
        hold_until = pd.Timestamp(lot["date"]) + pd.Timedelta(days=min_hold)
        if lot["exit_day"] < hold_until:
            p = m.price(lot["isin"], min(hold_until, m.end))
            if p is not None:
                return p
    hits = []
    if stop is not None:
        below = path[path <= entry * (1.0 - stop)]
        if len(below):
            hits.append((below.index[0], float(below.iloc[0])))
    if take is not None:
        above = path[path >= entry * (1.0 + take)]
        if len(above):
            hits.append((above.index[0], float(above.iloc[0])))
    if hits:
        return min(hits)[1]
    return lot["actual"]


def _overlays(lots: list[dict[str, Any]], m: _Market, split: pd.Timestamp) -> dict[str, Any]:
    olots = _overlay_lots(lots, m)
    setups: list[tuple[str, float | None, float | None, int | None]] = []
    setups += [(f"Stop-loss at -{s:.0%}", s, None, None) for s in STOPS]
    setups += [(f"Take profit at +{t:.0%}", None, t, None) for t in TAKE_PROFITS]
    setups += [(f"Hold at least {MIN_HOLD_DAYS} days", None, None, MIN_HOLD_DAYS)]
    setups += [(f"Stop -{s:.0%} and take profit +{t:.0%}", s, t, None) for s in STOPS for t in TAKE_PROFITS]
    rows = []
    for label, stop, take, hold in setups:
        diffs, first, second = [], 0.0, 0.0
        by_isin: dict[str, float] = defaultdict(float)  # lots of one name move together: one observation per name
        for lot in olots:
            d = lot["shares"] * (_apply_rule(lot, m, stop, take, hold) - lot["actual"])
            diffs.append(d)
            by_isin[lot["isin"]] += d
            if pd.Timestamp(lot["date"]) < split:
                first += d
            else:
                second += d
        t = _tstat(list(by_isin.values()))
        fired = sum(1 for d in diffs if abs(d) > 1e-9)
        rows.append({"rule": label, "lots": len(olots), "names": len(by_isin), "changed": fired, "eur": float(sum(diffs)),
                     "first_half": first, "second_half": second, "weaker_half": min(first, second), "t": t})
    rows.sort(key=lambda r: -r["weaker_half"])
    best = rows[0] if rows else None
    adopt = bool(best and best["first_half"] > 0 and best["second_half"] > 0 and best["t"] is not None and best["t"] >= 2.0)
    verdict = (f"{best['rule']} would have added {_eur_abs(best['eur'])}, ahead in both halves with t {best['t']:.1f}: worth a closer look."
               if adopt else
               "Keep what you do: no rule improved the result in both halves with |t| of at least 2"
               + (f" (best by its weaker half: {best['rule']}, {_eur(best['eur'])} in total)." if best else "."))
    return {"rows": rows, "best": best, "adopt": adopt, "verdict": verdict,
            "protocol": "Rules fixed before the run, applied to every lot of a priced stock or fund; a rule must beat "
                        "the actual exits in both halves of the period (by entry date) with t of at least 2. "
                        "Exits happen at the day's close; the cash then sits idle; fees are left out."}


# ================================================================== FastValue replay


def _fastvalue(ledger: dict[str, Any], m: _Market, orders: list[dict[str, Any]], lots: list[dict[str, Any]],
               split: pd.Timestamp, progress: Progress) -> dict[str, Any]:
    from config.weights import load_scoring_config
    from core import lab

    stock_isins = [i for i, info in ledger["instruments"].items() if info["asset_class"] == "STOCK" and i in m.tickers]
    tickers = sorted({m.tickers[i] for i in stock_isins})
    by_ticker = {m.tickers[i]: i for i in stock_isins}
    first = pd.Timestamp(ledger["summary"]["first_trade"]) - pd.offsets.MonthEnd(2)
    dates = [d for d in pd.date_range(first, m.end, freq="ME") if d < m.end]
    panel, status = lab.build_panel(tickers, dates, progress)
    decided = lab.evaluate_setup(panel, load_scoring_config())
    decided = decided.assign(isin=decided["ticker"].map(by_ticker))
    by_isin: dict[str, pd.DataFrame] = {i: g.sort_values("date") for i, g in decided.groupby("isin")}

    def at(isin: str, day: pd.Timestamp) -> dict[str, Any] | None:
        g = by_isin.get(isin)
        if g is None:
            return None
        before = g[g["date"] < day]
        if before.empty:
            return None
        r = before.iloc[-1]
        return {"decision": r["decision"], "owner_score": r["owner_score"], "date": r["date"]}

    def bucket(dec: str | None) -> str:
        return {"STRONG_BUY": "BUY", "BUY": "BUY", "HOLD": "HOLD", "SELL": "SELL", "STRONG_SELL": "SELL"}.get(dec or "", "not rated")

    rated = []
    for o in orders:
        if o["side"] == "buy" and o["isin"] in by_ticker.values() and o["priced"]:
            r = at(o["isin"], pd.Timestamp(o["date"]))
            rated.append({**o, "owner_decision": bucket(r["decision"] if r else None),
                          "owner_score": r["owner_score"] if r else None})
    groups = {k: _pattern([o for o in rated if o["owner_decision"] == k], f"excess_{MAIN_HORIZON}", split)
              for k in ("BUY", "HOLD", "SELL", "not rated")}
    compare = _compare(groups["BUY"], _merge(groups["HOLD"], groups["SELL"]), split)

    # FastValue's sell rule on your entries: leave at the first month-end rated SELL after the buy.
    diffs, first_h, second_h, changed = [], 0.0, 0.0, 0
    per_name: dict[str, float] = defaultdict(float)
    for lot in _overlay_lots(lots, m):
        g = by_isin.get(lot["isin"])
        if g is None:
            continue
        window = g[(g["date"] > pd.Timestamp(lot["date"])) & (g["date"] < lot["exit_day"])]
        sell = window[window["decision"].isin(["SELL", "STRONG_SELL"])]
        d = 0.0
        if len(sell):
            p = m.price(lot["isin"], sell["date"].iloc[0])
            if p is not None:
                d = lot["shares"] * (p - lot["actual"])
                changed += 1
        diffs.append(d)
        per_name[lot["isin"]] += d
        if pd.Timestamp(lot["date"]) < split:
            first_h += d
        else:
            second_h += d
    backfilled = _backfill_entry_ratings(at, m)
    return {"tickers": len(tickers), "status": status, "entries": groups, "compare": compare, "rated_buys": len(rated),
            "rated_rows": [{"order": r["order"], "owner_decision": r["owner_decision"], "owner_score": r["owner_score"]} for r in rated],
            "sell_rule": {"lots": len(diffs), "names": len(per_name), "changed": changed, "eur": float(sum(diffs)),
                          "first_half": first_h, "second_half": second_h, "t": _tstat(list(per_name.values()))},
            "backfilled": backfilled}


def _backfill_entry_ratings(at: Callable[[str, pd.Timestamp], dict[str, Any] | None], m: _Market) -> int:
    """Store the replayed owner rating at entry on imported positions that have none yet."""
    from data.brokers import SOURCES as BROKER_SOURCES
    from data.portfolio import load_portfolio, save_portfolio

    portfolio = load_portfolio()
    n = 0
    for p in portfolio["positions"]:
        if p.get("source") not in BROKER_SOURCES or p.get("owner_score_at_entry") is not None or not p.get("isin") or not p.get("opened"):
            continue
        r = at(p["isin"], pd.Timestamp(p["opened"]))
        if r is None or r["decision"] == "NOT_RATED" or r["owner_score"] is None or not np.isfinite(r["owner_score"]):
            continue
        p["owner_score_at_entry"] = round(float(r["owner_score"]), 1)
        p["owner_decision_at_entry"] = r["decision"]
        p["rated_at"] = f"replay {pd.Timestamp(r['date']).strftime('%Y-%m-%d')}"
        n += 1
    if n:
        save_portfolio(portfolio)
    return n


# ================================================================== learnings


def _learning(key: str, title: str, finding: str, eur: float | None, confidence: str, why: str, kind: str,
              lesson: str, columns: list[str] | None = None, rows: list[list[Any]] | None = None,
              n: int | None = None, section: str = "overview") -> dict[str, Any]:
    return {"key": key, "title": title, "finding": finding, "eur": eur, "confidence": confidence, "why": why,
            "kind": kind, "lesson": lesson, "n": n, "section": section,
            "evidence": {"columns": columns or [], "rows": rows or []},
            "tier": CONFIDENCE_TIER.get(confidence, 3)}


def _eur(v: float | None) -> str:
    """A signed difference: +€2,873, -€33."""
    return "n/a" if v is None else f"{'-' if round(v) < 0 else '+'}€{abs(v):,.0f}"


def _eur_abs(v: float | None) -> str:
    """A total or an amount without a sign of its own: €13,367 (-€33 when it is below zero)."""
    return "n/a" if v is None else f"{'-' if round(v) < 0 else ''}€{abs(v):,.0f}"


def _pct(v: float | None, signed: bool = True) -> str:
    return "n/a" if v is None else (f"{v:+.1%}" if signed else f"{v:.1%}")


def _weak(confidence: str) -> str | None:
    """The lesson when the evidence cannot carry one (the owner's rule: say so and keep the current way)."""
    return ("Not enough evidence to call it a habit yet: keep doing what you do on this point and look again when "
            "more trades are in.") if confidence == "low" else None


def _learnings(perf: dict[str, Any], beh: dict[str, Any], instruments: list[dict[str, Any]],
               whatif: dict[str, Any], overlays: dict[str, Any], fv: dict[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = sum(r["total_pnl"] for r in instruments)

    # 1. Against the market: the invested part (cash left out) against the primary index; the others are shown, not tested.
    benches = perf["benchmarks"]
    primary = benches[0]
    act = primary["active"]
    agree = _same_sign(act["first_half"], act["second_half"])
    conf, why = _confidence(act["months"], act["t"], agree)
    others = "".join(f"; against the {b['label']} t {b['active']['t']:+.1f}" for b in benches[1:] if b["active"]["t"] is not None)
    names = [f"{_pct(b['annual'])} for {b['label']}" for b in benches]
    diff = primary["buys_difference"]
    if primary["buys_end_value"] is not None:
        same = (f"The same buys and sales put into {primary['label']} on the same days would be worth "
                f"{_eur_abs(primary['buys_end_value'])} against your {_eur_abs(perf['invested_value'])} ({_eur(diff)}).")
    else:
        same = (f"Against the same buys and sales in {primary['label']}, your invested part is {_eur(diff)} ahead or behind; "
                "your sales took out more than the index position would have held, so only the difference is meaningful.")
    out.append(_learning(
        "market", "Against the market",
        f"Your invested part made {_pct(perf['twr_invested_annual'])} a year time-weighted, against "
        + ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] + f". {same} "
        "Cash is left out here; it has its own section.",
        diff, conf, why + f" (monthly returns of the invested part against {primary['label']}{others}, shown, not tested)", "what-if",
        "With this many months the gap is mostly luck or exposure (sectors, metals, single names), not proven skill. "
        "Judge the next year on the same yardstick before changing anything." if conf == "low" else
        (f"Your picks have beaten {primary['label']} with some consistency; keep the process that produced them." if diff > 0 else
         "A plain index has beaten your picks consistently; consider a larger index core."),
        ["Year", "Invested part (TWR)"] + [b["label"] for b in benches] + [f"Against {primary['label']}"],
        [[y["year"], _pct(y["twr_invested"])] + [_pct((y.get("benchmarks") or {}).get(b["id"])) for b in benches]
         + [_pct(y["twr_invested"] - (y.get("benchmarks") or {})[primary["id"]])
            if (y.get("benchmarks") or {}).get(primary["id"]) is not None else "n/a"]
         for y in perf["years"]], act["months"]))

    # 2. Where the result came from.
    if instruments:
        top = sorted(instruments, key=lambda r: -r["total_pnl"])[:3]
        bottom = sorted(instruments, key=lambda r: r["total_pnl"])[:3]
        top_sum = sum(r["total_pnl"] for r in top)
        out.append(_learning(
            "winners", "Where the gains came from",
            f"Three holdings made {_eur_abs(top_sum)} of the {_eur_abs(total)} total: "
            + ", ".join(f"{r['name']} ({_eur(r['total_pnl'])})" for r in top) + ".",
            top_sum, "exact", "accounting from the export and today's prices", "attribution",
            "A few winners carry the record. Know why each worked (business quality, price paid, luck of timing) "
            "before scaling the approach.",
            ["Holding", "Group", "Total P&L", "Return"],
            [[r["name"], r["group"], _eur(r["total_pnl"]), _pct(r["return_pct"])] for r in top], len(instruments)))
        bottom_sum = sum(r["total_pnl"] for r in bottom)
        if bottom_sum < 0:
            out.append(_learning(
                "losers", "Where the losses came from",
                f"Three holdings lost {_eur_abs(abs(bottom_sum))}: "
                + ", ".join(f"{r['name']} ({_eur(r['total_pnl'])})" for r in bottom) + ".",
                bottom_sum, "exact", "accounting from the export and today's prices", "attribution",
                "Write down the thesis each loser was bought on and what broke; check whether FastValue's owner score "
                "would have flagged it at entry.",
                ["Holding", "Group", "Total P&L", "Return", "Status"],
                [[r["name"], r["group"], _eur(r["total_pnl"]), _pct(r["return_pct"]), r["status"]] for r in bottom],
                len(instruments)))

    # 3. Asset classes.
    groups = beh["groups"]
    if len(groups) > 1:
        worst = groups[0]
        out.append(_learning(
            "groups", "Results by kind of instrument",
            "By group: " + "; ".join(f"{g['group']} {_eur(g['pnl'])} ({_pct(g['return_pct'])} on {_eur_abs(g['bought'])} bought)"
                                     for g in sorted(groups, key=lambda g: -g["pnl"])) + ".",
            None, "exact", "accounting from the export and today's prices", "attribution",
            f"{worst['group']} did worst. If it is outside your value framework, cap it or drop it." if worst["pnl"] < 0
            else "Every group made money.",
            ["Group", "Instruments", "Bought", "P&L", "Return", "Winners"],
            [[g["group"], g["instruments"], f"{g['bought']:,.0f}", _eur(g["pnl"]), _pct(g["return_pct"]),
              f"{g['winners']}/{g['instruments']}"] for g in groups], len(instruments)))

    # 4. Exits.
    ex = beh["exits"]
    if ex["n"]:
        conf, why = _confidence(ex["n_eff"], ex["t"], _same_sign(ex["first_half"], ex["second_half"]))
        mean = ex["mean"] or 0.0
        verdict = ("it kept rising: you sold early" if mean >= 0.01 else "it lagged: good exits" if mean <= -0.01
                   else "about in line with the market")
        out.append(_learning(
            "exits", "What happened after you sold",
            f"{MAIN_HORIZON} days after your {ex['n']} sales the price had moved {_pct(ex['mean'])} against MSCI World on "
            f"average ({verdict}). Against the index, the shares you sold made {_eur(ex['eur'])} in those {MAIN_HORIZON} days.",
            ex["eur"], conf, why + f" ({ex['n_eff']} independent name-months)", "pattern",
            _weak(conf) or ("You tend to sell before the move is over. Before selling, write down why; if the business case "
                            "still holds (owner score above the exit level), the framework says keep it." if mean >= 0.01 else
                            "Your exits have been well timed; keep selling on a broken thesis rather than on price alone."
                            if mean <= -0.01 else "Your exits neither helped nor hurt against the market."),
            ["Date", "Holding", "Sold at", f"After {MAIN_HORIZON} days", "vs MSCI World", "EUR vs MSCI World"],
            [[r["date"], r["name"], f"{r['price']:.2f}", _pct(r[f"ret_{MAIN_HORIZON}"]), _pct(r[f"excess_{MAIN_HORIZON}"]),
              _eur(r[f"eur_excess_{MAIN_HORIZON}"])]
             for r in sorted(ex["rows"], key=lambda r: -abs(r.get(f"eur_excess_{MAIN_HORIZON}") or 0))[:15]],
            ex["n"]))

    sk = whatif["sold_kept"]
    if sk["n"]:
        conf, why = _confidence(sk["names"], sk["t"], _same_sign(sk["first_half"], sk["second_half"]))
        out.append(_learning(
            "never_sell", "If you had kept everything you sold",
            f"The shares from your {sk['n']} sales ({sk['names']} names) would be worth {_eur(sk['eur'])} against what you got "
            f"for them, at today's prices. Most of that is the market rising: against the same proceeds put into MSCI World, "
            f"keeping them would have made {_eur(sk['eur_vs_index'])}.",
            sk["eur_vs_index"], conf, why + " (against MSCI World, one observation per name)", "what-if",
            _weak(conf) or ("The names you sold went on to beat the index: you have been selling good businesses too soon. "
                            "Sell on a broken thesis, not on a gain." if sk["eur_vs_index"] > 0 else
                            "The names you sold went on to lag the index: your exits were worth making."),
            ["Date", "Holding", "Proceeds", "Value today", "Same money in MSCI World", "Kept vs MSCI World"],
            [[r["date"], r["name"], f"{r['proceeds']:,.0f}", f"{r['value_today']:,.0f}", f"{r['index_value']:,.0f}",
              _eur(r["vs_index"])] for r in sk["rows"][:15]], sk["n"]))

    # 5. Entries: each pattern against its comparison group.
    def pattern_rows(groups: dict[str, dict[str, Any]]) -> list[list[Any]]:
        return [[k, v["n"], _pct(v["mean"]), f"{v['t']:.1f}" if v["t"] is not None else "n/a", _eur(v["eur"])]
                for k, v in groups.items() if v["n"]]

    columns = ["Buys", "Count", f"vs MSCI World after {MAIN_HORIZON} days", "t (clustered)", "EUR vs MSCI World"]
    kinds, kc = beh["kinds"], beh["kinds_compare"]
    loser = kinds.get("add to a loser", {})
    if loser.get("n") and kc["difference"] is not None:
        conf, why = kc["confidence"]
        out.append(_learning(
            "averaging_down", "Adding to losing positions",
            f"{loser['n']} buys added to a position below its average cost; {MAIN_HORIZON} days later they were "
            f"{_pct(loser['mean'])} against MSCI World, {_pct(kc['difference'])} against your other buys.",
            loser["eur"], conf, why + " (difference from your other buys)", "pattern",
            _weak(conf) or ("Averaging down has paid, but add only when the owner score is still at or above the entry level: a lower price alone "
                            "is not a thesis." if kc["difference"] > 0 else
                            "Averaging down has done worse than your other buys: add to a loser only when the owner score "
                            "is still at or above the entry level."),
            columns, pattern_rows(kinds), loser["n"]))

    rng, rc = beh["range"], beh["range_compare"]
    high, low = rng.get("near the 52-week high", {}), rng.get("near the 52-week low", {})
    if high.get("n") and low.get("n") and rc["difference"] is not None:
        conf, why = rc["confidence"]
        out.append(_learning(
            "chasing", "Buying near highs or near lows",
            f"Buys near the 52-week high ({high['n']}) did {_pct(high['mean'])} against MSCI World over {MAIN_HORIZON} days, "
            f"buys near the low ({low['n']}) {_pct(low['mean'])}: a gap of {_pct(rc['difference'])}.",
            high["eur"], conf, why + " (highs against lows)", "pattern",
            _weak(conf) or ("Chasing strength has cost you. The S&P 500 study found stocks at or above the entry level bought far below their high "
                            "did best (decision_log.md D-19)." if rc["difference"] < 0 else
                            "Buying strength worked for you; the S&P 500 study still favours pullbacks for stocks at or above the entry level."),
            columns, pattern_rows(rng), high["n"] + low["n"]))

    cl, cc = beh["clusters"], beh["clusters_compare"]
    clustered = cl.get("clustered impulse buys", {})
    if clustered.get("n") and cc["difference"] is not None:
        conf, why = cc["confidence"]
        out.append(_learning(
            "clusters", "Bursts of buying",
            f"{clustered['n']} one-off buys came in bursts ({CLUSTER_NAMES} or more names within {CLUSTER_DAYS} days); against "
            f"MSCI World they did {_pct(clustered['mean'])} over {MAIN_HORIZON} days, {_pct(cc['difference'])} against isolated buys.",
            clustered["eur"], conf, why + " (bursts against isolated buys)", "pattern",
            _weak(conf) or ("Bursts of buying did worse: when you want several names at once, analyze one at a time and "
                            "wait a day." if cc["difference"] < 0 else "Your bursts of buying did not do worse than single buys."),
            columns, pattern_rows(cl), clustered["n"]))

    sv, pc = beh["savings_vs_oneoff"], beh["plans_compare"]
    if len(sv) == 2:
        plan, one = sv[0], sv[1]
        conf, why = pc["confidence"]
        better = (plan["return_pct"] or 0) > (one["return_pct"] or 0)
        out.append(_learning(
            "plans", "Savings plans against one-off buys",
            f"Savings-plan purchases made {_eur(plan['pnl'])} ({_pct(plan['return_pct'])}, {_pct(plan['xirr'])} a year "
            f"money-weighted); one-off buys {_eur(one['pnl'])} ({_pct(one['return_pct'])}, {_pct(one['xirr'])} a year). "
            f"The one-off money would have made {_eur((plan['return_pct'] or 0.0) * one['paid'] - one['pnl'])} more at the "
            "plans' return.",
            one["pnl"] - (plan["return_pct"] or 0.0) * one["paid"], conf,
            why + "; P&L per FIFO lot, sale fees left out", "attribution",
            _weak(conf) or ("The automatic plans did better per euro: let them do more of the work." if better
                            else "Your one-off picks did better per euro than the plans."),
            ["Kind", "Lots", "Paid", "P&L", "Return", "Money-weighted a year"],
            [[r["group"], r["lots"], f"{r['paid']:,.0f}", _eur(r["pnl"]), _pct(r["return_pct"]), _pct(r["xirr"])] for r in sv],
            plan["lots"] + one["lots"]))

    # 6. Disposition effect.
    dp = beh["disposition"]
    if dp["pgr"] is not None and dp["plr"] is not None:
        n = dp["realised_gains"] + dp["realised_losses"]
        conf, why = _confidence(n, dp["t"], _same_sign(dp["first_half"], dp["second_half"]))
        out.append(_learning(
            "disposition", "Selling winners faster than losers",
            f"On days you sold, you realised {dp['pgr']:.0%} of the gains you had and {dp['plr']:.0%} of the losses: "
            f"you sell winners {'more' if dp['pgr'] > dp['plr'] else 'less'} readily than losers. Closed winners were held {dp['median_days_winners'] or 0:.0f} "
            f"days (median), closed losers {dp['median_days_losers'] or 0:.0f}.",
            None, conf, why + " (Odean's test; sale days are not fully independent)", "pattern",
            _weak(conf) or ("Classic disposition effect: you lock in gains and sit on losses. Judge losers by the same "
                            "question as winners: is the owner thesis intact?" if dp["pgr"] > dp["plr"]
                            else "No sign of the disposition effect."),
            ["Measure", "Value"],
            [["Realised gains / losses", f"{dp['realised_gains']} / {dp['realised_losses']}"],
             ["Paper gains / losses on sale days", f"{dp['paper_gains']} / {dp['paper_losses']}"],
             ["Share of gains / losses realised (PGR / PLR)", f"{dp['pgr']:.2f} / {dp['plr']:.2f}"],
             ["t of the difference", f"{dp['t']:.1f}" if dp["t"] is not None else "n/a"]], n))

    # 7. Sizing.
    sz = beh["sizing"]
    if sz:
        conf, why = _confidence(sz["n"], sz["t"], _same_sign(sz["first_half"], sz["second_half"]))
        out.append(_learning(
            "sizing", "Size of positions and their results",
            f"Your five largest positions made {_eur(sz['largest5_pnl'])} of {_eur(sz['all_pnl'])}; rank correlation between "
            f"size and return {sz['spearman']:+.2f} over {sz['n']} round trips.",
            sz["largest5_pnl"], conf, why, "pattern",
            _weak(conf) or ("Your conviction (size) has not matched results: size up only on stocks at or above the entry level with a margin of "
                            "safety." if sz["spearman"] < 0 else
                            "Larger positions did better: your conviction has been informative."),
            ["Holding", "Peak cost", "P&L"],
            [[r["name"], f"{r['peak_cost']:,.0f}", _eur(r["pnl"])] for r in sz["largest5"]], sz["n"]))

    # 8. Costs and cash.
    c = beh["costs"]
    out.append(_learning(
        "costs", "Fees, taxes and turnover",
        f"Fees {_eur_abs(abs(c['fees']))} ({_pct(c['fees_share_of_gains'], False)} of gross realised gains), taxes withheld "
        f"{_eur_abs(abs(c['taxes']))}, turnover {_pct(c['turnover_per_year'], False)} of the invested value a year.",
        c["fees"], "exact", "accounting from the export", "attribution",
        "High turnover: every sale also starts the tax clock and the fee meter. Trade less, mostly on a broken thesis."
        if (c["turnover_per_year"] or 0) > 1.0 else "Costs are small next to the results.",
        ["Item", "EUR"], [["Fees", f"{c['fees']:,.2f}"], ["Taxes withheld", f"{c['taxes']:,.2f}"],
                          ["Gross realised gains (average cost)", f"{c['gross_realised_gains']:,.2f}"]]))
    cash = beh["cash"]
    if cash["avg_cash_share"] is not None and cash["drag_pp"] is not None:
        effect = -cash["drag_pp"]  # negative: cash cost that many points a year
        out.append(_learning(
            "cash", "Uninvested cash",
            f"On average {cash['avg_cash_share']:.0%} of the account sat in cash, earning {_eur_abs(cash['interest_net'])} of "
            f"interest after tax. Cash {'took' if effect < 0 else 'added'} {abs(effect) * 100:.1f} points a year "
            f"{'off' if effect < 0 else 'to'} the time-weighted return: the whole account made {_pct(perf['twr_annual'])} a "
            f"year, the invested part {_pct(perf['twr_invested_annual'])}. Every deposit put into {primary['label']} on its day "
            f"would be {_eur_abs(primary['deposits_end_value'])} against your {_eur_abs(perf['end_value'])} "
            f"({_eur(primary['deposits_difference'])}): {_eur(primary['buys_difference'])} from your invested choices and "
            f"{_eur(primary['cash_part'])} from the cash.",
            primary["cash_part"], "exact",
            "arithmetic on your export and the index's prices: a what-if, not a forecast; in a falling market the cash would have helped",
            "what-if",
            "Keep a deliberate cash reserve for crashes (the framework buys when the VIX passes 40), but size it on purpose. "
            "The rules lab replays a 10% reserve with the rest in MSCI World (Idle cash into MSCI World).",
            ["Index", "Every deposit in it", "You against it", "From your invested choices", "From the cash"],
            [[b["label"], f"{_eur_abs(b['deposits_end_value'])}", _eur(b["deposits_difference"]), _eur(b["buys_difference"]),
              _eur(b["cash_part"])] for b in benches], section="cash"))

    # 9. Turbos.
    turbo = next((g for g in groups if g["group"] == "Turbos"), None)
    if turbo:
        out.append(_learning(
            "turbos", "Knock-out turbos",
            f"{turbo['instruments']} turbo trades made {_eur(turbo['pnl'])} on {_eur_abs(turbo['bought'])} bought; "
            f"{turbo['winners']} of {turbo['instruments']} were winners. Without them the account would be "
            f"{_eur_abs(abs(turbo['pnl']))} {'higher' if turbo['pnl'] < 0 else 'lower'}.",
            turbo["pnl"], "exact", "accounting from the export; too few trades to judge skill" if turbo["instruments"] < MIN_CASES
            else "accounting from the export", "attribution",
            "Leveraged short-term bets sit outside a 12-month value framework; keep them out or cap them hard.",
            n=turbo["instruments"]))

    # 10. Rules on your entries.
    best = overlays.get("best")
    if best:
        conf, why = _confidence(best["names"], best["t"], _same_sign(best["first_half"], best["second_half"]))
        why += " (one observation per name)"
        out.append(_learning(
            "rules", "Stop-loss, take-profit and minimum-holding rules",
            overlays["verdict"], best["eur"], conf, why, "what-if",
            "No mechanical exit rule beat your own exits robustly; the review's rule (exit at or below the exit level, or on a red flag) "
            "stays." if not overlays["adopt"] else "A rule beat your exits in both halves; test it on more data first.",
            ["Rule", "Lots changed", "EUR", "1st half", "2nd half", "t"],
            [[r["rule"], f"{r['changed']}/{r['lots']}", _eur(r["eur"]), _eur(r["first_half"]), _eur(r["second_half"]),
              f"{r['t']:.1f}" if r["t"] is not None else "n/a"] for r in overlays["rows"]], best["lots"]))

    # 11. FastValue.
    if fv:
        e = fv["entries"]
        buy = e.get("BUY", {})
        rest = _merge(e.get("HOLD", {}), e.get("SELL", {}))
        cmp_ = fv["compare"]
        conf, why = cmp_["confidence"]
        out.append(_learning(
            "fastvalue", "Your buys against FastValue's owner score at the time",
            f"Of {fv['rated_buys']} stock buys, {buy.get('n', 0)} had an owner score at or above the entry level at the month-end before; "
            f"they did {_pct(buy.get('mean'))} against MSCI World over {MAIN_HORIZON} days, against {_pct(rest['mean'])} for "
            f"the {rest['n']} buys below it.",
            buy.get("eur"), conf, why + " (buys at or above the entry level against buys below it)", "pattern",
            _weak(conf) or ("Checking the owner score before buying would have helped." if (cmp_["difference"] or 0) > 0
                            else "On your trades the owner score did not separate winners from losers."),
            ["Owner score at entry", "Buys", f"vs MSCI World after {MAIN_HORIZON} days", "t (clustered)", "EUR vs MSCI World"],
            [[_ENTRY_BUCKET_WORDS.get(k, k), v["n"], _pct(v["mean"]), f"{v['t']:.1f}" if v["t"] is not None else "n/a", _eur(v["eur"])]
             for k, v in e.items() if v["n"]], fv["rated_buys"]))
        sr = fv["sell_rule"]
        if sr["lots"]:
            conf, why = _confidence(sr["names"], sr["t"], _same_sign(sr["first_half"], sr["second_half"]))
            out.append(_learning(
                "fastvalue_sell", "FastValue's exit rule on your entries",
                f"Selling at the first month-end the owner score was at or below the exit level would have changed {sr['changed']} of {sr['lots']} "
                f"lots and {'added' if sr['eur'] >= 0 else 'cost'} {_eur_abs(abs(sr['eur']))} against your actual exits.",
                sr["eur"], conf, why, "what-if",
                "Following the exit level would have helped your exits." if sr["eur"] > 0 else
                "The exit-level rule would not have improved your exits on these trades.", n=sr["lots"]))

    out.sort(key=lambda x: (x["tier"], -abs(x["eur"] or 0.0)))
    return out


def _cash_section(perf: dict[str, Any], beh: dict[str, Any], a: dict[str, Any], m: _Market) -> dict[str, Any]:
    """Cash on its own: how much, what it earned, what it did to the return, and the whole account against every deposit in each index."""
    first = perf["first_trade"]
    acct = a["account"]
    live = (acct > 1.0) & (acct.index >= first)
    share = (a["cash"] / acct).where(live).dropna()
    interest = beh["cash"]["interest_net"]
    span_years = (m.end - first).days / 365.25
    balance = float(a["cash"][live].mean()) if live.any() else 0.0
    rate = interest / balance / span_years if balance >= 1.0 and span_years >= 0.25 else None
    twr, twr_inv = perf["twr_annual"], perf["twr_invested_annual"]
    return {
        "now": perf["cash"], "share_now": perf["cash"] / perf["end_value"] if perf["end_value"] > 1.0 else None,
        "avg_share": perf["avg_cash_share"], "interest_net": interest, "interest_rate": rate,
        "effect_annual": twr - twr_inv if twr is not None and twr_inv is not None else None,
        "twr_total": perf["twr_total"], "twr_annual": twr, "xirr": perf["xirr"],
        "opportunity_eur": beh["cash"]["opportunity_eur"],
        "share": [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
                  for d, v in share.resample("W-FRI").last().dropna().items()],
    }


def _rules_learning(lab: dict[str, Any]) -> dict[str, Any] | None:
    """One learning for the rules lab: the strongest scenario by its weaker half, judged by the protocol."""
    rows = [r for r in lab.get("scenarios", []) if r["key"] != "actual"]
    if not rows:
        return None
    best = rows[0]
    adopted = [r for r in rows if r["adopt"]]
    finding = (f"Of {len(rows)} rules replayed on your whole history, \"{best['label']}\" did best by its weaker half: "
               f"{_eur(best['difference'])} over the period ({_pct(best['difference_pct'])} of the account), "
               f"{_eur(best['first_half'])} when applied in the first half alone and {_eur(best['second_half'])} in the second.")
    if adopted:
        lesson = ("These rules won in both halves with t of at least 2: " + ", ".join(f'"{r["label"]}"' for r in adopted)
                  + ". Write each down as a rule for the next year and judge it on new trades, not on this replay.")
    else:
        lesson = ("No rule passed the protocol (both halves and t of at least 2), so keep what you do. The rules ahead in both "
                  "halves are leads to watch: " + (", ".join(f'"{r["label"]}"' for r in rows if r["first_half"] > 0 and r["second_half"] > 0)
                                                  or "none") + ".")
    return _learning(
        "rules_lab", "Rules replayed on your whole history", finding, best["difference"], best["confidence"],
        best["why"] + " (names the rule changed, one observation each)", "what-if", lesson,
        ["Rule", "Against what you did", "1st half", "2nd half", "t", "Verdict"],
        [[r["label"], _eur(r["difference"]), _eur(r["first_half"]), _eur(r["second_half"]),
          "n/a" if r["t"] is None else f"{r['t']:.1f}", r["verdict"]] for r in rows], len(rows))


# ================================================================== entry point


def result_path() -> Path:
    """The last full track record as JSON (git-ignored output folder; personal). The share report reads it."""
    return settings.OUTPUT_DIR / "track_record.json"


def load_result() -> dict[str, Any] | None:
    import json

    try:
        return json.loads(result_path().read_text())
    except (OSError, ValueError):
        return None


def _save_result(result: dict[str, Any]) -> None:
    import json

    try:
        path = result_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    except (OSError, TypeError) as exc:
        logger.warning("Track record not saved: %s", exc)


def account_history(progress: Progress | None = None) -> dict[str, Any]:
    """The imported account's value and time-weighted growth over time, against MSCI World (Portfolio tab).

    The first part of the track record only: ledger, prices and the daily account value, without the
    trade diagnostics, so it is quick once prices are cached.
    """
    progress = progress or (lambda *_: None)
    stored = tr.load_ledger()
    if not stored:
        raise ValueError("Import your broker export first")
    ledger = build_ledger(stored["events"])
    mapping = ins.resolve(ledger["instruments"], ins.trade_prices(ledger), progress=progress)
    m = _market(ledger, mapping, progress)
    progress(0, 1, "Valuing the account day by day")
    a = _account(ledger, m)
    perf = _performance(ledger, m, a)
    for key in ("_bench_value", "_bench_values", "_mirror_values"):
        perf.pop(key)
    first = perf["first_trade"]
    daily = lambda s: [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 3)}  # noqa: E731
                       for d, v in s[s.index >= first].dropna().items() if d.weekday() < 5]
    growth = (1.0 + a["r_account"][a["r_account"].index >= first]).cumprod() * 100.0
    sleeve = (1.0 + a["r_sleeve"][a["r_sleeve"].index >= first]).cumprod() * 100.0
    bench = m.bench[m.bench.index >= first]
    index_growth = {meta["id"]: daily(series[series.index >= first] / float(series[series.index >= first].iloc[0]) * 100.0)
                    for meta, series in m.benchmarks()}
    progress(1, 1, "Done")
    return _clean({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_label": m.bench_label,
        "period": {"first_trade": first, "end": perf["end"]},
        "benchmark_labels": {meta["id"]: meta["label"] for meta, _ in m.benchmarks()},
        "growth": {"account": daily(growth), "invested_part": daily(sleeve), "benchmark": daily(bench / float(bench.iloc[0]) * 100.0),
                   "benchmarks": index_growth},
        "value": {"account": daily(a["account"]), "put_in": daily(a["flows"].cumsum())},
        "summary": {k: perf[k] for k in ("end_value", "net_external_flows", "gain", "twr_total", "twr_annual", "xirr",
                                         "twr_invested_total", "twr_invested_annual", "benchmark_total", "benchmark_annual",
                                         "avg_cash_share", "invested_xirr")}
        | {"benchmarks": [{k: b[k] for k in ("id", "label", "ticker", "total", "annual", "excess_annual")}
                          for b in perf["benchmarks"]]},
        "years": [{k: y[k] for k in ("year", "twr", "twr_invested", "benchmark", "gain", "partial")} for y in perf["years"]],
        "notes": m.warnings,
    })


def run_track_record(method: str = "fifo", fastvalue: bool = False, progress: Progress | None = None) -> dict[str, Any]:
    """The whole track record of the imported account (Backtest tab, "My track record")."""
    progress = progress or (lambda *_: None)
    stored = tr.load_ledger()
    if not stored:
        raise ValueError("Import your broker export in the Portfolio tab first")
    ledger = build_ledger(stored["events"], method)
    mapping = ins.resolve(ledger["instruments"], ins.trade_prices(ledger), progress=progress)
    m = _market(ledger, mapping, progress)
    progress(0, 1, "Valuing the account day by day")
    a = _account(ledger, m)
    perf = _performance(ledger, m, a)
    instruments = _instruments(ledger, m, a, method)
    trips = _round_trips(ledger, m, a, method)
    orders = _orders(ledger, m)
    lots = _lots(ledger, m)
    beh = _behaviour(ledger, m, a, perf, orders, trips, instruments, lots)
    split = beh["split_date"]
    progress(0, 1, "Testing exit rules on your entries")
    whatif = {"sold_kept": _sold_kept(orders, m, split)}
    overlays = _overlays(lots, m, split)
    fv = None
    fv_error = None
    if fastvalue:
        try:
            fv = _fastvalue(ledger, m, orders, lots, split, progress)
        except Exception as exc:  # noqa: BLE001 - the rest of the record stands without it
            logger.exception("FastValue replay failed")
            fv_error = str(exc)
    learnings = _learnings(perf, beh, instruments, whatif, overlays, fv)
    turbo = next((g for g in beh["groups"] if g["group"] == "Turbos"), None)

    progress(0, 1, "Reading your style and replaying rules on your history")
    style, rules_lab, style_error = None, None, None
    try:
        from core import history_lab, style_profile, weights_review

        macro = None
        try:  # the market regime at each buy: optional, the rest of the profile stands without it
            macro = weights_review.load_macro([o["date"] for o in orders if o["side"] == "buy" and o["priced"] and o["group"] != "Turbos"])
        except Exception:  # noqa: BLE001
            logger.exception("Market regime at entry failed")
        style = style_profile.build_profile(ledger, m, a, perf, orders, trips, instruments, lots, beh, fv,
                                            style_profile._vix_series(m), (macro or {}).get("by_date"))
        try:
            style["weights"] = weights_review.build_review(style, macro, split)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Your history against the weights failed")
            style["weights"] = {"error": str(exc)}
        rules_lab = history_lab.run_rules_lab(ledger, m, a, split, perf["first_trade"], style["volatility"]["classes"],
                                              style["style"]["bets"]["discretionary_median"])
        history_lab.track_runs(rules_lab, ledger["summary"].get("last_event"))
        extra = _rules_learning(rules_lab)
        if extra:
            learnings.append(extra)
            learnings.sort(key=lambda x: (x["tier"], -abs(x["eur"] or 0.0)))
    except Exception as exc:  # noqa: BLE001 - the rest of the record stands without it
        logger.exception("Style profile or rules lab failed")
        style_error = str(exc)

    bench_val = perf.pop("_bench_value")
    bench_values, mirror_values = perf.pop("_bench_values"), perf.pop("_mirror_values")
    weekly = lambda s: [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}  # noqa: E731
                        for d, v in s.resample("W-FRI").last().dropna().items()]
    invested_line = a["flows"].cumsum()
    growth = lambda r: (1.0 + r[r.index >= perf["first_trade"]]).cumprod() * 100.0  # noqa: E731
    since = lambda s: s[s.index >= perf["first_trade"]]  # noqa: E731
    benches = perf["benchmarks"]
    primary = benches[0]
    end_cash = perf["cash"]
    cash = _cash_section(perf, beh, a, m)
    cash_core = next((r for r in (rules_lab or {}).get("scenarios", []) if r["key"] == "cash_to_core"), None)
    # A broker can give different funds the same name: the ISIN and the dates say which one is meant.
    unpriced = [f"{r['name']} ({r['isin']}, " + ("held today" if r["status"] == "open" else f"held {r['first_date']} to {r['last_date']}") + ")"
                for r in instruments if not r["priced"] and r["group"] != "Turbos"]
    notes = [
        "Prices from Yahoo, converted to EUR at ECB rates. The indices are MSCI World, the S&P 500 and the Nasdaq 100 through "
        "their accumulating iShares UCITS funds (IWDA, CSPX and CNDX in London); each order's 30, 90 and 180-day comparison "
        "uses MSCI World only.",
        "Time-weighted return (TWR) removes the effect of deposits and withdrawals: it measures the choices. "
        "Money-weighted (XIRR) includes their timing: it measures what your money earned.",
        "Private-market funds are left out; their payments count as money moved out of the account.",
        "Turbos have no price history: they are valued at cost while held, so their result shows on the day they were sold.",
    ] + ([f"Valued at cost while held (no listing fits your trade prices): {', '.join(unpriced)}."] if unpriced else []) + m.warnings \
      + list(dict.fromkeys(perf["benchmark_notes"] + [n for b in benches for n in b["notes"]])) + ([f"FastValue replay failed: {fv_error}"] if fv_error else []) \
      + ([f"Style profile and rules lab failed: {style_error}"] if style_error else [])
    result = {
        "schema": RESULT_SCHEMA,
        "method": method,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": perf["start"], "first_trade": perf["first_trade"], "end": perf["end"], "split": split},
        "summary": {k: v for k, v in perf.items() if k != "years"},
        "years": perf["years"],
        "equity": {"account": weekly(a["account"]), "invested": weekly(invested_line), "benchmark": weekly(bench_val),
                   "benchmarks": {i: weekly(v) for i, v in bench_values.items()}},
        "growth": {"account": weekly(growth(a["r_account"])), "invested_part": weekly(growth(a["r_sleeve"])),
                   "benchmark": weekly(since(m.bench) / float(since(m.bench).iloc[0]) * 100.0),
                   "benchmarks": {meta["id"]: weekly(since(series) / float(since(series).iloc[0]) * 100.0)
                                  for meta, series in m.benchmarks()}},
        "invested_equity": {"invested_part": weekly(since(a["sleeve"])),
                            "benchmarks": {i: weekly(since(v)) for i, v in mirror_values.items()}},
        "cash": cash,
        "alternatives": [
            {"key": "actual", "label": "What you did", "section": "overview", "end_value": perf["end_value"], "difference": 0.0,
             "note": f"invested part {_pct(perf['twr_invested_annual'])} a year time-weighted; whole account TWR "
                     f"{_pct(perf['twr_annual'])}, XIRR {_pct(perf['xirr'])}"},
        ] + [
            {"key": f"buys_{b['id']}", "label": f"Same buys and sales in {b['label']}", "section": "overview",
             "end_value": end_cash + b["buys_end_value"] if b["buys_end_value"] is not None else None,
             "difference": -b["buys_difference"],
             "note": f"every buy's money into {b['label']} on its day, every sale and dividend out again; your cash as it was; "
                     f"money-weighted {_pct(b['buys_xirr'])} a year against {_pct(perf['invested_xirr'])} for your invested part"}
            for b in benches
        ] + [
            {"key": "sold_kept", "label": "Sold shares kept until today", "section": "overview", "end_value": perf["end_value"] + whatif["sold_kept"]["eur"],
             "difference": whatif["sold_kept"]["eur"],
             "note": f"{_n(whatif['sold_kept']['n'], 'sale', 'sales')} of priced stocks and funds; against the same money in MSCI World "
                     f"{_eur(whatif['sold_kept'].get('eur_vs_index'))}"},
        ] + ([{"key": "no_turbos", "label": "Without turbos", "section": "overview", "end_value": perf["end_value"] - turbo["pnl"],
               "difference": -turbo["pnl"], "note": f"{_n(turbo['instruments'], 'turbo trade', 'turbo trades')}"}] if turbo else []) + (
            [{"key": "rule", "label": f"Best exit rule: {overlays['best']['rule']}", "section": "overview",
              "end_value": perf["end_value"] + overlays["best"]["eur"], "difference": overlays["best"]["eur"],
              "note": (f"best of {len(overlays['rows'])} exit rules by its weaker half: 1st half {_eur(overlays['best']['first_half'])}, "
                       f"2nd half {_eur(overlays['best']['second_half'])}, t "
                       + (f"{overlays['best']['t']:.1f}" if overlays["best"]["t"] is not None else "n/a")
                       + ("; passes the protocol" if overlays["adopt"] else "; does not pass the protocol, so keep what you do"))}]
            if overlays.get("best") else []) + (
            [{"key": "fastvalue_sell", "label": "FastValue's sell rule on your entries", "section": "overview",
              "end_value": perf["end_value"] + fv["sell_rule"]["eur"], "difference": fv["sell_rule"]["eur"],
              "note": f"{fv['sell_rule']['changed']} lots would have been sold earlier"}] if fv else []) + [
            {"key": f"deposits_{b['id']}", "label": f"Every deposit into {b['label']}", "section": "cash",
             "end_value": b["deposits_end_value"], "difference": b["deposits_end_value"] - perf["end_value"],
             "note": f"same deposits and withdrawals on the same days, nothing kept in cash; XIRR {_pct(b['deposits_xirr'])}; "
                     f"your lead or lag: {_eur(b['buys_difference'])} from your invested choices, {_eur(b['cash_part'])} from the cash"}
            for b in benches
        ] + ([{"key": "cash_to_core", "label": "Idle cash above 10% into MSCI World", "section": "cash",
               "end_value": cash_core["end_value"], "difference": cash_core["difference"],
               "note": f"rules lab: month-end sweep with a 10% reserve and the account's fee; {cash_core['verdict']}"}]
             if cash_core else []),
        "learnings": learnings,
        "instruments": instruments,
        "round_trips": sorted(trips, key=lambda t: t["start"]),
        "behaviour": {k: v for k, v in beh.items() if k not in ("exits",)} | {"exits": {k: v for k, v in beh["exits"].items() if k != "rows"}},
        "overlays": {k: v for k, v in overlays.items() if k != "best"} | {"best": overlays.get("best")},
        "orders": orders,
        "sold_kept": whatif["sold_kept"],
        "fastvalue": fv,
        "style": style,
        "rules_lab": rules_lab,
        "ledger_summary": ledger["summary"],
        "notes": notes,
    }
    result = _clean(result)
    progress(1, 1, "Writing the report")
    try:
        write_report(result)
        result["report"] = REPORT_NAME
    except OSError as exc:
        logger.warning("Report not written: %s", exc)
    _save_result(result)
    return result


def run_rules(rules: list[Any] | None = None, include_presets: bool = True, method: str = "fifo",
              progress: Progress | None = None) -> dict[str, Any]:
    """The rules lab alone (Backtest tab, custom rules): the same market and account as the track record."""
    from core import history_lab, style_profile

    progress = progress or (lambda *_: None)
    stored = tr.load_ledger()
    if not stored:
        raise ValueError("Import your broker export in the Portfolio tab first")
    ledger = build_ledger(stored["events"], method)
    mapping = ins.resolve(ledger["instruments"], ins.trade_prices(ledger), progress=progress)
    m = _market(ledger, mapping, progress)
    progress(0, 1, "Replaying your history under the rules")
    a = _account(ledger, m)
    perf = _performance(ledger, m, a)
    split = _split_date(m, perf["first_trade"])
    vol = style_profile.volatility_classes(ledger, m)
    disc = [abs(o["amount"]) for o in ledger["orders"] if o["side"] == "buy" and not o["savings_plan"]
            and ledger["instruments"][o["isin"]]["asset_class"] != "DERIVATIVE"]
    median = float(np.median(disc)) if disc else None
    result = history_lab.run_rules_lab(ledger, m, a, split, perf["first_trade"], {i: r["class"] for i, r in vol["classes"].items()},
                                       median, list(rules or []), include_presets)
    progress(1, 1, "Done")
    return _clean(result)


# ================================================================== HTML report


def report_path() -> Path:
    return settings.OUTPUT_DIR / REPORT_NAME


def _table(columns: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape('' if v is None else str(v))}</td>" for v in r) + "</tr>" for r in rows)
    return f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


REPORT_COLORS = {"invested_part": "#2a78d6", "account": "#2a78d6", "invested": "#9ca3af", "msci_world": "#eb6834",
                 "sp500": "#1baf7a", "nasdaq100": "#eda100", "benchmark": "#eb6834"}


def _svg_lines(series: dict[str, list[dict[str, Any]]], colors: dict[str, str], labels: dict[str, str] | None = None,
               dashed: tuple[str, ...] = ()) -> str:
    labels = labels or {}
    pts = {k: [(pd.Timestamp(p["time"]), p["value"]) for p in v if p["value"] is not None] for k, v in series.items()}
    allp = [p for v in pts.values() for p in v]
    if not allp:
        return ""
    t0, t1 = min(p[0] for p in allp), max(p[0] for p in allp)
    v0, v1 = min(p[1] for p in allp), max(p[1] for p in allp)
    w, h, pad = 900, 260, 30
    span_t = max((t1 - t0).days, 1)
    span_v = (v1 - v0) or 1.0
    lines = []
    for k, v in pts.items():
        d = " ".join(f"{pad + (p[0] - t0).days / span_t * (w - 2 * pad):.1f},{h - pad - (p[1] - v0) / span_v * (h - 2 * pad):.1f}" for p in v)
        dash = " stroke-dasharray='5 4'" if k in dashed else ""
        lines.append(f"<polyline fill='none' stroke='{colors.get(k, '#888')}' stroke-width='2'{dash} points='{d}'/>")
    legend = " ".join(f"<span style='color:{colors.get(k, '#888')}'>&#9632; {html.escape(labels.get(k, k))}</span>" for k in pts)
    labels = (f"<text x='{pad}' y='{h - 8}' class='ax'>{t0.date()}</text><text x='{w - pad}' y='{h - 8}' class='ax' text-anchor='end'>{t1.date()}</text>"
              f"<text x='4' y='{pad}' class='ax'>{v1:,.0f}</text><text x='4' y='{h - pad}' class='ax'>{v0:,.0f}</text>")
    return f"<p class='legend'>{legend}</p><svg viewBox='0 0 {w} {h}' role='img'>{''.join(lines)}{labels}</svg>"


def write_report(r: dict[str, Any], path: Path | None = None) -> Path:
    """A static HTML report of the whole track record (git-ignored output folder; personal)."""
    path = path or report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    s = r["summary"]
    pct = lambda v: "n/a" if v is None else f"{v:+.1%}"  # noqa: E731
    eur = _eur_abs  # the app's money format: €13,367, -€33
    benches = s.get("benchmarks") or []  # missing in a result made before D-36
    labels = {b["id"]: b["label"] for b in benches}
    primary = benches[0] if benches else None
    growth, equity = r.get("growth") or {}, r.get("equity") or {}
    c = r.get("cash")
    stats = [("Account value", f"{eur(s['end_value'])}"), ("Net deposited", f"{eur(s['net_external_flows'])}"),
             ("Gain", f"{eur(s['gain'])}")]
    if benches:
        stats += [("Invested part, TWR a year", pct(s["twr_invested_annual"])),
                  ("Invested part, money-weighted", pct(s.get("invested_xirr")))]
        stats += [(f"{b['label']} a year", pct(b["annual"])) for b in benches]
        stats += [(f"Same buys and sales in {primary['label']}", _eur(primary["buys_difference"]))]
    else:
        stats += [("TWR a year", pct(s["twr_annual"])), ("XIRR", pct(s["xirr"])),
                  ("Invested part, TWR a year", pct(s["twr_invested_annual"])), ("MSCI World a year", pct(s["benchmark_annual"])),
                  ("Same flows in MSCI World", f"{eur(s['benchmark_end_value'])}")]
    stats_html = lambda items: "<div class='stats'>" + "".join(  # noqa: E731
        f"<div><b>{html.escape(k)}</b><span>{html.escape(v)}</span></div>" for k, v in items) + "</div>"
    if benches and growth.get("benchmarks"):
        top = ("<h2>Invested part against the indices</h2>"
               + _svg_lines({"invested_part": growth["invested_part"], **growth["benchmarks"]}, REPORT_COLORS,
                            {"invested_part": "Invested part", **labels}))
    else:
        top = "<h2>Account value</h2>" + _svg_lines(equity, REPORT_COLORS, {"benchmark": "Same flows in MSCI World"})
    cash_html = ""
    if c and benches:
        cash_stats = [("Cash now", f"{eur(c['now'])}"), ("Average cash share", "n/a" if c["avg_share"] is None else f"{c['avg_share']:.0%}"),
                      ("Interest after tax", f"{eur(c['interest_net'])}"),
                      ("Cash effect", "n/a" if c["effect_annual"] is None else f"{c['effect_annual'] * 100:+.1f} points a year"),
                      ("Whole account, TWR a year", pct(c["twr_annual"])), ("XIRR", pct(c["xirr"]))]
        cash_stats += [(f"Every deposit into {b['label']}: {_eur(b['buys_difference'])} your choices, {_eur(b['cash_part'])} your cash",
                        f"{eur(b['deposits_end_value'])}") for b in benches]
        cash_html = ("<h2>Cash</h2>" + stats_html(cash_stats)
                     + _svg_lines({"account": equity.get("account", []), "invested": equity.get("invested", []),
                                   **(equity.get("benchmarks") or {})}, REPORT_COLORS,
                                  {"invested": "Put in, net", "account": "Your account",
                                   **{k: f"Every deposit into {v}" for k, v in labels.items()}}, dashed=("invested",)))
    parts = [
        f"<h1>Track record</h1><p class='muted'>{r['period']['first_trade']} to {r['period']['end']} · cost method "
        f"{r['method']} · generated {r['generated_at']}. Personal: this file stays on your machine.</p>",
        stats_html(stats),
        top,
        cash_html,
        "<h2>Learnings, ranked</h2>" + "".join(
            f"<div class='card'><h3>{i + 1}. {html.escape(x['title'])} <small>{html.escape(x['confidence'])} confidence · "
            f"{html.escape(_eur(x['eur']))}</small></h3><p>{html.escape(x['finding'])}</p><p><i>{html.escape(x['lesson'])}</i></p>"
            f"<p class='muted'>{html.escape(x['why'])}</p>"
            + (_table(x["evidence"]["columns"], x["evidence"]["rows"]) if x["evidence"]["rows"] else "") + "</div>"
            for i, x in enumerate(r["learnings"])),
        "<h2>What if</h2>" + _table(["Alternative", "End value", "Difference", "Note"],
                                    [[a["label"], eur(a["end_value"]), eur(a["difference"]), a["note"]] for a in r["alternatives"]]),
        "<h2>By year</h2>" + _table(["Year", "Start", "End", "Net flows", "Gain", "TWR", "Invested part", "Modified Dietz",
                                     "MSCI World"] + [b["label"] for b in benches[1:]] + (["Cash share"] if benches else [])
                                    + ["Realised (avg)", "Realised (FIFO)", "Dividends", "Interest"],
                                    [[y["year"], eur(y["start_value"]), eur(y["end_value"]), eur(y["net_flows"]), eur(y["gain"]),
                                      pct(y["twr"]), pct(y["twr_invested"]), pct(y["dietz"]), pct(y["benchmark"])]
                                     + [pct((y.get("benchmarks") or {}).get(b["id"])) for b in benches[1:]]
                                     + (["n/a" if y.get("avg_cash_share") is None else f"{y['avg_cash_share']:.0%}"] if benches else [])
                                     + [eur(y["realised_average"]), eur(y["realised_fifo"]), eur(y["dividends_net"]),
                                        eur(y["interest_net"])] for y in r["years"]]),
        "<h2>Exit rules on your entries</h2><p>" + html.escape(r["overlays"]["verdict"]) + "</p><p class='muted'>"
        + html.escape(r["overlays"]["protocol"]) + "</p>"
        + _table(["Rule", "Lots changed", "EUR", "1st half", "2nd half", "t"],
                 [[o["rule"], f"{o['changed']}/{o['lots']}", eur(o["eur"]), eur(o["first_half"]), eur(o["second_half"]),
                   "n/a" if o["t"] is None else f"{o['t']:.1f}"] for o in r["overlays"]["rows"]]),
        "<h2>Holdings</h2>" + _table(["Holding", "Ticker", "Group", "Status", "Bought", "Sold", "Dividends", "Fees",
                                      "Total P&L", "Return", "XIRR"],
                                     [[i["name"], i["ticker"], i["group"], i["status"], eur(i["bought"]), eur(i["sold"]),
                                       eur(i["dividends"]), eur(i["fees"]), eur(i["total_pnl"]), pct(i["return_pct"]),
                                       pct(i["xirr"])] for i in r["instruments"]]),
        "<h2>Round trips</h2>" + _table(["Holding", "From", "To", "Days", "Outcome", "Bought", "Avg buy", "Avg sell",
                                         "Total P&L", "Return", "Worst while held", "Best while held"],
                                        [[t["name"], t["start"], t["end"] or "open", t["days"], t["outcome"], eur(t["bought"]),
                                          "n/a" if t["avg_buy"] is None else f"{t['avg_buy']:.2f}",
                                          "n/a" if t["avg_sell"] is None else f"{t['avg_sell']:.2f}", eur(t["total_pnl"]),
                                          pct(t["return_pct"]), pct(t["mae"]), pct(t["mfe"])] for t in r["round_trips"]]),
        "<h2>Every order</h2><p class='muted'>Each order is compared with MSCI World only.</p>" + _table(["Date", "Holding", "Side", "Kind", "Price", "EUR", "Plan", "52w range",
                                         "+30d", "+90d", "+180d", "+90d vs MSCI World"],
                                        [[o["date"], o["name"], o["side"], o.get("kind"), f"{o['price']:.2f}" if o["price"] else "",
                                          eur(o["amount"]), "yes" if o["savings_plan"] else "",
                                          "" if o.get("range_52w") is None else f"{o['range_52w']:.0%}",
                                          pct(o.get("ret_30")), pct(o.get("ret_90")), pct(o.get("ret_180")),
                                          pct(o.get("excess_90"))] for o in r["orders"]]),
        "<h2>Notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in r["notes"]) + "</ul>",
    ]
    style = """
    :root{--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--card:#f9fafb}
    @media (prefers-color-scheme:dark){:root{--bg:#0b0f17;--fg:#e5e7eb;--muted:#9ca3af;--line:#1f2937;--card:#111827}}
    body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px}
    h1{margin:0 0 4px}h2{margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:4px}
    .muted,.ax{color:var(--muted);fill:var(--muted);font-size:12px}.card{background:var(--card);border:1px solid var(--line);
    border-radius:8px;padding:8px 14px;margin:10px 0}.card h3{margin:6px 0}.card small{color:var(--muted);font-weight:400}
    .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}.stats div{background:var(--card);
    border:1px solid var(--line);border-radius:8px;padding:8px}.stats b{display:block;font-size:12px;color:var(--muted);font-weight:500}
    .stats span{font-size:18px}.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px}
    th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
    svg{width:100%;height:auto}.legend span{margin-right:14px}"""
    path.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                    f"<title>Track record</title><style>{style}</style></head><body>{''.join(parts)}</body></html>")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Track record of the imported broker account (writes an HTML report).")
    parser.add_argument("--csv", help="import this Trade Republic export first (replaces the stored import)")
    parser.add_argument("--method", choices=("fifo", "average"), default="fifo")
    parser.add_argument("--fastvalue", action="store_true", help="replay FastValue's owner rating at each entry (slower)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.csv:
        from core.broker_import import import_trade_republic

        import_trade_republic(Path(args.csv).expanduser().read_bytes())
    r = run_track_record(args.method, args.fastvalue, progress=lambda d, t, s: print(f"  {s}", flush=True))
    s = r["summary"]
    b = s["benchmarks"]
    print(f"Invested part {s['twr_invested_annual']:+.1%} a year ("
          + ", ".join(f"{x['label']} {x['annual']:+.1%}" for x in b)
          + f"); same buys and sales in {b[0]['label']} {_eur(b[0]['buys_difference'])}. Account {_eur_abs(s['end_value'])}, "
          f"cash {(r['cash']['effect_annual'] or 0) * 100:+.1f} points a year.")
    for i, x in enumerate(r["learnings"][:5], 1):
        print(f"{i}. [{x['confidence']}] {x['title']}: {x['finding']}")
    print(f"Report: {report_path()}")


if __name__ == "__main__":
    main()
