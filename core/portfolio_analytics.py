"""Portfolio analytics: allocation, risk, correlation, performance, winners and losers, currency and cycle.

The review (``core.portfolio``) rates each holding; this module looks at the
holdings together. It is pure: prices, profiles, trades and macro readings are
passed in, so every number can be checked offline. Nothing here changes a
decision: the owner score still decides BUY / HOLD / SELL (decision_log.md D-24).

Method:

- One currency, one calendar. Each holding's daily close (dividend-adjusted)
  is converted into the portfolio currency at that day's ECB rate and put on a
  business-day calendar; a missing day carries the last price for up to a
  week. Returns are what you earned in your own currency, currency moves
  included.
- Today's mix: today's weights held constant through history, cash earning
  nothing. It measures the risk you hold now, not your past results (those
  come from your actual trades, in My track record).
- Risk and correlation use weekly returns (Friday to Friday) over the last two
  years. Daily returns understate how markets move together when they close
  at different hours: a European and a US stock react to the same news on
  different days. Weekly returns remove most of that bias.
- Share of risk: each holding's part of the mix's variance, weight times its
  covariance with the mix divided by the mix's variance (Euler); the shares
  add up to 100%. A holding with a larger share of risk than of value swings
  the portfolio more than its size suggests.
- Diversification ratio: the weighted average volatility of the holdings over
  the volatility of the mix. 1 means nothing cancels out; 2 means half of the
  holdings' swings cancel.
- Bad month: the 5th percentile of 21-trading-day returns of today's mix over
  three years (one month in twenty was worse).
- Period P&L of a holding: value now, minus value at the start, minus money
  put in during the period, plus dividends. Imported holdings use their
  actual trades (savings plans included); positions entered by hand count as
  one purchase on the date opened. The return is Modified Dietz.
- Currency split: P&L = price effect + currency effect. With C the cost in
  the listing's currency (each lot at its day's rate), V the value in that
  currency and f the rate now: price effect = (V - C) x f, currency effect =
  C x f - cost. The two add up to the P&L exactly.
- Exposures look through funds where Yahoo publishes their sector mix; a fund's
  region comes from its name. Sectors are grouped into cyclical, sensitive and
  defensive (Morningstar's super sectors) and compared with the benchmark.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from data.fx import amount_text, major
from data.profiles import SUPER_SECTOR_TIPS, SUPER_SECTORS

FxFn = Callable[[str, pd.DatetimeIndex], pd.Series]

RISK_WEEKS = 104  # two years of weekly returns for risk, beta and correlation
MIN_WEEKS = 26  # a holding needs half a year of prices to enter the risk figures
PERF_YEARS = 5
FFILL_DAYS = 5
MATRIX_MAX = 20  # largest holdings shown in the correlation heatmap
BAD_MONTH_DAYS = 21
BAD_MONTH_YEARS = 3
PERIODS: tuple[tuple[str, int | None], ...] = (("1M", 1), ("3M", 3), ("6M", 6), ("YTD", None), ("1Y", 12), ("3Y", 36), ("5Y", 60))
MOVER_PERIODS = ("1M", "3M", "YTD", "1Y")
EPS = 1e-9

# Thresholds for the plain-language alerts. They only choose what to point out; no decision depends on them.
HIGH_CORRELATION = 0.7  # weekly correlation from which two holdings count as one bet
RISK_SHARE_ALERT = 0.15  # share of risk worth a note when it is also ...
RISK_TO_WEIGHT_ALERT = 2.0  # ... at least twice the holding's share of the invested value
SECTOR_ACTIVE_ALERT = 0.20  # 20 points above the benchmark's sector weight
FOREIGN_CURRENCY_ALERT = 0.50  # half of the portfolio priced in other currencies
BENCHMARK_LIKE_CORRELATION = 0.95
BENCHMARK_LIKE_BETA_BAND = 0.10
CURRENCY_SHOCK = 0.10  # the "what a 10% currency move costs" illustration

CASH = "Cash"
NON_SECTOR = {"Commodity": "Commodities", "Bond": "Bonds", "Crypto": "Crypto"}


def _f(x: Any, digits: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _points(s: pd.Series, digits: int = 3) -> list[dict[str, Any]]:
    return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), digits)} for d, v in s.dropna().items()]


def period_start(as_of: pd.Timestamp, months: int | None) -> pd.Timestamp:
    """Start of a look-back period: ``months`` back, or the last day of the previous year (YTD)."""
    if months is None:
        return pd.Timestamp(as_of.year - 1, 12, 31)
    return as_of - pd.DateOffset(months=months)


def _asof(s: pd.Series | None, day: pd.Timestamp) -> float | None:
    if s is None:
        return None
    sub = s.loc[:day].dropna()
    return float(sub.iloc[-1]) if len(sub) else None


# ================================================================== prices


def price_frame(closes: dict[str, pd.Series], currencies: dict[str, str], currency: str, fx: FxFn
                ) -> tuple[pd.DataFrame, list[str]]:
    """Daily prices in the portfolio currency on one business-day calendar (gaps carried up to a week)."""
    cols: dict[str, pd.Series] = {}
    notes: list[str] = []
    for t, s in closes.items():
        s = pd.Series(s, dtype=float).dropna()
        s = s[s > 0]
        if s.empty:
            continue
        idx = pd.DatetimeIndex(s.index)
        s.index = idx.tz_localize(None) if idx.tz is not None else idx
        try:
            factor = fx(currencies.get(t) or currency, s.index)
        except Exception as exc:  # noqa: BLE001 - one missing rate must not sink the analytics
            notes.append(f"{t}: no exchange-rate history ({exc}); left out of the risk figures.")
            continue
        conv = (s * np.asarray(factor, dtype=float)).dropna()
        if not conv.empty:
            cols[t] = conv
    if not cols:
        return pd.DataFrame(), notes
    raw = pd.DataFrame(cols).sort_index()
    days = pd.bdate_range(raw.index.min(), raw.index.max())
    frame = raw.reindex(raw.index.union(days)).ffill(limit=FFILL_DAYS).reindex(days)
    return frame, notes


def _on_calendar(s: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    return s.reindex(s.index.union(days)).ffill(limit=FFILL_DAYS).reindex(days)


def _mix_returns(frame: pd.DataFrame, w: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Daily return of today's mix (a holding without a price that day counts as cash) and the invested share covered."""
    cols = [t for t in w.index if t in frame.columns]
    r = frame[cols].pct_change(fill_method=None).iloc[1:]
    wv = w[cols]
    mix = (r.fillna(0.0) * wv).sum(axis=1)
    invested = float(wv.sum())
    coverage = (r.notna() * wv).sum(axis=1) / invested if invested > 0 else pd.Series(0.0, index=r.index)
    return mix, coverage


# ================================================================== performance


def _performance(frame: pd.DataFrame, w: pd.Series, bench: pd.Series | None,
                 others: dict[str, pd.Series] | None = None) -> dict[str, Any]:
    mix, coverage = _mix_returns(frame, w)
    if mix.empty:
        return {"series": {}, "periods": [], "as_of": None}
    as_of = mix.index[-1]
    start = as_of - pd.DateOffset(years=PERF_YEARS)
    window = mix[mix.index > start]
    index = 100.0 * (1.0 + window).cumprod()
    series: dict[str, Any] = {"mix": _points(index)}
    # Today's holdings: the invested weights scaled to 100%, cash left out (the track record leaves it out too).
    hold = None
    if float(w.sum()) > 0:
        hold = _mix_returns(frame, w / w.sum())[0]
        series["holdings"] = _points(100.0 * (1.0 + hold[hold.index > start]).cumprod())
    returns: dict[str, pd.Series] = {}  # daily returns of each index on the mix's days
    prices: dict[str, pd.Series] = {}
    if bench is not None and bench.notna().sum() > 1:
        returns["benchmark"] = bench.pct_change(fill_method=None).iloc[1:].reindex(mix.index)
        prices["benchmark"] = bench
        bw = returns["benchmark"][returns["benchmark"].index > start].fillna(0.0)
        series["benchmark"] = _points(100.0 * (1.0 + bw).cumprod())
    for key, other in (others or {}).items():
        returns[key] = other.pct_change(fill_method=None).iloc[1:].reindex(mix.index)
        prices[key] = other
    if others:
        series["benchmarks"] = {
            key: _points(100.0 * (1.0 + returns[key][returns[key].index > start].fillna(0.0)).cumprod()) for key in others}

    def annual(total: float | None, years: float) -> float | None:
        return _f((1.0 + total) ** (1.0 / years) - 1.0) if years >= 1.5 and total is not None and total > -1 else None

    def total_of(key: str, s: pd.Timestamp) -> float | None:
        """An index's return from ``s``; None when it has no price that early (not a zero-filled return)."""
        if key not in returns or _asof(prices[key], s) is None:
            return None
        return float((1.0 + returns[key][returns[key].index > s].fillna(0.0)).prod() - 1.0)

    periods = []
    first = frame.index[0]
    for key, months in PERIODS:
        s = period_start(as_of, months)
        if s < first:
            continue
        sel = mix[mix.index > s]
        if sel.empty:
            continue
        years = (as_of - s).days / 365.25
        total = float((1.0 + sel).prod() - 1.0)
        b_total = total_of("benchmark", s)
        h_total = float((1.0 + hold[hold.index > s]).prod() - 1.0) if hold is not None else None
        o_totals = {k: total_of(k, s) for k in (others or {})}
        cov_at = coverage[coverage.index > s]
        period = {
            "key": key, "start": s.strftime("%Y-%m-%d"),
            "mix": _f(total), "benchmark": _f(b_total),
            "mix_annual": _f((1.0 + total) ** (1.0 / years) - 1.0) if years >= 1.5 and total > -1 else None,
            "benchmark_annual": annual(b_total, years),
            "coverage": _f(cov_at.iloc[0], 3) if len(cov_at) else None,
        }
        if hold is not None:
            period.update({"holdings": _f(h_total), "holdings_annual": annual(h_total, years)})
        if others:
            period.update({"benchmarks": {k: _f(v) for k, v in o_totals.items()},
                           "benchmarks_annual": {k: annual(v, years) for k, v in o_totals.items()}})
        periods.append(period)
    return {"series": series, "periods": periods, "as_of": as_of.strftime("%Y-%m-%d"),
            "start": window.index[0].strftime("%Y-%m-%d") if len(window) else None}


# ================================================================== risk and correlation


def _weekly_returns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.resample("W-FRI").last().pct_change(fill_method=None).iloc[1:]


def _greedy_order(corr: pd.DataFrame) -> list[str]:
    """Order names so that similar ones sit next to each other: start from the most connected, then nearest neighbour."""
    names = list(corr.columns)
    if len(names) <= 2:
        return names
    c = corr.fillna(-2.0)
    mean = (c.sum() - 1.0) / (len(names) - 1)
    order = [str(mean.idxmax())]
    left = [n for n in names if n != order[0]]
    while left:
        nxt = max(left, key=lambda n: c.loc[order[-1], n])
        order.append(nxt)
        left.remove(nxt)
    return order


def _risk(frame: pd.DataFrame, w: pd.Series, bench: pd.Series | None) -> dict[str, Any]:
    out: dict[str, Any] = {"basis": f"weekly returns in the portfolio currency, last {RISK_WEEKS // 52} years",
                           "volatility": None, "positions": {}, "excluded": [], "correlation": None}
    weekly = _weekly_returns(frame[[t for t in w.index if t in frame.columns]]).tail(RISK_WEEKS)
    if weekly.empty:
        return out
    counts = weekly.notna().sum()
    usable = [t for t in w.index if t in counts.index and counts[t] >= MIN_WEEKS]
    out["excluded"] = [t for t in w.index if t not in usable]
    out["weeks"] = int(len(weekly))
    if not usable:
        return out
    R = weekly[usable]
    Rf = R.fillna(0.0)
    wu = w[usable].to_numpy(float)
    cov = Rf.cov().to_numpy() * 52.0
    var_p = float(wu @ cov @ wu)
    vol_p = math.sqrt(var_p) if var_p > 0 else 0.0
    asset_vol = R.std() * math.sqrt(52.0)
    port = Rf.to_numpy() @ wu
    port_s = pd.Series(port, index=R.index)
    rc = (wu * (cov @ wu) / var_p) if var_p > 0 else np.full(len(usable), np.nan)

    b = None
    if bench is not None:
        b = bench.resample("W-FRI").last().pct_change(fill_method=None).reindex(R.index)
    beta_p = corr_p = te = b_vol = None
    if b is not None and b.notna().sum() >= MIN_WEEKS:
        ok = b.notna()
        bv = b[ok]
        var_b = float(bv.var())
        if var_b > 0:
            beta_p = float(np.cov(port_s[ok], bv)[0, 1] / var_b)
            corr_p = float(np.corrcoef(port_s[ok], bv)[0, 1])
            te = float((port_s[ok] - bv).std() * math.sqrt(52.0))
            b_vol = float(bv.std() * math.sqrt(52.0))

    weighted_vol = float(np.nansum(wu * asset_vol.to_numpy()))
    positions = {}
    for i, t in enumerate(usable):
        rest = port_s - wu[i] * Rf[t]
        ok = R[t].notna()
        corr_rest = float(np.corrcoef(R[t][ok], rest[ok])[0, 1]) if ok.sum() >= MIN_WEEKS and rest[ok].std() > 0 else None
        beta = None
        if b is not None:
            both = ok & b.notna()
            if both.sum() >= MIN_WEEKS and b[both].var() > 0:
                beta = float(np.cov(R[t][both], b[both])[0, 1] / b[both].var())
        positions[t] = {"volatility": _f(asset_vol[t]), "risk_share": _f(rc[i]), "beta": _f(beta, 2),
                        "corr_to_rest": _f(corr_rest, 2), "weeks": int(counts[t])}
    out.update({
        "volatility": _f(vol_p), "benchmark_volatility": _f(b_vol),
        "beta": _f(beta_p, 2), "correlation_to_benchmark": _f(corr_p, 2), "tracking_error": _f(te),
        "diversification_ratio": _f(weighted_vol / vol_p, 2) if vol_p > 0 else None,
        "positions": positions,
    })

    # Correlation: heatmap of the largest holdings, and the pairs that move together across all of them.
    corr_all = R.corr(min_periods=MIN_WEEKS)
    pairs = []
    for i, a in enumerate(usable):
        for c in usable[i + 1:]:
            v = corr_all.loc[a, c]
            if pd.notna(v):
                pairs.append((a, c, float(v)))
    top = sorted(usable, key=lambda t: -float(w[t]))[:MATRIX_MAX]
    sub = corr_all.loc[top, top]
    order = _greedy_order(sub)
    matrix = [[_f(sub.loc[a, c], 2) for c in order] for a in order]
    high = sorted([p for p in pairs if p[2] >= HIGH_CORRELATION], key=lambda p: -p[2])
    hi = max(pairs, key=lambda p: p[2], default=None)
    lo = min(pairs, key=lambda p: p[2], default=None)
    out["correlation"] = {
        "tickers": order, "matrix": matrix,
        "average": _f(np.mean([p[2] for p in pairs]), 3) if pairs else None,
        "max": {"pair": [hi[0], hi[1]], "value": _f(hi[2], 3)} if hi else None,
        "min": {"pair": [lo[0], lo[1]], "value": _f(lo[2], 3)} if lo else None,
        "high_pairs": [{"pair": [a, c], "value": _f(v, 2), "weight": _f(float(w[a] + w[c]))} for a, c, v in high[:12]],
        "high_threshold": HIGH_CORRELATION, "shown": len(order), "total": len(usable),
    }
    return out


def _drawdowns(mix: pd.Series, total: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if mix.empty:
        return out
    as_of = mix.index[-1]
    for key, years in (("1y", 1), ("3y", 3)):
        sel = mix[mix.index > as_of - pd.DateOffset(years=years)]
        if len(sel) < 60:
            continue
        idx = (1.0 + sel).cumprod()
        dd = idx / idx.cummax() - 1.0
        trough = dd.idxmin()
        peak = idx.loc[:trough].idxmax()
        out[f"max_drawdown_{key}"] = {"value": _f(dd.min()), "peak": peak.strftime("%Y-%m-%d"),
                                      "trough": trough.strftime("%Y-%m-%d")}
        out[f"return_{key}"] = _f(idx.iloc[-1] - 1.0)
    sel = mix[mix.index > as_of - pd.DateOffset(years=BAD_MONTH_YEARS)]
    if len(sel) >= BAD_MONTH_DAYS * 6:
        roll = np.expm1(np.log1p(sel).rolling(BAD_MONTH_DAYS).sum()).dropna()
        q = float(roll.quantile(0.05))
        worst_end = roll.idxmin()
        pos = sel.index.get_loc(worst_end)
        worst_start = sel.index[max(pos - BAD_MONTH_DAYS + 1, 0)]
        out["bad_month"] = {"value": _f(q), "amount": _f(q * total, 2), "years": BAD_MONTH_YEARS}
        out["worst_month"] = {"value": _f(roll.min()), "amount": _f(float(roll.min()) * total, 2),
                              "start": worst_start.strftime("%Y-%m-%d"), "end": worst_end.strftime("%Y-%m-%d")}
    return out


# ================================================================== exposures


def _pieces(profile: dict[str, Any], dim: str, ticker: str, label: str, price_ccy: str) -> list[tuple[str, float]]:
    kind = profile.get("asset_type") or "Other"
    if dim == "position":
        return [(label, 1.0)]
    if dim == "asset_type":
        return [(kind, 1.0)]
    if dim in ("sector", "cycle"):
        if kind == "Stock":
            pieces = [(profile.get("sector") or "Unknown sector", 1.0)]
        elif profile.get("fund_sectors"):
            pieces = list(profile["fund_sectors"].items())
        elif kind in NON_SECTOR:
            pieces = [(NON_SECTOR[kind], 1.0)]
        elif kind in ("ETF", "Fund"):
            pieces = [("Fund, sector mix unknown", 1.0)]
        else:
            pieces = [("Unknown sector", 1.0)]
        if dim == "cycle":
            return [(SUPER_SECTORS.get(s, s if s in NON_SECTOR.values() else "Unclassified"), f) for s, f in pieces]
        return pieces
    if dim == "industry":
        if kind == "Stock":
            return [(profile.get("industry") or "Unknown industry", 1.0)]
        return [({"ETF": "Funds (many industries)", "Fund": "Funds (many industries)"}.get(kind, NON_SECTOR.get(kind, "Other")), 1.0)]
    if dim == "country":
        return [(profile.get("country") or "Unknown", 1.0)]
    if dim == "region":
        return [(profile.get("region") or "Unknown", 1.0)]
    if dim == "currency":
        return [(major(price_ccy)[0], 1.0)]
    raise ValueError(dim)


DIMENSIONS = ("position", "asset_type", "sector", "cycle", "industry", "country", "region", "currency")


def _exposures(items: list[dict[str, Any]], cash: float, total: float, currency: str,
               risk_share: dict[str, float], bench_sectors: dict[str, float] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dim in DIMENSIONS:
        groups: dict[str, dict[str, Any]] = {}
        for it in items:
            for label, frac in _pieces(it["profile"], dim, it["ticker"], it["label"], it["price_currency"]):
                g = groups.setdefault(label, {"label": label, "value": 0.0, "risk_share": 0.0, "tickers": [], "_risk": False})
                g["value"] += it["value"] * frac
                if it["ticker"] in risk_share:
                    g["risk_share"] += risk_share[it["ticker"]] * frac
                    g["_risk"] = True
                if it["ticker"] not in g["tickers"]:
                    g["tickers"].append(it["ticker"])
        rows = sorted(groups.values(), key=lambda g: -g["value"])
        if cash > 0:
            label = currency if dim == "currency" else CASH
            hit = next((g for g in rows if g["label"] == label), None)
            if hit is not None:
                hit["value"] += cash
                hit["cash"] = True
            else:
                rows.append({"label": label, "value": cash, "risk_share": 0.0, "tickers": [], "_risk": True, "cash": True})
        for g in rows:
            g["weight"] = _f(g["value"] / total) if total > 0 else None
            g["value"] = round(g["value"], 2)
            g["risk_share"] = _f(g["risk_share"]) if g.pop("_risk") and risk_share else None
        if dim in ("sector", "cycle"):
            base = SUPER_SECTORS if dim == "sector" else SUPER_SECTOR_TIPS
            equity = sum(g["value"] for g in rows if g["label"] in base)
            bench = None
            if bench_sectors:
                bench = dict(bench_sectors) if dim == "sector" else {}
                if dim == "cycle":
                    for s, v in bench_sectors.items():
                        bench[SUPER_SECTORS.get(s, "Unclassified")] = bench.get(SUPER_SECTORS.get(s, "Unclassified"), 0.0) + v
            for g in rows:
                if g["label"] in base and equity > 0:
                    g["share_of_equity"] = _f(g["value"] / equity)
                    if bench is not None:
                        g["benchmark"] = _f(bench.get(g["label"], 0.0))
                        g["active"] = _f(g["share_of_equity"] - g["benchmark"])
                if dim == "cycle" and g["label"] in SUPER_SECTOR_TIPS:
                    g["tip"] = SUPER_SECTOR_TIPS[g["label"]]
            if bench is not None:
                present = {g["label"] for g in rows}
                for label in base:
                    if label not in present and bench.get(label):
                        rows.append({"label": label, "value": 0.0, "weight": 0.0, "risk_share": None, "tickers": [],
                                     "share_of_equity": 0.0, "benchmark": _f(bench[label]), "active": _f(-bench[label]),
                                     **({"tip": SUPER_SECTOR_TIPS[label]} if dim == "cycle" else {})})
        out[dim] = rows
    return out


# ================================================================== winners, losers and currency


def _trades_for(row: dict[str, Any], hist: dict[str, Any] | None) -> tuple[list[tuple], list[tuple], str]:
    if hist and hist.get("trades"):
        return hist["trades"], hist.get("dividends") or [], "trades"
    if row.get("opened"):
        return [(row["opened"], float(row["shares"]), float(row["shares"]) * float(row["avg_cost"]))], [], "opened"
    return [], [], "held"


def _aligned(factor: pd.Series, index: pd.Index) -> pd.Series:
    f = pd.Series(factor, dtype=float).dropna()
    f.index = pd.DatetimeIndex(f.index).tz_localize(None) if pd.DatetimeIndex(f.index).tz is not None else pd.DatetimeIndex(f.index)
    return f.reindex(f.index.union(index)).ffill().reindex(index)


def _time_weighted(prices: pd.Series | None, start: pd.Timestamp, as_of: pd.Timestamp, shares_start: float,
                   after: list[tuple], dividends: list[tuple]) -> float | None:
    """Return of the holding over the part of the period it was held: price change (plus dividends received) chained
    between trades, from the period start or from the first purchase inside the period to ``as_of``.

    Unlike a money-weighted figure it cannot fall below -100%: money that came in late in the period is not
    divided by the little time it was exposed. None when a price is missing.
    """
    if prices is None:
        return None
    shares, since, growth = shares_start, (start if shares_start > 0 else None), 1.0
    marks = sorted({pd.Timestamp(t[0]) for t in after})
    moves = {d: sum(float(t[1]) for t in after if pd.Timestamp(t[0]) == d) for d in marks}
    for end in marks + [as_of]:
        if shares > 0 and since is not None and end > since:
            p_a, p_b = _asof(prices, since), _asof(prices, end)
            if p_a is None or p_b is None or p_a <= 0:
                return None
            per_share = sum(float(a) for d, a in dividends if since < pd.Timestamp(d) <= end) / shares
            growth *= (p_b + per_share) / p_a
        shares = max(shares + moves.get(end, 0.0), 0.0)
        since = end if shares > 1e-6 else None
        shares = shares if shares > 1e-6 else 0.0
    return growth - 1.0


def period_pnl(row: dict[str, Any], prices: pd.Series | None, hist: dict[str, Any] | None,
               start: pd.Timestamp, as_of: pd.Timestamp, div_factor: pd.Series | None = None) -> dict[str, Any] | None:
    """P&L of one holding from ``start`` to ``as_of`` in the portfolio currency (None when a price is missing).

    ``prices`` are the live closes, which are total-return adjusted: a past close is lowered by every later
    dividend, so the price change already contains them. Adding the dividends received on top would count them
    twice. With ``div_factor`` (split-adjusted close / dividend-adjusted close, ``data.price_history``) the
    closes are turned back into the prices that traded and the dividends actually received are added; without
    it they stay in the price and are reported, not added (shares bought inside the period then miss theirs).
    """
    trades, dividends, basis = _trades_for(row, hist)
    div = sum(float(a) for d, a in dividends if pd.Timestamp(d) > start)
    in_price = False
    if div and prices is not None:
        factor = _aligned(div_factor, prices.index) if div_factor is not None else None
        if factor is not None and pd.notna(_asof(factor, start)):
            prices = prices * factor
        else:
            in_price = True
    after = [t for t in trades if pd.Timestamp(t[0]) > start]
    shares_start = float(row["shares"]) - sum(float(t[1]) for t in after)
    shares_start = 0.0 if shares_start < 1e-6 else shares_start
    p0 = _asof(prices, start) if shares_start > 0 else 0.0
    if p0 is None:
        return None
    v0 = shares_start * p0
    flows: list[tuple[pd.Timestamp, float]] = []
    for d, delta, cash in after:
        day = pd.Timestamp(d)
        if cash is None:
            p = _asof(prices, day)
            if p is None:
                return None
            cash = float(delta) * p
        flows.append((day, float(cash)))
    v1 = float(row["value"])
    net = sum(c for _, c in flows)
    pnl = v1 - v0 - net + (0.0 if in_price else div)
    span = max((as_of - start).days, 1)
    denom = v0 + sum(c * max((as_of - d).days, 0) / span for d, c in flows)
    twr = _time_weighted(prices, start, as_of, shares_start, after, [] if in_price else dividends)
    return {"pnl": round(pnl, 2), "return": _f(twr), "return_mwr": _f(pnl / denom) if denom > 1.0 else None,
            "start_value": round(v0, 2), "net_invested": round(net, 2), "dividends": round(div, 2),
            "dividends_in_price": in_price, "basis": basis}


def currency_split(row: dict[str, Any], hist: dict[str, Any] | None, currency: str, fx: FxFn) -> dict[str, Any] | None:
    """Split a holding's P&L into the price move in its own currency and the currency move."""
    ccy = row.get("price_currency") or currency
    if major(ccy)[0] == major(currency)[0] or not row.get("fx"):
        return None
    if hist and hist.get("lots"):
        lots = [(d, float(c), float(s)) for d, c, s in hist["lots"]]
        basis = "lots"
    elif row.get("opened"):
        lots = [(row["opened"], float(row["shares"]) * float(row["avg_cost"]), float(row["shares"]))]
        basis = "opened"
    else:
        return None
    try:
        factors = np.asarray(fx(ccy, pd.DatetimeIndex([pd.Timestamp(d) for d, _, _ in lots])), dtype=float)
    except Exception:  # noqa: BLE001 - no rate history: no split
        return None
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        return None
    lot_cost = sum(c for _, c, _ in lots)
    cost = float(row["shares"]) * float(row["avg_cost"])
    if lot_cost <= 0 or cost <= 0:
        return None
    local_cost = sum(c / f for (_, c, _), f in zip(lots, factors)) * cost / lot_cost
    f_now = float(row["fx"])
    local_value = float(row["value"]) / f_now
    price_effect = (local_value - local_cost) * f_now
    fx_effect = local_cost * f_now - cost
    entry_rate = cost / local_cost
    return {"price_effect": round(price_effect, 2), "fx_effect": round(fx_effect, 2),
            "local_return": _f(local_value / local_cost - 1.0), "currency_move": _f(f_now / entry_rate - 1.0),
            "currency": major(ccy)[0], "basis": basis}


def _movers(rows: list[dict[str, Any]], frame: pd.DataFrame, histories: dict[str, dict[str, Any]],
            as_of: pd.Timestamp | None, currency: str, fx: FxFn,
            dividend_factors: dict[str, pd.Series] | None = None) -> dict[str, Any]:
    since = []
    for r in rows:
        hist = histories.get(r["ticker"])
        opened = pd.Timestamp(r["opened"]) if r.get("opened") else None
        div = sum(float(a) for d, a in (hist or {}).get("dividends", []) if opened is None or pd.Timestamp(d) >= opened)
        split = currency_split(r, hist, currency, fx)
        since.append({"ticker": r["ticker"], "name": r.get("name"), "pnl": r["pnl"], "pnl_pct": r["pnl_pct"],
                      "dividends": round(div, 2), "total_pnl": round(r["pnl"] + div, 2), "weight": r["weight"],
                      "days_held": r.get("days_held"), "currency_split": split})
    since.sort(key=lambda m: -m["total_pnl"])
    periods: dict[str, Any] = {}
    if as_of is not None:
        for key, months in PERIODS:
            if key not in MOVER_PERIODS:
                continue
            start = period_start(as_of, months)
            items, missing = [], []
            for r in rows:
                prices = frame[r["ticker"]] if r["ticker"] in frame.columns else None
                res = period_pnl(r, prices, histories.get(r["ticker"]), start, as_of,
                                 (dividend_factors or {}).get(r["ticker"]))
                if res is None:
                    missing.append(r["ticker"])
                    continue
                items.append({"ticker": r["ticker"], "name": r.get("name"), **res})
            items.sort(key=lambda m: -m["pnl"])
            periods[key] = {"start": start.strftime("%Y-%m-%d"), "items": items, "missing": missing,
                            "total": round(sum(m["pnl"] for m in items), 2)}
    fx_total = sum((m["currency_split"] or {}).get("fx_effect", 0.0) for m in since)
    price_total = sum((m["currency_split"] or {}).get("price_effect", 0.0) for m in since)
    split_n = sum(1 for m in since if m["currency_split"])
    return {
        "since_entry": since,
        "periods": periods,
        "totals": {"pnl": round(sum(m["pnl"] for m in since), 2), "dividends": round(sum(m["dividends"] for m in since), 2),
                   "total_pnl": round(sum(m["total_pnl"] for m in since), 2),
                   "fx_effect": round(fx_total, 2) if split_n else None,
                   "price_effect_foreign": round(price_total, 2) if split_n else None, "split_positions": split_n},
    }


# ================================================================== cycle


def _cycle(macro: dict[str, Any] | None, exposures: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    m = macro or {}
    indicators = []
    if m.get("regime"):
        indicators.append({"key": "regime", "label": "Market regime", "value": m["regime"],
                           "detail": f"score {m['regime_score']:+.0f}" if isinstance(m.get("regime_score"), (int, float)) else ""})
    vix = m.get("vix_level")
    if isinstance(vix, (int, float)) and vix == vix:
        reading = ("extreme fear: historically one of the best times to buy" if vix >= 40 else
                   "high fear" if vix >= 30 else "above its long-run average of 20" if vix >= 20 else "calm, below its long-run average of 20")
        indicators.append({"key": "vix", "label": "VIX", "value": f"{vix:.1f}", "detail": reading})
    if m.get("yield_curve_status"):
        spread = m.get("yield_spread_10y_2y")
        indicators.append({"key": "curve", "label": "Yield curve 10Y-2Y", "value": str(m["yield_curve_status"]).lower().replace("_", " "),
                           "detail": f"{spread:+.2f} pp" if isinstance(spread, (int, float)) and spread == spread else ""})
    if m.get("credit_trend"):
        indicators.append({"key": "credit", "label": "Credit (high yield vs investment grade)", "value": str(m["credit_trend"]).lower().replace("_", " "), "detail": ""})
    if m.get("breadth_trend"):
        indicators.append({"key": "breadth", "label": "Market breadth", "value": str(m["breadth_trend"]).lower().replace("_", " "), "detail": ""})
    spy, sma = m.get("spy_price"), m.get("spy_sma_200")
    if isinstance(spy, (int, float)) and isinstance(sma, (int, float)) and sma:
        indicators.append({"key": "trend", "label": "S&P 500 vs its 200-day average", "value": f"{spy / sma - 1.0:+.1%}",
                           "detail": "above: uptrend" if spy >= sma else "below: downtrend"})
    groups = [g for g in exposures.get("cycle", []) if g["label"] in SUPER_SECTOR_TIPS]
    notes = []
    eq = {g["label"]: g for g in groups}
    if eq:
        cyc = sum((eq.get(k) or {}).get("share_of_equity") or 0.0 for k in ("Cyclical", "Sensitive"))
        bench = [eq[k].get("benchmark") for k in ("Cyclical", "Sensitive") if k in eq and eq[k].get("benchmark") is not None]
        text = f"Cyclical and sensitive sectors are {cyc:.0%} of your equity holdings (funds looked through)"
        if bench:
            text += f", against {sum(bench):.0%} in the benchmark"
        notes.append(text + ". These earn most in expansions and fall most in recessions.")
    if risk.get("beta") is not None:
        notes.append(f"Beta {risk['beta']:.2f}: over the last two years, when the benchmark moved 1% in a week, "
                     f"today's mix moved {risk['beta']:.2f}% on average.")
    notes.append("Read this as context. FastValue has not tested sector rotation by regime, so the owner score, "
                 "not the cycle, decides what to buy or sell.")
    return {"indicators": indicators, "groups": groups, "notes": notes, "as_of": m.get("as_of")}


# ================================================================== alerts


def _alerts(rows: list[dict[str, Any]], risk: dict[str, Any], exposures: dict[str, Any], total: float, currency: str,
            invested: float, cap: float, profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Portfolio alerts, each with a fix and ``where`` the page shows it: "todo" (asks for something now) or "risk"."""
    alerts: list[dict[str, Any]] = []
    pos = risk.get("positions") or {}
    for r in sorted(rows, key=lambda r: -((pos.get(r["ticker"]) or {}).get("risk_share") or 0.0)):
        if r.get("class") == "core":  # a broad fund is meant to carry most of the risk; it is not a bet to resize
            continue
        rs = (pos.get(r["ticker"]) or {}).get("risk_share")
        share_inv = r["value"] / invested if invested > 0 else 0.0
        if rs is not None and rs >= RISK_SHARE_ALERT and share_inv > 0 and rs >= RISK_TO_WEIGHT_ALERT * share_inv:
            alerts.append({"kind": "risk", "where": "todo", "ticker": r["ticker"], "title": f"{r['ticker']} drives the swings",
                           "text": f"{r['ticker']} is {share_inv:.0%} of the invested value but {rs:.0%} of the portfolio's risk "
                                   "(its swings and how they line up with the rest).",
                           "fix": "If that is not intended, size it by risk: hold less of it, or add holdings that move "
                                  "differently (Risk tab, correlation). The owner score still decides whether to own it."})
    corr = risk.get("correlation") or {}
    funds = {t for t, prof in profiles.items() if prof.get("asset_type") in ("ETF", "Fund")}
    for p in corr.get("high_pairs", [])[:3]:
        a, b = p["pair"]
        # A broad fund tracks the stocks it holds; that is not a doubled bet.
        if p["weight"] is not None and p["weight"] > cap and not ({a, b} & funds):
            alerts.append({"kind": "risk", "where": "todo", "ticker": None, "title": f"{a} and {b} move as one",
                           "text": f"{a} and {b} have a weekly correlation of {p['value']:.2f} over two years, and together "
                                   f"they are {p['weight']:.0%} of the portfolio, above the {cap:.0%} single-position cap.",
                           "fix": f"Treat them as one position when you size: keep the pair under {cap:.0%} together, "
                                  "for example by sending new money elsewhere."})
    foreign = [g for g in exposures.get("currency", []) if g["label"] != major(currency)[0]]
    foreign_value = sum(g["value"] for g in foreign)
    if total > 0 and foreign_value / total >= FOREIGN_CURRENCY_ALERT:
        top = max(foreign, key=lambda g: g["value"])
        alerts.append({"kind": "info", "where": "risk", "ticker": None, "title": "Currency exposure",
                       "text": f"{foreign_value / total:.0%} of the portfolio is priced in other currencies than {currency} "
                               f"({top['label']} {top['weight']:.0%}). A {CURRENCY_SHOCK:.0%} fall of those currencies against "
                               f"{currency} would cost about {amount_text(foreign_value * CURRENCY_SHOCK, currency)}.",
                       "fix": "Normal for owning foreign companies, and it evens out over long periods. If you want less of "
                              "it, prefer currency-hedged share classes for new money. Movers shows how much of "
                              "your P&L came from currencies."})
    for g in exposures.get("sector", []):
        if g.get("active") is not None and g["active"] >= SECTOR_ACTIVE_ALERT:
            alerts.append({"kind": "info", "where": "risk", "ticker": None, "title": f"{g['label']} overweight",
                           "text": f"{g['label']} is {g['share_of_equity']:.0%} of your equity holdings (funds looked through), "
                                   f"against {g['benchmark']:.0%} in the benchmark.",
                           "fix": "Fine if it is your conviction. To lower it without selling, send new money to other "
                                  "sectors; Risk > Exposure shows where you are light."})
    if (risk.get("correlation_to_benchmark") or 0) >= BENCHMARK_LIKE_CORRELATION and risk.get("beta") is not None \
            and abs(risk["beta"] - 1.0) <= BENCHMARK_LIKE_BETA_BAND:
        alerts.append({"kind": "info", "where": "risk", "ticker": None, "title": "Moves like the index",
                       "text": f"Today's mix moved almost exactly with the benchmark (correlation {risk['correlation_to_benchmark']:.2f}, "
                               f"beta {risk['beta']:.2f}) over the last two years.",
                       "fix": "Check in the Backtest tab whether your picks beat a plain index fund after costs; if not, "
                              "an index fund gives the same risk for less work."})
    if risk.get("excluded"):
        alerts.append({"kind": "info", "where": "risk", "ticker": None, "title": "Short price history",
                       "text": f"{', '.join(risk['excluded'])}: less than {MIN_WEEKS} weeks of prices, so left out of the "
                               "risk, beta and correlation figures (counted in values and weights).",
                       "fix": "Nothing to do: it enters the risk figures once it has half a year of prices."})
    return alerts


# ================================================================== candidate fit


def candidate_fit(ticker: str, holdings: list[dict[str, Any]], closes: dict[str, pd.Series], cash: float,
                  currency: str, profiles: dict[str, dict[str, Any]], fx: FxFn, add_value: float | None,
                  candidate_currency: str | None = None) -> dict[str, Any]:
    """How a ticker you do not hold would sit in the portfolio: correlation, overlap and risk before and after.

    ``holdings`` are dicts with ``ticker``, ``value`` (portfolio currency) and ``price_currency``; ``closes``
    includes the candidate. ``add_value`` is the suggested position, paid from cash (None: the new-position cap).
    """
    total = sum(h["value"] for h in holdings) + cash
    currencies = {h["ticker"]: h.get("price_currency") or currency for h in holdings}
    currencies[ticker] = candidate_currency or currency
    frame, _ = price_frame(closes, currencies, currency, fx)
    out: dict[str, Any] = {"ticker": ticker, "basis": f"weekly returns in {currency}, last {RISK_WEEKS // 52} years"}
    prof = profiles.get(ticker) or {}
    out["profile"] = {k: prof.get(k) for k in ("name", "asset_type", "sector", "industry", "country", "region")}
    if total <= 0 or frame.empty or ticker not in frame.columns:
        return out
    weekly = _weekly_returns(frame).tail(RISK_WEEKS)
    counts = weekly.notna().sum()
    if counts.get(ticker, 0) < MIN_WEEKS:
        out["note"] = f"Less than {MIN_WEEKS} weeks of prices: no correlation figures yet."
        return out
    held = [h["ticker"] for h in holdings if h["ticker"] in weekly.columns and counts[h["ticker"]] >= MIN_WEEKS]
    w = pd.Series({h["ticker"]: h["value"] / total for h in holdings if h["ticker"] in held}, dtype=float)
    Rf = weekly[held + [ticker]].fillna(0.0)
    if held:
        mix = Rf[held].to_numpy() @ w.to_numpy()
        ok = weekly[ticker].notna().to_numpy()
        out["correlation_to_portfolio"] = _f(np.corrcoef(weekly[ticker].to_numpy()[ok], mix[ok])[0, 1], 2) if mix[ok].std() > 0 else None
        corr = weekly[held + [ticker]].corr(min_periods=MIN_WEEKS)[ticker].drop(ticker).dropna().sort_values(ascending=False)
        out["most_similar"] = [{"ticker": t, "correlation": _f(v, 2)} for t, v in corr.head(3).items()]
    add = add_value if add_value is not None else 0.10 * total
    add = max(0.0, min(add, total))
    cov = Rf.cov().to_numpy() * 52.0
    before = np.append(w.to_numpy(), 0.0) if held else np.array([0.0])
    after = before.copy()
    after[-1] = add / total
    vol_before = math.sqrt(max(float(before @ cov @ before), 0.0))
    var_after = float(after @ cov @ after)
    out.update({
        "add_value": round(add, 2), "add_weight": _f(add / total),
        "volatility_before": _f(vol_before), "volatility_after": _f(math.sqrt(max(var_after, 0.0))),
        "risk_share_after": _f(after[-1] * (cov @ after)[-1] / var_after) if var_after > 0 else None,
        "own_volatility": _f(weekly[ticker].std() * math.sqrt(52.0)),
    })
    # Overlap with what you hold: the same sector, country and currency.
    def share(key: str, value: Any) -> float | None:
        if not value:
            return None
        return _f(sum(h["value"] for h in holdings if (profiles.get(h["ticker"]) or {}).get(key) == value) / total)
    out["overlap"] = {"sector": share("sector", prof.get("sector")), "country": share("country", prof.get("country")),
                      "currency": _f(sum(h["value"] for h in holdings if major(h.get("price_currency") or currency)[0]
                                         == major(currencies[ticker])[0]) / total)}
    return out


# ================================================================== entry point


def analyze(
    rows: list[dict[str, Any]],
    closes: dict[str, pd.Series],
    cash: float,
    currency: str,
    profiles: dict[str, dict[str, Any]],
    fx: FxFn,
    total: float | None = None,
    unpriced: list[dict[str, Any]] | None = None,
    benchmark: dict[str, Any] | None = None,
    histories: dict[str, dict[str, Any]] | None = None,
    macro: dict[str, Any] | None = None,
    cap: float = 0.20,
    dividend_factors: dict[str, pd.Series] | None = None,
    benchmarks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Allocation, risk, correlation, performance, winners and losers and cycle context of reviewed holdings.

    ``rows`` are review rows (value, weight, shares, avg_cost, price_currency, fx, opened, pnl ...);
    ``unpriced`` are positions valued at cost (no market data); ``histories`` maps tickers to the
    trades, dividends and open lots of imported holdings; ``benchmark`` has ``ticker``, ``label``,
    ``close`` (listing currency), ``currency`` and optionally ``sectors``. ``benchmarks``: the indices of the
    performance chart, primary first, each with ``id``, ``ticker``, ``label``, ``close`` (None when not fetched)
    and ``currency``; ``benchmark`` (the primary, with ``sectors``) still drives risk and exposures.
    """
    unpriced = unpriced or []
    histories = histories or {}
    invested = sum(r["value"] for r in rows) + sum(u["value"] for u in unpriced)
    total = total if total is not None else invested + cash
    notes: list[str] = []
    frame, fx_notes = price_frame(closes, {r["ticker"]: r.get("price_currency") or currency for r in rows}, currency, fx)
    notes += fx_notes
    w = pd.Series({r["ticker"]: r["value"] / total for r in rows if total > 0}, dtype=float)

    bench = None
    bench_info = None
    if benchmark and benchmark.get("close") is not None and not frame.empty:
        b_frame, b_notes = price_frame({"_b": benchmark["close"]}, {"_b": benchmark.get("currency") or currency}, currency, fx)
        notes += b_notes
        if not b_frame.empty:
            bench = _on_calendar(b_frame["_b"], frame.index)
            bench_info = {"ticker": benchmark.get("ticker"), "label": benchmark.get("label")}

    others: dict[str, pd.Series] = {}
    priced: list[dict[str, Any]] = []
    for entry in benchmarks or []:
        if entry.get("close") is None:
            notes.append(f"{entry['label']} ({entry['ticker']}): no prices today; left out of the performance chart.")
            continue
        if frame.empty:
            continue
        e_frame, e_notes = price_frame({"_b": entry["close"]}, {"_b": entry.get("currency") or currency}, currency, fx)
        notes += e_notes
        if e_frame.empty:
            continue
        others[entry["id"]] = _on_calendar(e_frame["_b"], frame.index)
        priced.append({"id": entry["id"], "ticker": entry["ticker"], "label": entry["label"]})

    performance = _performance(frame, w, bench, others) if not frame.empty and len(w) else {"series": {}, "periods": [], "as_of": None}
    risk = _risk(frame, w, bench) if not frame.empty and len(w) else {"volatility": None, "positions": {}, "excluded": [], "correlation": None}
    mix = _mix_returns(frame, w)[0] if not frame.empty and len(w) else pd.Series(dtype=float)
    risk.update(_drawdowns(mix, total))

    items = [{"ticker": r["ticker"], "label": r["ticker"], "value": r["value"], "price_currency": r.get("price_currency") or currency,
              "profile": profiles.get(r["ticker"]) or {}} for r in rows]
    items += [{"ticker": u["ticker"], "label": u["ticker"], "value": u["value"], "price_currency": u.get("price_currency") or currency,
               "profile": profiles.get(u["ticker"]) or {}} for u in unpriced]
    risk_share = {t: p["risk_share"] for t, p in (risk.get("positions") or {}).items() if p.get("risk_share") is not None}
    exposures = _exposures(items, cash, total, currency, risk_share, (benchmark or {}).get("sectors"))

    as_of = pd.Timestamp(performance["as_of"]) if performance.get("as_of") else None
    movers = _movers(rows, frame, histories, as_of, currency, fx, dividend_factors)
    alerts = _alerts(rows, risk, exposures, total, currency, invested, cap, profiles)
    profile_rows = {t: {k: p.get(k) for k in ("name", "asset_type", "sector", "industry", "country", "region", "source",
                                                      "expense_ratio")}
                    for t, p in profiles.items()}
    return {
        "currency": currency,
        "as_of": performance.get("as_of"),
        "benchmark": bench_info,
        "benchmarks": priced,
        "total_value": round(total, 2),
        "invested_value": round(invested, 2),
        "performance": performance,
        "risk": risk,
        "exposures": exposures,
        "movers": movers,
        "cycle": _cycle(macro, exposures, risk),
        "alerts": alerts,
        "profiles": profile_rows,
        "notes": notes,
        "method": {
            "mix": "Today's weights held constant through history; cash earns nothing.",
            "risk": f"Weekly returns in {currency} over the last {RISK_WEEKS // 52} years; a holding needs {MIN_WEEKS} weeks of prices.",
            "periods": "Value now minus value at the start, minus money put in, plus dividends. Imported holdings use their actual trades; "
                       "prices are the ones that traded, so dividends are counted once.",
        },
    }
