"""Point-in-time test of the value agents (Fundamentals, Shareholders) on the S&P 500.

Each month-end from 2011 on:

1. The universe is the S&P 500 as it was on that date (historical membership,
   not today's list), limited to members with SEC XBRL filings and Yahoo prices.
2. Each company's annual statements are rebuilt from the 10-K facts filed on or
   before that date (``data.sec_edgar``), priced with the share price and share
   count of that date (splits undone, ``data.price_history``).
3. The unchanged Fundamentals and Shareholder agents score it; their composite
   uses the same scoring engine as live analysis.
4. Every metric of the owner's framework is stored too, so each can be tested
   on its own (information coefficient and quintile spreads).

Portfolios are equal weight, rebalanced monthly (or yearly each April, after
most 10-Ks are out), with 10 bps per trade. Benchmarks are SPY and the equal
weight covered universe, which isolates selection skill from the universe.

Run: ``SEC_USER_AGENT="app you@example.com" python -m core.value_study``.
The report is written to analysis/output/value_study.json.
"""

from __future__ import annotations

import argparse
import warnings
import io
import json
import logging
import math
import os
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from agents.base import UNAVAILABLE, AgentAssessment, signal_from_score
from agents.fundamental_agent import FundamentalAgent
from agents.insider_agent import InsiderAgent
from agents.scoring import aggregate
from agents.shareholder_agent import ShareholderAgent
from analysis.fundamental_metrics import FCF_DEFINITIONS, _roic_for_year, compute_owner_metrics, is_excluded_sector
from config.weights import load_scoring_config
from data.fundamentals import FundamentalsData
from data.insider_trading import analyze_insider_transactions
from data.price_history import as_traded, fetch_price_history, split_events
from data.sec_edgar import SecConfigError, SecDataError, load_company_facts, load_ticker_map
from data.sec_insider import load_insider_history, to_yahoo_frame

logger = logging.getLogger(__name__)

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
PRICE_START = "2008-06-01"
STUDY_START = pd.Timestamp("2011-01-31")
COST_BPS = 10.0
LIQUIDITY_BARS = 30
INSIDER_HISTORY_YEARS = 3  # rows passed to the Insider agent (its window is 180 days; older rows feed track records)
INSIDER_PRICE_BARS = 800
# Same business under an older CIK before a holding-company reorganization (verified on EDGAR).
PREDECESSOR_CIKS: dict[str, tuple[int, ...]] = {
    "GOOGL": (1288776,),  # Google Inc. before Alphabet (2015)
    "GOOG": (1288776,),
    "DIS": (1001039,),  # TWDC Enterprises 18 (the pre-2019 Walt Disney Co)
}
# Exxon Mobil Corp keeps filing under its original CIK; the ticker map points to a new shell.
CIK_OVERRIDES: dict[str, int] = {"XOM": 34088}

SECTOR_SIC: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("Energy", ((1300, 1399), (2910, 2999), (4610, 4619))),
    ("Utilities", ((4900, 4999),)),
    ("Communication", ((4800, 4899), (2710, 2799), (7810, 7819))),
    ("Real estate", ((6500, 6599), (6798, 6798))),
    ("Financials", ((6000, 6499), (6700, 6797), (6799, 6799))),
    ("Health care", ((2830, 2836), (3840, 3851), (5047, 5047), (5122, 5122), (8000, 8099), (8730, 8734))),
    ("Technology", ((3570, 3579), (3600, 3699), (3820, 3829), (7370, 7379))),
    ("Consumer staples", ((2000, 2199), (2840, 2844), (5140, 5149), (5400, 5499), (5912, 5912))),
    ("Materials", ((1000, 1299), (1400, 1499), (2600, 2699), (2800, 2829), (2850, 2899), (3200, 3399))),
    ("Consumer discretionary", ((2300, 2399), (2500, 2599), (3710, 3716), (3940, 3949), (5000, 5999), (7000, 7099), (7900, 7999))),
    ("Industrials", ((1500, 1799), (3400, 3569), (3580, 3599), (3700, 3799), (3800, 3819), (4000, 4799), (7300, 7369), (7380, 8799))),
)


def sector_for_sic(sic: int | None) -> str:
    if sic is None:
        return "Other"
    for name, ranges in SECTOR_SIC:
        if any(lo <= sic <= hi for lo, hi in ranges):
            return name
    return "Other"


# ------------------------------------------------------------------ inputs


def _read_url(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fastvalue"}), timeout=60) as r:
        return r.read()


def load_membership() -> pd.Series:
    """Date -> frozenset of S&P 500 tickers (Yahoo style), from a public constituents history."""
    path = settings.DATA_DIR / "universe" / "sp500_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or (datetime.now().timestamp() - path.stat().st_mtime) > 7 * 86400:
        path.write_bytes(_read_url(MEMBERSHIP_URL))
    df = pd.read_csv(path)
    out = {pd.Timestamp(d): frozenset(t.strip().upper().replace(".", "-") for t in str(s).split(",") if t.strip())
           for d, s in zip(df["date"], df["tickers"])}
    return pd.Series(out).sort_index()


def members_asof(membership: pd.Series, date: pd.Timestamp) -> frozenset[str]:
    i = membership.index.searchsorted(date, side="right") - 1
    return membership.iloc[i] if i >= 0 else frozenset()


def load_fred(series: str, start: str) -> pd.Series:
    df = pd.read_csv(io.BytesIO(_read_url(FRED_URL.format(series=series, start=start))))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]))
    return s.dropna()


def month_ends(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="ME"))


# ------------------------------------------------------------------ scoring


def _product(splits: pd.Series, after: pd.Timestamp, until: pd.Timestamp) -> float:
    s = splits[(splits.index > after) & (splits.index <= until)]
    return float(s.prod()) if len(s) else 1.0


def _score_ticker(task: tuple) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score one ticker at each of its month-ends (runs in a worker process).

    An optional sixth task item is the SEC cache age limit in days (``math.inf``
    reuses cached filings without contacting the SEC, for runs without
    ``SEC_USER_AGENT``).
    """
    ticker, cik, predecessors, dates, treasury, *options = task
    stats = {"no_price_on_date": 0, "no_10k_yet": 0, "unavailable": 0}
    prices = fetch_price_history([ticker], PRICE_START).get(ticker)
    if prices is None or prices.empty:
        return [], stats
    traded = as_traded(prices)[["open", "high", "low", "close", "volume"]]
    splits = split_events(prices)
    try:
        age = {"max_age_days": options[0]} if options else {}
        cf = load_company_facts(ticker, cik, predecessors, splits=splits, **age)
    except SecConfigError:
        stats["no_sec_access"] = 1
        return [], stats
    except SecDataError:
        return [], stats
    fund_agent, sh_agent = FundamentalAgent(), ShareholderAgent()
    sector = sector_for_sic(cf.sic)
    config = load_scoring_config()
    scales = {"fundamentals": FundamentalAgent.score_scale, "shareholder": ShareholderAgent.score_scale}
    persistence_cache: dict[pd.Timestamp, tuple[float | None, int]] = {}
    rows: list[dict[str, Any]] = []

    for date in dates:
        i = int(prices.index.searchsorted(date, side="right")) - 1
        if i < LIQUIDITY_BARS or (date - prices.index[i]).days > 7:
            stats["no_price_on_date"] += 1
            continue
        window = traded.iloc[i - LIQUIDITY_BARS + 1: i + 1]
        price = float(window["close"].iloc[-1])
        st = cf.statements_asof(date, max_years=4)
        if not st.available:
            stats["no_10k_yet"] += 1
            continue

        years = st.years()
        shares, shares_source = None, None
        cover = cf.shares_outstanding_asof(date)
        if cover:
            shares = cover[0] * _product(splits, cover[1], date)
            shares_source = "cover page"
        # Already on today's share basis. A cover count far below it usually lists one share class only.
        diluted = st.value("income", ("Diluted Average Shares", "Basic Average Shares"))
        if diluted and (shares is None or shares / diluted < 0.85):
            shares, shares_source = diluted, "diluted average (cover page missing or one class only)"

        data = FundamentalsData(
            ticker=ticker, available=True, quote_type="EQUITY", short_name=cf.name,
            sector="Financial Services" if cf.is_financial else None, industry=cf.sic_description,
            currency="USD", financial_currency="USD", current_price=price, shares_outstanding=shares,
            treasury_10y=treasury.get(date),
        )
        m = compute_owner_metrics(st, data, price, window, today=date)
        fund = fund_agent.evaluate_with_data(data, st, window, asof=date)
        sh = sh_agent.evaluate_with_data(data, st, window, asof=date)
        if fund.data_quality == UNAVAILABLE and sh.data_quality == UNAVAILABLE:
            stats["unavailable"] += 1
            continue
        # Owner score: owner weights and thresholds, independent of the regime (agents/scoring.py).
        composite = aggregate([fund, sh], config, None, scales, weight_overrides=config.owner_weights,
                              thresholds=config.owner_thresholds).composite_score

        # ROIC persistence over up to 10 fiscal years (the live agent sees about 4).
        long_key = st.years()[0] if years else date
        if long_key not in persistence_cache:
            long_st = cf.statements_asof(date, max_years=10)
            history = [r for r in (_roic_for_year(long_st, y).roic for y in long_st.years()) if r is not None]
            persistence_cache[long_key] = (
                sum(r >= 0.15 for r in history) / len(history) if len(history) >= 5 else None, len(history))
        long_share, long_n = persistence_cache[long_key]

        history = [y.roic for y in m.years if y.roic is not None]
        components = fund.raw_data.get("components", {}) if fund.data_quality != UNAVAILABLE else {}
        rows.append({
            "date": date, "ticker": ticker, "sector": sector, "is_financial": m.is_financial,
            "fy_end": m.fiscal_year_end, "price": price, "shares": shares, "shares_source": shares_source,
            "market_cap": m.market_cap, "enterprise_value": m.enterprise_value,
            "fund_score": fund.score if fund.data_quality != UNAVAILABLE else np.nan,
            "fund_conf": fund.confidence, "fund_quality": fund.data_quality,
            "sh_score": sh.score if sh.data_quality != UNAVAILABLE else np.nan, "sh_conf": sh.confidence,
            "sh_quality": sh.data_quality,
            "composite": composite,
            **{f"comp_{k}": v for k, v in components.items()},
            "ev_to_fcf": m.ev_to_fcf, "fcf_ev_yield": m.fcf_yield_on_ev,
            "fcf": m.fcf, "reported_fcf": m.reported_fcf, "fcf_owner": m.fcf_owner, "fcf_all_capex": m.fcf_all_capex,
            "fcf_definition": m.fcf_definition, "ev_to_ebit": m.ev_to_ebit,
            "net_dilution_cost_m": m.net_dilution_cost, "asset_growth": m.asset_growth,
            "incremental_roic": m.incremental_roic, "delta_wc_years": m.delta_working_capital_years,
            "per": m.per, "per_ex_cash": m.per_ex_cash,
            "earnings_yield": (m.net_income / m.market_cap) if m.net_income is not None and m.market_cap else None,
            "ev_to_mc": m.ev_to_market_cap, "roic": m.roic,
            "roic_high_share": (sum(r >= 0.15 for r in history) / len(history)) if len(history) >= 3 else None,
            "roic_high_share_10y": long_share, "roic_years_10y": long_n,
            "reinvestment": m.reinvestment_rate, "roe": m.roe, "roa": m.roa,
            "growth_capex": m.growth_capex, "acquisitions": m.acquisitions, "sbc": m.sbc, "buybacks": m.buybacks,
            "revenue": m.revenue, "net_income": m.net_income, "missing": "; ".join(m.missing),
            "share_cagr": m.share_cagr, "sbc_to_revenue": m.sbc_to_revenue, "buyback_yield": m.buyback_yield,
            "dilutive_gap": m.dilutive_gap, "adv": m.avg_dollar_volume,
        })
    return rows, stats


def _score_insider(task: tuple) -> list[dict[str, Any]]:
    """Insider agent at each month-end, from Form 4 filings public on that date (worker process)."""
    ticker, dates, rows = task
    prices = fetch_price_history([ticker], PRICE_START).get(ticker)
    if prices is None or prices.empty:
        return []
    traded = as_traded(prices)[["open", "high", "low", "close", "volume"]]
    agent = InsiderAgent()
    out: list[dict[str, Any]] = []
    for date in dates:
        i = int(prices.index.searchsorted(date, side="right")) - 1
        if i < 60 or (date - prices.index[i]).days > 7:
            continue
        known = rows[(rows["filed"] <= date) & (rows["trade_date"] > date - pd.DateOffset(years=INSIDER_HISTORY_YEARS))]
        if known.empty:
            out.append({"date": date, "ticker": ticker, "ins_score": np.nan, "ins_conf": 0.0, "ins_quality": UNAVAILABLE})
            continue
        data = analyze_insider_transactions(to_yahoo_frame(known), traded.iloc[max(0, i - INSIDER_PRICE_BARS + 1): i + 1], ticker)
        a = agent.evaluate_with_data(data)
        ok = a.data_quality != UNAVAILABLE
        out.append({
            "date": date, "ticker": ticker, "ins_score": a.score if ok else np.nan, "ins_conf": a.confidence,
            "ins_quality": a.data_quality, "ins_buys": data.total_buy_transactions if ok else 0,
            "ins_buyers": data.distinct_buyers if ok else 0, "ins_net_buy": data.net_buy_value if ok else 0.0,
            "ins_cluster": bool(data.cluster_buy_detected) if ok else False,
        })
    return out


def build_insider_panel(prep: dict[str, Any], workers: int) -> pd.DataFrame:
    ciks = prep["ciks"]
    history = load_insider_history(set(ciks.values()), STUDY_START - pd.DateOffset(years=INSIDER_HISTORY_YEARS),
                                   prep["dates"][-1])
    print(f"Insider filings: {len(history):,} non-derivative Form 4/5 rows for {history['issuer_cik'].nunique()} issuers.")
    by_cik = {c: g for c, g in history.groupby("issuer_cik")}
    empty = history.iloc[0:0]
    tasks = []
    for t, cik in ciks.items():
        member_dates = [d for d in prep["dates"] if t in prep["per_date"][d]]
        if member_dates:
            tasks.append((t, member_dates, by_cik.get(cik, empty)))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(_score_insider, task) for task in tasks]), 1):
            rows.extend(fut.result())
            if n % 100 == 0:
                print(f"  insider scored {n}/{len(tasks)} tickers")
    return pd.DataFrame(rows)


def _assessment(key: str, score: float, confidence: float, quality: str) -> AgentAssessment:
    return AgentAssessment(agent_name=key, score=float(score) if pd.notna(score) else 0.0,
                           signal=signal_from_score(float(score) if pd.notna(score) else 0.0), confidence=float(confidence or 0.0),
                           summary="", agent_key=key, data_quality=quality if pd.notna(score) else UNAVAILABLE)


def add_insider_composite(panel: pd.DataFrame) -> pd.DataFrame:
    """Owner composite with the Insider agent given a 20% share (it has 0 weight in the live owner score)."""
    config = load_scoring_config()
    scales = {"fundamentals": FundamentalAgent.score_scale, "shareholder": ShareholderAgent.score_scale,
              "insider": InsiderAgent.score_scale}
    weights = {"fundamentals": 0.8 * config.owner_weights.get("fundamentals", 0.45),
               "shareholder": 0.8 * config.owner_weights.get("shareholder", 0.55), "insider": 0.2}
    values = []
    sh_conf = panel["sh_conf"] if "sh_conf" in panel else pd.Series(60.0, index=panel.index)
    for fund, fconf, fq, sh, sconf, shq, ins, iconf, iq in zip(
            panel["fund_score"], panel["fund_conf"], panel["fund_quality"], panel["sh_score"], sh_conf,
            panel["sh_quality"], panel["ins_score"], panel["ins_conf"], panel["ins_quality"]):
        assessments = [_assessment("fundamentals", fund, fconf, fq), _assessment("shareholder", sh, sconf, shq),
                       _assessment("insider", ins, iconf, iq if isinstance(iq, str) else UNAVAILABLE)]
        values.append(aggregate(assessments, config, None, scales, weight_overrides=weights,
                                thresholds=config.owner_thresholds).composite_score)
    panel = panel.copy()
    panel["composite_ins"] = values
    return panel


def apply_framework_exclusions(panel: pd.DataFrame) -> pd.DataFrame:
    """Utilities and REITs are NOT_RATED (decision_log.md T-12): their owner composite is blanked so they are
    never picked or ranked, exactly as the live tool treats them. Adds ``framework_applicable``."""
    panel = panel.copy()
    excluded = panel["sector"].map(is_excluded_sector).astype(bool)
    panel["framework_applicable"] = ~excluded
    for col in ("composite", "composite_ins"):
        if col in panel:
            panel.loc[excluded, col] = np.nan
    return panel


# ------------------------------------------------------------------ returns


def month_end_panel(prices: dict[str, pd.DataFrame], dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Total-return price (dividend adjusted) at each month-end: last close on or before it."""
    idx = pd.DatetimeIndex(dates)
    cols = {}
    for t, df in prices.items():
        s = df["adj_close"].dropna()
        if s.empty:
            continue
        pos = s.index.searchsorted(idx, side="right") - 1
        vals = np.where(pos >= 0, s.values[np.clip(pos, 0, None)], np.nan)
        # No price after the stock stopped trading: carry the last one (the position sits in cash).
        cols[t] = vals
    return pd.DataFrame(cols, index=idx)


def forward_returns(level: pd.DataFrame, months: int) -> pd.DataFrame:
    return level.shift(-months) / level - 1.0


def performance(monthly: pd.Series, rf: pd.Series) -> dict[str, float]:
    r = monthly.dropna()
    rf = rf.reindex(r.index).fillna(0.0)
    eq = (1 + r).cumprod()
    years = len(r) / 12
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    vol = float(r.std() * math.sqrt(12))
    sharpe = float((r - rf).mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    return {"cagr": cagr, "volatility": vol, "sharpe": sharpe, "max_drawdown": dd, "total_return": float(eq.iloc[-1] - 1)}


def newey_west_t(x: np.ndarray, lags: int) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(var / n)) if var > 0 else float("nan")


def portfolio_returns(
    panel: pd.DataFrame,
    fwd1: pd.DataFrame,
    column: str,
    pick: str,
    rebalance_months: tuple[int, ...] | None = None,
    ascending: bool = False,
    cost_bps: float = COST_BPS,
    mask: pd.Series | None = None,
    keep_if: str | None = None,
) -> tuple[pd.Series, float]:
    """Monthly returns of an equal-weight portfolio picked by ``column``.

    ``pick``: "top_q" (top 20%), "bottom_q", "top_30" or "all". With
    ``rebalance_months`` (for example (4,)), holdings are refreshed only in
    those months and kept otherwise. ``keep_if`` names a column that must be
    positive for a picked stock to be held (for example a momentum filter).
    """
    out: dict[pd.Timestamp, float] = {}
    turnovers: list[float] = []
    holdings: set[str] = set()
    for date, group in panel.groupby("date"):
        if mask is not None:
            group = group[mask.loc[group.index]]
        valid = group.dropna(subset=[column])
        if len(valid) < 50:
            continue
        refresh = not holdings or rebalance_months is None or date.month in rebalance_months
        if refresh:
            ranked = valid.sort_values(column, ascending=ascending)["ticker"].tolist()
            q = max(1, len(ranked) // 5)
            new = set(ranked[:q] if pick == "top_q" else ranked[-q:] if pick == "bottom_q"
                      else ranked[:30] if pick == "top_30" else ranked)
            if keep_if is not None:
                passing = set(valid.loc[valid[keep_if] > 0, "ticker"])
                new = (new & passing) or new
            turnover = len(new ^ holdings) / (2 * max(len(new), 1)) if holdings else 1.0
            holdings = new
        else:
            holdings = {t for t in holdings if t in set(group["ticker"])}
            turnover = 0.0
        r = fwd1.loc[date, sorted(holdings)] if date in fwd1.index else pd.Series(dtype=float)
        r = r.dropna()
        if r.empty:
            continue
        out[date] = float(r.mean()) - turnover * 2 * cost_bps / 10_000
        turnovers.append(turnover)
    s = pd.Series(out).sort_index()
    s.index = s.index + pd.offsets.MonthEnd(1)  # return is earned over the following month
    return s, float(np.mean(turnovers)) if turnovers else float("nan")


def factor_test(panel: pd.DataFrame, fwd1: pd.DataFrame, fwd12: pd.DataFrame, column: str, higher_is_better: bool,
                subset: pd.Series | None = None, by_sector: bool = False) -> dict[str, Any] | None:
    """Monthly rank IC (1 and 12 months ahead) and top-minus-bottom quintile return."""
    frame = panel if subset is None else panel[subset]
    frame = frame.dropna(subset=[column])
    sign = 1.0 if higher_is_better else -1.0
    ic1, ic12, spread, dates = [], [], [], []
    for date, g in frame.groupby("date"):
        if len(g) < 50 or date not in fwd1.index:
            continue
        x = sign * g.set_index("ticker")[column].astype(float)
        if by_sector:
            x = x.groupby(g.set_index("ticker")["sector"]).rank(pct=True)
        r1 = fwd1.loc[date].reindex(x.index)
        r12 = fwd12.loc[date].reindex(x.index)
        ok = r1.notna()
        if ok.sum() < 50:
            continue
        ic1.append(x[ok].rank().corr(r1[ok].rank()))
        ok12 = r12.notna()
        ic12.append(x[ok12].rank().corr(r12[ok12].rank()) if ok12.sum() >= 50 else np.nan)
        xs, rs = x[ok], r1[ok]
        hi, lo = xs.quantile(0.8), xs.quantile(0.2)
        top, bottom = (rs[xs >= hi], rs[xs <= lo]) if hi > lo else (rs[xs > lo], rs[xs <= lo])
        spread.append(top.mean() - bottom.mean() if len(top) and len(bottom) else np.nan)
        dates.append(date)
    if len(ic1) < 24:
        return None
    ic1, ic12, spread = np.array(ic1), np.array(ic12), np.array(spread)
    half = len(spread) // 2
    ann = lambda a: float((1 + np.nanmean(a)) ** 12 - 1)  # noqa: E731
    return {
        "months": len(ic1),
        "avg_names": int(frame.groupby("date").size().mean()),
        "ic_1m": float(np.nanmean(ic1)), "ic_1m_t": newey_west_t(ic1, 3),
        "ic_12m": float(np.nanmean(ic12)), "ic_12m_t": newey_west_t(ic12, 12),
        "q5_minus_q1_annual": ann(spread),
        "q5_minus_q1_first_half": ann(spread[:half]), "q5_minus_q1_second_half": ann(spread[half:]),
        "hit_rate": float((spread > 0).mean()),
    }


# ------------------------------------------------------------------ VIX rule


def vix_study(threshold_levels: tuple[float, ...] = (30.0, 40.0, 50.0), gap_sessions: int = 60) -> dict[str, Any]:
    """SPY total return after the VIX first closes above a level (owner's rule: above 40 is a buying chance).

    An event is the first close above the level after ``gap_sessions`` sessions
    without one, so one panic counts once.
    """
    from data.fetcher import fetch_ohlcv

    vix = load_fred("VIXCLS", "1990-01-01")
    spy = fetch_ohlcv(["SPY"], start="1993-02-01")["SPY"]["close"]
    vix = vix.reindex(spy.index).ffill()
    horizons = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
    fwd = {k: spy.shift(-h) / spy - 1 for k, h in horizons.items()}
    dd = {k: (spy[::-1].rolling(h, min_periods=1).min()[::-1] / spy - 1) for k, h in (("worst_3m", 63),)}
    out: dict[str, Any] = {"all_days": {k: float(v.mean()) for k, v in fwd.items()}}
    for level in threshold_levels:
        above = vix >= level
        events, last = [], None
        for i, (date, flag) in enumerate(above.items()):
            if flag and (last is None or i - last > gap_sessions):
                events.append(date)
            if flag:
                last = i
        rows = [{"date": d.strftime("%Y-%m-%d"), "vix": float(vix[d]), **{k: float(v[d]) for k, v in fwd.items()},
                 "further_drop_3m": float(dd["worst_3m"][d])} for d in events]
        out[f"vix_{level:g}"] = {
            "events": rows,
            "mean": {k: float(np.nanmean([r[k] for r in rows])) for k in horizons},
            "hit_rate_12m": float(np.nanmean([r["12m"] > 0 for r in rows if not np.isnan(r["12m"])])),
        }
    return out


# ------------------------------------------------------------------ main


def prepare(end: pd.Timestamp) -> dict[str, Any]:
    membership = load_membership()
    dates = [d for d in month_ends(STUDY_START, end) if d <= membership.index[-1] + pd.offsets.MonthEnd(1)]
    per_date = {d: members_asof(membership, d) for d in dates}
    tickers = sorted(set().union(*per_date.values()))
    ticker_map = load_ticker_map()
    ciks = {t: CIK_OVERRIDES.get(t, ticker_map.get(t)) for t in tickers}
    mapped = {t: c for t, c in ciks.items() if c is not None}
    print(f"Universe: {len(tickers)} tickers were S&P 500 members between {dates[0]:%Y-%m} and {dates[-1]:%Y-%m}; "
          f"{len(mapped)} map to a current SEC filer.")

    prices = fetch_price_history(sorted(mapped) + ["SPY", "RSP"], PRICE_START)
    have = {t: c for t, c in mapped.items() if t in prices}
    print(f"Prices: {len(have)} of {len(mapped)} mapped tickers have Yahoo history.")

    def fetch(t: str) -> tuple[str, bool]:
        try:
            load_company_facts(t, have[t], PREDECESSOR_CIKS.get(t, ()))
            return t, True
        except SecDataError as exc:
            logger.warning("No SEC facts for %s: %s", t, exc)
            return t, False

    ok: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(fetch, t) for t in have]):
            t, good = fut.result()
            if good:
                ok[t] = have[t]
    print(f"SEC facts: {len(ok)} companies.")
    return {"membership": membership, "dates": dates, "per_date": per_date, "ciks": ok, "prices": prices,
            "all_tickers": tickers}


def build_panel(prep: dict[str, Any], workers: int) -> tuple[pd.DataFrame, dict[str, int]]:
    treasury = load_fred("DGS10", "2008-01-01")
    dates = prep["dates"]
    tre = treasury.reindex(pd.DatetimeIndex(dates), method="ffill")
    tre_map = {d: float(v) for d, v in tre.items() if pd.notna(v)}
    tasks = []
    for t, cik in prep["ciks"].items():
        member_dates = [d for d in dates if t in prep["per_date"][d]]
        if member_dates:
            tasks.append((t, cik, PREDECESSOR_CIKS.get(t, ()), member_dates, tre_map))
    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score_ticker, task) for task in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            r, s = fut.result()
            rows.extend(r)
            for k, v in s.items():
                stats[k] = stats.get(k, 0) + v
            if n % 100 == 0:
                print(f"  scored {n}/{len(tasks)} tickers")
    panel = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Point-in-time test of the value agents on the S&P 500")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--rebuild", action="store_true", help="Rescore even if a cached panel exists")
    parser.add_argument("--skip-insider", action="store_true", help="Leave out the Insider agent (no Form 4 download)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    end = pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)
    prep = prepare(end)
    panel_path = settings.DATA_DIR / "value_panel.parquet"
    if panel_path.exists() and not args.rebuild:
        panel = pd.read_parquet(panel_path)
        stats: dict[str, int] = {}
    else:
        panel, stats = build_panel(prep, args.workers)
        panel.to_parquet(panel_path, index=False)
    insider_path = settings.DATA_DIR / "insider_panel.parquet"
    if args.skip_insider:
        insider = pd.DataFrame(columns=["date", "ticker", "ins_score", "ins_conf", "ins_quality"])
    elif insider_path.exists() and not args.rebuild:
        insider = pd.read_parquet(insider_path)
    else:
        insider = build_insider_panel(prep, args.workers)
        insider.to_parquet(insider_path, index=False)
    panel = add_insider_composite(panel.merge(insider, on=["date", "ticker"], how="left"))
    report = analyze(panel, prep, stats)
    try:
        report["vix"] = vix = vix_study()
        print("\nVIX RULE  (SPY total return after the VIX first closes above a level; average over events)\n")
        print(f"{'':<22}{'events':>7}{'1m':>8}{'3m':>8}{'6m':>8}{'12m':>8}{'12m > 0':>9}")
        print(f"{'All days':<22}{'':>7}" + "".join(f"{vix['all_days'][k]:>8.1%}" for k in ("1m", "3m", "6m", "12m")))
        for key in ("vix_30", "vix_40", "vix_50"):
            v = vix[key]
            print(f"{'VIX first close >= ' + key[4:]:<22}{len(v['events']):>7}" + "".join(f"{v['mean'][k]:>8.1%}" for k in ("1m", "3m", "6m", "12m"))
                  + f"{v['hit_rate_12m']:>9.0%}")
        for e in vix["vix_40"]["events"]:
            print(f"   {e['date']}  VIX {e['vix']:5.1f}  next 12m {e['12m']:+.1%}  worst further drop within 3m {e['further_drop_3m']:+.1%}")
    except Exception as exc:  # noqa: BLE001 - optional section
        logger.warning("VIX study skipped: %s", exc)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "value_study.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


def analyze(panel: pd.DataFrame, prep: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    dates = prep["dates"]
    prices = prep["prices"]
    level = month_end_panel(prices, dates + [dates[-1] + pd.offsets.MonthEnd(k) for k in range(1, 13)])
    level.loc[level.index > prices["SPY"].index[-1] + pd.Timedelta(days=3)] = np.nan  # months not finished yet
    fwd1 = forward_returns(level, 1)
    fwd12 = forward_returns(level, 12)
    # A missing next price means the stock stopped trading; its last price is carried (cash).
    rf = load_fred("DTB3", "2010-01-01") / 100 / 12
    rf_m = rf.resample("ME").last().ffill()

    coverage = panel.groupby("date").size() / pd.Series({d: len(prep["per_date"][d]) for d in dates})
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "months": int(panel["date"].nunique()),
        "coverage": {"mean": float(coverage.mean()), "first": float(coverage.dropna().iloc[0]),
                     "last": float(coverage.dropna().iloc[-1])},
        "scoring_stats": stats,
    }
    print(f"\nPanel: {len(panel):,} company-months, {panel['ticker'].nunique()} companies, "
          f"{panel['date'].min():%Y-%m} to {panel['date'].max():%Y-%m}. "
          f"Coverage of index members: {coverage.dropna().iloc[0]:.0%} at the start, "
          f"{coverage.dropna().iloc[-1]:.0%} at the end.")

    fill = {c: float(panel[c].notna().mean()) for c in ("fund_score", "sh_score", "ev_to_fcf", "fcf_ev_yield", "per",
                                                          "roic", "roic_high_share", "reinvestment", "share_cagr",
                                                          "sbc_to_revenue", "market_cap")}
    report["field_fill_rate"] = fill
    print("Share of company-months with each input: " + ", ".join(f"{k} {v:.0%}" for k, v in fill.items()))

    spy = month_end_panel({"SPY": prices["SPY"]}, list(level.index))["SPY"]
    spy.loc[spy.index > prices["SPY"].index[-1] + pd.Timedelta(days=3)] = np.nan
    spy_ret = (spy.shift(-1) / spy - 1)
    spy_ret.index = spy_ret.index + pd.offsets.MonthEnd(1)

    # ---- derived columns
    panel = apply_framework_exclusions(panel)
    excluded_rows = int((~panel["framework_applicable"]).sum())
    report["excluded_sectors"] = {"rows": excluded_rows, "share": float(excluded_rows / max(len(panel), 1)),
                                  "sectors": sorted(panel.loc[~panel["framework_applicable"], "sector"].unique().tolist())}
    print(f"Not rated (utilities, REITs; T-12): {excluded_rows:,} company-months ({excluded_rows / max(len(panel), 1):.0%}) "
          "left out of the owner composite.")
    panel["inv_per_ex_cash"] = 1.0 / panel["per_ex_cash"].where(panel["per_ex_cash"] > 0)
    ev = panel["enterprise_value"].where(panel["enterprise_value"] > 0)
    panel["reported_fcf_yield"] = panel["reported_fcf"] / ev
    positive_fcf = panel["fcf"].where(panel["fcf"] > 0)
    panel["organic_reinvestment"] = panel["growth_capex"] / positive_fcf
    panel["acquisition_rate"] = panel["acquisitions"] / positive_fcf
    panel["net_dilution_cost"] = (panel["sbc"] - panel["buybacks"].fillna(0)) / panel["market_cap"]
    mom = (level.shift(1) / level.shift(12) - 1).stack().rename("mom_12_1").reset_index()
    mom.columns = ["date", "ticker", "mom_12_1"]
    panel = panel.merge(mom, on=["date", "ticker"], how="left")
    if "ins_buys" in panel:
        has = panel["ins_quality"].isin(["LIVE", "PARTIAL"])
        panel["ins_any_buy"] = (panel["ins_buys"].fillna(0) > 0).astype(float).where(has)
        panel["ins_cluster_f"] = panel["ins_cluster"].fillna(False).astype(float).where(has)
    else:
        panel["ins_any_buy"] = panel["ins_cluster_f"] = np.nan
    panel["compounder"] = ((panel["roic"] >= 0.15) & (panel["reinvestment"] >= 0.20)).astype(float).where(panel["reinvestment"].notna())
    panel["owner_screen"] = ((panel["ev_to_fcf"] < 17) & (panel["roic"] >= 0.135)).astype(float).where(panel["roic"].notna())
    nonfin = ~panel["is_financial"].astype(bool)
    fin = panel["is_financial"].astype(bool)
    for comp in ("valuation", "returns", "reinvestment", "cash_yield"):
        if f"comp_{comp}" not in panel:
            panel[f"comp_{comp}"] = np.nan

    # ---- portfolios
    variants: dict[str, tuple[pd.Series, float]] = {}
    variants["Equal-weight universe"] = portfolio_returns(panel, fwd1, "composite", "all", cost_bps=0)
    for col, label in (("composite", "Value composite (fundamentals + shareholder)"),
                       ("fund_score", "Fundamentals agent"), ("sh_score", "Shareholder agent")):
        variants[f"{label}: top 20%"] = portfolio_returns(panel, fwd1, col, "top_q")
        variants[f"{label}: bottom 20%"] = portfolio_returns(panel, fwd1, col, "bottom_q")
    variants["Value + insider composite: top 20%"] = portfolio_returns(panel, fwd1, "composite_ins", "top_q")
    variants["Value + insider composite: bottom 20%"] = portfolio_returns(panel, fwd1, "composite_ins", "bottom_q")
    variants["Value composite: top 30, monthly"] = portfolio_returns(panel, fwd1, "composite", "top_30")
    variants["Value composite: top 30, yearly (April)"] = portfolio_returns(panel, fwd1, "composite", "top_30", (4,))
    variants["Value composite: top 20%, yearly (April)"] = portfolio_returns(panel, fwd1, "composite", "top_q", (4,))
    variants["Value + insider: top 20%, yearly (April)"] = portfolio_returns(panel, fwd1, "composite_ins", "top_q", (4,))
    variants["Value composite top 20%, only 12-1 momentum > 0"] = portfolio_returns(
        panel, fwd1, "composite", "top_q", keep_if="mom_12_1")

    common = None
    for s, _ in variants.values():
        common = s.index if common is None else common.intersection(s.index)
    common = common[common <= spy_ret.dropna().index[-1]]
    rsp = month_end_panel({"RSP": prices["RSP"]}, list(level.index))["RSP"]
    rsp_ret = rsp.shift(-1) / rsp - 1
    rsp_ret.index = rsp_ret.index + pd.offsets.MonthEnd(1)
    rows = {"S&P 500 (SPY)": performance(spy_ret.reindex(common), rf_m.reindex(common)),
            "S&P 500 equal weight ETF (RSP), no survivorship": performance(rsp_ret.reindex(common), rf_m.reindex(common))}
    for name, (s, turnover) in variants.items():
        rows[name] = {**performance(s.reindex(common), rf_m.reindex(common)), "turnover": turnover}
    report["portfolios"] = rows
    report["period"] = {"start": str(common[0].date()), "end": str(common[-1].date())}

    print(f"\nPORTFOLIOS {common[0]:%Y-%m} to {common[-1]:%Y-%m} (equal weight, {COST_BPS:g} bps per trade)\n")
    halves = [(common[0], common[len(common) // 2 - 1]), (common[len(common) // 2], common[-1])]
    print(f"{'':<56}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'Turn':>6}{'1st half':>10}{'2nd half':>10}")
    series_by_name = {"S&P 500 (SPY)": spy_ret, "S&P 500 equal weight ETF (RSP), no survivorship": rsp_ret,
                      **{k: v[0] for k, v in variants.items()}}
    for name, st in rows.items():
        turn = f"{st['turnover']:.0%}" if "turnover" in st else ""
        sub = [performance(series_by_name[name].reindex(common).loc[a:b], rf_m.reindex(common).loc[a:b])["cagr"] for a, b in halves]
        st["cagr_first_half"], st["cagr_second_half"] = sub
        print(f"{name:<56}{st['cagr']:>7.1%}{st['volatility']:>7.1%}{st['sharpe']:>8.2f}{st['max_drawdown']:>8.1%}{turn:>6}"
              f"{sub[0]:>10.1%}{sub[1]:>10.1%}")
    print(f"  halves: {halves[0][0]:%Y-%m} to {halves[0][1]:%Y-%m} and {halves[1][0]:%Y-%m} to {halves[1][1]:%Y-%m}")

    yearly = pd.DataFrame({
        "SPY": spy_ret.reindex(common),
        "EW universe": variants["Equal-weight universe"][0].reindex(common),
        "Composite top 20%": variants["Value composite (fundamentals + shareholder): top 20%"][0].reindex(common),
    })
    yearly = (1 + yearly).groupby(yearly.index.year).prod() - 1
    report["yearly"] = {int(y): {k: float(v) for k, v in r.items()} for y, r in yearly.iterrows()}
    print("\n  Year     SPY   EW univ   Composite top 20%   vs EW")
    for y, r in yearly.iterrows():
        print(f"  {y}  {r['SPY']:>6.1%}  {r['EW universe']:>7.1%}  {r['Composite top 20%']:>12.1%}  {r['Composite top 20%'] - r['EW universe']:>+10.1%}")

    # ---- factors
    tests = [
        ("Fundamentals agent score", "fund_score", True, None, False),
        ("Shareholder agent score", "sh_score", True, None, False),
        ("Value composite", "composite", True, None, False),
        ("Insider agent score", "ins_score", True, None, False),
        ("Value + insider composite", "composite_ins", True, None, False),
        ("Any open-market insider buy (180 days)", "ins_any_buy", True, None, False),
        ("Insider cluster buy (2+ buyers in 30 days)", "ins_cluster_f", True, None, False),
        ("  valuation points (EV/FCF, PER, EV/MC)", "comp_valuation", True, nonfin, False),
        ("  returns points (ROIC, persistence)", "comp_returns", True, nonfin, False),
        ("  reinvestment points", "comp_reinvestment", True, nonfin, False),
        ("FCF yield on EV (1 / EV/FCF), owner FCF", "fcf_ev_yield", True, nonfin, False),
        ("  same, ranked within sector", "fcf_ev_yield", True, nonfin, True),
        ("Earnings yield (1 / PER)", "earnings_yield", True, None, False),
        ("1 / PER ex-cash", "inv_per_ex_cash", True, nonfin, False),
        ("EV / market cap (debt load)", "ev_to_mc", False, nonfin, False),
        ("ROIC", "roic", True, nonfin, False),
        ("  same, ranked within sector", "roic", True, nonfin, True),
        ("ROIC >= 15% share of years (4 years)", "roic_high_share", True, nonfin, False),
        ("ROIC >= 15% share of years (up to 10 years)", "roic_high_share_10y", True, nonfin, False),
        ("Reported FCF yield (operating cash flow - CapEx) / EV", "reported_fcf_yield", True, nonfin, False),
        ("Reinvestment rate", "reinvestment", True, nonfin, False),
        ("  organic part: (CapEx - D&A) / FCF", "organic_reinvestment", True, nonfin, False),
        ("  acquisitions / FCF", "acquisition_rate", False, nonfin, False),
        ("Compounder: ROIC >= 15% and reinvestment >= 20%", "compounder", True, nonfin, False),
        ("Owner screen: EV/FCF < 17 and ROIC >= 13.5%", "owner_screen", True, nonfin, False),
        ("Share count change per year (cannibals)", "share_cagr", False, None, False),
        ("Stock-based pay / revenue", "sbc_to_revenue", False, None, False),
        ("Buyback yield", "buyback_yield", True, None, False),
        ("Stock pay net of buybacks / market cap", "net_dilution_cost", False, None, False),
        ("12-1 month price momentum (reference)", "mom_12_1", True, None, False),
        ("ROE (financials)", "roe", True, fin, False),
        ("ROA (financials)", "roa", True, fin, False),
        ("Earnings yield (financials)", "earnings_yield", True, fin, False),
    ]
    report["factors"] = {}
    print("\nFACTORS  (rank IC vs next 1 and 12 months, Newey-West t; top minus bottom quintile per year)\n")
    print(f"{'':<50}{'names':>6}{'IC 1m':>8}{'t':>6}{'IC 12m':>8}{'t':>6}{'Q5-Q1':>8}{'1st half':>9}{'2nd half':>9}{'hit':>6}")
    for label, col, higher, subset, by_sector in tests:
        res = factor_test(panel, fwd1, fwd12, col, higher, subset, by_sector)
        if res is None:
            continue
        report["factors"][label.strip()] = res
        print(f"{label:<50}{res['avg_names']:>6}{res['ic_1m']:>+8.3f}{res['ic_1m_t']:>6.1f}{res['ic_12m']:>+8.3f}"
              f"{res['ic_12m_t']:>6.1f}{res['q5_minus_q1_annual']:>+8.1%}{res['q5_minus_q1_first_half']:>+9.1%}"
              f"{res['q5_minus_q1_second_half']:>+9.1%}{res['hit_rate']:>6.0%}")

    # ---- valuation level of the market over time (FCF definition from settings)
    nf = panel[~panel["is_financial"].astype(bool)]
    by_year = nf.groupby(nf["date"].dt.year)
    regime = pd.DataFrame({
        "median_ev_to_fcf": by_year["ev_to_fcf"].median(),
        "share_below_17x": by_year["ev_to_fcf"].apply(lambda x: float((x.dropna() < 17).mean())),
        "median_per": panel.groupby(panel["date"].dt.year)["per"].median(),
        "median_roic": by_year["roic"].median(),
    })
    report["valuation_by_year"] = {int(y): {k: float(v) for k, v in r.items()} for y, r in regime.iterrows()}
    fcf_label = FCF_DEFINITIONS.get(settings.FCF_DEFINITION, settings.FCF_DEFINITION)
    report["fcf_definition"] = settings.FCF_DEFINITION
    print(f"\nMARKET VALUATION BY YEAR (non-financial S&P 500 members, EV/FCF on {fcf_label})\n")
    print("  Year  median EV/FCF  share below 17x  median PER  median ROIC")
    for y, r in regime.iterrows():
        print(f"  {y}  {r['median_ev_to_fcf']:>12.1f}x  {r['share_below_17x']:>14.0%}  {r['median_per']:>9.1f}x  {r['median_roic']:>10.1%}")

    # ---- what the top picks look like
    top = panel.dropna(subset=["composite"]).copy()
    top["q"] = top.groupby("date")["composite"].rank(pct=True)
    tilt = pd.DataFrame({
        "top 20%": top[top["q"] > 0.8]["sector"].value_counts(normalize=True),
        "universe": top["sector"].value_counts(normalize=True),
    }).fillna(0.0).sort_values("top 20%", ascending=False)
    report["sector_tilt"] = {s: {k: float(v) for k, v in r.items()} for s, r in tilt.iterrows()}
    print("\nSECTOR MIX (share of company-months)\n")
    for s, r in tilt.iterrows():
        print(f"  {s:<24}{r['top 20%']:>7.0%} of top 20%   {r['universe']:>6.0%} of universe")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
