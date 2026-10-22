"""Point-in-time test of the Competitive Position agent on the S&P 500, 2011-2026.

The agent ships at weight 0 (decision_log.md D-34). This study decides, with a test written down before the first
run, whether it earns a vote in the owner score. It replays the agent's scored part (days to pay suppliers, days
to get paid, gross margin and its stability, ROIC, operating margin, growth, share of peer-group sales) with the
same ``company_metrics``, ``choose_peers_by_sic`` and ``score_competitive`` the live agent runs, from the 10-K
facts known on each month-end. Peers are the SEC industry-code peers among that month's index members (Yahoo
industry membership has no history). Not replayed, so 0 points in the agent: the 10-K text (it would need every
historical 10-K, and the extractors were checked on 24 recent filings only) and EV/FCF against peers.

Acceptance test, fixed before the first run (``PROTOCOL``). Target: ``cp_score`` only, on company-months that have
one.

- A1 standalone: 12-month rank IC above 0 with Newey-West t of at least 2.0 over the whole period AND in each half.
- A2 spread: the next-month top-minus-bottom quintile return, annualized, above 0 over the whole period and in each half.
- A3 incremental: the 12-month partial IC given the owner composite above 0 with t of at least 2.0, whole and per half.
- A4 blend: share of the owner layer in {0, 10%, 20%}; the share whose weaker half beats 0 by most is chosen (the
  smaller share when two are within 0.3 points a year) and must beat 0 in both halves.

Pass needs all four; the shipped default then becomes ``owner_weights.competitive = share / (1 - share)``. Any fail
keeps weight 0. The study changes no setting by itself and writes only ``analysis/output/competitive_study.json``
and two caches in ``data/cache/``.

Inputs are the cached value panel (``core/value_study.py``), prices and SEC extracts, so the first run needs only
the SEC contact (cost of revenue and gross profit are downloaded once per company, about 3 minutes).

Run: ``SEC_USER_AGENT="app you@example.com" python -m core.competitive_study [--workers N] [--rebuild]
[--limit N] [--yahoo-industries]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from agents.competitive_agent import CompetitiveAgent
from agents.fundamental_agent import FundamentalAgent
from agents.scoring import aggregate
from agents.shareholder_agent import ShareholderAgent
from analysis.competitive_metrics import (
    SCORING_VERSION,
    Candidate,
    CompanyMetrics,
    SicIndex,
    card_years,
    choose_peers,
    choose_peers_by_sic,
    company_metrics,
    score_competitive,
    usable_peers,
)
from config.weights import load_scoring_config
from core.all_agents_study import _assess
from core.value_study import (
    CIK_OVERRIDES,
    PREDECESSOR_CIKS,
    PRICE_START,
    apply_framework_exclusions,
    forward_returns,
    month_end_panel,
    newey_west_t,
    portfolio_returns,
)
from data import peers as peers_mod
from data.price_history import fetch_price_history
from data.sec_edgar import SecConfigError, SecDataError, has_sec_access, load_company_facts, load_ticker_map
from data.sec_margins import load_margin_facts, margins_asof

logger = logging.getLogger(__name__)

PANEL_FILE = "competitive_panel.parquet"
FEATURES_FILE = "competitive_features.parquet"
REPORT_FILE = "competitive_study.json"
FEATURES_VERSION = 1  # bump when the figures behind the features change


def panel_signature(panel: pd.DataFrame) -> str:
    """First and last month, months, tickers and rows of the panel, stored beside each cache so a grown panel rebuilds it."""
    return f"{panel['date'].min():%Y-%m-%d}|{panel['date'].max():%Y-%m-%d}|{panel['date'].nunique()}|{panel['ticker'].nunique()}|{len(panel)}"


def _cache_fits(cached: pd.DataFrame | None, version_column: str, version: int, signature: str) -> bool:
    return (cached is not None and not cached.empty and cached[version_column].iloc[0] == version
            and "panel_signature" in cached and cached["panel_signature"].iloc[0] == signature)
SHARES = (0.0, 0.10, 0.20)
SHARE_TIE = 0.003  # shares whose weaker halves are within 0.3 points a year: take the smaller
MIN_T = 2.0
MIN_NAMES = 50
FEATURE_FIELDS = ("fy_end", "gross_margin", "gross_margin_std", "operating_margin", "dso", "dpo", "dio", "roic",
                  "revenue_cagr_3y", "revenue_usd", "revenue_3y_ago_usd", "has_cost_of_revenue")
SCORE_COLUMNS = ("cp_score", "cp_suppliers", "cp_customers", "cp_standing")
DIAGNOSTICS = ("dpo_ratio", "dso_ratio", "gm_gap", "gm_std_ratio", "roic_gap", "opm_gap", "growth_gap", "share_change")

PROTOCOL = {
    "target": "cp_score, on company-months that have one (no picking of a component after the run)",
    "A1": "12-month rank IC > 0 with Newey-West t >= 2.0 over the whole period and in each half",
    "A2": "next-month top-minus-bottom quintile return, annualized, > 0 over the whole period and in each half",
    "A3": "12-month partial IC given the owner composite > 0 with t >= 2.0, whole period and each half",
    "A4": "owner-layer share in {0, 10%, 20%}: the share with the best weaker half (the smaller when within 0.3 points a "
          "year) beats share 0 in both halves, top 20% of the owner score, 10 bps per trade",
    "decision": "pass needs A1 to A4: owner_weights.competitive = share / (1 - share); any fail keeps weight 0",
    "not_tested": "10-K text notes and EV/FCF against peers (carry 0 points)",
}


# ------------------------------------------------------------------ inputs


def prepare_cached(panel: pd.DataFrame, limit: int | None = None) -> dict[str, Any]:
    """Dates, SEC CIKs, prices and forward returns from the caches the value study left behind (no membership download)."""
    dates = [pd.Timestamp(d) for d in sorted(panel["date"].unique())]
    tickers = sorted(panel["ticker"].unique())
    if limit:
        tickers = tickers[:limit]
    ticker_map = load_ticker_map(max_age_days=math.inf)
    ciks = {t: CIK_OVERRIDES.get(t, ticker_map.get(t)) for t in tickers}
    ciks = {t: c for t, c in ciks.items() if c is not None}
    prices = fetch_price_history(sorted(ciks) + ["SPY", "RSP"], PRICE_START, max_age_days=math.inf)
    level = month_end_panel(prices, dates + [dates[-1] + pd.offsets.MonthEnd(k) for k in range(1, 13)])
    level.loc[level.index > prices["SPY"].index[-1] + pd.Timedelta(days=3)] = np.nan  # months not finished yet
    return {"dates": dates, "ciks": ciks, "prices": prices, "fwd1": forward_returns(level, 1), "fwd12": forward_returns(level, 12)}


def ensure_margin_facts(ciks: dict[str, int], workers: int = 4) -> dict[str, int]:
    """Download (once) cost of revenue and gross profit for every company; returns the tickers that have them."""
    ok: dict[str, int] = {}

    def fetch(item: tuple[str, int]) -> tuple[str, int, bool]:
        ticker, cik = item
        wanted = (cik, *PREDECESSOR_CIKS.get(ticker, ()))
        try:
            for c in wanted:
                load_margin_facts(c, max_age_days=math.inf)
            return ticker, cik, True
        except SecDataError as exc:
            logger.warning("No margin facts for %s: %s", ticker, exc)
            return ticker, cik, False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(fetch, item) for item in ciks.items()]):
            ticker, cik, good = fut.result()
            if good:
                ok[ticker] = cik
    return ok


# ------------------------------------------------------------------ features (worker process)


def _features_for_ticker(task: tuple) -> list[dict[str, Any]]:
    """Company figures at each month-end of one ticker (like ``value_study._score_ticker``)."""
    ticker, cik, predecessors, dates = task
    try:
        cf = load_company_facts(ticker, cik, predecessors, max_age_days=math.inf)
        margin_frames = [load_margin_facts(c, max_age_days=math.inf) for c in (cik, *predecessors)]
    except (SecConfigError, SecDataError):
        return []
    margins_frame = pd.concat(margin_frames, ignore_index=True) if len(margin_frames) > 1 else margin_frames[0]
    filings = cf.filing_dates()
    margin_filings = sorted(margins_frame["filed"].unique()) if not margins_frame.empty else []
    memo: dict[tuple[int, int], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for date in dates:
        key = (bisect_right(filings, np.datetime64(date)), bisect_right(margin_filings, np.datetime64(date)))
        if key not in memo:
            st = cf.statements_asof(date, max_years=6)
            memo[key] = card_years(st, margins_asof(margins_frame, date), max_years=6, roic_years=1) if st.available else []
        years = memo[key]
        if not years:
            continue
        m = company_metrics(years, 1.0, None, ticker)
        rows.append({"date": date, "ticker": ticker, **{f: getattr(m, f) for f in FEATURE_FIELDS}})
    return rows


def build_features(panel: pd.DataFrame, ciks: dict[str, int], workers: int) -> pd.DataFrame:
    members = panel.groupby("ticker")["date"].apply(list)
    tasks = [(t, cik, PREDECESSOR_CIKS.get(t, ()), members[t]) for t, cik in ciks.items() if t in members.index]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_features_for_ticker, task) for task in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if n % 100 == 0:
                print(f"  figures built for {n}/{len(tasks)} tickers")
    frame = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    frame["features_version"] = FEATURES_VERSION
    return frame


# ------------------------------------------------------------------ peers and scores


def _metrics_from(row: Any) -> CompanyMetrics:
    return CompanyMetrics(
        ticker=row.ticker, fy_end=row.fy_end, gross_margin=_none(row.gross_margin), gross_margin_std=_none(row.gross_margin_std),
        operating_margin=_none(row.operating_margin), dso=_none(row.dso), dpo=_none(row.dpo), dio=_none(row.dio),
        roic=_none(row.roic), revenue_cagr_3y=_none(row.revenue_cagr_3y), revenue_usd=_none(row.revenue_usd),
        revenue_3y_ago_usd=_none(row.revenue_3y_ago_usd), has_cost_of_revenue=bool(row.has_cost_of_revenue))


def _none(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def score_panel(
    universe: pd.DataFrame, features: pd.DataFrame, records: dict[str, dict[str, Any]], max_peers: int = 8,
    allow_wide: bool = True, industry_of: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Score every company-month of ``universe`` (date, ticker, market_cap) against the other members of its month.

    Peers come from the SEC industry-code ladder among the members that have figures that month; with
    ``industry_of`` (ticker -> Yahoo industry key) from today's Yahoo industry instead (a robustness check with
    look-ahead in the classification).
    """
    merged = universe.merge(features, on=["date", "ticker"], how="inner")
    out: list[dict[str, Any]] = []
    for date, group in merged.groupby("date"):
        metrics = {r.ticker: _metrics_from(r) for r in group.itertuples(index=False)}
        caps = dict(zip(group["ticker"], group["market_cap"]))
        candidates = [Candidate(t, records[t]["name"], caps[t], records[t]["sic"], records[t]["cik"])
                      for t in metrics if t in records]
        index = SicIndex(candidates)
        by_industry: dict[str, list[Candidate]] = {}
        for c in candidates:
            if industry_of and industry_of.get(c.ticker):
                by_industry.setdefault(industry_of[c.ticker], []).append(c)
        for t, subject in metrics.items():
            rec = records.get(t)
            if rec is None:
                continue
            identity = {"subject_ticker": t, "subject_name": rec["name"], "subject_cik": rec["cik"]}
            if industry_of is not None:
                choice = choose_peers(caps[t], by_industry.get(industry_of.get(t, ""), []), max_peers, allow_wide=allow_wide, **identity)
                choice.source = "yahoo_industry"
            else:
                choice = choose_peers_by_sic(caps[t], rec["sic"], index, max_peers, allow_wide=allow_wide, **identity)
            usable = usable_peers(subject, [metrics[p] for p in choice.tickers])
            score = score_competitive(subject, usable, None, weak_peer_group=choice.wide_band or choice.source == "sic2")
            ok = score.quality != "UNAVAILABLE"
            out.append({
                "date": date, "ticker": t, "scoring_version": SCORING_VERSION,
                "cp_score": score.total if ok else np.nan, "cp_conf": score.confidence if ok else 0.0, "cp_quality": score.quality,
                **{f"cp_{c}": (v if ok else np.nan) for c, v in (score.components or {}).items()},
                "n_peers": len(usable), "peer_level": choice.source or "none", "wide_band": choice.wide_band,
                **{k: (score.diagnostics.get(k) if ok else None) for k in DIAGNOSTICS},
            })
    return pd.DataFrame(out)


# ------------------------------------------------------------------ statistics


def monthly_ic(panel: pd.DataFrame, fwd: pd.DataFrame, column: str, min_names: int = MIN_NAMES) -> pd.Series:
    """Spearman rank correlation of ``column`` with the forward return, one value per month."""
    out: dict[pd.Timestamp, float] = {}
    for date, g in panel.dropna(subset=[column]).groupby("date"):
        if date not in fwd.index:
            continue
        r = fwd.loc[date].reindex(g["ticker"]).to_numpy()
        x = g[column].to_numpy(dtype=float)
        ok = ~np.isnan(r)
        if ok.sum() < min_names:
            continue
        out[date] = float(pd.Series(x[ok]).rank().corr(pd.Series(r[ok]).rank()))
    return pd.Series(out, dtype=float).sort_index()


def _block(values: np.ndarray, lags: int) -> dict[str, Any]:
    values = values[~np.isnan(values)]
    return {"mean": float(values.mean()) if len(values) else float("nan"),
            "t": newey_west_t(values, lags) if len(values) else float("nan"), "months": int(len(values))}


def summarize(series: pd.Series, lags: int) -> dict[str, dict[str, Any]]:
    """Mean, Newey-West t and month count over the whole period and over its first and second half."""
    values = series.dropna().to_numpy(dtype=float)
    half = len(values) // 2
    return {"whole": _block(values, lags), "first_half": _block(values[:half], lags), "second_half": _block(values[half:], lags)}


def quintile_spread(panel: pd.DataFrame, fwd1: pd.DataFrame, column: str, min_names: int = MIN_NAMES) -> pd.Series:
    """Next-month return of the top 20% minus the bottom 20% by ``column`` (equal weight), one value per month."""
    out: dict[pd.Timestamp, float] = {}
    for date, g in panel.dropna(subset=[column]).groupby("date"):
        if date not in fwd1.index:
            continue
        r = fwd1.loc[date].reindex(g["ticker"]).to_numpy()
        x = g[column].to_numpy(dtype=float)
        ok = ~np.isnan(r)
        if ok.sum() < min_names:
            continue
        xs, rs = x[ok], r[ok]
        hi, lo = np.quantile(xs, 0.8), np.quantile(xs, 0.2)
        top, bottom = (rs[xs >= hi], rs[xs <= lo]) if hi > lo else (rs[xs > lo], rs[xs <= lo])
        if len(top) and len(bottom):
            out[date] = float(top.mean() - bottom.mean())
    return pd.Series(out, dtype=float).sort_index()


def summarize_spread(series: pd.Series) -> dict[str, dict[str, Any]]:
    """Annualized mean and t of a monthly spread series, whole period and halves."""
    base = summarize(series, 3)
    for block in base.values():
        block["annual"] = float((1 + block["mean"]) ** 12 - 1) if block["months"] else float("nan")
    return base


def partial_ic(panel: pd.DataFrame, fwd: pd.DataFrame, column: str, controls: list[str], min_names: int = MIN_NAMES) -> pd.Series:
    """Rank IC of what ``column`` adds beyond ``controls``: each month the feature's rank is regressed on the controls'
    ranks (OLS with intercept) and the residual's rank IC with the forward return is taken."""
    out: dict[pd.Timestamp, float] = {}
    for date, g in panel.dropna(subset=[column, *controls]).groupby("date"):
        if date not in fwd.index:
            continue
        r = fwd.loc[date].reindex(g["ticker"]).to_numpy()
        ok = ~np.isnan(r)
        if ok.sum() < min_names:
            continue
        g = g[ok]
        y = g[column].rank(pct=True).to_numpy()
        design = np.column_stack([np.ones(len(g))] + [g[c].rank(pct=True).to_numpy() for c in controls])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        with np.errstate(all="ignore"):  # a residual of exactly zero has no rank correlation
            out[date] = float(pd.Series(resid).rank().corr(pd.Series(r[ok]).rank()))
    return pd.Series(out, dtype=float).sort_index()


def blend(panel: pd.DataFrame, share: float) -> pd.Series:
    """Owner composite with the competitive agent given ``share`` of the layer (when all three have data).

    Rows without a competitive score keep today's composite exactly; share 0 is the panel's composite.
    """
    if share <= 0:
        return panel["composite"].copy()
    config = load_scoring_config()
    scales = {"fundamentals": FundamentalAgent.score_scale, "shareholder": ShareholderAgent.score_scale,
              "competitive": CompetitiveAgent.score_scale}
    weights = {"fundamentals": config.owner_weights.get("fundamentals", 0.45),
               "shareholder": config.owner_weights.get("shareholder", 0.55), "competitive": share / (1.0 - share)}
    out = panel["composite"].copy()
    rows = panel[panel["cp_score"].notna() & panel["composite"].notna()]
    for idx, fs, fc, ss, sc, cs, cc in zip(rows.index, rows["fund_score"], rows["fund_conf"], rows["sh_score"], rows["sh_conf"],
                                           rows["cp_score"], rows["cp_conf"]):
        assessments = [_assess("fundamentals", fs, fc), _assess("shareholder", ss, sc), _assess("competitive", cs, cc)]
        out.at[idx] = aggregate(assessments, config, None, scales, weight_overrides=weights,
                                thresholds=config.owner_thresholds).composite_score
    return out


def _cagr(monthly: pd.Series) -> float:
    r = monthly.dropna()
    return float((1 + r).prod() ** (12 / len(r)) - 1) if len(r) else float("nan")


def blend_grid(panel: pd.DataFrame, fwd1: pd.DataFrame, shares: tuple[float, ...] = SHARES) -> dict[str, Any]:
    """Annual return of the owner top 20% at each share of the layer, whole period and halves, and the excess over share 0."""
    series: dict[float, pd.Series] = {}
    turnover: dict[float, float] = {}
    for share in shares:
        frame = panel.assign(_blend=blend(panel, share))
        series[share], turnover[share] = portfolio_returns(frame, fwd1, "_blend", "top_q")
    if any(s.empty for s in series.values()):
        return {"months": 0, "shares": {}}
    common = series[shares[0]].index
    for s in series.values():
        common = common.intersection(s.index)
    half = len(common) // 2
    rows = {}
    for share, s in series.items():
        s = s.reindex(common)
        rows[share] = {"cagr": _cagr(s), "first_half": _cagr(s.iloc[:half]), "second_half": _cagr(s.iloc[half:]),
                       "turnover": turnover[share]}
    base = rows[shares[0]]
    for share, row in rows.items():
        row["excess_first_half"] = row["first_half"] - base["first_half"]
        row["excess_second_half"] = row["second_half"] - base["second_half"]
    return {"months": int(len(common)), "shares": {f"{s:g}": r for s, r in rows.items()}}


# ------------------------------------------------------------------ verdict


def _passes(block: dict[str, dict[str, Any]], need_t: bool = True) -> bool:
    return all(
        b["mean"] > 0 and (not need_t or b["t"] >= MIN_T)
        for b in block.values()
    )


def choose_share(grid: dict[str, Any]) -> tuple[float, bool]:
    """(chosen share, whether it beats share 0 in both halves). The best weaker half wins; the smaller share when within 0.3 points a year."""
    candidates = []
    for key, row in grid["shares"].items():
        share = float(key)
        if share <= 0:
            continue
        candidates.append((share, min(row["excess_first_half"], row["excess_second_half"])))
    if not candidates:
        return 0.0, False
    best = max(w for _, w in candidates)
    near = [s for s, w in candidates if best - w <= SHARE_TIE + 1e-12]
    chosen = min(near)
    row = grid["shares"][f"{chosen:g}"]
    return chosen, bool(row["excess_first_half"] > 0 and row["excess_second_half"] > 0)


def verdict(ic12: dict, spread: dict, partial12: dict, grid: dict) -> dict[str, Any]:
    """Apply the acceptance test of ``PROTOCOL`` to the computed statistics."""
    chosen, a4 = choose_share(grid)
    checks = {
        "A1": _passes(ic12),
        "A2": all(b["annual"] > 0 for b in spread.values()),
        "A3": _passes(partial12),
        "A4": a4,
    }
    failing = [k for k, ok in checks.items() if not ok]
    passed = not failing
    return {
        "verdict": "pass" if passed else "fail", "failing": failing, "checks": checks, "chosen_share": chosen,
        "adopted_owner_weight": round(chosen / (1.0 - chosen), 4) if passed else 0.0,
        "message": (f"Pass: owner_weights.competitive = {chosen / (1.0 - chosen):.3f} ({chosen:.0%} of the owner layer)"
                    if passed else f"Fail ({', '.join(failing)}): the weight stays 0 and the agent stays evidence only"),
    }


# ------------------------------------------------------------------ report


def _factor_table(panel: pd.DataFrame, fwd1: pd.DataFrame, fwd12: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        if col not in panel or panel[col].notna().sum() < MIN_NAMES:
            continue
        out[col] = {"ic_1m": summarize(monthly_ic(panel, fwd1, col), 3), "ic_12m": summarize(monthly_ic(panel, fwd12, col), 12),
                    "spread": summarize_spread(quintile_spread(panel, fwd1, col))}
    return out


def analyze(panel: pd.DataFrame, prep: dict[str, Any], robustness: dict[str, Any] | None = None) -> dict[str, Any]:
    """The full report on a panel with the value columns and the ``cp_*`` columns."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    fwd1, fwd12 = prep["fwd1"], prep["fwd12"]
    sample = panel[panel["cp_score"].notna()]
    columns = ["cp_score", *SCORE_COLUMNS[1:], *DIAGNOSTICS]
    features = _factor_table(sample, fwd1, fwd12, columns)
    partial = {col: {"given_composite": summarize(partial_ic(sample, fwd12, col, ["composite"]), 12),
                     "given_fund_and_shareholder": summarize(partial_ic(sample, fwd12, col, ["fund_score", "sh_score"]), 12)}
               for col in ("cp_score", "cp_suppliers", "cp_customers", "cp_standing") if sample[col].notna().sum() >= MIN_NAMES}
    grid = blend_grid(panel, fwd1)
    if "cp_score" in features and "cp_score" in partial:
        main = features["cp_score"]
        decision = verdict(main["ic_12m"], main["spread"], partial["cp_score"]["given_composite"], grid)
    else:
        decision = {"verdict": "fail", "failing": ["A1", "A2", "A3", "A4"], "checks": dict.fromkeys(("A1", "A2", "A3", "A4"), False),
                    "chosen_share": 0.0, "adopted_owner_weight": 0.0,
                    "message": f"Fail: fewer than {MIN_NAMES} scored companies in a month; nothing to test"}

    universe_rows = panel.groupby("date").size()
    coverage = (sample.groupby("date").size() / universe_rows).dropna()
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {**PROTOCOL, "run_date": datetime.now().strftime("%Y-%m-%d")},
        "scoring_version": SCORING_VERSION,
        "coverage": {"rows": int(len(panel)), "scored_rows": int(len(sample)), "mean": float(coverage.mean()),
                     "by_year": {int(y): float(v) for y, v in coverage.groupby(coverage.index.year).mean().items()},
                     "avg_peers": float(sample["n_peers"].mean()),
                     "peer_level_share": {k: float(v) for k, v in sample["peer_level"].value_counts(normalize=True).items()},
                     "wide_band_share": float(sample["wide_band"].mean())},
        "features": features, "partial_ic": partial, "blend": grid, "verdict": decision,
        "robustness": robustness or {},
        "risk_free_note": "Sharpe and risk-free rates are not used by this test",
    }
    return report


def _robustness(panel: pd.DataFrame, prep: dict[str, Any], variants: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Reported, never part of the decision: IC within sectors, one-month IC and the alternative peer rules."""
    fwd1, fwd12 = prep["fwd1"], prep["fwd12"]
    out: dict[str, Any] = {}
    sample = panel[panel["cp_score"].notna()].copy()
    sample["cp_in_sector"] = sample.groupby(["date", "sector"])["cp_score"].rank(pct=True)
    out["ic_12m_within_sector"] = summarize(monthly_ic(sample, fwd12, "cp_in_sector"), 12)
    out["ic_1m"] = summarize(monthly_ic(sample, fwd1, "cp_score"), 3)
    for name, scores in variants.items():
        alt = panel.drop(columns=[c for c in panel.columns if c.startswith("cp_") or c in DIAGNOSTICS or c in ("n_peers", "peer_level", "wide_band", "scoring_version")])
        alt = alt.merge(scores, on=["date", "ticker"], how="left")
        alt = alt[alt["cp_score"].notna()]
        out[name] = {"rows": int(len(alt)), "ic_12m": summarize(monthly_ic(alt, fwd12, "cp_score"), 12),
                     "spread": summarize_spread(quintile_spread(alt, fwd1, "cp_score")),
                     "partial_ic_12m": summarize(partial_ic(alt, fwd12, "cp_score", ["composite"]), 12)}
    return out


def _fmt(block: dict[str, Any], key: str = "mean", digits: int = 3) -> str:
    return "n/a" if block is None or not np.isfinite(block.get(key, float("nan"))) else f"{block[key]:+.{digits}f}"


def print_report(report: dict[str, Any]) -> None:
    cov = report["coverage"]
    print(f"\nCompetitive position study: {cov['scored_rows']:,} scored company-months of {cov['rows']:,} "
          f"({cov['mean']:.0%} on average), {cov['avg_peers']:.1f} peers on average; "
          f"peer level {', '.join(f'{k} {v:.0%}' for k, v in cov['peer_level_share'].items())}.")
    print(f"\n{'':<16}{'IC 12m':>8}{'t':>6}{'1st half':>10}{'t':>6}{'2nd half':>10}{'t':>6}{'Q5-Q1 a yr':>12}{'1st':>8}{'2nd':>8}")
    for name, row in report["features"].items():
        ic, sp = row["ic_12m"], row["spread"]
        print(f"{name:<16}{_fmt(ic['whole']):>8}{_fmt(ic['whole'], 't', 1):>6}{_fmt(ic['first_half']):>10}{_fmt(ic['first_half'], 't', 1):>6}"
              f"{_fmt(ic['second_half']):>10}{_fmt(ic['second_half'], 't', 1):>6}{_fmt(sp['whole'], 'annual'):>12}"
              f"{_fmt(sp['first_half'], 'annual'):>8}{_fmt(sp['second_half'], 'annual'):>8}")
    print("\nPartial 12-month IC given the owner composite:")
    for name, row in report["partial_ic"].items():
        p = row["given_composite"]
        print(f"  {name:<14} whole {_fmt(p['whole'])} (t {_fmt(p['whole'], 't', 1)})  1st half {_fmt(p['first_half'])} "
              f"(t {_fmt(p['first_half'], 't', 1)})  2nd half {_fmt(p['second_half'])} (t {_fmt(p['second_half'], 't', 1)})")
    print("\nBlend (owner top 20%, share of the owner layer):")
    for share, row in report["blend"]["shares"].items():
        print(f"  share {float(share):>4.0%}  CAGR {row['cagr']:.1%}  1st half {row['first_half']:.1%} ({row['excess_first_half']:+.1%})  "
              f"2nd half {row['second_half']:.1%} ({row['excess_second_half']:+.1%})")
    v = report["verdict"]
    checks = ", ".join(f"{k} {'pass' if ok else 'FAIL'}" for k, ok in v["checks"].items())
    print(f"\nChecks: {checks}\nVerdict: {v['message']}")


# ------------------------------------------------------------------ main


def _yahoo_industries(tickers: list[str]) -> dict[str, str]:
    """Today's Yahoo industry key of each ticker (about two minutes for 600 tickers)."""
    from data import fundamentals

    out: dict[str, str] = {}

    def one(t: str) -> tuple[str, str | None]:
        try:
            return t, fundamentals._load_info(t).get("industryKey")
        except Exception:  # noqa: BLE001 - a ticker without a Yahoo profile just has no industry peers
            return t, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(one, t) for t in tickers]):
            t, key = fut.result()
            if key:
                out[t] = key
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Point-in-time test of the Competitive Position agent")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the cached figures and scores")
    parser.add_argument("--limit", type=int, help="Smoke test on the first N tickers")
    parser.add_argument("--yahoo-industries", action="store_true", help="Also run peers by today's Yahoo industry (robustness)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    value_path = settings.DATA_DIR / "value_panel.parquet"
    if not value_path.exists():
        print("No value panel: run `python -m core.value_study` first.")
        return 1
    columns = ["date", "ticker", "sector", "is_financial", "market_cap", "composite", "fund_score", "fund_conf", "fund_quality",
               "sh_score", "sh_conf", "sh_quality"]
    panel = apply_framework_exclusions(pd.read_parquet(value_path, columns=columns))
    panel = panel[panel["framework_applicable"] & ~panel["is_financial"].astype(bool) & (panel["market_cap"] > 0)].reset_index(drop=True)
    prep = prepare_cached(panel, args.limit)
    panel = panel[panel["ticker"].isin(prep["ciks"])].reset_index(drop=True)

    features_path, panel_path = settings.DATA_DIR / FEATURES_FILE, settings.DATA_DIR / PANEL_FILE
    signature = panel_signature(panel)
    cached = pd.read_parquet(features_path) if features_path.exists() and not args.rebuild and not args.limit else None
    if _cache_fits(cached, "features_version", FEATURES_VERSION, signature):
        features = cached
    else:
        if not has_sec_access():
            print("The SEC asks for a contact email with every request: start with "
                  "SEC_USER_AGENT=\"app you@example.com\" (needed once, to download cost of revenue and gross profit).")
            return 1
        print(f"Downloading cost of revenue and gross profit for {len(prep['ciks'])} companies (cached afterwards)...")
        prep["ciks"] = ensure_margin_facts(prep["ciks"])
        print("Building company figures at every month-end...")
        features = build_features(panel, prep["ciks"], args.workers)
        features["panel_signature"] = signature
        if not args.limit:
            features.to_parquet(features_path, index=False)

    records = peers_mod.sic_records()
    cached_scores = pd.read_parquet(panel_path) if panel_path.exists() and not args.rebuild and not args.limit else None
    if _cache_fits(cached_scores, "scoring_version", SCORING_VERSION, signature):
        scores = cached_scores
    else:
        print("Choosing peers and scoring...")
        scores = score_panel(panel[["date", "ticker", "market_cap"]], features, records)
        scores["panel_signature"] = signature
        if not args.limit:
            scores.to_parquet(panel_path, index=False)
    panel = panel.merge(scores.drop(columns="panel_signature", errors="ignore"), on=["date", "ticker"], how="left")

    print("Replaying the test...")
    variants = {
        "peers_capped_at_5": score_panel(panel[["date", "ticker", "market_cap"]], features, records, max_peers=5),
        "strict_band_only": score_panel(panel[["date", "ticker", "market_cap"]], features, records, allow_wide=False),
    }
    if args.yahoo_industries:
        industries = _yahoo_industries(sorted(panel["ticker"].unique()))
        variants["yahoo_industries_today"] = score_panel(panel[["date", "ticker", "market_cap"]], features, records, industry_of=industries)
    report = analyze(panel, prep, _robustness(panel, prep, variants))
    print_report(report)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = REPORT_FILE.replace(".json", "_smoke.json") if args.limit else REPORT_FILE  # a smoke test never replaces the report
    (settings.OUTPUT_DIR / name).write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
