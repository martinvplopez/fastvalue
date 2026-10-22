"""Does the stored news tone predict returns? The test that decides the News Sentiment agent's weight.

The agent stores one tone snapshot per ticker per day (``data/store/sentiment_history/``, built by the daily
check in ``core/monitor.py``). This script lines each snapshot up with what the stock did next, against SPY:

- Entry is the close of the first session after the snapshot day (a snapshot taken during the day may already
  be in that day's close), exit 5 and 20 sessions later. Returns are in excess of SPY over the same sessions.
- Rank IC per day across the tickers with a snapshot that day (at least ``MIN_TICKERS``), averaged over days,
  with a Newey-West t that allows for the overlap of the 20-session windows; also by halves of the period and
  as mean excess return by tone bucket.

Acceptance rule, fixed in October 2026 before any history existed (decision_log.md D-26). The agent gets timing
weight only if all of these hold, and otherwise stays at 0:
1. At least ``MIN_DAYS`` snapshot days with ``MIN_TICKERS`` or more tickers each (about six months of trading).
2. Mean rank IC at 20 sessions above 0 with Newey-West t of at least 2.
3. Mean IC at 20 sessions above 0 in both halves of the period.
4. Mean IC at 5 sessions not negative (the signal should not flip sign with the horizon).
Even then it is a timing weight: news tone never votes on BUY / HOLD / SELL.

    python -m core.sentiment_validation        # report: analysis/output/sentiment_validation.json (git-ignored)
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings

logger = logging.getLogger(__name__)

HORIZONS = (5, 20)
MIN_DAYS = 120
MIN_TICKERS = 5
MIN_T = 2.0
BENCHMARK = "SPY"
BUCKETS = ((-1.0, -0.15, "negative"), (-0.15, 0.15, "neutral"), (0.15, 1.01, "positive"))


def load_snapshots(store=None) -> pd.DataFrame:
    """Every stored snapshot: ticker, date, score, headlines, provider_scored."""
    folder = (store or settings.STORE_DIR) / "sentiment_history"
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "date" in item and isinstance(item.get("score"), (int, float)):
                rows.append({"ticker": path.stem.upper(), "date": pd.Timestamp(item["date"]), "score": float(item["score"]),
                             "headlines": item.get("headlines"), "provider_scored": item.get("provider_scored"),
                             "social_score": (item.get("social") or {}).get("score"),
                             "social_posts": (item.get("social") or {}).get("posts")})
    return pd.DataFrame(rows, columns=["ticker", "date", "score", "headlines", "provider_scored", "social_score", "social_posts"])


def forward_excess(snaps: pd.DataFrame, closes: dict[str, pd.Series], horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Add ``fwd_<h>``: the ticker's return minus SPY's from the first close after the snapshot day to h sessions later."""
    bench = closes.get(BENCHMARK)
    out = snaps.copy()
    for h in horizons:
        out[f"fwd_{h}"] = np.nan
    if bench is None or out.empty:
        return out
    bench = bench.dropna()
    for i, r in out.iterrows():
        s = closes.get(r["ticker"])
        if s is None:
            continue
        s = s.dropna()
        pos = s.index.searchsorted(r["date"], side="right")  # first session strictly after the snapshot day
        bpos = bench.index.searchsorted(r["date"], side="right")
        for h in horizons:
            if pos + h < len(s) and bpos + h < len(bench):
                own = s.iloc[pos + h] / s.iloc[pos] - 1.0
                mkt = bench.iloc[bpos + h] / bench.iloc[bpos] - 1.0
                out.at[i, f"fwd_{h}"] = own - mkt
    return out


def daily_ic(panel: pd.DataFrame, column: str) -> pd.Series:
    """Spearman rank IC between tone and ``column`` on each day with at least MIN_TICKERS names."""
    ics: dict[pd.Timestamp, float] = {}
    for day, g in panel.dropna(subset=[column]).groupby("date"):
        if len(g) < MIN_TICKERS or g["score"].nunique() < 2 or g[column].nunique() < 2:
            continue
        ics[day] = float(g["score"].rank().corr(g[column].rank()))
    return pd.Series(ics, dtype=float).sort_index()


def _summary(ic: pd.Series, lags: int) -> dict[str, Any]:
    from core.value_study import newey_west_t

    if ic.empty:
        return {"days": 0, "mean_ic": None, "t": None}
    t = newey_west_t(ic.to_numpy(float), lags)
    return {"days": int(len(ic)), "mean_ic": round(float(ic.mean()), 4), "t": None if t != t else round(t, 2)}


def evaluate(panel: pd.DataFrame) -> dict[str, Any]:
    """IC by horizon and half, returns by tone bucket, and the verdict under the acceptance rule."""
    by_h: dict[str, Any] = {}
    for h in HORIZONS:
        ic = daily_ic(panel, f"fwd_{h}")
        mid = len(ic) // 2
        by_h[str(h)] = {**_summary(ic, max(h - 1, 0)),
                        "first_half": _summary(ic.iloc[:mid], max(h - 1, 0)),
                        "second_half": _summary(ic.iloc[mid:], max(h - 1, 0))}
    buckets = []
    for lo, hi, label in BUCKETS:
        g = panel[(panel["score"] >= lo) & (panel["score"] < hi)]
        buckets.append({"tone": label, "snapshots": int(len(g)),
                        **{f"mean_excess_{h}": (None if g[f"fwd_{h}"].dropna().empty else round(float(g[f"fwd_{h}"].mean()), 5))
                           for h in HORIZONS}})
    days = by_h["20"]["days"]
    m20, t20 = by_h["20"]["mean_ic"], by_h["20"]["t"]
    halves = [by_h["20"]["first_half"]["mean_ic"], by_h["20"]["second_half"]["mean_ic"]]
    m5 = by_h["5"]["mean_ic"]
    checks = [
        {"rule": f"At least {MIN_DAYS} days with {MIN_TICKERS}+ tickers", "ok": days >= MIN_DAYS, "value": days},
        {"rule": f"IC 20 sessions > 0 with t >= {MIN_T:.0f}", "ok": bool(m20 is not None and t20 is not None and m20 > 0 and t20 >= MIN_T),
         "value": None if m20 is None else f"{m20:+.3f} (t {t20})"},
        {"rule": "IC 20 sessions > 0 in both halves", "ok": all(h is not None and h > 0 for h in halves),
         "value": [h for h in halves]},
        {"rule": "IC 5 sessions not negative", "ok": m5 is not None and m5 >= 0, "value": m5},
    ]
    if days < MIN_DAYS:
        verdict = (f"Not enough history yet: {days} of {MIN_DAYS} days with {MIN_TICKERS}+ tickers. Keep the News Sentiment "
                   "weight at 0 and keep the daily check running (python -m core.monitor --install).")
    elif all(c["ok"] for c in checks):
        verdict = ("Every condition holds. News tone may get a small timing weight (it never votes); record the numbers in "
                   "decision_log.md before changing config/agent_weights.json.")
    else:
        verdict = "The acceptance rule is not met: keep the News Sentiment weight at 0."
    return {"by_horizon": by_h, "tone_buckets": buckets, "checks": checks, "verdict": verdict,
            "accepted": days >= MIN_DAYS and all(c["ok"] for c in checks)}


def run(closes: dict[str, pd.Series] | None = None) -> dict[str, Any]:
    snaps = load_snapshots()
    tickers = sorted(set(snaps["ticker"])) if not snaps.empty else []
    if closes is None and tickers:
        from data.fetcher import fetch_ohlcv

        start = (snaps["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        data = fetch_ohlcv(tickers + [BENCHMARK], start=start)
        closes = {t: df["close"] for t, df in data.items()}
    panel = forward_excess(snaps, closes or {})
    report = {
        "snapshots": int(len(snaps)), "tickers": len(tickers),
        "first_day": None if snaps.empty else snaps["date"].min().strftime("%Y-%m-%d"),
        "last_day": None if snaps.empty else snaps["date"].max().strftime("%Y-%m-%d"),
        "provider_scored_share": None if snaps.empty else round(float((snaps["provider_scored"].fillna(0) > 0).mean()), 3),
        **evaluate(panel),
    }
    social = panel.dropna(subset=["social_score"])  # Reddit / StockTwits readings, same rule, judged apart (weight 0 until it passes)
    if not social.empty:
        report["social"] = {"snapshots": int(len(social)), **evaluate(social.assign(score=social["social_score"]))}
    return report


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Test the stored news tone against forward returns").parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    report = run()
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "sentiment_validation.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"News tone snapshots: {report['snapshots']} from {report['tickers']} ticker(s), "
          f"{report['first_day'] or 'n/a'} to {report['last_day'] or 'n/a'}.")
    for h, r in report["by_horizon"].items():
        print(f"  IC {h:>2} sessions: {r['mean_ic'] if r['mean_ic'] is not None else 'n/a'} (t {r['t']}), {r['days']} day(s)")
    for c in report["checks"]:
        print(f"  [{'x' if c['ok'] else ' '}] {c['rule']}: {c['value']}")
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
