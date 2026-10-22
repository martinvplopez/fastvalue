"""Out-of-sample check of the timing agents at the weekly, single-ticker level.

The "pullback inside an uptrend" idea was found in-sample on SPY, QQQ, AAPL,
MSFT, JPM and XOM. This replays the point-in-time timing agents (Pullback,
Momentum, Volume) weekly on other tickers and reports, per agent, the rank
correlation (IC) between the score and the 5, 20 and 60 session forward
returns, averaged across tickers with a t-stat, plus the mean forward return
after each pullback reading. Nothing here tunes anything: it is a yes/no on
whether the pullback agent deserves weight in the timing panel.

Run: ``python -m core.timing_validation [--tickers A B C] [--years 10]``.
Default tickers: the cached price files minus the in-sample set. Report:
analysis/output/timing_validation.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import config.settings as settings
from agents.base import UNAVAILABLE
from agents.momentum_agent import MomentumAgent
from agents.pullback_agent import PullbackAgent
from agents.volume_agent import VolumeAgent
from data.fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

IN_SAMPLE = frozenset({"SPY", "QQQ", "AAPL", "MSFT", "JPM", "XOM"})
HORIZONS = (5, 20, 60)
WINDOW_BARS = 300
WARMUP_BARS = 210
STEP = 5
SETUPS = ("pullback_in_uptrend", "neutral", "extended", "trend_broken")


def default_tickers() -> list[str]:
    files = sorted(p.stem for p in settings.DATA_DIR.glob("*.parquet"))
    return [t for t in files if t.isupper() and t not in IN_SAMPLE and "_" not in t and len(t) <= 6]


def replay_ticker(ticker: str, ohlcv: pd.DataFrame, years: float | None) -> pd.DataFrame:
    """Weekly scores of the three timing agents with forward returns."""
    agents = {"pullback": PullbackAgent(), "momentum": MomentumAgent(), "volume": VolumeAgent()}
    close = ohlcv["close"].to_numpy(dtype=float)
    n = len(ohlcv)
    start = WARMUP_BARS
    if years is not None:
        cutoff = ohlcv.index[-1] - pd.Timedelta(days=int(years * 365.25))
        start = max(start, int(ohlcv.index.searchsorted(cutoff)))
    rows = []
    for i in range(start, n - max(HORIZONS), STEP):
        window = ohlcv.iloc[max(0, i - WINDOW_BARS + 1): i + 1]
        row: dict[str, Any] = {"ticker": ticker, "date": ohlcv.index[i]}
        for key, agent in agents.items():
            try:
                a = agent.evaluate(ticker, window)
            except Exception:  # noqa: BLE001
                a = agent.unavailable("agent raised")
            row[key] = a.score if a.data_quality != UNAVAILABLE else np.nan
            if key == "pullback":
                row["setup"] = a.raw_data.get("setup") if a.data_quality != UNAVAILABLE else None
        for h in HORIZONS:
            row[f"fwd_{h}"] = close[i + h] / close[i] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _work(task: tuple[str, pd.DataFrame, float | None]) -> pd.DataFrame:
    ticker, ohlcv, years = task
    return replay_ticker(ticker, ohlcv, years)


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"tickers": sorted(frame["ticker"].unique()), "evaluations": int(len(frame)), "agents": {}, "setups": {}}
    for key in ("pullback", "momentum", "volume"):
        report["agents"][key] = {}
        for h in HORIZONS:
            ics = []
            for _, g in frame.groupby("ticker"):
                g = g.dropna(subset=[key, f"fwd_{h}"])
                if len(g) >= 40 and g[key].nunique() > 1:
                    ics.append(float(g[key].rank().corr(g[f"fwd_{h}"].rank())))
            arr = np.array(ics)
            report["agents"][key][f"{h}d"] = {
                "tickers": int(len(arr)),
                "mean_ic": float(arr.mean()) if len(arr) else None,
                "t": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))) if len(arr) > 2 and arr.std(ddof=1) > 0 else None,
                "share_positive": float((arr > 0).mean()) if len(arr) else None,
            }
    base = {f"{h}d": float(frame[f"fwd_{h}"].mean()) for h in HORIZONS}
    report["baseline"] = base
    for setup in SETUPS:
        sub = frame[frame["setup"] == setup]
        if sub.empty:
            continue
        entry = {"evaluations": int(len(sub)), "share": float(len(sub) / len(frame))}
        for h in HORIZONS:
            r = sub[f"fwd_{h}"].dropna()
            # t-stat across tickers of the per-ticker mean excess return over the ticker's own baseline.
            excess = []
            for t, g in sub.groupby("ticker"):
                if len(g) >= 10:
                    excess.append(float(g[f"fwd_{h}"].mean() - frame.loc[frame["ticker"] == t, f"fwd_{h}"].mean()))
            arr = np.array(excess)
            entry[f"{h}d"] = {"mean": float(r.mean()), "hit": float((r > 0).mean()), "excess_vs_ticker": float(arr.mean()) if len(arr) else None,
                              "t": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))) if len(arr) > 2 and arr.std(ddof=1) > 0 else None}
        report["setups"][setup] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Out-of-sample validation of the timing agents")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--years", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    tickers = [t.upper() for t in (args.tickers or default_tickers())]
    data = fetch_ohlcv(tickers)
    tasks = [(t, data[t], args.years) for t in tickers if t in data and len(data[t]) > WARMUP_BARS + max(HORIZONS) + 50]
    print(f"Replaying {len(tasks)} tickers weekly over the last {args.years:g} years: {', '.join(t for t, _, _ in tasks)}")
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_work, task) for task in tasks]):
            frames.append(fut.result())
    frame = pd.concat(frames, ignore_index=True)
    report = summarize(frame)
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    report["in_sample_excluded"] = sorted(IN_SAMPLE)

    print(f"\n{len(frame):,} weekly evaluations on {len(report['tickers'])} tickers (in-sample set excluded).\n")
    print(f"{'Agent':<12}" + "".join(f"{'IC ' + str(h) + 'd':>10}{'t':>6}{'pos':>6}" for h in HORIZONS))
    for key, res in report["agents"].items():
        line = f"{key:<12}"
        for h in HORIZONS:
            r = res[f"{h}d"]
            line += f"{(r['mean_ic'] if r['mean_ic'] is not None else float('nan')):>+10.3f}{(r['t'] or float('nan')):>6.1f}{(r['share_positive'] or 0):>6.0%}"
        print(line)
    print(f"\n{'Pullback reading':<22}{'share':>7}" + "".join(f"{str(h) + 'd mean':>10}{'excess':>9}{'t':>6}" for h in HORIZONS))
    print(f"{'all evaluations':<22}{'100%':>7}" + "".join(f"{report['baseline'][f'{h}d']:>10.2%}{'':>9}{'':>6}" for h in HORIZONS))
    for setup, res in report["setups"].items():
        line = f"{setup:<22}{res['share']:>7.0%}"
        for h in HORIZONS:
            r = res[f"{h}d"]
            line += f"{r['mean']:>10.2%}{(r['excess_vs_ticker'] if r['excess_vs_ticker'] is not None else float('nan')):>+9.2%}{(r['t'] or float('nan')):>6.1f}"
        print(line)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (settings.OUTPUT_DIR / "timing_validation.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
