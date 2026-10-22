"""Backtest the timing panel's composite score.

Replays point-in-time agents (those whose ``evaluate`` depends only on the
OHLCV passed in: trend, volume, macro, pullback) at a chosen frequency, trades
long when their composite crosses ``entry_threshold`` and exits at
``exit_threshold``, and measures:

- strategy vs buy-and-hold performance over the same window,
- calibration: forward returns grouped by decision bucket, plus the rank
  correlation between score and forward return (information coefficient).

Agents without point-in-time history (options chains, insider filings as
served today, current fundamentals, news) are excluded rather than replayed
with today's data, which would leak the future into the past.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from agents.base import UNAVAILABLE
from agents.registry import create_agents, get_agent_classes
from agents.scoring import aggregate, decide
from analysis.metrics import compute_buy_and_hold_returns, compute_metrics
from config.settings import INITIAL_CASH
from config.weights import ScoringConfig, load_scoring_config
from core.engine import run_backtest

logger = logging.getLogger(__name__)

FREQUENCIES: dict[str, int] = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}
WARMUP_BARS = 200
WINDOW_BARS = 300
MAX_EVALUATIONS = 3000


def _is_timing(key: str, cls: Any, config: ScoringConfig) -> bool:
    """Timing-panel agent in the user's roles (the score backtest replays only the timing panel)."""
    return config.roles.get(key, cls.role) == "timing"


def backtestable_agents(config: ScoringConfig | None = None) -> list[str]:
    config = config or load_scoring_config()
    return [k for k, cls in get_agent_classes().items()
            if cls.point_in_time and config.is_enabled(k) and _is_timing(k, cls, config)]


def _series_points(s: pd.Series, digits: int = 2) -> list[dict[str, Any]]:
    return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), digits)} for d, v in s.items() if pd.notna(v)]


def _calibration(scores: pd.DataFrame, close: pd.Series, horizon: int, thresholds: dict[str, float]) -> dict[str, Any]:
    idx = close.index
    pos = idx.get_indexer(scores.index)
    fwd = np.full(len(pos), np.nan)
    valid = pos + horizon < len(close)
    fwd[valid] = close.values[pos[valid] + horizon] / close.values[pos[valid]] - 1.0
    frame = scores.assign(forward_return=fwd).dropna(subset=["forward_return"])

    buckets = []
    for decision in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"):
        sub = frame[frame["decision"] == decision]["forward_return"]
        buckets.append({
            "decision": decision,
            "count": int(len(sub)),
            "mean_forward_return": round(float(sub.mean()), 4) if len(sub) else None,
            "median_forward_return": round(float(sub.median()), 4) if len(sub) else None,
            "hit_rate": round(float((sub > 0).mean()), 3) if len(sub) else None,
        })

    ic = None
    if len(frame) >= 10 and frame["composite"].nunique() > 1:
        ic = float(frame["composite"].rank().corr(frame["forward_return"].rank()))

    return {
        "horizon_bars": horizon,
        "observations": int(len(frame)),
        "baseline_mean_forward_return": round(float(frame["forward_return"].mean()), 4) if len(frame) else None,
        "baseline_hit_rate": round(float((frame["forward_return"] > 0).mean()), 3) if len(frame) else None,
        "information_coefficient": round(ic, 3) if ic is not None else None,
        "buckets": buckets,
        "thresholds": thresholds,
    }


def _trades(close: pd.Series, entries: pd.Series, exits: pd.Series) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    open_at: pd.Timestamp | None = None
    for date in close.index:
        if open_at is None and entries.get(date, False):
            open_at = date
        elif open_at is not None and exits.get(date, False):
            trades.append({"entry": open_at, "exit": date})
            open_at = None
    if open_at is not None:
        trades.append({"entry": open_at, "exit": None})
    out = []
    for t in trades:
        entry_px = float(close.loc[t["entry"]])
        exit_date = t["exit"] or close.index[-1]
        exit_px = float(close.loc[exit_date])
        out.append({
            "entry_date": t["entry"].strftime("%Y-%m-%d"),
            "entry_price": round(entry_px, 2),
            "exit_date": exit_date.strftime("%Y-%m-%d") if t["exit"] is not None else None,
            "exit_price": round(exit_px, 2),
            "return": round(exit_px / entry_px - 1.0, 4),
            "bars_held": int(close.index.get_loc(exit_date) - close.index.get_loc(t["entry"])),
            "open": t["exit"] is None,
        })
    return out


def resolve_agents(agent_keys: list[str] | None, config: ScoringConfig) -> tuple[list[str], list[dict[str, str]]]:
    """Split requested agents into replayable keys and excluded ones (with reasons)."""
    classes = get_agent_classes()
    requested = agent_keys if agent_keys is not None else list(classes)
    excluded: list[dict[str, str]] = []
    keys: list[str] = []
    for k in requested:
        cls = classes.get(k)
        if cls is None:
            excluded.append({"agent": k, "reason": "Unknown agent"})
        elif not cls.point_in_time:
            excluded.append({"agent": cls.name, "reason": "No point-in-time history; replaying today's data would leak the future"})
        elif not config.is_enabled(k):
            excluded.append({"agent": cls.name, "reason": "Disabled in weights config"})
        elif not _is_timing(k, cls, config):
            excluded.append({"agent": cls.name, "reason": "Not a timing agent in your weights; this backtest replays the entry timing agents (timing panel)"})
        else:
            keys.append(k)
    if not keys:
        raise ValueError("No point-in-time agents selected")
    return keys, excluded


def replay_composite_scores(
    ticker: str,
    ohlcv: pd.DataFrame,
    keys: list[str],
    config: ScoringConfig,
    positions: list[int],
) -> pd.DataFrame:
    """
    Composite score at each bar position, using only data up to that bar.

    Returns a frame indexed by date with composite, decision, regime and one
    ``agent_<key>`` column per agent (normalized score, NaN when excluded).
    """
    classes = get_agent_classes()
    agents = create_agents(keys)
    scales = {k: classes[k].score_scale for k in keys}
    records: list[dict[str, Any]] = []
    for i in positions:
        window = ohlcv.iloc[max(0, i - WINDOW_BARS + 1): i + 1]
        assessments = []
        for agent in agents:
            try:
                assessments.append(agent.evaluate(ticker, window))
            except Exception as exc:  # noqa: BLE001
                assessments.append(agent.unavailable("agent raised an error", error=str(exc)))
        regime = next(
            (a.raw_data.get("regime") for a in assessments
             if classes[a.agent_key].provides_regime and a.data_quality != UNAVAILABLE),
            None,
        )
        result = aggregate(assessments, config, regime, scales)
        row = {
            "date": ohlcv.index[i],
            "composite": result.composite_score,
            "decision": decide(result.composite_score, config.thresholds),
            "regime": result.regime,
        }
        for c in result.contributions:
            row[f"agent_{c.agent_key}"] = c.normalized_score if c.included else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("date")


def run_score_backtest(
    ticker: str,
    ohlcv: pd.DataFrame,
    agent_keys: list[str] | None = None,
    frequency: str = "weekly",
    entry_threshold: float | None = None,
    exit_threshold: float = 0.0,
    forward_days: int = 20,
    years: float | None = None,
    config: ScoringConfig | None = None,
) -> dict[str, Any]:
    """
    Replay the composite score and backtest threshold-based entries/exits.

    Args:
        ticker: Symbol.
        ohlcv: Full daily OHLCV history (completed sessions).
        agent_keys: Agents to include (non point-in-time ones are dropped with a note).
        frequency: daily, weekly, biweekly or monthly evaluation cadence.
        entry_threshold: Enter long when composite >= this (default: config "buy").
        exit_threshold: Exit when composite <= this.
        forward_days: Horizon (bars) for the calibration table.
        years: Only evaluate the last N years (None = all history after warmup).
        config: Scoring config (weights, thresholds); defaults to the saved one.
    """
    config = config or load_scoring_config()
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {sorted(FREQUENCIES)}")
    step = FREQUENCIES[frequency]
    entry = float(entry_threshold if entry_threshold is not None else config.thresholds["buy"])
    exit_ = float(exit_threshold)
    if exit_ >= entry:
        raise ValueError("exit_threshold must be below entry_threshold")

    keys, excluded = resolve_agents(agent_keys, config)
    classes = get_agent_classes()
    requested = agent_keys if agent_keys is not None else list(classes)

    n = len(ohlcv)
    start_i = WARMUP_BARS
    if years is not None:
        cutoff = ohlcv.index[-1] - pd.Timedelta(days=int(years * 365.25))
        start_i = max(start_i, int(ohlcv.index.searchsorted(cutoff)))
    if n - start_i < 20:
        raise ValueError(f"Need at least {WARMUP_BARS + 20} bars of history; got {n}")

    eval_positions = list(range(start_i, n, step))
    if eval_positions[-1] != n - 1:
        eval_positions.append(n - 1)
    if len(eval_positions) > MAX_EVALUATIONS:
        raise ValueError(f"Too many evaluations ({len(eval_positions)}); use a lower frequency or fewer years")

    scores = replay_composite_scores(ticker, ohlcv, keys, config, eval_positions)
    close = ohlcv["close"]
    window_ohlcv = ohlcv.loc[scores.index[0]:]

    # Position state machine on evaluation dates.
    raw_entries = pd.Series(False, index=window_ohlcv.index)
    raw_exits = pd.Series(False, index=window_ohlcv.index)
    in_position = False
    for date, value in scores["composite"].items():
        if not in_position and value >= entry:
            raw_entries.loc[date] = True
            in_position = True
        elif in_position and value <= exit_:
            raw_exits.loc[date] = True
            in_position = False

    # Trade on the bar after the signal (no same-bar execution).
    signals = pd.DataFrame({
        "entries": raw_entries.shift(1, fill_value=False).astype(bool),
        "exits": raw_exits.shift(1, fill_value=False).astype(bool),
    })
    result = run_backtest(window_ohlcv, signals)
    strategy_stats = compute_metrics(result.returns, signals=signals)

    bench_returns = compute_buy_and_hold_returns(window_ohlcv).reindex(result.returns.index).fillna(0.0)
    bench_stats = compute_metrics(bench_returns)

    position = signals["entries"].astype(int) - signals["exits"].astype(int)
    exposure = float((position.cumsum().clip(0, 1)).mean())

    equity = (1 + result.returns).cumprod() * INITIAL_CASH
    bench_equity = (1 + bench_returns).cumprod() * INITIAL_CASH

    agent_series = {
        k: _series_points(scores[f"agent_{k}"], 1) for k in keys if f"agent_{k}" in scores
    }

    notes = []
    if len(keys) < len(requested):
        notes.append(
            "Only point-in-time agents are replayed: this tests entry timing (trend, volume, macro, pullback), "
            "which does not vote on the owner decision. The owner layer is tested with SEC data in core/value_study.py."
        )

    return {
        "ticker": ticker.upper(),
        "frequency": frequency,
        "entry_threshold": entry,
        "exit_threshold": exit_,
        "agents_used": [{"key": k, "name": classes[k].name} for k in keys],
        "agents_excluded": excluded,
        "evaluations": len(scores),
        "date_range": {
            "start": scores.index[0].strftime("%Y-%m-%d"),
            "end": scores.index[-1].strftime("%Y-%m-%d"),
        },
        "strategy_stats": {**strategy_stats, "exposure": round(exposure, 3)},
        "benchmark_stats": bench_stats,
        "equity": _series_points(equity),
        "benchmark_equity": _series_points(bench_equity),
        "score_series": _series_points(scores["composite"], 1),
        "agent_series": agent_series,
        "regime_series": [
            {"time": d.strftime("%Y-%m-%d"), "regime": r} for d, r in scores["regime"].items()
        ],
        "trades": _trades(close.loc[window_ohlcv.index], signals["entries"], signals["exits"]),
        "calibration": _calibration(scores, close, forward_days, config.thresholds),
        "notes": notes,
    }
