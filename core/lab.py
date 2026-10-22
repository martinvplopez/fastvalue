"""FastValue backtest lab: your setup on one ticker or your whole portfolio, against the alternatives.

It answers two questions:

- "If I had followed FastValue with my weights on these holdings, would I have
  beaten buy and hold, the market and the classic strategies?" (``run_lab``)
- "Which setup would have done best, and does that survive out of sample?"
  (``run_setup_search``)

Two stages keep batches cheap:

1. Score panel (slow, cached). Every agent that can be replayed point in time,
   scored at each month-end for each ticker: the owner agents (Fundamentals,
   Shareholders) from the 10-K facts filed by that date (``core.value_study``),
   the timing agents (Momentum, Volume, Pullback) from the prices that traded
   (``core.all_agents_study``), and the Macro agent once per date. Insider,
   Options and News Sentiment have no point-in-time history here and are left
   out. Scores are cached per ticker under data/cache/lab/<code hash>/, so a
   change to an agent rescores instead of serving stale numbers.
2. Setups (fast). Roles, weights, thresholds and the margin-of-safety cap are
   applied to the stored scores with the arithmetic of ``agents/scoring.py``
   (``tests/test_lab.py`` checks it against ``aggregate_layers``), then traded
   month by month.

Trading rules (the portfolio review's, decision_log.md D-12): each holding has
a slot sized by its allocation weight. A slot is invested while the owner
decision is BUY or better (entry) and goes to cash on SELL or worse; HOLD keeps
what the slot was doing (a buy/hold band keeps turnover low). Tickers the
framework does not rate (funds, indexes, utilities, REITs, no filings yet) stay
invested, as in the review. An optional entry gate uses the timing panel.
Cash earns 3-month T-bills. Slots are rebalanced to the target weights on the
chosen schedule and every trade costs ``cost_bps``. Decisions use month-end
data and trade at that month-end close, the convention of the value study.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import config.settings as settings
from agents.base import OWNER, TIMING
from agents.registry import get_agent_classes
from agents.scoring import NOT_RATED, TRADING_VOTE_EVIDENCE
from analysis.fundamental_metrics import is_excluded_sector
from config.weights import ScoringConfig, load_scoring_config, registered_roles
from strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

Progress = Callable[[int, int, str], None]

PANEL_START = pd.Timestamp("2011-01-31")  # first month-end with XBRL 10-K facts for most filers (value study)
PRICE_START = "2008-06-01"  # timing agents need about 210 bars before the first month-end
COST_BPS = 10.0
MAX_TICKERS = 40
MAX_SETUPS = 1000
BENCHMARKS = ("SPY", "RSP")
BUYS = ("BUY", "STRONG_BUY")
SELLS = ("SELL", "STRONG_SELL")

# Agent key -> (score column, confidence column) in the score panel. Agents not listed are not replayed.
AGENT_COLUMNS: dict[str, tuple[str, str]] = {
    "fundamentals": ("fund_score", "fund_conf"),
    "shareholder": ("sh_score", "sh_conf"),
    "momentum": ("mom_score", "mom_conf"),
    "volume": ("vol_score", "vol_conf"),
    "pullback": ("pb_score", "pb_conf"),
    "macro": ("macro_score", "macro_conf"),
}
OWNER_COLUMNS = ("sector", "is_financial", "fund_score", "fund_conf", "sh_score", "sh_conf", "ev_to_fcf", "fcf")
TIMING_COLUMNS = ("mom_score", "mom_conf", "vol_score", "vol_conf", "pb_score", "pb_conf", "pb_setup", "trend")
MACRO_COLUMNS = ("macro_score", "macro_conf", "regime")

REBALANCE: dict[str, tuple[int, ...]] = {
    "never": (),
    "yearly": (4,),  # April, after most 10-Ks are out (value study)
    "quarterly": (1, 4, 7, 10),
    "monthly": tuple(range(1, 13)),
}
ENTRY_GATES: dict[str, str] = {
    "none": "No entry gate",
    "not_unfavorable": "Entry timing score above its exit level",
    "favorable": "Entry timing score at or above its entry level",
    "not_extended": "Price not extended (Pullback agent)",
}
CLASSIC_LABELS: dict[str, str] = {
    "sma_crossover": "SMA 50/200 crossover",
    "rsi": "RSI mean reversion",
    "macd": "MACD crossover",
    "bollinger_bands": "Bollinger bands",
    "momentum": "Price momentum",
    "options_flow": "Volume surge (price proxy)",
    "insider_buys": "Dip recovery (price proxy)",
    "multi_factor": "Price and volume confluence",
    "trend_filter": "Price above its 200-day average",
    "pb_ema": "PB EMA channel",
    "abs_momentum": "Absolute momentum, 12 months",
    "msci_momentum": "MSCI momentum on one asset",
    "dip_rsi2": "Short dip in an uptrend (RSI 2)",
    "dip_drawdown": "Buy 10% below the 1-year high",
}

# Files whose code decides the stored scores: any change rescores the cache.
_SCORE_CODE = (
    "agents/base.py", "agents/fundamental_agent.py", "agents/shareholder_agent.py", "agents/momentum_agent.py",
    "agents/volume_agent.py", "agents/pullback_agent.py", "agents/regime_agent.py", "analysis/fundamental_metrics.py",
    "data/sec_edgar.py", "data/financial_statements.py", "data/fundamentals.py", "data/macro_data.py",
    "core/value_study.py", "core/all_agents_study.py", "config/settings.py",
)


@dataclass(frozen=True)
class Rules:
    """How decisions become trades (everything that is not a scoring setting)."""

    entry_gate: str = "none"
    start: str = "invested"  # "invested": slots start in the stock; "cash": they wait for the entry level
    rebalance: str = "yearly"
    cost_bps: float = COST_BPS

    def validate(self) -> "Rules":
        if self.entry_gate not in ENTRY_GATES:
            raise ValueError(f"entry_gate must be one of {', '.join(ENTRY_GATES)}")
        if self.start not in ("invested", "cash"):
            raise ValueError("start must be 'invested' or 'cash'")
        if self.rebalance not in REBALANCE:
            raise ValueError(f"rebalance must be one of {', '.join(REBALANCE)}")
        if not 0.0 <= float(self.cost_bps) <= 200.0:
            raise ValueError("cost_bps must be between 0 and 200")
        return self


# ================================================================== score panel (slow, cached)


def scores_version() -> str:
    h = hashlib.sha256()
    for rel in _SCORE_CODE:
        path = settings.PROJECT_ROOT / rel
        h.update(rel.encode())
        h.update(path.read_bytes() if path.exists() else b"")
    return h.hexdigest()[:12]


def _cache_dir() -> Path:
    base = settings.DATA_DIR / "lab"
    path = base / scores_version()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        for old in base.iterdir():  # scores from older agent code are never reused
            if old.is_dir() and old != path:
                shutil.rmtree(old, ignore_errors=True)
    return path


def _read_part(name: str) -> tuple[pd.DataFrame, set[pd.Timestamp], dict[str, Any]]:
    base = _cache_dir()
    meta_path = base / f"{name}.json"
    if not meta_path.exists():
        return pd.DataFrame(), set(), {}
    meta = json.loads(meta_path.read_text())
    frame_path = base / f"{name}.parquet"
    frame = pd.read_parquet(frame_path) if frame_path.exists() else pd.DataFrame()
    return frame, {pd.Timestamp(d) for d in meta.get("dates", [])}, meta


def _write_part(name: str, frame: pd.DataFrame, dates: set[pd.Timestamp], **meta: Any) -> None:
    base = _cache_dir()
    if not frame.empty:
        frame.to_parquet(base / f"{name}.parquet", index=False)
    (base / f"{name}.json").write_text(json.dumps({"dates": sorted(d.strftime("%Y-%m-%d") for d in dates), **meta}))


def _safe(ticker: str) -> str:
    return ticker.upper().replace("/", "_").replace("^", "_")


def _fred(series: str) -> pd.Series:
    """FRED series cached for a day; a stale copy is used when FRED is unreachable."""
    from core.value_study import load_fred

    path = settings.DATA_DIR / "lab" / f"fred_{series}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < 86400:
        return pd.read_parquet(path)["value"]
    try:
        s = load_fred(series, "2008-01-01")
        path.parent.mkdir(parents=True, exist_ok=True)
        s.rename("value").to_frame().to_parquet(path)
        return s
    except Exception as exc:  # noqa: BLE001 - rates are optional; cash then earns 0
        logger.warning("FRED %s unavailable (%s)", series, exc)
        return pd.read_parquet(path)["value"] if path.exists() else pd.Series(dtype=float)


def _score_task(task: tuple) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Owner and timing agents for one ticker at the given month-ends (worker process)."""
    from core.all_agents_study import _score_timing
    from core.value_study import _score_ticker

    ticker, cik, predecessors, owner_dates, timing_dates, treasury, max_age = task
    owner_rows: list[dict[str, Any]] = []
    status: str | None = "no SEC filings found for this ticker"
    if cik is not None and owner_dates:
        rows, stats = _score_ticker((ticker, cik, predecessors, owner_dates, treasury, max_age))
        owner_rows = [{"date": r["date"], **{c: r.get(c) for c in OWNER_COLUMNS}} for r in rows]
        status = ("no SEC access: set SEC_USER_AGENT to download filings" if stats.get("no_sec_access")
                  else "ok" if owner_rows else "no usable 10-K data")
    elif cik is not None:
        status = None  # nothing new to score: keep the cached status
    timing_rows = [{"date": r["date"], **{c: r.get(c) for c in TIMING_COLUMNS}} for r in _score_timing((ticker, timing_dates))] \
        if timing_dates else []
    return ticker, owner_rows, timing_rows, status


def _macro_rows(dates: list[pd.Timestamp]) -> pd.DataFrame:
    from core.all_agents_study import macro_by_date

    return macro_by_date(dates)


def build_panel(tickers: list[str], dates: list[pd.Timestamp], progress: Progress | None = None,
                workers: int | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Score panel (one row per ticker and month-end) plus each ticker's owner-data status."""
    from core.value_study import CIK_OVERRIDES, PREDECESSOR_CIKS
    from data.sec_edgar import CACHE_MAX_AGE_DAYS, SecConfigError, SecDataError, has_sec_access, load_ticker_map

    progress = progress or (lambda *_: None)
    has_access = has_sec_access()
    max_age = CACHE_MAX_AGE_DAYS if has_access else math.inf  # without access, reuse cached filings
    try:
        ticker_map = load_ticker_map(max_age_days=max_age)
    except (SecConfigError, SecDataError, OSError) as exc:
        logger.warning("SEC ticker map unavailable: %s", exc)
        ticker_map = {}
    treasury = _fred("DGS10")
    tre = treasury.reindex(pd.DatetimeIndex(dates), method="ffill") if len(treasury) else pd.Series(dtype=float)
    tre_map = {d: float(v) for d, v in tre.items() if pd.notna(v)}

    wanted = set(dates)
    parts: dict[str, dict[str, Any]] = {}
    tasks = []
    for t in tickers:
        owner, owner_done, owner_meta = _read_part(f"{_safe(t)}_owner")
        timing, timing_done, _ = _read_part(f"{_safe(t)}_timing")
        cik = CIK_OVERRIDES.get(t, ticker_map.get(t))
        parts[t] = {"owner": owner, "owner_done": owner_done, "timing": timing, "timing_done": timing_done,
                    "status": owner_meta.get("status", "ok" if cik else "no SEC filings found for this ticker")}
        owner_todo = sorted(wanted - owner_done) if cik is not None else []
        timing_todo = sorted(wanted - timing_done)
        if owner_todo or timing_todo:
            tasks.append((t, cik, PREDECESSOR_CIKS.get(t, ()), owner_todo, timing_todo, tre_map, max_age))

    def absorb(result: tuple[str, list[dict[str, Any]], list[dict[str, Any]], str | None], task: tuple) -> None:
        t, owner_rows, timing_rows, status = result
        p = parts[t]
        if status is None:
            status = p["status"]
        elif owner_rows or status != "no SEC access: set SEC_USER_AGENT to download filings":
            # Dates without a 10-K yet are done too; only a missing SEC connection is retried next time.
            p["owner"] = pd.concat([p["owner"], pd.DataFrame(owner_rows)], ignore_index=True) if owner_rows else p["owner"]
            if not p["owner"].empty:
                status = "ok"  # a month without a row does not undo earlier ones
            p["owner_done"] |= set(task[3])
            _write_part(f"{_safe(t)}_owner", p["owner"], p["owner_done"], status=status)
        p["status"] = status
        if task[4]:
            p["timing"] = pd.concat([p["timing"], pd.DataFrame(timing_rows)], ignore_index=True) if timing_rows else p["timing"]
            p["timing_done"] |= set(task[4])
            _write_part(f"{_safe(t)}_timing", p["timing"], p["timing_done"])

    total = len(tasks) + 1
    progress(0, total, f"Scoring {len(tasks)} ticker{'' if len(tasks) == 1 else 's'} for the first time" if tasks else "Scores ready")
    if workers is None:
        workers = min(len(tasks), max(1, (os.cpu_count() or 2) - 1), 6)
    if workers <= 1 or len(tasks) <= 1:
        for n, task in enumerate(tasks, 1):
            progress(n - 1, total, f"Scoring {task[0]} ({n} of {len(tasks)})")
            absorb(_score_task(task), task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_task, task): task for task in tasks}
            for n, fut in enumerate(as_completed(futures), 1):
                task = futures[fut]
                absorb(fut.result(), task)
                progress(n, total, f"Scored {task[0]} ({n} of {len(tasks)})")

    progress(len(tasks), total, "Market regime by month")
    macro, macro_done, _ = _read_part("macro")
    todo = sorted(wanted - macro_done)
    if todo:
        fresh = _macro_rows(todo)
        good = fresh[fresh["macro_score"].notna()] if not fresh.empty else fresh
        macro = pd.concat([macro, good], ignore_index=True) if not good.empty else macro
        macro_done |= {pd.Timestamp(d) for d in good["date"]} if not good.empty else set()
        _write_part("macro", macro, macro_done)
    progress(total, total, "Scores ready")

    frames = []
    for t, p in parts.items():
        owner = p["owner"] if not p["owner"].empty else pd.DataFrame(columns=["date", *OWNER_COLUMNS])
        timing = p["timing"] if not p["timing"].empty else pd.DataFrame(columns=["date", *TIMING_COLUMNS])
        for f in (owner, timing):
            f["date"] = pd.to_datetime(f["date"])
        merged = pd.merge(owner, timing, on="date", how="outer")
        merged = merged[merged["date"].isin(wanted)]
        merged["ticker"] = t
        frames.append(merged)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker"])
    if not macro.empty:
        macro = macro.copy()
        macro["date"] = pd.to_datetime(macro["date"])
        panel = panel.merge(macro[["date", *MACRO_COLUMNS]], on="date", how="left")
    for col in (*OWNER_COLUMNS, *TIMING_COLUMNS, *MACRO_COLUMNS):
        if col not in panel:
            panel[col] = np.nan
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel, {t: p["status"] for t, p in parts.items()}


# ================================================================== setups (fast)


def _class_roles() -> dict[str, str]:
    return registered_roles()


def _scales(config: ScoringConfig) -> dict[str, float]:
    out = {key: cls.score_scale for key, cls in get_agent_classes().items()}
    return {**out, **config.score_scales}


def _layer(panel: pd.DataFrame, keys: list[str], weights: np.ndarray, config: ScoringConfig,
           scales: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Weighted composite per row and whether any agent counted (agents/scoring.py ``aggregate``)."""
    n = len(panel)
    if not keys:
        return np.zeros(n), np.zeros(n, dtype=bool)
    scores = np.column_stack([pd.to_numeric(panel[AGENT_COLUMNS[k][0]], errors="coerce").to_numpy(float) for k in keys])
    conf = np.column_stack([pd.to_numeric(panel[AGENT_COLUMNS[k][1]], errors="coerce").fillna(0.0).to_numpy(float)
                            for k in keys])
    enabled = np.array([config.is_enabled(k) for k in keys], dtype=bool)
    w = weights * enabled
    included = (w > 0) & ~np.isnan(scores)
    scale = np.array([max(scales.get(k, 50.0), 1e-6) for k in keys])
    normalized = np.tanh(np.nan_to_num(scores) / scale) * 100.0
    floor = config.confidence_floor
    # The engine rounds each effective weight to 4 decimals before averaging; so does this, to match it exactly.
    eff = np.where(included, np.round(w * (floor + (1.0 - floor) * np.clip(conf, 0.0, 100.0) / 100.0), 4), 0.0)
    total = eff.sum(axis=1)
    composite = np.where(total > 0, (normalized * eff).sum(axis=1) / np.where(total > 0, total, 1.0), 0.0)
    return np.clip(composite, -100.0, 100.0), included.any(axis=1)


def _bands(score: np.ndarray, th: dict[str, float]) -> np.ndarray:
    return np.select(
        [score >= th["strong_buy"], score >= th["buy"], score <= th["strong_sell"], score <= th["sell"]],
        ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"], default="HOLD").astype(object)


def evaluate_setup(panel: pd.DataFrame, config: ScoringConfig, class_roles: dict[str, str] | None = None) -> pd.DataFrame:
    """Owner score and decision, timing score and label for every panel row under ``config``."""
    class_roles = class_roles if class_roles is not None else _class_roles()
    roles = {**class_roles, **config.roles}
    scales = _scales(config)
    owner_keys = [k for k in AGENT_COLUMNS if roles.get(k, TIMING) == OWNER]
    timing_keys = [k for k in AGENT_COLUMNS if roles.get(k, TIMING) == TIMING]

    owner_w = np.tile(np.array([config.owner_weights.get(k, 0.0) for k in owner_keys], dtype=float), (len(panel), 1))
    owner_score, owner_any = _layer(panel, owner_keys, owner_w, config, scales)

    regimes = [r if isinstance(r, str) else None for r in panel["regime"].tolist()]
    profiles = {r: config.weights_for(r) for r in set(regimes)}
    timing_w = np.array([[profiles[r].get(k, 0.0) for k in timing_keys] for r in regimes], dtype=float) \
        if timing_keys else np.zeros((len(panel), 0))
    timing_score, timing_any = _layer(panel, timing_keys, timing_w, config, scales)

    excluded = panel["sector"].map(lambda s: is_excluded_sector(s) if isinstance(s, str) else False).to_numpy(bool)
    applicable = ~excluded & owner_any
    band = _bands(owner_score, config.owner_thresholds)
    cap = config.owner_max_ev_to_fcf
    if cap is not None:
        evf = pd.to_numeric(panel["ev_to_fcf"], errors="coerce").to_numpy(float)
        fcf = pd.to_numeric(panel["fcf"], errors="coerce").to_numpy(float)
        has_fund = pd.to_numeric(panel["fund_score"], errors="coerce").notna().to_numpy()
        financial = np.array([bool(v) if isinstance(v, (bool, np.bool_, int, float)) and v == v else False
                              for v in panel["is_financial"].tolist()], dtype=bool)
        with np.errstate(invalid="ignore"):
            expensive = (evf >= cap) | (np.isnan(evf) & (fcf <= 0))
        capped = applicable & np.isin(band, BUYS) & has_fund & ~financial & expensive
        band = np.where(capped, "HOLD", band)
    th = config.thresholds
    shown = np.array([round(float(x), 1) for x in timing_score])  # the engine labels the rounded timing score
    label = np.where(~timing_any, "NEUTRAL",
                     np.where(shown >= th["buy"], "FAVORABLE", np.where(shown <= th["sell"], "UNFAVORABLE", "NEUTRAL")))
    return pd.DataFrame({
        "date": panel["date"].to_numpy(), "ticker": panel["ticker"].to_numpy(),
        "owner_score": np.where(applicable, owner_score, np.nan),
        "decision": np.where(applicable, band, NOT_RATED),
        "timing_score": np.where(timing_any, timing_score, np.nan),
        "timing_label": label,
        "setup": panel["pb_setup"].to_numpy(),
    }, index=panel.index)


def positions(decided: pd.DataFrame, dates: list[pd.Timestamp], tickers: list[str], rules: Rules) -> pd.DataFrame:
    """Whether each slot holds its stock for the month after each date (True) or sits in cash."""
    wide = {col: decided.pivot(index="date", columns="ticker", values=col).reindex(index=dates, columns=tickers)
            for col in ("decision", "timing_label", "setup")}
    dec, lab, setup = (wide[c].to_numpy(object) for c in ("decision", "timing_label", "setup"))
    out = np.zeros((len(dates), len(tickers)), dtype=bool)
    held = np.full(len(tickers), rules.start == "invested")
    for i in range(len(dates)):
        for j in range(len(tickers)):
            d = dec[i, j]
            if d == NOT_RATED:
                held[j] = True
            elif d in SELLS:
                held[j] = False
            elif d in BUYS and not held[j]:
                gate = rules.entry_gate
                held[j] = (gate == "none" or (gate == "not_unfavorable" and lab[i, j] != "UNFAVORABLE")
                           or (gate == "favorable" and lab[i, j] == "FAVORABLE")
                           or (gate == "not_extended" and setup[i, j] != "extended"))
            # HOLD, or no score this month: keep what the slot was doing.
            out[i, j] = held[j]
    return pd.DataFrame(out, index=dates, columns=tickers)


def simulate(fwd: pd.DataFrame, invested: pd.DataFrame | None, weights: pd.Series, cash: pd.Series,
             rebalance: str, cost_bps: float, start_invested: bool = True) -> dict[str, Any]:
    """Monthly returns of slot portfolio. ``fwd`` holds each ticker's return over the month after each date."""
    dates = list(fwd.index)
    w = weights.reindex(fwd.columns).fillna(0.0).to_numpy(float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / max(len(w), 1))
    F = fwd.to_numpy(float)
    I = np.ones_like(F, dtype=bool) if invested is None else invested.reindex(index=fwd.index, columns=fwd.columns).fillna(False).to_numpy(bool)
    cash_r = cash.reindex(fwd.index).fillna(0.0).to_numpy(float)
    months = REBALANCE[rebalance]
    slots = w.copy()
    held = np.full(len(w), start_invested)
    rets, traded, invested_share = [], 0.0, 0.0
    for i, d in enumerate(dates):
        total = slots.sum()
        trade = np.zeros(len(w))
        if i > 0 and d.month in months:
            target = w * total
            trade += np.abs(target - slots)
            slots = target
        want = I[i] & ~np.isnan(F[i])
        trade += np.where(want != held, slots, 0.0)
        held = want
        slots = slots - trade * cost_bps / 10_000.0
        invested_share += slots[held].sum() / max(slots.sum(), 1e-12)
        slots = slots * (1.0 + np.where(held, np.nan_to_num(F[i]), cash_r[i]))
        rets.append(slots.sum() / total - 1.0)
        traded += trade.sum() / total
    years = max(len(dates) / 12.0, 1e-9)
    return {"returns": pd.Series(rets, index=fwd.index, dtype=float), "turnover": traded / years,
            "invested_share": invested_share / max(len(dates), 1)}


def classic_monthly(prices: pd.DataFrame, key: str, level_dates: list[pd.Timestamp], tbill_daily: pd.Series,
                    cost_bps: float) -> tuple[pd.Series, float, float]:
    """A Strategy Lab strategy on one ticker as month-to-month returns (cash earns T-bills), with its trades a
    year and the share of days it held the stock."""
    entry = STRATEGY_REGISTRY[key]
    ohlcv = prices[["open", "high", "low", "close", "volume"]]
    signals = entry["class"](**entry["defaults"]).generate_signals(ohlcv)
    entries = signals["entries"].reindex(ohlcv.index).fillna(False).astype(bool)
    exits = signals["exits"].reindex(ohlcv.index).fillna(False).astype(bool)
    pos = pd.Series(np.where(exits, 0.0, np.where(entries, 1.0, np.nan)), index=ohlcv.index).ffill().fillna(0.0)
    held_before = pos.shift(1).fillna(0.0)  # a signal trades at that day's close
    daily = prices["adj_close"].pct_change().fillna(0.0)
    rf = (tbill_daily.reindex(ohlcv.index, method="ffill").fillna(0.0) if len(tbill_daily)
          else pd.Series(0.0, index=ohlcv.index))
    switches = pos.diff().abs().fillna(pos.iloc[0] if len(pos) else 0.0)
    growth = (1.0 + held_before * daily + (1.0 - held_before) * rf - switches * cost_bps / 10_000.0).cumprod()
    idx = pd.DatetimeIndex(level_dates)
    level = growth.reindex(growth.index.union(idx)).ffill().reindex(idx)
    fwd = (level.shift(-1) / level - 1.0).iloc[:-1]
    window = (switches.index > idx[0]) & (switches.index <= idx[-1])
    trades = float(switches[window].sum()) / max(len(fwd) / 12.0, 1e-9)
    return fwd, trades, float(held_before[window].mean()) if window.any() else float("nan")


# ================================================================== reporting


def _perf(monthly: pd.Series, rf: pd.Series) -> dict[str, float]:
    r = monthly.dropna()
    if len(r) < 3:
        return {"cagr": float("nan"), "volatility": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan"),
                "total_return": float("nan")}
    rf = rf.reindex(r.index).fillna(0.0)
    eq = (1.0 + r).cumprod()
    years = len(r) / 12.0
    sd = r.std()
    return {
        "cagr": float(eq.iloc[-1] ** (1.0 / years) - 1.0),
        "volatility": float(sd * math.sqrt(12)),
        "sharpe": float((r - rf).mean() / sd * math.sqrt(12)) if sd > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
        "total_return": float(eq.iloc[-1] - 1.0),
    }


def _halves(monthly: pd.Series, rf: pd.Series) -> tuple[dict[str, float], dict[str, float]]:
    mid = len(monthly) // 2
    return _perf(monthly.iloc[:mid], rf), _perf(monthly.iloc[mid:], rf)


def _row(key: str, label: str, kind: str, sim: dict[str, Any], rf: pd.Series, base: dict[str, Any] | None,
         extra: dict[str, Any] | None = None) -> dict[str, Any]:
    stats = _perf(sim["returns"], rf)
    first, second = _halves(sim["returns"], rf)
    row = {"key": key, "label": label, "kind": kind, "stats": stats, "first_half": first, "second_half": second,
           "turnover": sim.get("turnover"), "invested_share": sim.get("invested_share"), **(extra or {})}
    if base is not None:
        row["excess_cagr"] = stats["cagr"] - base["stats"]["cagr"]
        row["excess_first"] = first["cagr"] - base["first_half"]["cagr"]
        row["excess_second"] = second["cagr"] - base["second_half"]["cagr"]
    return row


def _buy_hold_row(sim: dict[str, Any], rf: pd.Series) -> dict[str, Any]:
    plain = _row("buy_hold", "Buy and hold (same allocation)", "benchmark", sim, rf, None)
    return _row("buy_hold", "Buy and hold (same allocation)", "benchmark", sim, rf, plain)


def _finite(x: float | None) -> float:
    return x if x is not None and math.isfinite(x) else -math.inf


def _weaker(a: float, b: float) -> float:
    return min(a, b) if math.isfinite(a) and math.isfinite(b) else float("nan")


def _equity(returns: pd.Series) -> list[dict[str, Any]]:
    eq = (1.0 + returns.fillna(0.0)).cumprod() * 100.0
    start = returns.index[0] - pd.offsets.MonthEnd(1) if len(returns) else None
    points = ([{"time": start.strftime("%Y-%m-%d"), "value": 100.0}] if start is not None else [])
    return points + [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 3)} for d, v in eq.items()]


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN and infinities become None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def describe_setup(config: ScoringConfig, class_roles: dict[str, str] | None = None) -> dict[str, Any]:
    roles = {**(class_roles if class_roles is not None else _class_roles()), **config.roles}
    owner = {k: config.owner_weights.get(k, 0.0) for k in AGENT_COLUMNS if roles.get(k) == OWNER}
    return {
        "owner_weights": owner,
        "deciding_agents": [k for k, w in owner.items() if w > 0],
        "buy": config.owner_thresholds["buy"], "sell": config.owner_thresholds["sell"],
        "cap": config.owner_max_ev_to_fcf,
    }


# ================================================================== inputs


def load_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    from data.price_history import fetch_price_history

    return fetch_price_history(tickers, PRICE_START)


def load_rates(dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Monthly and daily T-bill returns (3-month bill, FRED DTB3); 0 when unavailable."""
    rate = _fred("DTB3")
    if rate.empty:
        return pd.Series(0.0, index=dates), pd.Series(dtype=float)
    monthly = rate.reindex(rate.index.union(dates)).ffill().reindex(dates).fillna(0.0) / 100.0 / 12.0
    return monthly, rate / 100.0 / 252.0


def resolve_universe(kind: str, ticker: str | None = None, allocation: str = "current",
                     custom: dict[str, float] | None = None) -> tuple[list[str], dict[str, float], str]:
    """Tickers and raw weights. Current weights need prices, so they are filled in later."""
    from data.portfolio import TICKER_RE

    if kind == "ticker":
        t = str(ticker or "").strip().upper()
        if not TICKER_RE.match(t):
            raise ValueError(f"Invalid ticker symbol: {t!r}")
        return [t], {t: 1.0}, t
    if kind != "portfolio":
        raise ValueError("universe must be 'ticker' or 'portfolio'")
    from data.portfolio import load_portfolio

    positions = load_portfolio().get("positions", [])
    if not positions:
        raise ValueError("The portfolio is empty: add positions in the Portfolio tab first")
    if len(positions) > MAX_TICKERS:
        raise ValueError(f"At most {MAX_TICKERS} holdings can be backtested at once")
    tickers = [p["ticker"] for p in positions]
    if allocation == "equal":
        weights = {t: 1.0 for t in tickers}
    elif allocation == "custom":
        custom = {str(k).upper(): v for k, v in (custom or {}).items()}
        weights = {t: max(0.0, float(custom.get(t, 0.0))) for t in tickers}
        if sum(weights.values()) <= 0:
            raise ValueError("Custom weights must add up to more than 0")
    else:
        weights = {p["ticker"]: float(p["shares"]) for p in positions}  # times the latest price, below
    return tickers, weights, "My portfolio"


@dataclass
class _Inputs:
    tickers: list[str]
    weights: pd.Series
    label: str
    dates: list[pd.Timestamp]
    fwd: pd.DataFrame
    bench_fwd: pd.DataFrame
    prices: dict[str, pd.DataFrame]
    level_dates: list[pd.Timestamp]
    cash: pd.Series  # T-bill return over the month after each date
    rf: pd.Series  # the same, indexed by period end (for Sharpe)
    tbill_daily: pd.Series
    panel: pd.DataFrame
    owner_status: dict[str, str]
    first_dates: dict[str, str | None]
    currencies: dict[str, str] | None = None  # listing currency per holding
    base_currency: str = "USD"  # the portfolio's currency, for current-value weights


def _prepare(universe: dict[str, Any], years: float | None, progress: Progress) -> _Inputs:
    from core.value_study import month_end_panel

    kind = str(universe.get("kind", "ticker"))
    allocation = str(universe.get("allocation", "current"))
    tickers, raw_weights, label = resolve_universe(kind, universe.get("ticker"), allocation, universe.get("weights"))
    progress(0, 1, "Loading prices")
    prices = load_prices(tickers + list(BENCHMARKS))
    missing = [t for t in tickers if t not in prices]
    if missing:
        raise LookupError(f"No price history for {', '.join(missing)}")
    currencies: dict[str, str] = {}
    base_currency = "USD"
    if kind == "portfolio":
        from data.fx import FxError, convert_latest, quote_currency
        from data.portfolio import load_portfolio

        base_currency = load_portfolio().get("currency", "USD")
        currencies = {t: quote_currency(t) for t in tickers}
    if kind == "portfolio" and allocation == "current":
        # Market values in one currency: a price in pence or CAD must not be weighed as if it were dollars.
        try:
            raw_weights = {t: raw_weights[t] * float(prices[t]["close"].iloc[-1])
                           * convert_latest(1.0, currencies[t], base_currency) for t in tickers}
        except FxError as exc:
            raise LookupError(f"{exc}: cannot weigh the holdings by market value") from exc
    total = sum(raw_weights.values())
    weights = pd.Series({t: raw_weights[t] / total for t in tickers})

    last = max(df.index[-1] for t, df in prices.items() if t in tickers)
    # Midnight: 365.25-day years leave a time of day (06:00, 18:00, 12:00) on every month-end, which never matched
    # the whole-day dates of the score cache, so each period re-scored every holding (and failed on its second run).
    start = PANEL_START if not years else max(PANEL_START, (last - pd.Timedelta(days=365.25 * float(years))).normalize())
    dates = [d for d in pd.date_range(start, last, freq="ME") if d < last]
    if len(dates) < 12:
        raise ValueError("Need at least 12 month-ends of history (the backtest starts in January 2011)")
    level_dates = dates + [last]
    level = month_end_panel({t: prices[t] for t in tickers}, level_dates).reindex(columns=tickers)
    fwd = (level.shift(-1) / level - 1.0).iloc[:-1]
    bench = month_end_panel({t: prices[t] for t in BENCHMARKS if t in prices}, level_dates)
    bench_fwd = (bench.shift(-1) / bench - 1.0).iloc[:-1]
    monthly_rf, daily_rf = load_rates(pd.DatetimeIndex(dates))
    rf = monthly_rf.copy()
    rf.index = pd.DatetimeIndex(level_dates[1:])

    panel, status = build_panel(tickers, dates, progress)
    first_dates = {}
    for t in tickers:
        s = fwd[t].dropna()
        first_dates[t] = s.index[0].strftime("%Y-%m-%d") if len(s) else None
    return _Inputs(tickers, weights, label, dates, fwd, bench_fwd, prices, level_dates, monthly_rf, rf, daily_rf,
                   panel, status, first_dates, currencies, base_currency)


def _period_index(inp: _Inputs) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(inp.level_dates[1:])


def _by_period_end(sim: dict[str, Any], inp: _Inputs) -> dict[str, Any]:
    out = dict(sim)
    r = sim["returns"].copy()
    r.index = _period_index(inp)
    out["returns"] = r
    return out


def _notes(inp: _Inputs, config: ScoringConfig, class_roles: dict[str, str], rules: Rules) -> list[str]:
    roles = {**class_roles, **config.roles}
    notes = []
    skipped = [k for k, r in roles.items() if k not in AGENT_COLUMNS and r in (OWNER, TIMING)
               and ((r == OWNER and config.owner_weights.get(k, 0) > 0)
                    or (r == TIMING and any(p.get(k, 0) > 0 for p in config.regime_weights.values())))]
    if skipped:
        notes.append(f"Not replayed (no point-in-time history): {', '.join(skipped)}. They count in live analysis, "
                     "not here.")
    traders = [k for k in AGENT_COLUMNS if roles.get(k) == OWNER and class_roles.get(k) != OWNER
               and config.owner_weights.get(k, 0) > 0]
    if traders:
        notes.append(f"Your setup lets {', '.join(traders)} decide. {TRADING_VOTE_EVIDENCE}")
    unrated = [t for t, s in inp.owner_status.items() if s != "ok"]
    if unrated:
        notes.append("No owner rating, so held as in buy and hold: "
                     + "; ".join(f"{t} ({inp.owner_status[t]})" for t in unrated) + ".")
    late = [f"{t} from {d[:7]}" for t, d in inp.first_dates.items() if d and d > inp.dates[0].strftime("%Y-%m-%d")]
    if late:
        notes.append("Shorter price history, the slot sits in cash until then: " + ", ".join(late) + ".")
    listed = sorted({c for c in (inp.currencies or {}).values()})
    if len(listed) > 1 or (listed and listed[0] != "USD"):
        notes.append(f"Each holding's return is in its own listing currency ({', '.join(listed)}); exchange-rate moves "
                     f"are left out and the market lines are in USD. Weights use today's market values in "
                     f"{inp.base_currency}.")
    if len(inp.tickers) > 1:
        notes.append("These are the holdings you own today. Picking them with hindsight flatters every strategy that "
                     "holds them, so compare the strategies with each other more than with the market.")
    notes.append(f"Decisions at each month-end, traded at that close; {rules.cost_bps:g} bps per trade; cash earns "
                 "3-month T-bills; rebalanced to the target weights " + ("never" if rules.rebalance == "never"
                                                                         else rules.rebalance) + ".")
    return notes


# ================================================================== entry points


def run_lab(universe: dict[str, Any], years: float | None = None, rules: Rules | None = None,
            compare: list[str] | None = None, config: ScoringConfig | None = None,
            progress: Progress | None = None) -> dict[str, Any]:
    """Your setup against buy and hold, the rebalanced allocation, SPY, RSP and the classic strategies."""
    progress = progress or (lambda *_: None)
    rules = (rules or Rules()).validate()
    config = config or load_scoring_config()
    compare = list(compare) if compare is not None else ["buy_hold", "rebalanced", "SPY", "RSP", "classic"]
    class_roles = _class_roles()
    inp = _prepare(universe, years, progress)

    decided = evaluate_setup(inp.panel, config, class_roles)
    held = positions(decided, inp.dates, inp.tickers, rules)
    start_invested = rules.start == "invested"
    ours = _by_period_end(simulate(inp.fwd, held, inp.weights, inp.cash, rules.rebalance, rules.cost_bps,
                                   start_invested), inp)
    bh = _by_period_end(simulate(inp.fwd, None, inp.weights, inp.cash, "never", rules.cost_bps), inp)
    bh_row = _buy_hold_row(bh, inp.rf)  # the reference every other row is measured against

    rows = [_row("fastvalue", "FastValue (your setup)", "agents", ours, inp.rf, bh_row), bh_row]
    equity = {"fastvalue": _equity(ours["returns"]), "buy_hold": _equity(bh["returns"])}
    if "rebalanced" in compare and rules.rebalance != "never" and len(inp.tickers) > 1:
        sim = _by_period_end(simulate(inp.fwd, None, inp.weights, inp.cash, rules.rebalance, rules.cost_bps), inp)
        rows.append(_row("rebalanced", f"Allocation rebalanced {rules.rebalance}", "benchmark", sim, inp.rf, bh_row))
        equity["rebalanced"] = _equity(sim["returns"])
    for t in BENCHMARKS:
        if t in compare and t in inp.bench_fwd:
            sim = _by_period_end(simulate(inp.bench_fwd[[t]], None, pd.Series({t: 1.0}), inp.cash, "never", 0.0), inp)
            label = "S&P 500 (SPY)" if t == "SPY" else "Equal-weight S&P 500 (RSP)"
            rows.append(_row(t, label, "market", sim, inp.rf, bh_row))
            equity[t] = _equity(sim["returns"])
    if "classic" in compare:
        keys = [k for k in CLASSIC_LABELS if k in STRATEGY_REGISTRY]
        for n, key in enumerate(keys, 1):
            progress(n, len(keys), f"Step 2 of 2: classic strategies ({CLASSIC_LABELS[key]})")
            slot, trades, in_market = {}, [], []
            for t in inp.tickers:
                fwd_t, per_year, held_share = classic_monthly(inp.prices[t], key, inp.level_dates, inp.tbill_daily,
                                                              rules.cost_bps)
                slot[t] = fwd_t.reindex(inp.fwd.index).where(inp.fwd[t].notna())
                trades.append(per_year * inp.weights[t])
                in_market.append(held_share * inp.weights[t])
            sim = _by_period_end(simulate(pd.DataFrame(slot), None, inp.weights, inp.cash, rules.rebalance,
                                          rules.cost_bps), inp)
            # Weighted by allocation: trades a year per holding and the share of time in the stock.
            sim["invested_share"] = float(np.nansum(in_market))
            rows.append(_row(key, CLASSIC_LABELS[key], "classic", sim, inp.rf, bh_row,
                             {"trades_per_year": float(np.nansum(trades))}))
            equity[key] = _equity(sim["returns"])

    dec_wide = decided.pivot(index="date", columns="ticker", values="decision").reindex(index=inp.dates, columns=inp.tickers)
    code = {"STRONG_BUY": "B", "BUY": "B", "HOLD": "H", "SELL": "S", "STRONG_SELL": "S", NOT_RATED: "N"}
    per_ticker = []
    for t in inp.tickers:
        col = held[t].to_numpy(bool)
        start_state = start_invested
        entries = int(((~np.r_[start_state, col[:-1]]) & col).sum())
        exits = int((np.r_[start_state, col[:-1]] & ~col).sum())
        r_t = inp.fwd[t]
        per_ticker.append({
            "ticker": t, "weight": float(inp.weights[t]), "owner_status": inp.owner_status.get(t),
            "invested_share": float(col[r_t.notna().to_numpy()].mean()) if r_t.notna().any() else None,
            "entries": entries, "exits": exits, "first_date": inp.first_dates.get(t),
            "decisions": [code.get(x, "") if isinstance(x, str) else "" for x in dec_wide[t].tolist()],
            "invested": [bool(x) for x in col],
        })
    kind_order = {"agents": 0, "benchmark": 1, "market": 2, "classic": 3}
    rows.sort(key=lambda r: (kind_order.get(r["kind"], 9), -_finite(r["stats"]["cagr"])))
    return _clean({
        "universe": {"kind": universe.get("kind"), "label": inp.label, "tickers": inp.tickers,
                     "weights": {t: float(w) for t, w in inp.weights.items()},
                     "allocation": universe.get("allocation", "current")},
        "period": {"start": inp.dates[0].strftime("%Y-%m-%d"), "end": inp.level_dates[-1].strftime("%Y-%m-%d"),
                   "months": len(inp.dates), "split": inp.level_dates[len(inp.dates) // 2].strftime("%Y-%m-%d")},
        "setup": describe_setup(config, class_roles),
        "rules": {"entry_gate": rules.entry_gate, "start": rules.start, "rebalance": rules.rebalance,
                  "cost_bps": rules.cost_bps},
        "rows": rows,
        "equity": equity,
        "dates": [d.strftime("%Y-%m-%d") for d in inp.dates],
        "per_ticker": per_ticker,
        "notes": _notes(inp, config, class_roles, rules),
    })


# ------------------------------------------------------------------ batch search

GRID: dict[str, dict[str, Any]] = {
    "shareholder_share": {"label": "Shareholders share of Fundamentals + Shareholders", "values": [0.35, 0.45, 0.55, 0.65]},
    "buy": {"label": "Entry level (buy at or above)", "values": [10, 15, 20, 25]},
    "sell": {"label": "Exit level (sell at or below)", "values": [-15, -25, -35]},
    "cap": {"label": "No new money at or above EV/FCF (none = off)", "values": [25, 35, 50, None]},
    "momentum_vote": {"label": "Momentum weight in the vote (0 = timing only)", "values": [0.0, 0.2]},
    "entry_gate": {"label": "Entry gate", "values": ["none", "not_unfavorable"]},
    "rebalance": {"label": "Rebalance", "values": ["yearly"]},
}
_CONFIG_KEYS = ("shareholder_share", "buy", "sell", "cap", "momentum_vote")
_LIMITS = {"shareholder_share": (0.0, 1.0), "buy": (-100.0, 100.0), "sell": (-100.0, 100.0), "cap": (1.0, 500.0),
           "momentum_vote": (0.0, 5.0)}


def validate_grid(grid: dict[str, Any] | None) -> dict[str, list[Any]]:
    """Clean grid: known settings, parsed values, at most ``MAX_SETUPS`` combinations. Raises ValueError."""
    if not grid:
        return {k: list(spec["values"]) for k, spec in GRID.items()}
    out: dict[str, list[Any]] = {}
    for key, values in grid.items():
        if key not in GRID:
            raise ValueError(f"Unknown setting {key!r}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Pick at least one value for {GRID[key]['label']}")
        clean: list[Any] = []
        for v in values:
            if key == "entry_gate":
                if v not in ENTRY_GATES:
                    raise ValueError(f"Unknown entry gate {v!r}")
            elif key == "rebalance":
                if v not in REBALANCE:
                    raise ValueError(f"Unknown rebalance schedule {v!r}")
            elif key == "cap" and v in (None, "", "off"):
                v = None
            else:
                try:
                    v = float(v)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{GRID[key]['label']}: {v!r} is not a number") from exc
                lo, hi = _LIMITS[key]
                if not (lo <= v <= hi) or not math.isfinite(v):
                    raise ValueError(f"{GRID[key]['label']}: {v:g} is outside {lo:g} to {hi:g}")
                v = int(v) if key in ("buy", "sell") and v == int(v) else v
            if v not in clean:
                clean.append(v)
        out[key] = clean
    count = math.prod(len(v) for v in out.values())
    if count > MAX_SETUPS:
        raise ValueError(f"{count} setups is more than the limit of {MAX_SETUPS}: select fewer values")
    return out


def apply_overrides(config: ScoringConfig, combo: dict[str, Any]) -> ScoringConfig | None:
    """``config`` with one grid point applied, or None when the combination is invalid."""
    ow = dict(config.owner_weights)
    th = dict(config.owner_thresholds)
    roles = dict(config.roles)
    base = (ow.get("fundamentals", 0.0) + ow.get("shareholder", 0.0)) or 1.0
    if "shareholder_share" in combo:
        x = float(combo["shareholder_share"])
        if not 0.0 <= x <= 1.0:
            return None
        ow["shareholder"], ow["fundamentals"] = round(x * base, 4), round((1.0 - x) * base, 4)
    if "buy" in combo:
        th["buy"] = float(combo["buy"])
        th["strong_buy"] = max(th["strong_buy"], th["buy"])
    if "sell" in combo:
        th["sell"] = float(combo["sell"])
        th["strong_sell"] = min(th["strong_sell"], th["sell"])
    if not th["buy"] > th["sell"]:
        return None
    cap = config.owner_max_ev_to_fcf
    if "cap" in combo:
        cap = None if combo["cap"] in (None, "", "off") else float(combo["cap"])
    if "momentum_vote" in combo and float(combo["momentum_vote"]) > 0:
        roles["momentum"] = OWNER
        ow["momentum"] = round(float(combo["momentum_vote"]) * base, 4)
    return replace(config, owner_weights=ow, owner_thresholds=th, owner_max_ev_to_fcf=cap, roles=roles)


def _spearman(a: list[float], b: list[float]) -> float | None:
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5 or s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return None
    return float(s["a"].rank().corr(s["b"].rank()))


def run_setup_search(universe: dict[str, Any], years: float | None = None, rules: Rules | None = None,
                     grid: dict[str, list[Any]] | None = None, config: ScoringConfig | None = None,
                     progress: Progress | None = None, top: int = 50) -> dict[str, Any]:
    """Every combination of ``grid`` on the same stored scores, ranked by the weaker of its two halves.

    The protocol is fixed before looking at results (the owner's rule: data based, no overfitting):
    each setup's excess return over buy and hold is measured separately in the first and second half of
    the period, setups are ranked by the weaker half, and the summary reports whether the first half
    predicts the second (rank correlation) and how the first-half winner did in the second half against
    your saved setup.
    """
    progress = progress or (lambda *_: None)
    base_rules = (rules or Rules()).validate()
    config = config or load_scoring_config()
    grid = validate_grid(grid)
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]
    class_roles = _class_roles()
    inp = _prepare(universe, years, progress)
    bh = _by_period_end(simulate(inp.fwd, None, inp.weights, inp.cash, "never", base_rules.cost_bps), inp)
    bh_row = _buy_hold_row(bh, inp.rf)

    decided_cache: dict[str, pd.DataFrame] = {}

    def run_one(cfg: ScoringConfig, r: Rules) -> dict[str, Any]:
        sig = json.dumps(describe_setup(cfg, class_roles), sort_keys=True, default=str)
        if sig not in decided_cache:
            decided_cache[sig] = evaluate_setup(inp.panel, cfg, class_roles)
        held = positions(decided_cache[sig], inp.dates, inp.tickers, r)
        return _by_period_end(simulate(inp.fwd, held, inp.weights, inp.cash, r.rebalance, r.cost_bps,
                                       r.start == "invested"), inp)

    results = []
    baseline = _row("baseline", "Your saved setup", "agents", run_one(config, base_rules), inp.rf, bh_row,
                    {"setup": describe_setup(config, class_roles), "rules": {"entry_gate": base_rules.entry_gate,
                                                                              "rebalance": base_rules.rebalance}})
    skipped = 0
    for n, combo in enumerate(combos, 1):
        if n % 25 == 0 or n == len(combos):
            progress(n, len(combos), f"Setup {n} of {len(combos)}")
        cfg = apply_overrides(config, {k: v for k, v in combo.items() if k in _CONFIG_KEYS})
        if cfg is None:
            skipped += 1
            continue
        r = replace(base_rules, **{k: combo[k] for k in ("entry_gate", "rebalance") if k in combo}).validate()
        sim = run_one(cfg, r)
        row = _row(f"s{n}", "", "setup", sim, inp.rf, bh_row,
                   {"combo": combo, "setup": describe_setup(cfg, class_roles),
                    "apply": {"owner_weights": cfg.owner_weights, "owner_thresholds": cfg.owner_thresholds,
                              "owner_max_ev_to_fcf": cfg.owner_max_ev_to_fcf, "roles": cfg.roles}})
        row["robust"] = _weaker(row["excess_first"], row["excess_second"])
        row["_returns"] = sim["returns"].to_numpy(float)
        results.append(row)
    if not results:
        raise ValueError("No valid setup in the grid (the entry level must be above the exit level)")
    baseline["robust"] = _weaker(baseline["excess_first"], baseline["excess_second"])

    # Setups that trade identically are one outcome: group them, so the list shows real differences.
    # ``values`` lists, per grid setting, the values that all lead to this same outcome.
    groups: dict[bytes, dict[str, Any]] = {}
    for row in results:
        sig = np.round(np.nan_to_num(row.pop("_returns")), 10).tobytes()
        if sig in groups:
            groups[sig]["_combos"].append(row["combo"])
        else:
            row["_combos"] = [row["combo"]]
            groups[sig] = row
    outcomes = list(groups.values())
    for row in outcomes:
        combos_in = row.pop("_combos")
        row["equivalent"] = len(combos_in) - 1
        row["values"] = {k: sorted({c[k] for c in combos_in}, key=lambda v: (v is None, str(type(v)), v if v is not None else 0))
                         for k in keys}
    ranked = sorted(outcomes, key=lambda r: _finite(r["robust"]), reverse=True)
    pick = max(results, key=lambda r: _finite(r["first_half"]["sharpe"]))
    rho = _spearman([r["excess_first"] for r in results], [r["excess_second"] for r in results])
    both = sum(1 for r in results if r["excess_first"] > 0 and r["excess_second"] > 0)
    pick_beats = pick["excess_second"] > baseline["excess_second"]
    strong = bool(both) and rho is not None and rho >= 0.3 and pick_beats and pick["excess_second"] > 0
    if not both:
        verdict = ("No setup beat buy and hold of this allocation in both halves. On these holdings the agents' exits "
                   "cost return, so buy and hold is the result to beat; a different grid will not change that much.")
    elif rho is None or rho < 0.3:
        verdict = ("Weak evidence: setups that did well in the first half did not reliably do well in the second, so "
                   "the differences look like noise. Keep your current setup.")
    elif not pick_beats or pick["excess_second"] <= 0:
        verdict = ("The setup that won the first half did not beat your saved setup (or buy and hold) in the second "
                   "half. Keep your current setup.")
    else:
        verdict = ("The first half predicts the second and its winner also won out of sample. Worth a closer look: "
                   "check it on tickers it was not tuned on before changing your weights.")
    return _clean({
        "universe": {"kind": universe.get("kind"), "label": inp.label, "tickers": inp.tickers,
                     "weights": {t: float(w) for t, w in inp.weights.items()}},
        "period": {"start": inp.dates[0].strftime("%Y-%m-%d"), "end": inp.level_dates[-1].strftime("%Y-%m-%d"),
                   "split": inp.level_dates[len(inp.dates) // 2].strftime("%Y-%m-%d"), "months": len(inp.dates)},
        "grid": grid,
        "tested": len(results), "skipped": skipped, "outcomes": len(outcomes),
        "buy_hold": bh_row,
        "baseline": baseline,
        "ranked": ranked[:top],
        "summary": {
            "beat_both_halves": both,
            "rank_correlation": rho,
            "first_half_winner": {"combo": pick["combo"], "first_half_sharpe": pick["first_half"]["sharpe"],
                                  "excess_second": pick["excess_second"]},
            "baseline_excess_second": baseline["excess_second"],
            "winner_beats_baseline_out_of_sample": pick_beats,
            "strong": strong,
            "verdict": verdict,
        },
        "notes": _notes(inp, config, class_roles, base_rules),
    })


# ================================================================== background jobs

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
MAX_JOBS = 20


class JobCancelled(Exception):
    """Raised from a job's progress callback once the user has asked it to stop."""


def start_job(kind: str, fn: Callable[..., dict[str, Any]], **kwargs: Any) -> str:
    """Run ``fn(progress=..., **kwargs)`` in a thread; poll with ``get_job``."""
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "kind": kind, "status": "running", "error": None, "result": None,
           "progress": {"done": 0, "total": 0, "stage": "Starting"}, "started": time.time(), "cancelled": False}

    def progress(done: int, total: int, stage: str) -> None:
        if job["cancelled"]:
            raise JobCancelled()
        job["progress"] = {"done": done, "total": total, "stage": stage}

    def run() -> None:
        try:
            job["result"] = fn(progress=progress, **kwargs)
            if not job["cancelled"]:
                job["status"] = "done"
        except JobCancelled:
            pass  # cancel_job already marked it
        except (ValueError, LookupError) as exc:
            job["status"], job["error"] = "error", str(exc)
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            logger.exception("Backtest job %s failed", job_id)
            job["status"], job["error"] = "error", f"Backtest failed: {exc}"
        job["finished"] = time.time()

    with _JOBS_LOCK:
        _JOBS[job_id] = job
        for old in sorted(_JOBS, key=lambda k: _JOBS[k]["started"])[:-MAX_JOBS]:
            _JOBS.pop(old, None)
    threading.Thread(target=run, daemon=True, name=f"lab-{job_id}").start()
    return job_id


def cancel_job(job_id: str) -> bool:
    """Ask a running job to stop at its next progress report; its status is "cancelled" at once. False if unknown."""
    job = _JOBS.get(job_id)
    if job is None:
        return False
    if job["status"] == "running":
        job["cancelled"] = True
        job["status"] = "cancelled"
        job["finished"] = time.time()
    return True


def get_job(job_id: str) -> dict[str, Any] | None:
    job = _JOBS.get(job_id)
    if job is None:
        return None
    out = {k: v for k, v in job.items() if k != "result"}
    if job["status"] == "done":
        out["result"] = job["result"]
    return out
