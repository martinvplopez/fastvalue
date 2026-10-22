"""News-tone validation: the acceptance rule on synthetic snapshots and prices."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config.settings as settings
from core import sentiment_validation as sv


def _world(n_days: int, informative: bool, seed: int = 3):
    """Snapshots for 8 tickers on each of ``n_days`` sessions; tone predicts the next 20 sessions when informative."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_days + 40)
    tickers = [f"T{i}" for i in range(8)]
    folder = settings.STORE_DIR / "sentiment_history"
    folder.mkdir(parents=True, exist_ok=True)
    drift = {t: np.zeros(len(days)) for t in tickers}
    files = {t: [] for t in tickers}
    for k, day in enumerate(days[:n_days]):
        for t in tickers:
            score = float(rng.uniform(-0.6, 0.6))
            files[t].append(json.dumps({"date": day.strftime("%Y-%m-%d"), "score": round(score, 4), "headlines": 5}))
            if informative:  # spread the tone over the 20 sessions after the next one
                drift[t][k + 2:k + 22] += score * 0.001
    for t in tickers:
        (folder / f"{t}.jsonl").write_text("\n".join(files[t]) + "\n")
    closes = {t: pd.Series(100 * np.cumprod(1 + drift[t] + rng.normal(0, 0.002, len(days))), index=days) for t in tickers}
    closes["SPY"] = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.001, len(days))), index=days)
    return closes


def test_too_little_history_keeps_the_weight_at_zero():
    report = sv.run(_world(30, informative=True))
    assert not report["accepted"] and "Not enough history" in report["verdict"]
    assert report["tickers"] == 8 and report["by_horizon"]["20"]["days"] == 30


def test_an_informative_tone_passes_and_noise_does_not():
    good = sv.run(_world(sv.MIN_DAYS + 20, informative=True))
    assert good["accepted"], good["checks"]
    assert good["by_horizon"]["20"]["mean_ic"] > 0.2
    assert good["tone_buckets"][2]["mean_excess_20"] > good["tone_buckets"][0]["mean_excess_20"]
    for path in (settings.STORE_DIR / "sentiment_history").glob("*.jsonl"):
        path.unlink()
    noise = sv.run(_world(sv.MIN_DAYS + 20, informative=False, seed=11))
    assert not noise["accepted"]


def test_no_snapshots_at_all():
    report = sv.run({})
    assert report["snapshots"] == 0 and not report["accepted"]
