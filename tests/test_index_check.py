"""S&P 500 out-of-sample check of a lab setup, on a synthetic panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.weights import load_scoring_config
from core import index_check, lab


def _world(months: int = 96, names: int = 40, seed: int = 5):
    """Fundamentals predict next month's return; shareholder scores are noise."""
    rng = np.random.default_rng(seed)
    dates = list(pd.date_range("2012-01-31", periods=months, freq="ME"))
    rows, fwd = [], {}
    for d in dates:
        fund = rng.uniform(-100, 100, names)
        sh = rng.uniform(-100, 100, names)
        tickers = [f"T{i}" for i in range(names)]
        for t, f, s in zip(tickers, fund, sh):
            row = {"date": d, "ticker": t, "sector": "Industrials", "is_financial": False, "ev_to_fcf": 15.0, "fcf": 1e9,
                   "regime": "BULL", "pb_setup": "neutral", "fund_score": f, "fund_conf": 90.0, "sh_score": s, "sh_conf": 90.0}
            for key in ("momentum", "volume", "pullback", "macro"):
                score_col, conf_col = lab.AGENT_COLUMNS[key]
                row[score_col], row[conf_col] = 0.0, 50.0
            rows.append(row)
        fwd[d] = pd.Series(0.008 + fund / 100 * 0.02 + rng.normal(0, 0.03, names), index=tickers)
    panel = pd.DataFrame(rows)
    fwd1 = pd.DataFrame(fwd).T
    rf = pd.Series(0.001, index=[d + pd.offsets.MonthEnd(1) for d in dates])
    return panel, fwd1, rf


def test_a_setup_that_follows_the_signal_generalizes():
    panel, fwd1, rf = _world()
    saved = load_scoring_config()
    candidate = lab.apply_overrides(saved, {"shareholder_share": 0.0})
    out = index_check.compare(panel, fwd1, rf, candidate, saved)
    assert out["level"] == "generalizes", out["difference"]
    assert out["difference"]["first_half"] > 0 and out["difference"]["second_half"] > 0
    assert [r["label"] for r in out["rows"]][1] == "Your saved setup"


def test_the_same_setup_is_reported_as_the_same():
    panel, fwd1, rf = _world(months=40)
    saved = load_scoring_config()
    out = index_check.compare(panel, fwd1, rf, saved, saved)
    assert out["level"] == "same" and out["difference"]["cagr"] == pytest.approx(0.0)


def test_book_exits_on_sell_and_respects_the_entry_gate():
    dates = list(pd.date_range("2020-01-31", periods=3, freq="ME"))
    tickers = [f"T{i}" for i in range(12)]

    def frame(decisions, labels):
        return pd.DataFrame([{"date": d, "ticker": t, "decision": decisions[k], "timing_label": labels[k], "setup": "neutral"}
                             for k, d in enumerate(dates) for t in tickers])

    fwd1 = pd.DataFrame(0.01, index=dates, columns=tickers)
    held = index_check.book(frame(["BUY", "HOLD", "SELL"], ["NEUTRAL"] * 3), fwd1)
    assert len(held["returns"]) == 2  # bought in month 1, kept through HOLD, sold on SELL
    gated = index_check.book(frame(["BUY", "BUY", "BUY"], ["UNFAVORABLE", "UNFAVORABLE", "NEUTRAL"]), fwd1,
                             entry_gate="not_unfavorable")
    assert len(gated["returns"]) == 1  # enters only once the label is no longer unfavorable


def test_api_rejects_unknown_settings():
    from web.app import app

    client = app.test_client()
    assert client.post("/api/lab/index-check", json={"combo": {"evil": 1}}).status_code == 400
    assert client.post("/api/lab/index-check", json={"combo": "x"}).status_code == 400
