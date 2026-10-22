"""Tests for the market regime history view: labels, per-regime stats, halves, the why-today breakdown and the API shape.

Synthetic series only; nothing reads the network or the owner's files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config.settings as settings
from core import regime_history as rh
from core import regime_study as rs
from data.macro_data import classify_regime, compute_macro_snapshot
from tests.test_regime_study import rewrite_future, synthetic_history

START = "2020-01-01"


@pytest.fixture(scope="module")
def history():
    return synthetic_history(n=900, seed=5)


@pytest.fixture(scope="module")
def qqq(history):
    s = history.series["spy"]
    return (s * 1.3).iloc[400:]  # starts later than SPY, like the real QQQ


@pytest.fixture(scope="module")
def built(history, qqq):
    return rs._clean(rh.build_history(history, qqq, start=START))


def test_labels_match_the_classifier_and_the_cutoffs(history, built):
    assert built["monthly"]
    for row in built["monthly"][::5]:
        snap = compute_macro_snapshot(history, row["date"])
        assert row["regime"] == snap.regime == classify_regime(snap.regime_score)
        assert row["regime_score"] == pytest.approx(snap.regime_score)
    codes = built["weekly"]["regime"]
    assert set(codes) <= {0, 1, 2, 3}
    n = len(built["weekly"]["dates"])
    for key in ("regime", "regime_score", "macro_score", "spy", "qqq"):
        assert len(built["weekly"][key]) == n


def test_view_reads_nothing_after_its_dates(history, built):
    cut = pd.Timestamp(built["monthly"][len(built["monthly"]) // 2]["date"])
    other = rh.build_history(rewrite_future(history, cut, "scale"), None, start=START)
    k = next(i for i, r in enumerate(other["monthly"]) if r["date"] == str(cut.date()))
    assert other["monthly"][k]["regime"] == built["monthly"][k]["regime"]
    assert other["monthly"][k]["regime_score"] == pytest.approx(built["monthly"][k]["regime_score"])


def test_qqq_only_where_it_exists(built):
    w = built["weekly"]
    assert w["qqq"][0] is None and w["qqq"][-1] is not None
    assert w["spy"][0] is not None
    assert built["stats"]["assets"]["QQQ"]["first_month"] >= built["stats"]["assets"]["SPY"]["first_month"]


def test_stats_per_regime_and_halves(history, built):
    stats = built["stats"]
    spy = stats["assets"]["SPY"]
    months = len(built["monthly"])
    assert stats["months"] == months
    for period in rh.PERIODS:
        assert sum(spy[period][r]["months"] for r in rh.REGIMES) == months or period != "whole"
        shares = [spy[period][r]["share"] for r in rh.REGIMES]
        assert sum(shares) == pytest.approx(1.0)
    assert sum(spy["first_half"][r]["months"] for r in rh.REGIMES) + sum(spy["second_half"][r]["months"] for r in rh.REGIMES) == months
    assert stats["split"] in {m["date"] for m in built["monthly"]}


def test_stats_equal_a_direct_calculation(history, built):
    spy = history.series["spy"]
    dates = [pd.Timestamp(m["date"]) for m in built["monthly"]]
    fwd = rs.forward_spy(spy, dates)
    labels = np.array([m["regime"] for m in built["monthly"]])
    for r in rh.REGIMES:
        vals = fwd["vol_3m"].to_numpy()[labels == r]
        vals = vals[~np.isnan(vals)]
        got = built["stats"]["assets"]["SPY"]["whole"][r]["vol_3m"]
        if len(vals):
            assert got == pytest.approx(float(vals.mean()))
        else:
            assert got is None


def test_eras_list_missing_inputs(built):
    eras = built["eras"]
    assert eras and sum(e["months"] for e in eras) == len(built["monthly"])
    for e in eras:
        assert set(e["inputs"]) | set(e["missing"]) == set(rs.INPUTS)
        assert sum(e["shares"].values()) == pytest.approx(1.0)


def test_why_today_breakdown(history):
    why = rh.explain(compute_macro_snapshot(history, None))
    keys = [i["key"] for i in why["inputs"]]
    assert keys == list(rs.INPUTS)
    pts = [i["points"] for i in why["inputs"] if i["available"]]
    assert sum(pts) == pytest.approx(why["total"])
    assert why["regime"] == classify_regime(why["total"])
    assert [c["regime"] for c in why["cutoffs"]] == ["BULL", "CORRECTION", "BEAR", "CRISIS"]
    assert [c["from"] for c in why["cutoffs"]] == [4.0, 1.0, -3.0, None]
    for i in why["inputs"]:
        assert i["rule"] and i["label"] and i["best"] >= (i["points"] or 0) >= i["worst"] - 0.5 or not i["available"]
    assert -100 <= why["agent"]["score"] <= 100
    assert why["agent"]["stance"] == settings.VIX_STANCE


def test_why_today_marks_missing_inputs(history):
    short = type(history)(series={k: v for k, v in history.series.items() if k not in ("hyg", "lqd")}, errors={}, fetched_at="")
    why = rh.explain(compute_macro_snapshot(short, None))
    credit = next(i for i in why["inputs"] if i["key"] == "credit")
    assert credit["available"] is False and credit["points"] is None
    assert "credit" in why["missing"]


def test_api_shape_and_failure(monkeypatch, built):
    from web.app import app

    monkeypatch.setattr("core.regime_history.get_regime_history", lambda refresh=False: built)
    client = app.test_client()
    ok = client.get("/api/regime/history")
    assert ok.status_code == 200
    body = ok.get_json()
    for key in ("weekly", "monthly", "stats", "eras", "today", "regimes"):
        assert key in body

    def boom(refresh=False):
        raise RuntimeError("Yahoo refused")

    monkeypatch.setattr("core.regime_history.get_regime_history", boom)
    bad = client.get("/api/regime/history")
    assert bad.status_code == 503 and "Try again" in bad.get_json()["error"]


def test_cache_for_a_day(monkeypatch, tmp_path, history, qqq):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(rs, "load_long_history", lambda refresh=False: calls.append(1) or history)
    monkeypatch.setattr(rh, "load_qqq", lambda: qqq)
    monkeypatch.setattr(rh, "START", START)
    monkeypatch.setattr(rh, "build_history", lambda h, q, start=START: _small(h, q))
    a = rh.get_regime_history()
    b = rh.get_regime_history()
    assert a == b and len(calls) == 1
    assert (tmp_path / "regime_history" / "history.json").exists()


def _small(h, q):
    return {"stub": True, "start": START}


def test_no_qqq_is_not_cached(monkeypatch, tmp_path, history):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rs, "load_long_history", lambda refresh=False: history)
    monkeypatch.setattr(rh, "load_qqq", lambda: None)
    monkeypatch.setattr(rh, "build_history", lambda h, q, start=START: _small(h, q))
    out = rh.get_regime_history()
    assert "QQQ" in out["warning"]
    assert not (tmp_path / "regime_history" / "history.json").exists()
