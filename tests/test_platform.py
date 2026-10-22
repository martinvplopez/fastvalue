"""Tests for macro snapshots, data freshness, history, score backtesting and the web API."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from agents.base import LIVE, AgentAssessment
from data import history
from data.fetcher import CLOSE, LIVE as FRESH_LIVE, MARKET_TZ, STALE, assess_freshness, last_completed_session, market_is_open
from data.macro_data import MacroHistory, compute_macro_snapshot
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------- macro


def _macro_history(days: int = 400, spy_trend: float = 0.3, vix: float = 14.0, drop_spy: bool = False) -> MacroHistory:
    idx = pd.bdate_range("2023-01-02", periods=days)
    t = np.linspace(0, 1, days)
    series = {
        "spy": pd.Series(400 * (1 + spy_trend * t), index=idx),
        "vix": pd.Series(np.full(days, vix), index=idx),
        "vix3m": pd.Series(np.full(days, vix * 1.15), index=idx),
        "rsp": pd.Series(150 * (1 + spy_trend * t * 1.1), index=idx),
        "hyg": pd.Series(75 * (1 + 0.05 * t), index=idx),
        "lqd": pd.Series(np.full(days, 105.0), index=idx),
        "tnx": pd.Series(np.full(days, 4.2), index=idx),
        "t10y2y": pd.Series(np.full(days, 0.8), index=idx),
        "t10y3m": pd.Series(np.full(days, 1.0), index=idx),
    }
    if drop_spy:
        series.pop("spy")
    return MacroHistory(series=series, errors={}, fetched_at="now")


def test_macro_snapshot_bull_regime():
    snap = compute_macro_snapshot(_macro_history())
    assert snap.regime == "BULL"
    assert snap.data_quality == "FULL"
    assert snap.vix_term_ratio == pytest.approx(1 / 1.15, abs=1e-3)
    assert {c["component"] for c in snap.components} >= {"SPY vs 200-day SMA", "VIX level", "VIX term structure"}


def test_macro_snapshot_missing_spy_is_skipped_not_bearish():
    full = compute_macro_snapshot(_macro_history())
    missing = compute_macro_snapshot(_macro_history(drop_spy=True))
    assert "spy_trend" in missing.missing
    assert missing.data_quality == "PARTIAL"
    # The old fallback scored missing SPY as "below the 200 SMA" (-3). Now it adds nothing.
    spy_points = sum(c["points"] for c in full.components if c["component"].startswith("SPY"))
    assert missing.regime_score == pytest.approx(full.regime_score - spy_points, abs=0.01)


def test_macro_snapshot_is_point_in_time():
    hist = _macro_history()
    asof = hist.series["spy"].index[300]
    snap = compute_macro_snapshot(hist, asof)
    assert snap.as_of == asof.strftime("%Y-%m-%d")
    # Changing data after asof must not change the snapshot.
    hist.series["vix"].iloc[301:] = 60.0
    assert compute_macro_snapshot(hist, asof).vix_level == snap.vix_level


def test_macro_snapshot_crisis_on_high_vix():
    snap = compute_macro_snapshot(_macro_history(spy_trend=-0.3, vix=45.0))
    assert snap.regime in ("BEAR", "CRISIS")
    assert snap.regime_score < 0


# ------------------------------------------------------------------ freshness


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=MARKET_TZ)


def test_market_clock():
    assert market_is_open(_et(2026, 9, 28, 10, 0))  # Monday 10:00
    assert not market_is_open(_et(2026, 9, 28, 9, 0))
    assert not market_is_open(_et(2026, 9, 27, 12, 0))  # Sunday
    assert last_completed_session(_et(2026, 9, 27, 12, 0)).isoformat() == "2026-09-25"  # Friday
    assert last_completed_session(_et(2026, 9, 28, 17, 0)).isoformat() == "2026-09-28"
    assert last_completed_session(_et(2026, 9, 28, 11, 0)).isoformat() == "2026-09-25"


def test_assess_freshness_statuses():
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-09-24", "2026-09-25"]))
    assert assess_freshness("X", df, _et(2026, 9, 27, 12, 0)).status == CLOSE
    stale = assess_freshness("X", df, _et(2026, 9, 30, 17, 0))
    assert stale.status == STALE and stale.sessions_behind == 3
    live_df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-09-25", "2026-09-28"]))
    live = assess_freshness("X", live_df, _et(2026, 9, 28, 11, 0))
    assert live.status == FRESH_LIVE and live.is_intraday


# -------------------------------------------------------------------- history


class _Rec:
    def __init__(self, ticker, decision, score, generated_at):
        self.ticker = ticker
        self.decision = decision
        self.composite_score = score
        self.overall_confidence = 70.0
        self.market_regime = "BULL"
        self.suggested_entry = 100.0
        self.generated_at = generated_at
        self.freshness = {"as_of": "2026-09-25"}
        self.contributions = [{"agent_key": "momentum", "raw_score": 40, "normalized_score": 60, "confidence": 80,
                               "data_quality": LIVE, "included": True}]


def test_history_roundtrip_and_previous():
    history.record_analysis(_Rec("AAA", "HOLD", 5.0, "2026-09-25T10:00:00+00:00"))
    later = _Rec("AAA", "BUY", 25.0, "2026-09-26T10:00:00+00:00")
    later.owner = {"score": 25.0}
    later.timing = {"score": -10.0, "label": "NEUTRAL"}
    history.record_analysis(later)
    rows = history.get_history("aaa")
    assert [r["decision"] for r in rows] == ["HOLD", "BUY"]
    assert rows[0]["agent_scores"]["momentum"]["score"] == 40
    assert rows[0]["owner_score"] == 5.0 and rows[0]["timing_score"] is None  # legacy record without layers
    assert rows[1]["owner_score"] == 25.0 and rows[1]["timing_label"] == "NEUTRAL"
    prev = history.get_previous("AAA", "2026-09-26T10:00:00+00:00")
    assert prev["decision"] == "HOLD"
    assert history.get_previous("AAA", "2026-09-25T10:00:00+00:00") is None
    assert history.get_first_after("AAA", "2026-09-26")["decision"] == "BUY"
    assert history.get_latest("AAA")["decision"] == "BUY"


def test_batched_history_matches_one_call_per_ticker():
    import sqlite3

    history.record_analyses([_Rec("AAA", "HOLD", 1.0, "2026-09-25T10:00:00+00:00"),
                             _Rec("AAA", "BUY", 2.0, "2026-09-25T10:00:00+00:00"),  # same time: the later id wins
                             _Rec("BBB", "SELL", -30.0, "2026-09-20T10:00:00+00:00")])
    history.record_analyses([])
    pairs = [("aaa", "2026-09-26"), ("BBB", "2026-09-20T10:00:00+00:00"), ("ZZZ", "2026-09-26"), ("AAA", "2026-09-26")]
    assert history.get_previous_many(pairs) == [history.get_previous(t, b) for t, b in pairs]
    assert history.get_previous_many(pairs)[0]["decision"] == "BUY" and history.get_previous_many([]) == []
    connections = []
    real = sqlite3.connect
    sqlite3.connect = lambda *a, **k: connections.append(a) or real(*a, **k)
    try:
        history.record_analyses([_Rec(t, "HOLD", 0.0, "2026-09-27T10:00:00+00:00") for t in ("C1", "C2", "C3")])
    finally:
        sqlite3.connect = real
    assert len(connections) == 1 and [r["ticker"] for r in history.get_latest_all()] == ["AAA", "BBB", "C1", "C2", "C3"]


def test_watchlist_persistence_dedupes_and_uppercases():
    assert history.load_watchlist() == history.DEFAULT_WATCHLIST
    assert history.save_watchlist(["aapl", "MSFT", "aapl", " ", "nvda"]) == ["AAPL", "MSFT", "NVDA"]
    assert history.load_watchlist() == ["AAPL", "MSFT", "NVDA"]


# ------------------------------------------------------------- score backtest


def test_score_backtest_runs_point_in_time_agents_only():
    from core.score_backtest import run_score_backtest

    ohlcv = make_ohlcv(periods=700, start=100, end=180, noise=0.012)
    result = run_score_backtest("TEST", ohlcv, agent_keys=["momentum", "volume", "options"], frequency="weekly")
    assert [a["key"] for a in result["agents_used"]] == ["momentum", "volume"]
    assert any(e["agent"] == "Options Flow Agent" for e in result["agents_excluded"])
    assert result["evaluations"] > 50
    assert len(result["score_series"]) == result["evaluations"]
    assert sum(b["count"] for b in result["calibration"]["buckets"]) == result["calibration"]["observations"]
    assert result["notes"]
    for trade in result["trades"]:
        assert trade["entry_date"] <= (trade["exit_date"] or "9999")


def test_score_backtest_replays_only_agents_that_are_timing_in_the_users_roles():
    from config.weights import load_scoring_config
    from core.score_backtest import backtestable_agents, resolve_agents

    cfg = load_scoring_config()
    cfg.roles = {**cfg.roles, "volume": "info"}
    keys, excluded = resolve_agents(["momentum", "volume"], cfg)
    assert keys == ["momentum"]
    assert excluded == [{"agent": "Volume & Accumulation Agent", "reason": excluded[0]["reason"]}]
    assert "timing panel" in excluded[0]["reason"]
    assert "volume" not in backtestable_agents(cfg) and "momentum" in backtestable_agents(cfg)


def test_score_backtest_validates_thresholds():
    from core.score_backtest import run_score_backtest

    with pytest.raises(ValueError):
        run_score_backtest("TEST", make_ohlcv(periods=400), agent_keys=["momentum"], entry_threshold=10, exit_threshold=20)
    with pytest.raises(ValueError):
        run_score_backtest("TEST", make_ohlcv(periods=400), agent_keys=["options"])


# ---------------------------------------------------------------------- web


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_api_agents_lists_registry(client):
    data = client.get("/api/agents").get_json()
    keys = {a["key"] for a in data["agents"]}
    assert {"momentum", "volume", "macro"} <= keys
    assert set(data["regimes"]) == {"BULL", "CORRECTION", "BEAR", "CRISIS"}
    assert "momentum" in data["backtestable"]


def test_api_answers_only_local_hosts_and_same_origin_writes(client):
    assert client.get("/api/agents").status_code == 200
    assert client.get("/api/agents", headers={"Host": "127.0.0.1:8501"}).status_code == 200
    # DNS rebinding: a foreign domain resolved to 127.0.0.1 must not read the portfolio.
    assert client.get("/api/portfolio", headers={"Host": "evil.example:8501"}).status_code == 403
    # Cross-site writes (a form post from another page) are refused; the page's own requests pass.
    assert client.post("/api/config/reset", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/config/reset", headers={"Origin": "null"}).status_code == 403
    assert client.post("/api/config/reset", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    # Another local server is another origin (same host, other port).
    assert client.post("/api/config/reset", headers={"Host": "localhost:8501", "Origin": "http://localhost:3000"}).status_code == 403
    assert client.post("/api/config/reset", headers={"Host": "127.0.0.1:8501", "Origin": "http://127.0.0.1:8501"}).status_code == 200
    assert client.post("/api/config/reset", headers={"Host": "localhost:8501", "Origin": "http://localhost:8501",
                                                     "Sec-Fetch-Site": "same-origin"}).status_code == 200
    page = client.get("/")
    assert "script-src 'self'" in page.headers["Content-Security-Policy"]
    assert b"unpkg.com" not in page.data and b"/static/vendor/lightweight-charts" in page.data


def test_api_rejects_bad_ticker_and_config(client):
    assert client.post("/api/analyze", json={"ticker": "<script>"}).status_code == 400
    bad = {"thresholds": {"strong_buy": 0, "buy": 10, "sell": -10, "strong_sell": -45}}
    assert client.put("/api/config", json=bad).status_code == 400


def test_api_rescore_with_draft_weights(client):
    assessments = [
        {"agent_key": "fundamentals", "agent_name": "Fundamentals", "score": 60, "confidence": 90, "data_quality": "LIVE"},
        {"agent_key": "shareholder", "agent_name": "Shareholder", "score": -60, "confidence": 90, "data_quality": "LIVE"},
        {"agent_key": "momentum", "agent_name": "Momentum", "score": 60, "confidence": 90, "data_quality": "LIVE"},
        {"agent_key": "volume", "agent_name": "Volume", "score": -60, "confidence": 90, "data_quality": "LIVE"},
    ]
    config = client.get("/api/agents").get_json()["config"]
    config["owner_weights"] = {"fundamentals": 1.0, "shareholder": 0.0, "insider": 0.0}
    config["regime_weights"]["BULL"] = {"momentum": 1.0, "volume": 0.0}
    first = client.post("/api/rescore", json={"assessments": assessments, "config": config, "regime": "BULL"}).get_json()
    config["owner_weights"] = {"fundamentals": 0.0, "shareholder": 1.0, "insider": 0.0}
    config["regime_weights"]["BULL"] = {"momentum": 0.0, "volume": 1.0}
    second = client.post("/api/rescore", json={"assessments": assessments, "config": config, "regime": "BULL"}).get_json()
    # Owner weights drive the owner score and the decision; regime weights drive only the timing panel.
    assert first["owner"]["composite_score"] > 0 > second["owner"]["composite_score"]
    assert first["decision"].endswith("BUY") and second["decision"].endswith("SELL")
    assert first["timing"]["composite_score"] > 0 > second["timing"]["composite_score"]
    assert first["timing_label"] == "FAVORABLE" and second["timing_label"] == "UNFAVORABLE"


def test_api_config_roles_and_order_roundtrip_and_rescore(client):
    config = client.get("/api/agents").get_json()["config"]
    assert config["roles"]["fundamentals"] == "owner" and config["roles"]["momentum"] == "timing"
    assert config["layer_order"] == ["owner", "timing", "info"] and config["agent_order"][0] == "fundamentals"

    config["roles"]["momentum"] = "owner"
    config["owner_weights"]["momentum"] = 1.0
    config["layer_order"] = ["timing", "owner", "info"]
    config["agent_order"] = ["momentum"] + [k for k in config["agent_order"] if k != "momentum"]
    saved = client.put("/api/config", json=config).get_json()["config"]
    assert saved["roles"]["momentum"] == "owner" and saved["layer_order"][0] == "timing"
    assert saved["agent_order"][0] == "momentum"
    assert client.get("/api/agents").get_json()["config"]["roles"]["momentum"] == "owner"

    assessments = [
        {"agent_key": "fundamentals", "agent_name": "Fundamentals", "score": 10, "confidence": 90, "data_quality": "LIVE"},
        {"agent_key": "momentum", "agent_name": "Momentum", "score": 90, "confidence": 90, "data_quality": "LIVE"},
    ]
    res = client.post("/api/rescore", json={"assessments": assessments, "config": saved, "regime": "BULL"}).get_json()
    assert any(c["agent_key"] == "momentum" and c["included"] for c in res["owner"]["contributions"])
    assert any(w.startswith("Your weights let Momentum decide") for w in res["warnings"])
    saved["roles"]["momentum"] = "info"
    res = client.post("/api/rescore", json={"assessments": assessments, "config": saved, "regime": "BULL"}).get_json()
    assert [c["agent_key"] for c in res["informative"]] == ["momentum"]
    everyone_informative = {**saved, "roles": {k: "info" for k in saved["roles"]}}
    assert client.put("/api/config", json=everyone_informative).status_code == 400


def test_api_analyze_serializes_recommendation(client, monkeypatch):
    from agents.master_agent import MasterDecisionAgent
    import web.app as web_app

    ohlcv = make_ohlcv(noise=0.01)

    def fake_analyze(ticker, record=True):
        master = MasterDecisionAgent(agent_keys=["momentum", "volume"])
        return master.analyze(ticker, ohlcv, {"status": "CLOSE", "as_of": "2024-01-01"}), ohlcv

    monkeypatch.setattr(web_app, "analyze_ticker", fake_analyze)
    res = client.post("/api/analyze", json={"ticker": "test"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["ticker"] == "TEST"
    assert data["decision"] == "NOT_RATED"  # only timing agents ran
    assert data["timing"]["label"] in ("FAVORABLE", "NEUTRAL", "UNFAVORABLE")
    assert len(data["chart"]["candles"]) == 260
    assert {c["agent_key"] for c in data["contributions"]} == {"momentum", "volume"}


# ------------------------------------------------------------ benchmark study


def test_benchmark_timing_math():
    from core.benchmark_study import performance, timing_returns

    idx = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series([100.0, 101.0, 102.0, 99.0, 100.0, 101.0], index=idx)
    rf = pd.Series(0.0001, index=idx)
    # In from the first evaluation, out at the fourth bar.
    scores = pd.Series([50.0, 50.0, -10.0], index=[idx[0], idx[2], idx[3]])
    ret, held = timing_returns(close, scores, rf, entry=15, exit_=0, cost_bps=10)
    assert held.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]  # trades on the bar after the signal
    assert ret.iloc[1] == pytest.approx(0.01 - 0.001)  # entry cost on the first invested bar
    assert ret.iloc[4] == pytest.approx(0.0001 - 0.001)  # back in T-bills, minus the exit cost
    stats = performance(pd.Series([0.01] * 252, index=pd.bdate_range("2024-01-01", periods=252)), pd.Series(0.0, index=pd.bdate_range("2024-01-01", periods=252)), 252)
    assert stats["cagr"] == pytest.approx(1.01 ** 252 - 1, rel=1e-6)
    assert stats["max_drawdown"] == 0.0


def test_clean_handles_every_value_type_as_before():
    import dataclasses
    import json
    from datetime import date

    import numpy as np
    import pandas as pd

    from web.app import _clean

    @dataclasses.dataclass
    class Point:
        x: float
        when: date

    payload = {1: [float("nan"), float("inf"), -0.5, (1, "a"), np.float64("nan"), np.float64(2.5), np.int64(3)],
               "ts": pd.Timestamp("2026-10-21"), "d": date(2026, 1, 2), "flag": True, "none": None,
               "dc": Point(float("nan"), date(2026, 1, 3)), "nested": {"t": (np.bool_(True), 7)}}
    out = _clean(payload)
    assert out == {"1": [None, None, -0.5, [1, "a"], None, 2.5, 3], "ts": "2026-10-21T00:00:00", "d": "2026-01-02",
                   "flag": True, "none": None, "dc": {"x": None, "when": "2026-01-03"}, "nested": {"t": [True, 7]}}
    assert type(out["1"][5]) is float and type(out["1"][6]) is int and type(out["nested"]["t"][0]) is bool
    json.dumps(out, allow_nan=False)


def test_candles_equal_the_row_by_row_build():
    import numpy as np
    import pandas as pd

    from web.app import _candles

    idx = pd.bdate_range("2026-01-01", periods=6)
    frame = pd.DataFrame({"open": np.linspace(10, 11, 6), "high": np.linspace(10.5, 11.5, 6),
                          "low": np.linspace(9.5, 10.5, 6).astype("float32"), "close": [10.123, 10.555, np.nan, 10.0049, 1e6 / 3, 2.675],
                          "volume": np.arange(6, dtype="int64")}, index=idx)
    rows = [{"time": d.strftime("%Y-%m-%d"), "open": round(float(r.open), 2), "high": round(float(r.high), 2),
             "low": round(float(r.low), 2), "close": round(float(r.close), 2)} for d, r in frame.iterrows()]
    new = _candles(frame)
    assert [{k: v for k, v in r.items() if k != "close"} for r in new] == [{k: v for k, v in r.items() if k != "close"} for r in rows]
    assert all((a["close"] == b["close"]) or (a["close"] != a["close"] and b["close"] != b["close"]) for a, b in zip(new, rows))


def test_personal_answers_are_not_stored_by_the_browser(client):
    assert client.get("/api/portfolio").headers["Cache-Control"] == "no-store"
    assert client.get("/api/agents").headers["Cache-Control"] == "no-store"
    assert "no-store" not in client.get("/").headers.get("Cache-Control", "")
    assert "no-store" not in client.get("/static/app.js").headers.get("Cache-Control", "")
    assert "immutable" not in client.get("/static/app.js").headers.get("Cache-Control", "")  # edited by hand
    vendored = client.get("/static/vendor/inter/inter-v20-latin.woff2").headers["Cache-Control"]
    assert "max-age=31536000" in vendored and "immutable" in vendored  # versioned file names


def test_lab_options_reports_an_import_like_the_import_status(client, monkeypatch):
    import core.broker_import as bi
    import data.trade_republic as tr

    assert client.get("/api/lab/options").get_json()["has_import"] is False is bi.import_status()["imported"]
    monkeypatch.setattr(tr, "load_ledger", lambda: {"events": []})
    assert client.get("/api/lab/options").get_json()["has_import"] is True


def test_page_loads_nothing_from_third_parties_and_fonts_are_vendored(client):
    import hashlib
    import re
    from pathlib import Path

    page = client.get("/")
    html = page.data.decode()
    assert not re.search(r'(src|href)="https?://', html)  # every script, style and font from this app
    csp = page.headers["Content-Security-Policy"]
    assert "font-src 'self' data:;" in csp  # data: is the share page's embedded Inter and "googleapis" not in csp and "gstatic" not in csp
    css = client.get("/static/style.css").data.decode()
    fonts = re.findall(r"url\((vendor/inter/[^)]+\.woff2)\)", css)
    assert len(fonts) == 7 and "https://" not in css
    root = Path(__file__).resolve().parents[1] / "web" / "static"
    sums = dict(line.split()[::-1] for line in (root / "vendor/inter/SHA256SUMS").read_text().splitlines())
    for rel in fonts:
        assert client.get(f"/static/{rel}").status_code == 200
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == sums[Path(rel).name]
    assert 'href="/static/vendor/inter/inter-v20-latin.woff2" as="font" type="font/woff2" crossorigin' in html
