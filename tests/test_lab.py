"""Backtest lab: vectorized scoring vs the live engine, trading rules, simulation, cache, batch search and API."""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import config.settings as settings
from agents.base import LIVE, OWNER, UNAVAILABLE, AgentAssessment, signal_from_score
from agents.scoring import aggregate_layers
from analysis.fundamental_metrics import is_excluded_sector
from config.weights import load_scoring_config, registered_roles
from core import lab
from data import portfolio as store


# ------------------------------------------------------------------ scoring parity


def _random_panel(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {
        "date": pd.Timestamp("2015-01-31"), "ticker": [f"T{i}" for i in range(n)],
        "sector": rng.choice(["Technology", "Utilities", "Real estate", "Industrials", None], n),
        "is_financial": rng.random(n) < 0.15,
        "ev_to_fcf": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(5, 80, n)),
        "fcf": rng.uniform(-5e8, 5e9, n),
        "regime": rng.choice(["BULL", "CORRECTION", "BEAR", "CRISIS", None], n),
        "pb_setup": rng.choice(["pullback_in_uptrend", "extended", "neutral", "trend_broken"], n),
    }
    for key, (score_col, conf_col) in lab.AGENT_COLUMNS.items():
        data[score_col] = np.where(rng.random(n) < 0.15, np.nan, rng.uniform(-100, 100, n))
        data[conf_col] = rng.uniform(0, 100, n)
    return pd.DataFrame(data)


def _assessments(row: pd.Series) -> list[AgentAssessment]:
    out = []
    excluded = is_excluded_sector(row["sector"]) if isinstance(row["sector"], str) else False
    for key, (score_col, conf_col) in lab.AGENT_COLUMNS.items():
        ok = pd.notna(row[score_col])
        score = float(row[score_col]) if ok else 0.0
        raw = {}
        if key in ("fundamentals", "shareholder"):
            raw = {"excluded_sector": excluded, "sector": row["sector"]}
        if key == "fundamentals":
            raw |= {"ev_to_fcf": None if pd.isna(row["ev_to_fcf"]) else float(row["ev_to_fcf"]),
                    "fcf": float(row["fcf"]), "is_financial": bool(row["is_financial"])}
        out.append(AgentAssessment(key, score, signal_from_score(score), float(row[conf_col]), "", agent_key=key,
                                   data_quality=LIVE if ok else UNAVAILABLE, raw_data=raw))
    return out


@pytest.mark.parametrize("variant", ["shipped", "trading_decides", "no_cap", "tight_thresholds", "disabled"])
def test_vectorized_scoring_matches_the_live_engine(variant):
    cfg = load_scoring_config()
    if variant == "trading_decides":
        cfg = replace(cfg, roles={**cfg.roles, "momentum": OWNER, "shareholder": "info"},
                      owner_weights={**cfg.owner_weights, "momentum": 0.3})
    elif variant == "no_cap":
        cfg = replace(cfg, owner_max_ev_to_fcf=None)
    elif variant == "tight_thresholds":
        cfg = replace(cfg, owner_thresholds={"strong_buy": 30.0, "buy": 5.0, "sell": -5.0, "strong_sell": -30.0},
                      confidence_floor=0.1)
    elif variant == "disabled":
        cfg = replace(cfg, enabled={**cfg.enabled, "fundamentals": False, "volume": False})
    panel = _random_panel()
    classes = registered_roles()
    scales = lab._scales(cfg)
    ours = lab.evaluate_setup(panel, cfg, classes)
    for i, row in panel.iterrows():
        ref = aggregate_layers(_assessments(row), cfg, row["regime"], scales, classes)
        assert ours.at[i, "decision"] == ref.decision, (variant, i)
        if ref.owner_applicable:
            assert ours.at[i, "owner_score"] == pytest.approx(ref.owner.composite_score, abs=0.051)
        assert ours.at[i, "timing_label"] == ref.timing_label, (variant, i)


# ------------------------------------------------------------------ trading rules


def _decided(decisions: list[str | None], labels: list[str] | None = None, setups: list[str] | None = None):
    dates = list(pd.date_range("2020-01-31", periods=len(decisions), freq="ME"))
    frame = pd.DataFrame({"date": dates, "ticker": "A", "decision": decisions,
                          "timing_label": labels or ["NEUTRAL"] * len(decisions),
                          "setup": setups or ["neutral"] * len(decisions)})
    return frame, dates


def test_positions_follow_the_owner_rules():
    frame, dates = _decided(["HOLD", "SELL", "HOLD", "BUY", "HOLD", "STRONG_SELL", None, "NOT_RATED", "HOLD"])
    held = lab.positions(frame, dates, ["A"], lab.Rules(start="invested"))["A"].tolist()
    # HOLD keeps, SELL exits, BUY enters, a missing score keeps, not rated is held.
    assert held == [True, False, False, True, True, False, False, True, True]
    from_cash = lab.positions(frame, dates, ["A"], lab.Rules(start="cash"))["A"].tolist()
    assert from_cash[0] is False and from_cash[3] is True


def test_entry_gate_waits_for_the_timing_panel():
    frame, dates = _decided(["BUY", "BUY", "BUY"], labels=["UNFAVORABLE", "NEUTRAL", "UNFAVORABLE"],
                            setups=["extended", "extended", "neutral"])
    gate = lambda g: lab.positions(frame, dates, ["A"], lab.Rules(start="cash", entry_gate=g))["A"].tolist()  # noqa: E731
    assert gate("none") == [True, True, True]
    assert gate("not_unfavorable") == [False, True, True]  # once in, a later unfavorable label does not exit
    assert gate("favorable") == [False, False, False]
    assert gate("not_extended") == [False, False, True]


# ------------------------------------------------------------------ simulation


def test_simulate_buy_and_hold_cash_costs_and_rebalancing():
    idx = pd.DatetimeIndex(["2020-03-31", "2020-04-30"])
    fwd = pd.DataFrame({"A": [0.10, 0.10], "B": [0.0, 0.0]}, index=idx)
    w = pd.Series({"A": 0.5, "B": 0.5})
    cash = pd.Series(0.01, index=idx)
    bh = lab.simulate(fwd, None, w, cash, "never", 0.0)["returns"].tolist()
    assert bh == pytest.approx([0.05, 1.105 / 1.05 - 1])
    out_b = pd.DataFrame({"A": [True, True], "B": [False, False]}, index=idx)
    first = lab.simulate(fwd, out_b, w, cash, "never", 10.0)["returns"].iloc[0]
    assert first == pytest.approx((0.55 + (0.5 - 0.0005) * 1.01) / 1.0 - 1)  # B moves to cash once, at 10 bps
    # April rebalance resets the drifted slots to 50/50 before the second month.
    reb = lab.simulate(fwd, None, w, cash, "yearly", 0.0)["returns"].tolist()
    assert reb[1] == pytest.approx(0.05)
    missing = fwd.copy()
    missing.loc[idx[0], "B"] = np.nan  # no price yet: the slot sits in cash
    assert lab.simulate(missing, None, w, cash, "never", 0.0)["returns"].iloc[0] == pytest.approx(0.055)


def _prices(start: str = "2008-01-01", end: str = "2016-12-31", drift: float = 0.0004, seed: int = 1) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "adj_close": close,
                         "volume": 1e6, "dividends": 0.0, "splits": 0.0}, index=idx)


def test_classic_buy_and_hold_slot_matches_the_stock():
    prices = _prices()
    dates = list(pd.date_range("2011-01-31", "2012-12-31", freq="ME"))
    fwd, trades, held = lab.classic_monthly(prices, "buy_and_hold", dates, pd.Series(dtype=float), 0.0)
    level = prices["adj_close"].reindex(prices.index.union(pd.DatetimeIndex(dates))).ffill().reindex(dates)
    assert fwd.to_numpy() == pytest.approx((level.shift(-1) / level - 1).iloc[:-1].to_numpy())
    assert trades == 0 and held == pytest.approx(1.0)


# ------------------------------------------------------------------ end to end (synthetic inputs)


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "cache")
    prices = {"AAA": _prices(seed=1), "BBB": _prices(seed=2, drift=0.0002), "SPY": _prices(seed=3), "RSP": _prices(seed=4)}
    monkeypatch.setattr(lab, "load_prices", lambda tickers: {t: prices[t] for t in tickers if t in prices})
    monkeypatch.setattr(lab, "load_rates", lambda dates: (pd.Series(0.001, index=dates), pd.Series(dtype=float)))

    def panel(tickers, dates, progress=None, workers=None):
        rows = []
        for i, d in enumerate(dates):
            for t in tickers:
                good = (i // 12) % 2 == 0 if t == "AAA" else True
                rows.append({"date": d, "ticker": t, "sector": "Technology", "is_financial": False,
                             "fund_score": 60.0 if good else -60.0, "fund_conf": 80.0,
                             "sh_score": 40.0 if good else -50.0, "sh_conf": 80.0, "ev_to_fcf": 18.0, "fcf": 1e9,
                             "mom_score": 20.0, "mom_conf": 60.0, "vol_score": 0.0, "vol_conf": 50.0,
                             "pb_score": 0.0, "pb_conf": 40.0, "pb_setup": "neutral", "trend": "UP",
                             "macro_score": 10.0, "macro_conf": 60.0, "regime": "BULL"})
        return pd.DataFrame(rows), {t: "ok" for t in tickers}

    monkeypatch.setattr(lab, "build_panel", panel)
    store.save_portfolio({"cash": 0, "positions": [{"ticker": "AAA", "shares": 10, "avg_cost": 40},
                                                   {"ticker": "BBB", "shares": 30, "avg_cost": 40}]})


def test_run_lab_compares_everything_on_the_portfolio(synthetic):
    res = lab.run_lab({"kind": "portfolio", "allocation": "current"}, rules=lab.Rules())
    keys = [r["key"] for r in res["rows"]]
    assert keys[:2] == ["fastvalue", "buy_hold"]
    assert {"rebalanced", "SPY", "RSP", "sma_crossover", "multi_factor"} <= set(keys)
    ours = res["rows"][0]
    assert ours["excess_cagr"] == pytest.approx(ours["stats"]["cagr"] - res["rows"][1]["stats"]["cagr"])
    aaa = next(p for p in res["per_ticker"] if p["ticker"] == "AAA")
    assert aaa["exits"] >= 2 and aaa["entries"] >= 2 and set(aaa["decisions"]) <= {"B", "H", "S", "N", ""}
    assert sum(res["universe"]["weights"].values()) == pytest.approx(1.0)
    json.dumps(res, allow_nan=False)  # NaN-free, ready for the browser


@pytest.mark.parametrize("years", [3, 5, 10])
def test_period_month_ends_are_whole_days_so_the_score_cache_matches(synthetic, monkeypatch, years):
    seen = []
    panel = lab.build_panel
    monkeypatch.setattr(lab, "build_panel", lambda tickers, dates, progress=None, workers=None:
                        (seen.extend(dates), panel(tickers, dates, progress, workers))[1])
    lab.run_lab({"kind": "portfolio", "allocation": "current"}, years=years, rules=lab.Rules())
    assert seen and all(d == d.normalize() for d in seen)


def test_setup_search_ranks_by_the_weaker_half_and_groups_identical_outcomes(synthetic):
    grid = {"shareholder_share": [0.3, 0.7], "buy": [-20, 15], "sell": [-15, -30], "entry_gate": ["none", "favorable"]}
    res = lab.run_setup_search({"kind": "ticker", "ticker": "AAA"}, grid=grid)
    assert res["tested"] + res["skipped"] == 16 and res["skipped"] == 4  # BUY -20 is not above SELL -15
    robust = [r["robust"] for r in res["ranked"]]
    assert robust == sorted(robust, reverse=True)
    assert res["outcomes"] == len(res["ranked"]) and sum(r["equivalent"] + 1 for r in res["ranked"]) == res["tested"]
    assert set(res["summary"]) >= {"rank_correlation", "beat_both_halves", "verdict", "strong"}
    assert res["baseline"]["robust"] is not None
    json.dumps(res, allow_nan=False)


def test_validate_grid_rejects_bad_input():
    assert lab.validate_grid(None) == {k: v["values"] for k, v in lab.GRID.items()}
    assert lab.validate_grid({"cap": ["off", 30], "buy": ["15"]}) == {"cap": [None, 30.0], "buy": [15]}
    for bad in ({"nope": [1]}, {"entry_gate": ["sometimes"]}, {"buy": [500]}, {"buy": []},
                {"shareholder_share": [i / 100 for i in range(50)], "buy": list(range(30))}):
        with pytest.raises(ValueError):
            lab.validate_grid(bad)


# ------------------------------------------------------------------ cache


def test_panel_cache_scores_each_month_once(monkeypatch, tmp_path):
    import data.sec_edgar as sec

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "cache")
    monkeypatch.setattr(sec, "load_ticker_map", lambda max_age_days=None: {"AAA": 1})
    monkeypatch.setattr(lab, "_fred", lambda series: pd.Series(dtype=float))
    monkeypatch.setattr(lab, "_macro_rows", lambda dates: pd.DataFrame(
        {"date": dates, "macro_score": 5.0, "macro_conf": 50.0, "regime": "BULL", "vix": 15.0}))
    calls = []

    def score(task):
        ticker, cik, _, owner_dates, timing_dates, _, _ = task
        calls.append((len(owner_dates), len(timing_dates)))
        owner = [{"date": d, "fund_score": 10.0, "fund_conf": 70.0, "sh_score": 5.0, "sh_conf": 70.0,
                  "sector": "Technology"} for d in owner_dates]
        timing = [{"date": d, "mom_score": 1.0, "mom_conf": 50.0} for d in timing_dates]
        return ticker, owner, timing, "ok"

    monkeypatch.setattr(lab, "_score_task", score)
    dates = list(pd.date_range("2015-01-31", periods=6, freq="ME"))
    first, status = lab.build_panel(["AAA"], dates, workers=1)
    assert len(first) == 6 and status == {"AAA": "ok"} and calls == [(6, 6)]
    again, _ = lab.build_panel(["AAA"], dates, workers=1)
    assert calls == [(6, 6)] and again["fund_score"].tolist() == first["fund_score"].tolist()
    lab.build_panel(["AAA"], dates + [pd.Timestamp("2015-07-31")], workers=1)
    assert calls[-1] == (1, 1)  # only the new month
    assert first["regime"].eq("BULL").all()


# ------------------------------------------------------------------ API


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    return app.test_client()


def _wait(client, job_id: str) -> dict:
    for _ in range(100):
        job = client.get(f"/api/lab/jobs/{job_id}").get_json()
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_api_lab_runs_jobs_and_validates(client, monkeypatch):
    seen = {}

    def fake_run(progress=None, **kwargs):
        seen.update(kwargs)
        progress(1, 1, "done")
        return {"rows": []}

    monkeypatch.setattr(lab, "run_lab", fake_run)
    job = client.post("/api/lab/run", json={"universe": "ticker", "ticker": "ko", "years": 5, "entry_gate": "favorable"}).get_json()["job"]
    done = _wait(client, job)
    assert done["status"] == "done" and done["result"] == {"rows": []}
    assert seen["universe"] == {"kind": "ticker", "allocation": "current", "ticker": "KO"}
    assert seen["years"] == 5.0 and seen["rules"].entry_gate == "favorable"
    assert client.post("/api/lab/run", json={"universe": "everything"}).status_code == 400
    assert client.post("/api/lab/run", json={"universe": "ticker", "ticker": "KO", "rebalance": "hourly"}).status_code == 400
    assert client.post("/api/lab/search", json={"universe": "ticker", "ticker": "KO", "grid": {"nope": [1]}}).status_code == 400
    assert client.get("/api/lab/jobs/doesnotexist").status_code == 404
    options = client.get("/api/lab/options").get_json()
    assert set(options["grid"]) == set(lab.GRID) and "not_unfavorable" in options["entry_gates"]


def test_api_lab_job_reports_errors(client, monkeypatch):
    def failing(progress=None, **kwargs):
        raise ValueError("The portfolio is empty: add positions in the Portfolio tab first")

    monkeypatch.setattr(lab, "run_lab", failing)
    job = client.post("/api/lab/run", json={"universe": "portfolio"}).get_json()["job"]
    done = _wait(client, job)
    assert done["status"] == "error" and "portfolio is empty" in done["error"]


def test_api_lab_job_can_be_cancelled(client, monkeypatch):
    def slow_run(progress=None, **kwargs):
        for n in range(500):
            progress(n, 500, "Setup")  # raises once the job is cancelled
            time.sleep(0.02)
        return {"rows": []}

    monkeypatch.setattr(lab, "run_lab", slow_run)
    job = client.post("/api/lab/run", json={"universe": "ticker", "ticker": "KO"}).get_json()["job"]
    time.sleep(0.1)
    assert client.post(f"/api/lab/jobs/{job}/cancel").get_json()["status"] == "cancelled"
    assert client.get(f"/api/lab/jobs/{job}").get_json()["status"] == "cancelled"
    time.sleep(0.2)  # the worker unwinds without turning it back into done or error
    assert client.get(f"/api/lab/jobs/{job}").get_json()["status"] == "cancelled"
    assert client.post("/api/lab/jobs/doesnotexist/cancel").status_code == 404


def test_classic_param_labels_are_sentence_case_and_percent_fields_flagged():
    from strategies import STRATEGY_REGISTRY

    params = STRATEGY_REGISTRY["momentum"]["params"]
    assert params["entry_threshold"]["label"] == "Entry threshold (%)" and params["entry_threshold"]["percent"]
    assert STRATEGY_REGISTRY["sma_crossover"]["params"]["fast_window"]["label"] == "Fast SMA window"
