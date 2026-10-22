"""Trend, momentum and dip rules and their studies (core/systematic.py, D-48). Synthetic data only, no network."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from analysis.trend_indicators import (
    drawdown_from_high,
    excess_momentum,
    msci_scores,
    pb_ema_channel,
    weekly_volatility,
    wilder_rsi,
)
from core import monitor, systematic
from strategies import STRATEGY_REGISTRY
from strategies.base import exposure_signals, latch, month_end_mask
from strategies.buy_the_dip import DipInUptrend, DrawdownDip
from strategies.trend_following import AbsoluteMomentum, MSCIMomentumTrend, PBEmaChannel, TrendFilter


def _prices(days: int = 2600, seed: int = 3, drift: float = 0.0004, vol: float = 0.012, start: str = "2000-01-03",
            crash_at: int | None = None) -> pd.DataFrame:
    """A random walk with an optional 40% crash over 60 days, as a price-history frame."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, days)
    if crash_at is not None:
        r[crash_at:crash_at + 60] = math.log(0.6) / 60
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range(start, periods=days)
    return pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996, "close": close,
                         "adj_close": close * 1.0, "volume": 1e6, "dividends": 0.0, "splits": 0.0}, index=idx)


@pytest.fixture
def offline(monkeypatch):
    """The study's data sources replaced by synthetic prices and a 2% T-bill rate."""
    frames = {"QQQ": _prices(crash_at=1500), "SPY": _prices(seed=5, crash_at=1500)}
    frames["^NDX"] = _prices(days=3000, seed=9, start="1998-11-02")
    lev = _prices(seed=11)
    frames["TQQQ"] = lev.iloc[1800:]
    monkeypatch.setattr(systematic, "load_prices", lambda tickers: {t: frames[t] for t in tickers if t in frames})
    monkeypatch.setattr(systematic, "_fred_long", lambda *a, **k: pd.Series(2.0, index=pd.bdate_range("1990-01-01", "2030-01-01")))
    monkeypatch.setattr(systematic, "_vix", lambda index: None)
    return frames


# ------------------------------------------------------------------ indicators


def test_pb_ema_channel_states():
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(np.r_[np.full(300, 100.0), np.full(50, 120.0), np.full(50, 80.0)], index=idx)
    ch = pb_ema_channel(close * 1.01, close, 50)
    assert (ch["state"].iloc[:49] == 0).all()  # before the EMAs exist
    assert ch["state"].iloc[310] == 1 and ch["state"].iloc[399] == -1
    assert (ch["ema_high"] > ch["ema_close"]).iloc[60:].all()


def test_excess_momentum_leaves_out_the_latest_month_and_subtracts_bills():
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(np.linspace(100, 200, 400), index=idx)
    rf = pd.Series(0.0001, index=idx)
    m = excess_momentum(close, rf, months=12, skip=1)
    t = 399
    expected = close.iloc[t - 21] / close.iloc[t - 273] - 1 - ((1.0001) ** 252 - 1)
    assert m.iloc[t] == pytest.approx(expected, rel=1e-9)
    # A jump in the latest month does not move the reading.
    bumped = close.copy()
    bumped.iloc[-5:] *= 1.5
    assert excess_momentum(bumped, rf, 12, 1).iloc[t] == pytest.approx(m.iloc[t])


def test_msci_scores_follow_the_methodology():
    ra6 = pd.Series({"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0})
    ra12 = pd.Series({"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": np.nan})
    sc = msci_scores(ra6, ra12)
    assert list(sc["z"].sort_values(ascending=False).index) == ["A", "B", "C", "D", "E"]
    z = sc["z"]
    assert sc.loc["A", "score"] == pytest.approx(1 + z["A"])
    assert sc.loc["E", "score"] == pytest.approx(1 / (1 - z["E"]))
    assert ((sc["score"] > 0) & (sc["score"] <= 4)).all()
    # Winsorised at +/-3: an outlier's score is capped at 1 + 3.
    wide = msci_scores(pd.Series([100.0] + [0.0] * 20, index=list("ABCDEFGHIJKLMNOPQRSTU")))
    assert wide.loc["A", "z"] > 3 and wide.loc["A", "score"] == pytest.approx(4.0)
    # No 6-month value: left out.
    assert "X" not in msci_scores(pd.Series({"X": np.nan, "A": 1.0, "B": 0.0})).index


def test_weekly_volatility_uses_closed_weeks_only():
    p = _prices(days=900)["close"]
    vol = weekly_volatility(p, years=3)
    later = p.copy()
    later.iloc[-3:] *= 3.0  # a shock in the last days of the current week
    cut = p.index[-4]
    assert weekly_volatility(later, years=3).loc[:cut].equals(vol.loc[:cut])


def test_drawdown_and_rsi():
    close = pd.Series([100, 110, 99, 88, 120.0], index=pd.bdate_range("2020-01-01", periods=5))
    assert drawdown_from_high(close).tolist() == pytest.approx([0, 0, -0.1, -0.2, 0])
    rsi = wilder_rsi(pd.Series(np.linspace(1, 50, 60)), 2)
    assert rsi.iloc[-1] == 100.0 and rsi.isna().iloc[0]


# ------------------------------------------------------------------ rules


def test_latch_keeps_state_and_exit_wins():
    enter = pd.Series([False, True, False, False, True, True])
    exit_ = pd.Series([False, False, False, True, False, True])
    assert latch(enter, exit_).tolist() == [0, 1, 1, 0, 1, 0]


def test_exposure_signals_trade_the_next_bar():
    exp = pd.Series([0, 1, 1, 0, 0.0], index=pd.bdate_range("2020-01-01", periods=5))
    sig = exposure_signals(exp)
    assert sig["entries"].tolist() == [False, False, True, False, False]
    assert sig["exits"].tolist() == [False, False, False, False, True]


def test_month_end_mask_skips_an_unfinished_month():
    idx = pd.bdate_range("2024-01-25", "2024-03-12")
    mask = month_end_mask(idx)
    assert list(idx[mask.to_numpy()].strftime("%Y-%m-%d")) == ["2024-01-31", "2024-02-29"]


def test_trend_filter_buffer_delays_the_reentry():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(np.r_[np.full(20, 100.0), np.full(10, 99.0), [100.5, 102.0], np.full(28, 102.0)], index=idx)
    frame = pd.DataFrame({"close": close, "high": close})
    plain = TrendFilter(window=10).exposure(frame)
    buffered = TrendFilter(window=10, buffer=0.02).exposure(frame)
    assert plain.iloc[25] == 0 and buffered.iloc[25] == 0  # below the average: cash
    assert plain.iloc[30] == 1 and buffered.iloc[30] == 0  # 100.5 clears the average (99.15) but not by 2%
    assert buffered.iloc[31] == 1


def test_fast_engine_exits_on_volatility_and_an_inverted_vix_curve():
    p = _prices(days=600)
    frame = p.assign(vix=15.0, vix3m=18.0)
    calm = TrendFilter(window=50, vol_cap=5.0, vix_term=True)
    base = calm.exposure(frame)
    inverted = frame.copy()
    inverted.loc[inverted.index[400]:, "vix"] = 25.0
    out = calm.exposure(inverted)
    assert out.iloc[401:].eq(0).all() and out.iloc[:400].equals(base.iloc[:400])
    assert TrendFilter(window=50, vol_cap=0.01).exposure(frame).iloc[60:].eq(0).all()  # nothing is that calm


@pytest.mark.parametrize("rule", [
    TrendFilter(window=50, buffer=0.01, vol_cap=0.3), TrendFilter(window=50, check="monthly"), PBEmaChannel(50),
    AbsoluteMomentum(6, 1), MSCIMomentumTrend(), DipInUptrend(trend_window=50), DrawdownDip(0.05, 60),
])
def test_rules_never_read_the_future(rule):
    p = _prices(days=1200)
    p["rf"] = 0.0001
    base = rule.exposure(p)
    future = p.copy()
    future.iloc[900:, :4] *= np.linspace(0.5, 2.0, 300)[:, None]
    assert rule.exposure(future).iloc[:900].equals(base.iloc[:900])


def test_dip_rules():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(np.r_[np.linspace(100, 150, 250), [140, 130, 125], np.linspace(126, 160, 47)], index=idx)
    frame = pd.DataFrame({"close": close, "high": close})
    dd = DrawdownDip(dip=0.10, lookback=100).exposure(frame)
    assert dd.iloc[249] == 0 and dd.iloc[252] == 1 and dd.iloc[-1] == 0  # bought below 135, sold at a new high
    rsi2 = DipInUptrend(entry=10, trend_window=250, exit_window=5).exposure(frame)
    assert rsi2.iloc[251] == 1 and rsi2.iloc[-1] == 0


def test_registry_entries_build_with_their_defaults():
    ohlcv = _prices(days=600)[["open", "high", "low", "close", "volume"]]
    for key in ("trend_filter", "pb_ema", "abs_momentum", "msci_momentum", "dip_rsi2", "dip_drawdown"):
        e = STRATEGY_REGISTRY[key]
        sig = e["class"](**e["defaults"]).generate_signals(ohlcv)
        assert set(sig.columns) == {"entries", "exits"} and sig.index.equals(ohlcv.index)
        assert not sig["entries"].iloc[0]


# ------------------------------------------------------------------ engine


def test_simulate_lags_and_charges_costs():
    idx = pd.bdate_range("2020-01-01", periods=6)
    exp = pd.Series([1, 1, 1, 1, 1, 1.0], index=idx)
    asset = pd.Series(0.01, index=idx)
    rf = pd.Series(0.0, index=idx)
    nxt = systematic.simulate(exp, asset, rf, systematic.EXECUTION_LAG["next_close"], cost_bps=10)
    same = systematic.simulate(exp, asset, rf, systematic.EXECUTION_LAG["same_close"], cost_bps=10)
    assert nxt["held"].tolist() == [0, 0, 1, 1, 1, 1] and same["held"].tolist() == [0, 1, 1, 1, 1, 1]
    assert nxt["ret"].iloc[2] == pytest.approx(0.01 - 0.001) and nxt["ret"].iloc[3] == pytest.approx(0.01)


def test_perf_and_trades():
    idx = pd.bdate_range("2020-01-01", periods=504)
    r = pd.Series((1.10 ** (1 / 252)) - 1, index=idx)
    s = systematic.perf(r)
    assert s["cagr"] == pytest.approx(0.10, abs=1e-9) and s["max_drawdown"] == 0 and math.isnan(s["calmar"])
    held = pd.Series([0, 1, 1, 0, 1.0], index=idx[:5])
    ret = pd.Series([0, 0.1, 0.1, 0, -0.05], index=idx[:5])
    trades = systematic.trade_list(held, ret, cost_bps=0)
    assert [round(t["return"], 4) for t in trades] == [0.21, -0.05] and trades[1]["open"]
    st = systematic.trade_stats(trades, 1.0)
    assert st["win_rate"] == 0.5 and st["expectancy"] == pytest.approx((0.21 - 0.05) / 2)


def test_splice_meets_the_fund_on_its_first_day():
    index = _prices(days=500, start="2019-01-01")
    fund = _prices(days=300, seed=4, start="2019-10-01")
    fund["close"] = fund["close"] * 0.5
    spliced = systematic.splice_index(fund, index)
    first = fund.index[0]
    assert spliced.loc[first, "close"] == fund.loc[first, "close"]
    before = spliced.index[spliced.index.get_loc(first) - 1]
    ratio = spliced.loc[before, "close"] / spliced.loc[first, "close"]
    assert ratio == pytest.approx(index.loc[before, "close"] / index.loc[first, "close"])
    assert set(spliced["source"]) == {"index", "fund"}


def test_synthetic_leverage_and_the_real_fund(offline):
    m = systematic.load_market("QQQ", leverage=3, vehicle="TQQQ", fee=0.01, spread=0.005)
    rf = m.frame["rf"]
    day = m.frame.index[100]
    expected = 3 * m.base_ret.loc[day] - 2 * (rf.loc[day] + 0.005 / 252) - 0.01 / 252
    assert m.asset_ret.loc[day] == pytest.approx(expected)
    after = m.frame.index[m.frame.index > m.first_real][5]
    assert m.asset_ret.loc[after] == pytest.approx(offline["TQQQ"]["adj_close"].pct_change().loc[after])
    assert m.tracking and "tracking_error" in m.tracking
    assert m.vehicle_label == "3x QQQ (TQQQ)" and any("TQQQ from" in n for n in m.notes)


def test_extend_with_the_index(offline):
    m = systematic.load_market("QQQ", extend=True)
    assert m.frame.index[0] == offline["^NDX"].index[0] and any("^NDX" in n for n in m.notes)


def test_run_rule_end_to_end(offline):
    out = systematic.run_rule("trend_filter", {"window": 150}, "QQQ", leverage=3)
    json.dumps(out)  # JSON-safe
    keys = [r["key"] for r in out["rows"]]
    assert keys == ["rule", "buy_hold", "rule_1x", "signal_hold", "cash"]
    rule, bh = out["rows"][0], out["rows"][1]
    assert rule["stats"]["max_drawdown"] > bh["stats"]["max_drawdown"]  # the 40% crash is mostly avoided
    assert out["sweep"]["summary"]["settings"] == 20 and sum(r["is_default"] for r in out["sweep"]["rows"]) == 1
    assert out["walk_forward"]["picks"] and out["bootstrap"]["paths"] == systematic.BOOTSTRAP_PATHS
    assert out["verdict"]["level"] in ("strong", "mixed", "weak") and out["verdict"]["lines"]
    assert out["now"]["reason"].startswith("close")
    again = systematic.run_rule("trend_filter", {"window": 150}, "QQQ", leverage=3)
    assert again["bootstrap"] == out["bootstrap"]  # seeded


def test_run_rule_rejects_bad_settings(offline):
    with pytest.raises(ValueError, match="between"):
        systematic.run_rule("trend_filter", {"window": 5})
    with pytest.raises(ValueError, match="Unknown rule"):
        systematic.run_rule("nope")


def test_grid_keeps_the_users_other_settings():
    combos = systematic.grid_combos("two_engine", systematic.validate_params("two_engine", {"vol_window": 40}))
    assert len(combos) == 25 and all(c["vol_window"] == 40 and c["vix_term"] for c in combos)


def test_block_bootstrap_is_paired():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 1000), index=idx)
    boot = systematic.block_bootstrap(r, r, paths=200)
    assert boot["rule"] == boot["buy_hold"] and boot["p_higher_cagr"] == 0.0


# ------------------------------------------------------------------ savings and rotation


def test_savings_at_once_compounds_each_deposit(offline):
    out = systematic.run_savings("SPY", thresholds=(0.10, 0.85))
    json.dumps(out)
    now, ten, never = out["rows"]
    assert now["cash_share"] == 0.0 and now["whole"]["still_waiting"] == 0.0
    assert ten["dip_buys"] >= 1 and 0 < ten["cash_share"] < 1  # the crash is a 10% dip
    assert never["whole"]["still_waiting"] == 1.0 and never["dip_buys"] == 0  # never 85% down: all in T-bills
    assert never["whole"]["irr"] == pytest.approx(0.02, abs=0.003)


def test_savings_values_math():
    asset = np.array([1.0, 1.1, 1.21, 1.331])
    cash = np.array([1.0, 1.0, 1.0, 1.0])
    v = systematic._savings_values(np.array([0, 1]), np.array([0, 2]), asset, cash, end=3)
    assert v.tolist() == pytest.approx([1.331, 1.1])


def test_rotation_picks_the_trending_fund(offline, monkeypatch):
    frames = {f"F{i}": _prices(days=2200, seed=20 + i, drift=0.0, vol=0.004) for i in range(5)}
    frames["F0"] = _prices(days=2200, seed=40, drift=0.001, vol=0.004)
    frames["SPY"] = _prices(days=2200, seed=41)
    monkeypatch.setattr(systematic, "load_prices", lambda tickers: {t: frames[t] for t in tickers if t in frames})
    out = systematic.run_rotation(tickers=list(frames)[:5], top=1, review="quarterly", adhoc=False)
    json.dumps(out)
    held = [h["holdings"][0]["ticker"] for h in out["reviews"]]
    assert held.count("F0") > 0.8 * len(held)
    assert out["rows"][0]["stats"]["cagr"] > out["rows"][2]["stats"]["cagr"]
    assert len(out["sweep"]) == 8 and out["latest"][0]["ticker"] == "F0"


# ------------------------------------------------------------------ the alert


def test_trend_check_and_monitor_alert(monkeypatch):
    import config.settings as settings

    monkeypatch.setattr(settings, "TREND_ALERT_RULES", ("trend_filter",))
    frames = {"SPY": _prices(days=800, seed=2, drift=0.001, vol=0.004)}
    crashed = frames["SPY"].copy()
    crashed.iloc[-30:, :5] *= np.linspace(1.0, 0.6, 30)[:, None]
    import data.macro_data as macro
    import data.price_history as ph

    monkeypatch.setattr(macro, "load_macro_history", lambda years: macro.MacroHistory({}, {}, ""))
    monkeypatch.setattr(systematic, "_fred_long", lambda *a, **k: pd.Series(dtype=float))
    monkeypatch.setattr(ph, "fetch_full_history", lambda tickers, max_age_days=0: frames)
    calm = systematic.trend_check(["SPY"], ("trend_filter", "pb_ema"))
    frames["SPY"] = crashed
    stressed = systematic.trend_check(["SPY"], ("trend_filter", "pb_ema"))
    assert stressed["markets"][0]["rules"]["trend_filter"]["in_market"] is False
    assert stressed["markets"][0]["readings"]["from_high"] < -0.3

    assert calm["markets"][0]["rules"]["trend_filter"]["in_market"] is True
    before = monitor.snapshot(None, None, "2026-10-08T22:45:00", calm)
    after = monitor.snapshot(None, None, "2026-10-09T22:45:00", stressed)
    alerts = monitor.compare(before, after)
    assert len(alerts) == 1 and alerts[0]["scope"] == "market" and alerts[0]["kind"] == "risk"
    assert alerts[0]["title"] == "SPY to cash (Price above its moving average)"
    assert alerts[0]["link"].endswith("#backtest=trend") and "next close" in alerts[0]["fix"]
    assert set(after["trend"]) == {"SPY|trend_filter"}  # the PB EMA is shown in the view, not alerted on
    assert monitor.compare({"checked_at": "x", "holdings": {}}, after) == []  # first trend reading: a baseline


def test_run_check_includes_the_trend_watch(monkeypatch):
    import config.settings as settings

    monkeypatch.setattr(settings, "TREND_WATCH", ["SPY"])
    monkeypatch.setattr(settings, "TREND_ALERT_RULES", ("pb_ema",))
    reading = {"markets": [{"ticker": "SPY", "rules": {"pb_ema": {"label": "PB EMA channel", "in_market": True,
                                                                   "reason": "Close above both lines.", "date": "2026-10-08"}}}]}
    first = monitor.run_check(lambda p: {"positions": []}, lambda t: {"rows": []}, trend_fn=lambda: reading)
    assert first["markets"] == 1
    reading["markets"][0]["rules"]["pb_ema"] = {**reading["markets"][0]["rules"]["pb_ema"], "in_market": False}
    second = monitor.run_check(lambda p: {"positions": []}, lambda t: {"rows": []}, trend_fn=lambda: reading)
    assert [a["title"] for a in second["alerts"]] == ["SPY to cash (PB EMA channel)"]


# ------------------------------------------------------------------ API


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_api_options_and_validation(client):
    opts = client.get("/api/systematic/options").get_json()
    assert opts["rule_order"][0] == "trend_filter" and "two_engine" in opts["rules"]
    bad = client.post("/api/systematic/run", json={"study": "rule", "rule": "trend_filter", "params": {"window": 1}})
    assert bad.status_code == 400 and "between" in bad.get_json()["error"]
    assert client.post("/api/systematic/run", json={"study": "nope"}).status_code == 400
    assert client.post("/api/systematic/run", json={"signal": "bad ticker!"}).status_code == 400
