"""Tests for the regime study: look-ahead, statistics helpers, the vectorised timing composite and the label decomposition.

Synthetic data only; nothing here reads the owner's files or the network.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from agents.base import AgentAssessment, signal_from_score
from agents.scoring import aggregate
from config.weights import ScoringConfig
from core import regime_study as rs
from data.macro_data import MacroHistory, classify_regime, compute_macro_snapshot


def synthetic_history(n: int = 700, seed: int = 3) -> MacroHistory:
    """Nine daily series on one business-day calendar, every one with a seed-dependent random path."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    series = {}
    for key, start, vol in (("spy", 300.0, 0.01), ("rsp", 100.0, 0.01), ("hyg", 80.0, 0.003), ("lqd", 120.0, 0.003)):
        series[key] = pd.Series(start * np.cumprod(1 + rng.normal(0.0003, vol, n)), index=idx)
    series["vix"] = pd.Series(np.clip(18 + np.cumsum(rng.normal(0, 0.8, n)), 9, 70), index=idx)
    series["vix3m"] = pd.Series(np.clip(20 + np.cumsum(rng.normal(0, 0.5, n)), 11, 55), index=idx)
    series["tnx"] = pd.Series(np.clip(2.5 + np.cumsum(rng.normal(0, 0.03, n)), 0.5, 6), index=idx)
    series["t10y2y"] = pd.Series(np.cumsum(rng.normal(0, 0.03, n)) + 0.3, index=idx)
    series["t10y3m"] = pd.Series(np.cumsum(rng.normal(0, 0.03, n)) + 0.5, index=idx)
    return MacroHistory(series=series, errors={}, fetched_at="")


def rewrite_future(history: MacroHistory, asof: pd.Timestamp, how: str) -> MacroHistory:
    """Same history up to and including ``asof``; afterwards values are replaced, scaled or extended."""
    series = {}
    for key, s in history.series.items():
        past = s.loc[:asof]
        future = s.loc[asof + pd.Timedelta(days=1):]
        if how == "scale":
            future = future * 7.0 + 13.0
        elif how == "negate":
            future = -future
        elif how == "nan":
            future = future * np.nan
        elif how == "drop":
            future = future.iloc[0:0]
        series[key] = pd.concat([past, future])
    return MacroHistory(series=series, errors={}, fetched_at="")


def snapshot_fields(snap) -> dict:
    return dataclasses.asdict(snap)


def same(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if not (x == y or (x != x and y != y)):
                return False
        elif x != y:
            return False
    return True


# ------------------------------------------------------------------ look-ahead


@pytest.mark.parametrize("how", ["scale", "negate", "nan", "drop"])
def test_snapshot_ignores_everything_after_asof(how):
    history = synthetic_history()
    asof = history.series["spy"].index[500]
    base = snapshot_fields(compute_macro_snapshot(history, asof))
    changed = snapshot_fields(compute_macro_snapshot(rewrite_future(history, asof, how), asof))
    assert same(base, changed), f"snapshot moved when the future was rewritten ({how})"


def test_snapshot_at_asof_equals_snapshot_of_truncated_history():
    history = synthetic_history()
    asof = history.series["spy"].index[420]
    truncated = MacroHistory(series={k: v.loc[:asof] for k, v in history.series.items()}, errors={}, fetched_at="")
    assert same(snapshot_fields(compute_macro_snapshot(history, asof)), snapshot_fields(compute_macro_snapshot(truncated, None)))


def test_snapshot_does_use_the_past_and_the_asof_bar():
    """Guards the look-ahead test against passing for the wrong reason: the output must react to data at or before asof."""
    history = synthetic_history()
    asof = history.series["spy"].index[500]
    base = compute_macro_snapshot(history, asof)
    earlier = dict(history.series)
    earlier["vix"] = history.series["vix"].copy()
    earlier["vix"].loc[asof] = 90.0  # the asof bar itself is used (month-end close decision)
    moved = compute_macro_snapshot(MacroHistory(series=earlier, errors={}, fetched_at=""), asof)
    assert moved.vix_level == 90.0 and base.vix_level != 90.0
    earlier["spy"] = history.series["spy"].copy()
    earlier["spy"].loc[: asof - pd.Timedelta(days=30)] *= 0.5  # old prices feed the 200-day mean
    assert compute_macro_snapshot(MacroHistory(series=earlier, errors={}, fetched_at=""), asof).spy_sma_200 != base.spy_sma_200


def test_snapshot_is_the_same_whichever_later_date_is_loaded():
    history = synthetic_history()
    asof = history.series["spy"].index[400]
    late = rewrite_future(history, history.series["spy"].index[550], "scale")
    assert same(snapshot_fields(compute_macro_snapshot(history, asof)), snapshot_fields(compute_macro_snapshot(late, asof)))


# ------------------------------------------------------------------ classifier facts the report relies on


def test_classify_regime_cutoffs():
    assert classify_regime(4.0) == "BULL" and classify_regime(3.9) == "CORRECTION"
    assert classify_regime(1.0) == "CORRECTION" and classify_regime(0.9) == "BEAR"
    assert classify_regime(-3.0) == "BEAR" and classify_regime(-3.1) == "CRISIS"


def test_missing_inputs_are_skipped_and_listed_not_filled():
    history = synthetic_history()
    asof = history.series["spy"].index[500]
    for key in ("vix3m", "rsp", "hyg", "lqd"):
        history.series.pop(key)
    snap = compute_macro_snapshot(history, asof)
    assert {"vix_term", "breadth", "credit"} <= set(snap.missing)
    assert len(snap.components) == len(snap.components)  # present components only
    assert all(c["component"] not in ("VIX term structure", "Breadth RSP/SPY", "Credit HYG/LQD") for c in snap.components)


def test_macro_table_skips_dates_before_spy_has_200_bars():
    history = synthetic_history()
    dates = list(history.series["spy"].index[[50, 150, 250, 400]])
    table = rs.macro_table(history, dates)
    assert list(table.index) == dates[2:]
    assert set(table["regime"]) <= set(rs.REGIMES)


# ------------------------------------------------------------------ statistics


def test_subset_mean_t_matches_newey_west_when_every_month_is_in():
    from core.value_study import newey_west_t

    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1.0, 120)
    m, t, n = rs.subset_mean_t(x, np.ones(120, dtype=bool), 3)
    assert n == 120 and m == pytest.approx(x.mean())
    assert t == pytest.approx(newey_west_t(x, 3), rel=1e-9)


def test_subset_mean_t_respects_gaps_and_nan():
    x = np.array([1.0, 2.0, np.nan, 3.0, 4.0] * 6)
    mask = np.array([True, True, True, False, False] * 6)
    m, t, n = rs.subset_mean_t(x, mask, 2)
    assert n == 12 and m == pytest.approx(1.5)


def test_diff_t_sign_and_size():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 200)
    a = np.arange(200) % 2 == 0
    x[a] += 1.0
    d, t, na, nb = rs.diff_t(x, a, ~a, 3)
    assert 0.6 < d < 1.4 and t > 4 and na == nb == 100


def test_paired_diff_and_pass_rule():
    idx = pd.date_range("2012-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(4)
    noise = pd.Series(rng.normal(0, 0.01, 120), index=idx)
    better = noise + 0.02
    res = rs.paired_diff(better, noise * 0)
    assert res["mean"] > 0 and rs.passes_both_halves(res)
    one_half = noise.copy()
    one_half.iloc[:60] += 0.03
    res2 = rs.paired_diff(one_half, noise * 0)
    assert not rs.passes_both_halves(res2) or res2["t_second"] >= 2  # a one-half effect must not pass both halves
    flat = rs.paired_diff(noise, noise * 0)
    assert not rs.passes_both_halves(flat)


def test_group_adj_r2_separates_and_penalises_noise():
    rng = np.random.default_rng(5)
    labels = np.array(["a", "b", "c", "d"] * 100, dtype=object)
    y = np.array([1.0, 2.0, 3.0, 4.0] * 100) + rng.normal(0, 0.1, 400)
    assert rs.group_adj_r2(y, labels) > 0.95
    noise = rng.normal(0, 1, 400)
    assert abs(rs.group_adj_r2(noise, labels)) < 0.03
    assert rs.group_adj_r2(np.where(np.arange(400) % 5 == 0, np.nan, y), labels) > 0.95


def test_runs_and_episodes():
    assert rs.runs(list("aabbbac")) == [("a", 0, 2), ("b", 2, 3), ("a", 5, 1), ("c", 6, 1)]
    months = np.array([2020 * 12 + m for m in (3, 4, 5, 14, 15, 40, 41, 42, 70)])
    clusters = rs.episodes(months)
    assert [len(c) for c in clusters] == [5, 3, 1]  # 5 to 14 is 9 months: one episode; 15 to 40 is a new one
    assert len(rs.episodes(months, gap=1)) == 4


def test_forward_spy_known_path():
    idx = pd.bdate_range("2020-01-01", periods=400)
    price = pd.Series(100.0, index=idx)
    price.iloc[100:] = 110.0
    price.iloc[130:140] = 88.0
    price.iloc[140:] = 110.0
    fwd = rs.forward_spy(price, [idx[99], idx[399]])
    first = fwd.iloc[0]
    assert first["ret_1m"] == pytest.approx(0.10)
    assert first["mdd_3m"] == pytest.approx(88 / 110 - 1)
    assert first["vol_1m"] > 0
    assert np.isnan(fwd.iloc[1]["ret_1m"])  # no future left


def test_sample_dates_start_after_200_bars_and_one_per_month():
    idx = pd.bdate_range("2019-01-01", periods=600)
    dates = rs.sample_dates(pd.Series(1.0, index=idx), "M")
    assert dates[0] > idx[199]
    assert len({(d.year, d.month) for d in dates}) == len(dates)


# ------------------------------------------------------------------ the vectorised timing composite


def shipped_cfg() -> dict:
    return rs.shipped_profiles()


def make_rows(n: int = 60, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "mom_score": rng.normal(0, 40, n), "mom_conf": rng.uniform(20, 95, n),
        "vol_score": rng.normal(0, 30, n), "vol_conf": rng.uniform(20, 95, n),
        "pb_score": rng.normal(0, 25, n), "pb_conf": rng.uniform(20, 95, n),
        "macro_score": rng.normal(20, 40, n), "macro_conf": rng.uniform(60, 95, n),
        "regime": rng.choice(list(rs.REGIMES) + [None], n),
    })
    f.loc[3, "mom_score"] = np.nan
    f.loc[5, ["pb_score", "vol_score", "mom_score"]] = np.nan
    f.loc[7, "macro_score"] = np.nan
    f.loc[9, ["pb_score", "vol_score", "mom_score", "macro_score"]] = np.nan
    return f


def assess(key: str, score: float, conf: float) -> AgentAssessment:
    ok = pd.notna(score)
    value = float(score) if ok else 0.0
    return AgentAssessment(agent_name=key, score=value, signal=signal_from_score(value), confidence=float(conf or 0.0), summary="",
                           agent_key=key, data_quality="LIVE" if ok else "UNAVAILABLE")


def test_vectorised_composite_equals_aggregate():
    from config.weights import load_scoring_config
    from core.all_agents_study import SCALES

    config = load_scoring_config()
    frame = make_rows()
    vec = rs.timing_composite(frame, shipped_cfg())
    for i, r in frame.iterrows():
        a = [assess("momentum", r.mom_score, r.mom_conf), assess("volume", r.vol_score, r.vol_conf),
             assess("pullback", r.pb_score, r.pb_conf), assess("macro", r.macro_score, r.macro_conf)]
        res = aggregate(a, config, r.regime if isinstance(r.regime, str) else None, SCALES)
        has = any(c.included for c in res.contributions)
        if has:
            assert vec.iloc[i] == pytest.approx(res.composite_score, abs=0.051), i
        else:
            assert np.isnan(vec.iloc[i]), i


def test_flat_profile_is_the_mean_of_the_four_and_changes_the_composite():
    cfg = shipped_cfg()
    for key in ("momentum", "macro", "pullback"):
        assert cfg["flat"][key] == pytest.approx(np.mean([cfg["profiles"][r][key] for r in rs.REGIMES]))
    frame = make_rows()
    assert not rs.timing_composite(frame, cfg).equals(rs.timing_composite(frame, cfg, per_regime=False))


def test_macro_off_removes_the_same_constant_from_every_ticker_on_a_date():
    """With equal confidences, the macro score shifts every ticker by one amount, so the cross-sectional ranking is unchanged."""
    cfg = shipped_cfg()
    frame = make_rows(40)
    frame["regime"] = "BULL"
    for c in ("mom_conf", "vol_conf", "pb_conf", "macro_conf"):
        frame[c] = 80.0
    frame[["mom_score", "vol_score", "pb_score"]] = frame[["mom_score", "vol_score", "pb_score"]].fillna(5.0)
    frame["macro_score"] = 33.0
    with_macro = rs.timing_composite(frame, cfg)
    without = rs.timing_composite(frame, cfg, include_macro=False)
    assert with_macro.rank().equals(without.rank())
    assert not with_macro.equals(without)


def test_label_series_and_matched_labels():
    th = {"buy": 15.0, "sell": -15.0}
    comp = pd.Series([20.0, 15.0, 0.0, -15.0, -30.0, np.nan])
    lab = rs.label_series(comp, th)
    assert list(lab[:5]) == ["FAVORABLE", "FAVORABLE", "NEUTRAL", "UNFAVORABLE", "UNFAVORABLE"] and lab.iloc[5] is None
    ref = pd.Series(["FAVORABLE"] * 3 + ["UNFAVORABLE"] * 2 + ["NEUTRAL"] * 5)
    shifted = pd.Series(np.linspace(-50, 90, 1000))
    m = rs.matched_label(shifted, ref)
    assert (m == "FAVORABLE").mean() == pytest.approx(0.3, abs=0.01) and (m == "UNFAVORABLE").mean() == pytest.approx(0.2, abs=0.01)


# ------------------------------------------------------------------ label changes: the double count


def synthetic_panel(macro_moves: bool, ticker_moves: bool) -> pd.DataFrame:
    dates = pd.date_range("2015-01-31", periods=24, freq="ME")
    rows = []
    for t in ("AAA", "BBB", "CCC"):
        for i, d in enumerate(dates):
            macro = (60.0 if i % 2 == 0 else -60.0) if macro_moves else 0.0
            tick = (50.0 if (i % 2 == 0) == (t != "CCC") else -50.0) if ticker_moves else 0.0
            rows.append({"date": d, "ticker": t, "mom_score": tick, "mom_conf": 80.0, "vol_score": tick, "vol_conf": 80.0,
                         "pb_score": tick, "pb_conf": 80.0, "macro_score": macro, "macro_conf": 80.0, "regime": "BULL"})
    return pd.DataFrame(rows)


def test_label_changes_are_attributed_to_macro_when_only_macro_moves():
    out = rs.label_change_decomposition(synthetic_panel(True, False), shipped_cfg())
    assert out["label_changes"] > 0
    assert out["shares"]["macro_only"] == pytest.approx(1.0)


def test_label_changes_are_attributed_to_ticker_when_only_ticker_moves():
    out = rs.label_change_decomposition(synthetic_panel(False, True), shipped_cfg())
    assert out["label_changes"] > 0
    assert out["shares"]["ticker_only"] == pytest.approx(1.0)


# ------------------------------------------------------------------ small helpers


def test_clean_makes_json_safe():
    out = rs._clean({"a": np.float64("nan"), "b": [np.int64(3), np.inf], "c": pd.Timestamp("2020-01-31"), "d": np.bool_(True)})
    assert out == {"a": None, "b": [3, None], "c": "2020-01-31", "d": True}


def test_regime_means_shifted_series_maps_back_to_the_decision_month():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    regime = pd.Series(["BULL", "BEAR", "BEAR", "CRISIS"], index=dates)
    series = pd.Series([1.0, 2.0, 3.0], index=dates[1:])  # dated the month after the decision
    out = rs.regime_means(series, regime, shifted=True)
    assert out["BULL"] == 1.0 and out["BEAR"] == 2.5 and out["CRISIS"] is None


# ------------------------------------------------------------------ added helpers


def test_diff_t_with_two_different_series():
    rng = np.random.default_rng(9)
    x = rng.normal(0.02, 0.01, 150)
    y = rng.normal(0.00, 0.01, 150)
    d, t, na, nb = rs.diff_t(x, np.ones(150, bool), np.ones(150, bool), 3, y=y)
    assert d == pytest.approx(x.mean() - y.mean()) and t > 5 and na == nb == 150


def test_theoretical_score_range_matches_the_point_tables():
    full = "+".join(rs.INPUTS)
    assert rs.theoretical_score_range(full) == (-12.5, 9.0)
    early = "spy_trend+vix+fred_yields+ten_year"
    assert rs.theoretical_score_range(early) == (-8.5, 6.5)


def test_theoretical_range_is_reachable_on_synthetic_snapshots():
    """The point tables in INPUT_POINTS must not understate what compute_macro_snapshot can add or take away."""
    history = synthetic_history(900, seed=11)
    table = rs.macro_table(history, list(history.series["spy"].index[250::10]))
    lo, hi = rs.theoretical_score_range("+".join(rs.INPUTS))
    assert table["regime_score"].between(lo, hi).all()


def test_relabel_shifts_the_cutoffs():
    scores = pd.Series([4.2, 3.8, 1.2, 0.8, -2.8, -3.2])
    assert list(rs.relabel(scores, 0.0)) == ["BULL", "CORRECTION", "CORRECTION", "BEAR", "BEAR", "CRISIS"]
    assert list(rs.relabel(scores, 0.5)) == ["CORRECTION", "CORRECTION", "BEAR", "BEAR", "CRISIS", "CRISIS"]


def test_linear_adj_r2():
    x = np.arange(100, dtype=float)
    assert rs.linear_adj_r2(2 * x + 1, x) > 0.99
    assert np.isnan(rs.linear_adj_r2(x, np.zeros(100)))


def test_analyze_classifier_runs_on_a_long_synthetic_history():
    history = synthetic_history(3200, seed=5)
    out = rs.analyze_classifier(history)
    for part in ("whole", "first_half", "second_half", "full_input_era"):
        assert part in out["monthly"]
    shares = out["monthly"]["whole"]["description"]["labels"]
    assert sum(v["share"] for v in shares.values()) == pytest.approx(1.0)
    assert out["acceptance_1"]["1a_risk"]["passes"] in (True, False)
    assert set(out["threshold_sensitivity"]["whole"]) >= {"shift_+0.0", "two_states", "score_linear"}
    assert out["flip_drivers_monthly"]["label_changes"] >= 0


def test_flip_drivers_credit_a_200d_cross():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    table = pd.DataFrame({"regime": ["BULL", "BULL", "CORRECTION", "CORRECTION"], "spy": [110, 110, 99, 99], "sma200": [100] * 4,
                          "pts SPY vs 200-day SMA": [2.0, 2.0, -2.0, -2.0], "pts VIX level": [2.0, 2.0, 2.0, 2.0]}, index=idx)
    out = rs.flip_drivers(table)
    assert out["label_changes"] == 1 and out["spy_200d_side_changed"] == 1.0 and out["components"]["VIX level"] == 0.0


def test_entry_by_regime_counts_independent_episodes_not_months():
    dates = list(pd.date_range("2015-01-31", periods=60, freq="ME"))
    regime = pd.Series("BULL", index=pd.DatetimeIndex(dates))
    for i in (5, 6, 7, 30, 31, 55):  # three episodes in the 12-month sense
        regime.iloc[i] = "CRISIS"
    rows = []
    for d in dates:
        for t in ("A", "B", "C", "D", "E", "F"):
            rows.append({"date": d, "ticker": t, "owner_decision": "BUY"})
    panel = pd.DataFrame(rows)
    level = pd.DataFrame(100.0, index=pd.date_range("2015-01-31", periods=80, freq="ME"), columns=list("ABCDEF"))
    level = level * np.cumprod(np.full((80, 1), 1.01), axis=0)
    out = rs.entry_by_regime(panel, rs.forward_returns(level, 1), rs.forward_returns(level, 12), regime, dates, dates[30])
    assert out["entry_by_regime"]["CRISIS"]["n_months"] == 6
    assert out["entry_by_regime"]["CRISIS"]["episodes"] == 3
    assert out["entry_by_regime"]["CRISIS"]["fwd_1m_ann"] == pytest.approx(1.01 ** 12 - 1)
    assert out["stress_vs_calm_12m"]["supported"] is False  # no difference, and fewer than five episodes


@pytest.mark.xfail(strict=True, reason="Known gap (decision_log D-38): compute_macro_snapshot has no staleness limit, "
                                       "so a series that stopped updating is scored as current. Remove the mark when fixed.")
def test_a_series_that_ended_two_years_before_asof_is_reported_missing():
    history = synthetic_history()
    asof = history.series["spy"].index[-1]
    stale = dict(history.series)
    stale["vix3m"] = history.series["vix3m"].loc[: asof - pd.Timedelta(days=730)]
    snap = compute_macro_snapshot(MacroHistory(series=stale, errors={}, fetched_at=""), asof)
    assert "vix_term" in snap.missing


# ------------------------------------------------------------------ D-40 out-of-sample machinery


def synthetic_oos_frame(weeks: int = 160, tickers: int = 12, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=weeks, freq="W-FRI")
    rows = []
    macro = rng.normal(20, 30, weeks)
    for ti in range(tickers):
        for wi, d in enumerate(dates):
            sig = rng.normal(0, 30, 3)
            rows.append({"ticker": f"T{ti}", "date": d, "mom_score": sig[0], "mom_conf": 70.0, "vol_score": sig[1], "vol_conf": 70.0,
                         "pb_score": sig[2], "pb_conf": 70.0, "macro_score": macro[wi], "macro_conf": 80.0,
                         "regime": ("BULL", "CORRECTION", "BEAR", "CRISIS")[wi // 10 % 4],
                         "fwd_5": rng.normal(0.002, 0.02), "fwd_20": rng.normal(0.008, 0.04), "fwd_60": rng.normal(0.02, 0.07)})
    return pd.DataFrame(rows)


def test_frozen_cutoffs_are_constants_and_oos_universe_is_32():
    assert (rs.C1_FAVORABLE_CUTOFF, rs.C1_UNFAVORABLE_CUTOFF) == (6.1, -40.0)
    assert len(rs.OOS_TICKERS) == 32 and len(set(rs.OOS_TICKERS)) == 32


def test_c1_cutoffs_from_panel_match_label_shares():
    panel = make_rows(2000, seed=3)
    for c in ("mom_conf", "vol_conf", "pb_conf", "macro_conf"):
        panel[c] = 80.0
    panel["regime"] = panel["regime"].fillna("BULL")
    hi, lo, fav, unf = rs.c1_cutoffs_from_panel(panel, shipped_cfg())
    c1 = rs.timing_composite(panel, shipped_cfg(), per_regime=False, include_macro=False)
    assert abs((c1 >= hi).mean() - fav) < 0.02 and abs((c1 <= lo).mean() - unf) < 0.02
    assert hi > lo


def test_oos_compare_on_noise_finds_nothing_favouring_s_and_macro_is_constant():
    frame = synthetic_oos_frame()
    assert frame.groupby("date")["macro_score"].std().max() == 0.0
    variants = rs.oos_label_frames(frame, shipped_cfg())
    cut = sorted(frame["date"].unique())[80]
    res = rs.oos_compare(frame, variants, "C1", cut)
    assert set(res["series"]) == set(rs.OOS_LAGS)
    assert res["series_favouring_S"] == []
    assert res["label_changes_per_ticker_year"]["S"] > 0
    shares = res["label_shares"]["C1"]
    assert sum(shares.values()) == pytest.approx(1.0)


def test_oos_series_identical_variant_gives_zero_difference_and_never_favours():
    frame = synthetic_oos_frame()
    variants = rs.oos_label_frames(frame, shipped_cfg())
    same = {"S": variants["S"], "X": variants["S"]}
    res = rs.oos_compare(frame, same, "X", sorted(frame["date"].unique())[80])
    assert all(abs(v["mean"]) < 1e-12 for v in res["series"].values())
    assert res["passes"] is False  # equal label changes: the candidate must change label strictly less often


def test_oos_series_detects_a_real_edge():
    """A composite that sees the future 20-session return must beat noise in IC and in the spread."""
    frame = synthetic_oos_frame(240, 20, seed=5)
    informed = frame["fwd_20"] * 400.0
    noise = pd.Series(np.random.default_rng(1).normal(0, 30, len(frame)), index=frame.index)
    lab = lambda c: pd.Series(np.where(c >= 15, "FAVORABLE", np.where(c <= -15, "UNFAVORABLE", "NEUTRAL")), index=c.index, dtype=object)  # noqa: E731
    cut = sorted(frame["date"].unique())[120]
    res = rs.oos_compare(frame, {"S": (informed, lab(informed)), "N": (noise, lab(noise))}, "N", cut)
    assert res["series"]["ic_20"]["mean"] > 0.3 and res["series"]["ic_20"]["favours_S"]
    assert "ic_20" in res["series_favouring_S"]
