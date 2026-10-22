"""Competitive position study: statistics, blend and acceptance verdict on a toy panel (offline, synthetic data)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import yfinance

import config.settings as settings
from analysis.competitive_metrics import SCORING_VERSION
from core import competitive_study as study

DATES = pd.date_range("2011-01-31", periods=60, freq="ME")
TICKERS = [f"T{i:03d}" for i in range(120)]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch: pytest.MonkeyPatch):
    def blocked(*_a, **_k):
        raise RuntimeError("Network access is disabled in tests")

    monkeypatch.setattr(yfinance, "Industry", blocked, raising=False)
    monkeypatch.setattr(yfinance, "Sector", blocked, raising=False)
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "cache")


def _returns(seed: int = 1, scale: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.01, scale, (len(DATES), len(TICKERS))), index=DATES, columns=TICKERS)


def _panel(extra: dict[str, np.ndarray] | None = None, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(DATES) * len(TICKERS)
    frame = pd.DataFrame({
        "date": np.repeat(DATES, len(TICKERS)), "ticker": np.tile(TICKERS, len(DATES)),
        "sector": np.tile(["A", "B", "C", "D"], n // 4),
        "composite": rng.normal(0, 20, n), "fund_score": rng.normal(0, 20, n), "fund_conf": 70.0, "fund_quality": "LIVE",
        "sh_score": rng.normal(0, 10, n), "sh_conf": 60.0, "sh_quality": "LIVE",
    })
    for name, values in (extra or {}).items():
        frame[name] = values
    return frame


def _stacked(frame: pd.DataFrame) -> np.ndarray:
    return frame.stack().to_numpy()


def test_feature_equal_to_future_return_passes_ic_in_every_period_and_noise_does_not():
    fwd12 = _returns()
    rng = np.random.default_rng(3)
    signal = _stacked(fwd12) + rng.normal(0, 0.05, fwd12.size)
    noise = rng.normal(0, 1, fwd12.size)
    panel = _panel({"signal": signal, "noise": noise})
    good = study.summarize(study.monthly_ic(panel, fwd12, "signal"), 12)
    for period in ("whole", "first_half", "second_half"):
        assert good[period]["mean"] > 0.3 and good[period]["t"] > 2 and good[period]["months"] >= 29
    bad = study.summarize(study.monthly_ic(panel, fwd12, "noise"), 12)
    assert not study._passes(bad)  # pure noise cannot pass A1
    assert study._passes(good)


def test_partial_ic_of_a_copy_of_the_control_is_about_zero_and_an_independent_signal_is_recovered():
    fwd = _returns(4)
    rng = np.random.default_rng(5)
    control = rng.normal(0, 1, fwd.size)
    independent = _stacked(fwd) + rng.normal(0, 0.05, fwd.size)
    panel = _panel({"control": control, "copy": control, "mix": control + 0.5 * independent / independent.std() * control.std()})
    same = study.partial_ic(panel, fwd, "copy", ["control"])
    assert abs(same.mean()) < 0.06
    mixed = study.partial_ic(panel, fwd, "mix", ["control"])
    assert mixed.mean() > 0.1 and study.newey_west_t(mixed.to_numpy(), 12) > 2


def test_quintile_spread_sign_and_annualization():
    fwd1 = _returns(6)
    rng = np.random.default_rng(7)
    panel = _panel({"good": _stacked(fwd1) + rng.normal(0, 0.03, fwd1.size), "backwards": -_stacked(fwd1)})
    up = study.quintile_spread(panel, fwd1, "good")
    assert (up > 0).mean() > 0.95 and up.mean() > 0.05
    assert study.quintile_spread(panel, fwd1, "backwards").mean() < 0
    summary = study.summarize_spread(up)
    assert summary["whole"]["annual"] == pytest.approx((1 + up.mean()) ** 12 - 1)
    assert summary["first_half"]["months"] + summary["second_half"]["months"] == summary["whole"]["months"]


def _cp_panel(cp_share: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    n = len(DATES) * len(TICKERS)
    cp = rng.normal(0, 10, n)
    cp[rng.random(n) > cp_share] = np.nan
    return _panel({"cp_score": cp, "cp_conf": 70.0})


def test_blend_at_share_zero_is_the_composite_and_rows_without_a_competitive_score_never_change():
    panel = _cp_panel()
    pd.testing.assert_series_equal(study.blend(panel, 0.0), panel["composite"])
    missing = panel["cp_score"].isna()
    for share in (0.10, 0.20):
        mixed = study.blend(panel, share)
        pd.testing.assert_series_equal(mixed[missing], panel.loc[missing, "composite"])
        assert not np.allclose(mixed[~missing], panel.loc[~missing, "composite"])


def test_a_larger_share_moves_the_composite_further_toward_the_competitive_score():
    panel = _cp_panel(1.0)
    panel["cp_score"] = 30.0  # strongly positive everywhere
    low, high = study.blend(panel, 0.10), study.blend(panel, 0.20)
    assert (high - panel["composite"]).mean() > (low - panel["composite"]).mean() > 0


def _block(mean, t):
    return {"mean": mean, "t": t, "months": 60}


def _passing_stats():
    ic = {p: _block(0.03, 2.5) for p in ("whole", "first_half", "second_half")}
    spread = {p: {**_block(0.002, 1.0), "annual": 0.03} for p in ("whole", "first_half", "second_half")}
    grid = {"shares": {"0": {"excess_first_half": 0.0, "excess_second_half": 0.0},
                       "0.1": {"excess_first_half": 0.01, "excess_second_half": 0.006},
                       "0.2": {"excess_first_half": 0.012, "excess_second_half": 0.004}}}
    return ic, spread, ic, grid


def test_verdict_passes_only_when_all_four_checks_hold():
    ic, spread, partial, grid = _passing_stats()
    ok = study.verdict(ic, spread, partial, grid)
    assert ok["verdict"] == "pass" and ok["failing"] == [] and ok["chosen_share"] == 0.1
    assert ok["adopted_owner_weight"] == pytest.approx(0.1111, abs=1e-4)
    weak_half = {**ic, "second_half": _block(0.03, 1.9)}  # t below 2 in one half
    assert study.verdict(weak_half, spread, partial, grid)["failing"] == ["A1"]
    negative_spread = {**spread, "first_half": {**_block(-0.001, -0.5), "annual": -0.01}}
    assert study.verdict(ic, negative_spread, partial, grid)["failing"] == ["A2"]
    assert study.verdict(ic, spread, {**partial, "whole": _block(-0.01, -3)}, grid)["failing"] == ["A3"]
    losing = {"shares": {**grid["shares"], "0.1": {"excess_first_half": 0.01, "excess_second_half": -0.002},
                         "0.2": {"excess_first_half": 0.02, "excess_second_half": -0.001}}}
    failed = study.verdict(ic, spread, partial, losing)
    assert failed["verdict"] == "fail" and failed["failing"] == ["A4"] and failed["adopted_owner_weight"] == 0.0


def test_the_grid_picks_the_best_weaker_half_and_the_smaller_share_when_within_three_tenths_of_a_point():
    def grid(ten, twenty):
        return {"shares": {"0": {"excess_first_half": 0.0, "excess_second_half": 0.0},
                           "0.1": {"excess_first_half": ten[0], "excess_second_half": ten[1]},
                           "0.2": {"excess_first_half": twenty[0], "excess_second_half": twenty[1]}}}

    assert study.choose_share(grid((0.010, 0.004), (0.012, 0.009))) == (0.2, True)  # clearly better weaker half
    assert study.choose_share(grid((0.010, 0.006), (0.012, 0.008))) == (0.1, True)  # within 0.3 points: the smaller
    assert study.choose_share(grid((0.010, -0.001), (0.012, -0.002))) == (0.1, False)  # no share beats 0 in both halves


def test_features_are_built_from_the_facts_known_on_each_month_end(monkeypatch: pytest.MonkeyPatch):
    from tests.test_competitive import _stmts, _years

    class FakeFacts:
        def filing_dates(self):
            return [np.datetime64("2011-03-01"), np.datetime64("2012-03-01")]

        def statements_asof(self, date, max_years=4):
            n = int(bisect(self.filing_dates(), np.datetime64(date)))
            if n == 0:
                return _stmts([]).__class__(ticker="X", available=False)
            return _stmts(_years(n=5, end=f"{2009 + n}-12-31", rev=1000.0 * n))

    from bisect import bisect

    monkeypatch.setattr(study, "load_company_facts", lambda *a, **k: FakeFacts())
    monkeypatch.setattr(study, "load_margin_facts", lambda *a, **k: pd.DataFrame(columns=["concept", "start", "end", "val", "filed", "form"]))
    dates = [pd.Timestamp("2011-01-31"), pd.Timestamp("2011-06-30"), pd.Timestamp("2012-06-30")]
    rows = study._features_for_ticker(("X", 1, (), dates))
    assert [r["date"] for r in rows] == dates[1:]  # no 10-K before March 2011
    assert rows[0]["fy_end"] == "2010-12-31" and rows[1]["fy_end"] == "2011-12-31"
    assert rows[0]["gross_margin"] == pytest.approx(0.40) and rows[1]["revenue_usd"] == pytest.approx(2000.0)


def _records(n: int) -> dict:
    return {f"S{i}": {"name": f"Company {i} Inc", "sic": 3674, "cik": 1000 + i, "sic_description": "Semiconductors"} for i in range(n)}


def test_score_panel_gives_each_company_peers_from_the_same_month_without_itself():
    from tests.test_competitive import _years
    from analysis.competitive_metrics import company_metrics

    records = _records(10)
    universe, feats = [], []
    for date in DATES[:2]:
        for i, ticker in enumerate(records):
            universe.append({"date": date, "ticker": ticker, "market_cap": 1e10 * (1 + i / 20)})
            m = company_metrics(_years(gm=0.30 + 0.03 * i, dpo=40 + 3 * i), 1.0, None, ticker)
            feats.append({"date": date, "ticker": ticker, **{f: getattr(m, f) for f in study.FEATURE_FIELDS}})
    scores = study.score_panel(pd.DataFrame(universe), pd.DataFrame(feats), records)
    assert len(scores) == 20 and set(scores["scoring_version"]) == {SCORING_VERSION}
    assert (scores["n_peers"] == 8).all() and (scores["peer_level"] == "sic4").all() and not scores["wide_band"].any()
    best, worst = scores[scores.ticker == "S9"].iloc[0], scores[scores.ticker == "S0"].iloc[0]
    assert best["cp_score"] > worst["cp_score"]  # the higher margin and later payment score higher
    assert best["gm_gap"] > 0 > worst["gm_gap"]
    capped = study.score_panel(pd.DataFrame(universe), pd.DataFrame(feats), records, max_peers=5)
    assert (capped["n_peers"] == 5).all()


def test_the_toy_study_runs_end_to_end_within_three_seconds():
    fwd1 = fwd12 = _returns(9)  # one month ahead and twelve months ahead are the same toy returns
    rng = np.random.default_rng(11)
    n = fwd1.size
    cp = _stacked(fwd12) * 0.5 + rng.normal(0, 0.05, n)
    extra = {"cp_score": cp, "cp_conf": 70.0, "cp_suppliers": rng.normal(0, 1, n), "cp_customers": rng.normal(0, 1, n),
             "cp_standing": rng.normal(0, 1, n), "n_peers": 6, "peer_level": "sic4", "wide_band": False}
    extra.update({d: rng.normal(0, 1, n) for d in study.DIAGNOSTICS})
    panel = _panel(extra)
    started = time.monotonic()
    report = study.analyze(panel, {"fwd1": fwd1, "fwd12": fwd12})
    assert time.monotonic() - started < 3.0
    assert report["verdict"]["checks"]["A1"] and report["verdict"]["checks"]["A2"] and report["verdict"]["checks"]["A3"]
    assert set(report["blend"]["shares"]) == {"0", "0.1", "0.2"} and report["protocol"]["run_date"]
    assert {"cp_score", "gm_gap", "share_change"} <= set(report["features"])
    assert report["coverage"]["scored_rows"] == len(panel)


def test_cached_figures_rebuild_when_the_panel_grows():
    panel = _panel()
    signature = study.panel_signature(panel)
    cached = pd.DataFrame({"features_version": [study.FEATURES_VERSION], "panel_signature": [signature]})
    assert study._cache_fits(cached, "features_version", study.FEATURES_VERSION, signature)
    longer = pd.concat([panel, panel.assign(date=panel["date"] + pd.offsets.MonthEnd(60))], ignore_index=True)
    wider = pd.concat([panel, panel.assign(ticker="NEW")], ignore_index=True)
    for grown in (longer, wider):
        assert not study._cache_fits(cached, "features_version", study.FEATURES_VERSION, study.panel_signature(grown))
    assert not study._cache_fits(cached.drop(columns="panel_signature"), "features_version", study.FEATURES_VERSION, signature)  # older cache
    assert not study._cache_fits(cached.assign(features_version=0), "features_version", study.FEATURES_VERSION, signature)
    assert not study._cache_fits(None, "features_version", study.FEATURES_VERSION, signature)
