"""Your history against the weights: tag mapping, the regime tag (no look-ahead), verdicts, recommendations, panel evidence.

Synthetic data only. Unit tests build tagged rows by hand; the end-to-end test runs the track record on the synthetic
export of test_trade_republic with a mocked macro history (no network).
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

import config.settings as settings
from agents.momentum_agent import MomentumAgent
from agents.pullback_agent import PullbackAgent
from config.weights import REGIMES, load_scoring_config
from core import track_record as trk
from core import weights_review as wr
from data.macro_data import MacroHistory
from tests.test_style_lab import make_market, offline_record, pieces  # noqa: F401  (fixtures)

SPLIT = pd.Timestamp("2024-01-01")


def make_rows(spec: list[tuple[str, str, int, float]], noise: float = 0.01) -> list[dict]:
    """Tagged buys: (tag, bucket, count, mean excess return). Dates alternate between the two halves, one name each."""
    rows: list[dict] = []
    i = 0
    for tag, bucket, n, mean in spec:
        for k in range(n):
            day = (pd.Timestamp("2023-01-15") + pd.DateOffset(months=(k * 2) % 22)).strftime("%Y-%m-%d")
            if k % 2:
                day = (pd.Timestamp("2024-01-15") + pd.DateOffset(months=(k * 2) % 10)).strftime("%Y-%m-%d")
            rows.append({"order": f"o{i}", "isin": f"IS{i}", "name": f"N{i}", "date": day, VALUE: mean + noise * np.sin(i * 1.7),
                         "excess_180": mean, "money_return": mean, "eur_excess_90": 0.0, "tags": {tag: bucket}})
            i += 1
    return rows


VALUE = wr.VALUE


def merge_tags(rows: list[dict], extra: dict[str, str]) -> list[dict]:
    return [{**r, "tags": {**r["tags"], **extra}} for r in rows]


def evidence_for(**overrides) -> dict:
    ev = wr.load_panel_evidence()
    ev = copy.deepcopy(ev)
    ev.update(overrides)
    return ev


# ------------------------------------------------------------------ panel evidence


def test_panel_evidence_file_loads_and_validates():
    ev = wr.load_panel_evidence()
    assert ev["version"] == 1 and {"all_agents", "timing_validation"} <= set(ev["studies"])
    for s in ev["studies"].values():
        assert s["name"] and s["period"] and s["date"]
    assert [b["label"] for b in ev["by_regime"]["buckets"]] == list(REGIMES)
    assert {b["tag_buckets"][0] for b in ev["by_regime"]["buckets"]} == set(REGIMES)
    assert ev["by_regime"]["inside_regime"] == {}  # the slot the regime study fills


def test_panel_evidence_rejects_malformed_blocks():
    good = wr.load_panel_evidence()
    for mutate, message in [
        (lambda d: d.pop("studies"), "studies"),
        (lambda d: d["conditions"]["trend"].update(study="nope"), "unknown study"),
        (lambda d: d["conditions"]["trend"]["buckets"][0].pop("value"), "numeric value"),
        (lambda d: d["by_regime"]["inside_regime"].update({"SIDEWAYS": {}}), "unknown regime"),
        (lambda d: d["agents"]["pullback"].update(study="nope"), "unknown study"),
    ]:
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            wr.validate_panel_evidence(bad)


def test_panel_numbers_are_market_numbers_with_a_study_period_and_date():
    text = json.dumps(wr.load_panel_evidence())
    assert "EUR" not in text and "isin" not in text.lower()
    ev = wr.load_panel_evidence()
    for block in [*ev["conditions"].values(), ev["by_regime"]]:
        assert ev["studies"][block["study"]]["period"] and ev["studies"][block["study"]]["date"]


# ------------------------------------------------------------------ the map of tags to agents


def test_every_mapped_agent_exists_and_every_direction_is_two_sided():
    from agents.registry import get_agent_classes

    known = set(get_agent_classes()) | {wr.OWNER}
    tags = set()
    for line in wr.LINES:
        assert line["agent"] in known
        tags.add(line["tag"])
        if not line.get("derived"):
            plus, minus = wr._split(line["rewards"])
            assert plus and minus and not set(plus) & set(minus)
    # Style tags are not mapped to any agent.
    assert not tags & {"volatility of the name", "size of the order", "when in the build", "positions open at the time"}
    assert {"volatility of the name", "size of the order", "when in the build", "positions open at the time"} <= set(wr.STYLE_TAGS)


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=len(close))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


def test_directions_match_what_the_agents_actually_score():
    base = np.linspace(100, 150, 300)
    # A small dip after a long climb: the pullback agent rewards it; a name stretched above its mean it penalises.
    dip = base.copy()
    dip[-4:] = base[-5] * np.array([0.985, 0.97, 0.955, 0.94])
    stretched = base.copy()
    stretched[-3:] = base[-4] * np.array([1.03, 1.06, 1.09])
    pull = PullbackAgent()
    assert pull.evaluate("T", _ohlcv(dip)).score > pull.evaluate("T", _ohlcv(stretched)).score
    line = next(x for x in wr.LINES if x["agent"] == "pullback" and x["tag"] == "move in the month before")
    assert line["rewards"]["after a 10%+ drop"] > 0 > line["rewards"]["after a 10%+ rise"]
    # Momentum is the opposite on the month's move and contrarian on RSI(14).
    mom = next(x for x in wr.LINES if x["agent"] == "momentum" and x["tag"] == "move in the month before")
    assert mom["rewards"]["after a 10%+ rise"] > 0 > mom["rewards"]["after a 10%+ drop"]
    up, down = np.linspace(100, 125, 120), np.linspace(125, 100, 120)
    agent = MomentumAgent()
    assert agent.evaluate("T", _ohlcv(np.r_[base[:180], up])).raw_data["roc_20d"] > 0 > agent.evaluate("T", _ohlcv(np.r_[base[:180], down])).raw_data["roc_20d"]
    rsi = next(x for x in wr.LINES if x["agent"] == "momentum" and x["tag"] == "RSI(14)")
    assert rsi["rewards"]["oversold (under 35)"] > 0 > rsi["rewards"]["overbought (over 65)"]


def test_the_vix_direction_follows_the_stance(monkeypatch):
    line = next(x for x in wr.LINES if x.get("vix"))
    monkeypatch.setattr(settings, "VIX_STANCE", "contrarian")
    assert wr._vix_rewards(line)["VIX 30 or more"] > 0
    monkeypatch.setattr(settings, "VIX_STANCE", "risk")
    assert wr._vix_rewards(line)["VIX 30 or more"] < 0


def test_the_weights_are_the_live_configuration_shares():
    w = wr._weights_by_unit(load_scoring_config())
    owner = {k: v["owner"] for k, v in w.items() if v["role"] == "owner"}
    assert sum(owner.values()) == pytest.approx(1.0)
    for regime in REGIMES:
        assert sum(v["by_regime"][regime] for v in w.values() if v["by_regime"]) == pytest.approx(1.0)
    assert w["macro"]["by_regime"]["CRISIS"] > w["macro"]["by_regime"]["BULL"]


# ------------------------------------------------------------------ the regime at entry, point in time


def macro_history(crash_from: str = "2024-03-04", end: str = "2025-12-31") -> MacroHistory:
    """SPY climbs and VIX sits at 13 until ``crash_from``; from that day SPY falls hard and VIX jumps to 45."""
    idx = pd.bdate_range("2022-01-03", end)
    n = len(idx)
    spy = np.linspace(400, 520, n)
    vix = np.full(n, 13.0)
    k = int(np.searchsorted(idx, pd.Timestamp(crash_from)))
    spy[k:] = spy[k - 1] * np.linspace(0.97, 0.65, n - k)
    vix[k:] = 45.0
    s = lambda a: pd.Series(a, index=idx)  # noqa: E731
    series = {"spy": s(spy), "vix": s(vix), "vix3m": s(vix * 1.1), "rsp": s(spy * 0.3), "hyg": s(np.full(n, 75.0)),
              "lqd": s(np.full(n, 105.0)), "tnx": s(np.full(n, 4.2)), "t10y2y": s(np.full(n, 0.8)), "t10y3m": s(np.full(n, 1.0))}
    return MacroHistory(series=series, errors={}, fetched_at="now")


def test_regime_at_entry_never_sees_the_buy_day():
    hist = macro_history(crash_from="2024-03-04")
    out = wr.load_macro(["2024-03-04", "2024-03-05", "2024-02-01", "2024-06-03"], loader=lambda _: hist)
    by = out["by_date"]
    # The crash starts on 4 March: a buy that day only knows 1 March, a buy a day later knows the first crash day.
    assert by["2024-02-01"] == "BULL" and by["2024-03-04"] == "BULL"
    assert by["2024-03-05"] != "BULL"
    assert by["2024-06-03"] in ("BEAR", "CRISIS")
    # Same answer as the classifier called by hand on the day before.
    from data.macro_data import compute_macro_snapshot

    assert by["2024-03-05"] == compute_macro_snapshot(hist, pd.Timestamp("2024-03-04")).regime
    assert out["now"]["regime"] in REGIMES and out["agent_scores"]


def test_a_failed_macro_download_leaves_the_tag_out():
    def boom(_):
        raise RuntimeError("offline")

    assert wr.load_macro(["2024-03-04"], loader=boom) is None
    empty = MacroHistory(series={}, errors={"core": "SPY and VIX unavailable"}, fetched_at="now")
    assert wr.load_macro(["2024-03-04"], loader=lambda _: empty) is None


# ------------------------------------------------------------------ verdicts


TREND = "trend"
ABOVE, BELOW = "above its 200-day mean", "below its 200-day mean"


def momentum_trend_verdict(rows):
    gap = wr._gap(rows, TREND, [ABOVE], [BELOW], SPLIT)
    return wr._verdict(gap)


def test_verdict_in_line_when_the_rewarded_buckets_did_better_in_both_halves():
    rows = make_rows([(TREND, ABOVE, 20, 0.06), (TREND, BELOW, 20, -0.02)])
    assert momentum_trend_verdict(rows) == ("in line", "clear")


def test_verdict_disagrees_when_the_penalised_buckets_did_better():
    rows = make_rows([(TREND, ABOVE, 20, -0.03), (TREND, BELOW, 20, 0.05)])
    assert momentum_trend_verdict(rows) == ("your history disagrees", "clear")


def test_verdict_too_few_cases_for_a_thin_sample():
    rows = make_rows([(TREND, ABOVE, 4, 0.06), (TREND, BELOW, 4, -0.02)])
    assert momentum_trend_verdict(rows)[0] == "too few cases"
    assert wr._verdict(None)[0] == "too few cases"
    assert wr._verdict(wr._gap(make_rows([(TREND, ABOVE, 12, 0.05)]), TREND, [ABOVE], [BELOW], SPLIT))[0] == "too few cases"


def test_a_noisy_gap_is_leaning_not_clear():
    rows = make_rows([(TREND, ABOVE, 14, 0.01), (TREND, BELOW, 14, 0.0)], noise=0.2)
    verdict, strength = momentum_trend_verdict(rows)
    assert verdict in ("in line", "your history disagrees") and strength == "leaning"


def test_agent_rows_carry_weights_history_panel_and_verdict():
    rows = make_rows([(TREND, ABOVE, 20, 0.06), (TREND, BELOW, 20, -0.02), ("distance from the 52-week high", "within 5% of the 52-week high", 12, 0.0),
                      ("distance from the 52-week high", "20 to 40% below the high", 12, 0.05)])
    from core.style_profile import _rank_tags

    ranked = _rank_tags(rows, SPLIT)
    agents = wr.build_agent_rows(rows, ranked, SPLIT, load_scoring_config(), wr.load_panel_evidence(), None, "BULL")
    by = {a["key"]: a for a in agents}
    assert agents[0]["key"] == wr.OWNER and {p["agent"] for p in agents[0]["owner_parts"]} >= {"fundamentals", "shareholder"}
    mom = by["momentum"]
    assert mom["verdict"] in ("in line", "mixed") and mom["by_regime"]["BULL"] > 0 and mom["weight_today"] == mom["by_regime"]["BULL"]
    line = next(x for x in mom["lines"] if x["tag"] == TREND)
    assert line["verdict"] == "in line" and line["history"]["best"] == ABOVE and line["rewarded"] == [ABOVE]
    assert mom["panel"]["ic"] < 0  # the panel finds the momentum agent backwards as an entry timer
    # Pullback and momentum read the distance from the high in opposite directions.
    d = lambda k: next(x for x in by[k]["lines"] if x["tag"] == "distance from the 52-week high")  # noqa: E731
    assert d("pullback")["rewarded"] == ["5 to 20% below the high"] and d("momentum")["rewarded"] == ["within 5% of the 52-week high"]
    assert by["volume"]["verdict"] == "not testable" and by["volume"]["lines"] == []
    assert json.dumps(trk._clean(agents))


def test_a_changed_panel_is_read_without_touching_the_code():
    rows = make_rows([(wr.REGIME_TAG, "BULL", 12, 0.0), (wr.REGIME_TAG, "CRISIS", 12, 0.08)])
    base = supported_panel()
    regimes = wr.build_by_regime(rows, SPLIT, base, load_scoring_config())
    crisis = next(r for r in regimes if r["regime"] == "CRISIS")
    assert crisis["panel"]["value"] > next(r for r in regimes if r["regime"] == "BULL")["panel"]["value"]
    ev = copy.deepcopy(base)
    for b in ev["by_regime"]["buckets"]:
        b["value"], b["alt"] = (0.01, 0.01) if b["label"] == "CRISIS" else (0.2, 0.2)
    swapped = wr.build_by_regime(rows, SPLIT, ev, load_scoring_config())
    assert next(r for r in swapped if r["regime"] == "CRISIS")["panel"]["value"] == 0.01
    # The panel now says CRISIS is the worst regime, so today being CRISIS and your history liking it is history only.
    now = {"regime": "CRISIS"}
    old = wr.build_recommendations(rows, [], regimes, SPLIT, base, now, [])["items"][0]
    new = wr.build_recommendations(rows, [], swapped, SPLIT, ev, now, [])["items"][0]
    assert old["kind"] == new["kind"] == "regime"
    assert old["label"] == "Your history and the S&P 500 panel agree"
    assert new["label"] == "Your history only: a hypothesis to test forward"


# ------------------------------------------------------------------ by regime


def test_by_regime_counts_halves_and_thin_regimes():
    rows = make_rows([(wr.REGIME_TAG, "BULL", 30, 0.02), (wr.REGIME_TAG, "CRISIS", 4, 0.09)])
    rows = merge_tags(rows[:30], {"size of the order": "large"}) + rows[30:]
    for k, r in enumerate(rows[:30]):
        r["tags"]["size of the order"] = "large" if k % 2 else "small"
        r[VALUE] = r[VALUE] + (0.05 if k % 2 else 0.0)
    out = wr.build_by_regime(rows, SPLIT, wr.load_panel_evidence(), load_scoring_config())
    by = {r["regime"]: r for r in out}
    assert by["BULL"]["n"] == 30 and by["BULL"]["n_first"] + by["BULL"]["n_second"] == 30 and by["BULL"]["hit_rate"] > 0.5
    assert by["CRISIS"]["too_few"] and by["CRISIS"]["inside"] == [] and by["BEAR"]["n"] == 0 and by["BEAR"]["median"] is None
    inside = by["BULL"]["inside"]
    assert inside and inside[0]["tag"] == "size of the order" and inside[0]["best"] == "large"
    assert by["BULL"]["heaviest_agents"] and by["CRISIS"]["heaviest_agents"][0]["agent"] == "macro"


# ------------------------------------------------------------------ recommendations


def supported_panel():
    """The shipped panel with the regime block marked supported, to exercise the agreement path."""
    ev = copy.deepcopy(wr.load_panel_evidence())
    ev["by_regime"]["supported"] = True
    return ev


def recs_for(rows, now=None, ev=None):
    from core.style_profile import _rank_tags

    ranked = _rank_tags(rows, SPLIT)
    ev = ev or wr.load_panel_evidence()
    regimes = wr.build_by_regime(rows, SPLIT, ev, load_scoring_config())
    agents = wr.build_agent_rows(rows, ranked, SPLIT, load_scoring_config(), ev, None, (now or {}).get("regime"))
    return wr.build_recommendations(rows, ranked, regimes, SPLIT, ev, now, agents)


DEEP, NEAR, MID = "20 to 40% below the high", "within 5% of the 52-week high", "5 to 20% below the high"
HIGH = "distance from the 52-week high"


def test_history_and_panel_agree_is_the_strongest_label():
    # The panel found deep dips the best buckets and a name at its high the worst; so did this history.
    rows = make_rows([(HIGH, DEEP, 20, 0.07), (HIGH, NEAR, 20, -0.01)])
    items = recs_for(rows)["items"]
    top = next(x for x in items if x["tag"] == HIGH)
    assert top["label"] == "Your history and the S&P 500 panel agree" and top["rank"] == 0
    assert top["n"] >= 10 and top["t"] is not None and top["first_half"] is not None and top["second_half"] is not None
    assert "Weights tab" not in top["text"] and "weight" not in top["text"].lower().replace("weights", "")  # acts within the rules


def test_panel_only_when_your_history_is_thin():
    rows = make_rows([(HIGH, DEEP, 4, 0.07), (HIGH, NEAR, 4, -0.01)])
    top = next(x for x in recs_for(rows)["items"] if x["tag"] == HIGH)
    assert top["label"] == "S&P 500 panel only" and "cannot confirm" in top["text"]


def test_history_that_opposes_the_panel_is_a_hypothesis_stated_the_right_way_round():
    rows = make_rows([(HIGH, DEEP, 20, -0.04), (HIGH, NEAR, 20, 0.05)])
    top = next(x for x in recs_for(rows)["items"] if x["tag"] == HIGH)
    assert top["label"] == "Your history only: a hypothesis to test forward"
    assert top["bucket"] == NEAR and "points the other way" in top["text"]


def test_a_style_tag_with_a_clear_pattern_is_history_only():
    rows = make_rows([("size of the order", "about your median size", 20, 0.05), ("size of the order", "large (over twice your median)", 20, -0.03)])
    item = next(x for x in recs_for(rows)["items"] if x["tag"] == "size of the order")
    assert item["label"].startswith("Your history only") and item["bucket"] == "about your median size" and item["link"] is None


def test_a_thin_history_says_keep_what_you_do():
    rows = make_rows([(HIGH, DEEP, 3, 0.07), (HIGH, NEAR, 3, -0.01)])
    items = recs_for(rows)["items"]
    assert {x["label"] for x in items} == {"S&P 500 panel only", "Keep what you do"} and items[-1]["label"] == "Keep what you do"
    none = recs_for(make_rows([("size of the order", "about your median size", 3, 0.05)]))["items"]
    assert none[-1]["label"] == "Keep what you do" and "keep what you do" in none[-1]["text"].lower()


def test_today_in_a_thin_regime_keeps_what_you_do_and_a_clear_one_is_acted_on():
    rows = make_rows([(wr.REGIME_TAG, "BULL", 24, 0.01), (wr.REGIME_TAG, "CRISIS", 3, 0.2)])
    item = recs_for(rows, now={"regime": "CRISIS"})["items"][0]
    assert item["label"] == "Keep what you do" and item["n"] == 3 and "too few" in item["text"]
    rows = make_rows([(wr.REGIME_TAG, "BULL", 24, -0.02), (wr.REGIME_TAG, "CRISIS", 14, 0.09)])
    item = recs_for(rows, now={"regime": "CRISIS"}, ev=supported_panel())["items"][0]
    assert item["label"] == "Your history and the S&P 500 panel agree" and "staging the buy" in item["text"]
    assert item["n"] == 14 and item["t"] is not None and item["first_half"] is not None and item["second_half"] is not None


def test_a_regime_lean_the_study_rejected_is_never_quoted_as_panel_evidence():
    # The shipped panel marks the regime result as not supported (core/regime_study.py): the same history is then
    # history only, and a regime with no clear history difference never gets the "S&P 500 panel only" label.
    rows = make_rows([(wr.REGIME_TAG, "BULL", 24, -0.02), (wr.REGIME_TAG, "CRISIS", 14, 0.09)])
    assert wr.load_panel_evidence()["by_regime"]["supported"] is False
    item = recs_for(rows, now={"regime": "CRISIS"})["items"][0]
    assert item["label"] == "Your history only: a hypothesis to test forward"
    flat = make_rows([(wr.REGIME_TAG, "BULL", 24, 0.01), (wr.REGIME_TAG, "CRISIS", 14, 0.01)])
    item = recs_for(flat, now={"regime": "BULL"})["items"][0]
    assert item["label"] != "S&P 500 panel only"
    agents = wr.build_agent_rows(flat, [], SPLIT, load_scoring_config(), wr.load_panel_evidence(),
                                 {"BULL": {"mean": 50.0, "n": 10}, "CRISIS": {"mean": -40.0, "n": 5}}, "BULL")
    macro = next(a for a in agents if a["key"] == "macro")
    assert all(x["panel"]["direction"] == "none" for x in macro["lines"] if x["tag"] == wr.REGIME_TAG)


def test_a_weight_question_names_the_replay_and_never_changes_a_weight():
    rows = make_rows([(TREND, ABOVE, 20, -0.03), (TREND, BELOW, 20, 0.05)])
    out = recs_for(rows, now={"regime": "BULL"})
    test = next(t for t in out["weight_tests"] if t["agent"] == "momentum")
    assert "both halves" in test["text"] and "out of sample" in test["text"] and "Backtest lab" in test["text"]
    before = load_scoring_config().to_dict()
    recs_for(rows)
    assert load_scoring_config().to_dict() == before


# ------------------------------------------------------------------ end to end on the synthetic export


def test_track_record_carries_the_regime_tag_and_the_review(offline_record, monkeypatch):
    monkeypatch.setattr(wr, "_load_history", lambda first: macro_history(crash_from="2024-05-01"))
    r = trk.run_track_record("fifo", False)
    rows = r["style"]["entries"]["rows"]
    regimes = {x["date"]: x["tags"].get(wr.REGIME_TAG) for x in rows}
    assert regimes["2024-01-03"] == "BULL" and regimes["2024-06-03"] in ("BEAR", "CRISIS")
    review = r["style"]["weights"]
    assert "error" not in review and review["regime_available"]
    assert review["agents"][0]["key"] == wr.OWNER and {a["key"] for a in review["agents"]} >= {"momentum", "pullback", "macro"}
    assert {x["regime"] for x in review["regimes"]} == set(REGIMES)
    assert review["now"]["items"] and review["now"]["items"][-1]["label"]
    assert any(x["tag"] == "size of the order" or x["tag"] == "volatility of the name" for x in review["unread"]) or review["unread"] is not None
    json.dumps(r["style"])  # the stored result must serialise
    from core import style_report

    page = style_report.write_share_report(trk.load_result(), names=False).read_text()
    assert "By market regime" in page and "EUR" not in page and "Weights tab" not in page  # counts and returns only, no advice


def test_without_macro_data_the_rest_of_the_profile_stands(offline_record, monkeypatch):
    def boom(first):
        raise RuntimeError("offline")

    monkeypatch.setattr(wr, "_load_history", boom)
    r = trk.run_track_record("fifo", False)
    assert r["style"] and not r["style"]["weights"]["regime_available"]
    assert all(wr.REGIME_TAG not in x["tags"] for x in r["style"]["entries"]["rows"])
