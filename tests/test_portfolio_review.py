"""The portfolio review by class: classes, rules per class, the to-do list, your plan and the settings it reports
(synthetic data only, no network). Every review test passes profiles and histories, so no result depends on the
machine's profile cache."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import config.settings as settings
from agents.master_agent import MasterDecisionAgent
from agents.scoring import NOT_RATED
from core import portfolio as pf
from data import portfolio as store
from tests.conftest import make_ohlcv
from tests.test_multi_agent import StubAgent
from tests.test_owner_first import _cfg, _fetched, _OwnerByTicker

DD = {"from_high": -0.02, "basis": "since entry"}
NO_ENTRY = {"valid": False, "score": None, "decision": None, "at": None, "kind": "none", "days_after_open": None}


def _rec(decision: str = "HOLD", score: float = 0.0, warnings: list[str] | None = None, setup: str | None = None,
         coverage: float = 1.0, reason: str | None = None, missing: list[str] | None = None) -> SimpleNamespace:
    contributions = [{"layer": "owner", "agent_name": f"{m} Agent", "exclusion_reason": "No usable data"} for m in missing or []]
    return SimpleNamespace(decision=decision, composite_score=score, warnings=warnings or [], timing={"setup": setup},
                           owner={"score": score, "coverage": coverage, "not_rated_reason": reason},
                           contributions=contributions, agent_assessments=[])


def _decide(rec, cls="stock", weight=0.05, drawdown=DD, entry=NO_ENTRY, cash=1_000.0, volatility=None, **kw):
    return pf.decide_action(rec, cls, weight, drawdown, entry, cash, volatility, **kw)


# ------------------------------------------------------------------ classes


@pytest.mark.parametrize("profile, asset_class, name, priced, expected, bucket", [
    ({"asset_type": "Stock"}, "STOCK", "Example Corp", True, "stock", "Satellite"),
    ({"asset_type": "ETF", "name": "Core MSCI World"}, "FUND", None, True, "core", "Core"),
    ({"asset_type": "ETF", "name": "Global Clean Energy"}, "FUND", None, True, "fund", "Satellite"),
    ({"asset_type": "ETF", "name": "S&P 500 Index Fund", "fund_sectors": {"Technology": 0.7, "Healthcare": 0.3}},
     "FUND", None, True, "fund", "Satellite"),
    ({"asset_type": "ETF", "name": "S&P 500 Index Fund", "fund_sectors": {"Technology": 0.3, "Financial Services": 0.2, "Healthcare": 0.5}},
     "FUND", None, True, "core", "Core"),
    ({"asset_type": "Bond", "name": "Euro Government Bond"}, "FUND", None, True, "core", "Core"),
    ({"asset_type": "Commodity", "name": "Physical Gold"}, "FUND", None, True, "commodity", "Metals"),
    ({"asset_type": "Crypto", "name": "Bitcoin ETP"}, "CRYPTO", None, True, "commodity", "Speculative"),
    ({}, "FUND", "Core MSCI World", True, "core", "Core"),
    (None, "DERIVATIVE", "Example Turbo", True, "derivative", "Speculative"),
    ({"asset_type": "Stock"}, "STOCK", "Example Corp", False, "unpriced", "Satellite"),
])
def test_classify_by_profile_name_and_broker_class(profile, asset_class, name, priced, expected, bucket):
    cls = pf.classify(profile, asset_class, name, priced)
    assert cls == expected  # (a 70% technology look-through is a sector bet whatever the name says)
    assert pf.bucket_of_class(cls, profile) == bucket


def test_classify_physical_gold_listed_as_an_equity_is_a_commodity():
    from data.profiles import build_profile

    profile = build_profile("GOLD.DE", {"quoteType": "EQUITY", "longName": "Example Physical Gold ETC"}, name="Example Physical Gold ETC")
    assert profile["asset_type"] == "Commodity" and pf.classify(profile, "STOCK", None) == "commodity"


# ------------------------------------------------------------------ rules per class


def test_core_fund_is_never_reviewed_or_trimmed():
    rec = _rec(NOT_RATED, reason="No owner rating", warnings=["Low trading liquidity: about $1M traded per day."])
    v = _decide(rec, "core", weight=0.45, expense_ratio=0.0020)
    assert (v.action, v.rule) == ("HOLD", "core_fund") and "no owner score by design" in v.reasons[0] and "0.20%" in v.reasons[0]
    assert "key information document" in _decide(rec, "core", weight=0.45).reasons[0]  # no cost published
    assert "cap_exempt" in pf.limits() and pf.limits()["cap_exempt"] == ["core"]


def test_other_fund_and_commodity_keep_the_cap():
    rec = _rec("HOLD", 0.0)
    for cls, rule in (("fund", "other_fund"), ("commodity", "tracker")):
        over = _decide(rec, cls, weight=0.25, total_value=10_000.0, currency="EUR")
        assert (over.action, over.rule) == ("TRIM", "above_cap") and "Sell about €500" in over.fix
        ok = _decide(rec, cls, weight=0.10)
        assert (ok.action, ok.rule) == ("HOLD", rule)
    assert "one commodity" in _decide(rec, "commodity", weight=0.25).reasons[0]


def test_tracker_ignores_a_shareholder_only_owner_layer():
    # A tracker listed as an equity gets a "score" of 0 from the liquidity rule alone: 55% coverage, HOLD.
    rec = _rec("HOLD", 0.0, coverage=0.55, warnings=["Red flag: reverse split in the last 10 years (see Shareholders & Management)."])
    tracker = _decide(rec, "commodity", weight=0.05)
    assert (tracker.action, tracker.rule) == ("HOLD", "tracker")  # not REVIEW, and a red flag is no exit for a trust
    stock = _decide(_rec("HOLD", 0.0, coverage=0.55, missing=["Fundamentals"]))
    assert (stock.action, stock.rule) == ("REVIEW", "thin_data") and "missing: Fundamentals" in stock.reasons[0]


def test_owner_sell_and_red_flag_exit_unchanged():
    flag = "Red flag: reverse split in the last 10 years (see Shareholders & Management)."
    sell = _decide(_rec("SELL", -20.0))
    assert (sell.action, sell.rule) == ("EXIT", "owner_sell") and "exit rule says sell" in sell.reasons[0] and "T-14" not in sell.reasons[0]
    red = _decide(_rec("HOLD", 5.0, warnings=[flag]))
    assert (red.action, red.rule) == ("EXIT", "red_flag") and red.reasons[0].startswith("Red flag") and "exit on its own" in red.reasons[0]
    both = _decide(_rec("STRONG_SELL", -40.0, warnings=[flag]))
    assert both.rule == "owner_sell" and both.reasons[1] == flag
    assert _decide(_rec("HOLD", 0.0, warnings=[flag]), "commodity").action == "HOLD"


def test_exit_says_when_it_was_sell_at_entry():
    entry = {"valid": True, "score": -20.0, "decision": "SELL", "at": "replay 2024-03-31", "kind": "replay", "days_after_open": 0}
    v = _decide(_rec("SELL", -22.0), entry=entry)
    assert "already at or below the exit level when you bought (March 2024)" in v.reasons[0]
    assert "already at or below" not in _decide(_rec("SELL", -22.0), entry={**entry, "valid": False}).reasons[0]


def test_drawdown_trigger_scales_with_the_holdings_own_volatility():
    assert pf._drawdown_trigger(None) == 0.25 and pf._drawdown_trigger(0.28) == 0.25  # the 25% look level holds up to about 29%
    assert pf._drawdown_trigger(0.40) == pytest.approx(0.33, abs=0.005)
    assert pf._drawdown_trigger(0.60) == pytest.approx(0.45, abs=0.005)
    assert pf._drawdown_trigger(1.00) == pytest.approx(0.63, abs=0.005)
    calm = make_ohlcv(periods=300, start=100, end=100, noise=0.005, seed=1)
    wild = make_ohlcv(periods=300, start=100, end=100, noise=0.06, seed=2)
    assert pf._own_volatility(calm) < 0.2 and pf._own_volatility(wild) > 0.7
    assert pf._own_volatility(make_ohlcv(periods=60, start=100, end=100, noise=0.01)) is None  # under 26 weeks


def test_drawdown_review_is_volatility_scaled_and_hold_band_only():
    fall = {"from_high": -0.40, "basis": "since entry"}
    calm = _decide(_rec("HOLD", 5.0), drawdown=fall, volatility=0.15)
    assert (calm.action, calm.rule) == ("REVIEW", "drawdown")
    assert _decide(_rec("HOLD", 5.0), drawdown=fall, volatility=0.90).action == "HOLD"  # a normal year for this stock
    buy = _decide(_rec("BUY", 30.0), drawdown=fall, volatility=0.15)
    assert (buy.action, buy.rule) == ("HOLD", "thesis_intact") and "2011-2026" in buy.reasons[0]


def test_thesis_drop_needs_a_rating_from_entry(monkeypatch):
    stored = {"ticker": "AAA", "opened": "2025-01-10", "owner_score_at_entry": 30.0, "owner_decision_at_entry": "BUY",
              "rated_at": "2025-01-12T10:00:00"}
    ref = pf._entry_ref(stored)
    assert ref["kind"] == "stored" and ref["valid"] and ref["days_after_open"] == 2
    drop = _decide(_rec("HOLD", 5.0), entry=ref)
    assert (drop.action, drop.rule) == ("REVIEW", "thesis_drop") and "fell 25 points" in drop.reasons[0]

    late = {"ticker": "BBB", "opened": "2025-01-10"}
    monkeypatch.setattr(pf, "get_first_after", lambda t, d: {"owner_score": 30.0, "decision": "BUY", "generated_at": "2025-05-10T10:00:00"})
    ref = pf._entry_ref(late)
    assert ref["kind"] == "first_after" and not ref["valid"] and ref["days_after_open"] == 120
    assert _decide(_rec("HOLD", 5.0), entry=ref).action == "HOLD"
    monkeypatch.setattr(pf, "get_first_after", lambda t, d: None)
    assert pf._entry_ref(late)["kind"] == "none"
    assert pf._entry_ref({"ticker": "C", "owner_score_at_entry": 1.0, "rated_at": "replay 2024-12-31", "opened": "2025-01-10"})["valid"]


def test_excluded_sector_holds_without_a_standing_review():
    reason = "Not rated: Utilities is outside the value framework (owner's call, decision_log.md T-12). The debt rules exclude them."
    v = _decide(_rec(NOT_RATED, reason=reason))
    assert (v.action, v.rule) == ("HOLD", "outside_framework") and v.reasons[0].startswith("Not rated: Utilities is outside")
    deep = _decide(_rec(NOT_RATED, reason=reason), drawdown={"from_high": -0.40, "basis": "since entry"}, volatility=0.15)
    assert (deep.action, deep.rule) == ("REVIEW", "drawdown")
    nodata = _decide(_rec(NOT_RATED, reason="Not rated: no owner-layer data."))
    assert (nodata.action, nodata.rule) == ("REVIEW", "no_statements")


def test_room_to_add_without_a_pullback():
    buy = _rec("BUY", 30.0, setup="neutral")
    v = _decide(buy, weight=0.04)
    assert v.action == "HOLD" and pf._room_to_add(buy, "stock", v.action, 0.04, 1_000.0) == {"weight": 0.04, "start_weight": 0.10, "cash": 1_000.0}
    pull = _rec("BUY", 30.0, setup="pullback_in_uptrend")
    assert _decide(pull, weight=0.04).action == "ADD"
    assert _decide(buy, weight=0.04, cash=0.0).action == "HOLD" and pf._room_to_add(buy, "stock", "HOLD", 0.04, 0.0) is None
    assert _decide(pull, weight=0.04, cash=0.0).action == "HOLD"


def test_add_rule_unchanged():
    assert _decide(_rec("BUY", 30.0, setup="pullback_in_uptrend"), weight=0.12).action == "HOLD"  # above the starting size


# ------------------------------------------------------------------ to-do, plan, alerts


def _stock_row(ticker, value, weight, action="HOLD", cls="stock", decision="HOLD", **extra):
    row = {"ticker": ticker, "name": ticker, "class": cls, "class_label": pf.CLASSES[cls]["label"], "bucket": pf.CLASSES[cls]["bucket"],
           "value": value, "weight": weight, "action": action, "rule": "x", "why": f"{ticker} why", "fix": "fix", "decision": decision,
           "rated": cls == "stock" and decision != NOT_RATED, "pnl": 0.0, "shares": 1.0, "avg_cost": value, "room_to_add": None}
    return {**row, **extra}


def test_todo_groups_order_and_counts():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}),
                                 agents=[_OwnerByTicker({"BAD": -80.0, "BIG": 40.0, "OKAY": 5.0, "GOOD": 40.0}), StubAgent("mom", 0)])
    fetched = _fetched({"BAD": (50, 40), "BIG": (100, 100), "OKAY": (20, 21), "GOOD": (100, 105), "NODATA": (30, 31), "FUND": (300, 310)})
    portfolio = {"cash": 5_000.0, "currency": "USD", "positions": [
        {"ticker": "BAD", "shares": 100, "avg_cost": 55.0}, {"ticker": "BIG", "shares": 1000, "avg_cost": 90.0},
        {"ticker": "OKAY", "shares": 100, "avg_cost": 20.0}, {"ticker": "GOOD", "shares": 5, "avg_cost": 100.0},
        {"ticker": "NODATA", "shares": 10, "avg_cost": 30.0}, {"ticker": "FUND", "shares": 20, "avg_cost": 300.0, "name": "Core MSCI World"},
        {"ticker": "GONE", "shares": 50, "avg_cost": 20.0}]}
    profiles = {"FUND": {"asset_type": "ETF", "name": "Core MSCI World"}}
    res = pf.review_portfolio(portfolio, master=master, record=False, fetched=fetched, profiles=profiles, histories={})
    keys = [g["key"] for g in res["todo"]]
    order = ["sell", "trim", "look", "add", "price", "room", "rest"]
    assert keys == sorted(keys, key=order.index) and {"sell", "trim", "look", "price", "rest"} <= set(keys)
    assert {"add", "room"} & set(keys)  # GOOD is rated BUY at a tiny weight with cash: ADD on a pullback, else room to add
    for g in res["todo"]:
        assert g["count"] == len(g["items"]) and all(i["why"] and i["fix"] for i in g["items"])
        if g["key"] == "rest":
            assert all(i["action"] == "HOLD" for i in g["items"])
        elif g["action"]:
            assert all(i["action"] == g["action"] for i in g["items"])
    sell = next(g for g in res["todo"] if g["key"] == "sell")
    assert [i["ticker"] for i in sell["items"]] == ["BAD"] and sell["weight"] == pytest.approx(sell["items"][0]["weight"])
    assert next(g for g in res["todo"] if g["key"] == "price")["items"][0]["ticker"] == "GONE"
    rest = next(g for g in res["todo"] if g["key"] == "rest")
    assert "FUND" in [i["ticker"] for i in rest["items"]] and rest["counts"]["core"] == 1
    assert [c["key"] for c in res["classes"]][:2] == ["core", "stock"] and "Satellite" in res["plan"]["split"]


def test_plan_drift_and_budget():
    rows = [_stock_row("CORE", 1_300.0, 0.13, cls="core"), _stock_row("STK", 8_700.0, 0.87)]
    low = pf._plan_checks(rows, [], 0.0, 10_000.0, {"core_target": 0.6, "derivatives_budget": None}, None)
    drift = next(c for c in low["checks"] if c["key"] == "drift")
    assert drift["status"] == "look" and "Send new money and savings plans to the core fund" in drift["fix"]
    assert low["drift"] == pytest.approx(-0.47, abs=0.01) and low["split"]["Core"] == pytest.approx(0.13)
    rows[0]["value"], rows[1]["value"] = 5_800.0, 4_200.0
    assert next(c for c in pf._plan_checks(rows, [], 0.0, 10_000.0, {"core_target": 0.6}, None)["checks"] if c["key"] == "drift")["status"] == "ok"
    none = pf._plan_checks(rows, [], 0.0, 10_000.0, {}, None)
    assert next(c for c in none["checks"] if c["key"] == "drift")["status"] == "na"
    over = pf._plan_checks(rows, [], 0.0, 10_000.0, {"core_target": 0.2}, None)
    assert "pause the core savings plan" in next(c for c in over["checks"] if c["key"] == "drift")["fix"]
    spent = pf._plan_checks(rows, [], 0.0, 10_000.0, {"derivatives_budget": 500.0},
                            {"bought": 800.0, "sold": 0.0, "open_cost": 300.0, "open_count": 1, "year": 2026})
    assert next(c for c in spent["checks"] if c["key"] == "budget")["status"] == "look"
    nobudget = pf._plan_checks(rows, [], 0.0, 10_000.0, {}, {"bought": 0.0, "open_cost": 300.0, "open_count": 1})
    assert next(c for c in nobudget["checks"] if c["key"] == "budget")["status"] == "na"
    unpriced = [{"ticker": "DERIV", "value": 500.0, "class": "derivative", "bucket": "Speculative"}]
    assert pf._plan_checks(rows, unpriced, 0.0, 10_500.0, {}, None)["split"]["Speculative"] == pytest.approx(500 / 10_500, abs=1e-3)


def test_alerts_are_placed_and_deduplicated():
    stats = {"total_value": 100_000.0, "positions": 4, "single_positions": 4, "effective_positions": 1.5, "largest_weight": 0.40,
             "largest_ticker": "BIG", "avg_correlation": 0.7, "max_correlation": None, "owner_sell_weight": 0.2, "not_rated_weight": 0.1,
             "unpriced_value": 1_000.0, "cash_weight": 0.6, "sectors": {"Technology": 0.6}}
    alerts = pf._summary_alerts(stats, room=["AAA", "BBB"])
    titles = {a["title"]: a for a in alerts}
    assert "Rated SELL" not in titles and "Outside the framework" not in titles and "Counted at cost" not in titles
    assert titles["Above the position cap"]["where"] == "todo" and titles["Mostly cash"]["where"] == "plan"
    assert titles["One sector dominates"]["where"] == "risk" and "AAA, BBB" in titles["Few real bets"]["fix"]
    assert all(a["fix"] for a in alerts)
    groups = pf._todo([_stock_row("BIG", 40_000.0, 0.40, "TRIM"), _stock_row("BAD", 5_000.0, 0.05, "EXIT")],
                      [], [{"kind": "risk", "where": "todo", "ticker": "BAD", "title": "BAD drives the swings", "text": "BAD is 5% but 30% of the risk.", "fix": "f"},
                           titles["Above the position cap"]], {"checks": []}, 100_000.0, "EUR")
    sell = next(g for g in groups if g["key"] == "sell")
    assert "Also most of the portfolio's swings" in sell["items"][0]["risk_note"]
    assert titles["Above the position cap"]["attached"]


def test_effective_positions_exclude_core_funds():
    rows = [_stock_row("CORE", 60_000.0, 0.60, cls="core", decision=NOT_RATED)] + [_stock_row(f"S{i}", 8_000.0, 0.08) for i in range(5)]
    stats = pf._portfolio_stats(rows, 0.0)
    assert stats["effective_positions"] == pytest.approx(5.0) and stats["single_positions"] == 5
    assert stats["largest_ticker"] != "CORE" and stats["not_rated_weight"] == 0.0
    assert not any(a["title"] == "Few real bets" for a in pf._summary_alerts(stats))


def test_settings_and_freshness():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}), agents=[_OwnerByTicker({}), StubAgent("mom", 0)])
    used = pf._settings_used(master)
    assert used["saved"] is False and used["owner_weights"] == {"owner": 1.0}
    assert used["buy"] > 0 > used["sell"] and used["limits"]["max_position_weight"] == settings.MAX_POSITION_WEIGHT
    rows = [{"ticker": "A", "freshness": {"status": "LIVE", "as_of": "2026-01-02"}},
            {"ticker": "B", "freshness": {"status": "CLOSE", "as_of": "2026-01-01"}},
            {"ticker": "C", "freshness": {"status": "CLOSE", "as_of": "2026-01-01"}},
            {"ticker": "D", "freshness": {"status": "STALE", "as_of": "2025-12-20"}}]
    fresh = pf._freshness(rows, "2026-01-02T18:00:00")
    assert (fresh["live"], fresh["close"], fresh["stale"]) == (1, 2, ["D"])
    assert fresh["oldest_price"] == "2025-12-20" and fresh["newest_price"] == "2026-01-02"
    # Live rows are dated later than the closes (a European listing live, a US one at the previous close): the close day is the closes' own.
    assert fresh["close_days"] == {"2026-01-01": 2}
    mixed = pf._freshness(rows + [{"ticker": "E", "freshness": {"status": "CLOSE", "as_of": "2026-01-02"}}], "2026-01-02T18:00:00")
    assert mixed["close_days"] == {"2026-01-01": 2, "2026-01-02": 1}


def test_unpriced_import_holdings_join_at_cost():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}), agents=[_OwnerByTicker({"GOOD": 40.0}), StubAgent("mom", 0)])
    fetched = _fetched({"GOOD": (100, 120)})
    extra = [{"isin": "DE000TEST001", "name": "Example Turbo", "asset_class": "DERIVATIVE", "shares": 10.0, "cost": 500.0,
              "status": "unmapped", "reason": "knock-out product: no public price history"}]
    res = pf.review_portfolio({"cash": 1_000.0, "currency": "USD", "positions": [{"ticker": "GOOD", "shares": 100, "avg_cost": 100.0}]},
                              master=master, record=False, fetched=fetched, profiles={"GOOD": {"asset_type": "Stock"}}, histories={},
                              extra_unpriced=extra)
    deriv = res["unpriced"][0]
    assert deriv["class"] == "derivative" and deriv["source"] == "import" and deriv["value"] == 500.0
    assert res["summary"]["total_value"] == pytest.approx(res["positions"][0]["value"] + 500.0 + 1_000.0, rel=1e-6)
    price = next(g for g in res["todo"] if g["key"] == "price")["items"][0]
    assert "counted at cost ($500)" in price["why"] and "Set a derivatives budget" in price["fix"]
    assert {"derivative"} <= {c["key"] for c in res["classes"]} and res["plan"]["split"]["Speculative"] > 0


def test_review_row_carries_the_class_rule_and_trigger_fields():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}), agents=[_OwnerByTicker({}), StubAgent("mom", 0)])
    fetched = _fetched({"GLD": (150, 160), "WRLD": (90, 99)})
    profiles = {"GLD": {"asset_type": "Commodity", "name": "Physical Gold", "expense_ratio": 0.0012},
                "WRLD": {"asset_type": "ETF", "name": "Core MSCI World", "expense_ratio": 0.0020}}
    res = pf.review_portfolio({"cash": 0.0, "currency": "USD", "positions": [
        {"ticker": "GLD", "shares": 5, "avg_cost": 150.0}, {"ticker": "WRLD", "shares": 50, "avg_cost": 90.0}]},
        master=master, record=False, fetched=fetched, profiles=profiles, histories={})
    rows = {r["ticker"]: r for r in res["positions"]}
    assert rows["GLD"]["class"] == "commodity" and rows["GLD"]["rule"] == "tracker" and rows["GLD"]["rated"] is False
    assert rows["WRLD"]["class"] == "core" and rows["WRLD"]["weight"] > settings.MAX_POSITION_WEIGHT and rows["WRLD"]["action"] == "HOLD"
    assert rows["WRLD"]["rating_note"] and rows["WRLD"]["expense_ratio"] == 0.0020 and rows["WRLD"]["notices"] == []
    assert rows["WRLD"]["concerns"] == [] and {c["key"] for c in rows["WRLD"]["checks"]} >= {"cost", "overlap"}
    assert rows["GLD"]["entry_ref"]["kind"] == "none" and rows["GLD"]["drawdown_trigger"] >= settings.DRAWDOWN_REVIEW
    assert res["rules_version"] == pf.RULES_VERSION and res["plan"]["split"]["Core"] > 0.5
    assert not any(a["title"].endswith("drives the swings") and a["ticker"] == "WRLD" for a in res["alerts"])


# ------------------------------------------------------------------ candidate


def test_candidate_top_up_for_a_held_buy_and_fund_class_sentence():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}), agents=[_OwnerByTicker({"HELD": 40.0}), StubAgent("mom", 0)])
    fetched = _fetched({"HELD": (100, 110), "WRLD": (90, 99), "NEWF": (50, 55)}, noise=0.002)
    portfolio = {"cash": 20_000.0, "currency": "USD", "positions": [
        {"ticker": "HELD", "shares": 10, "avg_cost": 100.0}, {"ticker": "WRLD", "shares": 100, "avg_cost": 90.0}]}
    profiles = {"HELD": {"asset_type": "Stock"}, "WRLD": {"asset_type": "ETF", "name": "Core MSCI World"},
                "NEWF": {"asset_type": "ETF", "name": "Global Equity Index", "expense_ratio": 0.0007}}
    held = pf.assess_candidate("HELD", portfolio, master=master, record=False, fetched=fetched, profiles=profiles)
    assert held["held"] and held["class"] == "stock" and held["top_up"] and held["top_up"]["shares"] > 0
    room = settings.NEW_POSITION_WEIGHT * held["portfolio_value"] - held["position"]["value"]
    assert held["top_up"]["constraints"]["position cap"] == int(room / held["current_price"])  # the cap counts what you hold
    assert held["position"]["value"] + held["top_up"]["value"] <= settings.NEW_POSITION_WEIGHT * held["portfolio_value"] + 1.0
    fund = pf.assess_candidate("NEWF", portfolio, master=master, record=False, fetched=fetched, profiles=profiles)
    assert fund["verdict"] == "NOT_RATED" and fund["class"] == "core" and fund["sizing"] is None
    assert "no owner score by design" in fund["reasons"][0] and fund["checks"][0]["key"] == "cost" and fund["checks"][0]["status"] == "ok"


# ------------------------------------------------------------------ store and API


def test_plan_survives_reimport_and_validation():
    store.save_portfolio({"cash": 0.0, "currency": "EUR", "positions": [{"ticker": "BBB", "shares": 1, "avg_cost": 10}]})
    assert store.set_plan({"core_target": 0.6, "derivatives_budget": 500})["plan"] == {"core_target": 0.6, "derivatives_budget": 500.0}
    stored, _ = store.replace_imported([{"ticker": "CCC", "shares": 2, "avg_cost": 5}], 100.0, "EUR", "trade_republic")
    assert stored["plan"]["core_target"] == 0.6 and store.load_portfolio()["plan"]["derivatives_budget"] == 500.0
    assert store.save_portfolio({"cash": 5.0, "positions": []})["plan"]["core_target"] == 0.6  # a payload without a plan keeps it
    assert store.set_plan({"derivatives_budget": None})["plan"] == {"core_target": 0.6, "derivatives_budget": None}
    for bad in ({"core_target": 1.5}, {"core_target": -0.1}, {"derivatives_budget": -5}, {"core_target": "lots"}):
        with pytest.raises(store.PortfolioError):
            store.set_plan(bad)


def test_set_notes_keeps_the_position():
    store.save_portfolio({"cash": 0.0, "currency": "EUR", "positions": [{"ticker": "AAA", "shares": 1, "avg_cost": 10, "source": "trade_republic"}]})
    out = store.set_notes("aaa", "kept for the dividend")
    assert out["positions"][0]["notes"] == "kept for the dividend" and out["positions"][0]["source"] == "trade_republic"
    assert len(store.set_notes("AAA", "x" * 900)["positions"][0]["notes"]) == 500
    with pytest.raises(LookupError):
        store.set_notes("NOPE", "x")


def test_api_plan_and_notes():
    from web.app import app

    client = app.test_client()
    client.put("/api/portfolio", json={"cash": 10, "currency": "EUR", "positions": [{"ticker": "AAA", "shares": 1, "avg_cost": 10, "source": "trade_republic"}]})
    res = client.put("/api/portfolio/plan", json={"core_target": 0.5, "derivatives_budget": 250})
    assert res.status_code == 200 and res.get_json()["plan"] == {"core_target": 0.5, "derivatives_budget": 250.0}
    assert "limits" in res.get_json() and client.get("/api/portfolio").get_json()["plan"]["core_target"] == 0.5
    assert client.put("/api/portfolio/plan", json={"core_target": 2}).status_code == 400
    assert client.put("/api/portfolio/plan", json={"derivatives_budget": -1}).status_code == 400
    res = client.put("/api/portfolio/position/AAA/notes", json={"notes": "why I hold it"})
    pos = res.get_json()["positions"][0]
    assert res.status_code == 200 and pos["notes"] == "why I hold it" and pos["source"] == "trade_republic"
    assert client.put("/api/portfolio/position/NOPE/notes", json={"notes": "x"}).status_code == 404
    assert client.put("/api/portfolio/position/AAA/notes", json={"notes": 5}).status_code == 400
    assert client.put("/api/portfolio/position/<bad>/notes", json={"notes": "x"}).status_code in (400, 404)


# ------------------------------------------------------------------ broker import helpers


def test_unpriced_holdings_and_derivatives_activity_from_a_synthetic_export():
    from core import broker_import
    from data import brokers
    from data import trade_republic as tr

    csv = ("date,type,isin,name,shares,price,amount,fee,tax,currency,id,asset_class\n"
           "2026-01-02,deposit,,,,,10000,,,EUR,,\n"
           "2026-02-03,buy,DE000TEST001,Example Turbo,10,50,-500,-1,,EUR,,derivative\n"
           "2026-03-05,sell,DE000TEST001,Example Turbo,4,60,240,,,EUR,,derivative\n"
           "2025-12-05,buy,DE000TEST002,Old Turbo,5,20,-100,,,EUR,,derivative\n"
           "2026-04-01,buy,LU0000000009,Example Private Fund,1,100,-100,,,EUR,,fund\n")
    parsed = brokers.parse("generic", csv.encode())
    for e in parsed["events"]:
        if e.get("isin") == "LU0000000009":
            e["asset_class"] = "PRIVATE_FUND"
    tr.save_ledger(parsed)
    held = broker_import.unpriced_holdings()
    assert {h["isin"] for h in held} == {"DE000TEST001", "DE000TEST002"}  # the private fund is left out of the review
    turbo = next(h for h in held if h["isin"] == "DE000TEST001")
    assert turbo["asset_class"] == "DERIVATIVE" and turbo["shares"] == pytest.approx(6.0) and turbo["cost"] > 0
    act = broker_import.derivatives_activity(2026)
    assert act["bought"] == pytest.approx(501.0) and act["sold"] == pytest.approx(240.0) and act["open_count"] == 2
    assert broker_import.derivatives_activity(2025)["bought"] == pytest.approx(100.0)


# ------------------------------------------------------------------ fund cost in profiles


def test_profile_carries_the_fund_cost_as_a_share_and_refreshes_an_old_cache_once(tmp_path, monkeypatch):
    import time

    from data import profiles

    assert profiles.build_profile("F", {"quoteType": "ETF", "longName": "Core MSCI World", "netExpenseRatio": 0.2})["expense_ratio"] == pytest.approx(0.002)
    assert profiles.build_profile("F", {"quoteType": "ETF", "longName": "Core MSCI World"})["expense_ratio"] is None
    assert profiles.build_profile("S", {"quoteType": "EQUITY", "longName": "Example Corp", "netExpenseRatio": 0.2})["expense_ratio"] is None

    path = tmp_path / "profiles.json"
    monkeypatch.setattr(profiles, "_path", lambda: path)
    old = {"F": {"info": {"quoteType": "ETF", "longName": "Core MSCI World"}, "fund_sectors": None, "bond_share": None, "_at": time.time()},
           "S": {"info": {"quoteType": "EQUITY", "longName": "Example Corp"}, "fund_sectors": None, "bond_share": None, "_at": time.time()}}
    path.write_text(__import__("json").dumps(old))
    calls = []

    def fake_fetch(ticker):
        calls.append(ticker)
        return {"quoteType": "ETF", "longName": "Core MSCI World", "netExpenseRatio": 0.2}, None, None

    monkeypatch.setattr(profiles, "_fetch", fake_fetch)
    out = profiles.get_profiles([{"ticker": "F"}, {"ticker": "S"}])
    assert calls == ["F"] and out["F"]["expense_ratio"] == pytest.approx(0.002)  # the fund is looked up once, the stock is not
    profiles.get_profiles([{"ticker": "F"}])
    assert calls == ["F"]  # now cached with the key (even when Yahoo has no value)


# ------------------------------------------------------------------ review fixes


def test_capped_stock_is_rated_hold_and_says_why():
    rec = _rec("HOLD", 15.1)
    rec.owner.update({"capped": True, "max_ev_to_fcf": 35.0, "thresholds": {"buy": 15.0, "sell": -15.0}})
    verdict = _decide(rec)
    assert verdict.rule == "hold_band" and "+15.1 reaches the entry level" in verdict.reasons[0]
    assert "35x cap" in verdict.reasons[0] and "no new money" in verdict.reasons[0] and "between the entry and exit levels" not in verdict.reasons[0]
    dd = _decide(rec, drawdown={"from_high": -0.3, "basis": "since entry"})
    assert dd.action == "REVIEW" and "between the entry and exit levels" in dd.reasons[0] and "HOLD band" not in dd.reasons[0]


def test_score_near_a_threshold_shows_one_decimal():
    rec = _rec("HOLD", -14.6)
    rec.owner["thresholds"] = {"buy": 15.0, "sell": -15.0}
    assert "-14.6, between the entry and exit levels" in _decide(rec).reasons[0]
    assert "+3, between the entry and exit levels" in _decide(_rec("HOLD", 3.2)).reasons[0]


def test_a_priced_derivative_is_never_judged_on_the_owner_score():
    verdict = _decide(_rec("STRONG_SELL", -30.0), cls="derivative")
    assert (verdict.action, verdict.rule) == ("HOLD", "derivative") and "derivatives budget" in verdict.fix
    assert "fund" not in pf._cost_check({"ticker": "TURBO"}, "derivative")["text"]


def test_rating_note_for_an_excluded_stock_without_a_reason_is_never_a_bare_dot():
    rec = _rec(NOT_RATED, 0.0)
    rec.agent_assessments = [SimpleNamespace(raw_data={"excluded_sector": True})]
    assert pf._rating_note("stock", rec) == "Outside the framework by your choice (T-12): no owner score."
    rec.owner["not_rated_reason"] = "Not rated: utilities are outside the framework (T-12). More."
    assert pf._rating_note("stock", rec) == "Not rated: utilities are outside the framework (T-12)."


def test_plan_checks_follow_a_saved_plan_without_a_new_review():
    assert pf.plan_checks(0.4, 1000.0, None, {})[0]["status"] == "na"
    look = pf.plan_checks(0.4, 1000.0, None, {"core_target": 0.6})[0]
    assert look["status"] == "look" and "plan of 60%" in look["text"]
    assert pf.plan_checks(0.58, 1000.0, None, {"core_target": 0.6})[0]["status"] == "ok"
    over = pf.plan_checks(0.5, 1000.0, {"bought": 300.0, "open_count": 1}, {"derivatives_budget": 200.0})
    assert over[-1]["key"] == "budget" and over[-1]["status"] == "look"
    fresh = pf.refreshed_plan(0.4, 1000.0, {"core_target": 0.6, "derivatives_budget": None})
    assert fresh["drift"] == pytest.approx(-0.2) and fresh["checks"][0]["key"] == "drift" and fresh["core_target"] == 0.6


def test_api_plan_returns_the_refreshed_checks_for_the_last_review():
    from web.app import app

    client = app.test_client()
    client.put("/api/portfolio", json={"cash": 10, "currency": "EUR", "positions": [{"ticker": "AAA", "shares": 1, "avg_cost": 10}]})
    res = client.put("/api/portfolio/plan", json={"core_target": 0.6, "core_share": 0.3, "invested_value": 500.0})
    out = res.get_json()["review_plan"]
    assert res.status_code == 200 and out["checks"][0]["status"] == "look" and out["drift"] == pytest.approx(-0.3)
    assert "review_plan" not in client.put("/api/portfolio/plan", json={"core_target": 0.5}).get_json()
