"""Daily check: change detection between two checks, stored alerts, and the launchd job (synthetic data only)."""

from __future__ import annotations

import plistlib
from datetime import datetime, time

import pytest

from core import monitor
from data.history import save_watchlist
from data.portfolio import save_portfolio


def _review(*rows):
    return {"positions": [{"ticker": t, "action": a, "decision": d, "owner_score": s, "reasons": [f"{t} reason"]}
                          for t, a, d, s in rows], "errors": []}


def _scan(*rows):
    return {"rows": [{"ticker": t, "decision": d, "composite_score": s, "owner_applicable": True} for t, d, s in rows],
            "errors": []}


def test_first_check_is_a_baseline():
    current = monitor.snapshot(_review(("AAA", "HOLD", "HOLD", 5.0)), None, "2026-10-05T22:45:00")
    assert monitor.compare({}, current) == []


def test_action_and_decision_changes_raise_alerts_with_fixes():
    before = monitor.snapshot(_review(("AAA", "HOLD", "HOLD", 5.0), ("BBB", "HOLD", "BUY", 20.0)),
                              _scan(("CCC", "HOLD", 10.0), ("DDD", "BUY", 25.0)), "2026-10-05T22:45:00")
    after = monitor.snapshot(_review(("AAA", "EXIT", "SELL", -18.0), ("BBB", "HOLD", "BUY", 22.0)),
                             _scan(("CCC", "BUY", 16.0), ("DDD", "BUY", 30.0)), "2026-10-06T22:45:00")
    alerts = monitor.compare(before, after)
    assert [a["title"] for a in alerts] == ["AAA: HOLD -> EXIT", "CCC: owner score +16 rose to or above the entry level (+15)"]
    exit_alert, upgrade = alerts
    assert exit_alert["kind"] == "risk" and exit_alert["scope"] == "holding" and "-18" in exit_alert["text"]
    assert "exit rule says sell" in exit_alert["fix"] and exit_alert["link"].endswith("#portfolio=AAA")
    assert upgrade["kind"] == "note" and "Should I enter?" in upgrade["fix"]


def test_held_names_are_not_double_counted_in_the_watchlist():
    snap = monitor.snapshot(_review(("AAA", "HOLD", "HOLD", 5.0)), _scan(("AAA", "BUY", 30.0)), "2026-10-05")
    assert snap["watchlist"] == {}


def test_run_check_stores_state_and_keeps_failed_names(monkeypatch):
    save_portfolio({"cash": 100.0, "currency": "EUR", "positions": [
        {"ticker": "AAA", "shares": 1, "avg_cost": 10}, {"ticker": "BBB", "shares": 1, "avg_cost": 10}]})
    save_watchlist(["AAA", "CCC"])
    seen = {}

    def scan_fn(tickers):
        seen["watch"] = tickers
        return _scan(("CCC", "HOLD", 0.0))

    first = monitor.run_check(lambda p: _review(("AAA", "HOLD", "HOLD", 5.0), ("BBB", "HOLD", "HOLD", 0.0)), scan_fn,
                              now=datetime(2026, 10, 5, 22, 45))
    assert first["baseline"] and first["alerts"] == [] and seen["watch"] == ["CCC"]  # held AAA is reviewed, not scanned

    # BBB fails the next day: it keeps its last state instead of vanishing, and AAA's change alerts once.
    failing = {"positions": _review(("AAA", "REVIEW", "HOLD", -10.0))["positions"],
               "errors": [{"ticker": "BBB", "error": "No market data"}]}
    second = monitor.run_check(lambda p: failing, scan_fn, now=datetime(2026, 10, 6, 22, 45))
    assert [a["title"] for a in second["alerts"]] == ["AAA: HOLD -> REVIEW"]
    assert second["errors"] == ["BBB: No market data"]
    assert monitor.load_state()["holdings"]["BBB"]["action"] == "HOLD"
    assert [a["title"] for a in monitor.recent_alerts()] == ["AAA: HOLD -> REVIEW"]

    third = monitor.run_check(lambda p: _review(("AAA", "REVIEW", "HOLD", -9.0), ("BBB", "HOLD", "HOLD", 0.0)), scan_fn,
                              now=datetime(2026, 10, 7, 22, 45))
    assert third["alerts"] == []


def test_launchd_job_runs_on_weekdays_and_passes_only_named_env():
    job = monitor.build_plist(time(22, 45), python="/venv/bin/python", env={"SEC_USER_AGENT": "app a@b.example"})
    assert job["Label"] == monitor.LABEL
    assert job["ProgramArguments"] == ["/venv/bin/python", "-m", "core.monitor"]
    assert [s["Weekday"] for s in job["StartCalendarInterval"]] == [1, 2, 3, 4, 5]
    assert {s["Hour"] for s in job["StartCalendarInterval"]} == {22}
    assert job["EnvironmentVariables"] == {"SEC_USER_AGENT": "app a@b.example"}
    assert plistlib.loads(plistlib.dumps(job))["Label"] == monitor.LABEL


def test_notify_is_local_and_can_be_switched_off(monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(monitor.sys, "platform", "darwin")
    alert = {"kind": "risk", "title": "AAA: HOLD -> EXIT"}
    monkeypatch.setenv("FASTVALUE_NOTIFY", "0")
    assert monitor.notify([alert]) is False and calls == []
    monkeypatch.delenv("FASTVALUE_NOTIFY")
    assert monitor.notify([alert]) is True and calls[0][0][0] == "osascript"


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_api_monitor_status(client, monkeypatch):
    monkeypatch.setattr(monitor, "schedule_status", lambda: {"installed": False})
    data = client.get("/api/monitor").get_json()
    assert data["checked_at"] is None and data["alerts"] == []
    assert data["install_command"] == "python -m core.monitor --install"


def _rated_review(*rows):
    return {"positions": [{"ticker": t, "action": a, "decision": d, "owner_score": s, "rule": rule, "class": cls, "rated": cls == "stock",
                           "reasons": [f"{t} reason"], "fix": f"{t} fix"} for t, a, d, s, rule, cls in rows], "errors": []}


def test_snapshot_records_rule_and_rules_version():
    snap = monitor.snapshot(_rated_review(("AAA", "HOLD", "HOLD", 5.0, "hold_band", "stock"),
                                          ("FND", "HOLD", "NOT_RATED", 0.0, "core_fund", "core")), None, "2026-10-05T22:45:00")
    assert snap["rules"] == "D-35"
    assert snap["holdings"]["FND"] == {"action": "HOLD", "decision": "NOT_RATED", "owner_score": 0.0, "reason": "FND reason",
                                       "rule": "core_fund", "class": "core", "rated": False, "fix": "FND fix"}
    # Old snapshots (no rules, no fix) still compare: the keys are read with .get.
    old = {"checked_at": "2026-10-04", "holdings": {"AAA": {"action": "HOLD", "decision": "HOLD", "owner_score": 5.0, "reason": ""}}, "watchlist": {}}
    assert monitor.compare(old, monitor.snapshot(_review(("AAA", "HOLD", "HOLD", 5.0)), None, "2026-10-05")) == []


def test_the_alert_uses_the_rows_fix_and_class_sentence():
    before = monitor.snapshot(_rated_review(("AAA", "HOLD", "HOLD", 5.0, "hold_band", "stock"), ("FND", "HOLD", "NOT_RATED", 0.0, "core_fund", "core")),
                              None, "2026-10-05T22:45:00")
    after = monitor.snapshot(_rated_review(("AAA", "EXIT", "SELL", -18.0, "owner_sell", "stock"), ("FND", "TRIM", "NOT_RATED", 0.0, "above_cap", "core")),
                             None, "2026-10-06T22:45:00")
    alerts = {a["ticker"]: a for a in monitor.compare(before, after)}
    assert alerts["AAA"]["fix"] == "AAA fix" and "-18" in alerts["AAA"]["text"]
    assert alerts["FND"]["text"] == "FND reason"  # not rated: the class sentence, not an owner score


def test_rules_change_folds_non_exit_changes_into_one_note():
    before = monitor.snapshot(_review(("AAA", "REVIEW", "NOT_RATED", 0.0), ("BBB", "REVIEW", "NOT_RATED", 0.0), ("CCC", "HOLD", "HOLD", 1.0),
                                      ("DDD", "HOLD", "HOLD", 2.0)), None, "2026-10-05T22:45:00")
    before.pop("rules")  # a state written before the rules were recorded
    after = monitor.snapshot(_review(("AAA", "HOLD", "NOT_RATED", 0.0), ("BBB", "HOLD", "NOT_RATED", 0.0), ("CCC", "EXIT", "SELL", -20.0),
                                     ("DDD", "HOLD", "HOLD", 2.0)), None, "2026-10-06T22:45:00")
    alerts = monitor.compare(before, after)
    assert [a["title"] for a in alerts] == ["CCC: HOLD -> EXIT", "The review rules changed"]  # a move into EXIT is real, it still alerts
    note = alerts[1]
    assert note["kind"] == "note" and "2 holdings changed action without a new rating: AAA, BBB" in note["text"]
    assert note["text"].startswith("The review rules changed:") and "decision log" not in note["text"] and note["link"].endswith("#portfolio=review") and note["fix"].startswith("Nothing to do")
    # The same rules on both sides: every change alerts one by one again.
    same = monitor.compare(monitor.snapshot(_review(("AAA", "REVIEW", "NOT_RATED", 0.0)), None, "2026-10-05"),
                           monitor.snapshot(_review(("AAA", "HOLD", "NOT_RATED", 0.0)), None, "2026-10-06"))
    assert [a["title"] for a in same] == ["AAA: REVIEW -> HOLD"]


def test_add_to_hold_does_not_alert():
    before = monitor.snapshot(_review(("AAA", "ADD", "BUY", 30.0), ("BBB", "HOLD", "BUY", 30.0)), None, "2026-10-05")
    after = monitor.snapshot(_review(("AAA", "HOLD", "BUY", 30.0), ("BBB", "ADD", "BUY", 30.0)), None, "2026-10-06")
    assert [a["title"] for a in monitor.compare(before, after)] == ["BBB: HOLD -> ADD"]


def test_a_reason_that_already_states_the_rating_is_not_repeated():
    def snap(action, decision, score, at):
        return {"checked_at": at, "rules": "D-35", "watchlist": {}, "holdings": {"AAA": {
            "action": action, "decision": decision, "owner_score": score, "rated": True, "rule": "x", "class": "stock", "fix": "Sell it.",
            "reason": "Rated SELL (-15) on a 12-month view, so the exit rule says sell." if action == "EXIT" else "AAA reason"}}}
    alert = monitor.compare(snap("HOLD", "HOLD", 0.0, "2026-10-05"), snap("EXIT", "SELL", -15.0, "2026-10-06"))[0]
    assert alert["text"] == "Rated SELL (-15) on a 12-month view, so the exit rule says sell."
