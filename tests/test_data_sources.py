"""The footer's "Your data and costs" list, the delete-everything route and the daily check routes (synthetic data only)."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest

import config.settings as settings
from config.data_sources import DATA_SOURCES, HOST_RE, VERSION, hosts

ROOT = Path(__file__).resolve().parents[1]


def test_every_host_the_fetchers_call_is_listed():
    """Copy and code cannot drift: a new https host in data/ or core/ must be named in the panel's registry."""
    found: dict[str, str] = {}
    for folder in ("data", "core"):
        for path in (ROOT / folder).rglob("*.py"):
            for host in HOST_RE.findall(path.read_text()):
                found.setdefault(host.rstrip("."), str(path.relative_to(ROOT)))
    listed = hosts()
    missing = {h: where for h, where in found.items() if not any(h == s or h.endswith("." + s) for s in listed)}
    assert not missing, f"Add these to config/data_sources.py: {missing}"


def test_every_source_says_what_it_sends_and_why():
    assert len({s["host"] for s in DATA_SOURCES}) == len(DATA_SOURCES)
    for s in DATA_SOURCES:
        assert s["name"] and s["sends"] and s["why"] and isinstance(s["optional"], bool)
        assert re.fullmatch(r"[a-z0-9.-]+", s["host"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION)


def test_the_meta_payload_carries_the_registry_and_the_version():
    import web.app as web_app

    data = web_app.app.test_client().get("/api/agents").get_json()
    assert data["version"] == VERSION and {s["host"] for s in data["data_sources"]} == hosts()


def test_delete_all_data_clears_the_personal_files_and_keeps_market_data(tmp_path, monkeypatch):
    import web.app as web_app
    from core import monitor, style_report, track_record
    from data import portfolio as pf

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(settings, "OUTPUT_DIR", out)
    removed = []
    monkeypatch.setattr(monitor, "uninstall", lambda: removed.append(1) or "Removed.")
    monkeypatch.setattr(monitor, "schedule_status", lambda: {"installed": True, "time": "22:30"})

    pf.save_portfolio({"cash": 5.0, "currency": "EUR", "positions": [{"ticker": "AAPL", "shares": 2, "avg_cost": 100}]})
    store = settings.STORE_DIR
    for name in ("watchlist.json", "agent_weights.json", "local_settings.json", "history.db", "rules_lab_history.json",
                 "monitor_state.json", "alerts.jsonl", "monitor.log", "last_review.json", "broker_ledger.json", "instruments.json"):
        (store / name).write_text("x")
    (store / "iv_history").mkdir()
    (store / "iv_history" / "AAPL.json").write_text("{}")
    for path in (style_report.share_report_path(), track_record.report_path(), track_record.result_path()):
        path.write_text("x")

    response = web_app.app.test_client().delete("/api/data")
    assert response.status_code == 204 and removed == [1]
    assert pf.load_portfolio()["positions"] == []
    assert sorted(p.name for p in store.iterdir()) == ["iv_history"]
    assert (store / "iv_history" / "AAPL.json").exists()
    assert list(out.iterdir()) == []


def test_delete_all_data_refuses_other_web_pages():
    import web.app as web_app

    refused = web_app.app.test_client().delete("/api/data", headers={"Origin": "https://example.org"})
    assert refused.status_code == 403


def test_the_daily_check_can_be_scheduled_stopped_and_run_from_the_page(monkeypatch):
    import web.app as web_app
    from core import monitor

    calls = []
    monkeypatch.setattr(monitor, "install", lambda at=None: calls.append("install") or "Scheduled")
    monkeypatch.setattr(monitor, "uninstall", lambda: calls.append("uninstall") or "Removed")
    monkeypatch.setattr(monitor, "run_check", lambda *a, **k: calls.append("run") or {"alerts": []})
    monkeypatch.setattr(monitor, "schedule_status", lambda: {"installed": "install" in calls and "uninstall" not in calls, "time": "22:30"})
    monkeypatch.setattr("sys.platform", "darwin")
    client = web_app.app.test_client()

    assert client.post("/api/monitor/install").get_json()["schedule"] == {"installed": True, "time": "22:30"}
    assert client.post("/api/monitor/uninstall").get_json()["schedule"]["installed"] is False
    job = client.post("/api/monitor/run").get_json()["job"]
    import time
    for _ in range(50):
        state = client.get(f"/api/lab/jobs/{job}").get_json()
        if state["status"] != "running":
            break
        time.sleep(0.05)
    assert state["status"] == "done" and "run" in calls and "schedule" in state["result"]


def test_scheduling_is_skipped_where_there_is_no_scheduler(monkeypatch):
    import web.app as web_app
    from core import monitor

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(monitor, "install", lambda at=None: pytest.fail("must not install"))
    status = web_app.app.test_client().post("/api/monitor/install").get_json()
    assert status["schedule"]["unsupported"] is True


def test_the_share_page_is_a_download_on_request():
    import web.app as web_app
    from core import style_report

    style_report.share_report_path().parent.mkdir(parents=True, exist_ok=True)
    style_report.share_report_path().write_text("<html><body>page</body></html>")
    client = web_app.app.test_client()
    try:
        plain = client.get("/report/my-style")
        assert plain.status_code == 200 and "attachment" not in plain.headers.get("Content-Disposition", "")
        saved = client.get("/report/my-style?download=1")
        assert 'attachment; filename=my-investing-style.html' in saved.headers["Content-Disposition"]
    finally:
        style_report.share_report_path().unlink(missing_ok=True)


def test_static_files_are_compressed_and_cached_by_version():
    import web.app as web_app

    client = web_app.app.test_client()
    page = client.get("/").get_data(as_text=True)
    asset = re.search(r'src="(/static/app\.js\?v=[0-9a-f]{10})"', page)
    assert asset and re.search(r'href="/static/style\.css\?v=[0-9a-f]{10}"', page)
    plain = client.get(asset.group(1))
    packed = client.get(asset.group(1), headers={"Accept-Encoding": "gzip"})
    assert packed.headers["Content-Encoding"] == "gzip" and packed.headers["Vary"] == "Accept-Encoding"
    assert gzip.decompress(packed.get_data()) == plain.get_data()
    assert len(packed.get_data()) < len(plain.get_data()) / 3
    assert "immutable" in packed.headers["Cache-Control"] and "Content-Encoding" not in plain.headers
    assert "immutable" not in client.get("/static/theme.js").headers.get("Cache-Control", "")


def test_the_content_security_policy_allows_the_share_pages_embedded_font():
    import web.app as web_app

    assert "font-src 'self' data:" in web_app.CONTENT_SECURITY_POLICY
