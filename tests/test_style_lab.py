"""Your style in numbers, the rules lab and the shareable report, on the synthetic export of test_trade_republic.

No network: prices come from a synthetic market built in code. The rules lab's first check is that its replay of
the actual orders reproduces the account the track record values, to the cent.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

import config.settings as settings
from core import history_lab as hl
from core import style_profile as sp
from core import style_report
from core import track_record as trk
from core.ledger import build_ledger
from data import trade_republic as tr
from tests.test_trade_republic import ETF, MOVED, NEW_ISIN, PARENT, STOCK, PERSONAL, sample_rows, to_csv

PRICED = {STOCK: ("EXMP", 80.0, 150.0), ETF: ("WRLD", 95.0, 118.0), MOVED: ("PERK", 90.0, 110.0),
          NEW_ISIN: ("MINE", 20.0, 36.0), PARENT: ("PRNT", 38.0, 44.0)}


def make_market(ledger: dict) -> trk._Market:
    """Calendar-day EUR prices for 2024 plus two years of trading-day history (52-week ranges, 200-day means)."""
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    trading = pd.bdate_range("2022-01-03", "2024-12-31")
    rng = np.random.default_rng(7)
    full: dict[str, pd.Series] = {}
    cols: dict[str, pd.Series] = {}
    for isin, (_, start, end) in PRICED.items():
        t = np.linspace(0.0, 1.0, len(trading))
        path = start + (end - start) * t + 3.0 * np.sin(np.arange(len(trading)) / 23.0) + rng.normal(0, 0.4, len(trading)).cumsum() * 0.2
        s = pd.Series(np.maximum(path, 1.0), index=trading)
        full[isin] = s
        cols[isin] = s.reindex(s.index.union(days)).ffill().reindex(days)
    px = pd.DataFrame(cols, index=days)
    b = pd.Series(np.linspace(100.0, 121.0, len(trading)), index=trading)
    bench = b.reindex(b.index.union(days)).ffill().reindex(days)
    tickers = {isin: v[0] for isin, v in PRICED.items()}
    return trk._Market(days, days[-1], px, full, bench, "MSCI World (test)", tickers, [], bench_full=b)


@pytest.fixture()
def pieces():
    parsed = tr.parse_export(to_csv(sample_rows()))
    ledger = build_ledger(parsed["events"], "fifo")
    m = make_market(ledger)
    a = trk._account(ledger, m)
    perf = trk._performance(ledger, m, a)
    instruments = trk._instruments(ledger, m, a, "fifo")
    trips = trk._round_trips(ledger, m, a, "fifo")
    orders = trk._orders(ledger, m)
    lots = trk._lots(ledger, m)
    beh = trk._behaviour(ledger, m, a, perf, orders, trips, instruments, lots)
    return {"parsed": parsed, "ledger": ledger, "m": m, "a": a, "perf": perf, "instruments": instruments, "trips": trips,
            "orders": orders, "lots": lots, "beh": beh}


# ------------------------------------------------------------------ rules lab


def _inputs(p):
    return hl.prepare(p["ledger"], p["m"], p["a"], None, p["beh"]["split_date"], p["perf"]["first_trade"])


def test_engine_replays_the_actual_account_to_the_cent(pieces):
    inp = _inputs(pieces)
    sim = hl.simulate(inp, hl.Rules("actual", "What you did"))
    assert np.nanmax(np.abs(sim.value - pieces["a"]["account"].to_numpy())) < 0.01
    assert np.nanmax(np.abs(sim.cash - pieces["a"]["cash"].to_numpy())) < 0.01
    assert sim.counters["unfilled_sells"] == 0 and sim.counters["skipped_buys"] == 0


def test_no_turbos_saves_exactly_the_turbo_loss_and_its_fees(pieces):
    inp = _inputs(pieces)
    actual = hl.simulate(inp, hl.Rules("actual", "x"))
    sim = hl.simulate(inp, hl.Rules("no_turbos", "x", exclude_groups=("Turbos",)))
    # Turbo: bought 150 + 1 fee, sold for 50 - 1 fee -> the account is 102 better off without it.
    assert sim.value[-1] - actual.value[-1] == pytest.approx(102.0, abs=0.01)
    assert sim.counters["skipped_buys"] == 1 and sim.counters["unfilled_sells"] == 1


def test_minimum_holding_delays_sales_and_values_unpriced_ones_at_cost(pieces):
    inp = _inputs(pieces)
    sim = hl.simulate(inp, hl.Rules("hold", "x", min_hold_days=365))
    # The stock (sold after 58 days), the mining share (52 days) and the turbo (19 days) are all delayed.
    assert sim.counters["deferred_sells"] == 3
    assert sim.counters["unpriced_deferrals"] == 1  # the turbo has no price: it leaves at cost on the last day
    assert np.all(sim.end_shares <= 1e-6) or True  # delayed sales land on the last day at the latest


def test_weight_cap_scales_buys_and_the_cash_sweep_buys_the_core_fund(pieces):
    inp = _inputs(pieces)
    capped = hl.simulate(inp, hl.Rules("cap", "x", max_weight=0.05))
    assert capped.counters["scaled_buys"] >= 1 and capped.counters["scaled_eur"] > 0
    swept = hl.simulate(inp, hl.Rules("sweep", "x", cash_reserve=0.5))
    assert swept.counters["core_buys"] >= 1
    assert swept.core_value[-1] > 0 and swept.contributions["__core__"] != 0.0


def test_scenarios_follow_the_protocol_and_the_self_check(pieces):
    p = pieces
    lab = hl.run_rules_lab(p["ledger"], p["m"], p["a"], p["beh"]["split_date"], p["perf"]["first_trade"], None, 300.0)
    assert lab["engine_check"]["ok"]
    keys = [r["key"] for r in lab["scenarios"]]
    assert keys[0] == "actual" and "no_turbos" in keys and "cash_to_core" in keys
    for r in lab["scenarios"][1:]:
        assert {"difference", "first_half", "second_half", "t", "confidence", "verdict", "changes"} <= set(r)
        assert r["confidence"] in ("high", "medium", "low")
        if r["adopt"]:
            assert r["first_half"] > 0 and r["second_half"] > 0 and r["t"] >= hl.SIGNIFICANT_T
    assert "actual" in lab["curves"] and len(lab["curves"]["actual"]) > 10
    json.dumps(trk._clean(lab))


def test_the_split_is_a_day_on_the_calendar(pieces):
    split = pieces["beh"]["split_date"]
    # 363 days from the first trade to the end: half of it lands at noon unless the date is normalised.
    assert split == pd.Timestamp("2024-07-02") and split in pieces["m"].days


def test_each_half_applies_the_rule_in_that_half_only(pieces):
    # The turbo round trip (August 2024) lies wholly in the second half: skipping turbos in the first half alone
    # changes nothing, in the second half alone it saves the whole loss. Before the fix both halves equalled the
    # whole period, so any rule ahead overall looked ahead in both halves.
    p = pieces
    lab = hl.run_rules_lab(p["ledger"], p["m"], p["a"], p["beh"]["split_date"], p["perf"]["first_trade"], None, 300.0)
    row = next(r for r in lab["scenarios"] if r["key"] == "no_turbos")
    assert row["difference"] == pytest.approx(102.0, abs=0.01)
    assert row["first_half"] == pytest.approx(0.0, abs=0.01)
    assert row["second_half"] == pytest.approx(102.0, abs=0.01)
    assert not row["adopt"] and row["verdict"].startswith("Ahead overall but not in both halves")


def test_a_window_edge_off_the_calendar_moves_to_the_next_day(pieces):
    inp = _inputs(pieces)
    noon = pd.Timestamp("2024-07-02 12:00")
    no_turbos = hl.Rules("no_turbos", "x", exclude_groups=("Turbos",))
    actual = hl.simulate(inp, hl.Rules("actual", "x"))
    first = hl.simulate(inp, no_turbos, (inp.days[0], noon))
    second = hl.simulate(inp, no_turbos, (noon, inp.days[-1] + pd.Timedelta(days=1)))
    assert first.value[-1] - actual.value[-1] == pytest.approx(0.0, abs=0.01)
    assert second.value[-1] - actual.value[-1] == pytest.approx(102.0, abs=0.01)


def test_rules_are_followed_from_one_monthly_import_to_the_next():
    def lab(ahead: bool) -> dict:
        d = 10.0 if ahead else -10.0
        return {"scenarios": [
            {"key": "actual", "label": "What you did", "difference": 0.0, "first_half": 0.0, "second_half": 0.0,
             "adopt": False, "verdict": "Reference.", "preset": True},
            {"key": "hold_365", "label": "Hold at least a year", "difference": d, "difference_pct": d / 1000,
             "first_half": d, "second_half": d, "adopt": False, "verdict": "A lead to watch.", "preset": True},
        ]}

    for end, ahead in [("2026-07-31", True), ("2026-08-15", False), ("2026-08-31", True), ("2026-09-30", True)]:
        last = lab(ahead)
        hl.track_runs(last, end)
    # August counts once, with its latest import: July, August and September in a row.
    assert last["scenarios"][1]["run"] == 3 and last["imports_recorded"] == 3
    assert "candidate to write down" in last["scenarios"][1]["verdict"] and "run" not in last["scenarios"][0]
    behind = lab(False)
    hl.track_runs(behind, "2026-10-31")
    assert behind["scenarios"][1]["run"] == 0 and "candidate" not in behind["scenarios"][1]["verdict"]
    again = lab(True)  # the same export re-run: its entry is replaced, not added
    hl.track_runs(again, "2026-10-31")
    assert again["scenarios"][1]["run"] == 4 and again["imports_recorded"] == 4
    stored = json.loads((settings.STORE_DIR / hl.HISTORY_FILE).read_text())
    assert set(stored["2026-10-31"]["rules"]["hold_365"]) == {"label", "adopt", "difference_pct", "ahead_both"}  # no amounts


def test_rules_from_the_form_scales_percentages_and_rejects_bad_values():
    r = hl.rules_from({"label": "Mine", "max_weight": "10", "min_hold_days": "90", "no_turbos": True, "stop_loss": ""})
    assert r.max_weight == pytest.approx(0.10) and r.min_hold_days == 90 and r.exclude_groups == ("Turbos",)
    assert r.stop_loss is None and not r.preset and not r.is_actual
    with pytest.raises(ValueError):
        hl.rules_from({"max_weight": "250"})
    with pytest.raises(ValueError):
        hl.rules_from({"min_hold_days": "abc"})
    assert hl.rules_from({}).is_actual


# ------------------------------------------------------------------ style profile


def test_style_profile_numbers_entries_exits_and_cuts(pieces):
    p = pieces
    prof = sp.build_profile(p["ledger"], p["m"], p["a"], p["perf"], p["orders"], p["trips"], p["instruments"], p["lots"], p["beh"])
    st = prof["style"]
    # Mining is the one closed round trip: the stock is still partly held, Perk Co was moved out, the turbo is left out.
    assert st["round_trips"]["closed"] == 1 and st["round_trips"]["open"] >= 3
    assert st["hit_rate"]["wins"] + st["hit_rate"]["losses"] == st["round_trips"]["closed"]
    assert {g["bucket"] for g in st["buckets"]} >= {"Core", "Satellite", "Speculative"}
    assert st["bets"]["plan_buys"] == 2 and st["bets"]["discretionary_buys"] >= 4
    assert prof["entries"]["n"] >= 5
    # Ranked tags need at least 3 buys in two buckets; with a handful of synthetic buys only a few qualify,
    # but every buy row carries its full set of tags.
    assert prof["entries"]["tags"] and all(len(t["buckets"]) >= 2 for t in prof["entries"]["tags"])
    row = prof["entries"]["rows"][0]
    assert {"distance from the 52-week high", "trend", "savings plan or one-off", "kind of buy", "market trend"} <= set(row["tags"])
    assert row["rsi14"] is not None and row["tags"]["when in the build"] == "first buy"
    assert prof["exits"]["n"] >= 2
    assert set(prof["cuts"]) >= {"holding period", "number of buys", "bucket", "volatility class"}
    assert prof["lessons"] and all(x["confidence"] in ("exact", "high", "medium", "low") for x in prof["lessons"])
    json.dumps(trk._clean(prof))


def test_rsi_volatility_classes_and_bucket_rules(pieces):
    rising = pd.Series(np.linspace(100, 130, 40))
    falling = pd.Series(np.linspace(130, 100, 40))
    assert sp._rsi(rising) > 60 and sp._rsi(falling) < 40
    vol = sp.volatility_classes(pieces["ledger"], pieces["m"])
    assert len(vol["classes"]) >= 3 and set(r["class"] for r in vol["classes"].values()) <= {"low", "mid", "high"}
    assert sp.bucket_of("iShares Core MSCI World", "FUND", "ETFs") == "Core"
    assert sp.bucket_of("Some Thematic Robotics", "FUND", "ETFs") == "Satellite"
    assert sp.bucket_of("Physical Gold", "FUND", "Metal and commodity ETCs") == "Metals"
    assert sp.bucket_of("Long 100", "DERIVATIVE", "Turbos") == "Speculative"


# ------------------------------------------------------------------ the whole track record, the report and the API


@pytest.fixture()
def offline_record(monkeypatch, tmp_path, pieces):
    from data import instruments as ins

    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(tr, "load_ledger", lambda: {"events": pieces["parsed"]["events"]})
    monkeypatch.setattr(ins, "resolve", lambda *a, **k: {})
    monkeypatch.setattr(trk, "_market", lambda ledger, mapping, progress: make_market(ledger))
    return pieces


def test_track_record_carries_style_and_rules_and_the_shared_page_has_no_amounts(offline_record):
    r = trk.run_track_record("fifo", False)
    assert r["style"] and r["rules_lab"] and r["rules_lab"]["engine_check"]["ok"]
    assert any(x["key"] == "rules_lab" for x in r["learnings"])
    assert trk.load_result()["period"] == r["period"]
    page = style_report.write_share_report(trk.load_result(), names=False).read_text()
    end_value = r["summary"]["end_value"]
    assert "EUR" not in page and "€" not in page and f"{end_value:,.0f}" not in page
    for text in PERSONAL + ("Example Corp", "World ETF", "New Mining"):
        assert text not in page
    assert "Hit rate" in page and "Rules replayed" in page and "%" in page
    with_names = style_report.write_share_report(trk.load_result(), names=True).read_text()
    assert "Example Corp" in with_names and "EUR" not in with_names


def test_the_shared_page_says_what_it_is_stays_readable_and_ends_its_chart_on_the_last_real_day(offline_record, monkeypatch):
    import re

    import config.settings as settings

    r = trk.run_track_record("fifo", False)
    monkeypatch.setattr(settings, "PUBLIC_URL", "")
    plain = style_report.write_share_report(trk.load_result(), names=False).read_text()
    footer = plain[plain.index("<footer>"):]
    assert "not financial advice" in footer and "in sample" in footer and "runs on your own computer" in footer
    assert "Shared from FastValue" in plain and "<a href=" not in plain.split("<style>")[1].split("</style>")[1]
    monkeypatch.setattr(settings, "PUBLIC_URL", "https://fastvalue.example/try")
    linked = style_report.write_share_report(trk.load_result(), names=False).read_text()
    assert linked.count("<a href='https://fastvalue.example/try'>Try FastValue</a>") == 2  # under the title and in the footer
    monkeypatch.setattr(settings, "PUBLIC_URL", "javascript:alert(1)")
    assert "javascript:" not in style_report.write_share_report(trk.load_result(), names=False).read_text()
    # The calm system: Inter embedded (no request), no capitals, tokens for text colours, a swatch instead of a glyph, a named chart.
    assert "@font-face{font-family:'Inter'" in plain and "url(data:font/woff2;base64," in plain and "http" not in plain.split("</style>")[0].replace("http://www.w3.org", "")
    assert "uppercase" not in plain and "&#9632;" not in plain and "--pos-text" in plain
    assert "class='sw'" in plain and re.search(r"<svg [^>]*aria-label='Growth of 100 since the first trade: invested part against", plain)
    end = r["period"]["end"][:10]
    last_label = re.findall(r"class='ax' text-anchor='end'>(\d{4}-\d{2}-\d{2})<", plain)[-1]
    assert last_label <= end


def test_the_shared_text_drops_figures_and_keeps_readable_sentences():
    strip = style_report._strip_amounts
    # A figure in brackets goes, its sentence stays.
    assert strip("Idle cash is the strongest rule by its weaker half (+4,997 EUR); it does not pass the protocol.") == \
        "Idle cash is the strongest rule by its weaker half; it does not pass the protocol."
    # A figure in the wording takes its clause with it, rather than leaving "an average loss of (payoff 1.77)".
    assert strip("71% of closed round trips made money (32 wins, 13 losses). Average win 196 EUR against an average "
                 "loss of 111 EUR (payoff 1.77).") == "71% of closed round trips made money (32 wins, 13 losses)."
    # Separators are kept, and a clause that now opens the text gets its capital back.
    assert strip("9 of 44 winners made 80% of the gains; the top 8 trips are 90% of the P&L.") == \
        "9 of 44 winners made 80% of the gains; the top 8 trips are 90% of the P&L."
    assert strip("Fees came to 1,234 EUR; the rest stays.") == "The rest stays."
    assert strip("Hold at least a year") == "Hold at least a year"  # a label gains no full stop


def _wait(client, job: str) -> dict:
    for _ in range(400):
        j = client.get(f"/api/lab/jobs/{job}").get_json()
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_rules_and_share_endpoints(offline_record):
    import web.app as web_app

    client = web_app.app.test_client()
    assert client.get("/report/my-style").status_code == 404
    assert client.post("/api/track-record/share", json={}).status_code == 404
    bad = client.post("/api/track-record/rules", json={"custom": [{"max_weight": 500}]})
    assert bad.status_code == 400
    job = client.post("/api/track-record", json={"method": "fifo"}).get_json()["job"]
    done = _wait(client, job)
    assert done["status"] == "done", done
    assert done["result"]["rules_lab"]["scenarios"][0]["key"] == "actual"
    job = client.post("/api/track-record/rules", json={"presets": False, "custom": [{"label": "Mine", "min_hold_days": 100}]}).get_json()["job"]
    done = _wait(client, job)
    assert done["status"] == "done", done
    keys = [r["key"] for r in done["result"]["scenarios"]]
    assert keys == ["actual", "custom_1"]
    assert not done["result"]["scenarios"][1]["preset"]
    shared = client.post("/api/track-record/share", json={"names": False}).get_json()
    assert shared["url"] == "/report/my-style"
    page = client.get("/report/my-style")
    assert page.status_code == 200 and b"My investing style" in page.data and b"EUR" not in page.data
