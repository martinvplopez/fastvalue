"""Benchmarks without cash: the invested part against three indices, the Cash section and old stored results.

Synthetic data only, no network. The owner's figures never appear here: made-up ISINs, made-up prices.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

import config.settings as settings
from core import portfolio_analytics as pa
from core import style_report
from core import track_record as trk
from core.ledger import build_ledger
from data import generic_ledger as gl
from data import instruments as ins
from data import trade_republic as tr
from data.trade_republic import ISIN_RE
from tests.test_portfolio_analytics import _row, _unit_fx, _walk
from tests.test_style_lab import make_market, pieces  # noqa: F401 - the fixture

FUND = "XS0000000001"
HEADER = "date,type,isin,name,shares,price,amount,fee,tax,currency\n"
IDS = [b["id"] for b in settings.PORTFOLIO_BENCHMARKS]


def _ledger(csv_text: str) -> dict:
    parsed = gl.parse_export((HEADER + csv_text).encode())
    return build_ledger(parsed["events"], "fifo")


def _market(ledger: dict, prices: dict[str, pd.Series], benches: dict[str, pd.Series]) -> trk._Market:
    days = next(iter(benches.values())).index
    px = pd.DataFrame(prices, index=days)
    meta = [{**b, "primary": i == 0} for i, b in enumerate(settings.PORTFOLIO_BENCHMARKS) if b["id"] in benches]
    return trk._Market(days, days[-1], px, {k: v for k, v in prices.items()}, benches["msci_world"], "MSCI World (test)",
                       {isin: "TST" for isin in prices}, [], bench_full=benches["msci_world"], benches=benches, bench_meta=meta)


def _paths() -> dict[str, pd.Series]:
    """Three index paths on 2024's calendar days. All are flat to the 3rd of January, the day of the first buy."""
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rise = np.clip(np.arange(len(days)) - 2, 0, None) / (len(days) - 3)
    return {"msci_world": pd.Series(100.0 + 30.0 * rise, index=days),
            "sp500": pd.Series(100.0 + 20.0 * rise, index=days),
            "nasdaq100": pd.Series(100.0 + 50.0 * rise ** 2, index=days)}


def _all(ledger: dict, m: trk._Market) -> dict:
    a = trk._account(ledger, m)
    perf = trk._performance(ledger, m, a)
    instruments = trk._instruments(ledger, m, a, "fifo")
    trips = trk._round_trips(ledger, m, a, "fifo")
    orders = trk._orders(ledger, m)
    lots = trk._lots(ledger, m)
    beh = trk._behaviour(ledger, m, a, perf, orders, trips, instruments, lots)
    split = beh["split_date"]
    learnings = trk._learnings(perf, beh, instruments, {"sold_kept": trk._sold_kept(orders, m, split)},
                               trk._overlays(lots, m, split), None)
    return {"a": a, "perf": perf, "beh": beh, "learnings": {x["key"]: x for x in learnings}}


SAME_AS_INDEX = ("2024-01-02,deposit,,,,,10000,,,EUR\n"
                 "2024-01-03,buy," + FUND + ",Index Tracker Fund,20,100,-2000,,,EUR\n")


def test_idle_cash_no_longer_lowers_the_headline():
    paths = _paths()
    ledger = _ledger(SAME_AS_INDEX)
    m = _market(ledger, {FUND: paths["msci_world"]}, paths)
    out = _all(ledger, m)
    perf, b = out["perf"], out["perf"]["benchmarks"][0]
    assert perf["twr_invested_total"] == pytest.approx(b["total"], abs=1e-9)
    assert b["excess_annual"] == pytest.approx(0.0, abs=1e-9)
    assert b["buys_difference"] == pytest.approx(0.0, abs=0.01)
    market = out["learnings"]["market"]
    assert market["eur"] == pytest.approx(0.0, abs=0.01) and market["section"] == "overview"
    # The old view blamed the picks for the idle cash.
    assert perf["twr_total"] < perf["benchmark_total"] - 0.2
    assert b["deposits_difference"] == pytest.approx(-2400.0, abs=0.01)
    assert b["cash_part"] == pytest.approx(-2400.0, abs=0.01)
    # Interest is the cash's own, not the picks'.
    with_interest = _ledger(SAME_AS_INDEX + "2024-06-01,interest,,,,,50,,,EUR\n")
    out = _all(with_interest, _market(with_interest, {FUND: paths["msci_world"]}, paths))
    b = out["perf"]["benchmarks"][0]
    assert b["cash_part"] == pytest.approx(-2400.0 + 50.0, abs=0.01) and b["buys_difference"] == pytest.approx(0.0, abs=0.01)
    cash = out["learnings"]["cash"]
    assert cash["section"] == "cash" and cash["confidence"] == "exact" and cash["eur"] == pytest.approx(b["cash_part"])


@pytest.fixture()
def offline_three(monkeypatch, tmp_path, pieces):  # noqa: F811
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(tr, "load_ledger", lambda: {"events": pieces["parsed"]["events"]})
    monkeypatch.setattr(ins, "resolve", lambda *a, **k: {})
    monkeypatch.setattr(trk, "_market", lambda ledger, mapping, progress: _with_three(make_market(ledger)))
    return pieces


def _with_three(m: trk._Market) -> trk._Market:
    n = np.arange(len(m.days))
    base = m.bench
    m.benches = {"msci_world": base, "sp500": base * (1.0 + 0.0004 * n), "nasdaq100": base * (1.0 + 0.0009 * n)}
    m.bench_meta = [{**b, "primary": i == 0} for i, b in enumerate(settings.PORTFOLIO_BENCHMARKS)]
    return m


def test_cash_section_reproduces_the_old_account_vs_deposits_figure(pieces, offline_three):  # noqa: F811
    p = pieces
    s = p["perf"]
    first = s["benchmarks"][0]
    assert first["deposits_end_value"] == s["benchmark_end_value"] and first["deposits_xirr"] == s["benchmark_xirr"]
    r = trk.run_track_record("fifo", False)
    b = r["summary"]["benchmarks"][0]
    assert r["schema"] == trk.RESULT_SCHEMA == 2
    assert b["deposits_end_value"] == pytest.approx(r["summary"]["benchmark_end_value"], abs=1e-4)
    assert b["deposits_xirr"] == r["summary"]["benchmark_xirr"]
    alt = {x["key"]: x for x in r["alternatives"]}
    assert alt["deposits_msci_world"]["end_value"] == pytest.approx(r["summary"]["benchmark_end_value"], abs=1e-4)
    assert alt["deposits_msci_world"]["section"] == "cash" and alt["actual"]["section"] == "overview"
    assert "msci" not in alt and "buys_sp500" in alt and "deposits_nasdaq100" in alt
    assert b["buys_difference"] + b["cash_part"] == pytest.approx(b["deposits_difference"], abs=1e-6)
    # The cash's own result, worked out apart: cash today minus the cash movements put into the index on their days.
    m, a = _with_three(make_market(p["ledger"])), p["a"]
    cash_moves = a["flows"] - (a["buys"] - a["sells"] - a["divs"])
    idx = m.bench
    in_index = float((cash_moves / idx).sum() * idx.iloc[-1])
    assert b["cash_part"] == pytest.approx(p["perf"]["cash"] - in_index, abs=1e-4)
    learning = next(x for x in r["learnings"] if x["key"] == "cash")
    assert learning["section"] == "cash" and learning["confidence"] == "exact"
    assert learning["eur"] == pytest.approx(b["cash_part"], abs=1e-4)
    assert r["cash"]["share"] and 0.0 <= r["cash"]["share"][0]["value"] <= 1.0


def test_index_mirror_moves_money_in_and_out_and_may_go_below_zero():
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    value, dipped, notes = trk._index_mirror(pd.Series([100.0, 0.0, -100.0], index=days), pd.Series([10.0, 20.0, 20.0], index=days))
    assert value.tolist() == pytest.approx([100.0, 200.0, 100.0]) and not dipped and not notes
    value, dipped, _ = trk._index_mirror(pd.Series([100.0, -300.0, 0.0], index=days), pd.Series([10.0, 10.0, 10.0], index=days))
    assert value.iloc[-1] == pytest.approx(-200.0) and dipped
    # A flow on a day without an index price is left out and counted.
    _, _, notes = trk._index_mirror(pd.Series([100.0, 50.0, 0.0], index=days), pd.Series([10.0, np.nan, 10.0], index=days))
    assert notes and "1 flows" in notes[0]

    # Through _performance: the whole difference from the holdings stays exact while the mirror itself is not a holding.
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    flat = pd.Series(10.0, index=days)
    fund = pd.Series(np.where(days < pd.Timestamp("2024-03-01"), 10.0, 40.0), index=days)
    ledger = _ledger("2024-01-02,deposit,,,,,1000,,,EUR\n"
                     "2024-01-03,buy," + FUND + ",Index Tracker Fund,10,10,-100,,,EUR\n"
                     "2024-03-05,sell," + FUND + ",Index Tracker Fund,7.5,40,300,,,EUR\n")
    m = _market(ledger, {FUND: fund}, {"msci_world": flat})
    perf = trk._performance(ledger, m, trk._account(ledger, m))
    b = perf["benchmarks"][0]
    assert b["buys_dipped"] and b["buys_end_value"] is None and b["buys_xirr"] is None
    assert b["buys_difference"] == pytest.approx(perf["invested_value"] - (-200.0), abs=1e-6) == pytest.approx(300.0, abs=1e-6)
    # The every-deposit figure keeps its old rule: it is emptied, not shorted.
    flows = pd.Series([0.0, 100.0, -300.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    m3 = trk._Market(flows.index, flows.index[-1], pd.DataFrame(index=flows.index), {}, pd.Series(10.0, index=flows.index), "x", {}, [])
    value, notes = trk._all_in(m3, flows)
    assert value.iloc[-1] == 0.0 and notes


def _frame(index: pd.DatetimeIndex, level: float) -> pd.DataFrame:
    close = pd.Series(level * (1.0 + 0.0003 * np.arange(len(index))), index=index)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0, "splits": 0.0})


def test_three_benchmarks_share_the_account_calendar(monkeypatch):
    ledger = _ledger(SAME_AS_INDEX)
    end = pd.Timestamp.today().normalize()
    london = pd.bdate_range("2023-06-01", end).drop(pd.Timestamp("2024-04-01"))  # a London holiday
    no_friday = pd.bdate_range("2023-06-01", end)
    no_friday = no_friday[no_friday != pd.Timestamp("2024-05-10")]
    late = pd.bdate_range("2024-06-03", end)
    frames = {"IWDA.L": _frame(london, 100.0), "CSPX.L": _frame(no_friday, 50.0), "CNDX.L": _frame(late, 70.0)}
    monkeypatch.setattr("data.price_history.fetch_price_history", lambda tickers, start=None: {t: frames[t] for t in tickers if t in frames})
    monkeypatch.setattr("data.fx.factor_series", lambda f, t, dates: pd.Series(1.0, index=pd.DatetimeIndex(dates)))
    m = trk._market(ledger, {}, lambda *a: None)
    assert list(m.benches) == ["msci_world", "sp500"]
    for series in m.benches.values():
        assert series.index.equals(m.days) and not series.isna().any()
    assert m.benches["msci_world"].loc["2024-04-01"] == m.benches["msci_world"].loc["2024-03-29"]  # carried over the holiday
    assert m.benches["sp500"].loc["2024-05-11"] == m.benches["sp500"].loc["2024-05-09"]  # a Friday and a weekend carried
    assert any("Nasdaq 100" in w and "left out of the comparison" in w for w in m.warnings)
    perf = trk._performance(ledger, m, trk._account(ledger, m))
    assert [b["id"] for b in perf["benchmarks"]] == ["msci_world", "sp500"]
    assert all(set(y["benchmarks"]) == {"msci_world", "sp500"} for y in perf["years"])
    assert perf["benchmarks"][0]["deposits_end_value"] == perf["benchmark_end_value"]

    monkeypatch.setattr("data.price_history.fetch_price_history", lambda tickers, start=None: {t: frames[t] for t in tickers if t != "IWDA.L" and t in frames})
    with pytest.raises(LookupError):
        trk._market(ledger, {}, lambda *a: None)


def test_result_series_share_one_calendar(offline_three):
    r = trk.run_track_record("fifo", False)
    times = [p["time"] for p in r["growth"]["invested_part"]]
    assert times and all([p["time"] for p in v] == times for v in r["growth"]["benchmarks"].values())
    inv = r["invested_equity"]
    assert [p["time"] for p in inv["invested_part"]] == times
    assert all([p["time"] for p in v] == times for v in inv["benchmarks"].values())
    assert set(r["equity"]["benchmarks"]) == set(IDS)
    json.dumps(r)


def test_skill_test_uses_the_invested_part_against_the_primary_only():
    paths = _paths()
    rng = np.random.default_rng(3)
    days = paths["msci_world"].index
    walk = 100.0 * np.cumprod(1.0 + np.where(np.arange(len(days)) < 3, 0.0, rng.normal(0.0005, 0.01, len(days))))
    paths["msci_world"] = pd.Series(walk, index=days)
    paths["sp500"] = pd.Series(100.0 * np.cumprod(1.0 + np.where(np.arange(len(days)) < 3, 0.0, rng.normal(0.0, 0.012, len(days)))), index=days)
    price = float(paths["msci_world"].loc["2024-01-03"])
    ledger = _ledger("2024-01-02,deposit,,,,,10000,,,EUR\n"
                     f"2024-01-03,buy,{FUND},Index Tracker Fund,{2000.0 / price!r},{price!r},-2000,,,EUR\n")
    out = _all(ledger, _market(ledger, {FUND: paths["msci_world"]}, paths))
    primary = out["perf"]["benchmarks"][0]["active"]
    assert primary["monthly_mean"] == pytest.approx(0.0, abs=1e-9)
    others = {b["id"]: b["active"] for b in out["perf"]["benchmarks"]}
    assert abs(others["sp500"]["monthly_mean"]) > 1e-4  # the S&P 500's path differs
    market = out["learnings"]["market"]
    conf, why = trk._confidence(primary["months"], primary["t"], trk._same_sign(primary["first_half"], primary["second_half"]))
    assert market["confidence"] == conf and market["why"].startswith(why)
    assert "S&P 500 t" in market["why"] and "shown, not tested" in market["why"]


def _old_result(r: dict) -> dict:
    old = copy.deepcopy(r)
    old.pop("schema", None)
    for k in ("benchmarks", "invested_xirr"):
        old["summary"].pop(k, None)
    old["growth"].pop("benchmarks", None)
    old["equity"].pop("benchmarks", None)
    old.pop("invested_equity", None)
    old.pop("cash", None)
    for y in old["years"]:
        y.pop("benchmarks", None)
        y.pop("avg_cash_share", None)
    for x in old["alternatives"] + old["learnings"]:
        x.pop("section", None)
    for x in old["alternatives"]:
        if x["key"] == "deposits_msci_world":
            x["key"] = "msci"
    return old


def test_old_results_still_render(offline_three, tmp_path):
    new = trk.run_track_record("fifo", False)
    old = _old_result(new)
    page = style_report.write_share_report(old, path=tmp_path / "s.html").read_text()
    end_value = new["summary"]["end_value"]
    assert "EUR" not in page and "€" not in page and f"{end_value:,.0f}" not in page
    assert "run My track record again" in page
    assert trk.write_report(old, tmp_path / "r.html").exists()
    trk.result_path().parent.mkdir(parents=True, exist_ok=True)
    trk.result_path().write_text(json.dumps(old))
    import web.app as web_app

    shared = web_app.app.test_client().post("/api/track-record/share", json={"names": False})
    assert shared.status_code == 200


def test_share_page_draws_the_indices_and_carries_no_amounts(offline_three, tmp_path):
    r = trk.run_track_record("fifo", False)
    page = style_report.write_share_report(r, path=tmp_path / "s.html").read_text()
    assert "S&amp;P 500" in page and "Nasdaq 100" in page and "Growth of 100" in page and "<svg" in page
    assert "EUR" not in page and "€" not in page
    s = r["summary"]
    mirror = s["benchmarks"][0]["buys_end_value"]
    for amount in (s["end_value"], s["invested_value"], mirror):
        if amount is not None:
            assert f"{amount:,.0f}" not in page
    for series in (r["equity"]["account"], r["equity"]["benchmarks"]["msci_world"]):
        assert not any(str(p["value"]) in page for p in series if abs(p["value"]) > 1000)


def test_performance_card_series_for_several_benchmarks():
    a, b = _walk(1), _walk(2)
    c = _walk(3).iloc[-100:]
    total = 100_000.0
    rows = [_row("H", a, 50_000.0, total)]
    entries = [{"id": i, "ticker": i.upper(), "label": f"Index {i}", "close": s, "currency": "USD"}
               for i, s in (("a", a), ("b", b), ("c", c))]
    bench = {"ticker": "A", "label": "Index a", "close": a, "currency": "USD"}
    out = pa.analyze(rows, {"H": a}, 50_000.0, "USD", {}, _unit_fx, total=total, benchmark=bench, benchmarks=entries)
    perf = out["performance"]
    series = perf["series"]
    dates = [p["time"] for p in series["holdings"]]
    assert set(series["benchmarks"]) == {"a", "b", "c"}
    assert all([p["time"] for p in v] == dates for v in series["benchmarks"].values())
    for period in perf["periods"]:
        assert period["holdings"] == pytest.approx(period["benchmarks"]["a"], abs=1e-6)
        assert abs(period["mix"]) < abs(period["holdings"])  # cash earns nothing: the mix is about half
        if period["key"] in ("1Y", "3Y", "5Y"):
            assert period["benchmarks"]["c"] is None and period["benchmarks_annual"]["c"] is None
    assert any(p["benchmarks"]["c"] is not None for p in perf["periods"])
    alone = pa.analyze(rows, {"H": a}, 50_000.0, "USD", {}, _unit_fx, total=total, benchmark=bench)
    assert out["risk"]["beta"] == alone["risk"]["beta"] and "benchmarks" not in alone["performance"]["series"]
    assert [x["id"] for x in out["benchmarks"]] == ["a", "b", "c"]
    missing = pa.analyze(rows, {"H": a}, 50_000.0, "USD", {}, _unit_fx, total=total, benchmark=bench,
                         benchmarks=entries[:1] + [{**entries[1], "close": None}])
    assert any("Index b" in n for n in missing["notes"]) and [x["id"] for x in missing["benchmarks"]] == ["a"]


def test_settings_keep_the_single_benchmark_for_old_callers():
    first = settings.PORTFOLIO_BENCHMARKS[0]
    assert settings.PORTFOLIO_BENCHMARK == {"ticker": first["ticker"], "label": first["label"]}
    assert len(set(IDS)) == len(IDS) == 3
    for b in settings.PORTFOLIO_BENCHMARKS:
        assert b["ticker"] and ISIN_RE.match(b["isin"]) and b["currency"] and b["fund"]
    assert trk.BENCHMARK["isin"] == first["isin"]
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    series = pd.Series([1.0, 2.0, 3.0], index=days)
    m = trk._Market(days, days[-1], pd.DataFrame(index=days), {}, series, "x", {}, [])  # the old signature
    listed = m.benchmarks()
    assert [meta["id"] for meta, _ in listed] == ["msci_world"] and listed[0][1] is series and listed[0][0]["primary"]


def _flat_paths(start: str, end: str, bump_on: str | None = None) -> dict[str, pd.Series]:
    days = pd.date_range(start, end, freq="D")
    up = pd.Series(100.0, index=days)
    if bump_on:
        up[up.index >= pd.Timestamp(bump_on)] = 110.0
    return {"msci_world": up, "sp500": up.copy(), "nasdaq100": up.copy()}


def test_a_year_that_closes_before_the_first_trade_has_no_index_return():
    # A December deposit, the first buy in January: the December year holds nothing invested, so no index figure either.
    ledger = _ledger("2023-12-27,deposit,,,,,10000,,,EUR\n"
                     "2024-01-15,buy," + FUND + ",Index Tracker Fund,20,100,-2000,,,EUR\n")
    paths = _flat_paths("2023-12-27", "2024-03-31", bump_on="2024-01-05")
    m = _market(ledger, {FUND: pd.Series(100.0, index=paths["msci_world"].index)}, paths)
    perf = trk._performance(ledger, m, trk._account(ledger, m))
    first, second = perf["years"][0], perf["years"][1]
    assert first["year"] == 2023 and first["benchmark"] is None
    assert all(v is None for v in first["benchmarks"].values())
    assert second["benchmark"] is not None and all(v is not None for v in second["benchmarks"].values())


def test_shares_moved_in_count_as_money_into_the_invested_part():
    other = "XS0000000002"
    ledger = _ledger(SAME_AS_INDEX + "2024-02-01,transfer_in," + other + ",Moved Fund,20,100,,,,EUR\n")
    flat = _flat_paths("2024-01-01", "2024-12-31")
    days = flat["msci_world"].index
    m = _market(ledger, {FUND: pd.Series(100.0, index=days), other: pd.Series(100.0, index=days)}, flat)
    a = trk._account(ledger, m)
    assert a["buys"].loc["2024-02-01"] == pytest.approx(2000.0)
    assert a["flows"].loc["2024-02-01"] == pytest.approx(2000.0)
    perf = trk._performance(ledger, m, a)
    b = perf["benchmarks"][0]
    assert perf["twr_invested_total"] == pytest.approx(0.0, abs=1e-9)
    assert b["buys_difference"] == pytest.approx(0.0, abs=0.01)
    assert b["cash_part"] == pytest.approx(0.0, abs=0.01)


def test_only_the_primary_follows_the_instrument_map(monkeypatch):
    ledger = _ledger(SAME_AS_INDEX)
    end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range("2023-06-01", end)
    frames = {t: _frame(idx, 100.0) for t in ("IWDA.SW", "CSPX.L", "CNDX.L", "CSNDX.SW")}
    monkeypatch.setattr("data.price_history.fetch_price_history", lambda tickers, start=None: {t: frames[t] for t in tickers if t in frames})
    monkeypatch.setattr("data.fx.factor_series", lambda f, t, dates: pd.Series(1.0, index=pd.DatetimeIndex(dates)))
    by_id = {b["id"]: b for b in settings.PORTFOLIO_BENCHMARKS}
    mapping = {by_id[i]["isin"]: {"ticker": t, "currency": "USD", "status": "ok"}
               for i, t in (("msci_world", "IWDA.SW"), ("nasdaq100", "CSNDX.SW"))}
    m = trk._market(ledger, mapping, lambda *a: None)
    tickers = {b["id"]: b["ticker"] for b in m.bench_meta}
    assert tickers == {"msci_world": "IWDA.SW", "sp500": "CSPX.L", "nasdaq100": "CNDX.L"}
