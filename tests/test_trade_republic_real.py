"""The owner's real Trade Republic export against the check figures of its specification.

Skipped unless FASTVALUE_TR_CSV points to the export. The expected figures are personal, so they are not in this
file: they live in data/store/tr_expected.json (git-ignored), or wherever FASTVALUE_TR_EXPECTED points.
Run: FASTVALUE_TR_CSV=~/Downloads/transactions_<from>_<to>.csv pytest tests/test_trade_republic_real.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import config.settings as settings
from core.ledger import build_ledger
from data.trade_republic import parse_export

CSV = Path(os.environ.get("FASTVALUE_TR_CSV", "")).expanduser()
EXPECTED = Path(os.environ.get("FASTVALUE_TR_EXPECTED", settings.PROJECT_ROOT / "data" / "store" / "tr_expected.json"))

pytestmark = pytest.mark.skipif(not (os.environ.get("FASTVALUE_TR_CSV") and CSV.is_file() and EXPECTED.is_file()),
                                reason="set FASTVALUE_TR_CSV to the export (expected figures in data/store/tr_expected.json)")


@pytest.fixture(scope="module")
def real():
    parsed = parse_export(CSV.read_bytes())
    return parsed, build_ledger(parsed["events"]), json.loads(EXPECTED.read_text())


def test_rows_and_types(real):
    parsed, _, exp = real
    assert parsed["rows"] == exp["rows"]
    for t, n in exp["type_counts"].items():
        assert parsed["type_counts"][t] == n
    assert all(c["ok"] for c in parsed["checks"]), parsed["checks"]


def test_cash_flows_fees_and_income(real):
    _, ledger, exp = real
    s = ledger["summary"]
    for key in ("cash", "net_external_flows", "securities_moved_out_at_cost", "fees", "dividends_gross",
                "liquidation_proceeds", "interest_gross"):
        assert s[key] == pytest.approx(exp[key], abs=0.005), key
    for t, v in exp["fees_by_type"].items():
        assert s["fees_by_type"][t] == pytest.approx(v, abs=0.005)


def test_realised_pnl_and_open_positions(real):
    parsed, ledger, exp = real
    for year, v in exp["realised_average"].items():
        assert ledger["summary"]["realised_by_year"][year]["average"] == pytest.approx(v, abs=0.005)
    private = {e["isin"] for e in parsed["events"] if e["asset_class"] == "PRIVATE_FUND"}
    assert len(ledger["positions"]) + len(private) == exp["open_positions_including_private"]
    pos = next(p for p in ledger["positions"] if p["isin"] == exp["position"]["isin"])
    assert pos["shares"] == pytest.approx(exp["position"]["shares"], abs=1e-6)
    assert pos["cost_avg"] == pytest.approx(exp["position"]["cost_avg"], abs=0.005)


def test_cost_matches_the_brokers_records(real):
    _, ledger, exp = real
    mig = ledger["migration_check"]
    assert sum(1 for m in mig if m["diff_pct"] is not None and abs(m["diff_pct"]) <= 0.001) == exp["broker_cost_matches"]
