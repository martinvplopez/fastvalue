"""Tests for the company reports from SEC EDGAR (financial overview and recent filings), on synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config.settings as settings
import data.sec_reports as sr
from data.sec_edgar import CompanyFacts, SecDataError
from data.sec_filings import NO_ACCESS

# A company with a September fiscal year. Quarterly flows by fiscal year; the 10-K gives the year, the 10-Qs the
# first three quarters (income lines as discrete quarters and year to date, cash flow lines year to date only).
REV = {2023: [900, 950, 1000, 1150], 2024: [1000, 1050, 1100, 1250], 2025: [1200, 1300]}
OCF = {2023: [140, 160, 180, 320], 2024: [150, 200, 210, 240], 2025: [200, 220]}
CAPEX = {2023: [25, 30, 30, 35], 2024: [30, 40, 30, 40], 2025: [40, 50]}
SHARES = {2023: [100, 100, 100, 100], 2024: [99, 98, 97, 96], 2025: [95, 94]}
FILED = {(2023, 1): "2023-02-02", (2023, 2): "2023-05-04", (2023, 3): "2023-08-03", (2023, 4): "2023-11-09",
         (2024, 1): "2024-02-01", (2024, 2): "2024-05-02", (2024, 3): "2024-08-01", (2024, 4): "2024-11-08",
         (2025, 1): "2025-02-04", (2025, 2): "2025-05-06"}
SIC = 3571


def _bounds(fy: int, j: int) -> tuple[str, str]:
    starts = {1: f"{fy - 1}-10-01", 2: f"{fy}-01-01", 3: f"{fy}-04-01", 4: f"{fy}-07-01"}
    ends = {1: f"{fy - 1}-12-31", 2: f"{fy}-03-31", 3: f"{fy}-06-30", 4: f"{fy}-09-30"}
    return starts[j], ends[j]


class _Raw:
    def __init__(self) -> None:
        self.usd: dict[str, list] = {}
        self.shares: dict[str, list] = {}
        self.per_share: dict[str, list] = {}

    def add(self, concept: str, start: str | None, end: str, val: float, fy: int, j: int, unit: str = "USD") -> None:
        entry = {"end": end, "val": val, "filed": FILED[(fy, j)], "form": "10-K" if j == 4 else "10-Q",
                 "accn": f"0000000001-{FILED[(fy, j)][2:4]}-{j:06d}"}
        if start:
            entry["start"] = start
        target = {"USD": self.usd, "shares": self.shares, "USD/shares": self.per_share}[unit]
        target.setdefault(concept, []).append(entry)

    def flow(self, concept: str, by_year: dict[int, list[float]], discrete: bool, unit: str = "USD", average: bool = False) -> None:
        for fy, quarters in by_year.items():
            fy_start = _bounds(fy, 1)[0]
            for j, value in enumerate(quarters, start=1):
                start, end = _bounds(fy, j)
                running = quarters[:j]
                to_date = sum(running) / j if average else sum(running)
                if j < 4:
                    if discrete:
                        self.add(concept, start, end, value, fy, j, unit)
                    if j == 1 and not discrete:
                        self.add(concept, start, end, value, fy, j, unit)
                    if j >= 2:
                        self.add(concept, fy_start, end, to_date, fy, j, unit)
                else:
                    self.add(concept, fy_start, end, to_date, fy, j, unit)

    def instant(self, concept: str, value: float) -> None:
        for (fy, j) in FILED:
            self.add(concept, None, _bounds(fy, j)[1], value + fy, fy, j)

    def json(self) -> dict:
        facts = {"us-gaap": {c: {"units": {"USD": v}} for c, v in self.usd.items()}}
        for c, v in self.shares.items():
            facts["us-gaap"][c] = {"units": {"shares": v}}
        for c, v in self.per_share.items():
            facts["us-gaap"][c] = {"units": {"USD/shares": v}}
        facts["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2025-04-25", "val": 93.0, "filed": FILED[(2025, 2)], "form": "10-Q", "accn": "0000000001-25-000002"}]}}}
        return {"cik": 1, "entityName": "Test Co", "facts": facts}


def _raw(skip: tuple[str, ...] = ()) -> dict:
    raw = _Raw()
    ni = {fy: [r * 0.10 for r in q] for fy, q in REV.items()}
    flows = {
        "Revenues": REV, "NetIncomeLoss": ni, "OperatingIncomeLoss": {fy: [r * 0.15 for r in q] for fy, q in REV.items()},
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            {fy: [r * 0.13 for r in q] for fy, q in REV.items()},
        "IncomeTaxExpenseBenefit": {fy: [r * 0.03 for r in q] for fy, q in REV.items()},
        "InterestExpense": {fy: [r * 0.01 for r in q] for fy, q in REV.items()},
        "CostOfRevenue": {fy: [r * 0.60 for r in q] for fy, q in REV.items()},
        "PaymentsOfDividends": {fy: [10.0 for _ in q] for fy, q in REV.items()},
    }
    for concept, by_year in flows.items():
        if concept not in skip:
            raw.flow(concept, by_year, discrete=True)
    raw.flow("DepreciationDepletionAndAmortization", {fy: [20.0] * len(q) for fy, q in REV.items()}, discrete=False)
    raw.flow("NetCashProvidedByUsedInOperatingActivities", OCF, discrete=False)
    raw.flow("ShareBasedCompensation", {fy: [5.0] * len(q) for fy, q in REV.items()}, discrete=False)
    raw.flow("PaymentsForRepurchaseOfCommonStock", {fy: [12.0] * len(q) for fy, q in REV.items()}, discrete=False)
    if "PaymentsToAcquirePropertyPlantAndEquipment" not in skip:
        raw.flow("PaymentsToAcquirePropertyPlantAndEquipment", CAPEX, discrete=False)
    raw.flow("WeightedAverageNumberOfDilutedSharesOutstanding", SHARES, discrete=True, unit="shares", average=True)
    raw.flow("EarningsPerShareDiluted", {fy: [n / s for n, s in zip(ni[fy], SHARES[fy])] for fy in ni}, discrete=True, unit="USD/shares")
    for concept, value in (("Assets", 5000), ("StockholdersEquity", 2000), ("LongTermDebt", 500),
                           ("CashAndCashEquivalentsAtCarryingValue", 400), ("AccountsReceivableNetCurrent", 300),
                           ("AccountsPayableCurrent", 200)):
        raw.instant(concept, value)
    return raw.json()


def _submissions(sic: int = SIC) -> dict:
    forms = [("10-Q", "2025-05-06", "", "q2.htm", "2025-03-31"), ("10-K", "2024-11-08", "", "k24.htm", "2024-09-30")]
    return {
        "name": "Test Co", "sic": str(sic), "sicDescription": "Computers", "website": "", "investorWebsite": "",
        "filings": {"recent": {
            "form": [f[0] for f in forms], "filingDate": [f[1] for f in forms], "items": [f[2] for f in forms],
            "primaryDocument": [f[3] for f in forms], "reportDate": [f[4] for f in forms],
            "accessionNumber": ["0000000001-25-000002", "0000000001-24-000004"],
        }},
    }


def _prices(split: tuple[str, float] | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2022-06-01", "2025-06-30")
    frame = pd.DataFrame({"close": np.linspace(10.0, 20.0, len(index)), "splits": 0.0}, index=index)
    if split:
        frame.loc[pd.Timestamp(split[0]), "splits"] = split[1]
    return frame


@pytest.fixture
def sec(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """SEC access with one synthetic company, TEST, and a price history. Nothing touches the network or the real cache."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "cache")
    state = {"raw": _raw(), "subs": _submissions(), "prices": _prices()}
    monkeypatch.setattr(sr, "has_sec_access", lambda: True)
    monkeypatch.setattr(sr, "_ticker_map", lambda: {"TEST": 1, "SAP": 2})
    monkeypatch.setattr(sr, "load_report_facts", lambda cik, *a, **k: sr.extract_report_facts(state["raw"]))
    monkeypatch.setattr(sr, "_submissions", lambda cik: state["subs"])
    monkeypatch.setattr(sr, "_price_frame", lambda symbol: state["prices"])
    return state


def _rows(payload: dict) -> dict:
    return {r["key"]: r for s in payload["sections"] for r in s["rows"]}


def _col(payload: dict, key: str) -> int:
    return [c["key"] for c in payload["columns"]].index(key)


# ------------------------------------------------------------------ the annual table


def test_annual_table_shape_and_contract(sec):
    out = sr.financial_overview("test")
    assert out["status"] == "ok" and out["ticker"] == "TEST" and out["period"] == "annual"
    assert out["cik"] == 1 and out["currency"] == "USD" and out["is_financial"] is False
    assert [c["key"] for c in out["columns"]] == ["FY2023", "FY2024", "LTM"]
    assert out["columns"][1]["form"] == "10-K" and out["columns"][1]["filed"] == "2024-11-08"
    assert out["columns"][1]["url"].endswith("/k24.htm")  # primary document from the submissions JSON
    assert out["source"]["url"].endswith("CIK=0000000001&type=&dateb=&owner=include&count=40")
    assert [s["key"] for s in out["sections"]] == ["income", "cash_flow", "balance", "returns", "shareholders", "valuation"]
    rows = _rows(out)
    for key in ("revenue", "revenue_growth", "gross_margin", "ebitda", "eps_diluted", "fcf", "sbc", "acquisitions", "net_debt",
                "working_capital", "roic", "incremental_roic", "shares_change", "net_dilution", "price", "ev_to_fcf", "ev_to_ebit"):
        assert key in rows, key
    assert all(len(r["values"]) == 3 for r in rows.values())
    assert {k for k, r in rows.items() if r["highlight"]} == {
        "fcf", "roic", "ev_to_fcf", "pe", "reinvestment_rate", "net_dilution", "shares_diluted"}


def test_annual_values_follow_the_owner_definitions(sec):
    rows = _rows(sr.financial_overview("TEST"))
    fy24 = 1
    assert rows["revenue"]["values"][fy24] == pytest.approx(sum(REV[2024]))
    assert rows["revenue_growth"]["values"][fy24] == pytest.approx(sum(REV[2024]) / sum(REV[2023]) - 1)
    assert rows["gross_profit"]["values"][fy24] == pytest.approx(sum(REV[2024]) * 0.40)  # revenue minus cost of revenue
    assert rows["ebit"]["values"][fy24] == pytest.approx(sum(REV[2024]) * 0.14)  # pretax 13% plus interest 1%
    assert rows["ebitda"]["values"][fy24] == pytest.approx(sum(REV[2024]) * 0.14 + 80)
    assert rows["operating_cash_flow"]["values"][fy24] == pytest.approx(800)
    assert rows["capex"]["values"][fy24] == pytest.approx(140)
    assert rows["fcf"]["values"][fy24] == pytest.approx(660)  # FCF_DEFINITION "standard": operating cash flow minus CapEx
    assert "reported FCF" in rows["fcf"]["tip"]
    assert rows["total_debt"]["values"][fy24] == 500 + 2024
    assert rows["net_debt"]["values"][fy24] == pytest.approx((500 + 2024) - (400 + 2024))
    assert rows["working_capital"]["values"][fy24] == pytest.approx((300 + 2024) - (200 + 2024))  # no inventory line: zero
    assert rows["shares_diluted"]["values"][fy24] == pytest.approx(97.5)
    assert rows["shares_change"]["values"][fy24] == pytest.approx(97.5 / 100 - 1)
    assert rows["eps_diluted"]["values"][fy24] == pytest.approx(sum(r * 0.1 / s for r, s in zip(REV[2024], SHARES[2024])))
    assert rows["dividends"]["values"][fy24] == pytest.approx(40)
    assert rows["sbc"]["values"][fy24] == pytest.approx(20) and rows["buybacks"]["values"][fy24] == pytest.approx(48)
    assert rows["sbc_to_revenue"]["values"][fy24] == pytest.approx(20 / sum(REV[2024]))
    cap = rows["market_cap"]["values"][fy24]
    assert rows["net_dilution"]["values"][fy24] == pytest.approx((20 - 48) / cap)  # (stock pay - buybacks) / market cap


def test_valuation_uses_the_close_on_the_period_end(sec):
    out = sr.financial_overview("TEST")
    rows = _rows(out)
    close = sec["prices"]["close"]
    assert rows["price"]["values"][1] == pytest.approx(close.loc[:"2024-09-30"].iloc[-1])
    assert "period end" in rows["price"]["tip"]
    cap = rows["price"]["values"][1] * 97.5
    assert rows["market_cap"]["values"][1] == pytest.approx(cap)
    assert rows["enterprise_value"]["values"][1] == pytest.approx(cap + 100)  # debt 2524 minus cash 2424
    assert rows["ev_to_fcf"]["values"][1] == pytest.approx((cap + 100) / 660)
    assert rows["pe"]["values"][1] == pytest.approx(cap / (sum(REV[2024]) * 0.10))
    assert rows["fcf_yield"]["values"][1] == pytest.approx(660 / (cap + 100))
    # The LTM column uses the latest close and the latest cover-page share count.
    assert rows["price"]["values"][2] == pytest.approx(close.iloc[-1])
    assert rows["market_cap"]["values"][2] == pytest.approx(close.iloc[-1] * 93.0)


def test_ltm_column_adds_the_quarters_since_the_last_10k(sec):
    out = sr.financial_overview("TEST")
    ltm = out["columns"][_col(out, "LTM")]
    assert ltm["end"] == "2025-03-31" and ltm["derived"] is True and ltm["form"] == "10-Q"
    rows = _rows(out)
    assert rows["revenue"]["values"][2] == pytest.approx(1100 + 1250 + 1200 + 1300)  # year plus the new half minus the old half
    assert rows["operating_cash_flow"]["values"][2] == pytest.approx(800 + 420 - 350)
    assert rows["capex"]["values"][2] == pytest.approx(140 + 90 - 70)
    assert any("LTM" in n and "2025-03-31" in n for n in out["notes"])


def test_missing_concept_stays_null_with_a_note(sec):
    out = sr.financial_overview("TEST")
    rows = _rows(out)
    assert rows["acquisitions"]["values"] == [None, None, None]  # never tagged: not estimated, not zero
    assert rows["sbc"]["values"][1] == pytest.approx(20)
    assert any(n.startswith("Acquisitions is not tagged") for n in out["notes"])
    sec["raw"] = _raw(skip=("PaymentsToAcquirePropertyPlantAndEquipment",))
    sr._model.cache_clear()
    out = sr.financial_overview("TEST")
    rows = _rows(out)
    assert rows["capex"]["values"] == [None, None, None] and rows["fcf"]["values"] == [None, None, None]
    assert any(n.startswith("CapEx is not tagged") for n in out["notes"])


def test_annual_statements_match_the_value_study_extract(sec):
    """The overview maps values with the same function as CompanyFacts, so a fiscal year reads identically."""
    model = sr._model("TEST", 1)
    cf = CompanyFacts(ticker="TEST", cik=1, name="Test Co", sic=SIC, sic_description="x", facts=sr.extract_report_facts(sec["raw"]))
    study = cf.statements_asof(pd.Timestamp("2025-06-30"), max_years=4)
    end = pd.Timestamp("2024-09-30")
    for statement, row in (("income", "Total Revenue"), ("income", "EBITDA"), ("income", "Diluted Average Shares"),
                           ("cashflow", "Operating Cash Flow"), ("cashflow", "Capital Expenditure"), ("balance", "Total Debt"),
                           ("balance", "Cash Cash Equivalents And Short Term Investments")):
        assert model.st_ttm.value(statement, row, end) == pytest.approx(study.value(statement, row, end)), row


def test_checklist_uses_the_framework_references(sec):
    out = sr.financial_overview("TEST")
    items = {c["key"]: c for c in out["checklist"]}
    assert list(items) == ["ev_fcf", "pe", "roic", "roic_persistence", "reinvestment_rate", "net_dilution", "sbc_to_revenue",
                           "net_debt_to_fcf", "fcf_margin"]
    assert items["ev_fcf"]["reference"] == 16.5 and items["ev_fcf"]["better"] == "lower"
    assert items["pe"]["reference"] == 16.0
    assert items["roic"]["reference"] == 0.135 and items["roic"]["better"] == "higher"
    ltm = _rows(out)
    assert items["ev_fcf"]["value"] == pytest.approx(ltm["ev_to_fcf"]["values"][2])  # LTM is the latest column
    assert items["ev_fcf"]["verdict"] in ("better", "worse", "in line")
    assert items["net_dilution"]["verdict"] == "better" and items["net_dilution"]["value"] < 0  # buybacks 48 a year against stock pay 20
    assert items["roic_persistence"]["unit"] == "years" and items["roic_persistence"]["value"] is not None
    assert items["sbc_to_revenue"]["value"] == pytest.approx(20 / 4850)
    assert items["sbc_to_revenue"]["verdict"] == "in line"
    # Only the market-average checks are worded against the market; the others say what their rule means.
    assert items["ev_fcf"]["verdict_text"] is None and items["roic"]["verdict_text"] is None
    assert items["sbc_to_revenue"]["verdict_text"] == "Below 6%"
    assert items["net_dilution"]["verdict_text"] == "Buybacks cover stock pay"
    persistence = items["roic_persistence"]
    assert persistence["reference"] is None and persistence["value_text"].endswith("years") and " of " in persistence["value_text"]


def test_verdict_bands(sec):
    out = sr.financial_overview("TEST")
    pe = next(c for c in out["checklist"] if c["key"] == "pe")
    assert pe["value"] < 16.0 * 0.8 and pe["verdict"] == "better"  # far below the market average of 16x


def test_bank_skips_ev_fcf_and_roic(sec):
    sec["subs"] = _submissions(sic=6021)
    out = sr.financial_overview("TEST")
    assert out["is_financial"] is True
    items = {c["key"]: c for c in out["checklist"]}
    assert items["ev_fcf"]["value"] is None and items["ev_fcf"]["verdict"] is None and "banks" in items["ev_fcf"]["tip"]
    assert items["roic"]["verdict"] is None
    assert {"roe", "roa", "pe"} <= set(items) and items["roe"]["value"] is not None
    rows = _rows(out)
    assert rows["fcf"]["values"] == [None] * 3 and rows["roic"]["values"] == [None] * 3 and rows["ev_to_fcf"]["values"] == [None] * 3
    assert rows["roe"]["values"][1] is not None and rows["pe"]["values"][1] is not None
    assert any("Bank or insurer" in n for n in out["notes"])


# ------------------------------------------------------------------ quarters


def test_quarterly_columns_and_year_to_date_derivation(sec):
    out = sr.financial_overview("TEST", "quarterly")
    keys = [c["key"] for c in out["columns"]]
    assert keys == ["FY2023Q1", "FY2023Q2", "FY2023Q3", "FY2023Q4", "FY2024Q1", "FY2024Q2", "FY2024Q3", "FY2024Q4", "FY2025Q1", "FY2025Q2"]
    rows = _rows(out)
    col = {k: i for i, k in enumerate(keys)}
    # Income lines are tagged for the discrete quarter; cash flow lines only year to date.
    assert rows["revenue"]["values"][col["FY2024Q2"]] == pytest.approx(1050)
    assert rows["operating_cash_flow"]["values"][col["FY2024Q1"]] == pytest.approx(150)
    assert rows["operating_cash_flow"]["values"][col["FY2024Q2"]] == pytest.approx(200)  # 6 months 350 minus 3 months 150
    assert rows["operating_cash_flow"]["values"][col["FY2024Q3"]] == pytest.approx(210)  # 560 minus 350
    assert rows["capex"]["values"][col["FY2024Q3"]] == pytest.approx(30)  # 100 minus 70
    assert rows["fcf"]["values"][col["FY2024Q2"]] == pytest.approx(200 - 40)
    derived = {c["key"]: c["derived"] for c in out["columns"]}
    assert derived["FY2024Q1"] is False and derived["FY2024Q2"] is True and derived["FY2024Q4"] is True
    assert any("year-to-date" in n for n in out["notes"])


def test_fourth_quarter_is_the_year_minus_nine_months(sec):
    out = sr.financial_overview("TEST", "quarterly")
    rows = _rows(out)
    q4 = [c["key"] for c in out["columns"]].index("FY2024Q4")
    assert rows["revenue"]["values"][q4] == pytest.approx(sum(REV[2024]) - sum(REV[2024][:3]))
    assert rows["operating_cash_flow"]["values"][q4] == pytest.approx(240)
    assert rows["capex"]["values"][q4] == pytest.approx(40)
    assert rows["shares_diluted"]["values"][q4] == pytest.approx(96)  # 4 x year average minus 3 x nine-month average
    assert out["columns"][q4]["form"] == "10-K" and out["columns"][q4]["end"] == "2024-09-30"


def test_quarter_ratios_use_trailing_twelve_months(sec):
    out = sr.financial_overview("TEST", "quarterly")
    rows = _rows(out)
    q = [c["key"] for c in out["columns"]].index("FY2024Q2")
    # Twelve months to 2024-03-31: Q3 FY23 + Q4 FY23 + Q1 FY24 + Q2 FY24.
    ttm_ni = 0.1 * (1000 + 1150 + 1000 + 1050)
    cap = rows["market_cap"]["values"][q]
    assert rows["pe"]["values"][q] == pytest.approx(cap / ttm_ni)
    ttm_fcf = (180 + 320 + 150 + 200) - (30 + 35 + 30 + 40)
    assert rows["ev_to_fcf"]["values"][q] == pytest.approx(rows["enterprise_value"]["values"][q] / ttm_fcf)
    first = rows["pe"]["values"][0]
    assert first is None  # no earlier year to build twelve months from
    assert any("lack enough income statement rows" in n for n in sec_notes(out))


def sec_notes(out: dict) -> list[str]:
    return out["notes"]


def test_quarterly_growth_is_year_on_year(sec):
    out = sr.financial_overview("TEST", "quarterly")
    rows = _rows(out)
    q = [c["key"] for c in out["columns"]].index("FY2025Q2")
    assert rows["revenue_growth"]["values"][q] == pytest.approx(1300 / 1050 - 1)
    assert rows["shares_change"]["values"][q] == pytest.approx(94 / 98 - 1)


# ------------------------------------------------------------------ splits and statuses


def test_later_split_restates_shares_and_per_share_figures():
    frame = pd.DataFrame({
        "concept": ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding", "EarningsPerShareDiluted", "Revenues"],
        "unit": ["shares", "shares", "USD/shares", "USD"],
        "start": pd.to_datetime(["2024-01-01"] * 4), "end": pd.to_datetime(["2024-03-31"] * 4),
        "val": [100.0, 100.0, 2.0, 500.0],
        "filed": pd.to_datetime(["2024-05-01", "2024-09-01", "2024-05-01", "2024-05-01"]), "form": ["10-Q"] * 4, "accn": ["a"] * 4,
    })
    splits = pd.Series([2.0], index=pd.to_datetime(["2024-07-01"]))
    out = sr._restate_for_splits(frame, splits, pd.Timestamp("2025-01-01"))
    assert list(out["val"]) == [200.0, 100.0, 1.0, 500.0]  # filed after the split: already restated


def test_no_access_names_the_fix(monkeypatch):
    monkeypatch.setattr(sr, "has_sec_access", lambda: False)
    for out in (sr.financial_overview("KO"), sr.recent_filings("KO")):
        assert out["status"] == "no_access" and out["fix"] == NO_ACCESS and out["message"]
    assert sr.financial_overview("KO")["columns"] == []


def test_ticker_missing_from_the_sec_map(sec):
    for out in (sr.financial_overview("ITX.MC"), sr.recent_filings("ITX.MC")):
        assert out["status"] == "not_sec_filer" and "not in the SEC" in out["message"] and out["fix"] is None


def test_foreign_filer_has_no_statements_but_filings(sec):
    sec["subs"] = {"name": "SAP", "sic": "7372", "filings": {"recent": {
        "form": ["6-K", "20-F"], "filingDate": ["2025-07-15", "2025-02-25"], "items": ["", ""], "primaryDocument": ["a.htm", "b.htm"],
        "reportDate": ["2025-06-29", "2024-12-31"], "accessionNumber": ["0000000002-25-000001", "0000000002-25-000002"]}}}
    out = sr.financial_overview("SAP")
    assert out["status"] == "error" and out["reason"] == "foreign_filer" and "20-F" in out["message"]
    filings = sr.recent_filings("SAP")
    assert filings["status"] == "ok" and [f["kind"] for f in filings["filings"]] == ["foreign_current", "foreign_annual"]
    assert filings["latest"]["annual"]["form"] == "20-F" and any("6-K exhibits are not itemised" in n for n in filings["notes"])
    assert "forms=6-K" in next(link["url"] for link in filings["links"] if link["label"] == "Call transcripts")


def test_network_failure_is_an_error_and_is_not_cached(sec, monkeypatch):
    calls = {"n": 0}

    def flaky(cik):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SecDataError("SEC request failed (503)")
        return sec["subs"]

    monkeypatch.setattr(sr, "_submissions", flaky)
    first = sr.financial_overview("TEST")
    assert first["status"] == "error" and "503" in first["message"] and first["fix"]
    assert sr.financial_overview("TEST")["status"] == "ok"  # the failure was not remembered


def test_no_company_facts_is_not_a_failure_to_retry_and_shows_no_address(sec, monkeypatch):
    url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    monkeypatch.setattr(sr, "_model", lambda symbol, cik: (_ for _ in ()).throw(SecDataError(f"SEC has no data at {url}")))
    out = sr.financial_overview("TEST")
    assert out["reason"] == "no_company_facts" and out["fix"] is None and "https://" not in out["message"]
    assert "no reported figures for TEST" in out["message"]
    monkeypatch.setattr(sr, "_model", lambda symbol, cik: (_ for _ in ()).throw(SecDataError(f"SEC request failed (503) at {url}")))
    other = sr.financial_overview("TEST")
    assert "reason" not in other and "https://" not in other["message"] and other["fix"]


def test_warm_call_reads_nothing_again(sec, monkeypatch):
    sr.financial_overview("TEST")
    monkeypatch.setattr(sr, "load_report_facts", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded again")))
    assert sr.financial_overview("TEST", "quarterly")["status"] == "ok"  # annual and quarterly share one model


# ------------------------------------------------------------------ filings

INDEX_HTML = """
<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td scope="row">1</td><td scope="row">8-K</td><td scope="row"><a href="/ix?doc=/Archives/edgar/data/1/000000000125000009/main.htm">main.htm</a></td><td scope="row">8-K</td><td>9</td></tr>
<tr><td scope="row">2</td><td scope="row">Press release</td><td scope="row"><a href="/Archives/edgar/data/1/000000000125000009/ex991.htm">ex991.htm</a></td><td scope="row">EX-99.1</td><td>9</td></tr>
<tr><td scope="row">3</td><td scope="row">Earnings call presentation</td><td scope="row"><a href="/Archives/edgar/data/1/000000000125000009/ex992.htm">ex992.htm</a></td><td scope="row">EX-99.2</td><td>9</td></tr>
<tr><td scope="row">4</td><td scope="row">Conference call transcript</td><td scope="row"><a href="/Archives/edgar/data/1/000000000125000009/ex993.htm">ex993.htm</a></td><td scope="row">EX-99.3</td><td>9</td></tr>
<tr><td scope="row">5</td><td scope="row">Supplemental data</td><td scope="row"><a href="/Archives/edgar/data/1/000000000125000009/ex994.htm">ex994.htm</a></td><td scope="row">EX-99.4</td><td>9</td></tr>
<tr><td scope="row">6</td><td scope="row">Cover Page Interactive Data File</td><td scope="row"><a href="/Archives/edgar/data/1/000000000125000009/cover.htm">cover.htm</a></td><td scope="row">EX-104</td><td>9</td></tr>
</table>
"""


def test_exhibits_are_labelled():
    out = sr.parse_filing_index(INDEX_HTML, 1, ["2.02", "9.01"])
    assert [(x["kind"], x["label"]) for x in out] == [
        ("earnings", "Earnings press release (Ex. 99.1)"),
        ("presentation", "Investor presentation or slides (Ex. 99.2)"),
        ("transcript", "Call transcript (Ex. 99.3)"),
        ("presentation", "Investor presentation or slides (Ex. 99.4)"),  # a supplement is slides or a data pack
    ]
    assert out[0]["url"] == "https://www.sec.gov/Archives/edgar/data/1/000000000125000009/ex991.htm"
    plain = '<tr><td>2</td><td></td><td><a href="/Archives/edgar/data/1/x/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>1</td></tr>'
    assert sr.parse_filing_index(plain, 1, ["2.02"])[0]["kind"] == "earnings"  # 99.1 with Item 2.02 is the results release
    assert sr.parse_filing_index(plain, 1, ["8.01"])[0]["kind"] == "other"


def test_form_and_item_classification():
    assert sr._classify_form("10-K", []) == ("annual", "Annual report (10-K)")
    assert sr._classify_form("10-Q", [])[0] == "quarterly"
    assert sr._classify_form("8-K", ["2.02", "9.01"]) == ("earnings", "Quarterly results (Item 2.02)")
    assert sr._classify_form("8-K", ["7.01"])[0] == "current" and "Regulation FD" in sr._classify_form("8-K", ["7.01"])[1]
    assert sr._classify_form("8-K", ["5.02", "9.01"])[1] == "Current report: executive or director change"
    assert sr._classify_form("DEF 14A", [])[0] == "proxy"
    assert sr._classify_form("20-F", [])[0] == "foreign_annual" and sr._classify_form("6-K", [])[0] == "foreign_current"
    assert sr._classify_form("10-K/A", [])[1].endswith("(amendment)")


def _filing_subs() -> dict:
    rows = [  # form, filed, items, accession, document
        ("8-K", "2025-05-06", "2.02,9.01", "0000000001-25-000009", "main.htm"),
        ("10-Q", "2025-05-06", "", "0000000001-25-000002", "q2.htm"),
        ("8-K", "2025-04-01", "7.01", "0000000001-25-000008", "fd.htm"),
        ("4", "2025-03-20", "", "0000000001-25-000007", "f4.xml"),
        ("8-K", "2025-03-01", "5.02", "0000000001-25-000006", "exec.htm"),
        ("DEF 14A", "2025-01-10", "", "0000000001-25-000005", "proxy.htm"),
        ("10-K", "2024-11-08", "", "0000000001-24-000004", "k24.htm"),
    ]
    return {"name": "Test Co", "sic": "3571", "website": "www.test.example", "filings": {"recent": {
        "form": [r[0] for r in rows], "filingDate": [r[1] for r in rows], "items": [r[2] for r in rows],
        "accessionNumber": [r[3] for r in rows], "primaryDocument": [r[4] for r in rows], "reportDate": ["2025-03-31"] * len(rows)}}}


FD_HTML = INDEX_HTML.replace("Press release", "Investor presentation").replace("Earnings call presentation", "Other")


def test_recent_filings_payload(sec, monkeypatch):
    sec["subs"] = _filing_subs()
    fetched: list[str] = []

    def fake_fetch(url: str) -> str:
        fetched.append(url)
        return INDEX_HTML if "000000000125000009" in url else FD_HTML

    monkeypatch.setattr(sr, "_fetch_text", fake_fetch)
    out = sr.recent_filings("TEST", limit=5)
    assert out["status"] == "ok" and out["cik"] == 1 and out["name"] == "Test Co"
    assert [f["form"] for f in out["filings"]] == ["8-K", "10-Q", "8-K", "8-K", "DEF 14A"]  # newest first, Form 4 left out, limit applied
    results = out["filings"][0]
    assert results["kind"] == "earnings" and results["items"] == ["2.02", "9.01"]
    assert results["url"].endswith("/000000000125000009/main.htm") and results["index_url"].endswith("0000000001-25-000009-index.htm")
    assert [x["kind"] for x in results["exhibits"]] == ["earnings", "presentation", "transcript", "presentation"]
    fd = out["filings"][2]
    assert fd["kind"] == "presentation" and fd["title"] == "Investor presentation (Item 7.01)"  # named after its exhibit
    assert fd["exhibits"][0]["kind"] == "presentation"
    assert out["filings"][3]["kind"] == "current" and out["filings"][3]["exhibits"] == []  # no exhibit lookup for other 8-Ks
    latest = out["latest"]
    assert latest["annual"]["form"] == "10-K" and latest["quarterly"]["form"] == "10-Q" and latest["proxy"]["form"] == "DEF 14A"
    assert latest["earnings"]["filed"] == "2025-05-06" and latest["presentation"]["filed"] == "2025-05-06"
    labels = {link["label"] for link in out["links"]}
    assert {"All filings on EDGAR", "Call transcripts", "Insider trades (Form 4)", "Company website"} <= labels
    transcripts = next(link for link in out["links"] if link["label"] == "Call transcripts")
    assert "ciks=0000000001" in transcripts["url"] and "not filed with the SEC" in transcripts["note"]
    assert next(link for link in out["links"] if link["label"] == "Company website")["url"] == "https://www.test.example"
    assert len(fetched) == 2  # one index page per 8-K with Item 2.02 or 7.01


def test_filings_are_cached_per_company_and_exhibits_for_good(sec, monkeypatch):
    sec["subs"] = _filing_subs()
    count = {"n": 0}

    def fake_fetch(url: str) -> str:
        count["n"] += 1
        return INDEX_HTML

    monkeypatch.setattr(sr, "_fetch_text", fake_fetch)
    first = sr.recent_filings("TEST")
    second = sr.recent_filings("TEST", limit=3)
    assert count["n"] == 2 and second["filings"] == first["filings"][:3]
    # A new payload (the day's cache expired) still finds the exhibit lists on disk.
    (sr.sec_cache_dir() / "filings_v1_0000000001.json").unlink()
    sr.recent_filings("TEST")
    assert count["n"] == 2


def test_failed_exhibit_lookup_is_not_cached_as_complete(sec, monkeypatch):
    sec["subs"] = _filing_subs()
    state = {"fail": True}

    def fake_fetch(url: str) -> str:
        if state["fail"]:
            raise SecDataError("SEC request failed (503)")
        return INDEX_HTML

    monkeypatch.setattr(sr, "_fetch_text", fake_fetch)
    broken = sr.recent_filings("TEST")
    assert broken["status"] == "ok" and broken["filings"][0]["exhibits"] == []
    state["fail"] = False
    assert sr.recent_filings("TEST")["filings"][0]["exhibits"]  # retried: the gap was not cached


# ------------------------------------------------------------------ routes


@pytest.fixture
def client(sec, monkeypatch):
    from web.app import app

    sec["subs"] = _filing_subs()
    monkeypatch.setattr(sr, "_fetch_text", lambda url: INDEX_HTML)
    return app.test_client()


def test_routes_return_the_payloads_and_record_no_history(client):
    r = client.get("/api/financials/test?period=quarterly")
    body = r.get_json()
    assert r.status_code == 200 and body["status"] == "ok" and body["period"] == "quarterly" and body["ticker"] == "TEST"
    assert len(body["columns"]) == 10 and all(len(row["values"]) == 10 for s in body["sections"] for row in s["rows"])
    assert client.get("/api/financials/TEST").get_json()["period"] == "annual"
    f = client.get("/api/filings/TEST?limit=3").get_json()
    assert f["status"] == "ok" and len(f["filings"]) == 3 and f["latest"]["annual"]["form"] == "10-K"
    assert not (settings.STORE_DIR / "history.db").exists()


def test_routes_validate_input(client):
    assert client.get("/api/financials/TEST?period=weekly").status_code == 400
    assert client.get("/api/financials/%24%24").status_code in (400, 404)
    assert client.get("/api/filings/TEST?limit=abc").status_code == 400
    assert client.get("/api/financials/ITX.MC").get_json()["status"] == "not_sec_filer"


def test_reinvestment_detail_flags_floored_capex_and_missing_acquisitions(sec):
    out = sr.financial_overview("TEST")
    item = next(i for i in out["checklist"] if i["key"] == "reinvestment_rate")
    assert "acquisitions not reported, counted as zero" in item["detail"]
    assert any(n.startswith("Reinvestment rate in the latest column") for n in out["notes"])
    rows = _rows(out)
    assert "rd_to_fcf" in rows and rows["rd_to_fcf"]["values"] == [None, None, None]  # no R&D tag in this filing: empty, not zero
    assert "not a reported zero" in rows["acquisitions"]["tip"]


def test_reinvestment_detail_text_for_a_floored_year():
    text = sr._reinvestment_detail({"capex": 9.4e9, "depreciation_amortization": 11.4e9, "acquisitions": None,
                                    "_acq_reported": 0.0, "reinvestment_rate": 0.0, "rd_to_fcf": 0.31})
    assert text.startswith("0%: CapEx $9.4B is below D&A $11.4B, so the proxy counts no growth CapEx; acquisitions not reported")
    assert "R&D is 31% of free cash flow" in text


def _filed_before(raw: dict, day: str) -> dict:
    """The facts as they stood before ``day``: a company whose latest filing is its 10-K."""
    out = {**raw, "facts": {}}
    for taxonomy, concepts in raw["facts"].items():
        out["facts"][taxonomy] = {c: {"units": {u: [e for e in entries if e["filed"] < day] for u, entries in body["units"].items()}}
                                  for c, body in concepts.items()}
    return out


def test_checklist_prices_the_newest_year_at_the_latest_close(sec):
    """Right after a 10-K there is no LTM column; the checklist must not keep the fiscal-year-end price."""
    sec["raw"] = _filed_before(sec["raw"], "2025-01-01")
    sr._model.cache_clear()
    out = sr.financial_overview("TEST")
    assert [c["key"] for c in out["columns"]][-1] == "FY2024"
    basis = out["checklist_basis"]
    close = sec["prices"]["close"]
    assert basis["kind"] == "fy" and basis["end"] == "2024-09-30"
    assert basis["price_date"] == "2025-06-30" and basis["price"] == pytest.approx(close.iloc[-1])
    items = {c["key"]: c for c in out["checklist"]}
    fy_pe = _rows(out)["pe"]["values"][-1]  # the table keeps the close on the period end
    assert items["pe"]["value"] == pytest.approx(fy_pe * close.iloc[-1] / close.loc[:"2024-09-30"].iloc[-1])


def test_checklist_basis_names_the_ltm_period(sec):
    basis = sr.financial_overview("TEST")["checklist_basis"]
    assert basis["kind"] == "ltm" and basis["end"] == "2025-03-31" and "last twelve months" in basis["label"]
