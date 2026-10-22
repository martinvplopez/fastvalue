"""Tests for point-in-time SEC fundamentals, as-traded prices and the value study helpers (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.fundamental_agent import FundamentalAgent
from data.fundamentals import FundamentalsData
from data.price_history import as_traded, split_events
from data.sec_edgar import CompanyFacts, SecConfigError, _total_debt, extract_facts, user_agent


def _fy(val, fy_end, filed, form="10-K", days=364, accn=None):
    end = pd.Timestamp(fy_end)
    return {"start": (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), "end": fy_end, "val": val,
            "filed": filed, "form": form, "accn": accn or f"acc-{filed}"}


def _inst(val, end, filed, form="10-K", accn=None):
    return {"end": end, "val": val, "filed": filed, "form": form, "accn": accn or f"acc-{filed}"}


def _raw(extra: dict | None = None, dei: list | None = None) -> dict:
    """Two 10-Ks: FY2019 (filed 2020-02-14) and FY2020 (filed 2021-02-12, restating FY2019 net income)."""
    k19, k20 = "2020-02-14", "2021-02-12"
    usd = {
        "NetIncomeLoss": [_fy(100, "2019-12-31", k19), _fy(90, "2019-12-31", k20), _fy(120, "2020-12-31", k20),
                          _fy(30, "2020-03-31", "2020-05-01", form="10-Q", days=90)],
        "Revenues": [_fy(1000, "2019-12-31", k19), _fy(1000, "2019-12-31", k20), _fy(1100, "2020-12-31", k20)],
        "OperatingIncomeLoss": [_fy(150, "2019-12-31", k19), _fy(170, "2020-12-31", k20)],
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            [_fy(130, "2019-12-31", k19), _fy(150, "2020-12-31", k20)],
        "IncomeTaxExpenseBenefit": [_fy(30, "2019-12-31", k19), _fy(30, "2020-12-31", k20)],
        "InterestExpense": [_fy(10, "2019-12-31", k19), _fy(10, "2020-12-31", k20)],
        "DepreciationDepletionAndAmortization": [_fy(50, "2019-12-31", k19), _fy(55, "2020-12-31", k20)],
        "PaymentsToAcquirePropertyPlantAndEquipment": [_fy(60, "2019-12-31", k19), _fy(70, "2020-12-31", k20)],
        "NetCashProvidedByUsedInOperatingActivities": [_fy(200, "2019-12-31", k19), _fy(220, "2020-12-31", k20)],
        "Assets": [_inst(2000, "2019-12-31", k19), _inst(2200, "2020-12-31", k20)],
        "StockholdersEquity": [_inst(800, "2019-12-31", k19), _inst(900, "2020-12-31", k20)],
        "LongTermDebt": [_inst(300, "2019-12-31", k19), _inst(280, "2020-12-31", k20)],
        "CashAndCashEquivalentsAtCarryingValue": [_inst(100, "2019-12-31", k19), _inst(150, "2020-12-31", k20)],
        "AccountsReceivableNetCurrent": [_inst(120, "2019-12-31", k19), _inst(130, "2020-12-31", k20)],
        "AccountsPayableCurrent": [_inst(90, "2019-12-31", k19), _inst(95, "2020-12-31", k20)],
    }
    for k, v in (extra or {}).items():
        usd[k] = v
    shares = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": [_fy(100e6, "2019-12-31", k19)],
    }
    facts = {"us-gaap": {k: {"units": {"USD": v}} for k, v in usd.items()}}
    for k, v in shares.items():
        facts["us-gaap"][k] = {"units": {"shares": v}}
    if dei:
        facts["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": dei}}}
    return {"cik": 1, "entityName": "Test Co", "facts": facts}


def _company(raw: dict, splits: pd.Series | None = None, sic: int = 3571) -> CompanyFacts:
    return CompanyFacts(ticker="TEST", cik=1, name="Test Co", sic=sic, sic_description="Electronic computers",
                        facts=extract_facts(raw), splits=splits if splits is not None else pd.Series(dtype=float))


def test_statements_use_only_filings_known_on_the_date():
    cf = _company(_raw())
    assert not cf.statements_asof(pd.Timestamp("2020-01-31")).available  # first 10-K not filed yet

    mid_2020 = cf.statements_asof(pd.Timestamp("2020-06-30"))
    assert [y.strftime("%Y") for y in mid_2020.years()] == ["2019"]
    assert mid_2020.value("income", "Net Income") == 100  # original figure; the restatement comes later
    assert mid_2020.source.startswith("SEC EDGAR")

    after = cf.statements_asof(pd.Timestamp("2021-03-01"))
    assert [y.strftime("%Y") for y in after.years()] == ["2020", "2019"]
    assert after.value("income", "Net Income") == 120
    assert after.value("income", "Net Income", pd.Timestamp("2019-12-31")) == 90  # restated in the FY2020 10-K


def test_row_mapping_signs_and_derived_lines():
    st = _company(_raw()).statements_asof(pd.Timestamp("2021-03-01"))
    assert st.value("income", "EBIT") == 160  # pretax 150 + interest 10 (Yahoo's definition)
    assert st.value("income", "EBITDA") == 215  # EBIT + D&A
    assert st.value("cashflow", "Capital Expenditure") == -70  # outflows negative, as Yahoo reports them
    assert st.value("balance", "Total Debt") == 280
    assert st.value("balance", "Cash Cash Equivalents And Short Term Investments") == 150
    assert not st.has_row("balance", "Inventory")  # never reported: the metrics treat it as zero


def test_depreciation_fallback_and_zero_debt():
    raw = _raw()
    raw["facts"]["us-gaap"].pop("DepreciationDepletionAndAmortization")
    raw["facts"]["us-gaap"].pop("LongTermDebt")
    raw["facts"]["us-gaap"]["Depreciation"] = {"units": {"USD": [_fy(40, "2020-12-31", "2021-02-12")]}}
    raw["facts"]["us-gaap"]["AmortizationOfIntangibleAssets"] = {"units": {"USD": [_fy(5, "2020-12-31", "2021-02-12")]}}
    st = _company(raw).statements_asof(pd.Timestamp("2021-03-01"))
    assert st.value("cashflow", "Depreciation And Amortization") == 45
    assert st.value("balance", "Total Debt") == 0.0  # balance sheet present, no debt line


def test_total_debt_combines_tags_without_double_counting():
    none = dict.fromkeys(("long_term_total", "long_term_noncurrent", "long_term_current", "debt_current", "short_term",
                          "operating_lease", "operating_lease_current", "operating_lease_noncurrent", "finance_lease",
                          "finance_lease_current", "finance_lease_noncurrent", "notes"))
    assert _total_debt(none) is None
    assert _total_debt({**none, "long_term_total": 100.0, "short_term": 20.0, "operating_lease": 30.0}) == 150.0
    # DebtCurrent already contains short-term borrowings.
    assert _total_debt({**none, "long_term_noncurrent": 100.0, "debt_current": 25.0, "short_term": 20.0}) == 125.0
    assert _total_debt({**none, "long_term_noncurrent": 100.0, "long_term_current": 5.0, "short_term": 20.0,
                        "operating_lease_current": 3.0, "operating_lease_noncurrent": 7.0}) == 135.0


def test_share_counts_are_restated_for_splits():
    splits = pd.Series([4.0], index=[pd.Timestamp("2020-08-31")])
    raw = _raw()
    diluted = raw["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"]["units"]["shares"]
    diluted.append(_fy(400e6, "2020-12-31", "2021-02-12"))  # post-split basis
    cf = _company(raw, splits)
    st = cf.statements_asof(pd.Timestamp("2021-03-01"))
    # FY2019 was only ever filed before the split: restated to the post-split basis.
    assert st.value("income", "Diluted Average Shares", pd.Timestamp("2019-12-31")) == pytest.approx(400e6)
    assert st.value("income", "Diluted Average Shares", pd.Timestamp("2020-12-31")) == pytest.approx(400e6)
    # Before the split the same figure stays on the old basis.
    assert cf.statements_asof(pd.Timestamp("2020-06-30")).value("income", "Diluted Average Shares") == pytest.approx(100e6)
    # A split after the latest filing is applied too, so counts match the price of the date.
    later = CompanyFacts(ticker="TEST", cik=1, name=None, sic=None, sic_description=None, facts=extract_facts(raw),
                         splits=pd.concat([splits, pd.Series([2.0], index=[pd.Timestamp("2021-06-01")])]))
    assert later.statements_asof(pd.Timestamp("2021-07-01")).value("income", "Diluted Average Shares") == pytest.approx(800e6)
    assert list(later.statements_asof(pd.Timestamp("2021-03-01")).splits.index) == [pd.Timestamp("2020-08-31")]


def test_cover_page_shares_sum_classes_and_ignore_stale_counts():
    dei = [
        _inst(300e6, "2021-04-20", "2021-04-28", form="10-Q", accn="q1"),
        _inst(50e6, "2021-04-20", "2021-04-28", form="10-Q", accn="q1"),  # second share class
        _inst(290e6, "2021-01-25", "2021-02-12", accn="k20"),
    ]
    cf = _company(_raw(dei=dei))
    assert cf.shares_outstanding_asof(pd.Timestamp("2021-03-31")) == (290e6, pd.Timestamp("2021-01-25"))
    assert cf.shares_outstanding_asof(pd.Timestamp("2021-05-31")) == (350e6, pd.Timestamp("2021-04-20"))
    assert cf.shares_outstanding_asof(pd.Timestamp("2023-06-30")) is None  # more than 15 months old


def test_fundamental_agent_scores_sec_statements():
    cf = _company(_raw(dei=[_inst(100e6, "2021-01-25", "2021-02-12")]))
    st = cf.statements_asof(pd.Timestamp("2021-03-01"))
    data = FundamentalsData(ticker="TEST", available=True, quote_type="EQUITY", currency="USD", financial_currency="USD",
                            shares_outstanding=100e6, treasury_10y=1.5)
    ohlcv = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1e6},
                         index=pd.bdate_range("2021-01-18", "2021-03-01"))
    result = FundamentalAgent().evaluate_with_data(data, st, ohlcv, asof=pd.Timestamp("2021-03-01"))
    assert result.data_quality != "UNAVAILABLE"
    assert result.raw_data["data_source_label"].startswith("SEC EDGAR")
    assert result.raw_data["ev_to_fcf"] is not None


def test_user_agent_requires_contact_email(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SecConfigError):
        user_agent()
    monkeypatch.setenv("SEC_USER_AGENT", "app contact@example.com")
    assert user_agent() == "app contact@example.com"


def test_as_traded_undoes_split_adjustment():
    idx = pd.bdate_range("2024-01-01", periods=4)
    # Yahoo-style: a 2-for-1 split on day 3, earlier prices halved and volume doubled.
    df = pd.DataFrame({"open": [50.0, 51, 52, 53], "high": [50.0, 51, 52, 53], "low": [50.0, 51, 52, 53],
                       "close": [50.0, 51, 52, 53], "adj_close": [50.0, 51, 52, 53],
                       "volume": [2000.0, 2000, 1000, 1000], "dividends": 0.0, "splits": [0, 0, 2.0, 0]}, index=idx)
    traded = as_traded(df)
    assert traded["close"].tolist() == [100.0, 102.0, 52.0, 53.0]
    assert traded["volume"].tolist() == [1000.0, 1000.0, 1000.0, 1000.0]
    assert np.allclose(traded["close"] * traded["volume"], df["close"] * df["volume"])  # dollar volume unchanged
    assert split_events(df).to_dict() == {idx[2]: 2.0}


def test_value_study_helpers():
    from core.value_study import factor_test, members_asof, newey_west_t, sector_for_sic

    membership = pd.Series({pd.Timestamp("2020-01-02"): frozenset({"A"}), pd.Timestamp("2020-06-01"): frozenset({"B"})})
    assert members_asof(membership, pd.Timestamp("2020-03-31")) == frozenset({"A"})
    assert members_asof(membership, pd.Timestamp("2019-12-31")) == frozenset()
    assert sector_for_sic(6021) == "Financials" and sector_for_sic(6798) == "Real estate" and sector_for_sic(7372) == "Technology"
    assert newey_west_t(np.array([0.1] * 5), 2) != newey_west_t(np.array([0.1] * 5), 2)  # too short: NaN

    # A factor that ranks next month's returns perfectly has an IC of 1.
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    tickers = [f"T{i}" for i in range(60)]
    rng = np.random.default_rng(0)
    fwd = pd.DataFrame(rng.normal(0, 0.05, (30, 60)), index=dates, columns=tickers)
    panel = pd.DataFrame([{"date": d, "ticker": t, "sector": "X", "f": fwd.loc[d, t]} for d in dates for t in tickers])
    res = factor_test(panel, fwd, fwd, "f", higher_is_better=True)
    assert res["ic_1m"] == pytest.approx(1.0) and res["q5_minus_q1_annual"] > 0


def test_insider_data_set_parsing_feeds_the_insider_agent():
    import io
    import zipfile

    from agents.insider_agent import InsiderAgent
    from data.insider_trading import analyze_insider_transactions
    from data.sec_insider import parse_quarter, to_yahoo_frame

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SUBMISSION.tsv", "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\n"
                                      "a1\t05-MAR-2024\t4\t42\na2\t06-MAR-2024\t4\t42\na3\t06-MAR-2024\t4\t99\n")
        zf.writestr("REPORTINGOWNER.tsv", "ACCESSION_NUMBER\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
                                          "a1\tDoe Jane\tOfficer\tChief Executive Officer\na2\tRoe Sam\tDirector\t\n"
                                          "a3\tOther Person\tDirector\t\n")
        zf.writestr("NONDERIV_TRANS.tsv", "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
                                          "a1\t01-MAR-2024\tP\t10000\t20.5\tA\na2\t04-MAR-2024\tP\t5000\t20\tA\n"
                                          "a3\t04-MAR-2024\tS\t100\t20\tD\n")
    rows = parse_quarter(buf.getvalue(), {42})
    assert len(rows) == 2 and set(rows["code"]) == {"P"}  # the other issuer's filing is dropped
    frame = to_yahoo_frame(rows)
    assert frame["Value"].tolist() == [205000.0, 100000.0]
    assert frame["Position"].tolist() == ["Chief Executive Officer", "Director"]

    ohlcv = pd.DataFrame({"open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0, "volume": 1e5},
                         index=pd.bdate_range("2023-06-01", "2024-03-29"))
    data = analyze_insider_transactions(frame, ohlcv, "TEST")
    assert data.distinct_buyers == 2 and data.cluster_buy_detected
    assert InsiderAgent().evaluate_with_data(data).score > 0


def test_ticker_map_is_parsed_once_while_the_file_is_unchanged(tmp_path, monkeypatch):
    import gzip
    import json
    import os

    import config.settings as settings
    import data.sec_edgar as se

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    path = se.sec_cache_dir() / "company_tickers.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({"0": {"ticker": "BRK.B", "cik_str": 1067983}, "1": {"ticker": "AAPL", "cik_str": 320193}}, fh)
    opened = []
    real_open = gzip.open
    monkeypatch.setattr(se.gzip, "open", lambda *a, **k: opened.append(a[0]) or real_open(*a, **k))
    first = se.load_ticker_map()
    first["AAPL"] = -1  # a caller changing its copy must not reach the next caller
    second = se.load_ticker_map()
    assert second == {"BRK-B": 1067983, "AAPL": 320193} and len(opened) == 1
    with real_open(path, "wt") as fh:
        json.dump({"0": {"ticker": "MSFT", "cik_str": 789019}}, fh)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert se.load_ticker_map() == {"MSFT": 789019} and len(opened) == 2


def test_one_companyfacts_download_feeds_every_extract(monkeypatch):
    import gzip
    import io
    import json

    import data.sec_edgar as se

    calls = []

    class Response(io.BytesIO):
        headers = {"Content-Encoding": "gzip"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return Response(gzip.compress(json.dumps({"entityName": "Test Co", "n": len(calls)}).encode()))

    monkeypatch.setattr(se, "MIN_REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr(se, "user_agent", lambda: "test agent")
    monkeypatch.setattr(se.urllib.request, "urlopen", fake_urlopen)
    url = se.FACTS_URL.format(cik=320193)
    payloads = [se._get_json(url) for _ in range(3)]
    assert len(calls) == 1 and payloads[0] == payloads[2] == {"entityName": "Test Co", "n": 1}
    se._get_json(se.SUBMISSIONS_URL.format(cik=320193))
    se._get_json(se.SUBMISSIONS_URL.format(cik=320193))
    assert len(calls) == 3  # other SEC files are fetched every time, as before
