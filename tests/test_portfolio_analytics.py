"""Portfolio analytics: one-currency calendar, weekly risk, risk shares, period P&L, currency split, exposures, fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.master_agent import MasterDecisionAgent
from core import portfolio as pf
from core import portfolio_analytics as pa
from data.profiles import build_profile, fund_region
from tests.test_owner_first import _cfg, _fetched, _OwnerByTicker
from tests.test_multi_agent import StubAgent

DAYS = pd.bdate_range("2023-01-02", "2026-09-30")


def _unit_fx(_ccy: str, dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(1.0, index=dates)


def _walk(seed: int, beta: float = 1.0, idio: float = 0.01, market: np.ndarray | None = None) -> pd.Series:
    rng = np.random.default_rng(seed)
    m = market if market is not None else np.zeros(len(DAYS))
    r = beta * m + rng.normal(0.0, idio, len(DAYS))
    return pd.Series(100.0 * np.cumprod(1.0 + r), index=DAYS)


def _row(ticker: str, close: pd.Series, value: float, total: float, **extra) -> dict:
    price = float(close.iloc[-1])
    shares = value / price
    return {"ticker": ticker, "value": value, "weight": value / total, "shares": shares, "avg_cost": price,
            "price_currency": "USD", "fx": 1.0, "opened": None, "pnl": 0.0, "pnl_pct": 0.0, "name": ticker, **extra}


# ------------------------------------------------------------------ prices and correlation


def test_price_frame_converts_currency_and_aligns_calendars():
    us = pd.Series([10.0, 11.0, 12.0], index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-08"]))
    eu = pd.Series([20.0, 21.0], index=pd.to_datetime(["2026-01-05", "2026-01-07"]))
    fx = lambda ccy, dates: pd.Series(2.0 if ccy == "USD" else 1.0, index=dates)  # noqa: E731
    frame, notes = pa.price_frame({"US": us, "EU": eu}, {"US": "USD", "EU": "EUR"}, "EUR", fx)
    assert notes == []
    assert list(frame.index) == list(pd.bdate_range("2026-01-05", "2026-01-08"))
    assert frame.loc["2026-01-07", "US"] == 22.0  # carried from the 6th and converted
    assert frame.loc["2026-01-06", "EU"] == 20.0 and frame.loc["2026-01-08", "EU"] == 21.0


def test_weekly_returns_see_markets_that_react_a_day_apart():
    """Why risk uses weekly returns: a market that prices the same news one day later looks unrelated day by day."""
    rng = np.random.default_rng(3)
    shocks = rng.normal(0.0, 0.01, len(DAYS))
    a = pd.Series(100.0 * np.cumprod(1.0 + shocks), index=DAYS)
    lagged = np.concatenate([[0.0], shocks[:-1]])
    b = pd.Series(100.0 * np.cumprod(1.0 + lagged), index=DAYS)
    daily = pd.concat([a, b], axis=1).pct_change().corr().iloc[0, 1]
    weekly = pa._weekly_returns(pd.concat([a.rename("A"), b.rename("B")], axis=1)).corr().iloc[0, 1]
    assert abs(daily) < 0.15
    assert weekly > 0.6


# ------------------------------------------------------------------ risk


def test_risk_shares_add_up_and_follow_volatility_and_correlation():
    market = np.random.default_rng(1).normal(0.0003, 0.01, len(DAYS))
    closes = {"HOT": _walk(2, 1.5, 0.02, market), "CORE": _walk(3, 1.0, 0.004, market), "CALM": _walk(4, 0.0, 0.006, market)}
    bench = pd.Series(100.0 * np.cumprod(1.0 + market), index=DAYS)
    total = 100_000.0
    rows = [_row("HOT", closes["HOT"], 30_000, total), _row("CORE", closes["CORE"], 40_000, total),
            _row("CALM", closes["CALM"], 20_000, total)]
    out = pa.analyze(rows, closes, 10_000.0, "USD", {}, _unit_fx,
                     benchmark={"ticker": "B", "label": "Bench", "close": bench, "currency": "USD"})
    risk = out["risk"]
    shares = {t: p["risk_share"] for t, p in risk["positions"].items()}
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)
    assert shares["HOT"] > 0.30 > shares["CALM"]  # 30% of value, but the most volatile and correlated
    assert risk["positions"]["CORE"]["beta"] == pytest.approx(1.0, abs=0.15)
    assert abs(risk["positions"]["CALM"]["beta"]) < 0.2
    assert risk["diversification_ratio"] > 1.0
    assert 0 < risk["volatility"] < risk["positions"]["HOT"]["volatility"]
    assert risk["bad_month"]["value"] < 0 and risk["bad_month"]["amount"] < 0
    corr = risk["correlation"]
    assert set(corr["tickers"]) == {"HOT", "CORE", "CALM"} and len(corr["matrix"]) == 3
    assert corr["max"]["pair"] in (["HOT", "CORE"], ["CORE", "HOT"])
    assert {p["key"] for p in out["performance"]["periods"]} >= {"1M", "1Y", "3Y"}
    assert out["performance"]["series"]["mix"][0]["value"] > 0 and "benchmark" in out["performance"]["series"]


def test_short_history_is_left_out_of_risk_but_kept_in_weights():
    closes = {"OLD": _walk(5), "NEW": _walk(6).iloc[-60:]}
    total = 50_000.0
    rows = [_row("OLD", closes["OLD"], 25_000, total), _row("NEW", closes["NEW"], 25_000, total)]
    out = pa.analyze(rows, closes, 0.0, "USD", {}, _unit_fx)
    assert out["risk"]["excluded"] == ["NEW"]
    assert set(out["risk"]["positions"]) == {"OLD"}
    assert any(a["title"] == "Short price history" for a in out["alerts"])
    assert sum(g["weight"] for g in out["exposures"]["position"]) == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------------------ winners, losers and currency


def test_period_pnl_does_not_count_savings_plan_money_as_gain():
    flat = pd.Series(50.0, index=DAYS)
    as_of = DAYS[-1]
    start = pa.period_start(as_of, 3)
    plan_day = (start + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    hist = {"trades": [("2024-01-02", 100.0, 5_000.0), (plan_day, 100.0, 5_000.0)],
            "dividends": [((start + pd.Timedelta(days=30)).strftime("%Y-%m-%d"), 40.0)], "lots": []}
    row = {"ticker": "X", "shares": 200.0, "avg_cost": 50.0, "value": 10_000.0, "opened": "2024-01-02"}
    as_traded = pd.Series(1.0, index=DAYS)  # these flat prices are the ones that traded: no dividend adjustment
    res = pa.period_pnl(row, flat, hist, start, as_of, as_traded)
    assert res["start_value"] == 5_000.0 and res["net_invested"] == 5_000.0
    assert res["pnl"] == pytest.approx(40.0)  # only the dividend: buying more shares is not a gain
    assert res["return_mwr"] == pytest.approx(40.0 / (5_000.0 + 5_000.0 * ((as_of - pd.Timestamp(plan_day)).days / (as_of - start).days)), abs=1e-4)
    assert res["return"] == pytest.approx(0.004)  # 0.20 a share on a flat price of 50, whatever came in when


def test_period_pnl_counts_a_dividend_once_on_total_return_prices():
    # 100 shares, traded flat at 100 all year, one 3.00 dividend: the P&L is the 300 received, not 600.
    as_of = DAYS[-1]
    start = pa.period_start(as_of, 12)
    ex_day = start + pd.Timedelta(days=90)
    adjusted = pd.Series(np.where(DAYS < ex_day, 97.0, 100.0), index=DAYS)  # Yahoo lowers closes before ex-date
    factor = pd.Series(np.where(DAYS < ex_day, 100.0 / 97.0, 1.0), index=DAYS)  # close / adj close
    hist = {"trades": [("2020-01-02", 100.0, 10_000.0)], "dividends": [(ex_day.strftime("%Y-%m-%d"), 300.0)], "lots": []}
    row = {"ticker": "D", "shares": 100.0, "avg_cost": 100.0, "value": 10_000.0, "opened": "2020-01-02"}
    exact = pa.period_pnl(row, adjusted, hist, start, as_of, factor)
    assert exact["pnl"] == pytest.approx(300.0) and exact["start_value"] == pytest.approx(10_000.0)
    assert exact["dividends"] == 300.0 and not exact["dividends_in_price"]
    # Without the factor the dividend stays in the price: still counted once.
    fallback = pa.period_pnl(row, adjusted, hist, start, as_of)
    assert fallback["pnl"] == pytest.approx(300.0) and fallback["dividends_in_price"]


def test_period_pnl_of_a_position_opened_by_hand_inside_the_period():
    prices = pd.Series(np.linspace(40.0, 60.0, len(DAYS)), index=DAYS)
    as_of = DAYS[-1]
    start = pa.period_start(as_of, 12)
    opened = (as_of - pd.Timedelta(days=100)).strftime("%Y-%m-%d")
    row = {"ticker": "Y", "shares": 10.0, "avg_cost": 55.0, "value": 600.0, "opened": opened}
    res = pa.period_pnl(row, prices, None, start, as_of)
    assert res["basis"] == "opened" and res["start_value"] == 0.0
    assert res["pnl"] == pytest.approx(600.0 - 550.0)
    held = pa.period_pnl({**row, "opened": None}, prices, None, start, as_of)
    assert held["basis"] == "held" and held["pnl"] == pytest.approx(600.0 - 10.0 * float(prices.loc[:start].iloc[-1]), abs=0.01)


def test_currency_split_adds_up_to_the_pnl():
    # Bought 10 shares at $100 when $1 = 1.00 EUR; now $110 and $1 = 0.90 EUR.
    fx = lambda _ccy, dates: pd.Series(1.0, index=dates)  # noqa: E731 - the rate on the buy date
    row = {"ticker": "Z", "shares": 10.0, "avg_cost": 100.0, "value": 10 * 110.0 * 0.9, "fx": 0.9,
           "price_currency": "USD", "opened": "2024-03-01", "pnl": 10 * 110.0 * 0.9 - 1_000.0}
    split = pa.currency_split(row, None, "EUR", fx)
    assert split["price_effect"] == pytest.approx(10 * 10.0 * 0.9)
    assert split["fx_effect"] == pytest.approx(1_000.0 * 0.9 - 1_000.0)
    assert split["price_effect"] + split["fx_effect"] == pytest.approx(row["pnl"])
    assert split["local_return"] == pytest.approx(0.10) and split["currency_move"] == pytest.approx(-0.10)
    assert pa.currency_split({**row, "price_currency": "EUR"}, None, "EUR", fx) is None


# ------------------------------------------------------------------ exposures and alerts


def test_exposures_look_through_funds_and_compare_with_the_benchmark():
    closes = {"CHIP": _walk(7), "WORLD": _walk(8), "GOLD": _walk(9)}
    total = 100_000.0
    rows = [_row("CHIP", closes["CHIP"], 40_000, total), _row("WORLD", closes["WORLD"], 40_000, total),
            _row("GOLD", closes["GOLD"], 10_000, total)]
    profiles = {
        "CHIP": build_profile("CHIP", {"quoteType": "EQUITY", "longName": "Chip Co", "sector": "Technology", "country": "Taiwan"}),
        "WORLD": build_profile("WORLD", {"quoteType": "ETF", "longName": "Core MSCI World UCITS ETF"},
                               {"technology": 0.25, "healthcare": 0.25, "financial_services": 0.5}),
        "GOLD": build_profile("GOLD", {"quoteType": "EQUITY", "longName": "Xetra-Gold"}),
    }
    out = pa.analyze(rows, closes, 10_000.0, "USD", profiles, _unit_fx,
                     benchmark={"ticker": "B", "label": "Bench", "close": _walk(10), "currency": "USD",
                                "sectors": {"Technology": 0.3, "Healthcare": 0.1, "Financial Services": 0.2, "Industrials": 0.4}})
    ex = out["exposures"]
    sector = {g["label"]: g for g in ex["sector"]}
    assert sector["Technology"]["weight"] == pytest.approx(0.50)  # 40% directly + 25% of the 40% fund
    assert sector["Commodities"]["weight"] == pytest.approx(0.10) and sector["Cash"]["weight"] == pytest.approx(0.10)
    assert sector["Technology"]["share_of_equity"] == pytest.approx(0.50 / 0.80)
    assert sector["Industrials"]["active"] == pytest.approx(-0.4)
    assert {g["label"] for g in ex["asset_type"]} == {"Stock", "ETF", "Commodity", "Cash"}
    region = {g["label"]: g["weight"] for g in ex["region"]}
    assert region["Emerging markets"] == pytest.approx(0.4) and region["Global (funds)"] == pytest.approx(0.4)
    for dim in pa.DIMENSIONS:
        assert sum(g["weight"] for g in ex[dim]) == pytest.approx(1.0, abs=1e-3), dim
    assert any(a["title"] == "Technology overweight" for a in out["alerts"])


def test_currency_alert_and_fund_pairs():
    market = np.random.default_rng(11).normal(0.0, 0.01, len(DAYS))
    closes = {"AAA": _walk(12, 1.0, 0.002, market), "IDX": _walk(13, 1.0, 0.001, market)}
    total = 100_000.0
    rows = [_row("AAA", closes["AAA"], 45_000, total, price_currency="USD"),
            _row("IDX", closes["IDX"], 45_000, total, price_currency="USD")]
    profiles = {"AAA": {"asset_type": "Stock"}, "IDX": {"asset_type": "ETF"}}
    out = pa.analyze(rows, closes, 10_000.0, "EUR", profiles, _unit_fx)
    titles = [a["title"] for a in out["alerts"]]
    assert "Currency exposure" in titles
    assert not any("move as one" in t for t in titles)  # a broad fund tracking a stock is not a doubled bet
    out = pa.analyze(rows, closes, 10_000.0, "EUR", {"AAA": {"asset_type": "Stock"}, "IDX": {"asset_type": "Stock"}}, _unit_fx)
    assert any("move as one" in a["title"] for a in out["alerts"])


def test_fund_region_and_asset_type_from_the_name():
    assert fund_region("iShares Core S&P 500 UCITS ETF") == ("North America", "United States")
    assert fund_region("Vanguard FTSE Emerging Markets")[0] == "Emerging markets"
    assert fund_region("Vanguard FTSE All-World")[0] == pa_global()
    assert build_profile("K", {"quoteType": "EQUITY", "longName": "Kinross Gold Corp"})["asset_type"] == "Stock"
    assert build_profile("G", {"quoteType": "ETF", "longName": "iShares Physical Gold ETC"})["asset_type"] == "Commodity"
    assert build_profile("T", {"quoteType": "ETF", "longName": "iShares USD Treasury Bond 7-10yr"})["asset_type"] == "Bond"
    assert build_profile("A", None, name="Apple", isin="US0378331005", broker_class="STOCK")["country"] == "United States"


def pa_global() -> str:
    from data.profiles import GLOBAL_FUND

    return GLOBAL_FUND


# ------------------------------------------------------------------ review and candidate


def test_review_keeps_unpriced_positions_at_cost_and_returns_analytics():
    master = MasterDecisionAgent(config=_cfg(owner={"owner": 1.0}, timing={"mom": 1.0}),
                                 agents=[_OwnerByTicker({"GOOD": 40.0}), StubAgent("mom", 0)])
    fetched = _fetched({"GOOD": (100, 120)})
    portfolio = {"cash": 1_000.0, "currency": "USD", "positions": [
        {"ticker": "GOOD", "shares": 100, "avg_cost": 100.0},
        {"ticker": "GONE", "shares": 50, "avg_cost": 20.0},
    ]}
    result = pf.review_portfolio(portfolio, master=master, record=False, fetched=fetched,
                                 profiles={"GOOD": {"asset_type": "Stock", "sector": "Technology"}}, histories={})
    s = result["summary"]
    good = result["positions"][0]
    assert result["unpriced"][0]["ticker"] == "GONE" and result["unpriced"][0]["value"] == 1_000.0
    assert s["total_value"] == pytest.approx(good["value"] + 1_000.0 + 1_000.0, rel=1e-6)
    assert good["weight"] == pytest.approx(good["value"] / s["total_value"], abs=1e-4)
    price = next(g for g in result["todo"] if g["key"] == "price")  # counted at cost, listed with its fix instead of an alert
    assert price["items"][0]["ticker"] == "GONE" and "counted at cost" in price["items"][0]["why"] and price["items"][0]["fix"]
    assert not any(a["title"] == "Counted at cost" for a in result["alerts"])
    assert all(a.get("fix") for a in result["alerts"])
    assert result["analytics"]["exposures"]["position"]
    assert good["risk"]["risk_share"] == pytest.approx(1.0, abs=1e-6)
    assert good["news"]["available"] is False and set(good["revisions"]) == {"d30", "d90"}


def test_candidate_fit_shows_overlap_and_risk_after_adding():
    market = np.random.default_rng(21).normal(0.0, 0.01, len(DAYS))
    closes = {"HELD": _walk(22, 1.0, 0.002, market), "TWIN": _walk(23, 1.0, 0.002, market), "OTHER": _walk(24, 0.0, 0.01)}
    holdings = [{"ticker": "HELD", "value": 50_000.0, "price_currency": "USD"}, {"ticker": "OTHER", "value": 30_000.0, "price_currency": "USD"}]
    profiles = {"HELD": {"sector": "Technology", "country": "United States"}, "OTHER": {"sector": "Utilities"},
                "TWIN": {"sector": "Technology", "country": "United States", "asset_type": "Stock"}}
    fit = pa.candidate_fit("TWIN", holdings, closes, 20_000.0, "USD", profiles, _unit_fx, 10_000.0, "USD")
    assert fit["most_similar"][0]["ticker"] == "HELD" and fit["most_similar"][0]["correlation"] > 0.9
    assert fit["correlation_to_portfolio"] > 0.5
    assert fit["volatility_after"] > fit["volatility_before"]
    assert fit["overlap"]["sector"] == pytest.approx(0.5) and fit["overlap"]["currency"] == pytest.approx(0.8)


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_api_portfolio_performance_starts_a_job(client, monkeypatch):
    from web import app as web_app

    monkeypatch.setattr(web_app.lab, "start_job", lambda kind, fn, **kw: f"job-{kind}")
    res = client.post("/api/portfolio/performance", json={})
    assert res.status_code == 200 and res.get_json()["job"] == "job-account_history"


# ------------------------------------------------------------------ currencies (data/fx.py)


def test_quote_currency_reads_pairs_and_does_not_store_offline_guesses():
    from data import fx

    assert fx.quote_currency("BTC-EUR") == "EUR" and fx.quote_currency("ETH-GBP") == "GBP"
    assert fx.quote_currency("BRK-B") == "USD" and fx.quote_currency("AAPL") == "USD"
    # Yahoo is unreachable in tests: the suffix rule answers, but nothing is written to the disk cache.
    assert fx.quote_currency("IWDA.L") == "GBp"
    assert "IWDA.L" not in fx._read_currency_cache()
    fx.set_quote_currency("IWDA.L", "USD")  # a real answer later wins
    assert fx.quote_currency("IWDA.L") == "USD"


def test_factor_series_accepts_two_lots_on_the_same_day(monkeypatch):
    from data import fx

    days = pd.bdate_range("2024-01-01", periods=30)
    monkeypatch.setattr(fx, "per_eur", lambda code: pd.Series(1.10, index=days))
    dates = pd.DatetimeIndex(["2024-01-10", "2024-01-10", "2024-01-15"])
    out = fx.factor_series("USD", "EUR", dates)
    assert len(out) == 3 and out.round(4).tolist() == [round(1 / 1.10, 4)] * 3


def test_expired_profile_is_used_when_yahoo_is_down(tmp_path, monkeypatch):
    import json as _json

    from data import profiles

    path = tmp_path / "profiles.json"
    old = {"info": {"quoteType": "EQUITY", "sector": "Technology", "country": "Germany", "longName": "Example AG"},
           "fund_sectors": None, "bond_share": None, "_at": 0}  # 1970: long expired
    path.write_text(_json.dumps({"EXM.DE": old}))
    monkeypatch.setattr(profiles, "_path", lambda: path)
    monkeypatch.setattr(profiles, "_fetch", lambda t: (_ for _ in ()).throw(RuntimeError("offline")))
    p = profiles.get_profiles([{"ticker": "EXM.DE", "name": "Example AG"}])["EXM.DE"]
    assert p["sector"] == "Technology" and p["country"] == "Germany"


def test_liquidity_amounts_are_in_dollars(monkeypatch):
    from data import fx

    fx.set_quote_currency("THIN.L", "GBp")
    offline = lambda code: (_ for _ in ()).throw(OSError("offline"))  # noqa: E731 - fail fast, no retry backoff
    monkeypatch.setattr(fx, "_fetch_ecb", offline)
    monkeypatch.setattr(fx, "_fetch_yahoo", offline)
    # No rates offline: pence still become pounds (the 100x part of the error).
    assert fx.to_usd(3_000_000_00.0, "THIN.L") == pytest.approx(3_000_000.0)
    # With rates: pence -> pounds -> dollars (1 EUR = 0.85 GBP = 1.10 USD).
    rates = {"GBP": 0.85, "USD": 1.10}
    monkeypatch.setattr(fx, "per_eur", lambda code: pd.Series([rates[code]], index=[pd.Timestamp("2026-10-01")]))
    assert fx.to_usd(3_000_000_00.0, "THIN.L") == pytest.approx(3_000_000.0 / 0.85 * 1.10)
    assert fx.to_usd(5e6, "AAPL") == 5e6 and fx.to_usd(None, "AAPL") is None


def test_period_return_is_time_weighted_for_money_added_late():
    # Bought in the last quarter and down 50%: the loss is 50%, not the huge figure a money-weighted
    # denominator of "a few days of exposure" gives (below -100%).
    as_of = DAYS[-1]
    start = pa.period_start(as_of, 12)
    buy_day = as_of - pd.Timedelta(days=60)
    prices = pd.Series(np.where(DAYS < buy_day, 100.0, 100.0 - 50.0 * np.clip((DAYS - buy_day).days / 60.0, 0, 1)), index=DAYS)
    row = {"ticker": "L", "shares": 100.0, "avg_cost": 100.0, "value": 5_000.0, "opened": buy_day.strftime("%Y-%m-%d")}
    res = pa.period_pnl(row, prices, None, start, as_of)
    assert res["pnl"] == pytest.approx(-5_000.0, abs=1.0)
    assert res["return"] == pytest.approx(-0.5, abs=0.02)
    assert res["return_mwr"] < -1.0  # the money-weighted figure is kept, and shows why it was replaced
