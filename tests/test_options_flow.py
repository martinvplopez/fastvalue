"""Tests for the options chain analytics and the Options Flow Agent.

All chains are synthetic DataFrames with yfinance column names. No network:
provider functions are monkeypatched and conftest blocks yfinance.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

import data.options_flow as options_flow
from agents.base import LIVE, PARTIAL, UNAVAILABLE
from agents.options_agent import OptionsFlowAgent
from data.options_flow import (
    OptionChainSet,
    OptionsFlowData,
    analyze_option_chains,
    fetch_option_chains,
    fetch_options_flow,
    iv_history_path,
    load_iv_history,
    max_pain,
    select_expirations,
)
from tests.conftest import make_ohlcv

TODAY = date(2026, 9, 28)  # a Monday
STRIKES = np.arange(80.0, 120.01, 2.5)
DASHES = re.compile("[\u2013\u2014]")


def _exp(days: int) -> str:
    return (TODAY + timedelta(days=days)).isoformat()


def _side(
    strikes: Any = STRIKES,
    iv: Any = 0.30,
    *,
    bid: Any = 1.0,
    ask: Any = 1.2,
    last: Any = 1.1,
    volume: Any = 0.0,
    oi: Any = 0.0,
    traded: date = TODAY,
) -> pd.DataFrame:
    """One side of a chain. Values may be scalars, callables of strike, or {strike: value} dicts."""
    ks = [float(k) for k in strikes]

    def col(v: Any) -> list[float]:
        if callable(v):
            return [float(v(k)) for k in ks]
        if isinstance(v, dict):
            return [float(v.get(k, 0.0)) for k in ks]
        return [v] * len(ks)

    ts = pd.Timestamp(traded).tz_localize("UTC") + pd.Timedelta(hours=19)
    return pd.DataFrame(
        {
            "strike": ks,
            "bid": col(bid),
            "ask": col(ask),
            "lastPrice": col(last),
            "volume": col(volume),
            "openInterest": col(oi),
            "impliedVolatility": col(iv),
            "lastTradeDate": [ts] * len(ks),
        }
    )


def _healthy_chains(near_iv: float = 0.30, far_iv: float = 0.30, **quote: Any) -> dict[str, tuple]:
    """Three near expirations plus one far, with volume and OI on a few strikes."""
    vol_c = {100.0: 800.0, 105.0: 1200.0, 110.0: 400.0}
    vol_p = {95.0: 600.0, 100.0: 500.0, 90.0: 300.0}
    oi_c = {100.0: 2000.0, 105.0: 5000.0, 110.0: 3000.0}
    oi_p = {90.0: 4000.0, 95.0: 3000.0, 100.0: 2000.0}
    chains = {}
    for days in (12, 30, 45):
        chains[_exp(days)] = (
            _side(iv=near_iv, volume=vol_c, oi=oi_c, **quote),
            _side(iv=near_iv, volume=vol_p, oi=oi_p, **quote),
        )
    chains[_exp(120)] = (_side(iv=far_iv, oi=oi_c, **quote), _side(iv=far_iv, oi=oi_p, **quote))
    return chains


@pytest.fixture
def noisy_ohlcv() -> pd.DataFrame:
    return make_ohlcv(periods=300, start=100.0, end=100.0, noise=0.01)


# --------------------------------------------------------------------------- registration and failures


def test_registered_under_options_key():
    from agents.registry import get_agent_class

    assert get_agent_class("options") is OptionsFlowAgent
    meta = OptionsFlowAgent.metadata()
    assert meta["name"] == "Options Flow Agent"
    assert meta["category"] == "Flow"
    assert meta["score_scale"] == 45.0
    assert meta["default_weight"] == 0.15
    assert meta["point_in_time"] is False


def test_provider_failure_is_unavailable_without_fabricated_numbers(ohlcv):
    # conftest makes yfinance.Ticker raise, so the real fetch path fails.
    data = fetch_options_flow("TEST", ohlcv)
    assert data.available is False
    assert data.quality == UNAVAILABLE
    assert "failed" in (data.error or "")
    assert data.atm_iv is None and data.put_call_volume_ratio is None and data.iv_rank is None

    result = OptionsFlowAgent().evaluate("TEST", ohlcv)
    assert result.data_quality == UNAVAILABLE
    assert result.score == 0.0 and result.confidence == 0.0
    for key in ("put_call_volume_ratio", "iv_rank", "iv_hv_ratio", "atm_iv", "metrics_table"):
        assert key not in result.raw_data
    assert result.raw_data["links"][0]["url"] == "https://finance.yahoo.com/quote/TEST/options/"


def test_no_listed_options_is_unavailable(monkeypatch, ohlcv):
    raw = OptionChainSet(ticker="TEST", chains={}, expirations_listed=0, as_of=TODAY.isoformat())
    monkeypatch.setattr(options_flow, "fetch_option_chains", lambda _t: raw)
    result = OptionsFlowAgent().evaluate("TEST", ohlcv)
    assert result.data_quality == UNAVAILABLE
    assert "No listed options" in result.summary


def test_fetch_option_chains_bounds_requests(monkeypatch):
    listed = [(TODAY + timedelta(days=d)).isoformat() for d in (0, 1, 3, 5, 8, 10, 12, 17, 24, 31, 38, 52, 59, 80, 115, 150, 175, 300)]
    calls_made: list[str] = []

    class FakeChain:
        def __init__(self) -> None:
            self.calls = _side()
            self.puts = _side()
            self.underlying = {"regularMarketPrice": 101.5, "marketState": "REGULAR"}

    class FakeTicker:
        options = tuple(listed)

        def __init__(self, _symbol: str) -> None:
            pass

        def option_chain(self, exp: str) -> FakeChain:
            calls_made.append(exp)
            return FakeChain()

    monkeypatch.setattr(options_flow, "_market_today", lambda: TODAY)
    monkeypatch.setattr(options_flow.yf, "Ticker", FakeTicker)
    raw = fetch_option_chains("TEST")
    assert len(calls_made) <= 8
    dtes = sorted((date.fromisoformat(e) - TODAY).days for e in calls_made)
    assert all(d >= 7 for d in dtes)
    assert sum(7 <= d <= 60 for d in dtes) == 6
    assert sum(90 <= d <= 180 for d in dtes) == 1
    assert 31 in dtes  # closest to 30 DTE always included
    assert raw.spot == 101.5


def test_select_expirations_skips_weeklies_and_picks_far():
    listed = [_exp(d) for d in (0, 2, 4, 9, 30, 44, 70, 118, 160, 400)]
    near, far = select_expirations(listed, TODAY)
    assert near == [_exp(9), _exp(30), _exp(44)]
    assert far == _exp(118)


# --------------------------------------------------------------------------- volatility surface


def test_atm_iv_and_skew_on_constructed_smile(noisy_ohlcv):
    def smile(k: float) -> float:
        return 0.30 - 0.005 * (k - 100.0)  # linear: puts richer than calls

    spot = 101.0
    calls = _side(iv=smile, volume=10.0, oi=100.0)
    puts = _side(iv=smile, volume=10.0, oi=100.0)
    # Junk that contract hygiene must ignore:
    calls = pd.concat([calls, _side([101.0], iv=6.0)], ignore_index=True)  # IV out of range
    puts = pd.concat([puts, _side([101.5], iv=0.90, bid=0.0, ask=0.0)], ignore_index=True)  # no live quote
    puts = pd.concat([puts, _side([40.0], iv=3.5)], ignore_index=True)  # deep OTM wing
    chains = {
        _exp(12): (_side(iv=0.50), _side(iv=0.50)),  # further from 30 DTE, must not be the ATM expiry
        _exp(31): (calls, puts),
        _exp(120): (_side(iv=0.28), _side(iv=0.28)),
    }
    d = analyze_option_chains(chains, spot, noisy_ohlcv, "TEST", as_of=TODAY)

    assert d.available
    assert d.atm_expiry == _exp(31) and d.atm_dte == 31
    assert d.atm_iv == pytest.approx(0.295, abs=1e-6)  # interpolated between 100 and 102.5
    assert d.put_iv_otm == pytest.approx(0.32025, abs=1e-6)  # at 0.95 * 101 = 95.95
    assert d.call_iv_otm == pytest.approx(0.26975, abs=1e-6)  # at 1.05 * 101 = 106.05
    assert d.skew == pytest.approx(5.05, abs=1e-3)  # vol points
    assert d.skew_normalized == pytest.approx(0.0505 / 0.295, abs=1e-3)
    assert d.hv20 is not None and d.iv_hv_ratio == pytest.approx(d.atm_iv / d.hv20, rel=1e-2)


def test_term_structure_backwardation_and_contango(noisy_ohlcv):
    inverted = analyze_option_chains(_healthy_chains(near_iv=0.45, far_iv=0.30), 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    assert inverted.term_structure_ratio == pytest.approx(1.5, abs=1e-6)
    assert inverted.term_structure_state == "BACKWARDATION"
    assert inverted.far_expiry == _exp(120)

    result = OptionsFlowAgent().evaluate_with_data(inverted)
    term = next(b for b in result.raw_data["score_breakdown"] if b["factor"].startswith("IV term structure"))
    assert term["points"] == -8.0
    assert any("Inverted IV term structure" in f for f in result.key_factors)

    # Earnings before the ATM expiry: same inversion is scored as event premium, at half weight.
    event = analyze_option_chains(
        _healthy_chains(near_iv=0.45, far_iv=0.30), 100.0, noisy_ohlcv, "TEST", as_of=TODAY, earnings_date=_exp(20)
    )
    assert event.earnings_before_atm_expiry and event.iv_crush_warning
    term = next(
        b for b in OptionsFlowAgent().evaluate_with_data(event).raw_data["score_breakdown"]
        if b["factor"].startswith("IV term structure")
    )
    assert term["points"] == -4.0

    calm = analyze_option_chains(_healthy_chains(near_iv=0.20, far_iv=0.30), 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    assert calm.term_structure_state == "CONTANGO"


# --------------------------------------------------------------------------- positioning


def test_max_pain_on_hand_checkable_chain(noisy_ohlcv):
    # Calls OI {95: 1000, 100: 500, 105: 200}; puts OI {95: 200, 100: 300, 105: 1600}.
    # Payout to holders if price settles at:
    #    95: puts 300*5 + 1600*10              = 17,500
    #   100: calls 1000*5 + puts 1600*5        = 13,000
    #   105: calls 1000*10 + 500*5             = 12,500  <- minimum
    strikes = [95.0, 100.0, 105.0]
    calls = _side(strikes, oi={95.0: 1000, 100.0: 500, 105.0: 200})
    puts = _side(strikes, oi={95.0: 200, 100.0: 300, 105.0: 1600})
    normalized = lambda df: pd.DataFrame({"strike": df["strike"], "oi": df["openInterest"]})  # noqa: E731
    assert max_pain(normalized(calls), normalized(puts)) == 105.0

    # A smaller expiry with a different max pain must not be chosen.
    small_calls = _side(strikes, oi={95.0: 10, 100.0: 10, 105.0: 10})
    small_puts = _side(strikes, oi={95.0: 10, 100.0: 10, 105.0: 10})
    chains = {_exp(19): (small_calls, small_puts), _exp(33): (calls, puts)}
    d = analyze_option_chains(chains, 101.0, noisy_ohlcv, "TEST", as_of=TODAY)
    assert d.max_pain == 105.0
    assert d.max_pain_expiry == _exp(33)


def test_oi_walls_land_on_the_right_side_of_spot(noisy_ohlcv):
    strikes = [60.0, 90.0, 95.0, 97.5, 100.0, 105.0, 110.0, 150.0]
    calls = _side(strikes, oi={95.0: 50_000, 105.0: 3_000, 110.0: 8_000, 150.0: 90_000})
    puts = _side(strikes, oi={60.0: 80_000, 90.0: 7_000, 97.5: 2_000, 105.0: 60_000})
    d = analyze_option_chains({_exp(30): (calls, puts)}, 100.0, noisy_ohlcv, "TEST", as_of=TODAY)

    # Big call OI below spot, big put OI above spot and far-away hedges are ignored.
    assert d.call_wall == 110.0 and d.call_wall > d.spot and d.call_wall_oi == 8_000
    assert d.put_wall == 90.0 and d.put_wall < d.spot and d.put_wall_oi == 7_000

    levels = OptionsFlowAgent().evaluate_with_data(d).raw_data["levels"]
    by_kind = {lvl["kind"]: lvl for lvl in levels}
    assert by_kind["resistance"]["price"] == 110.0
    assert by_kind["support"]["price"] == 90.0
    assert by_kind["pivot"]["price"] == d.max_pain


def test_put_call_ratio_excludes_short_dated_expirations(noisy_ohlcv):
    chains = {
        _exp(2): (_side(volume={100.0: 1_000}, oi={100.0: 500}), _side(volume={100.0: 50_000}, oi={100.0: 90_000})),
        _exp(30): (_side(volume={105.0: 2_000}, oi={105.0: 4_000}), _side(volume={95.0: 1_000}, oi={95.0: 1_000})),
        _exp(120): (_side(), _side()),
    }
    d = analyze_option_chains(chains, 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    assert d.total_put_volume == 1_000 and d.total_call_volume == 2_000
    assert d.put_call_volume_ratio == pytest.approx(0.5)
    assert d.put_call_oi_ratio == pytest.approx(0.25)
    assert all(c["dte"] >= 7 for c in d.unusual_contracts)


def test_unusual_activity_filters_and_table(noisy_ohlcv):
    strikes = [75.0, *STRIKES]
    calls = _side(
        strikes,
        volume={105.0: 5_000, 110.0: 300, 75.0: 9_000},
        oi={105.0: 100, 110.0: 10, 75.0: 0},
        bid=2.0,
        ask=2.2,
    )
    puts = _side(strikes, volume={95.0: 1_500}, oi={95.0: 5_000})  # volume below 2x OI
    chains = {
        _exp(3): (_side(volume={100.0: 20_000}, oi={100.0: 10}), _side()),  # under 7 DTE: ignored
        _exp(30): (calls, puts),
        _exp(120): (_side(), _side()),
    }
    d = analyze_option_chains(chains, 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    found = {(c["type"], c["strike"], c["expiry"]) for c in d.unusual_contracts}
    # Only the 105 call qualifies: the 110 call is too small, the 95 put trades under 2x OI,
    # the 75 call is deep ITM (25% from spot) and the 3 DTE print is weekly noise.
    assert found == {("CALL", 105.0, _exp(30))}
    assert d.unusual_count == 1
    top = next(c for c in d.unusual_contracts if c["strike"] == 105.0)
    assert top["premium"] == pytest.approx(5_000 * 2.1 * 100)
    assert top["moneyness_pct"] == pytest.approx(5.0)

    table = OptionsFlowAgent().evaluate_with_data(d).raw_data["table"]
    assert table["columns"] == ["Type", "Expiry", "Strike", "Moneyness %", "Volume", "OI", "Premium $"]
    assert table["rows"][0][0] in ("CALL", "PUT")


# --------------------------------------------------------------------------- IV history and quality


def test_iv_history_switches_from_proxy_to_history(monkeypatch, noisy_ohlcv):
    raw = OptionChainSet(ticker="TEST", chains=_healthy_chains(), spot=100.0, as_of=TODAY.isoformat())
    monkeypatch.setattr(options_flow, "fetch_option_chains", lambda _t: raw)

    first = fetch_options_flow("TEST", noisy_ohlcv)
    assert first.iv_rank_source == "hv_proxy" and first.iv_rank is not None
    path = iv_history_path("TEST")
    assert path.exists()
    fetch_options_flow("TEST", noisy_ohlcv)  # same day overwrites, no duplicate row
    history = load_iv_history("TEST")
    assert list(history.index) == [TODAY.isoformat()]
    assert history.iloc[0] == pytest.approx(0.30)

    agent = OptionsFlowAgent()
    proxy_result = agent.evaluate("TEST", noisy_ohlcv)
    metrics = [m["metric"] for m in proxy_result.raw_data["metrics_table"]]
    assert "IV vs 1Y HV range (proxy)" in metrics
    assert not any(m.startswith("IV rank") for m in metrics)
    assert any("IV history still building (1 day)" in f for f in proxy_result.key_factors)

    def write_history(n: int) -> None:
        dates = [(TODAY - timedelta(days=n - i)).isoformat() for i in range(n)]
        values = [0.20 + 0.01 * i for i in range(n)]
        pd.DataFrame({"date": dates, "atm_iv": values}).to_csv(path, index=False)

    write_history(18)  # 18 prior snapshots + today = 19: still the proxy
    assert fetch_options_flow("TEST", noisy_ohlcv).iv_rank_source == "hv_proxy"

    write_history(19)  # 19 prior snapshots + today = 20: true IV rank
    d = fetch_options_flow("TEST", noisy_ohlcv)
    assert d.iv_rank_source == "history"
    assert d.iv_history_days == 20
    # History spans 0.20 to 0.38 and today is 0.30: rank = 0.10 / 0.18, 10 of 19 prior days lower.
    assert d.iv_rank == pytest.approx(100 * 0.10 / 0.18, abs=0.1)
    assert d.iv_percentile == pytest.approx(100 * 10 / 19, abs=0.1)
    result = agent.evaluate("TEST", noisy_ohlcv)
    assert result.raw_data["iv_rank_source"] == "history"
    assert any(m["metric"] == "IV rank (1Y history)" for m in result.raw_data["metrics_table"])


def test_stale_weekend_quotes_are_partial(monkeypatch, noisy_ohlcv):
    live = analyze_option_chains(_healthy_chains(), 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    assert live.quality == LIVE and not live.quality_notes

    stale_chains = _healthy_chains(bid=0.0, ask=0.0)
    stale = analyze_option_chains(stale_chains, 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    assert stale.available and stale.quality == PARTIAL and stale.stale_quotes
    assert stale.atm_iv is not None and not stale.iv_snapshot_reliable

    agent = OptionsFlowAgent()
    live_result = agent.evaluate_with_data(live)
    stale_result = agent.evaluate_with_data(stale)
    assert live_result.data_quality == LIVE
    assert stale_result.data_quality == PARTIAL
    assert stale_result.confidence < live_result.confidence
    assert any("Quotes look stale" in f for f in stale_result.key_factors)

    # Unreliable weekend IVs are not written to the IV history.
    raw = OptionChainSet(ticker="TEST", chains=stale_chains, spot=100.0, as_of=TODAY.isoformat())
    monkeypatch.setattr(options_flow, "fetch_option_chains", lambda _t: raw)
    fetch_options_flow("TEST", noisy_ohlcv)
    assert not iv_history_path("TEST").exists()


def test_old_trades_are_partial(noisy_ohlcv):
    chains = _healthy_chains(traded=TODAY - timedelta(days=9))
    d = analyze_option_chains(chains, 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    assert d.quality == PARTIAL
    assert any("days old" in n for n in d.quality_notes)


# --------------------------------------------------------------------------- agent output contract


def test_raw_data_contract_and_score_breakdown(noisy_ohlcv):
    d = analyze_option_chains(_healthy_chains(), 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    result = OptionsFlowAgent().evaluate_with_data(d)
    raw = result.raw_data

    required = {
        "iv_hv_ratio", "iv_rank", "iv_percentile", "iv_rank_source", "atm_iv", "skew", "term_structure_ratio",
        "max_pain", "put_call_volume_ratio", "put_call_oi_ratio", "iv_crush_warning", "levels",
        "unusual_contracts", "metrics_table", "score_breakdown", "links", "table",
    }
    assert required <= set(raw)
    assert isinstance(raw["iv_hv_ratio"], float)
    assert isinstance(raw["iv_crush_warning"], bool)
    assert result.agent_key == "options" and result.data_as_of == TODAY.isoformat()
    assert result.score == pytest.approx(sum(b["points"] for b in raw["score_breakdown"]), abs=0.05)
    assert {link["url"] for link in raw["links"]} == {
        "https://finance.yahoo.com/quote/TEST/options/",
        "https://www.barchart.com/stocks/quotes/TEST/options",
    }
    for m in raw["metrics_table"]:
        assert isinstance(m["metric"], str) and isinstance(m["value"], str)
    for lvl in raw["levels"]:
        assert lvl["kind"] in ("support", "resistance", "pivot") and isinstance(lvl["price"], float)

    texts = [result.summary, *result.key_factors]
    texts += [m["metric"] + m["value"] for m in raw["metrics_table"]]
    texts += [b["factor"] for b in raw["score_breakdown"]]
    assert not any(DASHES.search(t) for t in texts)
    assert all(t.isascii() for t in texts)


def test_unavailable_when_no_near_expirations(noisy_ohlcv):
    d = analyze_option_chains({_exp(3): (_side(), _side()), _exp(120): (_side(), _side())}, 100.0, noisy_ohlcv, "TEST", as_of=TODAY)
    assert d.available is False and "7 and 60" in (d.error or "")
    assert OptionsFlowAgent().evaluate_with_data(d).data_quality == UNAVAILABLE


def test_unavailable_data_object_defaults_are_empty():
    d = OptionsFlowData(ticker="X")
    assert d.available is False and d.quality == UNAVAILABLE
    assert d.atm_iv is None and d.iv_rank is None and d.max_pain is None
