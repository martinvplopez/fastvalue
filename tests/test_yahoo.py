"""The shared Yahoo wrapper: a refusal yfinance only logs is raised, so no cache keeps its empty answer as data.

yfinance's default ``hide_exceptions`` turns "401 Invalid Crumb" or "429" into an ERROR log line and an empty
result. Without this check, the insider cache kept "no insider transactions reported" for six hours after one
refused request. No network: the calls are plain functions that log the way yfinance does.
"""

from __future__ import annotations

import logging
import threading

import pytest

import data.options_flow as options_flow
from data import yahoo

REFUSAL = 'HTTP Error 401: {"finance":{"result":null,"error":{"code":"Unauthorized","description":"Invalid Crumb"}}}'


@pytest.fixture(autouse=True)
def _no_session(monkeypatch):
    monkeypatch.setattr(yahoo, "warm_session", lambda force=False: None)
    monkeypatch.setattr(yahoo, "BACKOFF_SECONDS", (0.0, 0.0))


def test_a_refusal_yfinance_only_logs_is_retried_then_raised():
    calls = []

    def refused():
        calls.append(1)
        logging.getLogger("yfinance").error(REFUSAL)
        return None

    with pytest.raises(yahoo.YahooRefusedError, match="Invalid Crumb"):
        yahoo.yahoo_call(refused, "AAPL insider")
    assert len(calls) == yahoo.RETRIES + 1


def test_a_refusal_then_an_answer_returns_the_answer():
    answers = iter([None, "rows"])

    def flaky():
        value = next(answers)
        if value is None:
            logging.getLogger("yfinance").error(REFUSAL)
        return value

    assert yahoo.yahoo_call(flaky, "AAPL insider") == "rows"


def test_not_found_is_an_answer_not_a_refusal():
    def missing():
        logging.getLogger("yfinance").error('HTTP Error 404: {"quoteSummary":{"error":{"description":"Quote not found for symbol: ZZZ"}}}')
        return None

    assert yahoo.yahoo_call(missing, "ZZZ info") is None


def test_a_refusal_logged_by_another_thread_is_not_this_calls():
    started, logged = threading.Event(), threading.Event()
    out: dict[str, str] = {}

    def slow():
        started.set()
        logged.wait(5)
        return "mine"

    worker = threading.Thread(target=lambda: out.setdefault("value", yahoo.yahoo_call(slow, "MSFT insider")))
    worker.start()
    started.wait(5)
    logging.getLogger("yfinance").error(REFUSAL)  # a parallel call's refusal, logged while this one runs
    logged.set()
    worker.join(5)
    assert out["value"] == "mine"


def test_an_empty_option_list_is_asked_again_for_a_us_ticker_only(monkeypatch):
    asked: list[str] = []

    class NoOptions:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def options(self) -> tuple[str, ...]:
            asked.append(self.symbol)
            return ()

    monkeypatch.setattr(options_flow.yf, "Ticker", NoOptions)
    monkeypatch.setattr(options_flow, "warm_session", lambda force=False: None)
    options_flow.fetch_option_chains.cache_clear()
    raw = options_flow.fetch_option_chains("NOOPT")
    assert asked == ["NOOPT", "NOOPT"] and raw.expirations_listed == 0 and not raw.chains
    asked.clear()
    options_flow.fetch_option_chains("NOOPT.L")  # a London listing: Yahoo lists no options for it, no second ask
    assert asked == ["NOOPT.L"]
