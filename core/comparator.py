"""Strategy Comparison Engine for running and benchmarking multiple quantitative strategies side-by-side."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from analysis.metrics import compute_buy_and_hold_returns, compute_metrics
from config.settings import COMMISSION_PER_SHARE, INITIAL_CASH
from core.engine import run_backtest
from data.fetcher import fetch_ohlcv
from strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class StrategyComparisonResult:
    """Comparison matrix and equity curves across multiple backtested strategies."""

    ticker: str
    benchmark_name: str
    benchmark_stats: dict[str, float]
    strategy_results: list[dict[str, Any]]  # List of strategy metadata + metrics + equity curve
    rankings: list[dict[str, Any]]  # Sorted list of strategy names by Sharpe / Total Return
    date_range: dict[str, str]


def compare_strategies(
    ticker: str,
    strategy_names: list[str] | None = None,
    params_override: dict[str, dict[str, Any]] | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> StrategyComparisonResult:
    """
    Run and compare multiple strategies side-by-side on a given ticker.

    Args:
        ticker: Symbol (e.g. SPY, AAPL).
        strategy_names: Optional list of strategy registry keys to test. Defaults to all registered.
        params_override: Optional dict mapping strategy_name -> parameter dict overrides.
        ohlcv: Optional pre-loaded OHLCV DataFrame.

    Returns:
        StrategyComparisonResult with metrics matrix, equity curves, and rankings.
    """
    symbol = ticker.upper()

    if ohlcv is None:
        data = fetch_ohlcv([symbol])
        if symbol not in data:
            raise ValueError(f"No market data available for ticker {symbol}")
        ohlcv = data[symbol]

    if strategy_names is None:
        # Default to all registered strategies
        strategy_names = list(STRATEGY_REGISTRY.keys())

    params_override = params_override or {}

    # Calculate Benchmark
    benchmark_returns = compute_buy_and_hold_returns(ohlcv)
    benchmark_stats = compute_metrics(benchmark_returns)
    benchmark_equity = (1 + benchmark_returns).cumprod() * INITIAL_CASH

    strategy_results: list[dict[str, Any]] = []

    for name in strategy_names:
        if name not in STRATEGY_REGISTRY:
            logger.warning("Skipping unknown strategy: %s", name)
            continue

        entry = STRATEGY_REGISTRY[name]
        params = dict(entry["defaults"])
        if name in params_override:
            params.update(params_override[name])

        try:
            strat_inst = entry["class"](**params)
            signals = strat_inst.generate_signals(ohlcv)
            res = run_backtest(
                ohlcv=ohlcv,
                signals=signals,
                commission_per_share=COMMISSION_PER_SHARE,
                init_cash=INITIAL_CASH,
            )

            stats = compute_metrics(res.returns, signals=signals)
            equity = (1 + res.returns).cumprod() * INITIAL_CASH

            display_name = name.replace("_", " ").title()
            strategy_results.append({
                "strategy": name,
                "display_name": display_name,
                "params": params,
                "stats": stats,
                "returns": res.returns,
                "equity": [
                    {"time": d.strftime("%Y-%m-%d"), "value": round(v, 2)}
                    for d, v in zip(equity.index, equity.values)
                ],
            })
        except Exception as exc:
            logger.warning("Error running strategy %s on %s: %s", name, symbol, exc)

    # Rank strategies by Sharpe ratio, then Total Return
    rankings = sorted(
        strategy_results,
        key=lambda s: (s["stats"].get("sharpe", 0.0), s["stats"].get("total_return", 0.0)),
        reverse=True,
    )

    rank_list = [
        {
            "rank": i + 1,
            "strategy": r["strategy"],
            "display_name": r["display_name"],
            "total_return": r["stats"].get("total_return", 0.0),
            "cagr": r["stats"].get("cagr", 0.0),
            "sharpe": r["stats"].get("sharpe", 0.0),
            "max_drawdown": r["stats"].get("max_drawdown", 0.0),
            "win_rate": r["stats"].get("win_rate", 0.0),
        }
        for i, r in enumerate(rankings)
    ]

    return StrategyComparisonResult(
        ticker=symbol,
        benchmark_name="Buy & Hold",
        benchmark_stats=benchmark_stats,
        strategy_results=strategy_results,
        rankings=rank_list,
        date_range={
            "start": ohlcv.index[0].strftime("%Y-%m-%d"),
            "end": ohlcv.index[-1].strftime("%Y-%m-%d"),
        },
    )
