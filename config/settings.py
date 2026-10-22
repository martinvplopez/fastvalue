"""Central configuration for paths, tickers, and backtest defaults."""

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR: Path = PROJECT_ROOT / "analysis" / "output"

# Where a friend can try FastValue, for the "Try FastValue" link on the shareable page. Empty means no link.
PUBLIC_URL: str = os.environ.get("FASTVALUE_PUBLIC_URL", "")

DEFAULT_TICKERS: list[str] = ["SPY", "QQQ", "AAPL"]
LOOKBACK_YEARS: int = 10
COMMISSION_PER_SHARE: float = 0.005
INITIAL_CASH: float = 100_000.0

MIN_BARS_WARNING: int = 200
MAX_FORWARD_FILL_DAYS: int = 3

# Persistent local state (score history, saved weights, IV snapshots, watchlist).
# Kept out of git: it is personal and machine specific.
STORE_DIR: Path = PROJECT_ROOT / "data" / "store"
CONFIG_DIR: Path = PROJECT_ROOT / "config"

# Seconds each provider result stays cached in-process.
CACHE_TTL_QUOTES: int = 60
CACHE_TTL_OPTIONS: int = 300
CACHE_TTL_MACRO: int = 900
CACHE_TTL_NEWS: int = 900
CACHE_TTL_FUNDAMENTALS: int = 6 * 3600
CACHE_TTL_INSIDER: int = 6 * 3600

# Per-agent evaluation timeout inside the master agent (seconds).
AGENT_TIMEOUT_SECONDS: float = 30.0

# How the Macro agent's score treats volatility:
# "contrarian": high VIX is a buying opportunity (long-run average 20, above 40 is extreme fear)
# "risk":       high VIX is a reason to avoid stocks (trend-following view)
# The market regime (and so the weight profile) treats high VIX as CRISIS either way.
VIX_STANCE: str = "contrarian"

# Free cash flow definition used for EV/FCF scoring (analysis/fundamental_metrics.py):
#   "owner":           EBITDA - maintenance CapEx (D&A, capped at CapEx) - interest - taxes - change in working
#                      capital. The owner's definition (Buffett-style owner earnings).
#   "owner_all_capex": the same with all CapEx subtracted (growth CapEx too): the most conservative variant.
#   "standard":        operating cash flow - CapEx, as reported.
# Point-in-time S&P 500 test 2011-2026 (research_results.md, decision_log.md T-05), top minus bottom quintile a year:
# owner -1.5%, owner_all_capex +2.1%, standard +4.0%, and reported FCF exists for more companies (81% against 69%).
# Owner's call (September 2026): "standard" feeds EV/FCF and the cap; the other two are always shown in the
# fundamentals evidence table for comparison.
FCF_DEFINITION: str = "standard"
# The change in working capital is averaged over up to this many yearly changes (1 = latest year only).
# One-year swings are mostly noise; averaging keeps the definition and removes most of it.
WORKING_CAPITAL_AVERAGE_YEARS: int = 3

# News sentiment. Yahoo headlines need no key. Two API sources switch on when their key is set in the
# environment (never in tracked files): ALPHAVANTAGE_API_KEY (NEWS_SENTIMENT endpoint, sentiment score and
# relevance per ticker attached to each item) and FINNHUB_API_KEY (company news, scored with the lexicon).
NEWS_LOOKBACK_DAYS: int = 14  # window requested from dated news APIs
NEWS_MIN_RELEVANCE: float = 0.2  # floor on a provider's relevance weight so a barely related item still counts a little

# Portfolio review limits (core/portfolio.py). Evidence: concentrated portfolios with a 10-20% cap per
# position; tight stops hurt mean-reverting value holdings, so the ATR stop is a sizing reference only.
MAX_POSITION_WEIGHT: float = 0.20  # trim above this share of the portfolio
NEW_POSITION_WEIGHT: float = 0.10  # initial size of a new position (before adds)
RISK_PER_POSITION: float = 0.01  # share of the portfolio at risk between entry and the ATR stop reference
DRAWDOWN_REVIEW: float = 0.25  # drawdown from the high since entry that triggers a thesis review
MAX_LIQUIDATION_DAYS: float = 5.0  # days to exit at 10% of average daily volume before flagging liquidity
# Benchmarks of the portfolio analytics and My track record, converted to the portfolio currency (EUR) at the ECB
# rate of each day. The first is the primary: beta, tracking error, the sector comparison, the per-order "vs MSCI World"
# tables, the rules lab's core fund and the "skill or luck" test use it alone. Accumulating UCITS funds on purpose:
# their traded price includes the dividends (after US withholding tax), and a European investor can buy them. SPY and
# QQQ track the same indices but are US funds EU brokers do not sell to private investors, and their price alone
# leaves out about 1.8 and 0.8 points a year of dividends (decision_log.md D-36). A listing added here must be
# accumulating, because the track record uses the price that traded. "currency" is the listing's quote currency, so a
# failed Yahoo lookup cannot convert it at the wrong rate.
PORTFOLIO_BENCHMARKS: list[dict[str, str]] = [
    {"id": "msci_world", "label": "MSCI World", "ticker": "IWDA.L", "isin": "IE00B4L5Y983", "currency": "USD",
     "fund": "iShares Core MSCI World"},
    {"id": "sp500", "label": "S&P 500", "ticker": "CSPX.L", "isin": "IE00B5BMR087", "currency": "USD",
     "fund": "iShares Core S&P 500"},
    {"id": "nasdaq100", "label": "Nasdaq 100", "ticker": "CNDX.L", "isin": "IE00B53SZB19", "currency": "USD",
     "fund": "iShares Nasdaq 100"},
]
# The primary alone, for callers that compare with one index.
PORTFOLIO_BENCHMARK: dict[str, str] = {"ticker": PORTFOLIO_BENCHMARKS[0]["ticker"], "label": PORTFOLIO_BENCHMARKS[0]["label"]}

# Trend check (core/systematic.py, decision_log.md D-48): the markets the daily check watches, the rules the Trend and
# dips view shows for them, and the rule whose flip raises an alert. The two-engine rule re-enters only 2% above the
# 200-day average, a buffer against whipsaw; it still flips about 7 to 9 times a year on each market (1994-2026), so one
# rule alerts and the others are shown only. An alert informs; it never changes a score.
TREND_WATCH: list[str] = ["SPY", "QQQ", "IWDA.L"]
TREND_CHECK_RULES: tuple[str, ...] = ("trend_filter", "two_engine", "pb_ema")
TREND_ALERT_RULES: tuple[str, ...] = ("two_engine",)
