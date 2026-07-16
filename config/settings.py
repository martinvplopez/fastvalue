"""Central configuration for paths, tickers, and backtest defaults."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR: Path = PROJECT_ROOT / "analysis" / "output"

DEFAULT_TICKERS: list[str] = ["SPY", "QQQ", "AAPL"]
LOOKBACK_YEARS: int = 10
COMMISSION_PER_SHARE: float = 0.005
INITIAL_CASH: float = 100_000.0

MIN_BARS_WARNING: int = 200
MAX_FORWARD_FILL_DAYS: int = 3
