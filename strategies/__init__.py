"""Trading strategy implementations."""

from strategies.bollinger_bands import BollingerBands
from strategies.buy_and_hold import BuyAndHold
from strategies.buy_the_dip import DipInUptrend, DrawdownDip
from strategies.insider_buys import InsiderBuysStrategy
from strategies.macd import MACDCrossover
from strategies.momentum import Momentum
from strategies.multi_factor import MultiFactorStrategy
from strategies.options_flow import OptionsFlowStrategy
from strategies.rsi import RSIMeanReversion
from strategies.sma_crossover import SMACrossover
from strategies.trend_following import AbsoluteMomentum, MSCIMomentumTrend, PBEmaChannel, TrendFilter

# Registry maps user-friendly names → (class, default kwargs, description)
STRATEGY_REGISTRY: dict[str, dict] = {
    "buy_and_hold": {
        "class": BuyAndHold,
        "defaults": {},
        "description": "Buy on day 1 and hold forever. The passive benchmark.",
        "params": {},
    },
    "sma_crossover": {
        "class": SMACrossover,
        "defaults": {"fast_window": 50, "slow_window": 200},
        "description": "Enter on golden cross (fast SMA > slow SMA), exit on death cross.",
        "params": {
            "fast_window": {"type": "int", "default": 50, "min": 2, "max": 500, "label": "Fast SMA window"},
            "slow_window": {"type": "int", "default": 200, "min": 5, "max": 1000, "label": "Slow SMA window"},
        },
    },
    "rsi": {
        "class": RSIMeanReversion,
        "defaults": {"period": 14, "oversold": 30.0, "overbought": 70.0},
        "description": "Buy when RSI is oversold, sell when overbought. Mean reversion.",
        "params": {
            "period": {"type": "int", "default": 14, "min": 2, "max": 100, "label": "RSI period"},
            "oversold": {"type": "float", "default": 30.0, "min": 1, "max": 49, "label": "Oversold threshold"},
            "overbought": {"type": "float", "default": 70.0, "min": 51, "max": 99, "label": "Overbought threshold"},
        },
    },
    "macd": {
        "class": MACDCrossover,
        "defaults": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "description": "Buy on MACD line crossing above signal line, sell on cross below.",
        "params": {
            "fast_period": {"type": "int", "default": 12, "min": 2, "max": 100, "label": "Fast EMA period"},
            "slow_period": {"type": "int", "default": 26, "min": 5, "max": 200, "label": "Slow EMA period"},
            "signal_period": {"type": "int", "default": 9, "min": 2, "max": 50, "label": "Signal period"},
        },
    },
    "bollinger_bands": {
        "class": BollingerBands,
        "defaults": {"window": 20, "num_std": 2.0},
        "description": "Buy at the lower Bollinger Band, sell at the upper band.",
        "params": {
            "window": {"type": "int", "default": 20, "min": 5, "max": 200, "label": "Lookback window"},
            "num_std": {"type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "label": "Standard deviations"},
        },
    },
    "momentum": {
        "class": Momentum,
        "defaults": {"lookback": 20, "entry_threshold": 0.05, "exit_threshold": -0.02},
        "description": "Buy when momentum exceeds threshold, sell when it drops below.",
        "params": {
            "lookback": {"type": "int", "default": 20, "min": 2, "max": 252, "label": "Lookback period"},
            "entry_threshold": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.50, "label": "Entry threshold (%)", "percent": True},
            "exit_threshold": {"type": "float", "default": -0.02, "min": -0.50, "max": -0.01, "label": "Exit threshold (%)", "percent": True},
        },
    },
    "options_flow": {
        "class": OptionsFlowStrategy,
        "defaults": {"bullish_threshold": 0.20, "bearish_threshold": -0.20, "rolling_window": 14},
        "description": "Price and volume proxy for options sentiment (ROC plus volume surge). Uses no real options data; see the Options Flow agent for that.",
        "params": {
            "bullish_threshold": {"type": "float", "default": 0.20, "min": 0.05, "max": 0.80, "label": "Bullish threshold"},
            "bearish_threshold": {"type": "float", "default": -0.20, "min": -0.80, "max": -0.05, "label": "Bearish threshold"},
            "rolling_window": {"type": "int", "default": 14, "min": 3, "max": 50, "label": "Rolling window"},
        },
    },
    "insider_buys": {
        "class": InsiderBuysStrategy,
        "defaults": {"dip_lookback": 20, "holding_period": 30, "dip_threshold": -0.03},
        "description": "Buys short-term dips that start recovering, a price-only proxy for insider dip buying. Uses no Form 4 data; see the Insider agent for that.",
        "params": {
            "dip_lookback": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "Dip lookback"},
            "holding_period": {"type": "int", "default": 30, "min": 5, "max": 120, "label": "Holding period"},
            "dip_threshold": {"type": "float", "default": -0.03, "min": -0.20, "max": -0.01, "label": "Dip threshold"},
        },
    },
    "multi_factor": {
        "class": MultiFactorStrategy,
        "defaults": {"fast_window": 20, "slow_window": 50, "momentum_threshold": 0.02, "min_factor_confluence": 2},
        "description": "Confluence of trend, volume-surge and dip-recovery factors, all computed from price and volume only.",
        "params": {
            "fast_window": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "Fast window"},
            "slow_window": {"type": "int", "default": 50, "min": 20, "max": 200, "label": "Slow window"},
            "momentum_threshold": {"type": "float", "default": 0.02, "min": 0.0, "max": 0.10, "label": "Momentum threshold"},
            "min_factor_confluence": {"type": "int", "default": 2, "min": 1, "max": 3, "label": "Minimum factor confluence"},
        },
    },
    # Trend and dip rules (D-48). Price only here: the Trend and dips view adds T-bills, the VIX curve and leverage.
    "trend_filter": {
        "class": TrendFilter,
        "defaults": {"window": 200, "buffer": 0.0, "vol_cap": 0.0, "vol_window": 20},
        "description": "Hold while the close is above its moving average, cash below. A re-entry buffer asks the close to clear the average by that much before buying back; a volatility cap above 0 also exits while the recent volatility is at or above it.",
        "params": {
            "window": {"type": "int", "default": 200, "min": 20, "max": 400, "label": "Average (days)"},
            "buffer": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.10, "label": "Re-entry buffer (0.02 = 2%)"},
            "vol_cap": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.5, "label": "Volatility cap (0 = off, 0.30 = 30%)"},
            "vol_window": {"type": "int", "default": 20, "min": 5, "max": 120, "label": "Volatility window (days)"},
        },
    },
    "pb_ema": {
        "class": PBEmaChannel,
        "defaults": {"length": 200},
        "description": "The PB EMA channel (FFriZz's PlayBit EMA): in when the close is above the EMAs of the highs and of the closes, out when it is below both, unchanged inside the channel.",
        "params": {
            "length": {"type": "int", "default": 200, "min": 20, "max": 400, "label": "EMA length"},
        },
    },
    "abs_momentum": {
        "class": AbsoluteMomentum,
        "defaults": {"months": 12, "skip": 1},
        "description": "Absolute momentum, decided at month-ends: hold while the return over the last months, leaving out the latest, is above 0 (above T-bills in the Trend and dips view).",
        "params": {
            "months": {"type": "int", "default": 12, "min": 1, "max": 24, "label": "Look back (months)"},
            "skip": {"type": "int", "default": 1, "min": 0, "max": 3, "label": "Leave out the latest (months)"},
        },
    },
    "msci_momentum": {
        "class": MSCIMomentumTrend,
        "defaults": {},
        "description": "MSCI's momentum measure on one asset, decided at month-ends: hold while the 6- and 12-month returns (each without the latest month) over 3-year weekly volatility, averaged, are positive.",
        "params": {},
    },
    "dip_rsi2": {
        "class": DipInUptrend,
        "defaults": {"rsi_period": 2, "entry": 10.0, "trend_window": 200, "exit_window": 5},
        "description": "Buy a short dip in an uptrend: RSI(2) under the entry level while the close is above its 200-day average; sell when the close is back above its 5-day average.",
        "params": {
            "entry": {"type": "float", "default": 10.0, "min": 1.0, "max": 40.0, "label": "Buy when RSI is under"},
            "trend_window": {"type": "int", "default": 200, "min": 20, "max": 400, "label": "Only above the average (days)"},
            "exit_window": {"type": "int", "default": 5, "min": 2, "max": 50, "label": "Sell above the average (days)"},
            "rsi_period": {"type": "int", "default": 2, "min": 2, "max": 14, "label": "RSI period"},
        },
    },
    "dip_drawdown": {
        "class": DrawdownDip,
        "defaults": {"dip": 0.10, "lookback": 252},
        "description": "Buy a fall from the high: cash until the close is the set share below its highest close of the look-back, then hold until it is back at that high.",
        "params": {
            "dip": {"type": "float", "default": 0.10, "min": 0.02, "max": 0.60, "label": "Fall from the high (0.10 = 10%)"},
            "lookback": {"type": "int", "default": 252, "min": 20, "max": 2520, "label": "High of the last (days)"},
        },
    },
}

__all__ = [
    "AbsoluteMomentum",
    "BollingerBands",
    "BuyAndHold",
    "DipInUptrend",
    "DrawdownDip",
    "InsiderBuysStrategy",
    "MACDCrossover",
    "MSCIMomentumTrend",
    "Momentum",
    "MultiFactorStrategy",
    "OptionsFlowStrategy",
    "PBEmaChannel",
    "RSIMeanReversion",
    "SMACrossover",
    "STRATEGY_REGISTRY",
    "TrendFilter",
]
