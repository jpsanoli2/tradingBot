"""
Trading Bot - Base Strategy (v2.0)
Abstract base class with enhanced signal model.
"""

from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from config import settings


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """Represents a trading signal from a strategy."""
    action: SignalType
    strength: float  # 0.0 to 1.0
    strategy_name: str
    price: float
    stop_loss: float = None
    take_profit: float = None
    reason: str = ""
    regime: str = ""  # market regime at signal time
    urgency: float = 0.0  # 0-1, how time-sensitive the entry is
    
    @property
    def is_actionable(self) -> bool:
        return self.action != SignalType.HOLD and self.strength >= settings.trading.MIN_SIGNAL_STRENGTH


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> TradingSignal:
        """
        Analyze market data and generate a trading signal.

        Args:
            df: DataFrame with OHLCV data and calculated indicators

        Returns:
            TradingSignal with action, strength, and metadata
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return the strategy's current parameters."""
        pass
