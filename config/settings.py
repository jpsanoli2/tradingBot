"""
Trading Bot - Configuration Settings
Loads configuration from .env file with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class ExchangeConfig:
    """Exchange connection settings."""
    ID: str = os.getenv("EXCHANGE_ID", "binance")
    API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
    API_SECRET: str = os.getenv("EXCHANGE_API_SECRET", "")
    TESTNET: bool = os.getenv("EXCHANGE_TESTNET", "true").lower() == "true"
    MARKET_TYPE: str = os.getenv("EXCHANGE_MARKET_TYPE", "future")
    LEVERAGE: int = int(os.getenv("EXCHANGE_LEVERAGE", "100"))


class TradingConfig:
    """Trading parameters."""
    PAIR: str = os.getenv("TRADING_PAIR", "BTC/USDT")
    TIMEFRAME: str = os.getenv("TRADING_TIMEFRAME", "15m")
    MODE: str = os.getenv("TRADING_MODE", "paper")  # paper | live
    INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "1000.0"))
    MIN_SIGNAL_STRENGTH: float = float(os.getenv("MIN_SIGNAL_STRENGTH", "0.3"))
    MIN_CONSENSUS_STRENGTH: float = float(os.getenv("MIN_CONSENSUS_STRENGTH", "0.1"))


class RiskConfig:
    """Risk management parameters."""
    MAX_POSITION_SIZE_PCT: float = float(os.getenv("MAX_POSITION_SIZE_PCT", "8.0"))
    MAX_DAILY_DRAWDOWN_PCT: float = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "5.0"))
    MAX_TOTAL_DRAWDOWN_PCT: float = float(os.getenv("MAX_TOTAL_DRAWDOWN_PCT", "15.0"))
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "1.5"))
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
    MIN_TRADE_VALUE_USD: float = float(os.getenv("MIN_TRADE_VALUE_USD", "10.0"))
    # Trailing Stop Loss
    TRAILING_STOP_PCT: float = float(os.getenv("TRAILING_STOP_PCT", "1.0"))
    TRAILING_ACTIVATION_PCT: float = float(os.getenv("TRAILING_ACTIVATION_PCT", "0.5"))
    # Minimum R:R ratio
    MIN_RR_RATIO: float = float(os.getenv("MIN_RR_RATIO", "1.0"))
    # Partial Take Profit
    PARTIAL_TP_ENABLED: bool = os.getenv("PARTIAL_TP_ENABLED", "true").lower() == "true"
    PARTIAL_TP_CLOSE_PCT: float = float(os.getenv("PARTIAL_TP_CLOSE_PCT", "50.0"))



class AIConfig:
    """AI/ML model parameters."""
    LOOKBACK_PERIOD: int = int(os.getenv("MODEL_LOOKBACK_PERIOD", "60"))
    RETRAIN_INTERVAL_HOURS: int = int(os.getenv("MODEL_RETRAIN_INTERVAL_HOURS", "24"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.65"))


class DashboardConfig:
    """Dashboard settings."""
    HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("DASHBOARD_PORT", "8080"))


class DiscordConfig:
    """Discord notification settings."""
    BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
    CHANNEL_ID: int = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
    ENABLED: bool = os.getenv("DISCORD_ALERTS_ENABLED", "false").lower() == "true"


class LogConfig:
    """Logging settings."""
    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FILE: str = os.getenv("LOG_FILE", "logs/trading_bot.log")


# Singleton-style access
exchange = ExchangeConfig()
trading = TradingConfig()
risk = RiskConfig()
ai = AIConfig()
dashboard = DashboardConfig()
discord = DiscordConfig()
log = LogConfig()
