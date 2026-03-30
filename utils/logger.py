"""
Trading Bot - Centralized Logger
Uses loguru for powerful, simple logging with file rotation.
"""

import sys
from pathlib import Path
from loguru import logger

from config.settings import log as log_config, BASE_DIR


def setup_logger():
    """Configure the application logger."""
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_config.LEVEL,
        colorize=True,
    )

    # File handler with rotation
    log_path = BASE_DIR / log_config.FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=log_config.LEVEL,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    # Trade-specific log file
    trade_log_path = log_path.parent / "trades.log"
    logger.add(
        str(trade_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        rotation="5 MB",
        retention="90 days",
        filter=lambda record: "trade" in record["extra"],
    )

    return logger


# Initialize logger on import
setup_logger()


def get_trade_logger():
    """Get a logger specifically for trade events."""
    return logger.bind(trade=True)
