"""
Trading Bot - Main Entry Point
Starts the trading engine with scheduler for autonomous 24/7 operation.
"""

import sys
import signal
import argparse
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler

from config import settings
from core.engine import TradingEngine


# Timeframe to seconds mapping
TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument(
        "--mode", choices=["paper", "live"], default=None,
        help="Trading mode (overrides .env)"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train the AI model before starting"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train the AI model, don't start trading"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--cycle-once", action="store_true",
        help="Run a single trading cycle and exit"
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run a historical backtest"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days for backtest (default 30)"
    )
    args = parser.parse_args()

    # Override mode if specified
    if args.mode:
        settings.trading.MODE = args.mode

    # Banner
    print("""
    +----------------------------------------------------------+
    |         [BOT] AI TRADING BOT [BOT]                        |
    |         Autonomous 24/7 Trading Engine                  |
    +----------------------------------------------------------+
    """)

    # Initialize engine
    engine = TradingEngine()
    engine.start()

    # Train AI model if requested
    if args.train or args.train_only:
        logger.info("Training AI model...")
        metrics = engine.train_ai_model(epochs=args.epochs)
        logger.info(f"Training results: {metrics}")

        if args.train_only:
            logger.info("Training complete. Exiting (--train-only mode).")
            return

    # Backtest mode
    if args.backtest:
        from run_backtest import run_backtest
        run_backtest(days=args.days, use_ai=engine._ai_ready)
        return

    # Single cycle mode
    if args.cycle_once:
        logger.info("Running single cycle...")
        engine.run_cycle()
        status = engine.get_status()
        logger.info(f"Status: {status}")
        return

    # ── Scheduler for 24/7 operation ───────────────────────────
    interval_seconds = TIMEFRAME_SECONDS.get(settings.trading.TIMEFRAME, 3600)
    scheduler = BlockingScheduler()

    scheduler.add_job(
        engine.run_cycle,
        "interval",
        seconds=interval_seconds,
        id="trading_cycle",
        name=f"Trading Cycle ({settings.trading.TIMEFRAME})",
        max_instances=1,
    )

    # Run first cycle immediately
    scheduler.add_job(
        engine.run_cycle,
        "date",
        id="initial_cycle",
        name="Initial Trading Cycle",
    )

    # AI retraining schedule (every N hours)
    if settings.ai.RETRAIN_INTERVAL_HOURS > 0:
        scheduler.add_job(
            engine.train_ai_model,
            "interval",
            hours=settings.ai.RETRAIN_INTERVAL_HOURS,
            id="ai_retrain",
            name="AI Model Retraining",
        )

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Shutdown signal received...")
        engine.stop()
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(f"Scheduler started: trading every {interval_seconds}s ({settings.trading.TIMEFRAME})")
    logger.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        engine.stop()
        logger.info("Bot stopped gracefully")


if __name__ == "__main__":
    main()
