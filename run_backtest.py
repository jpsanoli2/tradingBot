"""
Trading Bot - Backtest Execution Script
Run this script to evaluate the current strategy performance.
"""

import argparse
import sys
import os
from loguru import logger
from datetime import datetime
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from exchange.connector import ExchangeConnector
from analysis.backtester import Backtester

def run_backtest(days: int = 30, use_ai: bool = False, pair: str = None, timeframe: str = None):
    """
    Fetch historical data and run the backtest.
    """
    pair = pair or settings.trading.PAIR
    timeframe = timeframe or settings.trading.TIMEFRAME
    
    logger.info(f"Preparing backtest for {pair} ({timeframe}) - Last {days} days")
    
    # Initialize exchange to fetch real data
    exchange = ExchangeConnector()
    
    # Calculate how many candles we need (roughly)
    # timeframe strings like '1h', '1m', '1d'
    mapping = {'m': 1, 'h': 60, 'd': 1440}
    unit = timeframe[-1]
    val = int(timeframe[:-1])
    minutes_per_candle = val * mapping.get(unit, 1)
    
    total_minutes = days * 24 * 60
    candles_needed = int(total_minutes / minutes_per_candle)
    
    # Fetch data
    # CCXT often limits to 500-1000 candles per request
    all_ohlcv = []
    current_limit = min(candles_needed, 1000)
    
    logger.info(f"Fetching approximately {candles_needed} candles...")
    
    # For now, let's fetch in chunks if needed
    try:
        # Simple fetch (we can improve this with a loop over 'since' if we want more data)
        df = exchange.get_ohlcv(pair, timeframe, limit=candles_needed)
        
        if df.empty:
            logger.error("Failed to fetch historical data.")
            return

        # Initialize and run backtester
        backtester = Backtester(initial_capital=settings.trading.INITIAL_CAPITAL)
        metrics = backtester.run(df, use_ai=use_ai)
        
        if not metrics or "error" in metrics:
            logger.error(f"Backtest failed: {metrics.get('error', 'Unknown error')}")
            return

        # Generate and display report
        report = backtester.generate_report(metrics)
        
        # Handle Windows encoding for console output
        try:
            print(report)
        except UnicodeEncodeError:
            print(report.encode('ascii', 'replace').decode('ascii'))
        
        # Save report to file
        report_file = f"backtest_report_{pair.replace('/', '_')}_{timeframe}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        # Print key metrics to console
        print(f"\n=== KEY METRICS ===")
        print(f"Return: {metrics['total_return_pct']:+.2f}%")
        print(f"Monthly Est: {metrics.get('monthly_return_pct', 0):+.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Trades: {metrics['total_trades']}")
        print(f"Max DD: {metrics['max_drawdown']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        
        # Plot results
        chart_file = f"backtest_chart_{pair.replace('/', '_')}_{timeframe}.png"
        backtester.plot_results(chart_file)
        
        logger.info(f"Backtest complete. Report saved to {report_file}")
        
    except Exception as e:
        logger.exception(f"An error occurred during backtest: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Backtest for Trading Bot")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest (default: 30)")
    parser.add_argument("--use-ai", action="store_true", help="Enhance signals with AI model")
    parser.add_argument("--pair", type=str, help="Trading pair (e.g. BTC/USDT)")
    parser.add_argument("--tf", type=str, help="Timeframe (e.g. 1h, 15m)")
    
    args = parser.parse_args()
    
    run_backtest(days=args.days, use_ai=args.use_ai, pair=args.pair, timeframe=args.tf)
