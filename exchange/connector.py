"""
Trading Bot - Exchange Connector
Wraps ccxt to provide a clean interface for interacting with Binance.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from config import settings


class ExchangeConnector:
    """Handles all communication with the exchange via ccxt."""

    def __init__(self):
        exchange_class = getattr(ccxt, settings.exchange.ID)
        config = {
            "apiKey": settings.exchange.API_KEY,
            "secret": settings.exchange.API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": settings.exchange.MARKET_TYPE},
        }

        if settings.exchange.TESTNET:
            config["sandbox"] = True

        self.exchange = exchange_class(config)

        if settings.exchange.TESTNET:
            self.exchange.set_sandbox_mode(True)

        if settings.exchange.MARKET_TYPE == "future":
            try:
                # Attempt to set leverage on the exchange side
                self.exchange.set_leverage(settings.exchange.LEVERAGE, settings.trading.PAIR)
                logger.info(f"Leverage successfully set to {settings.exchange.LEVERAGE}x for {settings.trading.PAIR}")
            except Exception as e:
                logger.warning(f"Could not automatically set leverage to {settings.exchange.LEVERAGE}x: {e}")

        logger.info(f"Exchange connector initialized: {settings.exchange.ID} (testnet={settings.exchange.TESTNET}, type={settings.exchange.MARKET_TYPE})")

    # ── Market Data ────────────────────────────────────────────

    def get_ticker(self, pair: str = None) -> dict:
        """Get current ticker data for a pair."""
        pair = pair or settings.trading.PAIR
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return {
                "pair": pair,
                "last": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "high": ticker["high"],
                "low": ticker["low"],
                "volume": ticker["baseVolume"],
                "timestamp": datetime.utcnow(),
            }
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {pair}: {e}")
            raise

    def get_ohlcv(self, pair: str = None, timeframe: str = None,
                  limit: int = 500, since: int = None) -> pd.DataFrame:
        """Fetch OHLCV candle data with pagination support for large limits."""
        pair = pair or settings.trading.PAIR
        timeframe = timeframe or settings.trading.TIMEFRAME

        all_ohlcv = []
        target_limit = limit
        
        # If no 'since' provided, calculate it based on limit * timeframe
        if since is None:
            # Approx minutes to go back
            mapping = {'m': 1, 'h': 60, 'd': 1440}
            unit = timeframe[-1]
            val = int(timeframe[:-1])
            minutes_per_candle = val * mapping.get(unit, 1)
            since = self.exchange.milliseconds() - (target_limit * minutes_per_candle * 60 * 1000)

        while len(all_ohlcv) < target_limit:
            current_limit = min(target_limit - len(all_ohlcv), 1000)
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    pair, timeframe=timeframe, limit=current_limit, since=since
                )
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + (1) # next ms
                
                # Rate limiting safety
                if len(all_ohlcv) < target_limit:
                    self.exchange.sleep(100) 
            except Exception as e:
                logger.error(f"Failed to fetch OHLCV chunk: {e}")
                break

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Remove duplicates from pagination overlap if any
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.drop_duplicates(subset="timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        
        logger.debug(f"Fetched {len(df)} candles total for {pair} {timeframe}")
        return df.tail(target_limit)

    def get_orderbook(self, pair: str = None, limit: int = 20) -> dict:
        """Fetch the order book."""
        pair = pair or settings.trading.PAIR
        try:
            return self.exchange.fetch_order_book(pair, limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {pair}: {e}")
            raise

    # ── Account ────────────────────────────────────────────────

    def get_balance(self) -> dict:
        """Get account balance."""
        try:
            balance = self.exchange.fetch_balance()
            return {
                "total": balance.get("total", {}),
                "free": balance.get("free", {}),
                "used": balance.get("used", {}),
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise

    # ── Orders ─────────────────────────────────────────────────

    def create_market_order(self, pair: str, side: str, amount: float) -> dict:
        """Create a market order."""
        pair = pair or settings.trading.PAIR
        try:
            order = self.exchange.create_order(pair, "market", side, amount)
            logger.info(f"Market {side} order placed: {amount} {pair} @ market")
            return order
        except Exception as e:
            logger.error(f"Failed to create market order: {e}")
            raise

    def create_limit_order(self, pair: str, side: str, amount: float,
                           price: float) -> dict:
        """Create a limit order."""
        pair = pair or settings.trading.PAIR
        try:
            order = self.exchange.create_order(pair, "limit", side, amount, price)
            logger.info(f"Limit {side} order placed: {amount} {pair} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Failed to create limit order: {e}")
            raise

    def cancel_order(self, order_id: str, pair: str = None) -> dict:
        """Cancel an open order."""
        pair = pair or settings.trading.PAIR
        try:
            result = self.exchange.cancel_order(order_id, pair)
            logger.info(f"Order {order_id} cancelled")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    def get_open_orders(self, pair: str = None) -> list:
        """Get all open orders."""
        pair = pair or settings.trading.PAIR
        try:
            return self.exchange.fetch_open_orders(pair)
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            raise

    def get_order(self, order_id: str, pair: str = None) -> dict:
        """Get details of a specific order."""
        pair = pair or settings.trading.PAIR
        try:
            return self.exchange.fetch_order(order_id, pair)
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            raise
