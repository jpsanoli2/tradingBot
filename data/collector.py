"""
Trading Bot - Data Collector
Fetches and stores OHLCV data from the exchange.
"""

import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from config import settings
from data.models import OHLCV, PriceTick
from data.database import db


class DataCollector:
    """Collects and manages market data."""

    def __init__(self, exchange):
        """
        Args:
            exchange: ExchangeConnector or PaperTrader instance
        """
        self.exchange = exchange

    def fetch_and_store(self, pair: str = None, timeframe: str = None,
                        limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data from exchange and store in database."""
        pair = pair or settings.trading.PAIR
        timeframe = timeframe or settings.trading.TIMEFRAME

        df = self.exchange.get_ohlcv(pair, timeframe, limit)

        if df.empty:
            logger.warning(f"No data received for {pair} {timeframe}")
            return df

        # Store in database
        session = db.get_session()
        try:
            for idx, row in df.iterrows():
                timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx)

                # Check if candle already exists
                existing = session.query(OHLCV).filter(
                    OHLCV.pair == pair,
                    OHLCV.timeframe == timeframe,
                    OHLCV.timestamp == timestamp,
                ).first()

                if not existing:
                    candle = OHLCV(
                        pair=pair,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                    )
                    session.add(candle)
                else:
                    # Update existing candle (essential for real-time price updates)
                    existing.open = row["open"]
                    existing.high = row["high"]
                    existing.low = row["low"]
                    existing.close = row["close"]
                    existing.volume = row["volume"]

            session.commit()
            logger.debug(f"Stored {len(df)} candles for {pair} {timeframe}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store OHLCV data: {e}")
        finally:
            session.close()

        return df

    def get_historical_data(self, pair: str = None, timeframe: str = None,
                            limit: int = 500) -> pd.DataFrame:
        """Get historical data from database, fetch from exchange if needed."""
        pair = pair or settings.trading.PAIR
        timeframe = timeframe or settings.trading.TIMEFRAME

        session = db.get_session()
        try:
            candles = (
                session.query(OHLCV)
                .filter(OHLCV.pair == pair, OHLCV.timeframe == timeframe)
                .order_by(OHLCV.timestamp.desc())
                .limit(limit)
                .all()
            )

            if len(candles) < limit:
                logger.info(f"Insufficient DB data ({len(candles)}/{limit}), fetching from exchange")
                return self.fetch_and_store(pair, timeframe, limit)

            df = pd.DataFrame(
                [
                    {
                        "timestamp": c.timestamp,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in reversed(candles)
                ]
            )
            df.set_index("timestamp", inplace=True)
            return df
        finally:
            session.close()
    def fetch_and_store_ticker(self, pair: str = None) -> dict:
        """Fetch current ticker and store in database."""
        pair = pair or settings.trading.PAIR
        ticker = self.exchange.get_ticker(pair)
        
        session = db.get_session()
        try:
            tick = PriceTick(
                pair=pair,
                price=ticker["last"],
                timestamp=ticker["timestamp"]
            )
            session.add(tick)
            session.commit()
            return ticker
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store ticker data: {e}")
            return ticker
        finally:
            session.close()
