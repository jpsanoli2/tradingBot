"""
Trading Bot - Paper Trader
Simulates trading without real funds. Implements the same interface as ExchangeConnector.
"""

import uuid
import pandas as pd
from datetime import datetime
from typing import Optional
from loguru import logger

from config import settings


class PaperTrader:
    """
    Simulated trading engine for backtesting and paper trading.
    Mirrors ExchangeConnector interface so they can be swapped seamlessly.
    """

    def __init__(self, initial_balance: float = None):
        self.initial_balance = initial_balance or settings.trading.INITIAL_CAPITAL
        self.balance = {
            "USDT": self.initial_balance,
            "BTC": 0.0,
        }
        self.orders = []
        self.open_orders = []
        self.trade_history = []
        self.current_prices = {}
        self._last_ohlcv = None

        logger.info(f"Paper trader initialized with {self.initial_balance} USDT")

    def set_current_price(self, pair: str, price: float):
        """Update the current price (used by the engine to feed live prices)."""
        self.current_prices[pair] = price

    def set_ohlcv_data(self, df: pd.DataFrame):
        """Set historical OHLCV data for paper trading."""
        self._last_ohlcv = df

    # ── Market Data ────────────────────────────────────────────

    def get_ticker(self, pair: str = None) -> dict:
        """Get simulated ticker."""
        pair = pair or settings.trading.PAIR
        price = self.current_prices.get(pair, 0.0)
        return {
            "pair": pair,
            "last": price,
            "bid": price * 0.9999,
            "ask": price * 1.0001,
            "high": price * 1.02,
            "low": price * 0.98,
            "volume": 1000.0,
            "timestamp": datetime.utcnow(),
        }

    def get_ohlcv(self, pair: str = None, timeframe: str = None,
                  limit: int = 500, since: int = None) -> pd.DataFrame:
        """Return stored OHLCV data."""
        if self._last_ohlcv is not None:
            return self._last_ohlcv.tail(limit)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_orderbook(self, pair: str = None, limit: int = 20) -> dict:
        """Simulated orderbook."""
        pair = pair or settings.trading.PAIR
        price = self.current_prices.get(pair, 50000.0)
        spread = price * 0.0001
        return {
            "bids": [[price - spread * i, 0.1] for i in range(1, limit + 1)],
            "asks": [[price + spread * i, 0.1] for i in range(1, limit + 1)],
        }

    # ── Account ────────────────────────────────────────────────

    def get_balance(self) -> dict:
        """Get simulated balance."""
        pair = settings.trading.PAIR
        btc_price = self.current_prices.get(pair, 0.0)
        btc_value = self.balance["BTC"] * btc_price

        return {
            "total": {
                "USDT": self.balance["USDT"] + btc_value,
                "BTC": self.balance["BTC"],
            },
            "free": {
                "USDT": self.balance["USDT"],
                "BTC": self.balance["BTC"],
            },
            "used": {"USDT": 0.0, "BTC": 0.0},
        }

    # ── Orders ─────────────────────────────────────────────────

    def create_market_order(self, pair: str, side: str, amount: float) -> dict:
        """Simulate a market order execution."""
        pair = pair or settings.trading.PAIR
        price = self.current_prices.get(pair, 0.0)

        if price <= 0:
            raise ValueError(f"No price available for {pair}")

        # Apply small simulated slippage (0.02% for major pairs)
        slippage = 1.0002 if side == "buy" else 0.9998
        fill_price = price * slippage
        cost = fill_price * amount

        # Handle Margin & Balance checking
        is_future = getattr(settings.exchange, "MARKET_TYPE", "spot") == "future"
        leverage = getattr(settings.exchange, "LEVERAGE", 1) if is_future else 1
        
        # Determine if this is a closing order (don't margin-check closes)
        is_closing = False
        if side == "sell" and self.balance["BTC"] > 0:
            is_closing = True  # Selling existing long
        elif side == "buy" and self.balance["BTC"] < 0:
            is_closing = True  # Buying back short
        
        if not is_closing:
            required_margin = cost / leverage
            portfolio_val = self.get_portfolio_value()
            if portfolio_val < required_margin:
                raise ValueError(f"Insufficient simulated margin: ${portfolio_val:,.2f} < ${required_margin:,.2f}")

        if side == "buy":
            self.balance["USDT"] -= cost
            self.balance["BTC"] += amount
        elif side == "sell":
            if not is_future and self.balance["BTC"] < amount:
                raise ValueError(f"Insufficient BTC for Spot sell: {self.balance['BTC']} < {amount}")
            self.balance["BTC"] -= amount
            self.balance["USDT"] += cost

        # Simulate fee (0.05% for futures, 0.1% for spot)
        fee_rate = 0.0005 if is_future else 0.001
        fee_amount = cost * fee_rate
        self.balance["USDT"] -= fee_amount

        order = {
            "id": str(uuid.uuid4())[:8],
            "pair": pair,
            "type": "market",
            "side": side,
            "amount": amount,
            "price": fill_price,
            "cost": cost,
            "fee": fee_amount,
            "status": "closed",
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.orders.append(order)
        self.trade_history.append(order)

        logger.info(
            f"[PAPER] Market {side}: {amount:.6f} BTC @ ${fill_price:,.2f} "
            f"(cost: ${cost:,.2f}, fee: ${fee_amount:.2f})"
        )

        return order

    def create_limit_order(self, pair: str, side: str, amount: float,
                           price: float) -> dict:
        """Simulate a limit order (adds to pending)."""
        pair = pair or settings.trading.PAIR

        order = {
            "id": str(uuid.uuid4())[:8],
            "pair": pair,
            "type": "limit",
            "side": side,
            "amount": amount,
            "price": price,
            "cost": price * amount,
            "fee": 0.0,
            "status": "open",
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.orders.append(order)
        self.open_orders.append(order)

        logger.info(f"[PAPER] Limit {side} order placed: {amount:.6f} BTC @ ${price:,.2f}")
        return order

    def cancel_order(self, order_id: str, pair: str = None) -> dict:
        """Cancel a simulated open order."""
        for order in self.open_orders:
            if order["id"] == order_id:
                order["status"] = "cancelled"
                self.open_orders.remove(order)
                logger.info(f"[PAPER] Order {order_id} cancelled")
                return order
        raise ValueError(f"Order {order_id} not found")

    def get_open_orders(self, pair: str = None) -> list:
        """Get simulated open orders."""
        if pair:
            return [o for o in self.open_orders if o["pair"] == pair]
        return self.open_orders

    def get_order(self, order_id: str, pair: str = None) -> dict:
        """Get a specific simulated order."""
        for order in self.orders:
            if order["id"] == order_id:
                return order
        raise ValueError(f"Order {order_id} not found")

    # ── Paper Trading Specific ─────────────────────────────────

    def get_portfolio_value(self) -> float:
        """Get total portfolio value in USDT."""
        btc_price = self.current_prices.get(settings.trading.PAIR, 0.0)
        return self.balance["USDT"] + (self.balance["BTC"] * btc_price)

    def get_pnl(self) -> dict:
        """Get profit/loss metrics."""
        current_value = self.get_portfolio_value()
        pnl = current_value - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        return {
            "initial_balance": self.initial_balance,
            "current_value": current_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "total_trades": len(self.trade_history),
        }
