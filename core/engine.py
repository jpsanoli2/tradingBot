"""
Trading Bot - Core Engine
The main orchestrator that coordinates all components for autonomous trading.
"""

import time
import json
from datetime import datetime
from loguru import logger

from config import settings
from data.collector import DataCollector
from data.database import db
from data.models import Trade, Signal, BalanceHistory
from analysis.indicators import TechnicalIndicators
from strategies.combined_strategy import CombinedStrategy
from strategies.base_strategy import SignalType
from risk.risk_manager import RiskManager
from ai.price_predictor import PricePredictor
from exchange.connector import ExchangeConnector
from exchange.paper_trader import PaperTrader
from utils.logger import get_trade_logger
from alerts.discord_bot import notifier


trade_logger = get_trade_logger()


class TradingEngine:
    """
    Main trading engine that orchestrates:
    1. Data collection → 2. Technical analysis → 3. AI prediction →
    4. Strategy signals → 5. Risk evaluation → 6. Order execution
    """

    def __init__(self):
        # Initialize exchange (paper or live)
        if settings.trading.MODE == "paper":
            self.exchange = PaperTrader()
            # We still need a real connector for market data fetching
            self.market_data_connector = ExchangeConnector()
            logger.info("🔵 Running in PAPER TRADING mode")
        else:
            self.exchange = ExchangeConnector()
            self.market_data_connector = self.exchange
            logger.info("🔴 Running in LIVE TRADING mode")

        # Components
        self.data_collector = DataCollector(self.market_data_connector)
        self.indicators = TechnicalIndicators()
        self.strategy = CombinedStrategy()
        self.risk_manager = RiskManager()
        self.ai_predictor = PricePredictor()
        self.notifier = notifier

        # State
        self.running = False
        self.cycle_count = 0
        self.open_trades = []
        self.last_signal = None
        self._ai_ready = False

        # Status tracking for dashboard
        self.last_run = None
        self.is_running = False
        
        logger.info("Trading Engine initialized successfully")

    def start(self):
        """Start the trading engine."""
        self.running = True
        self.is_running = True
        logger.info("=" * 60)
        logger.info("🚀 TRADING ENGINE STARTED")
        logger.info(f"   Pair: {settings.trading.PAIR}")
        logger.info(f"   Timeframe: {settings.trading.TIMEFRAME}")
        logger.info(f"   Mode: {settings.trading.MODE}")
        logger.info("=" * 60)

        # Initial data fetch
        self._initialize_data()

    def stop(self):
        """Stop the trading engine."""
        self.running = False
        self.is_running = False
        logger.info("🛑 Trading Engine stopped")

    def run_cycle(self):
        """
        Execute one trading cycle: analyze → decide → execute.
        This is called by the scheduler every timeframe interval.
        """
        self.cycle_count += 1
        cycle_start = time.time()

        try:
            logger.info(f"━━━ Cycle #{self.cycle_count} ━━━")

            # Step 1: Fetch latest data
            df = self.data_collector.fetch_and_store()
            if df.empty:
                logger.warning("No data available, skipping cycle")
                return

            # Update paper trader price
            current_price = df.iloc[-1]["close"]
            if isinstance(self.exchange, PaperTrader):
                self.exchange.set_current_price(settings.trading.PAIR, current_price)
                self.exchange.set_ohlcv_data(df)

            # Step 2: Calculate indicators
            df_with_indicators = self.indicators.calculate_all(df)
            indicator_summary = self.indicators.get_signal_summary(df_with_indicators)
            logger.info(f"Technical Indicators: {indicator_summary}")

            # Step 3: AI prediction (if model is ready)
            ai_prediction = {"direction": "neutral", "confidence": 0.0}
            if self._ai_ready:
                try:
                    ai_prediction = self.ai_predictor.predict(df_with_indicators)
                except Exception as e:
                    logger.error(f"AI prediction failed: {e}")

            # Step 4: Generate strategy signal
            signal = self.strategy.analyze(df_with_indicators)

            # Boost/reduce signal based on AI prediction
            signal = self._apply_ai_adjustment(signal, ai_prediction)

            self.last_signal = signal

            # Step 5: Record signal in DB
            self._record_signal(signal, ai_prediction)

            # Step 6: Check open trades for stop-loss / take-profit
            self._check_open_trades(current_price)

            # Step 7: Execute if signal is actionable
            if signal.is_actionable:
                self._execute_signal(signal, current_price)

            # Step 8: Record balance
            self._record_balance()

            # Log cycle summary
            elapsed = time.time() - cycle_start
            logger.info(
                f"Cycle #{self.cycle_count} complete in {elapsed:.2f}s | "
                f"Price: ${current_price:,.2f} | Signal: {signal.action.value} "
                f"({signal.strength:.2f}) | AI: {ai_prediction['direction']} "
                f"({ai_prediction['confidence']:.2f})"
            )

            # Update status tracking
            self.last_run = datetime.now()

        except Exception as e:
            logger.exception(f"Error in trading cycle #{self.cycle_count}: {e}")

    def _initialize_data(self):
        """Fetch initial historical data and try to load AI model."""
        logger.info("Fetching initial historical data...")
        df = self.data_collector.fetch_and_store(limit=500)

        if not df.empty:
            current_price = df.iloc[-1]["close"]
            if isinstance(self.exchange, PaperTrader):
                self.exchange.set_current_price(settings.trading.PAIR, current_price)
                self.exchange.set_ohlcv_data(df)
            logger.info(f"Loaded {len(df)} candles, current price: ${current_price:,.2f}")

        if self.ai_predictor.load_model():
            self._ai_ready = True
            logger.info("AI model loaded successfully")
        else:
            logger.info("No AI model found. Run training first for AI-enhanced signals.")

        # Reload state from database
        self._reload_state()
        self._reload_open_trades()

    def _reload_state(self):
        """Reload engine state (balance history, risk metrics) from DB."""
        session = db.get_session()
        try:
            # Restore peak balance for drawdown calculation
            peak_record = session.query(BalanceHistory).order_by(BalanceHistory.total_balance.desc()).first()
            if peak_record:
                self.risk_manager.set_peak_balance(peak_record.total_balance)
            
            # Restore daily PnL
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            daily_trades = session.query(Trade).filter(
                Trade.exit_time >= today_start,
                Trade.status == "closed"
            ).all()
            
            total_daily_pnl = sum(t.pnl or 0 for t in daily_trades)
            self.risk_manager.daily_pnl = total_daily_pnl
            if total_daily_pnl != 0:
                logger.info(f"Restored daily PnL: ${total_daily_pnl:,.2f} ({len(daily_trades)} trades)")
                
        except Exception as e:
            logger.error(f"Failed to reload state: {e}")
        finally:
            session.close()

    def _reload_open_trades(self):
        """Reload trades with status 'open' from the database."""
        session = db.get_session()
        try:
            active_trades = session.query(Trade).filter(Trade.status == "open").all()
            for trade in active_trades:
                trade_dict = {
                    "id": trade.id,
                    "side": trade.side,
                    "entry_price": trade.entry_price,
                    "amount": trade.amount,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "highest_price": trade.entry_price,
                    "lowest_price": trade.entry_price,
                    "partial_closed": False,
                }
                self.open_trades.append(trade_dict)
                logger.info(f"🔄 Reloaded active trade: {trade.side.upper()} {trade.pair} @ {trade.entry_price}")
            
            # Update risk manager
            self.risk_manager.update_positions(len(self.open_trades))
            
            if self.open_trades:
                logger.info(f"Total reloaded trades: {len(self.open_trades)}")
        except Exception as e:
            logger.error(f"Failed to reload open trades: {e}")
        finally:
            session.close()

    def train_ai_model(self, epochs: int = 50):
        """Train the AI model on historical data."""
        logger.info("Starting AI model training...")
        df = self.data_collector.fetch_and_store(limit=500)

        if df.empty:
            logger.error("No data available for training")
            return

        df_with_indicators = self.indicators.calculate_all(df)
        metrics = self.ai_predictor.train(df_with_indicators, epochs=epochs)

        if "error" not in metrics:
            self._ai_ready = True
            logger.info(f"AI model trained: {metrics}")
        else:
            logger.error(f"AI training failed: {metrics}")

        return metrics

    def _apply_ai_adjustment(self, signal, ai_prediction):
        """Adjust signal strength based on AI prediction."""
        if ai_prediction["confidence"] < settings.ai.CONFIDENCE_THRESHOLD:
            return signal

        # If AI agrees with strategy → boost signal
        if (signal.action == SignalType.BUY and ai_prediction["direction"] == "up") or \
           (signal.action == SignalType.SELL and ai_prediction["direction"] == "down"):
            signal.strength = min(signal.strength * 1.3, 1.0)
            signal.reason += f" | AI CONFIRMS ({ai_prediction['confidence']:.2f})"

        # If AI disagrees → reduce signal
        elif (signal.action == SignalType.BUY and ai_prediction["direction"] == "down") or \
             (signal.action == SignalType.SELL and ai_prediction["direction"] == "up"):
            signal.strength *= 0.5
            signal.reason += f" | AI DISAGREES ({ai_prediction['confidence']:.2f})"

        return signal

    def _execute_signal(self, signal, current_price):
        """Evaluate risk and execute the trade if approved."""
        # Get current balance
        if isinstance(self.exchange, PaperTrader):
            balance = self.exchange.get_portfolio_value()
        else:
            bal = self.exchange.get_balance()
            balance = float(bal["total"].get("USDT", 0))

        # Risk evaluation
        open_directions = [t["side"] for t in self.open_trades]
        risk_result = self.risk_manager.evaluate_signal(signal, balance, current_price, open_directions)

        if not risk_result["approved"]:
            logger.info(f"Trade rejected by risk manager: {risk_result['reason']}")
            return

        # Execute order
        pair = settings.trading.PAIR
        position_size = risk_result["position_size"]

        try:
            if signal.action == SignalType.BUY:
                order = self.exchange.create_market_order(pair, "buy", position_size)
            else:
                order = self.exchange.create_market_order(pair, "sell", position_size)

            # Record trade
            trade = Trade(
                pair=pair,
                side=signal.action.value,
                entry_price=current_price,
                amount=position_size,
                entry_time=datetime.utcnow(),
                status="open",
                strategy=signal.strategy_name,
                stop_loss=risk_result["stop_loss"],
                take_profit=risk_result["take_profit"],
                notes=signal.reason,
            )

            session = db.get_session()
            try:
                session.add(trade)
                session.commit()
                self.open_trades.append({
                    "id": trade.id,
                    "side": signal.action.value,
                    "entry_price": current_price,
                    "amount": position_size,
                    "stop_loss": risk_result["stop_loss"],
                    "take_profit": risk_result["take_profit"],
                    "highest_price": current_price,
                    "lowest_price": current_price,
                    "partial_closed": False,
                })
            finally:
                session.close()

            self.risk_manager.update_positions(len(self.open_trades))

            trade_logger.info(
                f"TRADE EXECUTED | {signal.action.value.upper()} | {pair} | "
                f"Size: {position_size:.6f} BTC | Price: ${current_price:,.2f} | "
                f"SL: ${risk_result['stop_loss']:,.2f} | TP: ${risk_result['take_profit']:,.2f}"
            )

            # Send Discord alert
            self._run_async(self.notifier.notify_trade_open({
                "pair": pair,
                "side": signal.action.value,
                "entry_price": current_price,
                "amount": position_size,
                "stop_loss": risk_result["stop_loss"],
                "take_profit": risk_result["take_profit"],
                "notes": signal.reason
            }))

        except Exception as e:
            logger.error(f"Order execution failed: {e}")

    def _check_open_trades(self, current_price):
        """Check open trades for SL/TP hits, apply trailing stop, and handle partial closes."""
        trades_to_close = []

        for trade in self.open_trades:
            should_close = False
            close_reason = ""

            # ── Update highest/lowest price for trailing stop ──
            if trade["side"] == "buy":
                if current_price > trade.get("highest_price", trade["entry_price"]):
                    trade["highest_price"] = current_price
            else:
                if current_price < trade.get("lowest_price", trade["entry_price"]):
                    trade["lowest_price"] = current_price

            # ── Apply trailing stop ────────────────────────────
            self._update_trailing_stop(trade, current_price)

            # ── Check SL/TP ────────────────────────────────────
            if trade["side"] == "buy":
                if current_price <= trade["stop_loss"]:
                    should_close = True
                    close_reason = "STOP-LOSS hit" if not trade.get("partial_closed") else "TRAILING STOP hit (post-partial)"
                elif current_price >= trade["take_profit"]:
                    # Try partial close first
                    if not trade.get("partial_closed", False) and settings.risk.PARTIAL_TP_ENABLED:
                        self._partial_close_trade(trade, current_price)
                        continue  # Don't close fully yet
                    else:
                        should_close = True
                        close_reason = "TAKE-PROFIT hit (final)"
            else:  # sell
                if current_price >= trade["stop_loss"]:
                    should_close = True
                    close_reason = "STOP-LOSS hit" if not trade.get("partial_closed") else "TRAILING STOP hit (post-partial)"
                elif current_price <= trade["take_profit"]:
                    if not trade.get("partial_closed", False) and settings.risk.PARTIAL_TP_ENABLED:
                        self._partial_close_trade(trade, current_price)
                        continue
                    else:
                        should_close = True
                        close_reason = "TAKE-PROFIT hit (final)"

            if should_close:
                success = self._close_trade(trade, current_price, close_reason)
                if success:
                    trades_to_close.append(trade)

        for trade in trades_to_close:
            self.open_trades.remove(trade)

        self.risk_manager.update_positions(len(self.open_trades))

    def _update_trailing_stop(self, trade, current_price):
        """Update trailing stop loss — moves SL in favor as price advances."""
        trailing_pct = settings.risk.TRAILING_STOP_PCT / 100
        activation_pct = settings.risk.TRAILING_ACTIVATION_PCT / 100

        if trade["side"] == "buy":
            profit_pct = (current_price - trade["entry_price"]) / trade["entry_price"]
            if profit_pct >= activation_pct:
                highest = trade.get("highest_price", current_price)
                new_sl = highest * (1 - trailing_pct)
                if new_sl > trade["stop_loss"]:
                    old_sl = trade["stop_loss"]
                    trade["stop_loss"] = round(new_sl, 2)
                    self._update_trade_sl_in_db(trade["id"], trade["stop_loss"])
                    logger.info(
                        f"📈 Trailing SL updated (BUY): ${old_sl:,.2f} → ${trade['stop_loss']:,.2f} "
                        f"(peak: ${highest:,.2f}, profit: {profit_pct:.2%})"
                    )
        else:  # sell
            profit_pct = (trade["entry_price"] - current_price) / trade["entry_price"]
            if profit_pct >= activation_pct:
                lowest = trade.get("lowest_price", current_price)
                new_sl = lowest * (1 + trailing_pct)
                if new_sl < trade["stop_loss"]:
                    old_sl = trade["stop_loss"]
                    trade["stop_loss"] = round(new_sl, 2)
                    self._update_trade_sl_in_db(trade["id"], trade["stop_loss"])
                    logger.info(
                        f"📉 Trailing SL updated (SELL): ${old_sl:,.2f} → ${trade['stop_loss']:,.2f} "
                        f"(trough: ${lowest:,.2f}, profit: {profit_pct:.2%})"
                    )

    def _partial_close_trade(self, trade, current_price):
        """Close a portion of the position at first TP, move SL to breakeven, extend TP."""
        partial_pct = settings.risk.PARTIAL_TP_CLOSE_PCT / 100
        close_amount = trade["amount"] * partial_pct
        remaining_amount = trade["amount"] - close_amount
        pair = settings.trading.PAIR

        try:
            # Execute partial close order
            if trade["side"] == "buy":
                self.exchange.create_market_order(pair, "sell", close_amount)
                pnl = (current_price - trade["entry_price"]) * close_amount
            else:
                self.exchange.create_market_order(pair, "buy", close_amount)
                pnl = (trade["entry_price"] - current_price) * close_amount

            # Update trade state
            trade["amount"] = remaining_amount
            trade["partial_closed"] = True
            trade["stop_loss"] = trade["entry_price"]  # Move to breakeven

            # Extend TP for remaining position (2x the original distance)
            tp_distance = abs(trade["take_profit"] - trade["entry_price"])
            if trade["side"] == "buy":
                trade["take_profit"] = round(trade["entry_price"] + (tp_distance * 2), 2)
            else:
                trade["take_profit"] = round(trade["entry_price"] - (tp_distance * 2), 2)

            # Record PnL
            if isinstance(self.exchange, PaperTrader):
                balance = self.exchange.get_portfolio_value()
            else:
                bal = self.exchange.get_balance()
                balance = float(bal["total"].get("USDT", 0))
            self.risk_manager.record_trade_result(pnl, balance)

            # Update DB
            session = db.get_session()
            try:
                db_trade = session.query(Trade).get(trade["id"])
                if db_trade:
                    db_trade.amount = remaining_amount
                    db_trade.stop_loss = trade["stop_loss"]
                    db_trade.take_profit = trade["take_profit"]
                    db_trade.notes = (db_trade.notes or "") + (
                        f" | 🎯 Partial TP: {partial_pct:.0%} closed @ ${current_price:,.2f}"
                        f" (PnL: ${pnl:+,.2f}) | SL→breakeven, TP extended"
                    )
                    session.commit()
            finally:
                session.close()

            pnl_pct = ((current_price / trade["entry_price"]) - 1) * 100
            if trade["side"] == "sell":
                pnl_pct = (1 - (current_price / trade["entry_price"])) * 100

            trade_logger.info(
                f"🎯 PARTIAL TP | Closed {partial_pct:.0%} ({close_amount:.6f} BTC) | "
                f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | "
                f"Remaining: {remaining_amount:.6f} BTC | "
                f"New SL: ${trade['stop_loss']:,.2f} (breakeven) | "
                f"New TP: ${trade['take_profit']:,.2f} (extended)"
            )

            # Discord notification
            self._run_async(self.notifier.notify_trade_closed({
                "pair": pair,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "entry_price": trade["entry_price"],
                "exit_price": current_price,
                "notes": f"🎯 Partial TP ({partial_pct:.0%}) — remaining {remaining_amount:.6f} BTC running with trailing stop"
            }))

        except Exception as e:
            logger.error(f"Failed to execute partial close for trade {trade['id']}: {e}")

    def _update_trade_sl_in_db(self, trade_id, new_sl):
        """Update stop loss in database."""
        session = db.get_session()
        try:
            db_trade = session.query(Trade).get(trade_id)
            if db_trade:
                db_trade.stop_loss = new_sl
                session.commit()
        except Exception as e:
            logger.error(f"Failed to update SL in DB: {e}")
        finally:
            session.close()

    def _close_trade(self, trade, exit_price, reason) -> bool:
        """
        Close a trade and record PnL.
        Returns True if successful, False otherwise.
        """
        pair = settings.trading.PAIR

        # Execute closing order
        try:
            if trade["side"] == "buy":
                self.exchange.create_market_order(pair, "sell", trade["amount"])
                pnl = (exit_price - trade["entry_price"]) * trade["amount"]
            else:
                self.exchange.create_market_order(pair, "buy", trade["amount"])
                pnl = (trade["entry_price"] - exit_price) * trade["amount"]

            pnl_pct = ((exit_price / trade["entry_price"]) - 1) * 100
            if trade["side"] == "sell":
                pnl_pct = -pnl_pct

            # Update DB
            session = db.get_session()
            try:
                db_trade = session.query(Trade).get(trade["id"])
                if db_trade:
                    db_trade.exit_price = exit_price
                    db_trade.exit_time = datetime.utcnow()
                    db_trade.pnl = pnl
                    db_trade.pnl_pct = pnl_pct
                    db_trade.status = "closed"
                    db_trade.notes = (db_trade.notes or "") + f" | Closed: {reason}"
                    session.commit()
            finally:
                session.close()

            # Record in risk manager
            if isinstance(self.exchange, PaperTrader):
                balance = self.exchange.get_portfolio_value()
            else:
                bal = self.exchange.get_balance()
                balance = float(bal["total"].get("USDT", 0))

            self.risk_manager.record_trade_result(pnl, balance)

            emoji = "✅" if pnl > 0 else "❌"
            trade_logger.info(
                f"{emoji} TRADE CLOSED | {reason} | {pair} | "
                f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | "
                f"Entry: ${trade['entry_price']:,.2f} → Exit: ${exit_price:,.2f}"
            )

            # Send Discord alert
            self._run_async(self.notifier.notify_trade_closed({
                "pair": pair,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "notes": reason
            }))
            
            return True

        except Exception as e:
            logger.error(f"Failed to close trade {trade['id']}: {e}")
            return False

    def _record_signal(self, signal, ai_prediction):
        """Record signal in database."""
        session = db.get_session()
        try:
            db_signal = Signal(
                pair=settings.trading.PAIR,
                strategy=signal.strategy_name,
                action=signal.action.value,
                strength=signal.strength,
                price_at_signal=signal.price,
                ai_confidence=ai_prediction.get("confidence"),
                indicators=json.dumps({"reason": signal.reason}),
            )
            session.add(db_signal)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record signal: {e}")
        finally:
            session.close()

    def _record_balance(self):
        """Record current balance in history."""
        try:
            if isinstance(self.exchange, PaperTrader):
                total = self.exchange.get_portfolio_value()
                free = self.exchange.balance["USDT"]
                in_pos = total - free
            else:
                bal = self.exchange.get_balance()
                total = float(bal["total"].get("USDT", 0))
                free = float(bal["free"].get("USDT", 0))
                in_pos = total - free

            session = db.get_session()
            try:
                record = BalanceHistory(
                    total_balance=total,
                    available_balance=free,
                    in_positions=in_pos,
                    daily_pnl=self.risk_manager.daily_pnl,
                )
                session.add(record)
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to record balance: {e}")

    def get_status(self) -> dict:
        """Get current engine status."""
        balance_info = {}
        if isinstance(self.exchange, PaperTrader):
            balance_info = self.exchange.get_pnl()
        else:
            try:
                bal = self.exchange.get_balance()
                balance_info = {"total_usdt": bal["total"].get("USDT", 0)}
            except Exception:
                pass

        return {
            "running": self.running,
            "mode": settings.trading.MODE,
            "pair": settings.trading.PAIR,
            "timeframe": settings.trading.TIMEFRAME,
            "cycles": self.cycle_count,
            "open_trades": len(self.open_trades),
            "ai_ready": self._ai_ready,
            "balance": balance_info,
            "risk": self.risk_manager.get_status(),
            "last_signal": {
                "action": self.last_signal.action.value if self.last_signal else None,
                "strength": self.last_signal.strength if self.last_signal else 0,
            },
        }

    def get_unrealized_metrics(self, current_price: float) -> dict:
        """
        Calculate metrics for open trades based on current price.
        Returns total unrealized PnL and a map of trade_id -> current_pnl.
        """
        total_unrealized_pnl = 0.0
        open_trades_metrics = {}

        for trade in self.open_trades:
            if trade["side"] == "buy":
                pnl = (current_price - trade["entry_price"]) * trade["amount"]
                pnl_pct = ((current_price / trade["entry_price"]) - 1) * 100
            else:  # sell
                pnl = (trade["entry_price"] - current_price) * trade["amount"]
                # For shorts, PnL % is (Entry - Exit) / Entry
                pnl_pct = ((trade["entry_price"] / current_price) - 1) * 100
                # Wait, the formula for short % is usually (Entry - Exit) / Entry
                # If entry is 100 and price is 90, (100-90)/100 = 10% profit.
                # The previous line used (100/90 - 1) which is (1.11 - 1) = 11.1%, which is wrong.
                pnl_pct = (1 - (current_price / trade["entry_price"])) * 100

            total_unrealized_pnl += pnl
            open_trades_metrics[str(trade["id"])] = {
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2)
            }

        # Get total realized PnL from history (simplified for dashboard)
        # In a real scenario, we'd query the DB or track it in memory
        
        return {
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "trades": open_trades_metrics
        }

    def _run_async(self, coro):
        """
        Safely run a coroutine from any thread.
        If an event loop is already running in the current thread, use it.
        Otherwise, run it in a new thread with its own loop.
        """
        import asyncio
        import threading

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(coro)
            else:
                # Loop exists but not running (unlikely for get_running_loop)
                asyncio.run(coro)
        except RuntimeError:
            # No loop in current thread
            def run_in_new_loop(c):
                asyncio.run(c)
            
            threading.Thread(target=run_in_new_loop, args=(coro,), daemon=True).start()
