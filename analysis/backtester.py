"""
Trading Bot - Backtesting Engine (v2.0)
Enhanced backtester with detailed trade analysis, multi-stage exits, and proper metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from config import settings
from exchange.paper_trader import PaperTrader
from analysis.indicators import TechnicalIndicators
from strategies.combined_strategy import CombinedStrategy
from strategies.base_strategy import SignalType
from risk.risk_manager import RiskManager


class Backtester:
    """
    Enhanced backtester with:
    - Proper PnL calculation per trade
    - Trailing stop logic
    - Partial take-profit
    - Detailed per-trade logging
    - Multiple exit strategies
    """

    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.exchange = PaperTrader(initial_balance=initial_capital)
        self.indicators = TechnicalIndicators()
        self.strategy = CombinedStrategy()
        self.risk_manager = RiskManager()

        # Reset risk manager for backtest
        self.risk_manager.peak_balance = initial_capital

        self.results = []
        self.equity_curve = []
        self.open_trades = []

    def run(self, df: pd.DataFrame, use_ai: bool = False):
        """Run the backtest on a DataFrame."""
        if len(df) < 200:
            logger.error("Insufficient data for backtest (minimum 200 candles)")
            return None

        logger.info(f"Starting backtest on {len(df)} candles...")

        initial_lookback = 150
        balances = []
        timestamps = []
        trades_log = []
        
        # Initialize daily reset date from data
        first_date = df.index[initial_lookback]
        if hasattr(first_date, 'date'):
            self.risk_manager.daily_reset_date = first_date.date()
        self.risk_manager._simulated_date = None

        for i in tqdm(range(initial_lookback, len(df))):
            current_df = df.iloc[max(0, i - 200):i + 1]
            current_candle = df.iloc[i]
            current_price = current_candle["close"]
            current_high = current_candle["high"]
            current_low = current_candle["low"]
            timestamp = current_candle.name

            # 1. Update exchange
            self.exchange.set_current_price(settings.trading.PAIR, current_price)
            self.exchange.set_ohlcv_data(current_df)
            
            # Update simulated date for daily drawdown tracking
            self.risk_manager.set_current_date(timestamp)

            # 2. Calculate indicators
            df_with_indicators = self.indicators.calculate_all(current_df)

            # 3. Strategy signal
            signal = self.strategy.analyze(df_with_indicators)

            # 4. Check open trades for exits (use high/low for SL/TP accuracy)
            self._process_exits(current_price, current_high, current_low, signal, timestamp)

            # 5. Execute new entries
            if signal.is_actionable:
                self._process_entry(signal, current_price, timestamp)

            # 6. Track equity
            portfolio_val = self.exchange.get_portfolio_value()
            balances.append(portfolio_val)
            timestamps.append(timestamp)

        self.equity_curve = pd.Series(balances, index=timestamps)
        return self.calculate_metrics()

    def _process_entry(self, signal, price, timestamp):
        """Process a potential new trade entry."""
        balance = self.exchange.get_portfolio_value()
        open_dirs = [t["side"] for t in self.open_trades]

        risk_result = self.risk_manager.evaluate_signal(signal, balance, price, open_dirs)

        if not risk_result["approved"]:
            return

        try:
            action = "buy" if signal.action == SignalType.BUY else "sell"
            self.exchange.create_market_order(settings.trading.PAIR, action, risk_result["position_size"])

            trade = {
                "id": len(self.results) + len(self.open_trades) + 1,
                "entry_time": timestamp,
                "side": signal.action.value,
                "entry_price": price,
                "capital_at_entry": balance,
                "amount": risk_result["position_size"],
                "stop_loss": risk_result["stop_loss"],
                "take_profit": risk_result["take_profit"],
                "highest_price": price,
                "lowest_price": price,
                "status": "open",
                "partial_closed": False,
                "strategy": signal.strategy_name,
                "regime": getattr(signal, 'regime', 'unknown'),
                "signal_strength": signal.strength,
            }
            self.open_trades.append(trade)
            self.risk_manager.update_positions(len(self.open_trades))

        except Exception as e:
            logger.error(f"Backtest entry failed: {e}")

    def _process_exits(self, current_price, current_high, current_low, current_signal, timestamp):
        """Exit Logic: uses candle high/low for more accurate SL/TP detection."""
        closed_this_tick = []

        for trade in self.open_trades:
            should_close = False
            exit_reason = ""
            exit_price = current_price

            # Update price extremes
            trade["highest_price"] = max(trade.get("highest_price", trade["entry_price"]), current_high)
            trade["lowest_price"] = min(trade.get("lowest_price", trade["entry_price"]), current_low)

            # 1. Inverse Signal Exit — only if better than SL
            if current_signal.is_actionable and current_signal.strength >= 0.5:
                if trade["side"] == "buy" and current_signal.action == SignalType.SELL:
                    # Only exit if we'd lose less at current price than at SL
                    if current_price >= trade["stop_loss"]:
                        should_close = True
                        exit_reason = "INVERSE SIGNAL"
                        exit_price = current_price
                elif trade["side"] == "sell" and current_signal.action == SignalType.BUY:
                    if current_price <= trade["stop_loss"]:
                        should_close = True
                        exit_reason = "INVERSE SIGNAL"
                        exit_price = current_price

            if not should_close:
                # Apply trailing stop
                self._apply_trailing_stop(trade, current_price)

                if trade["side"] == "buy":
                    # Check if candle low hit SL
                    if current_low <= trade["stop_loss"]:
                        should_close = True
                        exit_price = trade["stop_loss"]  # Fill at SL price
                        exit_reason = "SL" if not trade.get("partial_closed") else "TRAILING SL"
                    # Check if candle high hit TP
                    elif current_high >= trade["take_profit"]:
                        if not trade.get("partial_closed", False) and settings.risk.PARTIAL_TP_ENABLED:
                            self._handle_partial_tp(trade, trade["take_profit"], timestamp)
                            continue
                        else:
                            should_close = True
                            exit_price = trade["take_profit"]
                            exit_reason = "TP"
                else:  # sell
                    if current_high >= trade["stop_loss"]:
                        should_close = True
                        exit_price = trade["stop_loss"]
                        exit_reason = "SL" if not trade.get("partial_closed") else "TRAILING SL"
                    elif current_low <= trade["take_profit"]:
                        if not trade.get("partial_closed", False) and settings.risk.PARTIAL_TP_ENABLED:
                            self._handle_partial_tp(trade, trade["take_profit"], timestamp)
                            continue
                        else:
                            should_close = True
                            exit_price = trade["take_profit"]
                            exit_reason = "TP"

            if should_close:
                self._close_full(trade, exit_price, timestamp, exit_reason)
                closed_this_tick.append(trade)

        for t in closed_this_tick:
            self.open_trades.remove(t)
        self.risk_manager.update_positions(len(self.open_trades))

    def _handle_partial_tp(self, trade, price, timestamp):
        """Close partial position at TP, move SL to breakeven, extend TP."""
        partial_amount = trade["amount"] * (settings.risk.PARTIAL_TP_CLOSE_PCT / 100)
        action = "sell" if trade["side"] == "buy" else "buy"

        # Set price for the close order
        self.exchange.set_current_price(settings.trading.PAIR, price)
        self.exchange.create_market_order(settings.trading.PAIR, action, partial_amount)

        trade["amount"] -= partial_amount
        trade["partial_closed"] = True
        trade["stop_loss"] = trade["entry_price"]  # Move to breakeven

        # Extend TP
        tp_dist = abs(trade["take_profit"] - trade["entry_price"])
        if trade["side"] == "buy":
            trade["take_profit"] = price + tp_dist
        else:
            trade["take_profit"] = price - tp_dist

    def _close_full(self, trade, price, timestamp, reason):
        """Close full position and record results."""
        action = "sell" if trade["side"] == "buy" else "buy"

        self.exchange.set_current_price(settings.trading.PAIR, price)
        self.exchange.create_market_order(settings.trading.PAIR, action, trade["amount"])

        # Calculate PnL
        if trade["side"] == "buy":
            pnl_per_unit = price - trade["entry_price"]
        else:
            pnl_per_unit = trade["entry_price"] - price

        pnl = pnl_per_unit * trade["amount"]
        pnl_pct = (pnl_per_unit / trade["entry_price"]) * 100

        trade.update({
            "exit_time": timestamp,
            "exit_price": price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "status": "closed"
        })
        self.results.append(trade)

        capital_after = self.exchange.get_portfolio_value()
        self.risk_manager.record_trade_result(pnl, capital_after)

    def _apply_trailing_stop(self, trade, current_price):
        """Apply ATR-based trailing stop."""
        trailing_pct = settings.risk.TRAILING_STOP_PCT / 100
        activation_pct = settings.risk.TRAILING_ACTIVATION_PCT / 100

        if trade["side"] == "buy":
            profit_pct = (current_price - trade["entry_price"]) / trade["entry_price"]
            if profit_pct >= activation_pct:
                new_sl = trade["highest_price"] * (1 - trailing_pct)
                trade["stop_loss"] = max(trade["stop_loss"], new_sl)
        else:
            profit_pct = (trade["entry_price"] - current_price) / trade["entry_price"]
            if profit_pct >= activation_pct:
                new_sl = trade["lowest_price"] * (1 + trailing_pct)
                trade["stop_loss"] = min(trade["stop_loss"], new_sl)

    def calculate_metrics(self):
        """Calculate comprehensive backtest metrics."""
        if not self.results:
            return {"error": "No trades executed"}

        trades_df = pd.DataFrame(self.results)
        total_pnl = trades_df["pnl"].sum()
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / len(trades_df) * 100

        equity = self.equity_curve
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        gross_profits = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

        # Calculate monthly return estimate
        if len(equity) > 0:
            total_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
            if total_days > 0:
                daily_return = (equity.iloc[-1] / equity.iloc[0]) ** (1 / total_days) - 1
                monthly_return = ((1 + daily_return) ** 30 - 1) * 100
            else:
                monthly_return = 0
        else:
            monthly_return = 0
            total_days = 0

        # Trades by exit reason
        exit_reasons = trades_df["reason"].value_counts().to_dict() if "reason" in trades_df.columns else {}

        # Trades by strategy
        strategy_perf = {}
        if "strategy" in trades_df.columns:
            for strat in trades_df["strategy"].unique():
                strat_trades = trades_df[trades_df["strategy"] == strat]
                strategy_perf[strat] = {
                    "trades": len(strat_trades),
                    "pnl": round(strat_trades["pnl"].sum(), 2),
                    "win_rate": round((strat_trades["pnl"] > 0).mean() * 100, 1),
                }

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.exchange.get_portfolio_value(),
            "total_return_usd": total_pnl,
            "total_return_pct": (total_pnl / self.initial_capital) * 100,
            "monthly_return_pct": monthly_return,
            "win_rate": win_rate,
            "total_trades": len(trades_df),
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "avg_trade_pct": trades_df["pnl_pct"].mean(),
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "best_trade_usd": trades_df["pnl"].max(),
            "worst_trade_usd": trades_df["pnl"].min(),
            "exit_reasons": exit_reasons,
            "strategy_performance": strategy_perf,
            "total_days": total_days,
        }

    def generate_report(self, metrics):
        """Generate detailed backtest report."""
        strategy_lines = ""
        if "strategy_performance" in metrics and metrics["strategy_performance"]:
            strategy_lines = "\n## Rendimiento por Estrategia\n"
            for strat, perf in metrics["strategy_performance"].items():
                strategy_lines += f"- **{strat}**: {perf['trades']} trades, ${perf['pnl']:+,.2f}, WR: {perf['win_rate']}%\n"

        exit_lines = ""
        if "exit_reasons" in metrics and metrics["exit_reasons"]:
            exit_lines = "\n## Razones de Cierre\n"
            for reason, count in metrics["exit_reasons"].items():
                exit_lines += f"- **{reason}**: {count} trades\n"

        return f"""
# Informe de Backtesting (v2.0)

## Resumen de Rendimiento
- **Capital Inicial**: ${metrics['initial_capital']:,.2f}
- **Capital Final**: ${metrics['final_capital']:,.2f}
- **Retorno Total**: ${metrics['total_return_usd']:+,.2f} ({metrics['total_return_pct']:+.2f}%)
- **Retorno Mensual Estimado**: {metrics.get('monthly_return_pct', 0):+.2f}%
- **Max Drawdown**: {metrics['max_drawdown']:.2f}%
- **Profit Factor**: {metrics['profit_factor']:.2f}

## Estadísticas de Trades
- **Total de Operaciones**: {metrics['total_trades']}
- **Win Rate**: {metrics['win_rate']:.2f}%
- **Promedio por Trade**: {metrics['avg_trade_pct']:+.2f}%
- **Promedio Win**: ${metrics['avg_win_usd']:+,.2f}
- **Promedio Loss**: ${metrics['avg_loss_usd']:+,.2f}
- **Mejor Trade**: ${metrics['best_trade_usd']:+,.2f}
- **Peor Trade**: ${metrics['worst_trade_usd']:+,.2f}
- **Período**: {metrics.get('total_days', 0):.0f} días
{strategy_lines}
{exit_lines}
---
*Backtest generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    def plot_results(self, filename="backtest_result.png"):
        """Plot equity curve and drawdown."""
        if self.equity_curve.empty:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.patch.set_facecolor('#1a1a2e')

        # Equity Curve
        ax1 = axes[0]
        ax1.set_facecolor('#16213e')
        ax1.plot(self.equity_curve.index, self.equity_curve.values, color='#00d2ff', linewidth=2, label='Portfolio')
        ax1.axhline(y=self.initial_capital, color='#ff6b6b', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title("Curva de Equidad", fontsize=14, fontweight='bold', color='white')
        ax1.legend(facecolor='#16213e', edgecolor='#333', labelcolor='white')
        ax1.grid(True, alpha=0.2, color='#333')
        ax1.tick_params(colors='white')

        # Mark trades
        if self.results:
            for trade in self.results:
                if trade.get("pnl", 0) > 0:
                    ax1.axvline(x=trade["exit_time"], color='green', alpha=0.3, linewidth=0.5)
                else:
                    ax1.axvline(x=trade["exit_time"], color='red', alpha=0.3, linewidth=0.5)

        # Drawdown
        ax2 = axes[1]
        ax2.set_facecolor('#16213e')
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, 0, drawdown.values, color='#ff6b6b', alpha=0.4)
        ax2.set_title("Drawdown (%)", fontsize=12, color='white')
        ax2.grid(True, alpha=0.2, color='#333')
        ax2.tick_params(colors='white')

        # Trade PnL Distribution  
        ax3 = axes[2]
        ax3.set_facecolor('#16213e')
        if self.results:
            pnls = [t["pnl"] for t in self.results]
            colors = ['#00ff88' if p > 0 else '#ff4444' for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.8)
            ax3.set_title("PnL por Trade", fontsize=12, color='white')
            ax3.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
        ax3.grid(True, alpha=0.2, color='#333')
        ax3.tick_params(colors='white')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        return filename
