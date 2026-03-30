"""
Trading Bot - Risk Manager (v2.0)
Dynamic position sizing with volatility adjustment and smart risk management.
"""

from datetime import datetime, date
from loguru import logger

from config import settings
from strategies.base_strategy import TradingSignal, SignalType


class RiskManager:
    """
    Enhanced risk management:
    - Dynamic position sizing based on signal strength + ATR
    - Kelly criterion-inspired sizing
    - Streak-based adjustment (reduce after losses, increase after wins)
    - Smarter R:R filtering
    """

    def __init__(self):
        self.max_position_pct = settings.risk.MAX_POSITION_SIZE_PCT / 100
        self.max_daily_dd_pct = settings.risk.MAX_DAILY_DRAWDOWN_PCT / 100
        self.max_total_dd_pct = settings.risk.MAX_TOTAL_DRAWDOWN_PCT / 100
        self.default_sl_pct = settings.risk.STOP_LOSS_PCT / 100
        self.default_tp_pct = settings.risk.TAKE_PROFIT_PCT / 100
        self.max_open_positions = settings.risk.MAX_OPEN_POSITIONS

        # Tracking
        self.daily_pnl = 0.0
        self.daily_reset_date = date.today()
        self.peak_balance = settings.trading.INITIAL_CAPITAL
        self.open_positions = 0
        self.trades_today = 0

        # Win/Loss tracking for dynamic sizing
        self.recent_wins = 0
        self.recent_losses = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Simulated date for backtesting
        self._simulated_date = None

        logger.info(
            f"Risk Manager v2.0 initialized: max_pos={self.max_position_pct:.1%}, "
            f"daily_dd={self.max_daily_dd_pct:.1%}, total_dd={self.max_total_dd_pct:.1%}"
        )

    def set_current_date(self, dt):
        """Set simulated date for backtesting."""
        if hasattr(dt, 'date'):
            self._simulated_date = dt.date()
        else:
            self._simulated_date = dt

    def evaluate_signal(self, signal: TradingSignal, current_balance: float,
                        current_price: float, open_directions: list = None) -> dict:
        """
        Evaluate a trading signal against risk rules with dynamic sizing.
        """
        self._check_daily_reset()

        # ── Basic Checks ──────────────────────────────────────
        if signal.action == SignalType.HOLD:
            return self._reject("Signal is HOLD")

        # Daily drawdown check
        if current_balance > 0:
            daily_dd = self.daily_pnl / current_balance
            if daily_dd < -self.max_daily_dd_pct:
                return self._reject(
                    f"Daily drawdown limit ({daily_dd:.2%} < -{self.max_daily_dd_pct:.2%})"
                )

        # Total drawdown check
        if self.peak_balance > 0:
            total_dd = (current_balance - self.peak_balance) / self.peak_balance
            if total_dd < -self.max_total_dd_pct:
                return self._reject(
                    f"Total drawdown limit ({total_dd:.2%} < -{self.max_total_dd_pct:.2%})"
                )

        # Max open positions
        if self.open_positions >= self.max_open_positions:
            return self._reject(
                f"Max positions reached ({self.open_positions}/{self.max_open_positions})"
            )

        # Redundancy check
        if open_directions and signal.action.value in open_directions:
            return self._reject(
                f"Already have an open {signal.action.value.upper()} position"
            )

        # ── Dynamic Position Sizing ───────────────────────────
        # Base risk = configured % of capital
        base_risk_pct = self.max_position_pct

        # Adjust based on signal strength (stronger signal = more risk)
        strength_mult = 0.6 + (signal.strength * 0.6)  # 0.6x to 1.2x

        # Adjust based on win/loss streak
        streak_mult = self._get_streak_multiplier()

        # Final risk percentage
        risk_pct = base_risk_pct * strength_mult * streak_mult
        risk_pct = max(risk_pct, 0.01)  # Minimum 1% risk
        risk_pct = min(risk_pct, self.max_position_pct * 1.5)  # Cap at 1.5x max

        risk_amount = current_balance * risk_pct

        # ── Stop Loss ─────────────────────────────────────────
        if signal.stop_loss is not None:
            stop_loss = signal.stop_loss
            sl_pct = abs(current_price - stop_loss) / current_price
        else:
            sl_pct = self.default_sl_pct
            if signal.action == SignalType.BUY:
                stop_loss = current_price * (1 - sl_pct)
            else:
                stop_loss = current_price * (1 + sl_pct)

        if sl_pct <= 0:
            return self._reject("Invalid Stop Loss distance")

        # Ensure SL is not too tight (< 0.1%) or too wide (> 5%)
        if sl_pct < 0.001:
            sl_pct = 0.003
            if signal.action == SignalType.BUY:
                stop_loss = current_price * (1 - sl_pct)
            else:
                stop_loss = current_price * (1 + sl_pct)
        elif sl_pct > 0.05:
            sl_pct = 0.05
            if signal.action == SignalType.BUY:
                stop_loss = current_price * (1 - sl_pct)
            else:
                stop_loss = current_price * (1 + sl_pct)

        # ── Position Value ────────────────────────────────────
        # Risk-Based: Position = Risk / SL%
        position_value = risk_amount / sl_pct

        # Leverage cap
        leverage = getattr(settings.exchange, "LEVERAGE", 1)
        max_allowed_value = current_balance * leverage
        if position_value > max_allowed_value:
            position_value = max_allowed_value

        # Minimum trade value
        min_trade_usd = getattr(settings.risk, "MIN_TRADE_VALUE_USD", 10.0)
        if position_value < min_trade_usd:
            return self._reject(
                f"Position value (${position_value:.2f}) below minimum (${min_trade_usd:.2f})"
            )

        # Position size in BTC
        if current_price > 0:
            position_size = position_value / current_price
        else:
            return self._reject("Invalid price")

        # ── Take Profit ───────────────────────────────────────
        if signal.take_profit is not None:
            take_profit = signal.take_profit
        else:
            if signal.action == SignalType.BUY:
                take_profit = current_price * (1 + self.default_tp_pct)
            else:
                take_profit = current_price * (1 - self.default_tp_pct)

        # ── Risk:Reward Ratio ─────────────────────────────────
        risk_dist = abs(current_price - stop_loss)
        reward_dist = abs(take_profit - current_price)
        rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 0

        # More lenient R:R (1.0 minimum for momentum, 1.5 for swing)
        min_rr = getattr(settings.risk, "MIN_RR_RATIO", 1.0)
        if rr_ratio < min_rr:
            return self._reject(
                f"Poor R:R ratio ({rr_ratio:.2f} < {min_rr})"
            )

        result = {
            "approved": True,
            "position_size": round(position_size, 8),
            "position_value": round(position_value, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_reward_ratio": round(rr_ratio, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_pct": round(risk_pct * 100, 2),
            "reason": f"Risk=${risk_amount:.2f} ({risk_pct:.1%}) | R:R={rr_ratio:.1f} | Streak={streak_mult:.2f}",
        }

        logger.info(
            f"Risk approved: {signal.action.value} {position_size:.6f} BTC @ ${current_price:,.2f} "
            f"(SL: ${stop_loss:,.2f}, TP: ${take_profit:,.2f}, R:R: {rr_ratio:.1f}, "
            f"Risk: ${risk_amount:.2f})"
        )

        return result

    def _get_streak_multiplier(self) -> float:
        """
        Adjust position size based on recent performance.
        - After consecutive wins: slightly increase (max 1.3x)
        - After consecutive losses: reduce (min 0.5x)
        """
        if self.consecutive_losses >= 3:
            return 0.5  # Significant reduction after 3+ losses
        elif self.consecutive_losses >= 2:
            return 0.7
        elif self.consecutive_losses >= 1:
            return 0.85
        elif self.consecutive_wins >= 3:
            return 1.3  # Increase after hot streak
        elif self.consecutive_wins >= 2:
            return 1.15
        return 1.0

    def record_trade_result(self, pnl: float, balance: float):
        """Record a completed trade's PnL."""
        self.daily_pnl += pnl
        self.trades_today += 1

        if pnl > 0:
            self.recent_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.recent_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        if balance > self.peak_balance:
            self.peak_balance = balance

        logger.debug(
            f"Trade recorded: PnL=${pnl:.2f}, daily=${self.daily_pnl:.2f}, "
            f"streak: W{self.consecutive_wins}/L{self.consecutive_losses}"
        )

    def set_peak_balance(self, peak: float):
        """Manually set peak balance."""
        if peak > 0:
            self.peak_balance = peak

    def update_positions(self, open_count: int):
        """Update count of open positions."""
        self.open_positions = open_count

    def _check_daily_reset(self):
        """Reset daily metrics at start of new day."""
        today = self._simulated_date if self._simulated_date else date.today()
        if today > self.daily_reset_date:
            logger.info(f"Daily reset: PnL=${self.daily_pnl:.2f}, trades={self.trades_today}")
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.daily_reset_date = today

    def _reject(self, reason: str) -> dict:
        """Create rejection result."""
        logger.warning(f"Risk REJECTED: {reason}")
        return {
            "approved": False,
            "position_size": 0.0,
            "position_value": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward_ratio": 0.0,
            "risk_amount": 0.0,
            "risk_pct": 0.0,
            "reason": reason,
        }

    def get_status(self) -> dict:
        """Get current risk manager status."""
        return {
            "daily_pnl": self.daily_pnl,
            "trades_today": self.trades_today,
            "open_positions": self.open_positions,
            "peak_balance": self.peak_balance,
            "daily_dd_limit": f"{self.max_daily_dd_pct:.1%}",
            "total_dd_limit": f"{self.max_total_dd_pct:.1%}",
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "streak_mult": self._get_streak_multiplier(),
        }
