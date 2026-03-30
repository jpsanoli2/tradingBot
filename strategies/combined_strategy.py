"""
Trading Bot - Adaptive Combined Strategy (v2.0)
Regime-aware meta-strategy that dynamically weights sub-strategies
based on current market conditions.
"""

import pandas as pd
from loguru import logger
from typing import List

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
from config import settings


class CombinedStrategy(BaseStrategy):
    """
    Adaptive meta-strategy that:
    1. Detects market regime
    2. Weights sub-strategies based on regime
    3. Uses strongest signal rather than averaging (avoids dilution)
    4. Requires less consensus but higher individual signal quality
    """

    def __init__(self, strategies: List[tuple] = None):
        super().__init__("Adaptive")

        if strategies is None:
            from strategies.trend_following import MomentumBreakoutStrategy
            from strategies.swing_continuation import SwingContinuationStrategy
            
            self.strategy_instances = {
                "momentum": MomentumBreakoutStrategy(),
                "swing": SwingContinuationStrategy(),
            }
            
            # Base weights
            self.base_weights = {
                "momentum": 0.60,
                "swing": 0.40,
            }

        self.min_strength = settings.trading.MIN_CONSENSUS_STRENGTH

        # Regime-specific weight adjustments (momentum-dominant)
        self.regime_weights = {
            "trending": {"momentum": 0.55, "swing": 0.45},
            "volatile_trend": {"momentum": 0.70, "swing": 0.30},
            "ranging": {"momentum": 0.50, "swing": 0.50},
            "transitioning": {"momentum": 0.55, "swing": 0.45},
            "unknown": {"momentum": 0.60, "swing": 0.40},
        }

    def analyze(self, df: pd.DataFrame) -> TradingSignal:
        """Analyze using regime-adaptive strategy selection."""
        if df.empty:
            return TradingSignal(
                action=SignalType.HOLD, strength=0.0,
                strategy_name=self.name, price=0.0, reason="No data"
            )

        price = df.iloc[-1]["close"]
        regime = df.iloc[-1].get("regime", "unknown")

        # Get regime-specific weights
        weights = self.regime_weights.get(regime, self.base_weights)

        # Collect all signals
        signals = {}
        for name, strategy in self.strategy_instances.items():
            signal = strategy.analyze(df)
            signals[name] = {
                "signal": signal,
                "weight": weights.get(name, 0.33),
            }

        # ── Strategy Selection: Best Signal Approach ──────────
        # Instead of averaging (which dilutes), pick the BEST signal
        # but validate it doesn't conflict with high-weight strategies
        
        best_signal = None
        best_weighted_score = 0.0
        all_reasons = []
        
        for name, data in signals.items():
            signal = data["signal"]
            weight = data["weight"]
            
            if signal.action == SignalType.HOLD:
                all_reasons.append(f"[{name}:HOLD]")
                continue
            
            # Weighted score = strength * weight * urgency_factor
            urgency_factor = 1.0 + signal.urgency * 0.3  # Urgent signals get priority
            weighted_score = signal.strength * weight * urgency_factor
            
            all_reasons.append(
                f"[{name}:{signal.action.value}(s={signal.strength:.2f},w={weight:.2f})]"
            )
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_signal = signal

        # ── Conflict Detection ────────────────────────────────
        buy_signals = [s for s in signals.values() 
                       if s["signal"].action == SignalType.BUY and s["weight"] > 0.15]
        sell_signals = [s for s in signals.values() 
                        if s["signal"].action == SignalType.SELL and s["weight"] > 0.15]

        has_conflict = len(buy_signals) > 0 and len(sell_signals) > 0

        if has_conflict:
            # Only trade if one side is clearly dominant
            buy_power = sum(s["signal"].strength * s["weight"] for s in buy_signals)
            sell_power = sum(s["signal"].strength * s["weight"] for s in sell_signals)
            
            dominance_ratio = max(buy_power, sell_power) / (min(buy_power, sell_power) + 1e-10)
            
            if dominance_ratio < 2.0:
                # Signals are too balanced — hold
                all_reasons.append(f"CONFLICT: dominance={dominance_ratio:.1f}, holding")
                return TradingSignal(
                    action=SignalType.HOLD,
                    strength=0.0,
                    strategy_name=self.name,
                    price=price,
                    reason=" | ".join(all_reasons),
                    regime=regime,
                )
            else:
                all_reasons.append(f"Conflict resolved (dominance={dominance_ratio:.1f})")

        # ── No actionable signal ──────────────────────────────
        if best_signal is None or best_signal.action == SignalType.HOLD:
            return TradingSignal(
                action=SignalType.HOLD,
                strength=0.0,
                strategy_name=self.name,
                price=price,
                reason=" | ".join(all_reasons),
                regime=regime,
            )

        # ── Final strength check ──────────────────────────────
        final_strength = best_signal.strength
        
        # Boost if multiple strategies agree
        agreeing = sum(1 for s in signals.values() 
                       if s["signal"].action == best_signal.action)
        if agreeing >= 2:
            final_strength = min(final_strength * 1.2, 1.0)
            all_reasons.append(f"CONSENSUS: {agreeing} strategies agree")

        if final_strength < self.min_strength:
            return TradingSignal(
                action=SignalType.HOLD,
                strength=final_strength,
                strategy_name=self.name,
                price=price,
                reason=" | ".join(all_reasons) + f" | Below min strength ({final_strength:.2f})",
                regime=regime,
            )

        # ── Build final signal ────────────────────────────────
        signal = TradingSignal(
            action=best_signal.action,
            strength=final_strength,
            strategy_name=f"{self.name}>{best_signal.strategy_name}",
            price=price,
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
            reason=" | ".join(all_reasons),
            regime=regime,
            urgency=best_signal.urgency,
        )

        logger.info(
            f"[{self.name}] Regime: {regime} | Signal: {signal.action.value} "
            f"(strength={signal.strength:.2f}) via {best_signal.strategy_name}"
        )
        return signal

    def get_parameters(self) -> dict:
        params = {
            "min_strength": self.min_strength,
            "base_weights": self.base_weights,
            "regime_weights": self.regime_weights,
            "strategies": {},
        }
        for name, strategy in self.strategy_instances.items():
            params["strategies"][name] = strategy.get_parameters()
        return params
