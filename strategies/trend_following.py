"""
Trading Bot - Momentum Breakout Strategy (v2.0)
High-probability breakout entries using Keltner Channels + Volume + MACD momentum.
Designed for TRENDING markets (ADX > 20).
"""

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType


class MomentumBreakoutStrategy(BaseStrategy):
    """
    Momentum breakout strategy:
    - Keltner Channel breakout for entry
    - Volume surge confirmation (vol > 1.3x average)
    - MACD momentum alignment
    - EMA trend filter
    - Tight ATR-based stops for controlled risk
    
    Only activates in trending/volatile_trend regimes.
    """

    def __init__(self):
        super().__init__("MomentumBreakout")
        self.min_adx = 18  # Lowered to catch more moves
        self.vol_threshold = 1.2  # Volume surge threshold
        self.atr_sl_mult = 0.9  # Balanced stop-loss
        self.atr_tp_mult = 1.5  # Take-profit = 1.5x ATR (1.67:1 R:R)

    def analyze(self, df: pd.DataFrame) -> TradingSignal:
        """Analyze for momentum breakout opportunities."""
        if df.empty or len(df) < 60:
            return self._hold(0.0, "Insufficient data")

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        price = latest["close"]
        regime = latest.get("regime", "unknown")

        # Only trade in trending regimes
        if regime in ("ranging",):
            return self._hold(price, f"Wrong regime: {regime}")

        score = 0.0
        reasons = []
        
        # ── Directional Candle Confirmation ───────────────────
        # Count bullish vs bearish candles in last 3
        recent_candles = df.iloc[-4:-1]  # last 3 completed candles
        bullish_count = sum(1 for _, c in recent_candles.iterrows() if c["close"] > c["open"])
        bearish_count = len(recent_candles) - bullish_count
        candle_bias = bullish_count - bearish_count  # +3 to -3

        # ── Keltner Channel Breakout ──────────────────────────
        kc_upper = latest.get("KC_upper")
        kc_lower = latest.get("KC_lower")
        kc_mid = latest.get("KC_mid")
        prev_kc_upper = prev.get("KC_upper")
        prev_kc_lower = prev.get("KC_lower")

        if kc_upper is not None and kc_lower is not None:
            # BULLISH: Price breaks above KC upper
            if price > kc_upper:
                # Fresh breakout (prev candle was inside)
                if prev["close"] <= prev_kc_upper:
                    score += 0.40
                    reasons.append("KC BREAKOUT UP (fresh)")
                else:
                    # Continuation above KC
                    score += 0.20
                    reasons.append("Above KC upper (continuation)")
            
            # BEARISH: Price breaks below KC lower
            elif price < kc_lower:
                if prev["close"] >= prev_kc_lower:
                    score -= 0.40
                    reasons.append("KC BREAKOUT DOWN (fresh)")
                else:
                    score -= 0.20
                    reasons.append("Below KC lower (continuation)")
            
            # Price near KC boundary (anticipatory entry)
            elif kc_upper > 0:
                dist_to_upper = (kc_upper - price) / price
                dist_to_lower = (price - kc_lower) / price
                if dist_to_upper < 0.002:  # Within 0.2% of upper
                    score += 0.15
                    reasons.append("Approaching KC upper")
                elif dist_to_lower < 0.002:
                    score -= 0.15
                    reasons.append("Approaching KC lower")

        # ── MACD Momentum ─────────────────────────────────────
        macd_hist = latest.get("MACDh_8_21_5")
        prev_macd_hist = prev.get("MACDh_8_21_5")
        prev2_macd_hist = prev2.get("MACDh_8_21_5")

        if macd_hist is not None and prev_macd_hist is not None:
            # Histogram increasing = momentum building
            if macd_hist > 0 and macd_hist > prev_macd_hist:
                score += 0.20
                reasons.append("MACD momentum building (bull)")
            elif macd_hist < 0 and macd_hist < prev_macd_hist:
                score -= 0.20
                reasons.append("MACD momentum building (bear)")
            
            # Histogram flip (momentum change)
            if prev_macd_hist <= 0 and macd_hist > 0:
                score += 0.15
                reasons.append("MACD histogram flip bullish")
            elif prev_macd_hist >= 0 and macd_hist < 0:
                score -= 0.15
                reasons.append("MACD histogram flip bearish")

        # ── EMA Trend Alignment ───────────────────────────────
        ema_8 = latest.get("ema_8")
        ema_21 = latest.get("ema_21")
        ema_55 = latest.get("ema_55")

        if ema_8 is not None and ema_21 is not None and ema_55 is not None:
            # Perfect bullish alignment: price > ema8 > ema21 > ema55
            if price > ema_8 > ema_21 > ema_55:
                score += 0.15
                reasons.append("Perfect bullish EMA alignment")
            elif price < ema_8 < ema_21 < ema_55:
                score -= 0.15
                reasons.append("Perfect bearish EMA alignment")
            elif ema_8 > ema_21:
                score += 0.05
                reasons.append("Short-term bullish")
            elif ema_8 < ema_21:
                score -= 0.05
                reasons.append("Short-term bearish")

        # ── Volume Confirmation ───────────────────────────────
        vol_ratio = latest.get("vol_ratio", 1.0)
        if vol_ratio is not None and not pd.isna(vol_ratio):
            if vol_ratio > self.vol_threshold:
                # Volume surge — amplify signal
                amp = min(1.0 + (vol_ratio - 1.0) * 0.3, 1.5)
                score *= amp
                reasons.append(f"Volume surge ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.5:
                # Low volume — reduce confidence
                score *= 0.6
                reasons.append(f"Low volume ({vol_ratio:.1f}x)")

        # ── ADX Strength Amplifier ────────────────────────────
        adx_val = latest.get("ADX_14", 0)
        if adx_val is not None and not pd.isna(adx_val):
            if adx_val > 30:
                score *= 1.3  # Strong trend amplification
                reasons.append(f"Strong trend (ADX={adx_val:.0f})")
            elif adx_val > 25:
                score *= 1.1
                reasons.append(f"Moderate trend (ADX={adx_val:.0f})")
            elif adx_val < self.min_adx:
                score *= 0.5
                reasons.append(f"Weak trend (ADX={adx_val:.0f})")

        # ── Rate of Change confirmation ──────────────────────
        roc_5 = latest.get("roc_5", 0)
        if roc_5 is not None and not pd.isna(roc_5):
            if score > 0 and roc_5 > 0.5:
                score += 0.10
                reasons.append(f"Positive momentum ROC={roc_5:.2f}%")
            elif score < 0 and roc_5 < -0.5:
                score -= 0.10
                reasons.append(f"Negative momentum ROC={roc_5:.2f}%")

        # ── Candle Bias Filter (anti-fakeout) ─────────────────
        # Require candle direction to agree with signal
        if score > 0 and candle_bias < 0:
            # Bullish signal but bearish candles — likely fakeout
            score *= 0.3
            reasons.append(f"Candle bias bearish ({candle_bias}), reducing bull signal")
        elif score < 0 and candle_bias > 0:
            score *= 0.3
            reasons.append(f"Candle bias bullish ({candle_bias}), reducing bear signal")
        elif score > 0 and candle_bias >= 2:
            score *= 1.2
            reasons.append(f"Strong bullish candle bias (+{candle_bias})")
        elif score < 0 and candle_bias <= -2:
            score *= 1.2
            reasons.append(f"Strong bearish candle bias ({candle_bias})")

        # ── Generate Signal ───────────────────────────────────
        strength = min(abs(score), 1.0)
        
        if score > 0.30:
            action = SignalType.BUY
        elif score < -0.30:
            action = SignalType.SELL
        else:
            action = SignalType.HOLD

        # Calculate stops using adaptive ATR (use faster ATR for tighter response)
        atr = latest.get("atr")
        atr_fast = latest.get("atr_fast")
        stop_loss = None
        take_profit = None
        
        if atr is not None and not pd.isna(atr) and action != SignalType.HOLD:
            # Use the tighter of fast and standard ATR
            effective_atr = min(atr_fast, atr) if (atr_fast is not None and not pd.isna(atr_fast)) else atr
            
            sl_distance = effective_atr * self.atr_sl_mult
            tp_distance = effective_atr * self.atr_tp_mult
            
            # Cap SL at max 1.2% of price to limit downside with leverage
            max_sl_distance = price * 0.012
            sl_distance = min(sl_distance, max_sl_distance)
            
            if action == SignalType.BUY:
                stop_loss = price - sl_distance
                take_profit = price + tp_distance
            else:
                stop_loss = price + sl_distance
                take_profit = price - tp_distance

        signal = TradingSignal(
            action=action,
            strength=strength,
            strategy_name=self.name,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=" | ".join(reasons),
            regime=regime,
            urgency=0.8 if "fresh" in " ".join(reasons).lower() else 0.5,
        )

        logger.debug(f"[{self.name}] Signal: {signal.action.value} (str={signal.strength:.2f}, score={score:+.3f})")
        return signal

    def _hold(self, price, reason):
        return TradingSignal(
            action=SignalType.HOLD, strength=0.0,
            strategy_name=self.name, price=price, reason=reason
        )

    def get_parameters(self) -> dict:
        return {
            "min_adx": self.min_adx,
            "vol_threshold": self.vol_threshold,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
        }
