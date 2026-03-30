"""
Trading Bot - Swing Trend Continuation Strategy (v2.0)
Enters on pullbacks within established trends for high R:R trades.
"""

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType


class SwingContinuationStrategy(BaseStrategy):
    """
    Swing continuation strategy:
    - Waits for established trend (EMA alignment + ADX > 20)
    - Enters on pullbacks to EMA 21 or EMA 55 support
    - RSI dip to 40-55 zone in uptrend (45-60 in downtrend)
    - Wider stops, larger targets (2.5:1 R:R)
    - Fewer trades, higher quality
    """

    def __init__(self):
        super().__init__("SwingContinuation")
        self.pullback_rsi_bull = (38, 55)  # RSI zone for pullback in uptrend
        self.pullback_rsi_bear = (45, 62)  # RSI zone for pullback in downtrend
        self.min_adx = 20
        self.atr_sl_mult = 0.7  # Tighter SL for leveraged trading
        self.atr_tp_mult = 1.8  # Target with good R:R

    def analyze(self, df: pd.DataFrame) -> TradingSignal:
        """Analyze for trend continuation after pullback."""
        if df.empty or len(df) < 60:
            return self._hold(0.0, "Insufficient data")

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 3 else prev
        price = latest["close"]
        regime = latest.get("regime", "unknown")

        # Only trade in trending or volatile_trend regimes
        if regime in ("ranging",):
            return self._hold(price, f"Wrong regime for swing: {regime}")

        score = 0.0
        reasons = []

        # ── Trend Identification ──────────────────────────────
        ema_8 = latest.get("ema_8")
        ema_21 = latest.get("ema_21")
        ema_55 = latest.get("ema_55")
        ema_100 = latest.get("ema_100")
        trend_dir = latest.get("trend_direction", 0)

        if ema_21 is None or ema_55 is None:
            return self._hold(price, "Missing EMAs")

        is_uptrend = ema_21 > ema_55
        is_downtrend = ema_21 < ema_55

        # Need clear trend
        if not is_uptrend and not is_downtrend:
            return self._hold(price, "No clear trend direction")

        # ── ADX Confirmation ──────────────────────────────────
        adx_val = latest.get("ADX_14", 0)
        if adx_val is not None and not pd.isna(adx_val):
            if adx_val < self.min_adx:
                return self._hold(price, f"ADX too weak ({adx_val:.0f})")

        # ── Pullback Detection ────────────────────────────────
        rsi = latest.get("rsi")
        prev_rsi = prev.get("rsi")
        rsi_roc = latest.get("rsi_roc")

        if rsi is None or pd.isna(rsi):
            return self._hold(price, "No RSI data")

        if is_uptrend:
            # BULLISH PULLBACK: Price has dipped, now recovering
            # Price should have pulled back to EMA 21 or EMA 55
            pulled_back_to_ema21 = price <= ema_21 * 1.005  # Within 0.5% of EMA 21
            pulled_back_to_ema55 = price <= ema_55 * 1.005
            near_ema = pulled_back_to_ema21 or pulled_back_to_ema55

            rsi_in_zone = self.pullback_rsi_bull[0] <= rsi <= self.pullback_rsi_bull[1]
            rsi_turning_up = prev_rsi is not None and not pd.isna(prev_rsi) and rsi > prev_rsi

            if near_ema and rsi_in_zone:
                score += 0.35
                reasons.append("Pullback to EMA in uptrend + RSI in zone")
                
                if rsi_turning_up:
                    score += 0.20
                    reasons.append("RSI turning up (momentum returning)")
                
                if pulled_back_to_ema55:
                    score += 0.10
                    reasons.append("Deep pullback to EMA 55 (stronger support)")

            elif rsi_in_zone and rsi_turning_up:
                # RSI pullback without reaching EMA (still valid but weaker)
                ema_dist = (price - ema_21) / ema_21 * 100
                if ema_dist < 1.0:  # Within 1% of EMA 21
                    score += 0.25
                    reasons.append(f"RSI pullback near EMA 21 ({ema_dist:.1f}%)")
            
            # Check EMA 100 as ultimate support
            if ema_100 is not None and not pd.isna(ema_100):
                if price >= ema_100:
                    score += 0.05
                    reasons.append("Above EMA 100 (macro uptrend)")
                else:
                    score -= 0.30
                    reasons.append("Below EMA 100 (macro downtrend)")

        elif is_downtrend:
            # BEARISH PULLBACK: Price has bounced up, now falling again
            pulled_back_to_ema21 = price >= ema_21 * 0.995
            pulled_back_to_ema55 = price >= ema_55 * 0.995
            near_ema = pulled_back_to_ema21 or pulled_back_to_ema55

            rsi_in_zone = self.pullback_rsi_bear[0] <= rsi <= self.pullback_rsi_bear[1]
            rsi_turning_down = prev_rsi is not None and not pd.isna(prev_rsi) and rsi < prev_rsi

            if near_ema and rsi_in_zone:
                score -= 0.35
                reasons.append("Pullback to EMA in downtrend + RSI in zone")
                
                if rsi_turning_down:
                    score -= 0.20
                    reasons.append("RSI turning down (momentum returning)")
                
                if pulled_back_to_ema55:
                    score -= 0.10
                    reasons.append("Deep pullback to EMA 55 (stronger resistance)")

            elif rsi_in_zone and rsi_turning_down:
                ema_dist = (ema_21 - price) / ema_21 * 100
                if ema_dist < 1.0:
                    score -= 0.25
                    reasons.append(f"RSI pullback near EMA 21 ({ema_dist:.1f}%)")

            if ema_100 is not None and not pd.isna(ema_100):
                if price <= ema_100:
                    score -= 0.05
                    reasons.append("Below EMA 100 (macro downtrend confirmed)")
                else:
                    score += 0.30
                    reasons.append("Above EMA 100 (macro uptrend, don't short)")

        # ── MACD Momentum Check ───────────────────────────────
        macd_hist = latest.get("MACDh_8_21_5")
        prev_macd_hist = prev.get("MACDh_8_21_5")

        if macd_hist is not None and prev_macd_hist is not None:
            if not pd.isna(macd_hist) and not pd.isna(prev_macd_hist):
                if score > 0 and macd_hist > prev_macd_hist:
                    score += 0.10
                    reasons.append("MACD momentum recovering")
                elif score < 0 and macd_hist < prev_macd_hist:
                    score -= 0.10
                    reasons.append("MACD momentum falling")

        # ── Volume Filter ─────────────────────────────────────
        vol_ratio = latest.get("vol_ratio", 1.0)
        if vol_ratio is not None and not pd.isna(vol_ratio):
            if vol_ratio < 0.3:
                score *= 0.5
                reasons.append("Very low volume, reducing confidence")

        # ── ADX Amplifier ─────────────────────────────────────
        if adx_val is not None and not pd.isna(adx_val):
            if adx_val > 30:
                score *= 1.25
                reasons.append(f"Strong trend (ADX={adx_val:.0f})")

        # ── Generate Signal ───────────────────────────────────
        strength = min(abs(score), 1.0)
        
        if score > 0.30:
            action = SignalType.BUY
        elif score < -0.30:
            action = SignalType.SELL
        else:
            action = SignalType.HOLD

        # Wider stops for swing trades
        atr = latest.get("atr")
        stop_loss = None
        take_profit = None
        
        if atr is not None and not pd.isna(atr) and action != SignalType.HOLD:
            if action == SignalType.BUY:
                # SL below the pullback low or ATR-based
                recent_low = df["low"].iloc[-5:].min()
                sl_atr = price - (atr * self.atr_sl_mult)
                stop_loss = min(sl_atr, recent_low - atr * 0.2)
                take_profit = price + (atr * self.atr_tp_mult)
            else:
                recent_high = df["high"].iloc[-5:].max()
                sl_atr = price + (atr * self.atr_sl_mult)
                stop_loss = max(sl_atr, recent_high + atr * 0.2)
                take_profit = price - (atr * self.atr_tp_mult)

        signal = TradingSignal(
            action=action,
            strength=strength,
            strategy_name=self.name,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=" | ".join(reasons),
            regime=regime,
            urgency=0.3,  # Swing trades are less urgent
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
            "pullback_rsi_bull": self.pullback_rsi_bull,
            "pullback_rsi_bear": self.pullback_rsi_bear,
            "min_adx": self.min_adx,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
        }
