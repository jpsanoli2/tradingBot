"""
Trading Bot - VWAP Mean Reversion Strategy (v2.0)
High-frequency mean reversion scalping using VWAP + Bollinger + RSI.
Designed for RANGING markets (ADX < 25).
"""

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType


class MeanReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion strategy for ranging markets:
    - VWAP + EMA confluence zones as mean
    - Bollinger Band extremes for entries
    - RSI overbought/oversold confirmation
    - Stochastic RSI for precise timing
    - Tight stops, targets BB middle (quick scalps)
    
    Only activates in ranging/transitioning regimes.
    """

    def __init__(self):
        super().__init__("MeanReversion")
        self.rsi_oversold = 40  # Relaxed to get more trades
        self.rsi_overbought = 60
        self.bb_entry_pct = 0.85
        self.atr_sl_mult = 0.3  # Ultra-tight SL for scalps
        self.atr_tp_mult = 0.8  # Quick target

    def analyze(self, df: pd.DataFrame) -> TradingSignal:
        """Analyze for mean reversion opportunities."""
        if df.empty or len(df) < 30:
            return self._hold(0.0, "Insufficient data")

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 3 else prev
        price = latest["close"]
        regime = latest.get("regime", "unknown")

        # Only trade in strict ranging markets
        regime_ok = regime in ("ranging",)
        
        score = 0.0
        reasons = []

        # ── Bollinger Band Position ───────────────────────────
        bb_upper = latest.get("BBU_20_2.0")
        bb_lower = latest.get("BBL_20_2.0")
        bb_mid = latest.get("BBM_20_2.0")

        if bb_upper is not None and bb_lower is not None and bb_mid is not None:
            bb_width = bb_upper - bb_lower
            bb_position = (price - bb_lower) / bb_width if bb_width > 0 else 0.5

            # OVERSOLD: Price near/below lower BB
            if bb_position < 0.05:
                score += 0.35
                reasons.append(f"Price at BB lower extreme ({bb_position:.1%})")
            elif bb_position < 0.15:
                score += 0.25
                reasons.append(f"Price near BB lower ({bb_position:.1%})")
            elif bb_position < 0.30:
                score += 0.10
                reasons.append(f"Price in lower BB zone ({bb_position:.1%})")

            # OVERBOUGHT: Price near/above upper BB
            elif bb_position > 0.95:
                score -= 0.35
                reasons.append(f"Price at BB upper extreme ({bb_position:.1%})")
            elif bb_position > 0.85:
                score -= 0.25
                reasons.append(f"Price near BB upper ({bb_position:.1%})")
            elif bb_position > 0.70:
                score -= 0.10
                reasons.append(f"Price in upper BB zone ({bb_position:.1%})")

            # BB width squeeze = upcoming volatility, reduce MR signals  
            bb_pct_width = bb_width / bb_mid if bb_mid > 0 else 0
            if bb_pct_width < 0.02:
                score *= 0.3
                reasons.append("BB squeeze — low confidence for MR")

        # ── VWAP Confluence ───────────────────────────────────
        vwap = latest.get("vwap")
        if vwap is not None and not pd.isna(vwap):
            vwap_dist = (price - vwap) / vwap * 100  # Distance from VWAP in %

            if vwap_dist < -0.5:
                score += 0.15
                reasons.append(f"Below VWAP ({vwap_dist:.2f}%)")
            elif vwap_dist > 0.5:
                score -= 0.15
                reasons.append(f"Above VWAP ({vwap_dist:+.2f}%)")

        # ── RSI Confirmation ──────────────────────────────────
        rsi = latest.get("rsi")
        prev_rsi = prev.get("rsi")

        if rsi is not None and not pd.isna(rsi):
            if rsi < self.rsi_oversold:
                score += 0.20
                reasons.append(f"RSI oversold ({rsi:.1f})")
                # RSI turning up (stronger reversal signal)
                if prev_rsi is not None and rsi > prev_rsi:
                    score += 0.10
                    reasons.append("RSI turning up ↑")
            elif rsi > self.rsi_overbought:
                score -= 0.20
                reasons.append(f"RSI overbought ({rsi:.1f})")
                if prev_rsi is not None and rsi < prev_rsi:
                    score -= 0.10
                    reasons.append("RSI turning down ↓")
            
            # RSI divergence check (price making new lows but RSI not)
            if len(df) > 5:
                price_5_min = df["close"].iloc[-5:].min()
                rsi_5_min = df["rsi"].iloc[-5:].min() if "rsi" in df.columns else None
                
                if rsi_5_min is not None and not pd.isna(rsi_5_min):
                    if price <= price_5_min and rsi > rsi_5_min:
                        score += 0.15
                        reasons.append("Bullish RSI divergence")

        # ── Stochastic RSI Timing ─────────────────────────────
        stoch_k = latest.get("STOCHRSIk_14_14_3_3")
        stoch_d = latest.get("STOCHRSId_14_14_3_3")
        prev_stoch_k = prev.get("STOCHRSIk_14_14_3_3")

        if stoch_k is not None and stoch_d is not None:
            if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                # Oversold + crossing up
                if stoch_k < 20:
                    score += 0.10
                    if prev_stoch_k is not None and not pd.isna(prev_stoch_k):
                        if stoch_k > prev_stoch_k:
                            score += 0.10
                            reasons.append("StochRSI turning up from oversold")
                
                # Overbought + crossing down
                elif stoch_k > 80:
                    score -= 0.10
                    if prev_stoch_k is not None and not pd.isna(prev_stoch_k):
                        if stoch_k < prev_stoch_k:
                            score -= 0.10
                            reasons.append("StochRSI turning down from overbought")

        # ── Williams %R ───────────────────────────────────────
        willr = latest.get("willr")
        if willr is not None and not pd.isna(willr):
            if willr < -80:
                score += 0.10
                reasons.append(f"Williams %R oversold ({willr:.0f})")
            elif willr > -20:
                score -= 0.10
                reasons.append(f"Williams %R overbought ({willr:.0f})")

        # ── Candle Pattern Confirmation ───────────────────────
        body_ratio = latest.get("body_ratio", 0)
        lower_wick = latest.get("lower_wick", 0)
        upper_wick = latest.get("upper_wick", 0)

        if not pd.isna(body_ratio):
            # Hammer/pin bar at bottom
            if score > 0 and lower_wick > 0.6:
                score += 0.10
                reasons.append("Hammer candle pattern")
            # Shooting star at top
            elif score < 0 and upper_wick > 0.6:
                score -= 0.10
                reasons.append("Shooting star pattern")

        # ── Regime Penalty ────────────────────────────────────
        if not regime_ok:
            score *= 0.4
            reasons.append(f"Regime penalty ({regime})")

        # ── Generate Signal ───────────────────────────────────
        strength = min(abs(score), 1.0)
        
        if score > 0.30:
            action = SignalType.BUY
        elif score < -0.30:
            action = SignalType.SELL
        else:
            action = SignalType.HOLD

        # Stops based on ATR + BB
        atr = latest.get("atr")
        stop_loss = None
        take_profit = None
        
        if atr is not None and not pd.isna(atr) and action != SignalType.HOLD:
            if action == SignalType.BUY:
                stop_loss = price - (atr * self.atr_sl_mult)
                # Target: BB middle or ATR*1.5, whichever is closer
                tp_bb = bb_mid if bb_mid is not None and not pd.isna(bb_mid) else price + (atr * self.atr_tp_mult)
                tp_atr = price + (atr * self.atr_tp_mult)
                take_profit = min(tp_bb, tp_atr) if tp_bb > price else tp_atr
            else:
                stop_loss = price + (atr * self.atr_sl_mult)
                tp_bb = bb_mid if bb_mid is not None and not pd.isna(bb_mid) else price - (atr * self.atr_tp_mult)
                tp_atr = price - (atr * self.atr_tp_mult)
                take_profit = max(tp_bb, tp_atr) if tp_bb < price else tp_atr

        signal = TradingSignal(
            action=action,
            strength=strength,
            strategy_name=self.name,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=" | ".join(reasons),
            regime=regime,
            urgency=0.6 if "turning" in " ".join(reasons).lower() else 0.4,
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
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "bb_entry_pct": self.bb_entry_pct,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
        }
