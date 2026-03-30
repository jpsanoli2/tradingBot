"""
Trading Bot - Technical Indicators (v2.0)
Advanced indicator suite with market regime detection and volume profiling.
"""

import pandas as pd
import numpy as np
import pandas_ta_classic as ta
from loguru import logger


class TechnicalIndicators:
    """Calculates a comprehensive set of technical indicators with regime detection."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators and add them as columns.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with indicator columns added
        """
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for indicators: {len(df)} rows")
            return df

        df = df.copy()

        # ── Trend Indicators ──────────────────────────────────
        # EMAs for trend structure
        df["ema_8"] = ta.ema(df["close"], length=8)
        df["ema_13"] = ta.ema(df["close"], length=13)
        df["ema_21"] = ta.ema(df["close"], length=21)
        df["ema_55"] = ta.ema(df["close"], length=55)
        df["ema_100"] = ta.ema(df["close"], length=100)

        # MACD (fast momentum)
        macd = ta.macd(df["close"], fast=8, slow=21, signal=5)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)

        # ADX - Trend strength (critical for regime detection)
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None:
            df = pd.concat([df, adx], axis=1)

        # ── Momentum Indicators ───────────────────────────────
        # RSI
        df["rsi"] = ta.rsi(df["close"], length=14)
        
        # RSI rate of change (momentum of momentum)
        df["rsi_roc"] = df["rsi"].diff(3)

        # Stochastic RSI
        stoch_rsi = ta.stochrsi(df["close"], length=14)
        if stoch_rsi is not None:
            df = pd.concat([df, stoch_rsi], axis=1)

        # Williams %R (shorter-term momentum)
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        # ── Volatility Indicators ─────────────────────────────
        # Bollinger Bands (standard)
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)

        # Keltner Channels (for breakout detection)
        keltner = TechnicalIndicators._keltner_channels(df, length=20, mult=1.5)
        df = pd.concat([df, keltner], axis=1)

        # ATR (multiple periods)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_fast"] = ta.atr(df["high"], df["low"], df["close"], length=7)

        # ATR as percentage of price (normalized volatility)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100

        # ── Volume Indicators ─────────────────────────────────
        # OBV
        df["obv"] = ta.obv(df["close"], df["volume"])
        
        # Volume SMA (for volume surge detection)
        df["vol_sma_20"] = ta.sma(df["volume"], length=20)
        df["vol_ratio"] = df["volume"] / df["vol_sma_20"]

        # VWAP (if intraday)
        try:
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        except Exception:
            # Calculate simple VWAP approximation
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3

        # ── Market Regime Detection ───────────────────────────
        df = TechnicalIndicators._detect_regime(df)

        # ── Price Action Features ─────────────────────────────
        # Candle body ratio
        df["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
        
        # Upper/Lower wick ratio
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)

        # Price position within range
        df["range_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        # Momentum (rate of change)
        df["roc_5"] = ta.roc(df["close"], length=5)
        df["roc_10"] = ta.roc(df["close"], length=10)

        logger.debug(f"Calculated {len(df.columns)} indicators on {len(df)} candles")
        return df

    @staticmethod
    def _keltner_channels(df: pd.DataFrame, length: int = 20, mult: float = 1.5) -> pd.DataFrame:
        """Calculate Keltner Channels."""
        mid = ta.ema(df["close"], length=length)
        atr_val = ta.atr(df["high"], df["low"], df["close"], length=length)
        
        result = pd.DataFrame(index=df.index)
        result["KC_upper"] = mid + (atr_val * mult)
        result["KC_mid"] = mid
        result["KC_lower"] = mid - (atr_val * mult)
        result["KC_width"] = (result["KC_upper"] - result["KC_lower"]) / result["KC_mid"]
        return result

    @staticmethod
    def _detect_regime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime: trending, ranging, or volatile.
        Uses ADX + BB width + directional movement.
        """
        adx_col = "ADX_14"
        
        if adx_col not in df.columns:
            df["regime"] = "unknown"
            df["trend_direction"] = 0
            return df

        adx = df[adx_col]
        
        # Trend direction using EMA alignment
        ema_short = df.get("ema_21")
        ema_long = df.get("ema_55")
        
        trend_direction = pd.Series(0, index=df.index)
        if ema_short is not None and ema_long is not None:
            trend_direction = np.where(ema_short > ema_long, 1, np.where(ema_short < ema_long, -1, 0))
        
        df["trend_direction"] = trend_direction

        # Regime classification
        regimes = []
        for i in range(len(df)):
            adx_val = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0
            
            atr_pct = df["atr_pct"].iloc[i] if "atr_pct" in df.columns and not pd.isna(df["atr_pct"].iloc[i]) else 0
            
            if adx_val > 25 and atr_pct > 1.5:
                regimes.append("volatile_trend")
            elif adx_val > 20:
                regimes.append("trending")
            elif adx_val < 15:
                regimes.append("ranging")
            else:
                regimes.append("transitioning")
        
        df["regime"] = regimes
        
        return df

    @staticmethod
    def get_signal_summary(df: pd.DataFrame) -> dict:
        """
        Generate a summary of current indicator states.
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        summary = {}

        # Regime
        summary["regime"] = latest.get("regime", "unknown")
        summary["trend_direction"] = int(latest.get("trend_direction", 0))

        # RSI
        rsi_val = latest.get("rsi")
        if rsi_val is not None:
            summary["rsi"] = {
                "value": round(rsi_val, 2),
                "signal": "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral",
            }

        # MACD
        macd_val = latest.get("MACD_8_21_5")
        macd_signal = latest.get("MACDs_8_21_5")
        if macd_val is not None and macd_signal is not None:
            summary["macd"] = {
                "macd": round(macd_val, 4),
                "signal": round(macd_signal, 4),
                "histogram": round(latest.get("MACDh_8_21_5", 0), 4),
                "signal_type": "bullish" if macd_val > macd_signal else "bearish",
            }

        # ADX
        adx_val = latest.get("ADX_14")
        if adx_val is not None:
            summary["adx"] = {
                "value": round(adx_val, 2),
                "trend_strength": "strong" if adx_val > 25 else "moderate" if adx_val > 20 else "weak",
            }

        # Volume
        vol_ratio = latest.get("vol_ratio")
        if vol_ratio is not None:
            summary["volume"] = {
                "ratio": round(vol_ratio, 2),
                "surge": vol_ratio > 1.5,
            }

        # ATR volatility
        atr_pct = latest.get("atr_pct")
        if atr_pct is not None:
            summary["volatility"] = {
                "atr_pct": round(atr_pct, 3),
                "level": "high" if atr_pct > 2.0 else "normal" if atr_pct > 0.8 else "low",
            }

        return summary
