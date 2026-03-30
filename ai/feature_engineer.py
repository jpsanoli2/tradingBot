"""
Trading Bot - Feature Engineer
Prepares features from OHLCV + indicators for ML model input.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    """Transforms raw market data into ML-ready features."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self._fitted = False

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare feature matrix from OHLCV + indicators.
        
        Returns:
            (X, y, scaler) where X is the feature matrix, y is the target
        """
        if df.empty or len(df) < self.lookback + 10:
            logger.warning("Insufficient data for feature engineering")
            return None, None

        # Select feature columns
        feature_cols = self._get_feature_columns(df)
        data = df[feature_cols].copy()

        # Drop rows with NaN
        data = data.dropna()

        if len(data) < self.lookback + 10:
            logger.warning(f"After cleaning, only {len(data)} rows remain")
            return None, None

        # Create target: 1 if price goes up next period, 0 if down
        data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)
        data = data.dropna()

        # Scale features
        feature_data = data[feature_cols].values
        self.scaler.fit(feature_data)
        self._fitted = True
        scaled_data = self.scaler.transform(feature_data)

        # Create sequences (sliding windows)
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i])
            y.append(data["target"].iloc[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Features prepared: X shape={X.shape}, y shape={y.shape}")
        return X, y

    def prepare_prediction_input(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare the latest data window for prediction."""
        if not self._fitted:
            logger.warning("Scaler not fitted yet, call prepare_features first")
            return None

        feature_cols = self._get_feature_columns(df)
        data = df[feature_cols].dropna()

        if len(data) < self.lookback:
            logger.warning(f"Need {self.lookback} rows, got {len(data)}")
            return None

        # Take the last lookback rows
        recent_data = data.tail(self.lookback).values
        scaled = self.scaler.transform(recent_data)
        
        # Reshape for LSTM: (1, lookback, features)
        return scaled.reshape(1, self.lookback, -1)

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Select available feature columns from the DataFrame."""
        desired_cols = [
            "open", "high", "low", "close", "volume",
            "ema_9", "ema_21", "ema_50",
            "rsi", "atr", "obv",
            "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
            "BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0",
            "ADX_14",
        ]
        return [col for col in desired_cols if col in df.columns]
