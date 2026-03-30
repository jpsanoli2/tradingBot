"""
Trading Bot - AI Price Predictor
LSTM neural network for predicting price direction.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from config.settings import BASE_DIR
from config import settings
from ai.feature_engineer import FeatureEngineer

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class PricePredictor:
    """
    LSTM-based price direction predictor.
    Predicts whether price will go UP or DOWN in the next period.
    """

    def __init__(self, lookback: int = None):
        self.lookback = lookback or settings.ai.LOOKBACK_PERIOD
        self.feature_engineer = FeatureEngineer(lookback=self.lookback)
        self.model = None
        self.model_path = BASE_DIR / "models" / "saved" / "price_predictor.keras"
        self.scaler_path = self.model_path.parent / "scaler.joblib"
        self.history = None
        self._tf_imported = False

    def _import_tf(self):
        """Lazy import TensorFlow to speed up startup."""
        if not self._tf_imported:
            global tf, Sequential, LSTM, Dense, Dropout, EarlyStopping, ModelCheckpoint
            import tensorflow as tf_module
            tf = tf_module
            from tensorflow.keras.models import Sequential as Seq
            from tensorflow.keras.layers import LSTM as L, Dense as D, Dropout as Dr
            from tensorflow.keras.callbacks import EarlyStopping as ES, ModelCheckpoint as MC
            Sequential = Seq
            LSTM = L
            Dense = D
            Dropout = Dr
            EarlyStopping = ES
            ModelCheckpoint = MC
            self._tf_imported = True

    def build_model(self, input_shape: tuple) -> None:
        """Build the LSTM model architecture."""
        self._import_tf()

        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(f"Model built: {self.model.count_params()} parameters")

    def train(self, df: pd.DataFrame, epochs: int = 50, 
              validation_split: float = 0.2) -> dict:
        """
        Train the model on historical data.
        
        Returns:
            Training metrics dict
        """
        X, y = self.feature_engineer.prepare_features(df)
        if X is None:
            logger.error("Failed to prepare features for training")
            return {"error": "Insufficient data"}

        # Build model if needed
        if self.model is None:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))

        # Callbacks
        self._import_tf()
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=10,
                restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                str(self.model_path), monitor="val_accuracy",
                save_best_only=True, verbose=1
            ),
        ]

        # Train
        logger.info(f"Training model: {len(X)} samples, {epochs} epochs")
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        # Get final metrics
        val_loss = min(self.history.history["val_loss"])
        val_acc = max(self.history.history["val_accuracy"])

        metrics = {
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "epochs_trained": len(self.history.history["loss"]),
            "total_samples": len(X),
        }

        logger.info(f"Training complete: val_accuracy={val_acc:.4f}, val_loss={val_loss:.4f}")
        
        # Save results
        self.save_model()
        return metrics

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict price direction for the next period.
        
        Returns:
            dict with prediction, confidence, and direction
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                logger.warning("No trained model available")
                return {"direction": "neutral", "confidence": 0.0, "prediction": 0.5}

        X = self.feature_engineer.prepare_prediction_input(df)
        if X is None:
            return {"direction": "neutral", "confidence": 0.0, "prediction": 0.5}

        prediction = self.model.predict(X, verbose=0)[0][0]

        if prediction > 0.5:
            direction = "up"
            confidence = prediction
        else:
            direction = "down"
            confidence = 1 - prediction

        result = {
            "direction": direction,
            "confidence": round(float(confidence), 4),
            "prediction": round(float(prediction), 4),
        }

        logger.info(f"AI Prediction: {direction} (confidence: {confidence:.2%})")
        return result

    def load_model(self) -> bool:
        """Load a previously saved model."""
        self._import_tf()
        if self.model_path.exists():
            try:
                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info(f"Model loaded from {self.model_path}")
                
                # Load scaler
                if self.scaler_path.exists():
                    import joblib
                    self.feature_engineer.scaler = joblib.load(str(self.scaler_path))
                    self.feature_engineer._fitted = True
                    logger.info(f"Scaler loaded from {self.scaler_path}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        return False

    def save_model(self) -> bool:
        """Save the current model."""
        if self.model is None:
            return False
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.model_path))
            logger.info(f"Model saved to {self.model_path}")
            
            # Save scaler
            import joblib
            joblib.dump(self.feature_engineer.scaler, str(self.scaler_path))
            logger.info(f"Scaler saved to {self.scaler_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
