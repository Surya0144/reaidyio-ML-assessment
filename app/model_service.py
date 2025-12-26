import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger("api_logger")

class ModelService:
    def __init__(self):
        self.model = None
        self.class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

    def load_model(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = joblib.load(path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model: {e}")

    def predict(self, features: list):
        """Makes a prediction based on input features."""
        if not self.model:
            raise RuntimeError("Model is not loaded")
        
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        confidence = max(probabilities)

        return {
            "class_id": int(prediction),
            "class_name": self.class_names.get(int(prediction), "unknown"),
            "confidence_score": float(confidence)
        }

ml_service = ModelService()