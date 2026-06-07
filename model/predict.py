"""
Prediction engine for underwater mine detection.
Loads trained model and provides prediction with confidence scores.
"""
import numpy as np
import joblib
import os
import json


class MineDetector:
    """Mine detection prediction engine."""
    
    def __init__(self):
        """Load trained model and scaler."""
        model_dir = os.path.dirname(__file__)
        
        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Run train_model.py first."
            )
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
    
    def predict(self, features):
        """
        Predict mine or rock from sonar features.
        
        Args:
            features: numpy array of shape (60,) or (n, 60)
        
        Returns:
            dict with prediction, confidence, and probabilities
        """
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions and probabilities
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        results = []
        for i in range(len(predictions)):
            pred_label = "Mine" if predictions[i] == 1 else "Rock"
            confidence = float(max(probabilities[i])) * 100
            
            results.append({
                "prediction": pred_label,
                "prediction_code": int(predictions[i]),
                "confidence": round(confidence, 2),
                "probability_rock": round(float(probabilities[i][0]) * 100, 2),
                "probability_mine": round(float(probabilities[i][1]) * 100, 2)
            })
        
        return results[0] if len(results) == 1 else results
    
    def predict_random_sample(self, df):
        """
        Pick a random sample from dataset and predict.
        
        Args:
            df: pandas DataFrame with sonar features
        
        Returns:
            tuple of (sample_features, prediction_result)
        """
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        sample = df.sample(1)
        features = sample[feature_cols].values[0]
        result = self.predict(features)
        return features, result
    
    def get_feature_importance(self):
        """Get feature importance from the Random Forest model."""
        importance = self.model.feature_importances_
        feature_names = self.metadata["feature_columns"]
        
        importance_dict = dict(zip(feature_names, importance))
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        return sorted_importance
