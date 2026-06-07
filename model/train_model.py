"""
Train Random Forest model for underwater mine detection.
Trains on the Sonar dataset and saves the model to model/model.pkl
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json


def load_and_preprocess_data(data_path=None):
    """Load sonar dataset and preprocess it."""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sonar.csv")
    
    df = pd.read_csv(data_path)
    
    # Encode labels: R -> 0, M -> 1
    df["label_encoded"] = df["label"].map({"R": 0, "M": 1})
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    X = df[feature_cols].values
    y = df["label_encoded"].values
    
    return X, y, feature_cols, df


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier with optimized hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1 Score:  {metrics['f1_score']}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Rock', 'Mine'])}")
    
    return metrics


def save_model(model, scaler, feature_cols, metrics):
    """Save trained model and metadata."""
    model_dir = os.path.dirname(__file__)
    
    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "metrics": metrics
    }
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline."""
    print("=" * 50)
    print("UNDERWATER MINE DETECTION - MODEL TRAINING")
    print("=" * 50)
    
    # Load data
    X, y, feature_cols, df = load_and_preprocess_data()
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Rock={sum(y==0)}, Mine={sum(y==1)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining Random Forest model...")
    model = train_random_forest(X_train_scaled, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Save
    save_model(model, scaler, feature_cols, metrics)
    
    print("\n✓ Training complete!")
    return model, scaler, metrics


if __name__ == "__main__":
    main()
