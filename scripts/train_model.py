"""
VoxGuard Model Training Script
Trains the deepfake detection model using prepared features
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ML imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    import joblib
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn joblib")
    sys.exit(1)


def load_data(data_dir: Path) -> tuple:
    """Load prepared features and labels."""
    features_path = data_dir / "features.npy"
    labels_path = data_dir / "labels.npy"
    
    if not features_path.exists() or not labels_path.exists():
        print(f"Error: Data not found in {data_dir}")
        print("Please run prepare_dataset.py first")
        sys.exit(1)
    
    X = np.load(features_path)
    y = np.load(labels_path)
    
    return X, y


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "gradient_boosting"
) -> object:
    """
    Train the classification model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        
    Returns:
        Trained model
    """
    print(f"\nTraining {model_type} model...")
    
    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=1
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            verbose=1,
            n_jobs=-1
        )
    elif model_type == "svm":
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None
) -> dict:
    """
    Evaluate the trained model.
    
    Returns dictionary of metrics.
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    # Cross-validation if training data provided
    if X_train is not None and y_train is not None:
        X_full = np.vstack([X_train, X_test])
        y_full = np.hstack([y_train, y_test])
        
        cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def print_metrics(metrics: dict):
    """Print formatted metrics."""
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    if "cv_accuracy_mean" in metrics:
        print(f"\nCross-Validation Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
    print("=" * 50)


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description="Train VoxGuard deepfake detector")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed features"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gradient_boosting", "random_forest", "svm"],
        default="gradient_boosting",
        help="Type of model to train"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save metrics reports"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    reports_dir = Path(args.reports_dir)
    
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X, y = load_data(data_dir)
    print(f"Loaded {len(y)} samples with {X.shape[1]} features")
    print(f"  AI-generated: {np.sum(y == 1)}")
    print(f"  Human: {np.sum(y == 0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set:  {len(y_test)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train, args.model_type)
    
    # Evaluate model
    metrics = evaluate_model(
        model, X_test_scaled, y_test,
        X_train_scaled, y_train
    )
    print_metrics(metrics)
    
    # Save model and scaler
    model_path = model_dir / "ai_detector.pkl"
    scaler_path = model_dir / "scaler.pkl"
    
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    # Save metrics report
    report = {
        "model_type": args.model_type,
        "training_date": datetime.now().isoformat(),
        "num_training_samples": len(y_train),
        "num_test_samples": len(y_test),
        "feature_dimension": X.shape[1],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "cv_accuracy_mean": metrics.get("cv_accuracy_mean", None),
            "cv_accuracy_std": metrics.get("cv_accuracy_std", None)
        }
    }
    
    report_path = reports_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nMetrics saved to {report_path}")
    print("\n✅ Training complete!")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")


if __name__ == "__main__":
    main()
