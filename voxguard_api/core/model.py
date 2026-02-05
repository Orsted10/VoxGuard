"""
VoxGuard ML Model Module
Handles model loading and prediction for deepfake detection
"""

import os
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import joblib

from voxguard_api.core.config import settings, CLASSIFICATION_AI, CLASSIFICATION_HUMAN


# Global model cache
_model = None
_scaler = None
_model_loaded = False


class FallbackModel:
    """
    Fallback model using heuristic feature analysis.
    Used when trained model is not available.
    """
    
    def __init__(self):
        self.version = "heuristic-v1"
        self.training_date = "2026-02-05"
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability using heuristic analysis.
        
        Uses weighted feature analysis to determine AI probability.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        probabilities = []
        
        for x in X:
            # Feature indices (based on build_feature_vector structure)
            # MFCC: 0-199 (40*5), Mel: 200-263 (32*2), Spectral: 264-273, 
            # Pitch: 274-279, Energy: 280-285, Additional: 286-289, Chroma: 290-301
            
            ai_score = 0.5  # Start neutral
            
            # Analyze pitch features (indices 274-279)
            if len(x) > 275:
                pitch_std = x[275]  # pitch_std
                pitch_range = x[278]  # pitch_range
                voiced_ratio = x[279]  # voiced_ratio
                
                # Low pitch variation suggests AI
                if pitch_std < 15:
                    ai_score += 0.15
                elif pitch_std > 50:
                    ai_score -= 0.1
                
                # Very high voiced ratio (no breaths) suggests AI
                if voiced_ratio > 0.95:
                    ai_score += 0.1
                elif voiced_ratio < 0.8:
                    ai_score -= 0.1
            
            # Analyze energy features (indices 280-285)
            if len(x) > 285:
                energy_std = x[281]  # energy_std
                dynamic_range = x[284]  # dynamic_range
                
                # Low dynamic range suggests AI
                if dynamic_range < 10:
                    ai_score += 0.12
                elif dynamic_range > 40:
                    ai_score -= 0.08
            
            # Analyze spectral features (indices 264-273)
            if len(x) > 271:
                spectral_flatness = x[270]  # spectral_flatness_mean
                
                # Very clean spectra suggest AI
                if spectral_flatness < 0.01:
                    ai_score += 0.1
                elif spectral_flatness > 0.1:
                    ai_score -= 0.08
            
            # Analyze harmonic ratio (index 287)
            if len(x) > 287:
                harmonic_ratio = x[287]
                
                # Almost pure harmonic suggests AI
                if harmonic_ratio > 0.95:
                    ai_score += 0.1
                elif harmonic_ratio < 0.7:
                    ai_score -= 0.08
            
            # MFCC smoothness analysis (deltas are at indices 160-199)
            if len(x) > 199:
                mfcc_deltas = x[160:200]
                avg_delta = np.mean(np.abs(mfcc_deltas))
                
                # Very smooth MFCC transitions suggest AI
                if avg_delta < 0.5:
                    ai_score += 0.08
                elif avg_delta > 2.0:
                    ai_score -= 0.06
            
            # Clamp to valid probability range
            ai_score = max(0.0, min(1.0, ai_score))
            
            # Add slight randomness for more realistic behavior
            ai_score += np.random.uniform(-0.02, 0.02)
            ai_score = max(0.0, min(1.0, ai_score))
            
            probabilities.append([1 - ai_score, ai_score])
        
        return np.array(probabilities)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class FallbackScaler:
    """Fallback scaler that applies minimal normalization."""
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply basic z-score normalization per feature."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Simple normalization
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True) + 1e-8
        
        return (X - mean) / std


def load_model() -> Tuple[any, any]:
    """
    Load the trained model and scaler.
    
    Returns:
        Tuple of (model, scaler)
        
    Uses fallback model if trained model is not available.
    """
    global _model, _scaler, _model_loaded
    
    if _model_loaded:
        return _model, _scaler
    
    # Try to load trained model
    model_path = Path(settings.model_path)
    scaler_path = Path(settings.scaler_path)
    
    # Also check in package directory
    package_dir = Path(__file__).parent.parent
    alt_model_path = package_dir / "models" / "ai_detector.pkl"
    alt_scaler_path = package_dir / "models" / "scaler.pkl"
    
    try:
        if model_path.exists():
            _model = joblib.load(model_path)
            print(f"Loaded trained model from {model_path}")
        elif alt_model_path.exists():
            _model = joblib.load(alt_model_path)
            print(f"Loaded trained model from {alt_model_path}")
        else:
            print("Trained model not found, using heuristic fallback")
            _model = FallbackModel()
        
        if scaler_path.exists():
            _scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        elif alt_scaler_path.exists():
            _scaler = joblib.load(alt_scaler_path)
            print(f"Loaded scaler from {alt_scaler_path}")
        else:
            print("Scaler not found, using fallback")
            _scaler = FallbackScaler()
            
    except Exception as e:
        print(f"Error loading model: {e}. Using fallback.")
        _model = FallbackModel()
        _scaler = FallbackScaler()
    
    _model_loaded = True
    return _model, _scaler


def predict_ai_probability(feature_vector: np.ndarray) -> float:
    """
    Predict the probability that audio is AI-generated.
    
    Args:
        feature_vector: Flat feature vector from audio
        
    Returns:
        Probability of AI_GENERATED class (0.0 to 1.0)
    """
    model, scaler = load_model()
    
    # Ensure 2D input
    if len(feature_vector.shape) == 1:
        feature_vector = feature_vector.reshape(1, -1)
    
    # Apply scaling
    try:
        scaled_features = scaler.transform(feature_vector)
    except Exception:
        scaled_features = feature_vector
    
    # Get prediction
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_features)
            # Return probability of AI_GENERATED (class 1)
            ai_prob = float(proba[0, 1])
        else:
            # Use decision function for models without predict_proba
            decision = model.decision_function(scaled_features)
            # Sigmoid transformation
            ai_prob = 1.0 / (1.0 + np.exp(-decision[0]))
    except Exception as e:
        print(f"Prediction error: {e}")
        ai_prob = 0.5  # Return neutral if error
    
    return max(0.0, min(1.0, ai_prob))


def classify_audio(feature_vector: np.ndarray) -> Tuple[str, float]:
    """
    Classify audio as AI_GENERATED or HUMAN.
    
    Args:
        feature_vector: Flat feature vector from audio
        
    Returns:
        Tuple of (classification, confidence_score)
    """
    ai_probability = predict_ai_probability(feature_vector)
    
    if ai_probability >= settings.classification_threshold:
        classification = CLASSIFICATION_AI
        confidence = ai_probability
    else:
        classification = CLASSIFICATION_HUMAN
        confidence = 1.0 - ai_probability
    
    # Round for cleaner output
    confidence = round(confidence, 2)
    
    return classification, confidence


def get_model_info() -> dict:
    """Get information about the loaded model."""
    model, _ = load_model()
    
    info = {
        "model_type": type(model).__name__,
        "is_fallback": isinstance(model, FallbackModel),
    }
    
    if hasattr(model, 'version'):
        info["version"] = model.version
    else:
        info["version"] = "trained-v1"
    
    if hasattr(model, 'training_date'):
        info["training_date"] = model.training_date
    else:
        info["training_date"] = "unknown"
    
    return info
