"""
VoxGuard Model Tests
Tests for model loading and prediction
"""

import pytest
import numpy as np

from voxguard_api.core.model import (
    load_model,
    predict_ai_probability,
    classify_audio,
    get_model_info,
    FallbackModel,
    FallbackScaler
)
from voxguard_api.core.config import CLASSIFICATION_AI, CLASSIFICATION_HUMAN


class TestFallbackModel:
    """Tests for the fallback heuristic model."""
    
    def test_predict_proba_returns_array(self):
        """Test that predict_proba returns numpy array."""
        model = FallbackModel()
        X = np.random.randn(1, 302).astype(np.float32)
        
        proba = model.predict_proba(X)
        
        assert isinstance(proba, np.ndarray)
        
    def test_predict_proba_shape(self):
        """Test that predict_proba returns correct shape."""
        model = FallbackModel()
        X = np.random.randn(5, 302).astype(np.float32)
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (5, 2)
        
    def test_predict_proba_sums_to_one(self):
        """Test that probabilities sum to ~1."""
        model = FallbackModel()
        X = np.random.randn(1, 302).astype(np.float32)
        
        proba = model.predict_proba(X)
        
        np.testing.assert_almost_equal(np.sum(proba[0]), 1.0, decimal=5)
        
    def test_predict_proba_in_range(self):
        """Test that probabilities are in [0, 1]."""
        model = FallbackModel()
        X = np.random.randn(10, 302).astype(np.float32)
        
        proba = model.predict_proba(X)
        
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        
    def test_predict_returns_binary(self):
        """Test that predict returns 0 or 1."""
        model = FallbackModel()
        X = np.random.randn(10, 302).astype(np.float32)
        
        predictions = model.predict(X)
        
        assert np.all((predictions == 0) | (predictions == 1))


class TestFallbackScaler:
    """Tests for the fallback scaler."""
    
    def test_transform_returns_array(self):
        """Test that transform returns numpy array."""
        scaler = FallbackScaler()
        X = np.random.randn(1, 302).astype(np.float32)
        
        result = scaler.transform(X)
        
        assert isinstance(result, np.ndarray)
        
    def test_transform_preserves_shape(self):
        """Test that transform preserves shape."""
        scaler = FallbackScaler()
        X = np.random.randn(5, 302).astype(np.float32)
        
        result = scaler.transform(X)
        
        assert result.shape == X.shape


class TestLoadModel:
    """Tests for model loading."""
    
    def test_load_model_returns_tuple(self):
        """Test that load_model returns tuple of (model, scaler)."""
        model, scaler = load_model()
        
        assert model is not None
        assert scaler is not None
        
    def test_model_has_predict_proba(self):
        """Test that model has predict_proba method."""
        model, _ = load_model()
        
        assert hasattr(model, 'predict_proba')
        
    def test_scaler_has_transform(self):
        """Test that scaler has transform method."""
        _, scaler = load_model()
        
        assert hasattr(scaler, 'transform')


class TestPredictAIProbability:
    """Tests for predict_ai_probability function."""
    
    def test_returns_float(self):
        """Test that probability is a float."""
        feature_vector = np.random.randn(302).astype(np.float32)
        
        prob = predict_ai_probability(feature_vector)
        
        assert isinstance(prob, float)
        
    def test_returns_in_range(self):
        """Test that probability is in [0, 1]."""
        feature_vector = np.random.randn(302).astype(np.float32)
        
        prob = predict_ai_probability(feature_vector)
        
        assert 0 <= prob <= 1
        
    def test_handles_1d_input(self):
        """Test that 1D input is handled."""
        feature_vector = np.random.randn(302).astype(np.float32)
        
        prob = predict_ai_probability(feature_vector)
        
        assert isinstance(prob, float)
        
    def test_handles_2d_input(self):
        """Test that 2D input is handled."""
        feature_vector = np.random.randn(1, 302).astype(np.float32)
        
        prob = predict_ai_probability(feature_vector)
        
        assert isinstance(prob, float)


class TestClassifyAudio:
    """Tests for classify_audio function."""
    
    def test_returns_tuple(self):
        """Test that classify returns tuple."""
        feature_vector = np.random.randn(302).astype(np.float32)
        
        result = classify_audio(feature_vector)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
    def test_classification_is_valid(self):
        """Test that classification is AI_GENERATED or HUMAN."""
        feature_vector = np.random.randn(302).astype(np.float32)
        
        classification, _ = classify_audio(feature_vector)
        
        assert classification in [CLASSIFICATION_AI, CLASSIFICATION_HUMAN]
        
    def test_confidence_in_range(self):
        """Test that confidence is in [0, 1]."""
        feature_vector = np.random.randn(302).astype(np.float32)
        
        _, confidence = classify_audio(feature_vector)
        
        assert 0 <= confidence <= 1


class TestGetModelInfo:
    """Tests for get_model_info function."""
    
    def test_returns_dict(self):
        """Test that model info is a dictionary."""
        info = get_model_info()
        
        assert isinstance(info, dict)
        
    def test_contains_required_keys(self):
        """Test that info contains required keys."""
        info = get_model_info()
        
        assert "model_type" in info
        assert "version" in info
        assert "is_fallback" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
