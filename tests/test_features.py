"""
VoxGuard Feature Extraction Tests
Tests for audio decoding and feature extraction
"""

import pytest
import numpy as np
import base64

from voxguard_api.core.features import (
    extract_audio_features,
    build_feature_vector,
    get_feature_names
)


def generate_synthetic_audio(
    duration: float = 1.0,
    sample_rate: int = 22050,
    frequency: float = 440.0
) -> tuple:
    """Generate synthetic sine wave audio for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate multi-frequency audio for more realistic testing
    audio = (
        np.sin(2 * np.pi * frequency * t) +
        0.5 * np.sin(2 * np.pi * (frequency * 2) * t) +  # Harmonic
        0.3 * np.sin(2 * np.pi * (frequency * 3) * t)    # Harmonic
    )
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Add slight noise for realism
    audio += np.random.normal(0, 0.01, len(audio))
    
    return audio.astype(np.float32), sample_rate


class TestExtractAudioFeatures:
    """Tests for extract_audio_features function."""
    
    def test_returns_dict(self):
        """Test that extract returns a dictionary."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        
        assert isinstance(features, dict)
        
    def test_contains_mfcc_features(self):
        """Test that MFCC features are extracted."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        
        assert "mfcc_mean" in features
        assert "mfcc_std" in features
        assert "mfcc_min" in features
        assert "mfcc_max" in features
        assert "mfcc_delta" in features
        
    def test_contains_spectral_features(self):
        """Test that spectral features are extracted."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        
        assert "spectral_centroid_mean" in features
        assert "spectral_bandwidth_mean" in features
        assert "spectral_rolloff_mean" in features
        assert "spectral_flatness_mean" in features
        
    def test_contains_pitch_features(self):
        """Test that pitch features are extracted."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        
        assert "pitch_mean" in features
        assert "pitch_std" in features
        assert "pitch_range" in features
        assert "voiced_ratio" in features
        
    def test_contains_energy_features(self):
        """Test that energy features are extracted."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        
        assert "energy_mean" in features
        assert "energy_std" in features
        assert "dynamic_range" in features
        
    def test_handles_short_audio(self):
        """Test that short audio doesn't crash."""
        audio, sr = generate_synthetic_audio(duration=0.5)
        features = extract_audio_features(audio, sr)
        
        assert isinstance(features, dict)
        assert len(features) > 0


class TestBuildFeatureVector:
    """Tests for build_feature_vector function."""
    
    def test_returns_numpy_array(self):
        """Test that feature vector is a numpy array."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        vector = build_feature_vector(features)
        
        assert isinstance(vector, np.ndarray)
        
    def test_vector_is_1d(self):
        """Test that feature vector is 1-dimensional."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        vector = build_feature_vector(features)
        
        assert len(vector.shape) == 1
        
    def test_vector_has_no_nan(self):
        """Test that feature vector has no NaN values."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        vector = build_feature_vector(features)
        
        assert not np.any(np.isnan(vector))
        
    def test_vector_has_no_inf(self):
        """Test that feature vector has no infinite values."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        vector = build_feature_vector(features)
        
        assert not np.any(np.isinf(vector))
        
    def test_vector_is_float32(self):
        """Test that feature vector is float32."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        vector = build_feature_vector(features)
        
        assert vector.dtype == np.float32


class TestGetFeatureNames:
    """Tests for get_feature_names function."""
    
    def test_returns_list(self):
        """Test that feature names is a list."""
        names = get_feature_names()
        assert isinstance(names, list)
        
    def test_names_match_vector_length(self):
        """Test that number of names matches feature vector length."""
        audio, sr = generate_synthetic_audio()
        features = extract_audio_features(audio, sr)
        vector = build_feature_vector(features)
        names = get_feature_names()
        
        assert len(names) == len(vector)
        
    def test_all_names_are_strings(self):
        """Test that all feature names are strings."""
        names = get_feature_names()
        
        for name in names:
            assert isinstance(name, str)


class TestFeatureConsistency:
    """Tests for feature extraction consistency."""
    
    def test_same_audio_same_features(self):
        """Test that same audio produces same features."""
        audio, sr = generate_synthetic_audio(duration=1.0)
        
        features1 = extract_audio_features(audio.copy(), sr)
        features2 = extract_audio_features(audio.copy(), sr)
        
        # Check key features are identical
        np.testing.assert_array_almost_equal(
            features1["mfcc_mean"],
            features2["mfcc_mean"],
            decimal=5
        )
        
    def test_different_audio_different_features(self):
        """Test that different audio produces different features."""
        audio1, sr = generate_synthetic_audio(frequency=440)
        audio2, sr = generate_synthetic_audio(frequency=880)
        
        features1 = extract_audio_features(audio1, sr)
        features2 = extract_audio_features(audio2, sr)
        
        # Spectral centroid should differ for different frequencies
        assert features1["spectral_centroid_mean"] != features2["spectral_centroid_mean"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
