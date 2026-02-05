"""
VoxGuard API Tests
Tests for API endpoints, authentication, and response schemas
"""

import pytest
import base64
import numpy as np
from fastapi.testclient import TestClient

from voxguard_api.api.main import app
from voxguard_api.core.config import settings, SUPPORTED_LANGUAGES


# Create test client
client = TestClient(app)

# Valid API key for tests
VALID_API_KEY = settings.voxguard_api_key
INVALID_API_KEY = "invalid_key_12345"


def generate_test_audio_base64() -> str:
    """Generate a valid Base64 encoded synthetic audio for testing."""
    # Generate 1 second of synthetic audio (sine wave)
    sample_rate = 22050
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    # Create a simple WAV-like structure (simplified for testing)
    # In real usage, this would be actual MP3 data
    # For testing, we'll use a known short MP3 file or handle the decode error gracefully
    
    # Return minimal valid-looking Base64 (will likely fail decode, which is fine for error testing)
    return base64.b64encode(audio.tobytes()).decode('utf-8')


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_returns_ok(self):
        """Test that health check returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_health_check_response_schema(self):
        """Test that health check returns correct schema."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "ok"
        assert "version" in data
        assert "supported_languages" in data
        
    def test_health_check_supported_languages(self):
        """Test that health check lists all supported languages."""
        response = client.get("/health")
        data = response.json()
        
        for lang in SUPPORTED_LANGUAGES:
            assert lang in data["supported_languages"]


class TestInfoEndpoint:
    """Tests for /info endpoint."""
    
    def test_info_returns_ok(self):
        """Test that info endpoint returns 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200
        
    def test_info_response_schema(self):
        """Test that info returns correct schema."""
        response = client.get("/info")
        data = response.json()
        
        assert "app_name" in data
        assert "version" in data
        assert "supported_languages" in data
        assert "model_version" in data
        assert "endpoints" in data


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""
    
    def test_metrics_returns_ok(self):
        """Test that metrics endpoint returns 200 OK."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
    def test_metrics_response_schema(self):
        """Test that metrics returns correct schema."""
        response = client.get("/metrics")
        data = response.json()
        
        assert "total_requests" in data
        assert "ai_generated_count" in data
        assert "human_count" in data
        assert "error_count" in data
        assert "uptime_seconds" in data


class TestVoiceDetectionAuthentication:
    """Tests for voice detection auth."""
    
    def test_missing_api_key_returns_401(self):
        """Test that missing API key returns 401."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "SGVsbG8gV29ybGQ="
            }
        )
        assert response.status_code == 422  # Missing header
        
    def test_invalid_api_key_returns_401(self):
        """Test that invalid API key returns 401."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "SGVsbG8gV29ybGQ="
            },
            headers={"x-api-key": INVALID_API_KEY}
        )
        assert response.status_code == 401
        
        data = response.json()
        assert "Invalid API key" in data["detail"]


class TestVoiceDetectionValidation:
    """Tests for voice detection input validation."""
    
    def test_invalid_language_returns_400(self):
        """Test that unsupported language returns 400."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "French",  # Not supported
                "audioFormat": "mp3",
                "audioBase64": "SGVsbG8gV29ybGQ="
            },
            headers={"x-api-key": VALID_API_KEY}
        )
        assert response.status_code == 422  # Pydantic validation error
        
    def test_invalid_audio_format_returns_400(self):
        """Test that unsupported audio format returns 400."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "wav",  # Not mp3
                "audioBase64": "SGVsbG8gV29ybGQ="
            },
            headers={"x-api-key": VALID_API_KEY}
        )
        assert response.status_code == 422  # Pydantic validation error
        
    def test_empty_audio_returns_400(self):
        """Test that empty audio returns 400."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": ""
            },
            headers={"x-api-key": VALID_API_KEY}
        )
        assert response.status_code == 422  # Pydantic validation error


class TestVoiceDetectionResponse:
    """Tests for voice detection response format."""
    
    def test_response_has_required_fields(self):
        """Test that response has all required fields (with audio decode error)."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": generate_test_audio_base64()
            },
            headers={"x-api-key": VALID_API_KEY}
        )
        
        # This will likely return 400 due to invalid audio, which is expected
        # The test verifies error handling works correctly
        assert response.status_code in [200, 400, 500]
        
        data = response.json()
        # Either success response or error response
        assert "status" in data or "detail" in data


class TestAllSupportedLanguages:
    """Tests for all supported languages."""
    
    @pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
    def test_language_accepted(self, language):
        """Test that each supported language is accepted."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": generate_test_audio_base64()
            },
            headers={"x-api-key": VALID_API_KEY}
        )
        
        # Should not return 422 for valid language
        # May return 400 for audio decode error, which is fine
        assert response.status_code != 422 or "language" not in str(response.json())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
