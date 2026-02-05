"""
VoxGuard Configuration Module
Handles all settings, API keys, and constants
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    app_name: str = "VoxGuard API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API Key Authentication
    voxguard_api_key: str = "sk_test_123456789"
    
    # Model paths
    model_path: str = "models/ai_detector.pkl"
    scaler_path: str = "models/scaler.pkl"
    
    # Audio processing settings
    sample_rate: int = 22050
    max_audio_duration: float = 30.0  # seconds
    min_audio_duration: float = 0.5   # seconds
    
    # Confidence thresholds
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    classification_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Supported languages (fixed list per requirements)
SUPPORTED_LANGUAGES: List[str] = [
    "Tamil",
    "English", 
    "Hindi",
    "Malayalam",
    "Telugu"
]

# Classification labels
CLASSIFICATION_AI: str = "AI_GENERATED"
CLASSIFICATION_HUMAN: str = "HUMAN"

# Feature extraction parameters
MFCC_N_COEFFS: int = 40
N_MELS: int = 128
HOP_LENGTH: int = 512
N_FFT: int = 2048

# Rate limiting (requests per minute per API key)
RATE_LIMIT_PER_MINUTE: int = 60


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
