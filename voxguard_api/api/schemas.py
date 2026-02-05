"""
VoxGuard API Schemas
Pydantic models for request/response validation
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from voxguard_api.core.config import SUPPORTED_LANGUAGES


class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection endpoint"""
    
    language: str = Field(
        ...,
        description="Language of the audio. Must be one of: Tamil, English, Hindi, Malayalam, Telugu",
        examples=["Tamil", "English"]
    )
    audioFormat: str = Field(
        ...,
        description="Format of the audio file. Must be 'mp3'",
        examples=["mp3"]
    )
    audioBase64: str = Field(
        ...,
        description="Base64 encoded MP3 audio content",
        min_length=1
    )
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language. Must be one of: {', '.join(SUPPORTED_LANGUAGES)}")
        return v
    
    @field_validator("audioFormat")
    @classmethod
    def validate_audio_format(cls, v: str) -> str:
        if v.lower() != "mp3":
            raise ValueError("audioFormat must be 'mp3'")
        return v.lower()


class VoiceDetectionResponse(BaseModel):
    """Response model for successful voice detection"""
    
    status: Literal["success"] = Field(
        default="success",
        description="Status of the request"
    )
    language: str = Field(
        ...,
        description="Language of the analyzed audio"
    )
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ...,
        description="Classification result: AI_GENERATED or HUMAN"
    )
    confidenceScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the classification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.91,
                "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for error cases"""
    
    status: Literal["error"] = Field(
        default="error",
        description="Status indicating an error occurred"
    )
    message: str = Field(
        ...,
        description="Error message describing what went wrong"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    
    status: Literal["ok"] = "ok"
    version: str
    supported_languages: list[str]


class InfoResponse(BaseModel):
    """Response model for info endpoint"""
    
    app_name: str
    version: str
    supported_languages: list[str]
    model_version: str
    description: str
    endpoints: dict


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    
    total_requests: int
    ai_generated_count: int
    human_count: int
    error_count: int
    uptime_seconds: float
