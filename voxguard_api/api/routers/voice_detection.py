"""
VoxGuard Voice Detection Router
POST /api/voice-detection endpoint implementation
"""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from voxguard_api.api.schemas import (
    VoiceDetectionRequest,
    VoiceDetectionResponse,
    ErrorResponse
)
from voxguard_api.api.routers.auth import verify_api_key
from voxguard_api.core.features import (
    decode_base64_mp3_to_array,
    extract_audio_features,
    build_feature_vector
)
from voxguard_api.core.model import classify_audio
from voxguard_api.core.explanations import build_explanation
from voxguard_api.core.language_id import validate_language
from voxguard_api.core.config import CLASSIFICATION_AI, CLASSIFICATION_HUMAN

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["Voice Detection"])

# Request metrics (in-memory for showcase)
_metrics = {
    "total_requests": 0,
    "ai_generated_count": 0,
    "human_count": 0,
    "error_count": 0,
    "start_time": time.time()
}


def get_metrics() -> dict:
    """Get current request metrics."""
    return {
        **_metrics,
        "uptime_seconds": time.time() - _metrics["start_time"]
    }


@router.post(
    "/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        200: {"model": VoiceDetectionResponse, "description": "Successful classification"},
        400: {"model": ErrorResponse, "description": "Bad request - invalid input"},
        401: {"model": ErrorResponse, "description": "Unauthorized - invalid API key"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Detect AI-Generated Voice",
    description="""
    Analyze an audio file to determine if the voice is AI-generated or human.
    
    **Supported Languages**: Tamil, English, Hindi, Malayalam, Telugu
    
    **Audio Requirements**:
    - Format: MP3
    - Encoding: Base64
    - Duration: 0.5 - 30 seconds
    
    **Authentication**: Requires valid API key in `x-api-key` header.
    """
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
) -> VoiceDetectionResponse:
    """
    Detect if voice audio is AI-generated or human.
    
    Args:
        request: VoiceDetectionRequest with language, audioFormat, and audioBase64
        api_key: Validated API key from header
        
    Returns:
        VoiceDetectionResponse with classification, confidence, and explanation
    """
    global _metrics
    _metrics["total_requests"] += 1
    
    start_time = time.time()
    
    try:
        # 1. Validate language
        logger.info(f"Processing voice detection request for language: {request.language}")
        try:
            validated_language = validate_language(request.language)
        except ValueError as e:
            _metrics["error_count"] += 1
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # 2. Decode Base64 audio
        try:
            waveform, sample_rate = decode_base64_mp3_to_array(request.audioBase64)
            logger.info(f"Decoded audio: {len(waveform)} samples at {sample_rate}Hz")
        except ValueError as e:
            _metrics["error_count"] += 1
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio decoding failed: {str(e)}"
            )
        
        # 3. Extract audio features
        try:
            features = extract_audio_features(waveform, sample_rate)
            feature_vector = build_feature_vector(features)
            logger.info(f"Extracted feature vector of shape: {feature_vector.shape}")
        except Exception as e:
            _metrics["error_count"] += 1
            logger.error(f"Feature extraction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract audio features"
            )
        
        # 4. Classify audio
        try:
            classification, confidence_score = classify_audio(feature_vector)
            logger.info(f"Classification: {classification}, Confidence: {confidence_score}")
        except Exception as e:
            _metrics["error_count"] += 1
            logger.error(f"Classification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Classification failed"
            )
        
        # 5. Generate explanation
        explanation = build_explanation(
            features=features,
            probability=confidence_score if classification == CLASSIFICATION_AI else 1 - confidence_score,
            language=validated_language,
            classification=classification
        )
        
        # Update metrics
        if classification == CLASSIFICATION_AI:
            _metrics["ai_generated_count"] += 1
        else:
            _metrics["human_count"] += 1
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.3f}s")
        
        # Return response
        return VoiceDetectionResponse(
            status="success",
            language=validated_language,
            classification=classification,
            confidenceScore=confidence_score,
            explanation=explanation
        )
        
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
        _metrics["error_count"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        _metrics["error_count"] += 1
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.get(
    "/voice-detection",
    summary="Voice Detection Info",
    description="Information about the voice detection endpoint."
)
async def get_voice_detection_info():
    """
    Return information about how to use the voice detection endpoint.
    This handles cases where a user (or tool) sends a GET request instead of POST.
    """
    return {
        "status": "info",
        "message": "This endpoint requires a POST request with audio data.",
        "usage": {
            "method": "POST",
            "headers": {"x-api-key": "YOUR_API_KEY"},
            "body": {
                "language": "Target Language",
                "audioFormat": "mp3",
                "audioBase64": "BASE64_ENCODED_STRING"
            }
        }
    }
