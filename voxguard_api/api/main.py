"""
VoxGuard API - Main Application
FastAPI entrypoint with all routes and middleware
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from voxguard_api.core.config import settings, SUPPORTED_LANGUAGES
from voxguard_api.core.model import load_model, get_model_info
from voxguard_api.api.schemas import HealthResponse, InfoResponse, MetricsResponse
from voxguard_api.api.routers import voice_detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application start time
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - runs on startup and shutdown."""
    # Startup
    logger.info("VoxGuard API starting up...")
    logger.info(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    
    # Pre-load model
    try:
        load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Model pre-loading failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("VoxGuard API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="VoxGuard API",
    description="""
## 🎙️ VoxGuard - Multilingual AI Voice Deepfake Detector

VoxGuard is a production-grade REST API that detects AI-generated voice deepfakes 
across multiple Indian languages.

### Features
- **Multilingual Support**: Tamil, English, Hindi, Malayalam, Telugu
- **Real-time Analysis**: Fast audio processing and classification
- **Explainable AI**: Human-readable explanations for each classification
- **Production Ready**: API key authentication, rate limiting, comprehensive error handling

### How It Works
1. Send Base64-encoded MP3 audio
2. Audio is analyzed using MFCC, spectral, and prosodic features
3. ML model classifies as `AI_GENERATED` or `HUMAN`
4. Response includes confidence score and explanation

### Authentication
All endpoints (except `/health`) require the `x-api-key` header.
    """,
    version="1.0.0",
    contact={
        "name": "VoxGuard Team",
        "email": "support@voxguard.ai"
    },
    license_info={
        "name": "MIT License"
    },
    lifespan=lifespan
)

# Add CORS middleware for broad compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start = time.time()
    
    # Generate request ID
    request_id = f"req_{int(start * 1000) % 100000}"
    
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = time.time() - start
    logger.info(f"[{request_id}] Completed in {duration:.3f}s - Status: {response.status_code}")
    
    return response


# Include routers
app.include_router(voice_detection.router)


# Health check endpoint (no auth required)
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check if the API is running and healthy. No authentication required."
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        supported_languages=SUPPORTED_LANGUAGES
    )


# Info endpoint
@app.get(
    "/info",
    response_model=InfoResponse,
    tags=["System"],
    summary="API Information",
    description="Get detailed information about the API, supported languages, and model."
)
async def get_info() -> InfoResponse:
    """Get API information."""
    model_info = get_model_info()
    
    return InfoResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        supported_languages=SUPPORTED_LANGUAGES,
        model_version=model_info.get("version", "unknown"),
        description="Multilingual AI Voice Deepfake Detector for Tamil, English, Hindi, Malayalam, and Telugu",
        endpoints={
            "POST /api/voice-detection": "Analyze audio for AI-generated voice",
            "GET /health": "Health check",
            "GET /info": "API information",
            "GET /metrics": "Usage metrics"
        }
    )


# Metrics endpoint
@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["System"],
    summary="Usage Metrics",
    description="Get usage statistics for the API."
)
async def get_metrics() -> MetricsResponse:
    """Get API usage metrics."""
    from voxguard_api.api.routers.voice_detection import get_metrics as get_detection_metrics
    
    metrics = get_detection_metrics()
    
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        ai_generated_count=metrics["ai_generated_count"],
        human_count=metrics["human_count"],
        error_count=metrics["error_count"],
        uptime_seconds=time.time() - _start_time
    )


# Root endpoint
@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "Welcome to VoxGuard API",
        "docs": "/docs",
        "health": "/health",
        "version": settings.app_version
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected error occurred"
        }
    )
