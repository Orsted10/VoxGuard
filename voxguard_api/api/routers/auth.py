"""
VoxGuard API Key Authentication
Implements API key verification as a FastAPI dependency
"""

from fastapi import HTTPException, Header, status
from voxguard_api.core.config import settings


async def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")) -> str:
    """
    Verify the API key from request header.
    
    Args:
        x_api_key: The API key from the x-api-key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: 401 if API key is invalid or missing
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key or malformed request"
        )
    
    if x_api_key != settings.voxguard_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key or malformed request"
        )
    
    return x_api_key
