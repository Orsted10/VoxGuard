"""
VoxGuard API - Vercel Serverless Entry Point
"""

from voxguard_api.api.main import app

# Vercel expects 'app' or 'handler' as the ASGI application
handler = app
