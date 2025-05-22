"""
API package for Maxa AI.
"""
from fastapi import APIRouter

# Create main router
api_router = APIRouter()

# Import and include routers
from .endpoints import chat  # noqa

# Include routers
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
