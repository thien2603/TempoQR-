"""
TempoQR API v1 Router
Main router for v1 API endpoints
"""

from fastapi import APIRouter
from src.api.api.v1.endpoints.predict import router as predict_router

# Create main v1 router
router = APIRouter()

# Include prediction endpoints
router.include_router(
    predict_router,
    prefix="/predict",
    tags=["prediction"]
)

# Root endpoint for v1
@router.get("/")
async def v1_root():
    """
    v1 API root endpoint
    """
    return {
        "message": "TempoQR API v1",
        "version": "1.0.0",
        "endpoints": {
            "predict": {
                "single": "/predict/single",
                "batch": "/predict/batch",
                "async": "/predict/async",
                "status": "/predict/status/{task_id}"
            },
            "health": "/health",
            "model": "/model/info"
        },
        "docs": "/docs"
    }
