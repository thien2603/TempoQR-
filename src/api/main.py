"""
TempoQR FastAPI Application
Main application for TempoQR API
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from contextlib import asynccontextmanager
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import API components
from src.api.api.v1.router import router as v1_router
from src.api.services.tempoqr_service import tempoqr_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting TempoQR API v1")
    logger.info(f"Device: {tempoqr_service.model_manager.device}")
    logger.info("Model loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TempoQR API")

# Create FastAPI app
app = FastAPI(
    title="TempoQR API",
    version="1.0.0",
    description="TempoQR API for temporal knowledge graph question answering",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Include API routes
app.include_router(v1_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    health_data = tempoqr_service.health_check()
    return JSONResponse(content=health_data)

# Root endpoint - serve the web interface
@app.get("/")
async def root():
    """
    Root endpoint - serve web interface
    """
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={
        "message": "Welcome to TempoQR API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
        "web_interface": "/static/index.html"
    })

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
