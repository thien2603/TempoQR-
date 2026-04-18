"""
TempoQR API v1 Prediction Endpoints
REST API endpoints for TempoQR model predictions
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
import logging
import asyncio

from src.api.schemas.predict import (
    QuestionRequest,
    QuestionResponse,
    BatchQuestionRequest,
    BatchQuestionResponse,
    HealthResponse,
    ErrorResponse,
    ModelInfo
)
from src.api.services.tempoqr_service import tempoqr_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/single", response_model=QuestionResponse)
async def predict_question(request: QuestionRequest):
    """
    Predict answer for a single question
    
    Args:
        request: QuestionRequest with question and parameters
        
    Returns:
        QuestionResponse with predictions and metadata
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Processing single question: {request.question[:100]}...")
        
        # Validate input
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Get prediction from service
        response = await tempoqr_service.predict_single(request)
        
        logger.info(f"Successfully processed question in {response.processing_time:.3f}s")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@router.post("/predict/batch", response_model=BatchQuestionResponse)
async def predict_batch(request: BatchQuestionRequest):
    """
    Predict answers for multiple questions
    
    Args:
        request: BatchQuestionRequest with list of questions
        
    Returns:
        BatchQuestionResponse with all predictions
        
    Raises:
        HTTPException: If batch prediction fails
    """
    try:
        logger.info(f"Processing batch of {len(request.questions)} questions")
        
        # Validate input
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(
                status_code=400,
                detail="Questions list cannot be empty"
            )
        
        if len(request.questions) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 questions allowed per batch"
            )
        
        # Get batch prediction from service
        response = await tempoqr_service.predict_batch(request)
        
        logger.info(f"Successfully processed batch in {response.total_processing_time:.3f}s")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during batch prediction"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with service status
    """
    try:
        health_data = tempoqr_service.health_check()
        
        return HealthResponse(
            status=health_data["status"],
            message=health_data["message"],
            models_loaded=health_data["models_loaded"],
            device=health_data["device"],
            version=health_data["version"],
            uptime=health_data.get("uptime")
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            models_loaded=False,
            device="unknown",
            version="1.0.0",
            uptime=None
        )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get model information
    
    Returns:
        ModelInfo with model details
    """
    try:
        model_info = tempoqr_service.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get model information"
        )


@router.post("/predict/async")
async def predict_question_async(request: QuestionRequest, background_tasks: BackgroundTasks):
    """
    Async prediction endpoint (placeholder for future implementation)
    
    Args:
        request: QuestionRequest with question and parameters
        background_tasks: FastAPI BackgroundTasks
        
    Returns:
        Task ID for async processing
    """
    try:
        # Generate task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # This is a placeholder for future async implementation
        # background_tasks.add_task(process_prediction_async, task_id, request)
        
        logger.info(f"Async prediction task created: {task_id}")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Prediction task queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Async prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create async prediction task"
        )


@router.get("/predict/status/{task_id}")
async def get_prediction_status(task_id: str):
    """
    Get status of async prediction task (placeholder)
    
    Args:
        task_id: Task ID from async prediction
        
    Returns:
        Task status information
    """
    try:
        # This is a placeholder for future implementation
        return {
            "task_id": task_id,
            "status": "completed",
            "message": "Task completed (placeholder implementation)"
        }
        
    except Exception as e:
        logger.error(f"Task status error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get task status"
        )


# Note: Exception handlers are moved to main.py since APIRouter doesn't support exception_handler
