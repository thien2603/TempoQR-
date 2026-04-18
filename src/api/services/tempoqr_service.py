"""
TempoQR Service Layer
Business logic for TempoQR model predictions
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..schemas.predict import (
    QuestionRequest, 
    QuestionResponse, 
    PredictionResult,
    BatchQuestionRequest,
    BatchQuestionResponse,
    ModelInfo
)
from src.core.model_loader import model_manager
from src.core.config import settings

logger = logging.getLogger(__name__)


class TempoQRService:
    """Service class for TempoQR model operations"""
    
    def __init__(self):
        """Initialize service with model manager"""
        self.model_manager = model_manager
        self.start_time = time.time()
        
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.start_time
    
    def get_model_info(self) -> ModelInfo:
        """Get model information"""
        try:
            # Get model statistics from model manager
            qa_model = self.model_manager.qa_model
            
            # Try to get entity count from model
            num_entities = 0
            num_relations = 0
            num_timestamps = 0
            
            if hasattr(qa_model, 'tkbc_model'):
                tkbc_model = qa_model.tkbc_model
                if hasattr(tkbc_model, 'num_ent'):
                    num_entities = tkbc_model.num_ent
                if hasattr(tkbc_model, 'num_rel'):
                    num_relations = tkbc_model.num_rel
                if hasattr(tkbc_model, 'num_ts'):
                    num_timestamps = tkbc_model.num_ts
            
            return ModelInfo(
                model_name=settings.MODEL_TYPE,
                model_type="Temporal Knowledge Graph QA",
                dataset_name=settings.DATASET_NAME,
                num_entities=num_entities,
                num_relations=num_relations,
                num_timestamps=num_timestamps,
                device=str(self.model_manager.device),
                model_loaded=qa_model is not None
            )
        except Exception as e:
            logging.error(f"Error getting model info: {e}")
            return ModelInfo(
                model_name=settings.MODEL_TYPE,
                model_type="Temporal Knowledge Graph QA",
                dataset_name=settings.DATASET_NAME,
                num_entities=0,
                num_relations=0,
                num_timestamps=0,
                device=str(self.model_manager.device),
                model_loaded=False
            )
    
    async def predict_single(self, request: QuestionRequest) -> QuestionResponse:
        """
        Predict answer for a single question
        
        Args:
            request: QuestionRequest with question and parameters
            
        Returns:
            QuestionResponse with predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not request.question or not request.question.strip():
                raise ValueError("Question cannot be empty")
            
            # Preprocess question
            question = request.question.strip()
            top_k = min(request.top_k, 20)  # Limit to reasonable max
            
            # Get prediction from model manager
            predictions = self.model_manager.predict(question, k=top_k)
            
            # Debug: Log predictions from model manager
            logging.info(f"Model manager predictions: {predictions}")
            
            # Handle None predictions
            if predictions is None:
                predictions = []
            
            # Convert predictions to PredictionResult objects
            prediction_results = []
            for i, pred in enumerate(predictions, 1):
                # Simple confidence calculation (can be enhanced)
                confidence = 1.0 - (i - 1) * 0.1  # Decreasing confidence
                confidence = max(0.1, min(1.0, confidence))
                
                # Determine answer type (simplified)
                answer_type = "entity" if not pred.isdigit() else "time"
                
                prediction_results.append(PredictionResult(
                    answer=str(pred),
                    confidence=confidence,
                    answer_type=answer_type,
                    rank=i
                ))
            
            processing_time = time.time() - start_time
            
            # Get model info if requested
            model_info = None
            if request.include_metadata:
                model_info = self.get_model_info()
            
            return QuestionResponse(
                question=question,
                predictions=prediction_results,
                processing_time=processing_time,
                timestamp=datetime.utcnow(),
                model_info=model_info
            )
            
        except Exception as e:
            logging.error(f"Error in predict_single: {e}")
            processing_time = time.time() - start_time
            
            return QuestionResponse(
                question=request.question,
                predictions=[],
                processing_time=processing_time,
                timestamp=datetime.utcnow(),
                model_info=None
            )
    
    async def predict_batch(self, request: BatchQuestionRequest) -> BatchQuestionResponse:
        """
        Predict answers for multiple questions
        
        Args:
            request: BatchQuestionRequest with list of questions
            
        Returns:
            BatchQuestionResponse with all predictions
        """
        start_time = time.time()
        results = []
        
        try:
            # Validate input
            if not request.questions or len(request.questions) == 0:
                raise ValueError("Questions list cannot be empty")
            
            if len(request.questions) > 50:
                raise ValueError("Maximum 50 questions allowed per batch")
            
            # Process each question
            for question in request.questions:
                single_request = QuestionRequest(
                    question=question,
                    top_k=request.top_k,
                    include_metadata=request.include_metadata
                )
                
                # Use predict_single for each question
                result = await self.predict_single(single_request)
                results.append(result)
            
            total_processing_time = time.time() - start_time
            
            return BatchQuestionResponse(
                results=results,
                total_questions=len(request.questions),
                total_processing_time=total_processing_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logging.error(f"Error in predict_batch: {e}")
            total_processing_time = time.time() - start_time
            
            # Return empty results on error
            return BatchQuestionResponse(
                results=[],
                total_questions=len(request.questions) if request.questions else 0,
                total_processing_time=total_processing_time,
                timestamp=datetime.utcnow()
            )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on service and models
        
        Returns:
            Dict with health status information
        """
        try:
            model_info = self.get_model_info()
            uptime = self.get_uptime()
            
            return {
                "status": "healthy",
                "message": "TempoQR service is running",
                "models_loaded": model_info.model_loaded,
                "device": model_info.device,
                "version": settings.VERSION,
                "uptime": uptime,
                "model_info": model_info.dict() if model_info else None
            }
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "models_loaded": False,
                "device": "unknown",
                "version": settings.VERSION,
                "uptime": self.get_uptime()
            }


# Global service instance
tempoqr_service = TempoQRService()
