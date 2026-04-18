"""
Pydantic schemas for TempoQR API prediction requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionResult(BaseModel):
    """Single prediction result"""
    answer: str = Field(..., description="Predicted answer")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    answer_type: str = Field(..., description="Type of answer: 'entity' or 'time'")
    rank: int = Field(..., description="Rank of this prediction", ge=1)


class QuestionRequest(BaseModel):
    """Request schema for question answering"""
    question: str = Field(..., description="Question to answer", min_length=1, max_length=500)
    top_k: int = Field(default=5, description="Number of top predictions to return", ge=1, le=20)
    include_metadata: bool = Field(default=False, description="Include additional metadata in response")


class QuestionResponse(BaseModel):
    """Response schema for question answering"""
    model_config = {"protected_namespaces": ()}
    
    question: str = Field(..., description="Original question")
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    processing_time: float = Field(..., description="Processing time in seconds", ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Model information")


class BatchQuestionRequest(BaseModel):
    """Request schema for batch question answering"""
    questions: List[str] = Field(..., description="List of questions to answer", min_items=1, max_items=50)
    top_k: int = Field(default=5, description="Number of top predictions per question", ge=1, le=20)
    include_metadata: bool = Field(default=False, description="Include additional metadata in response")


class BatchQuestionResponse(BaseModel):
    """Response schema for batch question answering"""
    results: List[QuestionResponse] = Field(..., description="List of question responses")
    total_questions: int = Field(..., description="Total number of questions processed")
    total_processing_time: float = Field(..., description="Total processing time in seconds", ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    device: str = Field(..., description="Device being used")
    version: str = Field(..., description="API version")
    uptime: Optional[float] = Field(default=None, description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class ModelInfo(BaseModel):
    """Model information response"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    dataset_name: str = Field(..., description="Dataset name")
    num_entities: int = Field(..., description="Number of entities in knowledge graph")
    num_relations: int = Field(..., description="Number of relations in knowledge graph")
    num_timestamps: int = Field(..., description="Number of timestamps in knowledge graph")
    device: str = Field(..., description="Device being used")
    model_loaded: bool = Field(..., description="Whether model is loaded")
