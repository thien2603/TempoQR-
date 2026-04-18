"""
FastAPI Dependencies
Dependency injection for TempoQR API
"""

from fastapi import Depends
from typing import Generator
import logging

from src.api.services.tempoqr_service import tempoqr_service

logger = logging.getLogger(__name__)


async def get_tempoqr_service() -> tempoqr_service.TempoQRService:
    """
    Get TempoQR service instance
    
    Returns:
        TempoQRService: Service instance
    """
    return tempoqr_service


async def get_current_user():
    """
    Get current user (placeholder for future authentication)
    
    Returns:
        dict: User information
    """
    # This is a placeholder for future authentication implementation
    return {
        "id": "anonymous",
        "username": "anonymous",
        "permissions": ["predict", "health"]
    }


def get_query_params(
    limit: int = 10,
    offset: int = 0,
    sort_by: str = "confidence"
):
    """
    Common query parameters
    
    Args:
        limit: Maximum number of results
        offset: Number of results to skip
        sort_by: Field to sort by
        
    Returns:
        dict: Query parameters
    """
    return {
        "limit": min(limit, 100),  # Cap at 100
        "offset": max(offset, 0),
        "sort_by": sort_by if sort_by in ["confidence", "rank", "answer"] else "confidence"
    }


def validate_question_length(question: str) -> bool:
    """
    Validate question length
    
    Args:
        question: Question text
        
    Returns:
        bool: True if valid
    """
    if not question or not question.strip():
        return False
    
    if len(question.strip()) < 1:
        return False
    
    if len(question.strip()) > 500:
        return False
    
    return True


def validate_top_k(top_k: int) -> bool:
    """
    Validate top_k parameter
    
    Args:
        top_k: Number of predictions requested
        
    Returns:
        bool: True if valid
    """
    return isinstance(top_k, int) and 1 <= top_k <= 20


def get_pagination_params(
    page: int = 1,
    size: int = 10
) -> dict:
    """
    Get pagination parameters
    
    Args:
        page: Page number (1-based)
        size: Page size
        
    Returns:
        dict: Pagination parameters
    """
    if page < 1:
        page = 1
    
    if size < 1:
        size = 10
    if size > 100:
        size = 100
    
    offset = (page - 1) * size
    
    return {
        "page": page,
        "size": size,
        "offset": offset,
        "limit": size
    }
