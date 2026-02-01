"""
API Schemas

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================
# Request Models
# ============================================

class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    user_id: str = Field(..., description="User ID")
    num_recommendations: int = Field(default=20, ge=1, le=100, description="Number of recommendations")
    exclude_watched: bool = Field(default=True, description="Exclude already watched videos")


class FeedbackRequest(BaseModel):
    """User feedback/interaction request."""
    user_id: str = Field(..., description="User ID")
    video_id: str = Field(..., description="Video ID")
    interaction_type: str = Field(..., description="Type: view, like, dislike, share")
    watch_duration_seconds: Optional[int] = Field(default=None, ge=0)
    watch_percentage: Optional[float] = Field(default=None, ge=0, le=1)


class SimilarVideosRequest(BaseModel):
    """Request for similar videos."""
    video_id: str = Field(..., description="Source video ID")
    num_results: int = Field(default=10, ge=1, le=50)


# ============================================
# Response Models
# ============================================

class VideoItem(BaseModel):
    """A single video item."""
    video_id: str
    title: Optional[str] = None
    channel_name: Optional[str] = None
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    view_count: Optional[int] = None
    score: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Response with recommendations."""
    user_id: str
    recommendations: List[VideoItem]
    num_results: int
    retrieval_time_ms: Optional[float] = None
    ranking_time_ms: Optional[float] = None
    reranking_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None


class SimilarVideosResponse(BaseModel):
    """Response with similar videos."""
    source_video_id: str
    similar_videos: List[VideoItem]
    num_results: int


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    retrieval_service: str
    version: str
    timestamp: datetime


class StatsResponse(BaseModel):
    """System statistics response."""
    total_videos: int
    total_users: int
    total_interactions: int
    videos_with_embeddings: int
    users_with_embeddings: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None