"""
FastAPI Application

Main API entry point for the recommendation system.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from serving.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    FeedbackRequest,
    FeedbackResponse,
    SimilarVideosRequest,
    SimilarVideosResponse,
    HealthResponse,
    StatsResponse,
    VideoItem,
    ErrorResponse,
)
from models.retrieval import SimpleRetrievalService
from models.reranking import ReRankingPipeline, VideoCandidate

load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")
INDEX_PATH = os.getenv("INDEX_PATH", "models/retrieval/saved/simple_faiss_index")
API_VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Database Connection
# ============================================

def get_connection():
    """Get database connection."""
    return psycopg2.connect(DATABASE_URL)


def parse_embedding(emb):
    """Parse embedding from string or list format."""
    if emb is None:
        return None
    if isinstance(emb, str):
        emb = emb.strip('[]')
        emb = [float(x) for x in emb.split(',')]
    return np.array(emb, dtype=np.float32)


# ============================================
# Global Services
# ============================================

retrieval_service: Optional[SimpleRetrievalService] = None
reranking_pipeline: Optional[ReRankingPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global retrieval_service, reranking_pipeline
    
    # Startup
    logger.info("Starting YouTube RecSys API...")
    
    # Initialize retrieval service
    try:
        index_path = Path(INDEX_PATH)
        if index_path.with_suffix('.faiss').exists():
            retrieval_service = SimpleRetrievalService(embedding_dim=256)
            retrieval_service.load(str(index_path))
            logger.info(f"Loaded retrieval service from {index_path}")
        else:
            logger.warning(f"Index not found at {index_path}. Run build_index.py first.")
    except Exception as e:
        logger.error(f"Failed to load retrieval service: {e}")
    
    # Initialize re-ranking pipeline
    reranking_pipeline = ReRankingPipeline()
    logger.info("Initialized re-ranking pipeline")
    
    yield
    
    # Shutdown
    logger.info("Shutting down YouTube RecSys API...")


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="YouTube RecSys API",
    description="Video Recommendation System with Two-Tower Architecture",
    version=API_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Helper Functions
# ============================================

def get_user_embedding(user_id: str) -> Optional[np.ndarray]:
    """Get user embedding from database."""
    query = "SELECT user_embedding FROM user_embeddings WHERE user_id = %s"
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id,))
            row = cur.fetchone()
            if row and row[0]:
                return parse_embedding(row[0])
    return None


def get_user_watch_history(user_id: str, limit: int = 100) -> list:
    """Get user's recent watch history."""
    query = """
        SELECT video_id FROM user_interactions
        WHERE user_id = %s AND interaction_type = 'view'
        ORDER BY created_at DESC
        LIMIT %s
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, limit))
            return [row[0] for row in cur.fetchall()]


def get_video_metadata(video_ids: list) -> dict:
    """Get video metadata from database."""
    if not video_ids:
        return {}
    
    placeholders = ','.join(['%s'] * len(video_ids))
    query = f"""
        SELECT video_id, title, channel_name, category_id, category_name,
               thumbnail_url, duration_seconds, view_count
        FROM videos
        WHERE video_id IN ({placeholders})
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, video_ids)
            return {
                row[0]: {
                    'video_id': row[0],
                    'title': row[1],
                    'channel_name': row[2],
                    'category_id': row[3],
                    'category_name': row[4],
                    'thumbnail_url': row[5],
                    'duration_seconds': row[6],
                    'view_count': row[7],
                }
                for row in cur.fetchall()
            }


def get_video_embeddings(video_ids: list) -> dict:
    """Get video embeddings from database."""
    if not video_ids:
        return {}
    
    placeholders = ','.join(['%s'] * len(video_ids))
    query = f"""
        SELECT video_id, combined_embedding
        FROM video_embeddings
        WHERE video_id IN ({placeholders})
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, video_ids)
            return {row[0]: parse_embedding(row[1]) for row in cur.fetchall()}


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "service": "YouTube RecSys API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    # Check database
    db_status = "healthy"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check retrieval service
    retrieval_status = "healthy" if retrieval_service else "not initialized"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
        retrieval_service=retrieval_status,
        version=API_VERSION,
        timestamp=datetime.now(),
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Get system statistics."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM videos WHERE is_active = true")
                total_videos = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM users")
                total_users = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM user_interactions")
                total_interactions = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM video_embeddings")
                videos_with_emb = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM user_embeddings")
                users_with_emb = cur.fetchone()[0]
        
        return StatsResponse(
            total_videos=total_videos,
            total_users=total_users,
            total_interactions=total_interactions,
            videos_with_embeddings=videos_with_emb,
            users_with_embeddings=users_with_emb,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """Get personalized video recommendations for a user."""
    start_time = time.time()
    
    # Get user embedding
    user_embedding = get_user_embedding(request.user_id)
    if user_embedding is None:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found or has no embedding")
    
    # Get watched videos to exclude
    exclude_ids = None
    if request.exclude_watched:
        exclude_ids = get_user_watch_history(request.user_id)
    
    # Stage 1: Retrieval
    retrieval_start = time.time()
    
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not initialized")
    
    retrieved_ids, retrieval_scores = retrieval_service.retrieve(
        user_embedding,
        k=min(500, request.num_recommendations * 10),
        exclude_video_ids=exclude_ids,
    )
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    if not retrieved_ids:
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=[],
            num_results=0,
            retrieval_time_ms=retrieval_time,
            total_time_ms=(time.time() - start_time) * 1000,
        )
    
    # Stage 2: Ranking (using retrieval scores for now)
    ranking_start = time.time()
    ranked = list(zip(retrieved_ids, retrieval_scores.tolist()))
    ranking_time = (time.time() - ranking_start) * 1000
    
    # Stage 3: Re-ranking
    reranking_start = time.time()
    
    # Get metadata and embeddings for re-ranking
    video_ids = [vid for vid, _ in ranked]
    metadata = get_video_metadata(video_ids)
    embeddings = get_video_embeddings(video_ids)
    
    # Create candidates
    candidates = []
    for vid, score in ranked:
        meta = metadata.get(vid, {})
        candidates.append(VideoCandidate(
            video_id=vid,
            score=score,
            embedding=embeddings.get(vid),
            category_id=meta.get('category_id'),
            channel_id=meta.get('channel_id'),
            duration_seconds=meta.get('duration_seconds'),
        ))
    
    # Apply re-ranking
    reranked = reranking_pipeline.rerank(candidates, k=request.num_recommendations)
    reranking_time = (time.time() - reranking_start) * 1000
    
    # Build response
    recommendations = []
    final_ids = [c.video_id for c in reranked]
    final_metadata = get_video_metadata(final_ids)
    
    for candidate in reranked:
        meta = final_metadata.get(candidate.video_id, {})
        recommendations.append(VideoItem(
            video_id=candidate.video_id,
            title=meta.get('title'),
            channel_name=meta.get('channel_name'),
            category_id=meta.get('category_id'),
            category_name=meta.get('category_name'),
            thumbnail_url=meta.get('thumbnail_url'),
            duration_seconds=meta.get('duration_seconds'),
            view_count=meta.get('view_count'),
            score=candidate.score,
        ))
    
    total_time = (time.time() - start_time) * 1000
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        num_results=len(recommendations),
        retrieval_time_ms=retrieval_time,
        ranking_time_ms=ranking_time,
        reranking_time_ms=reranking_time,
        total_time_ms=total_time,
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_get(
    user_id: str,
    n: int = Query(default=20, ge=1, le=100, description="Number of recommendations"),
    exclude_watched: bool = Query(default=True),
):
    """Get recommendations via GET request."""
    return await get_recommendations(RecommendationRequest(
        user_id=user_id,
        num_recommendations=n,
        exclude_watched=exclude_watched,
    ))


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback/interaction."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_interactions 
                    (user_id, video_id, interaction_type, watch_duration_seconds, watch_percentage)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    request.user_id,
                    request.video_id,
                    request.interaction_type,
                    request.watch_duration_seconds,
                    request.watch_percentage,
                ))
                conn.commit()
        
        return FeedbackResponse(status="success", message="Feedback recorded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos/{video_id}", response_model=VideoItem, tags=["Videos"])
async def get_video(video_id: str):
    """Get video details."""
    metadata = get_video_metadata([video_id])
    
    if video_id not in metadata:
        raise HTTPException(status_code=404, detail="Video not found")
    
    meta = metadata[video_id]
    return VideoItem(**meta)


@app.get("/users/{user_id}/history", tags=["Users"])
async def get_user_history(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    """Get user's watch history."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ui.video_id, v.title, v.thumbnail_url, 
                           ui.watch_percentage, ui.interaction_type, ui.created_at
                    FROM user_interactions ui
                    LEFT JOIN videos v ON ui.video_id = v.video_id
                    WHERE ui.user_id = %s
                    ORDER BY ui.created_at DESC
                    LIMIT %s
                """, (user_id, limit))
                
                history = []
                for row in cur.fetchall():
                    history.append({
                        "video_id": row[0],
                        "title": row[1],
                        "thumbnail_url": row[2],
                        "watch_percentage": row[3],
                        "interaction_type": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                    })
        
        return {"user_id": user_id, "history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)