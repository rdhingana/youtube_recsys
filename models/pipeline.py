"""
Recommendation Pipeline

Full pipeline: Retrieval → Ranking → Re-ranking
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import psycopg2
from dotenv import load_dotenv

from models.retrieval import SimpleRetrievalService, RetrievalService
from models.ranking import RankingModel
from models.reranking import VideoCandidate, ReRankingPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")


def get_connection():
    return psycopg2.connect(DATABASE_URL)


class RecommendationPipeline:
    """
    Full recommendation pipeline.
    
    Stage 1: Retrieval - Get ~500 candidates using two-tower model + FAISS
    Stage 2: Ranking - Score candidates using ranking model
    Stage 3: Re-ranking - Apply diversity and business rules
    """
    
    def __init__(
        self,
        retrieval_model_path: str = None,
        ranking_model_path: str = None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize retrieval
        self.retrieval_service = None
        if retrieval_model_path and Path(retrieval_model_path).exists():
            self.retrieval_service = SimpleRetrievalService(embedding_dim=256)
            self.retrieval_service.load(retrieval_model_path)
            logger.info("Loaded retrieval service")
        
        # Initialize ranking model
        self.ranking_model = None
        if ranking_model_path and Path(ranking_model_path).exists():
            self.ranking_model = RankingModel(user_dim=256, video_dim=256)
            self.ranking_model.load_state_dict(torch.load(ranking_model_path, map_location=self.device))
            self.ranking_model.to(self.device)
            self.ranking_model.eval()
            logger.info("Loaded ranking model")
        
        # Initialize re-ranking pipeline
        self.reranking_pipeline = ReRankingPipeline()
        
        # Cache for video metadata
        self._video_cache = {}
        self._embedding_cache = {}
    
    def _load_video_metadata(self, video_ids: List[str]) -> Dict[str, dict]:
        """Load video metadata from database."""
        if not video_ids:
            return {}
        
        # Check cache first
        uncached = [vid for vid in video_ids if vid not in self._video_cache]
        
        if uncached:
            placeholders = ','.join(['%s'] * len(uncached))
            query = f"""
                SELECT video_id, title, category_id, channel_id, 
                       published_at, duration_seconds, view_count, thumbnail_url
                FROM videos
                WHERE video_id IN ({placeholders})
            """
            
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, uncached)
                    for row in cur.fetchall():
                        self._video_cache[row[0]] = {
                            'video_id': row[0],
                            'title': row[1],
                            'category_id': row[2],
                            'channel_id': row[3],
                            'published_at': row[4],
                            'duration_seconds': row[5],
                            'view_count': row[6],
                            'thumbnail_url': row[7],
                        }
        
        return {vid: self._video_cache.get(vid, {}) for vid in video_ids}
    
    def _load_video_embeddings(self, video_ids: List[str]) -> Dict[str, np.ndarray]:
        """Load video embeddings from database."""
        if not video_ids:
            return {}
        
        uncached = [vid for vid in video_ids if vid not in self._embedding_cache]
        
        if uncached:
            placeholders = ','.join(['%s'] * len(uncached))
            query = f"""
                SELECT video_id, combined_embedding
                FROM video_embeddings
                WHERE video_id IN ({placeholders})
            """
            
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, uncached)
                    for row in cur.fetchall():
                        emb = row[1]
                        if isinstance(emb, str):
                            emb = emb.strip('[]')
                            emb = [float(x) for x in emb.split(',')]
                        self._embedding_cache[row[0]] = np.array(emb, dtype=np.float32)
        
        return {vid: self._embedding_cache.get(vid) for vid in video_ids}
    
    def _load_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Load user embedding from database."""
        query = """
            SELECT user_embedding FROM user_embeddings WHERE user_id = %s
        """
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (user_id,))
                row = cur.fetchone()
                
                if row and row[0]:
                    emb = row[0]
                    if isinstance(emb, str):
                        emb = emb.strip('[]')
                        emb = [float(x) for x in emb.split(',')]
                    return np.array(emb, dtype=np.float32)
        
        return None
    
    def _get_user_watch_history(self, user_id: str, limit: int = 100) -> List[str]:
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
    
    def retrieve(
        self,
        user_embedding: np.ndarray,
        k: int = 500,
        exclude_video_ids: List[str] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Stage 1: Retrieval
        
        Get top-k candidates using ANN search.
        """
        if self.retrieval_service is None:
            logger.warning("Retrieval service not initialized")
            return [], np.array([])
        
        video_ids, scores = self.retrieval_service.retrieve(
            user_embedding,
            k=k,
            exclude_video_ids=exclude_video_ids,
        )
        
        return video_ids, scores
    
    def rank(
        self,
        user_embedding: np.ndarray,
        video_ids: List[str],
        retrieval_scores: np.ndarray = None,
    ) -> List[Tuple[str, float]]:
        """
        Stage 2: Ranking
        
        Score candidates with ranking model.
        """
        if not video_ids:
            return []
        
        # If no ranking model, use retrieval scores
        if self.ranking_model is None:
            if retrieval_scores is not None:
                return list(zip(video_ids, retrieval_scores.tolist()))
            else:
                return [(vid, 1.0) for vid in video_ids]
        
        # Load video embeddings
        video_embeddings = self._load_video_embeddings(video_ids)
        
        # Prepare tensors
        user_tensor = torch.tensor(user_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        user_batch = user_tensor.expand(len(video_ids), -1)
        
        video_embs = []
        valid_video_ids = []
        for vid in video_ids:
            if vid in video_embeddings and video_embeddings[vid] is not None:
                video_embs.append(video_embeddings[vid])
                valid_video_ids.append(vid)
        
        if not video_embs:
            return [(vid, 1.0) for vid in video_ids]
        
        video_tensor = torch.tensor(np.array(video_embs), dtype=torch.float32).to(self.device)
        user_batch = user_tensor.expand(len(video_embs), -1)
        
        # Get ranking scores
        with torch.no_grad():
            scores = self.ranking_model.predict_proba(user_batch, video_tensor)
            scores = scores.cpu().numpy()
        
        return list(zip(valid_video_ids, scores.tolist()))
    
    def rerank(
        self,
        ranked_candidates: List[Tuple[str, float]],
        k: int = 20,
    ) -> List[VideoCandidate]:
        """
        Stage 3: Re-ranking
        
        Apply diversity and business rules.
        """
        if not ranked_candidates:
            return []
        
        video_ids = [vid for vid, _ in ranked_candidates]
        scores = {vid: score for vid, score in ranked_candidates}
        
        # Load metadata and embeddings
        metadata = self._load_video_metadata(video_ids)
        embeddings = self._load_video_embeddings(video_ids)
        
        # Create VideoCandidate objects
        candidates = []
        for vid in video_ids:
            meta = metadata.get(vid, {})
            candidates.append(VideoCandidate(
                video_id=vid,
                score=scores[vid],
                embedding=embeddings.get(vid),
                category_id=meta.get('category_id'),
                channel_id=meta.get('channel_id'),
                published_at=meta.get('published_at'),
                duration_seconds=meta.get('duration_seconds'),
                view_count=meta.get('view_count'),
            ))
        
        # Apply re-ranking
        reranked = self.reranking_pipeline.rerank(candidates, k=k)
        
        return reranked
    
    def recommend(
        self,
        user_id: str,
        k: int = 20,
        retrieval_k: int = 500,
        exclude_watched: bool = True,
    ) -> List[Dict]:
        """
        Full recommendation pipeline.
        
        Args:
            user_id: User ID
            k: Number of final recommendations
            retrieval_k: Number of candidates to retrieve
            exclude_watched: Whether to exclude watched videos
            
        Returns:
            List of recommendation dicts with video info and scores
        """
        # Load user embedding
        user_embedding = self._load_user_embedding(user_id)
        
        if user_embedding is None:
            logger.warning(f"No embedding found for user {user_id}")
            return []
        
        # Get watched videos to exclude
        exclude_ids = None
        if exclude_watched:
            exclude_ids = self._get_user_watch_history(user_id)
        
        # Stage 1: Retrieval
        logger.info(f"Stage 1: Retrieving {retrieval_k} candidates...")
        retrieved_ids, retrieval_scores = self.retrieve(
            user_embedding,
            k=retrieval_k,
            exclude_video_ids=exclude_ids,
        )
        logger.info(f"Retrieved {len(retrieved_ids)} candidates")
        
        if not retrieved_ids:
            return []
        
        # Stage 2: Ranking
        logger.info("Stage 2: Ranking candidates...")
        ranked = self.rank(user_embedding, retrieved_ids, retrieval_scores)
        logger.info(f"Ranked {len(ranked)} candidates")
        
        # Stage 3: Re-ranking
        logger.info(f"Stage 3: Re-ranking to top {k}...")
        reranked = self.rerank(ranked, k=k)
        logger.info(f"Final {len(reranked)} recommendations")
        
        # Load metadata for final results
        final_ids = [c.video_id for c in reranked]
        metadata = self._load_video_metadata(final_ids)
        
        # Build response
        results = []
        for candidate in reranked:
            meta = metadata.get(candidate.video_id, {})
            results.append({
                'video_id': candidate.video_id,
                'score': candidate.score,
                'title': meta.get('title'),
                'category_id': meta.get('category_id'),
                'channel_id': meta.get('channel_id'),
                'thumbnail_url': meta.get('thumbnail_url'),
                'duration_seconds': meta.get('duration_seconds'),
            })
        
        return results


if __name__ == "__main__":
    print("Testing Recommendation Pipeline...")
    
    # Initialize pipeline (without trained models)
    pipeline = RecommendationPipeline()
    
    # Test with dummy user
    print("\nNote: This test requires data in the database.")
    print("Run the following first if you haven't:")
    print("  python scripts/load_data.py --videos data/raw/videos.json")
    print("  python scripts/load_data.py --simulate-users 100 --simulate-days 7")
    print("  python scripts/generate_embeddings.py --all --skip-thumbnails")
    print("  python scripts/build_index.py")
    
    # Try to get a user ID from database
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id::text FROM users LIMIT 1")
                row = cur.fetchone()
                
                if row:
                    user_id = row[0]
                    print(f"\nTesting with user: {user_id[:8]}...")
                    
                    # This will fail without trained models, but tests the pipeline structure
                    results = pipeline.recommend(user_id, k=10)
                    print(f"Got {len(results)} recommendations")
                    
                    for i, rec in enumerate(results[:5]):
                        print(f"  {i+1}. {rec['video_id']} (score: {rec['score']:.3f})")
                else:
                    print("\nNo users found in database.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the database is running and has data.")