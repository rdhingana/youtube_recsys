"""
User Encoder

Generates user embeddings by aggregating their watch history.
"""

import logging
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserEncoder:
    """Encodes user preferences from watch history."""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
    
    def encode_user(
        self,
        watched_video_embeddings: list,
        watch_percentages: list = None,
        interaction_types: list = None,
        decay_factor: float = 0.95,
    ) -> np.ndarray:
        """
        Create user embedding from watch history.
        
        Args:
            watched_video_embeddings: List of video combined_embeddings (most recent last)
            watch_percentages: Optional list of watch completion percentages (0-1)
            interaction_types: Optional list of interaction types ('view', 'like', etc.)
            decay_factor: Time decay factor (recent videos weighted more)
            
        Returns:
            User embedding vector
        """
        if not watched_video_embeddings:
            return np.zeros(self.embedding_dim)
        
        embeddings = np.array(watched_video_embeddings)
        n = len(embeddings)
        
        # Base weights: time decay (more recent = higher weight)
        time_weights = np.array([decay_factor ** (n - i - 1) for i in range(n)])
        
        # Adjust by watch percentage if provided
        if watch_percentages:
            watch_weights = np.array(watch_percentages)
            time_weights = time_weights * watch_weights
        
        # Boost for likes
        if interaction_types:
            interaction_weights = np.array([
                1.5 if t == 'like' else (0.5 if t == 'dislike' else 1.0)
                for t in interaction_types
            ])
            time_weights = time_weights * interaction_weights
        
        # Normalize weights
        time_weights = time_weights / time_weights.sum()
        
        # Weighted average of embeddings
        user_embedding = np.average(embeddings, axis=0, weights=time_weights)
        
        # Normalize
        norm = np.linalg.norm(user_embedding)
        if norm > 0:
            user_embedding = user_embedding / norm
        
        return user_embedding
    
    def encode_user_with_categories(
        self,
        watched_video_embeddings: list,
        video_categories: list,
        watch_percentages: list = None,
        category_boost: float = 0.2,
    ) -> np.ndarray:
        """
        Create user embedding with category preference boosting.
        
        Args:
            watched_video_embeddings: List of video embeddings
            video_categories: List of category IDs for each video
            watch_percentages: Optional watch percentages
            category_boost: How much to boost frequently watched categories
            
        Returns:
            User embedding vector
        """
        if not watched_video_embeddings:
            return np.zeros(self.embedding_dim)
        
        # Get base embedding
        user_embedding = self.encode_user(
            watched_video_embeddings,
            watch_percentages=watch_percentages,
        )
        
        # Calculate category preferences
        category_counts = {}
        for cat in video_categories:
            if cat:
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Store category preferences (could be used for filtering/boosting)
        self.category_preferences = category_counts
        
        return user_embedding
    
    def compute_similarity(
        self,
        user_embedding: np.ndarray,
        video_embedding: np.ndarray,
    ) -> float:
        """Compute cosine similarity between user and video."""
        if user_embedding is None or video_embedding is None:
            return 0.0
        
        dot_product = np.dot(user_embedding, video_embedding)
        norm_user = np.linalg.norm(user_embedding)
        norm_video = np.linalg.norm(video_embedding)
        
        if norm_user == 0 or norm_video == 0:
            return 0.0
        
        return dot_product / (norm_user * norm_video)
    
    def rank_videos(
        self,
        user_embedding: np.ndarray,
        video_embeddings: list,
        video_ids: list,
    ) -> list:
        """
        Rank videos by similarity to user.
        
        Returns:
            List of (video_id, score) tuples sorted by score descending
        """
        scores = []
        
        for vid, emb in zip(video_ids, video_embeddings):
            score = self.compute_similarity(user_embedding, emb)
            scores.append((vid, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores


class BatchUserEncoder:
    """Batch encoding for multiple users."""
    
    def __init__(self, encoder: UserEncoder = None):
        self.encoder = encoder or UserEncoder()
    
    def encode_users(
        self,
        user_histories: dict,
        show_progress: bool = True,
    ) -> dict:
        """
        Encode multiple users.
        
        Args:
            user_histories: Dict mapping user_id -> {
                'embeddings': list of video embeddings,
                'watch_percentages': list of watch percentages,
                'interaction_types': list of interaction types
            }
            
        Returns:
            Dict mapping user_id -> user_embedding
        """
        from tqdm import tqdm
        
        results = {}
        iterator = tqdm(user_histories.items(), desc="Encoding users") if show_progress else user_histories.items()
        
        for user_id, history in iterator:
            try:
                embedding = self.encoder.encode_user(
                    watched_video_embeddings=history.get('embeddings', []),
                    watch_percentages=history.get('watch_percentages'),
                    interaction_types=history.get('interaction_types'),
                )
                results[user_id] = embedding
            except Exception as e:
                logger.error(f"Error encoding user {user_id}: {e}")
                results[user_id] = np.zeros(self.encoder.embedding_dim)
        
        return results


if __name__ == "__main__":
    # Test the encoder
    print("Testing UserEncoder...")
    
    encoder = UserEncoder(embedding_dim=256)
    
    # Create dummy video embeddings
    np.random.seed(42)
    dummy_embeddings = [np.random.randn(256) for _ in range(10)]
    dummy_embeddings = [e / np.linalg.norm(e) for e in dummy_embeddings]  # Normalize
    
    watch_percentages = [0.3, 0.5, 0.8, 0.9, 1.0, 0.4, 0.7, 0.95, 0.6, 0.85]
    interaction_types = ['view', 'view', 'like', 'view', 'like', 'view', 'view', 'like', 'view', 'view']
    
    user_embedding = encoder.encode_user(
        watched_video_embeddings=dummy_embeddings,
        watch_percentages=watch_percentages,
        interaction_types=interaction_types,
    )
    
    print(f"\nUser embedding shape: {user_embedding.shape}")
    print(f"User embedding norm: {np.linalg.norm(user_embedding):.4f}")
    
    # Test ranking
    print("\nTesting video ranking...")
    new_videos = [np.random.randn(256) for _ in range(5)]
    new_videos = [e / np.linalg.norm(e) for e in new_videos]
    video_ids = ['vid_1', 'vid_2', 'vid_3', 'vid_4', 'vid_5']
    
    rankings = encoder.rank_videos(user_embedding, new_videos, video_ids)
    
    print("Rankings:")
    for vid, score in rankings:
        print(f"  {vid}: {score:.4f}")