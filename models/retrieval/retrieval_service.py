"""
Retrieval Service

Combines Two-Tower model and FAISS index for candidate retrieval.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch

from .two_tower import TwoTowerModel
from .faiss_index import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Retrieval service for candidate generation.
    
    Uses Two-Tower model to encode user/video and FAISS for ANN search.
    """
    
    def __init__(
        self,
        model_path: str = None,
        index_path: str = None,
        embedding_dim: int = 128,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        
        # Initialize model
        self.model = TwoTowerModel(
            user_input_dim=256,
            video_input_dim=256,
            embedding_dim=embedding_dim,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load model weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # Initialize FAISS index
        self.index = FAISSIndex(embedding_dim=embedding_dim)
        
        # Load index if provided
        if index_path and Path(index_path).with_suffix('.faiss').exists():
            self.index.load(index_path)
    
    def load_model(self, path: str):
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Model loaded from {path}")
    
    def save_model(self, path: str):
        """Save model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def build_index(self, video_ids: list, video_embeddings: np.ndarray):
        """
        Build FAISS index from video embeddings.
        
        Args:
            video_ids: List of video IDs
            video_embeddings: Raw video embeddings (256-dim)
        """
        # Encode videos through the video tower
        logger.info("Encoding videos through video tower...")
        
        with torch.no_grad():
            video_features = torch.tensor(video_embeddings, dtype=torch.float32).to(self.device)
            
            # Process in batches
            batch_size = 512
            encoded_embeddings = []
            
            for i in range(0, len(video_features), batch_size):
                batch = video_features[i:i + batch_size]
                encoded = self.model.encode_video(batch)
                encoded_embeddings.append(encoded.cpu().numpy())
            
            encoded_embeddings = np.vstack(encoded_embeddings)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        self.index.build(video_ids, encoded_embeddings)
    
    def retrieve(
        self,
        user_embedding: np.ndarray,
        k: int = 100,
        exclude_video_ids: list = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Retrieve top-k candidate videos for a user.
        
        Args:
            user_embedding: User embedding (256-dim raw embedding)
            k: Number of candidates to retrieve
            exclude_video_ids: Video IDs to exclude (e.g., already watched)
            
        Returns:
            Tuple of (video_ids, scores)
        """
        # Encode user through user tower
        with torch.no_grad():
            user_features = torch.tensor(user_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            user_encoded = self.model.encode_user(user_features)
            user_encoded = user_encoded.cpu().numpy().flatten()
        
        # Search FAISS index
        video_ids, scores = self.index.search(
            user_encoded,
            k=k,
            exclude_ids=exclude_video_ids,
        )
        
        return video_ids, scores
    
    def retrieve_batch(
        self,
        user_embeddings: np.ndarray,
        k: int = 100,
    ) -> Tuple[List[List[str]], List[np.ndarray]]:
        """
        Retrieve candidates for multiple users.
        
        Args:
            user_embeddings: User embeddings (n_users, 256)
            k: Number of candidates per user
            
        Returns:
            Tuple of (list of video_id lists, list of score arrays)
        """
        # Encode users
        with torch.no_grad():
            user_features = torch.tensor(user_embeddings, dtype=torch.float32).to(self.device)
            user_encoded = self.model.encode_user(user_features)
            user_encoded = user_encoded.cpu().numpy()
        
        # Search for each user
        all_ids = []
        all_scores = []
        
        for user_emb in user_encoded:
            video_ids, scores = self.index.search(user_emb, k=k)
            all_ids.append(video_ids)
            all_scores.append(scores)
        
        return all_ids, all_scores
    
    def save(self, directory: str):
        """Save model and index."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        self.save_model(str(directory / "two_tower_model.pt"))
        self.index.save(str(directory / "faiss_index"))
        
        logger.info(f"Retrieval service saved to {directory}")
    
    def load(self, directory: str):
        """Load model and index."""
        directory = Path(directory)
        
        self.load_model(str(directory / "two_tower_model.pt"))
        self.index.load(str(directory / "faiss_index"))
        
        logger.info(f"Retrieval service loaded from {directory}")


class SimpleRetrievalService:
    """
    Simple retrieval service without Two-Tower model.
    
    Uses raw embeddings directly with FAISS (for when model is not trained).
    """
    
    def __init__(
        self,
        index_path: str = None,
        embedding_dim: int = 256,
    ):
        self.embedding_dim = embedding_dim
        self.index = FAISSIndex(embedding_dim=embedding_dim, index_type="IVF")
        
        if index_path and Path(index_path).with_suffix('.faiss').exists():
            self.index.load(index_path)
    
    def build_index(self, video_ids: list, video_embeddings: np.ndarray):
        """Build index from raw embeddings."""
        self.index.build(video_ids, video_embeddings)
    
    def retrieve(
        self,
        user_embedding: np.ndarray,
        k: int = 100,
        exclude_video_ids: list = None,
    ) -> Tuple[List[str], np.ndarray]:
        """Retrieve using raw embedding similarity."""
        return self.index.search(user_embedding, k=k, exclude_ids=exclude_video_ids)
    
    def save(self, path: str):
        """Save index."""
        self.index.save(path)
    
    def load(self, path: str):
        """Load index."""
        self.index.load(path)


if __name__ == "__main__":
    # Test retrieval service
    print("Testing Retrieval Service...")
    
    # Create dummy data
    num_videos = 500
    num_users = 10
    
    np.random.seed(42)
    video_ids = [f"video_{i}" for i in range(num_videos)]
    video_embeddings = np.random.randn(num_videos, 256).astype(np.float32)
    user_embeddings = np.random.randn(num_users, 256).astype(np.float32)
    
    # Test SimpleRetrievalService (no model training needed)
    print("\nTesting SimpleRetrievalService...")
    simple_service = SimpleRetrievalService(embedding_dim=256)
    simple_service.build_index(video_ids, video_embeddings)
    
    # Retrieve for one user
    results, scores = simple_service.retrieve(user_embeddings[0], k=10)
    print(f"Top 10 for user 0: {results[:5]}...")
    print(f"Scores: {scores[:5]}")
    
    # Test with exclusion
    results, scores = simple_service.retrieve(
        user_embeddings[0], 
        k=10, 
        exclude_video_ids=results[:3]
    )
    print(f"After exclusion: {results[:5]}...")
    
    # Test full RetrievalService
    print("\nTesting RetrievalService with Two-Tower model...")
    service = RetrievalService(embedding_dim=128)
    service.build_index(video_ids, video_embeddings)
    
    results, scores = service.retrieve(user_embeddings[0], k=10)
    print(f"Top 10 for user 0: {results[:5]}...")
    print(f"Scores: {scores[:5]}")
    
    print("\nRetrieval Service test passed!")