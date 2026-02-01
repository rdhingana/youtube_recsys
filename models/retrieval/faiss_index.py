"""
FAISS Index Manager

Manages FAISS index for fast approximate nearest neighbor search.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import faiss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS Index for video retrieval.
    
    Supports multiple index types:
    - Flat: Exact search (slow for large datasets)
    - IVF: Inverted file index (faster, approximate)
    - HNSW: Hierarchical navigable small world (fast, good recall)
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        index_type: str = "IVF",
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10,  # Number of clusters to search
        use_gpu: bool = False,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        
        self.index = None
        self.video_ids = []  # Maps index position to video_id
        self.id_to_idx = {}  # Maps video_id to index position
        
        self._is_trained = False
    
    def _create_index(self, num_vectors: int = None) -> faiss.Index:
        """Create FAISS index based on type."""
        
        if self.index_type == "Flat":
            # Exact search - good for small datasets
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine for normalized vectors)
            
        elif self.index_type == "IVF":
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(self.nlist, num_vectors // 10) if num_vectors else self.nlist
            nlist = max(1, nlist)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
        elif self.index_type == "HNSW":
            # HNSW - fast and good recall
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if available and requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        
        return index
    
    def build(self, video_ids: list, embeddings: np.ndarray):
        """
        Build the index from embeddings.
        
        Args:
            video_ids: List of video IDs
            embeddings: numpy array of shape (num_videos, embedding_dim)
        """
        if len(video_ids) != len(embeddings):
            raise ValueError("video_ids and embeddings must have same length")
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Store mappings
        self.video_ids = list(video_ids)
        self.id_to_idx = {vid: idx for idx, vid in enumerate(self.video_ids)}
        
        # Create and train index
        self.index = self._create_index(num_vectors=len(embeddings))
        
        if self.index_type == "IVF":
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        
        # Add vectors
        logger.info(f"Adding {len(embeddings)} vectors to index...")
        self.index.add(embeddings)
        
        self._is_trained = True
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
        exclude_ids: list = None,
    ) -> Tuple[list, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: Query vector (embedding_dim,) or (n_queries, embedding_dim)
            k: Number of results to return
            exclude_ids: Video IDs to exclude from results
            
        Returns:
            Tuple of (video_ids, scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Ensure correct shape
        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize query
        faiss.normalize_L2(query)
        
        # Search more if we need to exclude some
        search_k = k
        if exclude_ids:
            search_k = min(k + len(exclude_ids) + 10, self.index.ntotal)
        
        # Search
        scores, indices = self.index.search(query, search_k)
        
        # Convert to video IDs and filter
        results_ids = []
        results_scores = []
        
        for i in range(len(query)):
            ids = []
            scs = []
            for idx, score in zip(indices[i], scores[i]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                vid = self.video_ids[idx]
                if exclude_ids and vid in exclude_ids:
                    continue
                ids.append(vid)
                scs.append(score)
                if len(ids) >= k:
                    break
            results_ids.append(ids)
            results_scores.append(scs)
        
        # If single query, return flat lists
        if len(query) == 1:
            return results_ids[0], np.array(results_scores[0])
        
        return results_ids, results_scores
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, str(path.with_suffix('.faiss')))
        else:
            faiss.write_index(self.index, str(path.with_suffix('.faiss')))
        
        # Save metadata
        metadata = {
            'video_ids': self.video_ids,
            'id_to_idx': self.id_to_idx,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
        }
        
        with open(path.with_suffix('.meta'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))
        
        # Load metadata
        with open(path.with_suffix('.meta'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.video_ids = metadata['video_ids']
        self.id_to_idx = metadata['id_to_idx']
        self.embedding_dim = metadata['embedding_dim']
        self.index_type = metadata['index_type']
        self.nlist = metadata['nlist']
        self.nprobe = metadata['nprobe']
        
        if self.index_type == "IVF":
            self.index.nprobe = self.nprobe
        
        self._is_trained = True
        logger.info(f"Index loaded from {path} with {self.index.ntotal} vectors")
    
    def get_embedding(self, video_id: str) -> Optional[np.ndarray]:
        """Get embedding for a video ID."""
        if video_id not in self.id_to_idx:
            return None
        
        idx = self.id_to_idx[video_id]
        return self.index.reconstruct(idx)
    
    @property
    def num_vectors(self) -> int:
        """Number of vectors in index."""
        return self.index.ntotal if self.index else 0


if __name__ == "__main__":
    # Test FAISS index
    print("Testing FAISS Index...")
    
    # Create dummy data
    num_videos = 1000
    embedding_dim = 128
    
    np.random.seed(42)
    video_ids = [f"video_{i}" for i in range(num_videos)]
    embeddings = np.random.randn(num_videos, embedding_dim).astype(np.float32)
    
    # Test different index types
    for index_type in ["Flat", "IVF"]:
        print(f"\nTesting {index_type} index...")
        
        index = FAISSIndex(
            embedding_dim=embedding_dim,
            index_type=index_type,
            nlist=50,
            nprobe=10,
        )
        
        # Build index
        index.build(video_ids, embeddings)
        print(f"  Built index with {index.num_vectors} vectors")
        
        # Search
        query = np.random.randn(embedding_dim).astype(np.float32)
        result_ids, scores = index.search(query, k=10)
        print(f"  Top 10 results: {result_ids[:5]}...")
        print(f"  Scores: {scores[:5]}")
        
        # Test with exclusion
        result_ids, scores = index.search(query, k=10, exclude_ids=result_ids[:3])
        print(f"  After excluding top 3: {result_ids[:5]}...")
    
    print("\nFAISS Index test passed!")