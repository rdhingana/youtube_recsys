"""
Build FAISS Index

Builds FAISS index from video embeddings without training the two-tower model.
Useful for quick testing with raw embeddings.
"""

import os
import sys
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.retrieval.faiss_index import FAISSIndex
from models.retrieval.retrieval_service import SimpleRetrievalService

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def load_video_embeddings():
    """Load video embeddings from database."""
    print("Loading video embeddings...")
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT video_id, combined_embedding 
                FROM video_embeddings 
                WHERE combined_embedding IS NOT NULL
            """)
            results = cur.fetchall()
    
    video_ids = []
    embeddings = []
    
    for row in results:
        video_ids.append(row[0])
        emb = row[1]
        
        # Handle different formats (string vs list)
        if isinstance(emb, str):
            # Parse string format: '[0.1, 0.2, ...]'
            emb = emb.strip('[]')
            emb = [float(x) for x in emb.split(',')]
        
        embeddings.append(emb)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    print(f"Loaded {len(video_ids)} video embeddings")
    return video_ids, embeddings


def load_user_embeddings():
    """Load user embeddings from database."""
    print("Loading user embeddings...")
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_id::text, user_embedding 
                FROM user_embeddings 
                WHERE user_embedding IS NOT NULL
            """)
            results = cur.fetchall()
    
    user_embeddings = {}
    for row in results:
        emb = row[1]
        
        # Handle different formats (string vs list)
        if isinstance(emb, str):
            emb = emb.strip('[]')
            emb = [float(x) for x in emb.split(',')]
        
        user_embeddings[row[0]] = np.array(emb, dtype=np.float32)
    
    print(f"Loaded {len(user_embeddings)} user embeddings")
    return user_embeddings


def test_retrieval(service, user_embeddings, n_tests: int = 5):
    """Test retrieval for a few users."""
    print(f"\nTesting retrieval for {n_tests} users...")
    
    user_ids = list(user_embeddings.keys())[:n_tests]
    
    for user_id in user_ids:
        user_emb = user_embeddings[user_id]
        
        video_ids, scores = service.retrieve(user_emb, k=10)
        
        print(f"\nUser {user_id[:8]}...")
        print(f"  Top 5 videos: {video_ids[:5]}")
        print(f"  Scores: {[f'{s:.3f}' for s in scores[:5]]}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS Index")
    parser.add_argument("--output-dir", type=str, default="models/retrieval/saved")
    parser.add_argument("--index-type", type=str, default="IVF", choices=["Flat", "IVF", "HNSW"])
    parser.add_argument("--test", action="store_true", help="Test retrieval after building")
    
    args = parser.parse_args()
    
    # Load embeddings
    video_ids, video_embeddings = load_video_embeddings()
    
    if len(video_ids) == 0:
        print("No video embeddings found. Run generate_embeddings.py first.")
        return
    
    # Create service
    embedding_dim = video_embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    service = SimpleRetrievalService(embedding_dim=embedding_dim)
    
    # Build index
    print(f"\nBuilding {args.index_type} index...")
    service.index = FAISSIndex(
        embedding_dim=embedding_dim,
        index_type=args.index_type,
        nlist=min(100, len(video_ids) // 10),
        nprobe=10,
    )
    service.build_index(video_ids, video_embeddings)
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / "simple_faiss_index"
    service.save(str(index_path))
    print(f"\nIndex saved to {index_path}")
    
    # Test
    if args.test:
        user_embeddings = load_user_embeddings()
        if user_embeddings:
            test_retrieval(service, user_embeddings)
        else:
            print("No user embeddings found for testing.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()