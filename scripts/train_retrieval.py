"""
Train Two-Tower Retrieval Model

Trains the two-tower model using user-video interactions.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.retrieval.two_tower import TwoTowerModel, TwoTowerDataset
from models.retrieval.faiss_index import FAISSIndex
from models.retrieval.retrieval_service import RetrievalService

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def load_embeddings_from_db():
    """Load user and video embeddings from database."""
    print("Loading embeddings from database...")
    
    def parse_embedding(emb):
        """Parse embedding from string or list format."""
        if isinstance(emb, str):
            emb = emb.strip('[]')
            emb = [float(x) for x in emb.split(',')]
        return np.array(emb, dtype=np.float32)
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Load video embeddings
            cur.execute("""
                SELECT video_id, combined_embedding 
                FROM video_embeddings 
                WHERE combined_embedding IS NOT NULL
            """)
            video_embeddings = {row[0]: parse_embedding(row[1]) for row in cur.fetchall()}
            
            # Load user embeddings
            cur.execute("""
                SELECT user_id::text, user_embedding 
                FROM user_embeddings 
                WHERE user_embedding IS NOT NULL
            """)
            user_embeddings = {row[0]: parse_embedding(row[1]) for row in cur.fetchall()}
    
    print(f"Loaded {len(video_embeddings)} video embeddings")
    print(f"Loaded {len(user_embeddings)} user embeddings")
    
    return user_embeddings, video_embeddings


def load_interactions_from_db(min_watch_pct: float = 0.3):
    """Load positive interactions from database."""
    print("Loading interactions from database...")
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get positive interactions (watched > min_watch_pct or liked)
            cur.execute("""
                SELECT DISTINCT user_id::text, video_id
                FROM user_interactions
                WHERE (watch_percentage >= %s OR interaction_type = 'like')
            """, (min_watch_pct,))
            
            interactions = [(row[0], row[1], 1.0) for row in cur.fetchall()]
    
    print(f"Loaded {len(interactions)} positive interactions")
    return interactions


class InteractionDataset(Dataset):
    """Dataset with in-batch negative sampling."""
    
    def __init__(self, user_embeddings, video_embeddings, interactions):
        self.user_embeddings = user_embeddings
        self.video_embeddings = video_embeddings
        
        # Filter to valid interactions
        self.interactions = [
            (u, v) for u, v, _ in interactions
            if u in user_embeddings and v in video_embeddings
        ]
        
        print(f"Dataset size: {len(self.interactions)} valid interactions")
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, video_id = self.interactions[idx]
        
        user_emb = torch.tensor(self.user_embeddings[user_id], dtype=torch.float32)
        video_emb = torch.tensor(self.video_embeddings[video_id], dtype=torch.float32)
        
        return user_emb, video_emb


def train_model(
    model: TwoTowerModel,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
):
    """Train the two-tower model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for user_emb, video_emb in pbar:
            user_emb = user_emb.to(device)
            video_emb = video_emb.to(device)
            
            # Compute contrastive loss with in-batch negatives
            loss = model.compute_contrastive_loss(user_emb, video_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for user_emb, video_emb in val_loader:
                    user_emb = user_emb.to(device)
                    video_emb = video_emb.to(device)
                    
                    loss = model.compute_contrastive_loss(user_emb, video_emb)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            print(f"         Val Loss = {avg_val_loss:.4f}")
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
    
    return model


def evaluate_recall(
    model: TwoTowerModel,
    user_embeddings: dict,
    video_embeddings: dict,
    test_interactions: list,
    k_values: list = [10, 50, 100],
    device: str = "cpu",
):
    """Evaluate recall@k."""
    print("\nEvaluating recall...")
    
    model.eval()
    model.to(device)
    
    # Build index with video tower
    video_ids = list(video_embeddings.keys())
    video_emb_array = np.array([video_embeddings[vid] for vid in video_ids])
    
    with torch.no_grad():
        video_features = torch.tensor(video_emb_array, dtype=torch.float32).to(device)
        encoded_videos = model.encode_video(video_features).cpu().numpy()
    
    # Build FAISS index
    index = FAISSIndex(embedding_dim=model.embedding_dim)
    index.build(video_ids, encoded_videos)
    
    # Group test interactions by user
    user_positives = {}
    for user_id, video_id, _ in test_interactions:
        if user_id in user_embeddings and video_id in video_embeddings:
            if user_id not in user_positives:
                user_positives[user_id] = set()
            user_positives[user_id].add(video_id)
    
    # Compute recall for each user
    recalls = {k: [] for k in k_values}
    
    for user_id, positive_videos in tqdm(user_positives.items(), desc="Computing recall"):
        user_emb = user_embeddings[user_id]
        
        with torch.no_grad():
            user_features = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(device)
            encoded_user = model.encode_user(user_features).cpu().numpy().flatten()
        
        # Get top-k recommendations
        max_k = max(k_values)
        retrieved_ids, _ = index.search(encoded_user, k=max_k)
        
        for k in k_values:
            retrieved_set = set(retrieved_ids[:k])
            hits = len(retrieved_set & positive_videos)
            recall = hits / len(positive_videos)
            recalls[k].append(recall)
    
    # Print results
    print("\nRecall Results:")
    for k in k_values:
        avg_recall = np.mean(recalls[k])
        print(f"  Recall@{k}: {avg_recall:.4f}")
    
    return recalls


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Two-Tower Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="models/retrieval/saved")
    parser.add_argument("--eval-only", action="store_true")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    user_embeddings, video_embeddings = load_embeddings_from_db()
    interactions = load_interactions_from_db()
    
    if len(interactions) == 0:
        print("No interactions found. Please load data first.")
        return
    
    # Split data
    np.random.seed(42)
    np.random.shuffle(interactions)
    
    split_idx = int(len(interactions) * 0.9)
    train_interactions = interactions[:split_idx]
    val_interactions = interactions[split_idx:]
    
    print(f"Train: {len(train_interactions)}, Val: {len(val_interactions)}")
    
    # Create datasets
    train_dataset = InteractionDataset(user_embeddings, video_embeddings, train_interactions)
    val_dataset = InteractionDataset(user_embeddings, video_embeddings, val_interactions)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # Create model
    model = TwoTowerModel(
        user_input_dim=256,
        video_input_dim=256,
        embedding_dim=args.embedding_dim,
    )
    
    output_dir = Path(args.output_dir)
    model_path = output_dir / "two_tower_model.pt"
    
    if args.eval_only and model_path.exists():
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Train
        print("\nStarting training...")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
        )
        
        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Evaluate
    evaluate_recall(
        model=model,
        user_embeddings=user_embeddings,
        video_embeddings=video_embeddings,
        test_interactions=val_interactions,
        k_values=[10, 50, 100],
        device=device,
    )
    
    # Build and save FAISS index
    print("\nBuilding FAISS index...")
    service = RetrievalService(embedding_dim=args.embedding_dim, device=device)
    service.model = model
    
    video_ids = list(video_embeddings.keys())
    video_emb_array = np.array([video_embeddings[vid] for vid in video_ids])
    service.build_index(video_ids, video_emb_array)
    
    service.save(str(output_dir))
    print(f"Retrieval service saved to {output_dir}")


if __name__ == "__main__":
    main()