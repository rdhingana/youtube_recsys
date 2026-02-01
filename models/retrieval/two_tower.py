"""
Two-Tower Model for Video Retrieval

User Tower: Encodes user features into embedding
Video Tower: Encodes video features into embedding
Similarity: Dot product between user and video embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UserTower(nn.Module):
    """
    User Tower: Transforms user features into a dense embedding.
    
    Input: User embedding from watch history (256-dim)
    Output: User retrieval embedding (128-dim)
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.network(x)
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)
        return x


class VideoTower(nn.Module):
    """
    Video Tower: Transforms video features into a dense embedding.
    
    Input: Video combined embedding (256-dim)
    Output: Video retrieval embedding (128-dim)
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.network(x)
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)
        return x


class TwoTowerModel(nn.Module):
    """
    Two-Tower Model for retrieval.
    
    Combines user and video towers with contrastive learning.
    """
    
    def __init__(
        self,
        user_input_dim: int = 256,
        video_input_dim: int = 256,
        embedding_dim: int = 128,
        hidden_dims: list = [256, 128],
        dropout: float = 0.2,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        self.user_tower = UserTower(
            input_dim=user_input_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        
        self.video_tower = VideoTower(
            input_dim=video_input_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        
        self.temperature = temperature
        self.embedding_dim = embedding_dim
    
    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features."""
        return self.user_tower(user_features)
    
    def encode_video(self, video_features: torch.Tensor) -> torch.Tensor:
        """Encode video features."""
        return self.video_tower(video_features)
    
    def forward(
        self,
        user_features: torch.Tensor,
        video_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass computing similarity scores.
        
        Args:
            user_features: (batch_size, user_input_dim)
            video_features: (batch_size, video_input_dim)
            
        Returns:
            Similarity scores: (batch_size,)
        """
        user_emb = self.encode_user(user_features)
        video_emb = self.encode_video(video_features)
        
        # Dot product similarity (embeddings are already normalized)
        similarity = torch.sum(user_emb * video_emb, dim=-1)
        
        return similarity
    
    def compute_contrastive_loss(
        self,
        user_features: torch.Tensor,
        positive_video_features: torch.Tensor,
        negative_video_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE / NT-Xent style).
        
        Uses in-batch negatives if negative_video_features is None.
        """
        user_emb = self.encode_user(user_features)  # (batch_size, emb_dim)
        pos_video_emb = self.encode_video(positive_video_features)  # (batch_size, emb_dim)
        
        if negative_video_features is None:
            # In-batch negatives: all other videos in batch are negatives
            # Compute all pairwise similarities
            logits = torch.matmul(user_emb, pos_video_emb.T) / self.temperature  # (batch_size, batch_size)
            
            # Labels: diagonal elements are positives (user i matches video i)
            labels = torch.arange(user_emb.size(0), device=user_emb.device)
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, labels)
        else:
            # Explicit negatives provided
            neg_video_emb = self.encode_video(negative_video_features)  # (batch_size, num_neg, emb_dim)
            
            # Positive scores
            pos_scores = torch.sum(user_emb * pos_video_emb, dim=-1, keepdim=True) / self.temperature
            
            # Negative scores
            if neg_video_emb.dim() == 2:
                neg_video_emb = neg_video_emb.unsqueeze(1)
            neg_scores = torch.bmm(neg_video_emb, user_emb.unsqueeze(-1)).squeeze(-1) / self.temperature
            
            # Concatenate and compute loss
            logits = torch.cat([pos_scores, neg_scores], dim=-1)
            labels = torch.zeros(user_emb.size(0), dtype=torch.long, device=user_emb.device)
            
            loss = F.cross_entropy(logits, labels)
        
        return loss


class TwoTowerDataset(torch.utils.data.Dataset):
    """Dataset for training two-tower model."""
    
    def __init__(
        self,
        user_embeddings: dict,
        video_embeddings: dict,
        interactions: list,
    ):
        """
        Args:
            user_embeddings: Dict mapping user_id -> embedding
            video_embeddings: Dict mapping video_id -> embedding
            interactions: List of (user_id, video_id, label) tuples
        """
        self.user_embeddings = user_embeddings
        self.video_embeddings = video_embeddings
        
        # Filter interactions where we have both embeddings
        self.interactions = [
            (u, v, label) for u, v, label in interactions
            if u in user_embeddings and v in video_embeddings
        ]
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, video_id, label = self.interactions[idx]
        
        user_emb = torch.tensor(self.user_embeddings[user_id], dtype=torch.float32)
        video_emb = torch.tensor(self.video_embeddings[video_id], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return user_emb, video_emb, label


if __name__ == "__main__":
    # Test the model
    print("Testing Two-Tower Model...")
    
    model = TwoTowerModel(
        user_input_dim=256,
        video_input_dim=256,
        embedding_dim=128,
    )
    
    # Dummy data
    batch_size = 32
    user_features = torch.randn(batch_size, 256)
    video_features = torch.randn(batch_size, 256)
    
    # Forward pass
    similarity = model(user_features, video_features)
    print(f"Similarity shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    
    # Contrastive loss
    loss = model.compute_contrastive_loss(user_features, video_features)
    print(f"Contrastive loss: {loss.item():.4f}")
    
    # Test encoding
    user_emb = model.encode_user(user_features)
    video_emb = model.encode_video(video_features)
    print(f"User embedding shape: {user_emb.shape}")
    print(f"Video embedding shape: {video_emb.shape}")
    
    print("\nModel test passed!")