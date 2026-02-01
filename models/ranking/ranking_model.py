"""
Ranking Model

Deep Cross Network for ranking retrieved candidates.
Predicts engagement probability (click, watch time, like).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossNetwork(nn.Module):
    """
    Cross Network for explicit feature interactions.
    
    Each layer computes: x_{l+1} = x_0 * x_l^T * w_l + b_l + x_l
    """
    
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01)
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for i in range(self.num_layers):
            # x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l
            xw = torch.matmul(x, self.weights[i])  # (batch, 1)
            x = x0 * xw + self.biases[i] + x
        return x


class DeepNetwork(nn.Module):
    """Deep network for implicit feature interactions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
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
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RankingModel(nn.Module):
    """
    Deep Cross Network (DCN) for ranking.
    
    Combines cross network (explicit interactions) with 
    deep network (implicit interactions).
    
    Input features:
    - User embedding (256-dim)
    - Video embedding (256-dim)
    - Context features (optional)
    """
    
    def __init__(
        self,
        user_dim: int = 256,
        video_dim: int = 256,
        context_dim: int = 0,
        cross_layers: int = 3,
        deep_hidden_dims: list = [512, 256, 128],
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.user_dim = user_dim
        self.video_dim = video_dim
        self.context_dim = context_dim
        
        # Total input dimension
        input_dim = user_dim + video_dim + context_dim
        
        # Cross network
        self.cross_network = CrossNetwork(input_dim, cross_layers)
        
        # Deep network
        self.deep_network = DeepNetwork(input_dim, deep_hidden_dims, dropout)
        
        # Combination layer
        combined_dim = input_dim + self.deep_network.output_dim
        self.output_layer = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    
    def forward(
        self,
        user_embedding: torch.Tensor,
        video_embedding: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_embedding: (batch_size, user_dim)
            video_embedding: (batch_size, video_dim)
            context_features: (batch_size, context_dim) optional
            
        Returns:
            Ranking scores: (batch_size,)
        """
        # Concatenate all features
        if context_features is not None:
            x = torch.cat([user_embedding, video_embedding, context_features], dim=-1)
        else:
            x = torch.cat([user_embedding, video_embedding], dim=-1)
        
        # Cross network
        cross_out = self.cross_network(x)
        
        # Deep network
        deep_out = self.deep_network(x)
        
        # Combine
        combined = torch.cat([cross_out, deep_out], dim=-1)
        
        # Output score
        score = self.output_layer(combined).squeeze(-1)
        
        return score
    
    def predict_proba(
        self,
        user_embedding: torch.Tensor,
        video_embedding: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict probability (sigmoid of score)."""
        score = self.forward(user_embedding, video_embedding, context_features)
        return torch.sigmoid(score)


class MultiTaskRankingModel(nn.Module):
    """
    Multi-task ranking model.
    
    Predicts multiple objectives:
    - Click probability
    - Watch time
    - Like probability
    """
    
    def __init__(
        self,
        user_dim: int = 256,
        video_dim: int = 256,
        context_dim: int = 0,
        shared_hidden_dims: list = [512, 256],
        task_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        input_dim = user_dim + video_dim + context_dim
        
        # Shared layers
        self.shared_network = DeepNetwork(input_dim, shared_hidden_dims, dropout)
        shared_out_dim = self.shared_network.output_dim
        
        # Task-specific towers
        self.click_tower = nn.Sequential(
            nn.Linear(shared_out_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Linear(task_hidden_dim, 1),
        )
        
        self.watch_tower = nn.Sequential(
            nn.Linear(shared_out_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Linear(task_hidden_dim, 1),
        )
        
        self.like_tower = nn.Sequential(
            nn.Linear(shared_out_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Linear(task_hidden_dim, 1),
        )
    
    def forward(
        self,
        user_embedding: torch.Tensor,
        video_embedding: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (click_score, watch_score, like_score)
        """
        if context_features is not None:
            x = torch.cat([user_embedding, video_embedding, context_features], dim=-1)
        else:
            x = torch.cat([user_embedding, video_embedding], dim=-1)
        
        # Shared representation
        shared = self.shared_network(x)
        
        # Task-specific predictions
        click_score = self.click_tower(shared).squeeze(-1)
        watch_score = self.watch_tower(shared).squeeze(-1)
        like_score = self.like_tower(shared).squeeze(-1)
        
        return click_score, watch_score, like_score
    
    def compute_combined_score(
        self,
        user_embedding: torch.Tensor,
        video_embedding: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
        click_weight: float = 0.3,
        watch_weight: float = 0.5,
        like_weight: float = 0.2,
    ) -> torch.Tensor:
        """Compute weighted combination of task scores."""
        click, watch, like = self.forward(user_embedding, video_embedding, context_features)
        
        # Normalize scores to [0, 1]
        click_prob = torch.sigmoid(click)
        watch_prob = torch.sigmoid(watch)
        like_prob = torch.sigmoid(like)
        
        # Weighted combination
        combined = (
            click_weight * click_prob +
            watch_weight * watch_prob +
            like_weight * like_prob
        )
        
        return combined


class RankingDataset(torch.utils.data.Dataset):
    """Dataset for ranking model training."""
    
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
                         label can be click (0/1), watch_pct (0-1), or like (0/1)
        """
        self.data = []
        
        for user_id, video_id, label in interactions:
            if user_id in user_embeddings and video_id in video_embeddings:
                self.data.append({
                    'user_emb': user_embeddings[user_id],
                    'video_emb': video_embeddings[video_id],
                    'label': label,
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['user_emb'], dtype=torch.float32),
            torch.tensor(item['video_emb'], dtype=torch.float32),
            torch.tensor(item['label'], dtype=torch.float32),
        )


if __name__ == "__main__":
    print("Testing Ranking Models...")
    
    batch_size = 32
    user_emb = torch.randn(batch_size, 256)
    video_emb = torch.randn(batch_size, 256)
    
    # Test RankingModel (DCN)
    print("\n1. Testing RankingModel (DCN)...")
    model = RankingModel(user_dim=256, video_dim=256)
    
    scores = model(user_emb, video_emb)
    print(f"   Score shape: {scores.shape}")
    print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    probs = model.predict_proba(user_emb, video_emb)
    print(f"   Prob range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Test MultiTaskRankingModel
    print("\n2. Testing MultiTaskRankingModel...")
    multi_model = MultiTaskRankingModel(user_dim=256, video_dim=256)
    
    click, watch, like = multi_model(user_emb, video_emb)
    print(f"   Click score shape: {click.shape}")
    print(f"   Watch score shape: {watch.shape}")
    print(f"   Like score shape: {like.shape}")
    
    combined = multi_model.compute_combined_score(user_emb, video_emb)
    print(f"   Combined score range: [{combined.min():.3f}, {combined.max():.3f}]")
    
    # Test loss computation
    print("\n3. Testing loss computation...")
    labels = torch.rand(batch_size)
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    print(f"   BCE Loss: {loss.item():.4f}")
    
    print("\nRanking model tests passed!")