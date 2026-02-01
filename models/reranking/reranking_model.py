"""
Re-ranking Model

Applies diversity, freshness, and business rules to ranked candidates.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoCandidate:
    """A video candidate with metadata for re-ranking."""
    video_id: str
    score: float
    embedding: Optional[np.ndarray] = None
    category_id: Optional[int] = None
    channel_id: Optional[str] = None
    published_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    view_count: Optional[int] = None


class DiversityReranker:
    """
    Maximal Marginal Relevance (MMR) based diversity re-ranking.
    
    Balances relevance (ranking score) with diversity (dissimilarity to already selected items).
    """
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Args:
            lambda_param: Trade-off between relevance and diversity.
                         1.0 = pure relevance, 0.0 = pure diversity
        """
        self.lambda_param = lambda_param
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _max_similarity_to_selected(
        self,
        candidate: VideoCandidate,
        selected: List[VideoCandidate],
    ) -> float:
        """Find maximum similarity between candidate and any selected item."""
        if not selected or candidate.embedding is None:
            return 0.0
        
        max_sim = 0.0
        for s in selected:
            if s.embedding is not None:
                sim = self._cosine_similarity(candidate.embedding, s.embedding)
                max_sim = max(max_sim, sim)
        
        return max_sim
    
    def rerank(
        self,
        candidates: List[VideoCandidate],
        k: int,
    ) -> List[VideoCandidate]:
        """
        Re-rank candidates using MMR.
        
        MMR = λ * relevance - (1 - λ) * max_similarity_to_selected
        
        Args:
            candidates: List of candidates with scores and embeddings
            k: Number of items to return
            
        Returns:
            Re-ranked list of k candidates
        """
        if len(candidates) <= k:
            return candidates
        
        # Normalize scores to [0, 1]
        scores = np.array([c.score for c in candidates])
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            norm_scores = (scores - min_score) / (max_score - min_score)
        else:
            norm_scores = np.ones_like(scores)
        
        # Update candidates with normalized scores
        for i, c in enumerate(candidates):
            c._norm_score = norm_scores[i]
        
        selected = []
        remaining = list(candidates)
        
        for _ in range(k):
            if not remaining:
                break
            
            best_idx = -1
            best_mmr = float('-inf')
            
            for i, candidate in enumerate(remaining):
                relevance = candidate._norm_score
                diversity = 1.0 - self._max_similarity_to_selected(candidate, selected)
                
                mmr = self.lambda_param * relevance + (1 - self.lambda_param) * diversity
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        
        return selected


class CategoryDiversityReranker:
    """Ensures diversity across categories."""
    
    def __init__(self, max_per_category: int = 3):
        self.max_per_category = max_per_category
    
    def rerank(
        self,
        candidates: List[VideoCandidate],
        k: int,
    ) -> List[VideoCandidate]:
        """Re-rank ensuring category diversity."""
        category_counts = {}
        selected = []
        
        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        for candidate in sorted_candidates:
            if len(selected) >= k:
                break
            
            cat_id = candidate.category_id or 'unknown'
            count = category_counts.get(cat_id, 0)
            
            if count < self.max_per_category:
                selected.append(candidate)
                category_counts[cat_id] = count + 1
        
        return selected


class FreshnessBooster:
    """Boosts scores of fresh/recent content."""
    
    def __init__(
        self,
        max_boost: float = 0.2,
        decay_days: int = 7,
    ):
        """
        Args:
            max_boost: Maximum score boost for very fresh content
            decay_days: Days after which boost becomes 0
        """
        self.max_boost = max_boost
        self.decay_days = decay_days
    
    def apply(self, candidates: List[VideoCandidate]) -> List[VideoCandidate]:
        """Apply freshness boost to candidate scores."""
        now = datetime.now()
        
        for candidate in candidates:
            if candidate.published_at:
                age_days = (now - candidate.published_at).days
                
                if age_days <= self.decay_days:
                    # Linear decay
                    boost = self.max_boost * (1 - age_days / self.decay_days)
                    candidate.score += boost
        
        return candidates


class BusinessRulesFilter:
    """Applies business rules and filters."""
    
    def __init__(
        self,
        min_duration: int = 30,
        max_duration: int = 7200,
        blocked_channels: List[str] = None,
        blocked_categories: List[int] = None,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.blocked_channels = set(blocked_channels or [])
        self.blocked_categories = set(blocked_categories or [])
    
    def filter(self, candidates: List[VideoCandidate]) -> List[VideoCandidate]:
        """Filter candidates based on business rules."""
        filtered = []
        
        for candidate in candidates:
            # Duration filter
            if candidate.duration_seconds:
                if candidate.duration_seconds < self.min_duration:
                    continue
                if candidate.duration_seconds > self.max_duration:
                    continue
            
            # Blocked channel filter
            if candidate.channel_id and candidate.channel_id in self.blocked_channels:
                continue
            
            # Blocked category filter
            if candidate.category_id and candidate.category_id in self.blocked_categories:
                continue
            
            filtered.append(candidate)
        
        return filtered


class ReRankingPipeline:
    """
    Complete re-ranking pipeline.
    
    Combines multiple re-ranking strategies.
    """
    
    def __init__(
        self,
        diversity_lambda: float = 0.7,
        max_per_category: int = 3,
        freshness_boost: float = 0.2,
        freshness_decay_days: int = 7,
        min_duration: int = 30,
        max_duration: int = 7200,
    ):
        self.diversity_reranker = DiversityReranker(lambda_param=diversity_lambda)
        self.category_reranker = CategoryDiversityReranker(max_per_category=max_per_category)
        self.freshness_booster = FreshnessBooster(
            max_boost=freshness_boost,
            decay_days=freshness_decay_days,
        )
        self.business_filter = BusinessRulesFilter(
            min_duration=min_duration,
            max_duration=max_duration,
        )
    
    def rerank(
        self,
        candidates: List[VideoCandidate],
        k: int,
        use_diversity: bool = True,
        use_category_diversity: bool = True,
        use_freshness: bool = True,
        use_business_rules: bool = True,
    ) -> List[VideoCandidate]:
        """
        Apply full re-ranking pipeline.
        
        Args:
            candidates: Ranked candidates from ranking model
            k: Number of final recommendations
            
        Returns:
            Final re-ranked recommendations
        """
        result = candidates
        
        # 1. Apply business rules filter
        if use_business_rules:
            result = self.business_filter.filter(result)
        
        # 2. Apply freshness boost
        if use_freshness:
            result = self.freshness_booster.apply(result)
        
        # 3. Apply diversity re-ranking (MMR)
        if use_diversity:
            result = self.diversity_reranker.rerank(result, k=min(k * 2, len(result)))
        
        # 4. Apply category diversity
        if use_category_diversity:
            result = self.category_reranker.rerank(result, k=k)
        
        # 5. Final sort by score and trim
        result = sorted(result, key=lambda x: x.score, reverse=True)[:k]
        
        return result


if __name__ == "__main__":
    print("Testing Re-ranking Models...")
    
    # Create dummy candidates
    np.random.seed(42)
    candidates = []
    
    for i in range(50):
        candidates.append(VideoCandidate(
            video_id=f"video_{i}",
            score=np.random.rand(),
            embedding=np.random.randn(128),
            category_id=np.random.randint(1, 6),
            channel_id=f"channel_{np.random.randint(1, 10)}",
            published_at=datetime.now() - timedelta(days=np.random.randint(0, 30)),
            duration_seconds=np.random.randint(60, 1800),
        ))
    
    print(f"\nInput: {len(candidates)} candidates")
    
    # Test DiversityReranker
    print("\n1. Testing DiversityReranker (MMR)...")
    div_reranker = DiversityReranker(lambda_param=0.7)
    div_result = div_reranker.rerank(candidates.copy(), k=10)
    print(f"   Output: {len(div_result)} items")
    print(f"   Top 5 video IDs: {[c.video_id for c in div_result[:5]]}")
    
    # Test CategoryDiversityReranker
    print("\n2. Testing CategoryDiversityReranker...")
    cat_reranker = CategoryDiversityReranker(max_per_category=2)
    cat_result = cat_reranker.rerank(candidates.copy(), k=10)
    print(f"   Output: {len(cat_result)} items")
    categories = [c.category_id for c in cat_result]
    print(f"   Categories: {categories}")
    
    # Test FreshnessBooster
    print("\n3. Testing FreshnessBooster...")
    fresh_booster = FreshnessBooster(max_boost=0.2, decay_days=7)
    fresh_result = fresh_booster.apply(candidates.copy())
    boosted_scores = [(c.video_id, c.score) for c in sorted(fresh_result, key=lambda x: x.score, reverse=True)[:5]]
    print(f"   Top 5 after boost: {boosted_scores}")
    
    # Test full pipeline
    print("\n4. Testing ReRankingPipeline...")
    pipeline = ReRankingPipeline()
    final_result = pipeline.rerank(candidates.copy(), k=10)
    print(f"   Final output: {len(final_result)} items")
    print(f"   Video IDs: {[c.video_id for c in final_result]}")
    print(f"   Categories: {[c.category_id for c in final_result]}")
    
    print("\nRe-ranking tests passed!")