"""
User Interaction Simulator

Generates realistic synthetic user interactions for training
the recommendation system.
"""

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UserPersona:
    """Defines a user persona with specific preferences."""
    name: str
    preferred_categories: list
    category_weights: dict
    avg_session_length: int
    avg_watch_completion: float
    watch_variance: float
    like_probability: float
    activity_level: str  # 'low', 'medium', 'high'


# User personas for simulation
USER_PERSONAS = [
    UserPersona(
        name="tech_enthusiast",
        preferred_categories=[28, 27],  # Science & Tech, Education
        category_weights={28: 0.6, 27: 0.3, 24: 0.1},
        avg_session_length=8,
        avg_watch_completion=0.75,
        watch_variance=0.15,
        like_probability=0.3,
        activity_level="high",
    ),
    UserPersona(
        name="gamer",
        preferred_categories=[20, 24],  # Gaming, Entertainment
        category_weights={20: 0.7, 24: 0.2, 17: 0.1},
        avg_session_length=12,
        avg_watch_completion=0.6,
        watch_variance=0.25,
        like_probability=0.25,
        activity_level="high",
    ),
    UserPersona(
        name="music_lover",
        preferred_categories=[10],  # Music
        category_weights={10: 0.8, 24: 0.15, 22: 0.05},
        avg_session_length=15,
        avg_watch_completion=0.85,
        watch_variance=0.1,
        like_probability=0.4,
        activity_level="medium",
    ),
    UserPersona(
        name="learner",
        preferred_categories=[27, 28, 26],  # Education, Science, Howto
        category_weights={27: 0.5, 28: 0.3, 26: 0.2},
        avg_session_length=5,
        avg_watch_completion=0.9,
        watch_variance=0.1,
        like_probability=0.35,
        activity_level="medium",
    ),
    UserPersona(
        name="casual_viewer",
        preferred_categories=[24, 23, 22],  # Entertainment, Comedy, Blogs
        category_weights={24: 0.4, 23: 0.3, 22: 0.2, 10: 0.1},
        avg_session_length=6,
        avg_watch_completion=0.5,
        watch_variance=0.3,
        like_probability=0.15,
        activity_level="low",
    ),
]


@dataclass
class SimulatedUser:
    """A simulated user."""
    user_id: str
    username: str
    persona_name: str
    preferred_categories: list
    account_age_days: int


@dataclass
class SimulatedInteraction:
    """A simulated user-video interaction."""
    user_id: str
    video_id: str
    interaction_type: str
    watch_duration_seconds: int
    watch_percentage: float
    session_id: str
    device_type: str
    recommendation_source: str
    position_in_list: Optional[int]
    created_at: datetime


class UserSimulator:
    """Generates synthetic users and their interactions."""
    
    def __init__(self, videos: list, seed: int = 42):
        """
        Args:
            videos: List of video dicts with video_id, category_id, duration_seconds
            seed: Random seed for reproducibility
        """
        self.videos = videos
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        
        # Group videos by category
        self.videos_by_category = {}
        for video in videos:
            cat_id = video.get('category_id')
            if cat_id:
                if cat_id not in self.videos_by_category:
                    self.videos_by_category[cat_id] = []
                self.videos_by_category[cat_id].append(video)
        
        logger.info(f"Loaded {len(videos)} videos across {len(self.videos_by_category)} categories")
    
    def generate_users(self, num_users: int) -> list:
        """Generate simulated users."""
        users = []
        
        for i in range(num_users):
            persona = random.choice(USER_PERSONAS)
            
            user = SimulatedUser(
                user_id=str(uuid.uuid4()),
                username=f"user_{i+1:05d}",
                persona_name=persona.name,
                preferred_categories=persona.preferred_categories.copy(),
                account_age_days=random.randint(30, 1000),
            )
            users.append(user)
        
        logger.info(f"Generated {len(users)} users")
        return users
    
    def _get_persona(self, persona_name: str) -> UserPersona:
        """Get persona by name."""
        for p in USER_PERSONAS:
            if p.name == persona_name:
                return p
        return USER_PERSONAS[0]
    
    def _select_video(self, persona: UserPersona, watched_ids: set) -> Optional[dict]:
        """Select a video based on user preferences."""
        # Get category weights
        weights = persona.category_weights.copy()
        
        # 20% exploration
        if random.random() < 0.2:
            available_cats = list(self.videos_by_category.keys())
            if available_cats:
                weights = {random.choice(available_cats): 1.0}
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Filter to available categories
        valid_cats = [c for c in weights.keys() if c in self.videos_by_category]
        if not valid_cats:
            valid_cats = list(self.videos_by_category.keys())
        
        if not valid_cats:
            return None
        
        # Sample category
        probs = [weights.get(c, 0.1) for c in valid_cats]
        total = sum(probs)
        probs = [p/total for p in probs]
        
        chosen_cat = self.rng.choice(valid_cats, p=probs)
        
        # Get videos from category
        available = [v for v in self.videos_by_category[chosen_cat] 
                    if v['video_id'] not in watched_ids]
        
        if not available:
            available = self.videos_by_category[chosen_cat]
        
        return random.choice(available) if available else None
    
    def _simulate_watch(self, persona: UserPersona, video: dict) -> tuple:
        """Simulate watching behavior. Returns (duration, percentage, liked)."""
        video_duration = video.get('duration_seconds', 300) or 300
        
        # Base completion
        base = persona.avg_watch_completion
        variance = persona.watch_variance
        
        # Boost for preferred categories
        if video.get('category_id') in persona.preferred_categories:
            base += 0.1
        
        watch_pct = self.rng.normal(base, variance)
        watch_pct = max(0.05, min(1.0, watch_pct))
        
        watch_duration = int(video_duration * watch_pct)
        
        # Determine like
        liked = False
        if watch_pct > 0.5:
            like_prob = persona.like_probability
            if video.get('category_id') in persona.preferred_categories:
                like_prob *= 1.5
            liked = random.random() < like_prob
        
        return watch_duration, watch_pct, liked
    
    def simulate_session(self, user: SimulatedUser, session_time: datetime) -> list:
        """Simulate a viewing session."""
        interactions = []
        session_id = str(uuid.uuid4())
        persona = self._get_persona(user.persona_name)
        
        # Session length
        length = max(1, int(self.rng.normal(persona.avg_session_length, 2)))
        
        # Device
        devices = ['mobile', 'desktop', 'tablet', 'tv']
        device_weights = [0.5, 0.3, 0.1, 0.1]
        device = self.rng.choice(devices, p=device_weights)
        
        watched_ids = set()
        current_time = session_time
        
        for position in range(length):
            # Source
            if position == 0:
                source = 'home'
            else:
                sources = ['home', 'search', 'related', 'trending']
                source = self.rng.choice(sources, p=[0.4, 0.2, 0.35, 0.05])
            
            video = self._select_video(persona, watched_ids)
            if not video:
                continue
            
            duration, pct, liked = self._simulate_watch(persona, video)
            
            # View interaction
            interactions.append(SimulatedInteraction(
                user_id=user.user_id,
                video_id=video['video_id'],
                interaction_type='view',
                watch_duration_seconds=duration,
                watch_percentage=pct,
                session_id=session_id,
                device_type=device,
                recommendation_source=source,
                position_in_list=position + 1 if source in ['home', 'related'] else None,
                created_at=current_time,
            ))
            
            # Like interaction
            if liked:
                interactions.append(SimulatedInteraction(
                    user_id=user.user_id,
                    video_id=video['video_id'],
                    interaction_type='like',
                    watch_duration_seconds=duration,
                    watch_percentage=pct,
                    session_id=session_id,
                    device_type=device,
                    recommendation_source=source,
                    position_in_list=None,
                    created_at=current_time + timedelta(seconds=duration),
                ))
            
            watched_ids.add(video['video_id'])
            current_time += timedelta(seconds=duration + random.randint(5, 60))
        
        return interactions
    
    def simulate_history(self, user: SimulatedUser, days: int = 30) -> list:
        """Simulate complete viewing history for a user."""
        all_interactions = []
        persona = self._get_persona(user.persona_name)
        
        # Sessions per day based on activity level
        sessions_per_day = {
            'low': (0.3, 1),
            'medium': (0.5, 2),
            'high': (1, 3),
        }
        min_s, max_s = sessions_per_day[persona.activity_level]
        
        for day in range(days):
            date = datetime.now() - timedelta(days=days - day)
            num_sessions = max(0, int(self.rng.uniform(min_s, max_s)))
            
            for _ in range(num_sessions):
                hour = random.randint(8, 23)
                minute = random.randint(0, 59)
                session_time = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                interactions = self.simulate_session(user, session_time)
                all_interactions.extend(interactions)
        
        return all_interactions


if __name__ == "__main__":
    # Test with dummy data
    dummy_videos = [
        {"video_id": f"vid_{i}", "category_id": random.choice([20, 24, 27, 28]), "duration_seconds": random.randint(60, 600)}
        for i in range(100)
    ]
    
    simulator = UserSimulator(dummy_videos, seed=42)
    users = simulator.generate_users(10)
    
    print(f"Generated {len(users)} users")
    
    total_interactions = 0
    for user in users[:3]:
        interactions = simulator.simulate_history(user, days=7)
        total_interactions += len(interactions)
        print(f"  {user.username} ({user.persona_name}): {len(interactions)} interactions")
    
    print(f"\nTotal interactions (3 users, 7 days): {total_interactions}")