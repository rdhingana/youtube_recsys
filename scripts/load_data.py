"""
Load Data into PostgreSQL

Loads scraped videos and simulated user interactions into the database.
"""

import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.simulator.user_simulator import UserSimulator, SimulatedUser, SimulatedInteraction

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")


def get_connection():
    """Get database connection."""
    return psycopg2.connect(DATABASE_URL)


def load_videos_from_json(filepath: str) -> list:
    """Load videos from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def insert_videos(videos: list) -> int:
    """Insert videos into database."""
    if not videos:
        return 0
    
    query = """
        INSERT INTO videos (
            video_id, title, description, channel_id, channel_name,
            category_id, category_name, tags, duration_seconds,
            view_count, like_count, comment_count, thumbnail_url,
            published_at, scraped_at
        ) VALUES %s
        ON CONFLICT (video_id) DO UPDATE SET
            title = EXCLUDED.title,
            view_count = EXCLUDED.view_count,
            like_count = EXCLUDED.like_count,
            updated_at = NOW()
    """
    
    values = []
    for v in videos:
        published_at = None
        if v.get('published_at'):
            try:
                published_at = datetime.fromisoformat(v['published_at'])
            except:
                pass
        
        values.append((
            v['video_id'],
            v['title'],
            v.get('description'),
            v.get('channel_id'),
            v.get('channel_name'),
            v.get('category_id'),
            v.get('category_name'),
            v.get('tags', []),
            v.get('duration_seconds'),
            v.get('view_count'),
            v.get('like_count'),
            v.get('comment_count'),
            v.get('thumbnail_url'),
            published_at,
            datetime.now(),
        ))
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, values)
            conn.commit()
    
    return len(values)


def insert_users(users: list) -> int:
    """Insert users into database."""
    if not users:
        return 0
    
    query = """
        INSERT INTO users (user_id, username, preferred_categories, account_age_days)
        VALUES %s
        ON CONFLICT (user_id) DO NOTHING
    """
    
    values = [(u.user_id, u.username, u.preferred_categories, u.account_age_days) 
              for u in users]
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, values)
            conn.commit()
    
    return len(values)


def insert_interactions(interactions: list) -> int:
    """Insert interactions into database."""
    if not interactions:
        return 0
    
    query = """
        INSERT INTO user_interactions (
            user_id, video_id, interaction_type, watch_duration_seconds,
            watch_percentage, session_id, device_type, recommendation_source,
            position_in_list, created_at
        ) VALUES %s
    """
    
    values = [(
        i.user_id, i.video_id, i.interaction_type, i.watch_duration_seconds,
        i.watch_percentage, i.session_id, i.device_type, i.recommendation_source,
        i.position_in_list, i.created_at
    ) for i in interactions]
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, values)
            conn.commit()
    
    return len(values)


def get_videos_from_db() -> list:
    """Get all videos from database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT video_id, category_id, duration_seconds
                FROM videos WHERE is_active = true
            """)
            return [
                {"video_id": r[0], "category_id": r[1], "duration_seconds": r[2]}
                for r in cur.fetchall()
            ]


def get_stats():
    """Get database statistics."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM videos")
            videos = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM users")
            users = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM user_interactions")
            interactions = cur.fetchone()[0]
    
    return {"videos": videos, "users": users, "interactions": interactions}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load data into PostgreSQL")
    parser.add_argument("--videos", type=str, help="Path to videos JSON file")
    parser.add_argument("--simulate-users", type=int, default=0, help="Number of users to simulate")
    parser.add_argument("--simulate-days", type=int, default=30, help="Days of history to simulate")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_stats()
        print("\nDatabase Statistics:")
        print(f"  Videos: {stats['videos']}")
        print(f"  Users: {stats['users']}")
        print(f"  Interactions: {stats['interactions']}")
        return
    
    # Load videos
    if args.videos:
        print(f"\nLoading videos from {args.videos}...")
        videos = load_videos_from_json(args.videos)
        count = insert_videos(videos)
        print(f"  Inserted {count} videos")
    
    # Simulate users
    if args.simulate_users > 0:
        print(f"\nSimulating {args.simulate_users} users...")
        
        # Get videos from database
        videos = get_videos_from_db()
        if not videos:
            print("  Error: No videos in database. Load videos first.")
            return
        
        print(f"  Found {len(videos)} videos in database")
        
        # Generate users
        simulator = UserSimulator(videos, seed=42)
        users = simulator.generate_users(args.simulate_users)
        
        # Insert users
        count = insert_users(users)
        print(f"  Inserted {count} users")
        
        # Generate interactions
        print(f"\nSimulating {args.simulate_days} days of interactions...")
        all_interactions = []
        
        for i, user in enumerate(users):
            interactions = simulator.simulate_history(user, days=args.simulate_days)
            all_interactions.extend(interactions)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(users)} users...")
        
        # Insert in batches
        batch_size = 5000
        total_inserted = 0
        
        for i in range(0, len(all_interactions), batch_size):
            batch = all_interactions[i:i + batch_size]
            count = insert_interactions(batch)
            total_inserted += count
        
        print(f"  Inserted {total_inserted} interactions")
    
    # Final stats
    stats = get_stats()
    print("\nFinal Database Statistics:")
    print(f"  Videos: {stats['videos']}")
    print(f"  Users: {stats['users']}")
    print(f"  Interactions: {stats['interactions']}")


if __name__ == "__main__":
    main()