"""
Generate Embeddings

Generates video and user embeddings and stores them in PostgreSQL.
"""

import os
import sys
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.video_encoder import VideoEncoder, BatchVideoEncoder
from features.user_encoder import UserEncoder

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")


def get_connection():
    """Get database connection."""
    return psycopg2.connect(DATABASE_URL)


def get_videos_without_embeddings(limit: int = None) -> list:
    """Get videos that don't have embeddings yet."""
    query = """
        SELECT v.video_id, v.title, v.description, v.thumbnail_url
        FROM videos v
        LEFT JOIN video_embeddings ve ON v.video_id = ve.video_id
        WHERE ve.video_id IS NULL AND v.is_active = true
    """
    if limit:
        query += f" LIMIT {limit}"
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = ['video_id', 'title', 'description', 'thumbnail_url']
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_all_videos_with_embeddings() -> dict:
    """Get all video embeddings as a dict."""
    query = """
        SELECT video_id, combined_embedding
        FROM video_embeddings
        WHERE combined_embedding IS NOT NULL
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return {row[0]: np.array(row[1]) for row in cur.fetchall()}


def get_user_watch_history(user_id: str) -> list:
    """Get user's watch history with video embeddings."""
    query = """
        SELECT ui.video_id, ui.watch_percentage, ui.interaction_type, ve.combined_embedding
        FROM user_interactions ui
        JOIN video_embeddings ve ON ui.video_id = ve.video_id
        WHERE ui.user_id = %s
        ORDER BY ui.created_at ASC
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id,))
            return cur.fetchall()


def get_users_without_embeddings(limit: int = None) -> list:
    """Get users that don't have embeddings yet."""
    query = """
        SELECT u.user_id
        FROM users u
        LEFT JOIN user_embeddings ue ON u.user_id = ue.user_id
        WHERE ue.user_id IS NULL
    """
    if limit:
        query += f" LIMIT {limit}"
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]


def save_video_embeddings(embeddings: list) -> int:
    """Save video embeddings to database."""
    if not embeddings:
        return 0
    
    query = """
        INSERT INTO video_embeddings (
            video_id, thumbnail_embedding, title_embedding, 
            description_embedding, combined_embedding
        ) VALUES %s
        ON CONFLICT (video_id) DO UPDATE SET
            thumbnail_embedding = EXCLUDED.thumbnail_embedding,
            title_embedding = EXCLUDED.title_embedding,
            description_embedding = EXCLUDED.description_embedding,
            combined_embedding = EXCLUDED.combined_embedding,
            updated_at = NOW()
    """
    
    values = []
    for emb in embeddings:
        values.append((
            emb['video_id'],
            emb['thumbnail_embedding'].tolist() if emb['thumbnail_embedding'] is not None else None,
            emb['title_embedding'].tolist() if emb['title_embedding'] is not None else None,
            emb['description_embedding'].tolist() if emb['description_embedding'] is not None else None,
            emb['combined_embedding'].tolist() if emb['combined_embedding'] is not None else None,
        ))
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, values)
            conn.commit()
    
    return len(values)


def save_user_embedding(user_id: str, embedding: np.ndarray) -> bool:
    """Save user embedding to database."""
    query = """
        INSERT INTO user_embeddings (user_id, user_embedding)
        VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            user_embedding = EXCLUDED.user_embedding,
            updated_at = NOW()
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, embedding.tolist()))
            conn.commit()
    
    return True


def generate_video_embeddings(limit: int = None, skip_thumbnails: bool = False):
    """Generate embeddings for videos without them."""
    print("Fetching videos without embeddings...")
    videos = get_videos_without_embeddings(limit)
    
    if not videos:
        print("No videos need embeddings.")
        return
    
    print(f"Found {len(videos)} videos to encode")
    
    # Initialize encoder
    print("Initializing video encoder...")
    encoder = VideoEncoder()
    
    # Encode in batches
    batch_size = 10
    all_embeddings = []
    
    for i in tqdm(range(0, len(videos), batch_size), desc="Encoding batches"):
        batch = videos[i:i + batch_size]
        
        for video in batch:
            try:
                thumbnail_url = None if skip_thumbnails else video.get('thumbnail_url')
                
                emb = encoder.encode_video(
                    title=video.get('title', ''),
                    description=video.get('description'),
                    thumbnail_url=thumbnail_url,
                )
                
                all_embeddings.append({
                    'video_id': video['video_id'],
                    **emb,
                })
            except Exception as e:
                print(f"Error encoding {video['video_id']}: {e}")
                continue
        
        # Save batch
        if len(all_embeddings) >= batch_size:
            saved = save_video_embeddings(all_embeddings)
            print(f"Saved {saved} embeddings")
            all_embeddings = []
    
    # Save remaining
    if all_embeddings:
        saved = save_video_embeddings(all_embeddings)
        print(f"Saved {saved} embeddings")
    
    print("Video embedding generation complete!")


def generate_user_embeddings(limit: int = None):
    """Generate embeddings for users without them."""
    print("Fetching users without embeddings...")
    user_ids = get_users_without_embeddings(limit)
    
    if not user_ids:
        print("No users need embeddings.")
        return
    
    print(f"Found {len(user_ids)} users to encode")
    
    # Initialize encoder
    encoder = UserEncoder(embedding_dim=256)
    
    success_count = 0
    
    for user_id in tqdm(user_ids, desc="Encoding users"):
        try:
            # Get watch history
            history = get_user_watch_history(user_id)
            
            if not history:
                # No history, create zero embedding
                embedding = np.zeros(256)
            else:
                embeddings = [np.array(row[3]) for row in history if row[3] is not None]
                watch_pcts = [row[1] or 0.5 for row in history]
                interaction_types = [row[2] for row in history]
                
                if embeddings:
                    embedding = encoder.encode_user(
                        watched_video_embeddings=embeddings,
                        watch_percentages=watch_pcts,
                        interaction_types=interaction_types,
                    )
                else:
                    embedding = np.zeros(256)
            
            save_user_embedding(user_id, embedding)
            success_count += 1
            
        except Exception as e:
            print(f"Error encoding user {user_id}: {e}")
            continue
    
    print(f"User embedding generation complete! Encoded {success_count} users.")


def get_stats():
    """Get embedding statistics."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM videos WHERE is_active = true")
            total_videos = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM video_embeddings")
            videos_with_emb = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM users")
            total_users = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM user_embeddings")
            users_with_emb = cur.fetchone()[0]
    
    return {
        "total_videos": total_videos,
        "videos_with_embeddings": videos_with_emb,
        "total_users": total_users,
        "users_with_embeddings": users_with_emb,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("--videos", action="store_true", help="Generate video embeddings")
    parser.add_argument("--users", action="store_true", help="Generate user embeddings")
    parser.add_argument("--all", action="store_true", help="Generate all embeddings")
    parser.add_argument("--limit", type=int, help="Limit number to process")
    parser.add_argument("--skip-thumbnails", action="store_true", help="Skip thumbnail encoding (faster)")
    parser.add_argument("--stats", action="store_true", help="Show embedding stats")
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_stats()
        print("\nEmbedding Statistics:")
        print(f"  Videos: {stats['videos_with_embeddings']}/{stats['total_videos']}")
        print(f"  Users: {stats['users_with_embeddings']}/{stats['total_users']}")
        return
    
    if args.videos or args.all:
        generate_video_embeddings(limit=args.limit, skip_thumbnails=args.skip_thumbnails)
    
    if args.users or args.all:
        generate_user_embeddings(limit=args.limit)
    
    if not (args.videos or args.users or args.all):
        print("Specify --videos, --users, or --all")
        parser.print_help()


if __name__ == "__main__":
    main()