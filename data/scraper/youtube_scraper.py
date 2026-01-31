"""
YouTube Video Metadata Scraper

Uses yt-dlp to collect video metadata from YouTube.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Represents metadata for a single YouTube video."""
    video_id: str
    title: str
    description: Optional[str]
    channel_id: Optional[str]
    channel_name: Optional[str]
    category_id: Optional[int]
    category_name: Optional[str]
    tags: list
    duration_seconds: Optional[int]
    view_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]
    thumbnail_url: Optional[str]
    published_at: Optional[str]


# YouTube category mapping
YOUTUBE_CATEGORIES = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    19: "Travel & Events",
    20: "Gaming",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
}


class YouTubeScraper:
    """Scrapes YouTube video metadata using yt-dlp."""
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
    def _run_ytdlp(self, url: str) -> Optional[dict]:
        """Run yt-dlp and return JSON output."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--no-warnings",
            "--ignore-errors",
            url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout fetching {url}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _parse_video_data(self, data: dict) -> Optional[VideoMetadata]:
        """Parse yt-dlp output into VideoMetadata."""
        try:
            video_id = data.get("id")
            if not video_id:
                return None
            
            # Parse upload date
            upload_date = data.get("upload_date")
            published_at = None
            if upload_date:
                try:
                    dt = datetime.strptime(upload_date, "%Y%m%d")
                    published_at = dt.isoformat()
                except ValueError:
                    pass
            
            # Get category
            category_name = data.get("categories", [None])[0] if data.get("categories") else None
            category_id = None
            for cid, cname in YOUTUBE_CATEGORIES.items():
                if cname == category_name:
                    category_id = cid
                    break
            
            # Get best thumbnail
            thumbnails = data.get("thumbnails", [])
            thumbnail_url = None
            if thumbnails:
                for thumb in reversed(thumbnails):
                    if thumb.get("url"):
                        thumbnail_url = thumb["url"]
                        break
            
            return VideoMetadata(
                video_id=video_id,
                title=data.get("title", ""),
                description=data.get("description"),
                channel_id=data.get("channel_id"),
                channel_name=data.get("channel") or data.get("uploader"),
                category_id=category_id,
                category_name=category_name,
                tags=data.get("tags", []) or [],
                duration_seconds=data.get("duration"),
                view_count=data.get("view_count"),
                like_count=data.get("like_count"),
                comment_count=data.get("comment_count"),
                thumbnail_url=thumbnail_url,
                published_at=published_at,
            )
            
        except Exception as e:
            logger.error(f"Error parsing video data: {e}")
            return None
    
    def scrape_video(self, video_id: str) -> Optional[VideoMetadata]:
        """Scrape metadata for a single video."""
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        for attempt in range(self.max_retries):
            data = self._run_ytdlp(url)
            if data:
                video = self._parse_video_data(data)
                if video:
                    return video
            
            if attempt < self.max_retries - 1:
                time.sleep(self.rate_limit_delay * (attempt + 1))
        
        return None
    
    def scrape_search(
        self,
        query: str,
        max_videos: int = 50
    ) -> Generator[VideoMetadata, None, None]:
        """Scrape videos from YouTube search results."""
        search_url = f"ytsearch{max_videos}:{query}"
        
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            search_url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            video_ids = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("id"):
                            video_ids.append(entry["id"])
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Found {len(video_ids)} videos for query '{query}'")
            
            for i, video_id in enumerate(video_ids):
                logger.info(f"Scraping video {i+1}/{len(video_ids)}: {video_id}")
                video = self.scrape_video(video_id)
                if video:
                    yield video
                time.sleep(self.rate_limit_delay)
                
        except Exception as e:
            logger.error(f"Error scraping search: {e}")
    
    def scrape_channel(
        self,
        channel_url: str,
        max_videos: int = 50
    ) -> Generator[VideoMetadata, None, None]:
        """Scrape recent videos from a channel."""
        if "/videos" not in channel_url:
            channel_url = channel_url.rstrip("/") + "/videos"
        
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            f"--playlist-end={max_videos}",
            channel_url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            video_ids = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("id"):
                            video_ids.append(entry["id"])
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Found {len(video_ids)} videos from channel")
            
            for i, video_id in enumerate(video_ids):
                logger.info(f"Scraping video {i+1}/{len(video_ids)}: {video_id}")
                video = self.scrape_video(video_id)
                if video:
                    yield video
                time.sleep(self.rate_limit_delay)
                
        except Exception as e:
            logger.error(f"Error scraping channel: {e}")
    
    def save_to_json(self, videos: list, filename: str):
        """Save videos to JSON file."""
        filepath = self.output_dir / filename
        
        data = [asdict(v) for v in videos]
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(videos)} videos to {filepath}")
        return filepath


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Video Scraper")
    parser.add_argument("--mode", choices=["search", "channel", "video"], default="search")
    parser.add_argument("--query", type=str, required=True, help="Search query, channel URL, or video ID")
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--output", type=str, default="videos.json")
    parser.add_argument("--delay", type=float, default=1.0)
    
    args = parser.parse_args()
    
    scraper = YouTubeScraper(rate_limit_delay=args.delay)
    
    videos = []
    
    if args.mode == "search":
        for video in scraper.scrape_search(args.query, args.max_videos):
            videos.append(video)
            print(f"  ✓ {video.title[:50]}...")
    
    elif args.mode == "channel":
        for video in scraper.scrape_channel(args.query, args.max_videos):
            videos.append(video)
            print(f"  ✓ {video.title[:50]}...")
    
    elif args.mode == "video":
        video = scraper.scrape_video(args.query)
        if video:
            videos.append(video)
            print(f"  ✓ {video.title}")
    
    if videos:
        scraper.save_to_json(videos, args.output)
        print(f"\nTotal: {len(videos)} videos saved to data/raw/{args.output}")