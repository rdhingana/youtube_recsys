"""
Batch Scraper for YouTube Videos

Scrapes videos across multiple categories to build a diverse corpus.
"""

import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path

from youtube_scraper import YouTubeScraper, VideoMetadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Search queries by category for diverse content
SCRAPE_CONFIG = {
    "technology": [
        "tech review 2024",
        "programming tutorial python",
        "AI machine learning explained",
        "software engineering tips",
    ],
    "gaming": [
        "gaming walkthrough",
        "game review 2024",
        "esports highlights",
    ],
    "education": [
        "science explained",
        "math tutorial",
        "history documentary",
    ],
    "entertainment": [
        "comedy sketch",
        "movie trailer 2024",
        "music video",
    ],
    "lifestyle": [
        "cooking recipe easy",
        "fitness workout home",
        "travel vlog",
    ],
}

# Popular channels to scrape
POPULAR_CHANNELS = [
    "https://www.youtube.com/@mkbhd",
    "https://www.youtube.com/@3blue1brown",
    "https://www.youtube.com/@Fireship",
    "https://www.youtube.com/@veritasium",
]


class BatchScraper:
    """Orchestrates batch scraping across multiple categories."""
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        videos_per_query: int = 10,
        videos_per_channel: int = 15,
        rate_limit_delay: float = 1.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.videos_per_query = videos_per_query
        self.videos_per_channel = videos_per_channel
        
        self.scraper = YouTubeScraper(
            output_dir=output_dir,
            rate_limit_delay=rate_limit_delay,
        )
        
        self.stats = {
            "total_videos": 0,
            "by_category": {},
            "errors": [],
        }
    
    def scrape_by_categories(self) -> list:
        """Scrape videos from search queries by category."""
        all_videos = []
        seen_ids = set()
        
        for category, queries in SCRAPE_CONFIG.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Category: {category}")
            logger.info(f"{'='*50}")
            
            category_videos = []
            
            for query in queries:
                logger.info(f"\nSearching: '{query}'")
                
                try:
                    for video in self.scraper.scrape_search(query, self.videos_per_query):
                        if video.video_id not in seen_ids:
                            seen_ids.add(video.video_id)
                            category_videos.append(video)
                            all_videos.append(video)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    self.stats["errors"].append(str(e))
                
                time.sleep(2)
            
            self.stats["by_category"][category] = len(category_videos)
            logger.info(f"\nCategory '{category}': {len(category_videos)} videos")
        
        return all_videos
    
    def scrape_channels(self) -> list:
        """Scrape videos from popular channels."""
        all_videos = []
        seen_ids = set()
        
        for channel_url in POPULAR_CHANNELS:
            channel_name = channel_url.split("@")[-1]
            logger.info(f"\n{'='*50}")
            logger.info(f"Channel: {channel_name}")
            logger.info(f"{'='*50}")
            
            try:
                for video in self.scraper.scrape_channel(channel_url, self.videos_per_channel):
                    if video.video_id not in seen_ids:
                        seen_ids.add(video.video_id)
                        all_videos.append(video)
            except Exception as e:
                logger.error(f"Error: {e}")
                self.stats["errors"].append(str(e))
            
            time.sleep(3)
        
        return all_videos
    
    def run(self, mode: str = "all") -> list:
        """Run the batch scraper."""
        logger.info("Starting batch scraper...")
        
        all_videos = []
        
        if mode in ["all", "categories"]:
            videos = self.scrape_by_categories()
            all_videos.extend(videos)
        
        if mode in ["all", "channels"]:
            videos = self.scrape_channels()
            # Dedupe
            seen_ids = {v.video_id for v in all_videos}
            for v in videos:
                if v.video_id not in seen_ids:
                    all_videos.append(v)
        
        self.stats["total_videos"] = len(all_videos)
        
        # Save all videos
        if all_videos:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"videos_{timestamp}.json"
            self.scraper.save_to_json(all_videos, filename)
        
        self._print_summary()
        
        return all_videos
    
    def _print_summary(self):
        """Print scraping summary."""
        print("\n" + "="*50)
        print("SCRAPING SUMMARY")
        print("="*50)
        print(f"Total videos: {self.stats['total_videos']}")
        print(f"Errors: {len(self.stats['errors'])}")
        
        if self.stats["by_category"]:
            print("\nBy Category:")
            for cat, count in self.stats["by_category"].items():
                print(f"  {cat}: {count}")
        print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch YouTube Scraper")
    parser.add_argument("--mode", choices=["all", "categories", "channels"], default="all")
    parser.add_argument("--videos-per-query", type=int, default=10)
    parser.add_argument("--videos-per-channel", type=int, default=15)
    parser.add_argument("--delay", type=float, default=1.5)
    
    args = parser.parse_args()
    
    scraper = BatchScraper(
        videos_per_query=args.videos_per_query,
        videos_per_channel=args.videos_per_channel,
        rate_limit_delay=args.delay,
    )
    
    scraper.run(mode=args.mode)