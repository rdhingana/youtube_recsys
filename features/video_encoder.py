"""
Video Encoder

Generates embeddings for videos using:
- CLIP (ViT-B/32) for thumbnail images
- Sentence Transformers (all-MiniLM-L6-v2) for title/description text
"""

import logging
from typing import Optional
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoEncoder:
    """Encodes video features into embeddings."""
    
    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        text_model: str = "all-MiniLM-L6-v2",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {clip_model}")
        import open_clip
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model)
        
        # Load Sentence Transformer
        logger.info(f"Loading Sentence Transformer: {text_model}")
        from sentence_transformers import SentenceTransformer
        self.text_model = SentenceTransformer(text_model, device=self.device)
        
        # Embedding dimensions
        self.clip_dim = 512  # ViT-B/32 output
        self.text_dim = 384  # all-MiniLM-L6-v2 output
        
        logger.info("Models loaded successfully")
    
    def _download_image(self, url: str, timeout: int = 10) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to download image: {e}")
            return None
    
    @torch.no_grad()
    def encode_thumbnail(self, image: Image.Image) -> np.ndarray:
        """Encode thumbnail image using CLIP."""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def encode_thumbnail_url(self, url: str) -> Optional[np.ndarray]:
        """Download and encode thumbnail from URL."""
        image = self._download_image(url)
        if image is None:
            return None
        return self.encode_thumbnail(image)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using Sentence Transformer."""
        if not text:
            return np.zeros(self.text_dim)
        
        # Truncate long text
        text = text[:1000]
        
        embedding = self.text_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_title(self, title: str) -> np.ndarray:
        """Encode video title."""
        return self.encode_text(title)
    
    def encode_description(self, description: str) -> np.ndarray:
        """Encode video description."""
        return self.encode_text(description or "")
    
    def encode_video(
        self,
        title: str,
        description: str = None,
        thumbnail_url: str = None,
    ) -> dict:
        """
        Encode all video features.
        
        Returns:
            dict with keys: thumbnail_embedding, title_embedding, description_embedding, combined_embedding
        """
        result = {
            "thumbnail_embedding": None,
            "title_embedding": None,
            "description_embedding": None,
            "combined_embedding": None,
        }
        
        # Encode thumbnail
        if thumbnail_url:
            result["thumbnail_embedding"] = self.encode_thumbnail_url(thumbnail_url)
        
        # Encode title
        result["title_embedding"] = self.encode_title(title)
        
        # Encode description
        result["description_embedding"] = self.encode_description(description)
        
        # Create combined embedding (for retrieval)
        result["combined_embedding"] = self._create_combined_embedding(
            result["thumbnail_embedding"],
            result["title_embedding"],
            result["description_embedding"],
        )
        
        return result
    
    def _create_combined_embedding(
        self,
        thumbnail_emb: Optional[np.ndarray],
        title_emb: np.ndarray,
        description_emb: np.ndarray,
        target_dim: int = 256,
    ) -> np.ndarray:
        """
        Create a combined embedding from all features.
        
        Strategy: Weighted average + PCA-like projection
        """
        embeddings = []
        weights = []
        
        # Add thumbnail (weight: 0.4)
        if thumbnail_emb is not None:
            embeddings.append(thumbnail_emb)
            weights.append(0.4)
        
        # Add title (weight: 0.4)
        embeddings.append(title_emb)
        weights.append(0.4)
        
        # Add description (weight: 0.2)
        embeddings.append(description_emb)
        weights.append(0.2)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Concatenate all embeddings
        concat = np.concatenate(embeddings)
        
        # Simple projection to target dimension using random but fixed projection
        np.random.seed(42)  # Fixed seed for reproducibility
        projection_matrix = np.random.randn(len(concat), target_dim)
        projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)
        
        combined = concat @ projection_matrix
        combined = combined / np.linalg.norm(combined)  # Normalize
        
        return combined


class BatchVideoEncoder:
    """Batch encoding for efficiency."""
    
    def __init__(self, encoder: VideoEncoder = None):
        self.encoder = encoder or VideoEncoder()
    
    def encode_videos(self, videos: list, show_progress: bool = True) -> list:
        """
        Encode multiple videos.
        
        Args:
            videos: List of dicts with keys: video_id, title, description, thumbnail_url
            
        Returns:
            List of dicts with video_id and embeddings
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(videos, desc="Encoding videos") if show_progress else videos
        
        for video in iterator:
            try:
                embeddings = self.encoder.encode_video(
                    title=video.get("title", ""),
                    description=video.get("description"),
                    thumbnail_url=video.get("thumbnail_url"),
                )
                
                results.append({
                    "video_id": video["video_id"],
                    **embeddings,
                })
            except Exception as e:
                logger.error(f"Error encoding video {video.get('video_id')}: {e}")
                continue
        
        return results


if __name__ == "__main__":
    # Test the encoder
    print("Initializing VideoEncoder...")
    encoder = VideoEncoder()
    
    # Test with a sample
    print("\nEncoding sample video...")
    result = encoder.encode_video(
        title="Python Tutorial for Beginners",
        description="Learn Python programming from scratch. This tutorial covers basics.",
        thumbnail_url=None,  # Skip thumbnail for quick test
    )
    
    print(f"\nResults:")
    print(f"  Title embedding shape: {result['title_embedding'].shape}")
    print(f"  Description embedding shape: {result['description_embedding'].shape}")
    print(f"  Combined embedding shape: {result['combined_embedding'].shape}")
    print(f"  Thumbnail embedding: {result['thumbnail_embedding']}")