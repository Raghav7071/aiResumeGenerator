import os
import logging
import httpx
from typing import List

logger = logging.getLogger(__name__)

# Use the same model via HF Inference API
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}/pipeline/feature-extraction"

class EmbeddingModel:
    """Utility class for generating text embeddings using Hugging Face Inference API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            logger.warning("HUGGINGFACE_API_KEY not found. Embeddings will fail unless provided at runtime.")
        
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def generate(self, text: str) -> List[float]:
        """Calculate the embedding for a single text string via HF API."""
        if not text or not text.strip():
            return []
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    HF_API_URL,
                    headers=self.headers,
                    json={"inputs": text, "options": {"wait_for_model": True}}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"HF API Embedding failed: {e}")
            return []

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Calculate embeddings for an array of text strings via HF API."""
        valid = [t for t in texts if t and t.strip()]
        if not valid:
            return []
        
        logger.info(f"Embedding {len(valid)} chunks via HF API")
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    HF_API_URL,
                    headers=self.headers,
                    json={"inputs": valid, "options": {"wait_for_model": True}}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"HF API Batch Embedding failed: {e}")
            # Fallback or return empty list (ChromaDB will error if empty)
            return []

# Initialize a shared instance
embedding_model = EmbeddingModel()
