"""
Embedder: Convert text to vector embeddings using SentenceTransformer
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """Generates embeddings for text using pre-trained SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list) -> np.ndarray:
        """Embed texts to vectors."""
        return self.model.encode(texts, show_progress_bar=False)
