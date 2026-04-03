"""
Retriever Module: Semantic search using FAISS vectorstore
"""
from pathlib import Path
import faiss
import pickle
import numpy as np
from src.embedder import Embedder


class RAGRetriever:
    """Retrieves relevant document chunks using FAISS."""

    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parent.parent

        self.index = faiss.read_index(str(BASE_DIR / "vectordb" / "index.faiss"))

        with open(BASE_DIR / "vectordb" / "index.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self.embedder = Embedder()

    def retrieve(self, query: str, k: int = 5) -> list:
        """Retrieve top-k relevant chunks for a query."""
        query_embedding = self.embedder.embed([query])
        distances, indices = self.index.search(np.array(query_embedding), k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx]["page_content"])

        return results

