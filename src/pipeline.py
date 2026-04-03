"""
Pipeline Module: Orchestrates the RAG pipeline (retriever + generator)
"""
from src.retriever import RAGRetriever
from src.generator import RAGGenerator


class RAGPipeline:
    """Orchestrates the complete retrieval-augmented generation pipeline."""

    def __init__(self):
        self.retriever = RAGRetriever()
        self.generator = RAGGenerator()

    def run(self, query: str, k: int = 10) -> str:
        """Process query and return non-streaming response."""
        if not query or not query.strip():
            return "Please provide a valid question."

        chunks = self.retriever.retrieve(query, k=k)

        if not chunks:
            return "No relevant documents found."

        return self.generator.generate(query, chunks)

    def stream_run(self, query: str, k: int = 10):
        """
        Process query with streaming response.
        Returns both stream generator and retrieved chunks for source display.
        """
        chunks = self.retriever.retrieve(query, k=k)
        stream = self.generator.stream_generate(query, chunks)
        return stream, chunks

