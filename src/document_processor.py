"""
Document Processor: Extract, clean, and chunk PDF documents
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("pymupdf not installed. Run: pip install pymupdf")


class DocumentProcessor:
    """Handles PDF extraction, cleaning, and chunking."""

    def __init__(
        self,
        chunk_size_words: int = 200,
        chunk_overlap_words: int = 50,
        min_chunk_words: int = 100
    ):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.min_chunk_words = min_chunk_words

    def extract_pdf_pages(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Extract text from all PDF pages."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        pages = []

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            pages.append({
                "page": page_num + 1,
                "text": text
            })

        doc.close()
        return pages

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing artifacts and normalizing whitespace.

        Args:
            text: Raw text from PDF

        Returns:
            Cleaned text
        """
        text = text.replace("\x00", " ")                          # null bytes
        text = re.sub(r"-\s*\n\s*", "", text)                    # join hyphenated line breaks
        text = re.sub(r"\n+", "\n", text)                         # collapse repeated newlines
        text = re.sub(r"[ \t]+", " ", text)                       # collapse spaces/tabs
        text = re.sub(r"\n\s+", "\n", text)                       # trim space after newlines
        text = re.sub(r'(\b\w+\b)(?:\s*\1\b)+', r'\1', text)    # remove duplicate words
        text = text.strip()
        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split cleaned text into sentences.

        Args:
            text: Cleaned text

        Returns:
            List of sentences
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """
        Chunk sentences into fixed-size word chunks with overlap.

        Args:
            sentences: List of sentences

        Returns:
            List of text chunks
        """
        chunks = []
        current_words = []

        for sentence in sentences:
            sent_words = sentence.split()
            if not sent_words:
                continue

            # If sentence itself is too long, split it safely
            while len(sent_words) > self.chunk_size_words:
                part = sent_words[:self.chunk_size_words]
                if part:
                    chunks.append(" ".join(part).strip())
                sent_words = (
                    sent_words[self.chunk_size_words - self.chunk_overlap_words:]
                    if self.chunk_overlap_words
                    else sent_words[self.chunk_size_words:]
                )

            # Normal case: add sentence to current chunk
            if len(current_words) + len(sent_words) <= self.chunk_size_words:
                current_words.extend(sent_words)
            else:
                if current_words:
                    chunks.append(" ".join(current_words).strip())
                    current_words = (
                        current_words[-self.chunk_overlap_words:]
                        if self.chunk_overlap_words
                        else []
                    )
                current_words.extend(sent_words)

        if current_words:
            chunks.append(" ".join(current_words).strip())

        return chunks

    def merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks smaller than minimum word threshold.

        Args:
            chunks: List of text chunks

        Returns:
            Merged chunks
        """
        merged = []

        for chunk in chunks:
            if merged and len(chunk.split()) < self.min_chunk_words:
                merged[-1] += " " + chunk
            else:
                merged.append(chunk)

        return merged

    def process_pdf(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Complete pipeline: extract → clean → sentence split → chunk.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of processed chunks with metadata
        """
        # Extract pages
        pdf_path = Path(pdf_path)
        pages = self.extract_pdf_pages(pdf_path)
        print(f"✓ Extracted {len(pages)} pages from PDF")

        all_chunks = []

        # Process each page
        for page_data in pages:
            raw_text = page_data["text"]
            cleaned = self.clean_text(raw_text)
            sentences = self.split_into_sentences(cleaned)
            page_chunks = self.chunk_sentences(sentences)

            # Add metadata to each chunk
            for idx, chunk in enumerate(page_chunks, start=1):
                all_chunks.append({
                    "chunk_id": len(all_chunks) + 1,
                    "page_content": chunk,
                    "metadata": {
                        "page": page_data["page"],
                        "page_chunk": idx,
                        "word_count": len(chunk.split())
                    }
                })

        print(f"✓ Created {len(all_chunks)} chunks")

        # Calculate stats
        word_counts = [c["metadata"]["word_count"] for c in all_chunks]
        print(f"  - Min words: {min(word_counts)}")
        print(f"  - Max words: {max(word_counts)}")
        print(f"  - Avg words: {sum(word_counts) / len(word_counts):.1f}")

        return all_chunks

    @staticmethod
    def save_chunks(chunks: List[Dict[str, Any]], output_path: Union[str, Path, None] = None) -> Path:
        """
        Save chunks to JSON file (compatible with retriever).

        Args:
            chunks: List of processed chunks
            output_path: Path to save (defaults to chunks/doc_chunks.json)

        Returns:
            Path where chunks were saved
        """
        if output_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            output_path = base_dir / "chunks" / "doc_chunks.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"✓ Chunks saved to: {output_path.resolve()}")
        return output_path


def process_and_save(pdf_path: Union[str, Path], output_path: Union[str, Path, None] = None) -> Path:
    """
    Convenience function: process PDF and save chunks in one call.

    Args:
        pdf_path: Path to PDF file
        output_path: Path to save chunks (optional)

    Returns:
        Path where chunks were saved
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    processor = DocumentProcessor(
        chunk_size_words=200,
        chunk_overlap_words=50,
        min_chunk_words=100
    )

    chunks = processor.process_pdf(pdf_path)
    return DocumentProcessor.save_chunks(chunks, Path(output_path) if output_path else None)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    pdf_file = BASE_DIR / "data" / "AI Training Document.pdf"

    process_and_save(pdf_file)
