from pathlib import Path
import json
import faiss
import pickle
import numpy as np

from embedder import Embedder


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_PATH = BASE_DIR / "chunks" / "doc_chunks.json"
VECTOR_DB_PATH = BASE_DIR / "vectordb"


def load_chunks():
    with open(CHUNK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_index(chunks):
    texts = [c["page_content"] for c in chunks]

    embedder = Embedder()
    embeddings = embedder.embed(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings


def save_index(index, chunks):
    VECTOR_DB_PATH.mkdir(exist_ok=True)

    faiss.write_index(index, str(VECTOR_DB_PATH / "index.faiss"))

    with open(VECTOR_DB_PATH / "index.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index saved")


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = load_chunks()

    print(f"Total chunks: {len(chunks)}")

    index, embeddings = build_index(chunks)

    print("Saving index...")
    save_index(index, chunks)

    print("Done!")
