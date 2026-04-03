from pathlib import Path
import faiss
import pickle
import numpy as np

from embedder import Embedder

BASE_DIR = Path(__file__).resolve().parent.parent

index_path = BASE_DIR / "vectordb" / "index.faiss"
pkl_path = BASE_DIR / "vectordb" / "index.pkl"


# Load index
index = faiss.read_index(str(index_path))

with open(pkl_path, "rb") as f:
    chunks = pickle.load(f)

print("Loaded FAISS index and chunks")


# Load embedder
embedder = Embedder()


# Query
query = "What does ebay do?"
query_embedding = embedder.embed([query])


# Search
k = 3
distances, indices = index.search(np.array(query_embedding), k)


print("\nTop Results:\n")

for i, idx in enumerate(indices[0]):
    print(f"Result {i+1}:")
    print(chunks[idx]["page_content"])
    print("-" * 80)
