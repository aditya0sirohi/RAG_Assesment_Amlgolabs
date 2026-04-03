"""
Configuration: Model names, paths, hyperparameters
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Try local .env first, then fallback to Streamlit Cloud secrets
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

# LLM Configuration
MODEL_NAME = "arcee-ai/trinity-large-preview:free"
TEMPERATURE = 0.7
MAX_TOKENS = 500

# API Keys - Support both local (.env) and Streamlit Cloud (st.secrets)
def _get_api_key(key_name):
    """Get API key from environment or Streamlit secrets."""
    # First try environment variables (local development)
    if os.getenv(key_name):
        return os.getenv(key_name)
    
    # Then try Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    
    return None

OPENROUTER_API_KEY = _get_api_key("OPENROUTER_API_KEY")
OPENAI_API_KEY = _get_api_key("OPENAI_API_KEY")
API_KEY = OPENROUTER_API_KEY or OPENAI_API_KEY

# Only log warning in development (when running locally)
if not API_KEY and not os.getenv("STREAMLIT_SERVER_HEADLESS"):
    import warnings
    warnings.warn(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in:\n"
        "  Local: Create .env file\n"
        "  Cloud: Streamlit Settings → Secrets",
        UserWarning
    )

# Retrieval Configuration
VECTORDB_DIR = "vectordb"
CHUNK_FILE = "chunks/doc_chunks.json"
TOP_K = 10

# Preprocessing Configuration
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
DATA_DIR = "data"
NOTEBOOKS_DIR = "notebooks"


def validate_paths():
    """Verify required directories and files exist."""
    issues = []
    
    if not Path(VECTORDB_DIR).exists():
        issues.append(f"Missing {VECTORDB_DIR}/ - Run: python src/build_faiss.py")
    
    if not Path(CHUNK_FILE).exists():
        issues.append(f"Missing {CHUNK_FILE} - Run: python src/document_processor.py")
    
    return issues


if __name__ == "__main__":
    print(f"Model: {MODEL_NAME}")
    print(f"API Key: {'Set' if API_KEY else 'NOT SET'}")
    print(f"Top-K: {TOP_K}")
    
    issues = validate_paths()
    if issues:
        print("\nSetup Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nAll paths validated")

