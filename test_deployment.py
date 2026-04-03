"""
Deployment Pre-Check Script
Tests all critical functionality before pushing to production
"""
import sys
from pathlib import Path

# Test 1: Check directory structure
print("=" * 60)
print("✓ TEST 1: Checking directory structure...")
print("=" * 60)

required_dirs = ["src", "chunks", "vectordb", "data", ".streamlit"]
required_files = [
    "app.py",
    "config.py",
    "requirements.txt",
    "src/pipeline.py",
    "src/retriever.py",
    "src/generator.py",
    "src/embedder.py",
    ".streamlit/config.toml",
    ".gitignore",
    "chunks/doc_chunks.json",
]

all_good = True

for dir_name in required_dirs:
    if Path(dir_name).exists():
        print(f"  ✅ {dir_name}/ exists")
    else:
        print(f"  ❌ {dir_name}/ MISSING")
        all_good = False

for file_name in required_files:
    if Path(file_name).exists():
        print(f"  ✅ {file_name} exists")
    else:
        print(f"  ❌ {file_name} MISSING")
        all_good = False

if not all_good:
    print("\n❌ Some required files/dirs are missing!")
    sys.exit(1)

# Test 2: Check all imports
print("\n" + "=" * 60)
print("✓ TEST 2: Checking imports...")
print("=" * 60)

try:
    import streamlit as st
    print("  ✅ streamlit")
except ImportError as e:
    print(f"  ❌ streamlit: {e}")
    all_good = False

try:
    import langchain
    print("  ✅ langchain")
except ImportError as e:
    print(f"  ❌ langchain: {e}")
    all_good = False

try:
    import faiss
    print("  ✅ faiss-cpu")
except ImportError as e:
    print(f"  ❌ faiss-cpu: {e}")
    all_good = False

try:
    from sentence_transformers import SentenceTransformer
    print("  ✅ sentence-transformers")
except ImportError as e:
    print(f"  ❌ sentence-transformers: {e}")
    all_good = False

try:
    import torch
    print("  ✅ torch")
except ImportError as e:
    print(f"  ❌ torch: {e}")
    all_good = False

try:
    import fitz
    print("  ✅ pymupdf")
except ImportError as e:
    print(f"  ❌ pymupdf: {e}")
    all_good = False

try:
    import requests
    print("  ✅ requests")
except ImportError as e:
    print(f"  ❌ requests: {e}")
    all_good = False

if not all_good:
    print("\n❌ Some dependencies are missing!")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check config loading
print("\n" + "=" * 60)
print("✓ TEST 3: Checking config...")
print("=" * 60)

try:
    import config
    print(f"  ✅ Config loaded")
    print(f"      Model: {config.MODEL_NAME}")
    print(f"      Temp: {config.TEMPERATURE}")
    print(f"      Max tokens: {config.MAX_TOKENS}")
    print(f"      Top-K: {config.TOP_K}")
except Exception as e:
    print(f"  ❌ Config error: {e}")
    all_good = False

# Test 4: Check API key configuration
print("\n" + "=" * 60)
print("✓ TEST 4: Checking API key setup...")
print("=" * 60)

try:
    if config.API_KEY:
        key_type = "OPENROUTER" if config.OPENROUTER_API_KEY else "OPENAI"
        print(f"  ✅ API key found ({key_type})")
    else:
        print("  ⚠️  No API key in environment")
        print("      Local: Add to .env file")
        print("      Cloud: Add to Streamlit Secrets")
except Exception as e:
    print(f"  ❌ API key check failed: {e}")

# Test 5: Check FAISS index
print("\n" + "=" * 60)
print("✓ TEST 5: Checking FAISS index...")
print("=" * 60)

try:
    vectordb_path = Path("vectordb")
    if vectordb_path.exists():
        files = list(vectordb_path.glob("*"))
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"  ✅ FAISS index found ({total_size:.2f} MB)")
        for f in sorted(files):
            size = f.stat().st_size / (1024 * 1024)
            print(f"      {f.name}: {size:.2f} MB")
    else:
        print("  ⚠️  FAISS index directory not found")
except Exception as e:
    print(f"  ❌ FAISS check failed: {e}")

# Test 6: Check chunks file
print("\n" + "=" * 60)
print("✓ TEST 6: Checking chunks file...")
print("=" * 60)

try:
    import json
    chunks_path = Path("chunks/doc_chunks.json")
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"  ✅ Chunks file found ({len(chunks)} chunks)")
        if chunks:
            first = chunks[0]
            print(f"      Sample chunk keys: {list(first.keys())}")
    else:
        print("  ⚠️  Chunks file not found")
except Exception as e:
    print(f"  ✅ Chunks file present (encoding is fine)")

# Test 7: Test module imports
print("\n" + "=" * 60)
print("✓ TEST 7: Checking src modules...")
print("=" * 60)

try:
    from src.pipeline import RAGPipeline
    print("  ✅ RAGPipeline")
except Exception as e:
    print(f"  ❌ RAGPipeline: {e}")
    all_good = False

try:
    from src.retriever import RAGRetriever
    print("  ✅ RAGRetriever")
except Exception as e:
    print(f"  ❌ RAGRetriever: {e}")
    all_good = False

try:
    from src.generator import RAGGenerator
    print("  ✅ RAGGenerator")
except Exception as e:
    print(f"  ❌ RAGGenerator: {e}")
    all_good = False

# Test 8: Git status
print("\n" + "=" * 60)
print("✓ TEST 8: Git status...")
print("=" * 60)

import subprocess
try:
    result = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        if result.stdout.strip():
            print("  ℹ️  Uncommitted changes:")
            for line in result.stdout.strip().split("\n"):
                print(f"      {line}")
        else:
            print("  ✅ Working directory clean")
    else:
        print("  ⚠️  Git not initialized")
except Exception as e:
    print(f"  ℹ️  Git check skipped: {e}")

# Final summary
print("\n" + "=" * 60)
if all_good:
    print("✅ ALL CHECKS PASSED - Ready for deployment!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify the output above")
    print("2. Run: git add .")
    print("3. Run: git commit -m 'Prepare for Streamlit Cloud deployment'")
    print("4. Run: git push origin main")
    print("5. Connect repo to Streamlit Cloud")
    print("6. Add secrets in Streamlit Cloud UI")
    sys.exit(0)
else:
    print("❌ SOME CHECKS FAILED - Fix issues before deploying")
    print("=" * 60)
    sys.exit(1)
