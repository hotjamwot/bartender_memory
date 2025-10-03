"""Pre-download and save the embedding model to the local models folder.

Run this once while online to cache the model for offline operation.

Usage:
    source venv/bin/activate
    python scripts/preload_model.py
"""
import os
from server.config import EMBEDDING_MODEL, EMBEDDING_MODEL_LOCAL_PATH
from sentence_transformers import SentenceTransformer


def main():
    os.makedirs(EMBEDDING_MODEL_LOCAL_PATH, exist_ok=True)
    print(f"Loading model {EMBEDDING_MODEL} (this may take a while)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    target = os.path.join(EMBEDDING_MODEL_LOCAL_PATH, f"sentence-transformers_{EMBEDDING_MODEL}")
    print(f"Saving model to {target}...")
    model.save(target)
    print("Done. You can now run the server offline using the local model folder.")


if __name__ == '__main__':
    main()
