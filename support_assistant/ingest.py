"""Ingest the documents into a ChromaDB collection using sentence-transformers.

Behavior:
- One chunk per document (no additional chunking).
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Create ChromaDB collection with metadata={"hnsw:space": "cosine"}

This script is defensive: it will print helpful messages and not crash on
transient failures. Running it will download the embedding model the first
time (allowed per project rules).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

# Use the sentence-transformers library for embeddings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DATA_DIR = Path(__file__).resolve().parent / "docs"
COLLECTION_NAME = "zepto_docs"
PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"


def load_docs(path: Path) -> List[tuple[str, str]]:
    """Return a list of (id, text) for each document file in the docs/ folder.

    Uses the filename stem (doc_01 etc.) as the document id.
    """
    docs = []
    for p in sorted(path.glob("doc_*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        docs.append((p.stem, text))
    return docs


def main() -> None:
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_docs(DATA_DIR)
    if not docs:
        print("No documents found in", DATA_DIR)
        return

    # Load the embedding model (downloads first time). This is allowed once.
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create a persistent Chroma client that stores its database under ./chroma_db
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(PERSIST_DIR))
    client = chromadb.Client(settings=settings)

    # If the collection already exists, delete it so runs are idempotent and reproducible
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' to rebuild it.")
    except Exception:
        pass

    # Create new collection with required metadata
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, texts, embeddings, metadatas = [], [], [], []
    for doc_id, text in docs:
        try:
            # One chunk per document — embed the entire document text
            emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
            ids.append(doc_id)
            texts.append(text)
            embeddings.append(emb.tolist())
            metadatas.append({"source": f"docs/{doc_id}.txt"})
            print(f"Prepared embedding for {doc_id}")
        except Exception as exc:
            print(f"Skipping {doc_id} due to embed error: {exc}")

    if not ids:
        print("No embeddings were created; exiting.")
        return

    # Add vectors to the collection and persist to disk
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    client.persist()

    print(f"Ingested {len(ids)} documents into persistent ChromaDB collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
