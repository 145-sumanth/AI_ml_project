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

DATA_DIR = Path(__file__).resolve().parent / "docs"
COLLECTION_NAME = "zepto_docs"


def load_docs(path: Path) -> List[tuple[str, str]]:
    """Return a list of (id, text) for each document file in the docs/ folder.

    Uses the basename (without extension) as the document id.
    """
    docs = []
    for p in sorted(path.glob("doc_*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        docs.append((p.stem, text))
    return docs


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_docs(DATA_DIR)
    if not docs:
        print("No documents found in", DATA_DIR)
        return

    # Load the embedding model (downloads first time)
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create a Chroma client and collection. The metadata is included per spec.
    client = chromadb.Client()

    # Remove any existing collection with the same name to keep runs idempotent.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, texts, embeddings, metadatas = [], [], [], []
    for doc_id, text in docs:
        try:
            # One chunk per doc — embed the full doc text
            emb = model.encode([text], show_progress_bar=False)[0]
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

    # Add to collection
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    # Persist if the chroma client supports persistence (some installations do)
    try:
        client.persist()
    except Exception:
        pass

    print(f"Ingested {len(ids)} documents into ChromaDB collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
