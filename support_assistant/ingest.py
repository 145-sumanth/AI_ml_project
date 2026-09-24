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
    # Newer chromadb versions may deprecate certain Settings; try the preferred Settings call
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(PERSIST_DIR))
        client = chromadb.Client(settings=settings)
    except Exception as exc:
        # Fall back to a more tolerant client creation so ingestion can proceed in this environment
        print('Warning: chromadb.Settings client creation failed:', exc)
        try:
            # Try PersistentClient fallback (some chromadb versions provide this)
            client = chromadb.PersistentClient(persist_directory=str(PERSIST_DIR))
            print('Using chromadb.PersistentClient fallback')
        except Exception:
            # Final fallback to in-memory client (non-persistent)
            print('PersistentClient not available; falling back to chromadb.Client() (in-memory).')
            client = chromadb.Client()

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

    # Add vectors to the collection
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    # Persist if the client implementation supports it (some chromadb clients are in-memory)
    if hasattr(client, 'persist') and callable(getattr(client, 'persist')):
        client.persist()
        print('Persisted ChromaDB collection to disk')
    else:
        print('Warning: chromadb client has no persist() method; collection is in-memory for this run')

    # ALSO write a fallback JSON with ids, texts, and embeddings so other processes
    # (the FastAPI server) can load the same vectors even if chromadb persistence
    # is unavailable in this environment.
    try:
        import json
        fallback_dir = PERSIST_DIR
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / 'fallback.json'
        fallback_data = [
            {'id': i, 'text': t, 'embedding': e}
            for i, t, e in zip(ids, texts, embeddings)
        ]
        with open(fallback_path, 'w', encoding='utf-8') as fh:
            json.dump(fallback_data, fh, ensure_ascii=False, indent=2)
        print(f'Wrote fallback JSON with {len(fallback_data)} docs to', fallback_path)
    except Exception as exc:
        print('Failed to write fallback JSON:', exc)

    print(f"Ingested {len(ids)} documents into ChromaDB collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
