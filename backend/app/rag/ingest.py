"""
ingest.py

Takes the raw document text, chunks it, embeds it, and stores it in a
FAISS index. Originally used chromadb for this but that pulls in
chroma-hnswlib which needs a C++ compiler to build on Windows - not
everyone has Visual C++ Build Tools installed and I didn't want that to
be a blocker for anyone trying to run this. FAISS ships pre-built
wheels for Windows/Mac/Linux so this just works with a plain pip install.

Kept it simple - one FAISS index + a parallel list of chunk texts per
document, held in memory. Fine for a demo project. Would move to a
persistent vector store if this needed to survive restarts or scale to
lots of concurrent documents.
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 200       # words per chunk
CHUNK_OVERLAP = 40     # words shared between consecutive chunks

_embedding_model = None

# doc_id -> {"index": faiss.Index, "chunks": [str, ...]}
# this is the in-memory "database" - resets every time the server restarts,
# which is fine for a portfolio project but wouldn't fly in production
_doc_indexes = {}


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but re-cover the overlap

    return chunks


def ingest_document(text: str, doc_id: str) -> int:
    """
    Chunks + embeds the doc and builds a FAISS index for it, keyed by
    doc_id so multiple documents don't collide with each other. Returns
    the number of chunks stored - mostly just handy for logging while
    testing this.
    """
    model = _get_embedding_model()

    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # normalizing so we can use inner product as cosine similarity -
    # cheaper than computing cosine distance manually every query
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = inner product
    index.add(embeddings)

    # overwrite if this doc_id was ingested before (shouldn't normally
    # happen since doc_id is a fresh uuid per upload, but just in case)
    _doc_indexes[doc_id] = {
        "index": index,
        "chunks": chunks,
    }

    return len(chunks)


def get_doc_index(doc_id: str):
    return _doc_indexes.get(doc_id)
