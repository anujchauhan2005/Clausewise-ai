"""
ingest.py

Takes the raw document text, chunks it, embeds it, and stores it in a
FAISS index.

NOTE (updated): embeddings used to come from a locally-loaded
sentence-transformers model (which pulls in torch). On a 512MB-RAM host
(Render free tier), just importing torch was eating enough memory that
combined with everything else running, the process would get OOM-killed.
Switched to calling the Hugging Face Inference API for embeddings too -
see hf_client.py. This means torch/transformers/sentence-transformers
aren't needed locally at all anymore (removed from requirements.txt).

Kept the rest simple - one FAISS index + a parallel list of chunk texts
per document, held in memory. Fine for a demo project. Would move to a
persistent vector store if this needed to survive restarts or scale to
lots of concurrent documents.
"""

import os
import faiss
import numpy as np
from app.nlp.hf_client import query as hf_query

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 200       # words per chunk
CHUNK_OVERLAP = 40     # words shared between consecutive chunks

# doc_id -> {"index": faiss.Index, "chunks": [str, ...]}
# this is the in-memory "database" - resets every time the server restarts,
# which is fine for a portfolio project but wouldn't fly in production
_doc_indexes = {}


def _mean_pool(token_embeddings: list) -> list:
    """
    Some HF Inference API responses for feature-extraction come back as
    per-token embeddings (shape: tokens x dims) rather than a single
    pooled sentence vector. If that happens, average over the token
    dimension to get one vector per input - a reasonable equivalent to
    what sentence-transformers' mean pooling does internally.
    """
    arr = np.array(token_embeddings)
    if arr.ndim == 2:
        return arr.mean(axis=0).tolist()
    return token_embeddings  # already a flat vector


def embed_texts(texts: list) -> np.ndarray:
    """
    Calls the HF Inference API's feature-extraction endpoint for the
    configured embedding model. Sentence-transformers models hosted on
    HF typically return one pooled vector per input already, but this
    defensively mean-pools if a response comes back as per-token
    embeddings instead.
    """
    response = hf_query(
        EMBEDDING_MODEL_NAME,
        {"inputs": texts, "options": {"wait_for_model": True}},
    )

    vectors = []
    for item in response:
        # item is either a flat list[float] (already pooled) or a
        # list[list[float]] (per-token) depending on the model/endpoint
        if isinstance(item[0], list):
            vectors.append(_mean_pool(item))
        else:
            vectors.append(item)

    return np.array(vectors).astype("float32")


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
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)

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