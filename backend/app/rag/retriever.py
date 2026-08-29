"""
retriever.py

Just the search part - given a doc_id and a question, pulls the most
relevant chunks back out of the FAISS index built in ingest.py. Kept
this dumb and simple on purpose, all the "what do we do with these
chunks" logic lives in qa.py.

NOTE (updated): the query embedding used to come from the same local
sentence-transformers model as ingest.py. Now goes through the same HF
Inference API call (embed_texts) so there's no local model load at all
for the RAG pipeline. See ingest.py / hf_client.py.
"""

import faiss

from app.rag.ingest import get_doc_index, embed_texts


def retrieve_relevant_chunks(doc_id: str, query: str, top_k: int = 4) -> list:
    doc_data = get_doc_index(doc_id)
    if doc_data is None:
        # doc hasn't been ingested (or was ingested under a different id)
        return []

    index = doc_data["index"]
    chunks = doc_data["chunks"]

    query_embedding = embed_texts([query])
    faiss.normalize_L2(query_embedding)

    # don't ask for more results than there actually are chunks, faiss
    # doesn't love that
    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # faiss pads with -1 if fewer than k results exist
        results.append({
            "text": chunks[idx],
            # since we normalized + used inner product, this score is
            # already basically cosine similarity, 1.0 = identical
            "relevance_score": round(float(score), 4),
        })

    return results