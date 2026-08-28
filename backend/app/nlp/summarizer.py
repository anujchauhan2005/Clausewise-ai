"""
summarizer.py

Plain-English summary of the whole contract.

One annoying thing - most summarization models cap out around 1024
tokens input, but contracts are obviously way longer than that. Dealt
with this by chunking the doc and summarizing each chunk, then doing a
second pass summarization over the combined chunk-summaries. Not the
most elegant solution but it works and doesn't lose too much info.

NOTE (updated): this used to load distilbart-cnn locally via
transformers.pipeline(). Combined with the classifier model, that was
enough to OOM-kill the process on a 512MB-RAM host. Switched to calling
the Hugging Face Inference API instead - see hf_client.py. Requires
HF_API_TOKEN to be set in the environment.
"""

import os
from app.nlp.hf_client import query as hf_query

SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")

# roughly how many words fit safely under the model's token limit -
# not exact since tokens != words but gives enough buffer
CHUNK_WORD_LIMIT = 600


def _chunk_text(text: str, word_limit: int = CHUNK_WORD_LIMIT) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), word_limit):
        chunks.append(" ".join(words[i:i + word_limit]))
    return chunks


def _summarize_via_api(text: str, max_length: int, min_length: int) -> str:
    result = hf_query(
        SUMMARIZER_MODEL,
        {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
            },
        },
    )
    # the Inference API returns a list of dicts, same shape as the local
    # pipeline used to
    return result[0]["summary_text"]


def summarize_document(text: str, max_length: int = 130, min_length: int = 40) -> str:
    chunks = _chunk_text(text)

    if len(chunks) == 1:
        return _summarize_via_api(chunks[0], max_length=max_length, min_length=min_length)

    # multi-chunk case - summarize each piece first
    chunk_summaries = []
    for chunk in chunks:
        # using smaller max_length per chunk since we're going to combine
        # and summarize again after this
        chunk_summaries.append(_summarize_via_api(chunk, max_length=80, min_length=20))

    combined = " ".join(chunk_summaries)

    # second pass - if the combined summaries are still long, summarize again
    if len(combined.split()) > CHUNK_WORD_LIMIT:
        return _summarize_via_api(combined[:4000], max_length=max_length, min_length=min_length)

    return combined


def summarize_clause(clause_text: str) -> str:
    """
    Shorter summary for a single clause - used when someone wants the
    plain-English version of just one section instead of the whole doc.
    """
    word_count = len(clause_text.split())
    if word_count < 30:
        # not worth summarizing something already that short, model tends
        # to just repeat it back or produce garbage on very short inputs
        return clause_text

    return _summarize_via_api(clause_text[:1000], max_length=60, min_length=15)