"""
summarizer.py

Plain-English summary of the whole contract. Using distilbart instead
of the full bart-large-cnn because the full model was way too slow on
my laptop (no GPU) and distilbart gets close enough quality for what
this needs. Can swap the model via env var if someone's running this
with actual GPU access.

One annoying thing - most summarization models cap out around 1024
tokens input, but contracts are obviously way longer than that. Dealt
with this by chunking the doc and summarizing each chunk, then doing a
second pass summarization over the combined chunk-summaries. Not the
most elegant solution but it works and doesn't lose too much info.
"""

import os
from functools import lru_cache
from transformers import pipeline

SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")

# roughly how many words fit safely under the model's token limit -
# not exact since tokens != words but gives enough buffer
CHUNK_WORD_LIMIT = 600


@lru_cache(maxsize=1)
def _get_summarizer():
    return pipeline("summarization", model=SUMMARIZER_MODEL)


def _chunk_text(text: str, word_limit: int = CHUNK_WORD_LIMIT) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), word_limit):
        chunks.append(" ".join(words[i:i + word_limit]))
    return chunks


def summarize_document(text: str, max_length: int = 130, min_length: int = 40) -> str:
    summarizer = _get_summarizer()
    chunks = _chunk_text(text)

    if len(chunks) == 1:
        result = summarizer(chunks[0], max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]["summary_text"]

    # multi-chunk case - summarize each piece first
    chunk_summaries = []
    for chunk in chunks:
        # using smaller max_length per chunk since we're going to combine
        # and summarize again after this
        out = summarizer(chunk, max_length=80, min_length=20, do_sample=False)
        chunk_summaries.append(out[0]["summary_text"])

    combined = " ".join(chunk_summaries)

    # second pass - if the combined summaries are still long, summarize again
    if len(combined.split()) > CHUNK_WORD_LIMIT:
        final = summarizer(combined[:4000], max_length=max_length, min_length=min_length, do_sample=False)
        return final[0]["summary_text"]

    return combined


def summarize_clause(clause_text: str) -> str:
    """
    Shorter summary for a single clause - used when someone wants the
    plain-English version of just one section instead of the whole doc.
    """
    summarizer = _get_summarizer()

    word_count = len(clause_text.split())
    if word_count < 30:
        # not worth summarizing something already that short, model tends
        # to just repeat it back or produce garbage on very short inputs
        return clause_text

    result = summarizer(clause_text[:1000], max_length=60, min_length=15, do_sample=False)
    return result[0]["summary_text"]
