"""
classifier.py

Splits a contract into clause-sized chunks and tags each one with a
category (Termination, Liability, Payment Terms etc.) using zero-shot
classification.

Went back and forth on this - originally wanted to fine-tune a proper
classifier on LegalBERT but didn't have a labeled dataset big enough to
make that worthwhile for v1. Zero-shot with bart-large-mnli gets
surprisingly decent results on clause-level text since the categories
are fairly distinct semantically. Fine-tuning is on the roadmap once
I've got more labeled examples (see eval_dataset.json - been adding to
it manually as I test).
"""

import os
import re
from functools import lru_cache
from transformers import pipeline

CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "valhalla/distilbart-mnli-12-3")
# NOTE (updated): originally used facebook/bart-large-mnli, but at ~400M
# params it was the single biggest contributor to slow /analyze times on
# CPU (each clause took 1-2s, and they were processed one at a time).
# Switched to this distilled MNLI model - noticeably faster with only a
# small accuracy tradeoff, which is a fine trade for a CPU-only demo.
# Set CLASSIFIER_MODEL in .env if you want to swap back for higher
# accuracy and have the compute budget for it.

# these are the categories I settled on after looking at ~10 different
# contract types (NDAs, employment, loan agreements, leases). Could
# probably add more granular ones later (e.g. splitting "Payment Terms"
# into "Late Fees" / "Payment Schedule") but this is a reasonable start
CLAUSE_LABELS = [
    "Termination Clause",
    "Liability Clause",
    "Payment Terms",
    "Confidentiality Clause",
    "Indemnification Clause",
    "Governing Law",
    "Force Majeure",
    "Non-Compete Clause",
    "Dispute Resolution",
    "General / Other",
]


@lru_cache(maxsize=1)
def _get_classifier():
    # lru_cache so the model only loads once per process, not per request.
    # this thing takes a good few seconds to load so definitely don't
    # want that happening on every API call
    return pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)


def split_into_clauses(text: str) -> list:
    """
    Contracts don't come with clean paragraph breaks half the time, so
    this splits on numbered sections (1., 2., Section 3, Article IV etc.)
    first, and falls back to plain paragraph splitting if none of those
    patterns match anything.
    """
    # matches things like "1.", "2.1", "Section 3", "Article IV", "(a)"
    section_pattern = re.compile(
        r"\n(?=\s*(?:\d+\.\d*|\(?[a-zA-Z]\)|Section \d+|Article [IVXLC]+)\s)"
    )
    chunks = section_pattern.split(text)

    if len(chunks) <= 1:
        # fallback - just split on blank lines
        chunks = [p for p in text.split("\n\n") if p.strip()]

    # filter out tiny fragments (headers, page numbers, stray whitespace)
    # anything under ~25 chars is basically never a real clause
    clauses = [c.strip() for c in chunks if len(c.strip()) > 25]

    return clauses


def classify_clause(clause_text: str) -> dict:
    classifier = _get_classifier()

    # truncating because the model has a token limit and some clauses
    # (looking at you, indemnification sections) run really long
    truncated = clause_text[:1000]

    result = classifier(truncated, candidate_labels=CLAUSE_LABELS, multi_label=False)

    return {
        "label": result["labels"][0],
        "confidence": round(result["scores"][0], 4),
        # keeping the runner-up in case confidence is low and someone
        # wants to see what the second guess was
        "runner_up": result["labels"][1] if len(result["labels"]) > 1 else None,
    }


def _classify_batch(clauses: list) -> list:
    """
    Runs all clauses through the classifier in one call instead of looping
    one at a time. This was the single biggest speed win when I profiled
    this - transformers pipelines batch internally when given a list,
    which uses the CPU a lot more efficiently than N separate forward
    passes. Went from ~1-2s per clause to something much closer to
    ~0.3-0.5s per clause on my machine once batched.
    """
    classifier = _get_classifier()
    truncated = [c[:1000] for c in clauses]

    # batch_size caps how many go through the model at once - keeps memory
    # usage sane on a CPU-only machine while still getting the batching
    # speedup. tuned this down from higher values after it started
    # thrashing memory on a bigger test document
    results = classifier(truncated, candidate_labels=CLAUSE_LABELS, multi_label=False, batch_size=8)

    # pipeline returns a single dict instead of a list if given a single
    # item - normalize so callers always get a list back
    if isinstance(results, dict):
        results = [results]

    return [
        {
            "label": r["labels"][0],
            "confidence": round(r["scores"][0], 4),
        }
        for r in results
    ]


def classify_document(text: str, max_clauses: int = 60) -> dict:
    """
    Returns both the classified clauses and a flag for whether we had to
    truncate. Added the max_clauses cap after testing this with a large
    (6MB+) document - on CPU, classifying thousands of clauses one at a
    time just isn't going to finish in any reasonable amount of time for
    a live demo. 60 clauses covers pretty much any real contract anyone
    would realistically upload to try this out; genuinely huge documents
    would need batched/async processing, which is out of scope for what
    this project is trying to demonstrate.
    """
    clauses = split_into_clauses(text)
    truncated = len(clauses) > max_clauses
    clauses_to_process = clauses[:max_clauses]

    if not clauses_to_process:
        return {"clauses": [], "total_clauses_found": 0, "clauses_processed": 0, "truncated": False}

    batch_results = _classify_batch(clauses_to_process)

    results = []
    for clause, classification in zip(clauses_to_process, batch_results):
        results.append(
            {
                "text": clause[:300] + ("..." if len(clause) > 300 else ""),
                "full_text": clause,
                "category": classification["label"],
                "confidence": classification["confidence"],
            }
        )

    return {
        "clauses": results,
        "total_clauses_found": len(clauses),
        "clauses_processed": len(clauses_to_process),
        "truncated": truncated,
    }
