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

NOTE (updated): this used to load the model locally via
transformers.pipeline(). On a 512MB-RAM host (Render free tier), that
load alone was enough to OOM-kill the process the moment /analyze was
hit. Switched to calling the Hugging Face Inference API instead - the
model runs on HF's infrastructure, and our process just makes HTTP
calls. Requires HF_API_TOKEN to be set (see hf_client.py).
"""

import re
import os
from app.nlp.hf_client import query as hf_query

CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "valhalla/distilbart-mnli-12-3")
# NOTE: originally used facebook/bart-large-mnli, but at ~400M params it
# was the single biggest contributor to slow /analyze times (and, once
# self-hosted, to OOM crashes). This distilled MNLI model is noticeably
# faster/lighter with only a small accuracy tradeoff - fine for a demo.
# Set CLASSIFIER_MODEL in .env if you want to swap models.

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


def _classify_via_api(clause_text: str) -> dict:
    truncated = clause_text[:1000]

    result = hf_query(
        CLASSIFIER_MODEL,
        {
            "inputs": truncated,
            "parameters": {"candidate_labels": CLAUSE_LABELS, "multi_label": False},
        },
    )

    return {
        "label": result["labels"][0],
        "confidence": round(result["scores"][0], 4),
        "runner_up": result["labels"][1] if len(result["labels"]) > 1 else None,
    }


def classify_clause(clause_text: str) -> dict:
    return _classify_via_api(clause_text)


def _classify_batch(clauses: list) -> list:
    """
    NOTE (updated): the local transformers pipeline used to batch these
    internally for a speed win. The hosted Inference API doesn't give us
    that same control, so this now calls the API once per clause. Each
    call is a lightweight HTTP request rather than a local forward pass,
    so this trades a bit of latency for not crashing the host - a fine
    trade for a demo app.
    """
    results = []
    for clause in clauses:
        classification = _classify_via_api(clause)
        results.append(
            {
                "label": classification["label"],
                "confidence": classification["confidence"],
            }
        )
    return results


def classify_document(text: str, max_clauses: int = 60) -> dict:
    """
    Returns both the classified clauses and a flag for whether we had to
    truncate. Added the max_clauses cap after testing this with a large
    (6MB+) document - classifying thousands of clauses one at a time
    just isn't going to finish in any reasonable amount of time for a
    live demo. 60 clauses covers pretty much any real contract anyone
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