"""
classifier.py

Splits a contract into clause-sized chunks and tags each one with a
category (Termination, Liability, Payment Terms etc.) using zero-shot
classification.

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
    section_pattern = re.compile(
        r"\n(?=\s*(?:\d+\.\d*|\(?[a-zA-Z]\)|Section \d+|Article [IVXLC]+)\s)"
    )
    chunks = section_pattern.split(text)

    if len(chunks) <= 1:
        chunks = [p for p in text.split("\n\n") if p.strip()]

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

    # NOTE (updated): the old Inference API returned a dict shaped like
    # {"labels": [...], "scores": [...]}. The new router (Inference
    # Providers) returns a list of {"label": ..., "score": ...} objects,
    # already sorted highest-score-first - confirmed via a real
    # TypeError in production when this still assumed the old dict shape.
    # Handling both here so this doesn't break again if it changes back.
    if isinstance(result, dict) and "labels" in result:
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        runner_up = result["labels"][1] if len(result["labels"]) > 1 else None
    else:
        top_label = result[0]["label"]
        top_score = result[0]["score"]
        runner_up = result[1]["label"] if len(result) > 1 else None

    return {
        "label": top_label,
        "confidence": round(top_score, 4),
        "runner_up": runner_up,
    }


def classify_clause(clause_text: str) -> dict:
    return _classify_via_api(clause_text)


def _classify_batch(clauses: list) -> list:
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
