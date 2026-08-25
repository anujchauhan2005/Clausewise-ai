"""
evaluate.py

Run this after the backend deps are installed to get actual numbers
for the README. Two things get evaluated:

1. Clause classification - accuracy + per-class F1 against the labeled
   examples in eval_dataset.json
2. Summarization - ROUGE-1/2/L against the reference summaries

Kept this as a standalone script rather than a pytest suite since I'm
using it more to generate numbers to report than as a pass/fail gate -
though could definitely wire this into GitHub Actions later as an
actual CI check (added that to the roadmap).

Usage:
    cd eval
    python evaluate.py
"""

import json
import os
import sys

# same deal as in main.py - avoid transformers dragging in tensorflow if
# it happens to be installed on this machine, which caused a protobuf
# version crash that had nothing to do with the actual eval logic
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

# ugly but works - lets this script import from the backend app package
# without needing to install it or mess with PYTHONPATH manually
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sklearn.metrics import accuracy_score, f1_score, classification_report
from rouge_score import rouge_scorer

from app.nlp.classifier import classify_clause
from app.nlp.summarizer import summarize_clause


def load_eval_data():
    path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_classification(examples):
    print("\n--- Clause Classification ---")
    print(f"Running against {len(examples)} labeled examples...\n")

    y_true = []
    y_pred = []

    for ex in examples:
        prediction = classify_clause(ex["text"])
        y_true.append(ex["label"])
        y_pred.append(prediction["label"])

        # printing the mismatches as we go so I can eyeball what's
        # actually going wrong, rather than just staring at a final number
        marker = "✓" if prediction["label"] == ex["label"] else "✗"
        if marker == "✗":
            print(f"{marker} predicted='{prediction['label']}' actual='{ex['label']}' (conf: {prediction['confidence']})")
            print(f"   text: {ex['text'][:90]}...")

    acc = accuracy_score(y_true, y_pred)
    # macro avg since some clause types have way fewer examples than
    # others in this small dataset, don't want that skewing the score
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-class breakdown:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"accuracy": acc, "macro_f1": macro_f1}


def evaluate_summarization(examples):
    print("\n--- Summarization (ROUGE) ---")
    print(f"Running against {len(examples)} reference summaries...\n")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for ex in examples:
        generated = summarize_clause(ex["text"])
        scores = scorer.score(ex["reference_summary"], generated)

        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

        print(f"generated: {generated}")
        print(f"reference: {ex['reference_summary']}")
        print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}\n")

    avg_r1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_r2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rL = sum(rougeL_scores) / len(rougeL_scores)

    print(f"Average ROUGE-1: {avg_r1:.4f}")
    print(f"Average ROUGE-2: {avg_r2:.4f}")
    print(f"Average ROUGE-L: {avg_rL:.4f}")

    return {"rouge1": avg_r1, "rouge2": avg_r2, "rougeL": avg_rL}


if __name__ == "__main__":
    data = load_eval_data()

    classification_results = evaluate_classification(data["clause_classification"])
    summarization_results = evaluate_summarization(data["summarization"])

    print("\n=== Summary (copy this into the README) ===")
    print(f"Clause Classification — Accuracy: {classification_results['accuracy']:.2f}, Macro F1: {classification_results['macro_f1']:.2f}")
    print(f"Summarization — ROUGE-1: {summarization_results['rouge1']:.2f}, ROUGE-2: {summarization_results['rouge2']:.2f}, ROUGE-L: {summarization_results['rougeL']:.2f}")
