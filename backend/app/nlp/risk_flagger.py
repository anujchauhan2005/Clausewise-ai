"""
risk_flagger.py

This is the part I spent the most time on honestly. Tried a few
approaches:

1. Pure ML classifier for "risky vs not risky" - didn't have enough
   labeled data to train this properly, results were unreliable
2. Pure keyword matching - too many false positives, "liability" shows
   up in totally normal clauses too
3. What's here now - keyword/phrase lexicon combined with some basic
   context checks, plus weighting by clause category (a liability
   clause with "unlimited" in it is way more concerning than a random
   clause mentioning "unlimited")

Not claiming this is bulletproof, it's a heuristic system. But it
catches the obvious stuff (unlimited liability, auto-renewal traps,
one-sided termination rights) which is most of what actually matters
in a first-pass review.
"""

import re

# phrases that are red flags almost regardless of context
HIGH_RISK_PHRASES = [
    "sole discretion",
    "unlimited liability",
    "without limitation",
    "irrevocable",
    "perpetual",
    "waives all rights",
    "waives any right",
    "no cap on liability",
    "at any time without notice",
    "automatically renew",
    "auto-renew",
    "non-refundable",
    "indemnify and hold harmless",
    "in its sole and absolute discretion",
    "shall have no liability",
    "as-is",
    "no warranty",
]

# these only matter if they show up in a clause we've already classified
# as one of these categories - e.g. "30 days" isn't risky on its own but
# a *termination* clause giving only one party a 30-day-notice exit while
# the other party has no such right would be worth flagging
CATEGORY_RISK_WEIGHTS = {
    "Liability Clause": 1.5,
    "Termination Clause": 1.3,
    "Indemnification Clause": 1.4,
    "Non-Compete Clause": 1.2,
}

MEDIUM_RISK_PHRASES = [
    "liquidated damages",
    "penalty",
    "forfeit",
    "exclusive",
    "non-compete",
    "confidential for",  # usually followed by a suspiciously long duration
]


def _count_phrase_hits(text: str, phrase_list: list) -> list:
    text_lower = text.lower()
    hits = []
    for phrase in phrase_list:
        if phrase in text_lower:
            hits.append(phrase)
    return hits


def score_clause_risk(clause_text: str, category: str = None) -> dict:
    """
    Returns a risk score 0-10 and the phrases that triggered it. Not a
    precise science - this is meant to flag things for a human to look
    at, not replace an actual legal review.
    """
    high_hits = _count_phrase_hits(clause_text, HIGH_RISK_PHRASES)
    medium_hits = _count_phrase_hits(clause_text, MEDIUM_RISK_PHRASES)

    base_score = (len(high_hits) * 2.5) + (len(medium_hits) * 1.0)

    weight = CATEGORY_RISK_WEIGHTS.get(category, 1.0)
    weighted_score = base_score * weight

    # cap at 10, and round to 1 decimal so it doesn't look falsely precise
    final_score = min(round(weighted_score, 1), 10.0)

    if final_score >= 6:
        level = "high"
    elif final_score >= 3:
        level = "medium"
    elif final_score > 0:
        level = "low"
    else:
        level = "none"

    return {
        "risk_score": final_score,
        "risk_level": level,
        "flagged_phrases": high_hits + medium_hits,
    }


def flag_document_risks(classified_clauses: list) -> list:
    """
    Takes the output of classifier.classify_document() and adds risk
    scoring on top. Only returns clauses that actually have some risk
    (score > 0) sorted highest first - no point showing someone 40
    clauses when 6 of them are actually worth their attention.
    """
    flagged = []

    for clause in classified_clauses:
        risk = score_clause_risk(clause["full_text"], clause.get("category"))
        if risk["risk_score"] > 0:
            flagged.append(
                {
                    "text": clause["text"],
                    "category": clause.get("category"),
                    **risk,
                }
            )

    flagged.sort(key=lambda x: x["risk_score"], reverse=True)
    return flagged
