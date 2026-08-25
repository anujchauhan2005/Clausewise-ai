"""
ner.py

Handles entity extraction from contract text. Uses spaCy's default NER
for people/orgs/dates and then some regex on top because spaCy alone
kept missing money amounts written in weird formats (e.g. "Rs. 5,00,000"
or "$25,000.00" or just "twenty five thousand dollars").

Ran this against ~15 sample contracts while building it and kept tweaking
the regex until it stopped missing obvious stuff. Still not perfect for
edge cases like ranges ("$5,000 - $10,000") but good enough for v1.
"""

import re
import spacy

# loading this once at module level so we don't reload the model on every
# request - learned this the hard way after the API was taking 3+ seconds
# per call in early testing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # in case someone forgets to run the download command
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# regex patterns for stuff spaCy's MONEY label misses half the time
MONEY_PATTERN = re.compile(
    r"(?:Rs\.?|INR|USD|\$)\s?[\d,]+(?:\.\d{1,2})?|"
    r"\b[\d,]+(?:\.\d{1,2})?\s?(?:rupees|dollars)\b",
    re.IGNORECASE,
)

# things like "30 days", "6 months", "2 years" - common in notice periods,
# termination clauses, warranty periods etc.
DURATION_PATTERN = re.compile(
    r"\b\d+\s?(?:day|days|month|months|year|years|week|weeks)\b",
    re.IGNORECASE,
)

PERCENT_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%")


def extract_entities(text: str) -> dict:
    """
    Runs spaCy NER + our custom regex over the text and returns everything
    bucketed by type. Doing dedup with dict.fromkeys instead of set() because
    I want to preserve the order they appear in the document.
    """
    doc = nlp(text)

    orgs = []
    people = []
    dates = []
    locations = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            orgs.append(ent.text.strip())
        elif ent.label_ == "PERSON":
            people.append(ent.text.strip())
        elif ent.label_ == "DATE":
            dates.append(ent.text.strip())
        elif ent.label_ in ("GPE", "LOC"):
            locations.append(ent.text.strip())

    money_matches = [m.strip() for m in MONEY_PATTERN.findall(text)]
    durations = [m.strip() for m in DURATION_PATTERN.findall(text)]
    percentages = [m.strip() for m in PERCENT_PATTERN.findall(text)]

    return {
        "organizations": list(dict.fromkeys(orgs)),
        "people": list(dict.fromkeys(people)),
        "dates": list(dict.fromkeys(dates)),
        "locations": list(dict.fromkeys(locations)),
        "monetary_amounts": list(dict.fromkeys(money_matches)),
        "durations": list(dict.fromkeys(durations)),
        "percentages": list(dict.fromkeys(percentages)),
    }


def extract_parties(text: str, max_lines_to_check: int = 40) -> list:
    """
    Cheap heuristic to guess who the contracting parties are - most
    contracts name the parties in the first ~30-40 lines ("This Agreement
    is entered into between X and Y..."), so we don't bother running NER
    on the whole doc for this, just the top chunk. Faster and honestly
    worked better in my testing than running it on everything.
    """
    lines = text.split("\n")[:max_lines_to_check]
    top_chunk = "\n".join(lines)

    doc = nlp(top_chunk)
    candidates = [ent.text.strip() for ent in doc.ents if ent.label_ in ("ORG", "PERSON")]

    # dedupe, keep order
    seen = set()
    result = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            result.append(c)

    return result[:6]  # realistically a contract won't have more than a few parties worth flagging
