"""
routes.py

All the endpoints live here. Kept it flat instead of splitting into
multiple router files since the app is small enough right now that
having 4-5 tiny router files would just be extra navigation for no
real benefit. Might split this up later (analyze.py, qa.py etc) if it
keeps growing.
"""

import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.nlp import ner, classifier, summarizer, risk_flagger
from app.rag import ingest, qa

router = APIRouter()

# capping upload size for the demo - this isn't a scalability limitation of
# the NLP techniques themselves, it's a practical limit for a CPU-only,
# single-process demo app. spaCy also has its own internal max_length limit
# that a multi-MB document would blow past anyway. 300KB is roughly a
# 50-60 page contract, which comfortably covers anything anyone would
# realistically upload to try this out.
MAX_DOCUMENT_CHARS = 300_000

# quick and dirty in-memory store mapping doc_id -> raw text, so we don't
# have to re-read the uploaded file every time someone hits a different
# endpoint. would move this to redis or a real db if this ever needed to
# survive a server restart or handle multiple concurrent users properly
_document_store = {}


def _read_uploaded_file(file: UploadFile) -> str:
    raw = file.file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # some files come in with weird encodings, this is a reasonable
        # fallback that won't crash the request
        return raw.decode("latin-1")


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    text = _read_uploaded_file(file)

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Document seems too short / empty, check the file")

    if len(text) > MAX_DOCUMENT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Document is too large for this demo ({len(text):,} chars, limit is {MAX_DOCUMENT_CHARS:,}). "
                   f"Try a shorter excerpt - the pipeline processes clause-by-clause on CPU, so very large "
                   f"documents would take an impractically long time.",
        )

    doc_id = str(uuid.uuid4())[:8]
    _document_store[doc_id] = text

    chunk_count = ingest.ingest_document(text, doc_id)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "char_count": len(text),
        "chunks_indexed": chunk_count,
    }


@router.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    One-shot endpoint that runs the whole pipeline (entities, clauses,
    risk, summary) in a single call - convenient for the demo/frontend
    so it's not making 4 separate round trips. Also ingests for Q&A
    while it's at it.
    """
    text = _read_uploaded_file(file)

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Document seems too short / empty, check the file")

    if len(text) > MAX_DOCUMENT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Document is too large for this demo ({len(text):,} chars, limit is {MAX_DOCUMENT_CHARS:,}). "
                   f"Try a shorter excerpt - the pipeline processes clause-by-clause on CPU, so very large "
                   f"documents would take an impractically long time.",
        )

    doc_id = str(uuid.uuid4())[:8]
    _document_store[doc_id] = text
    ingest.ingest_document(text, doc_id)

    entities = ner.extract_entities(text)
    parties = ner.extract_parties(text)
    classification_result = classifier.classify_document(text)
    clauses = classification_result["clauses"]
    risks = risk_flagger.flag_document_risks(clauses)
    summary = summarizer.summarize_document(text)

    return {
        "doc_id": doc_id,
        "summary": summary,
        "parties": parties,
        "entities": entities,
        "clauses": clauses,
        "risks": risks,
        "total_clauses_found": classification_result["total_clauses_found"],
        "clauses_processed": classification_result["clauses_processed"],
        "truncated": classification_result["truncated"],
    }


@router.get("/entities/{doc_id}")
async def get_entities(doc_id: str):
    text = _document_store.get(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="doc_id not found - did you upload first?")
    return ner.extract_entities(text)


@router.get("/clauses/{doc_id}")
async def get_clauses(doc_id: str):
    text = _document_store.get(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="doc_id not found - did you upload first?")
    return classifier.classify_document(text)["clauses"]


@router.get("/risks/{doc_id}")
async def get_risks(doc_id: str):
    text = _document_store.get(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="doc_id not found - did you upload first?")
    clauses = classifier.classify_document(text)["clauses"]
    return risk_flagger.flag_document_risks(clauses)


@router.get("/summary/{doc_id}")
async def get_summary(doc_id: str):
    text = _document_store.get(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="doc_id not found - did you upload first?")
    return {"summary": summarizer.summarize_document(text)}


@router.post("/ask")
async def ask_question(doc_id: str = Form(...), question: str = Form(...)):
    if doc_id not in _document_store:
        raise HTTPException(status_code=404, detail="doc_id not found - did you upload first?")

    result = qa.answer_question(doc_id, question)
    return result
