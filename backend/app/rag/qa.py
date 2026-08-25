"""
qa.py

Ties retriever.py output together with an LLM call to actually answer
the question. Using Groq because it's free-tier friendly and fast
enough that streaming isn't strictly necessary for a document-length
context.

Made this fall back to a plain extractive answer (just return the top
chunk) if no API key is set, so the project still works and is
demoable even without anyone having to sign up for anything. Figured
that's important for something going on a resume/portfolio - a
recruiter isn't going to go get an API key just to try it.
"""

import os
from groq import Groq
from app.rag.retriever import retrieve_relevant_chunks

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = (
    "You are a legal document assistant. Answer the user's question using "
    "ONLY the contract excerpts provided below. If the excerpts don't "
    "contain enough information to answer confidently, say so clearly - "
    "do not guess or make up clause details. Keep answers concise and in "
    "plain English, not legal jargon."
)


def _build_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Excerpt {i}]\n{c['text']}")
    return "\n\n".join(parts)


def _generate_with_groq(question: str, context: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        # NOTE (updated): Groq deprecated the llama-3.1/3.3 chat models this
        # was originally built with. Switched to their current recommended
        # general-purpose model. If this ever breaks again, check
        # https://console.groq.com/docs/models for what's currently live -
        # Groq rotates/deprecates models more often than I'd like.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contract excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,  # kept this low, don't want creative answers about legal terms
        max_tokens=600,
    )

    return response.choices[0].message.content


def answer_question(doc_id: str, question: str, top_k: int = 4) -> dict:
    chunks = retrieve_relevant_chunks(doc_id, question, top_k=top_k)

    if not chunks:
        return {
            "answer": "I couldn't find this document. Try re-uploading it.",
            "sources": [],
            "mode": "error",
        }

    context = _build_context(chunks)

    if GROQ_API_KEY:
        try:
            answer = _generate_with_groq(question, context)
            mode = "llm"
        except Exception as e:
            # if the API call fails for whatever reason (rate limit, bad
            # key, network blip) don't just 500 the whole request - fall
            # back to extractive so the user still gets something useful.
            # printing the real error to the console since the user-facing
            # message deliberately doesn't expose internals
            print(f"[qa.py] Groq call failed: {type(e).__name__}: {e}")
            answer = f"(LLM generation failed, showing most relevant excerpt instead)\n\n{chunks[0]['text']}"
            mode = "extractive_fallback"
    else:
        # no key configured - just hand back the best matching chunk(s)
        answer = chunks[0]["text"]
        mode = "extractive"

    return {
        "answer": answer,
        "sources": [c["text"][:200] + "..." for c in chunks],
        "mode": mode,
    }
