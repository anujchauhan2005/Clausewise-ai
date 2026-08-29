# 📜 ClauseWise — Legal Document Intelligence System

**Turns dense legal/financial contracts into plain-English insights, automatically.**

Upload a contract — a loan agreement, employment contract, NDA, lease, whatever — and ClauseWise reads it like a junior associate would: it classifies every clause, pulls out the key facts, flags anything unusual, summarizes the whole thing in plain English, and lets you ask follow-up questions about it.

## 🔗 Live Demo

| | |
|---|---|
| **Frontend (try it here)** | [clausewise-ai-legal9.streamlit.app](https://clausewise-ai-legal9.streamlit.app/) |
| **Backend API docs** | [clausewise-backend-fzmr.onrender.com/docs](https://clausewise-backend-fzmr.onrender.com/docs) |

> ⏳ Both are on free hosting tiers, so the backend spins down after inactivity — the first request after a while can take 30–60 seconds to wake up. Totally normal, just give it a moment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it actually does

1. **🏷️ Classifies every clause** — Termination, Liability, Payment Terms, Confidentiality, and 6 other categories — using zero-shot NLP classification (no training data needed).
2. **🔍 Extracts key entities** — money amounts, dates, durations, and the parties involved — using Named Entity Recognition.
3. **⚠️ Flags risky clauses** — unlimited liability, unilateral termination rights, auto-renewal traps — using a rule-based lexicon weighted by clause category.
4. **📝 Summarizes the whole document** in plain English using abstractive summarization.
5. **💬 Answers questions about the document** — "who are the parties?", "what's the termination notice period?" — using Retrieval-Augmented Generation (RAG).

Built to show applied NLP across a full stack — a real backend/frontend split, a genuine evaluation framework, and an actually-deployed, publicly usable app — not a single Jupyter notebook.

---

## 🏗️ Architecture

```
                    ┌───────────────┐
                    │   Frontend     │  Streamlit UI
                    │ (Streamlit     │  → Streamlit Community Cloud
                    │  Cloud)        │
                    └───────┬────────┘
                            │ HTTPS
                    ┌───────▼────────┐
                    │   FastAPI       │  backend/app/main.py
                    │   Backend       │  → Render (free tier)
                    └──┬─────────┬────┘
        ┌──────────────┘         └──────────────┐
┌───────▼─────────┐                    ┌─────────▼─────────┐
│  NLP Pipeline     │                   │   RAG Q&A          │
│  ──────────────   │                   │  ──────────────    │
│  • NER (spaCy,     │                  │  • chunk + embed    │
│    runs locally)   │                  │    (via HF API)     │
│  • Classification  │──── HF ─────►    │  • FAISS search     │
│  • Summarization    │  Inference      │    (runs locally)   │
│  • Risk Flagger    │   API            │  • Groq LLM answer   │
│    (rule-based,     │                  └─────────────────────┘
│    runs locally)    │
└─────────────────────┘
```

**Why it's split this way:** the backend runs on Render's free tier, which caps out at 512MB RAM. Loading heavy transformer models (bart-large-mnli, distilbart, sentence-transformers) directly into that process was enough to OOM-crash it the moment a real request came in. The fix was to offload every heavy model call — clause classification, summarization, and embeddings — to the **Hugging Face Inference API**, so the actual neural network forward passes happen on HF's infrastructure, not on the 512MB box. The backend itself only ever holds spaCy (lightweight) and FAISS (an in-memory index, not a model) in memory, plus a handful of HTTP client libraries. No `torch` or `transformers` needed locally at all anymore.

---

## 🧠 Techniques & Where They Run

| Module | Technique | Model | Runs on |
|---|---|---|---|
| Entity Extraction | NER + regex for money/dates/duration | spaCy `en_core_web_sm` | Backend (local — small enough to be safe) |
| Clause Classification | Zero-shot text classification | `valhalla/distilbart-mnli-12-3` | Hugging Face Inference API |
| Summarization | Abstractive summarization | `sshleifer/distilbart-cnn-12-6` | Hugging Face Inference API |
| Embeddings (for RAG) | Sentence embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face Inference API |
| Risk Flagging | Rule-based lexicon + category-weighted scoring | Custom | Backend (local — no model needed) |
| Vector Search | Cosine similarity via inner product | FAISS `IndexFlatIP` | Backend (local — just math, no model) |
| Document Q&A | Retrieval-Augmented Generation | `openai/gpt-oss-20b` | Groq API |

---

## 🚀 Tech Stack & Hosting

| Layer | Tech | Hosted on |
|---|---|---|
| Frontend | Streamlit | **Streamlit Community Cloud** (free) |
| Backend API | FastAPI + Uvicorn | **Render** (free web service) |
| NLP inference (classification, summarization, embeddings) | Hugging Face Inference Providers (`router.huggingface.co`) | Hugging Face (free tier, token-based) |
| LLM for Q&A | Groq (`openai/gpt-oss-20b`) | Groq API (free tier) |
| Vector store | FAISS (in-memory, per-document) | Runs inside the Render backend |
| Local NLP | spaCy | Runs inside the Render backend |

**Why this combination:** it's the cheapest possible way to run a real multi-model NLP pipeline without paying for GPU hosting anywhere. Every piece — Render, Streamlit Cloud, the HF Inference API, Groq — has a usable free tier, and none of them require a credit card to get started. The tradeoff is cold-start latency (spinning up after inactivity) and occasional API rate limits, which is a fair trade for a $0/month portfolio deployment.

> **Note on Hugging Face's API:** Hugging Face migrated their serverless inference from `api-inference.huggingface.co` to a new `router.huggingface.co`-based "Inference Providers" system in 2025/2026, with a different URL structure and slightly different response shapes per task. `backend/app/nlp/hf_client.py` is written against the current router API.

---

## 📈 Evaluation Results

Ran against a hand-labeled set of 20 clauses (10 categories) and 3 reference summaries — see `eval/eval_dataset.json` and `eval/evaluate.py`.

**Clause Classification (zero-shot):**

| Metric | Score |
|---|---|
| Accuracy | 0.85 |
| Macro F1 | 0.79 |
| Weighted F1 | 0.83 |

**Summarization:**

| Metric | Score |
|---|---|
| ROUGE-1 | 0.38 |
| ROUGE-2 | 0.14 |
| ROUGE-L | 0.31 |

Classification was strongest on structurally distinctive clauses (Termination, Governing Law, Dispute Resolution — all F1 1.00) and weakest where categories semantically overlap, e.g. **Force Majeure vs. Liability** ("liable for delays" reads as both). That's the expected failure mode of zero-shot classification without domain fine-tuning — see the LegalBERT fine-tuning item in the roadmap below.

Reproduce this yourself:
```bash
cd eval
pip install -r requirements.txt
python evaluate.py
```

---

## 🛠️ Running It Locally

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm

copy .env.example .env
```

Fill in `.env` with:
```
HF_API_TOKEN=your_huggingface_token      # free "Read" token from huggingface.co/settings/tokens
GROQ_API_KEY=your_groq_key               # free tier at console.groq.com
```

```bash
uvicorn app.main:app --reload
```
Backend: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

> No local model downloads needed — classification, summarization, and embeddings all call the Hugging Face Inference API instead of running on your machine. The first call to any given model may take a few seconds while it "wakes up" on HF's side; subsequent calls are fast.

### 2. Frontend

```bash
cd frontend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Frontend: `http://localhost:8501`

By default the frontend points at `http://localhost:8000`. To point it at the deployed backend instead, set a `BACKEND_URL` environment variable (this is exactly how the deployed frontend on Streamlit Cloud is configured, via a secret).

### 3. Try it without any UI

```bash
curl -X POST http://localhost:8000/analyze -F "file=@sample_data/sample_contract.txt"
```

---

## ☁️ How the Deployment Actually Works

**Backend → Render**
- Free "Web Service" plan, deployed straight from the GitHub repo (`backend/` as the root).
- Environment variables set in Render's dashboard: `HF_API_TOKEN`, `GROQ_API_KEY`, `PYTHON_VERSION=3.11.9`.
- Heavy NLP imports inside `routes.py` are lazy (imported inside each endpoint function, not at module load time), so the server binds to its port immediately on startup instead of timing out while loading libraries.

**Frontend → Streamlit Community Cloud**
- Deployed directly from the same GitHub repo, with `frontend/app.py` as the entry point.
- The backend's Render URL is passed in via a `BACKEND_URL` secret in Streamlit Cloud's app settings — the code itself never hardcodes which backend to talk to.

**Why not Hugging Face Spaces for the frontend?** HF Spaces changed its pricing during this project's development — Gradio and Docker Spaces now require a paid plan for personal accounts, with only static (HTML/JS) Spaces free. Since this app needs a live Python backend, Render + Streamlit Community Cloud ended up being the actually-free path.

---

## 📂 Folder Structure

```
legal-doc-intelligence/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── nlp/
│   │   │   ├── hf_client.py        # shared Hugging Face Inference API helper
│   │   │   ├── ner.py              # entity + key-term extraction (local spaCy)
│   │   │   ├── classifier.py       # zero-shot clause classification (via HF API)
│   │   │   ├── summarizer.py       # abstractive summarization (via HF API)
│   │   │   └── risk_flagger.py     # rule-based risk detection (local)
│   │   ├── rag/
│   │   │   ├── ingest.py           # chunk + embed document (via HF API) + FAISS index
│   │   │   ├── retriever.py        # vector similarity search (local FAISS)
│   │   │   └── qa.py               # RAG question-answering (Groq)
│   │   └── api/
│   │       └── routes.py           # /analyze, /ask, /clauses, /risks endpoints
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── eval/
│   ├── eval_dataset.json
│   ├── evaluate.py
│   └── requirements.txt
├── sample_data/
│   └── sample_contract.txt
├── .gitignore
└── README.md
```

---

## 🎯 What Makes This Project Resume-Worthy

- **Real backend/frontend separation** — a FastAPI service any client could call, not a single monolithic script.
- **A genuine evaluation framework** — actual accuracy/F1/ROUGE numbers against a hand-labeled dataset, not just "it works on my machine."
- **RAG with a real vector index (FAISS)** and a fallback mode (extractive answers) when no LLM API key is configured, so the project is demoable without requiring anyone to sign up for anything.
- **Solved a genuine production constraint** — free-tier RAM limits forced a real architectural decision (offload inference to hosted APIs instead of running models locally), which is exactly the kind of tradeoff real deployments have to make.
- **Deployed and publicly usable** — both frontend and backend are live, not just code that "runs on my machine."

---

## 🗺️ Roadmap

- [ ] Fine-tune a dedicated clause classifier (LegalBERT) instead of zero-shot — should directly address the Force Majeure / Liability confusion seen in evaluation
- [ ] Add PDF/DOCX ingestion (currently plain text)
- [ ] Add clause-level diffing between two versions of a contract
- [ ] Persist the FAISS index (currently in-memory, resets on backend restart)
- [ ] Wire `eval/evaluate.py` into CI so classification/summarization regressions get caught automatically

## 👤 Author
**Anuj Chauhan** — [GitHub](https://github.com/anujchauhan2005)
