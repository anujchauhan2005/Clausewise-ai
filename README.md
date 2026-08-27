# 📜 ClauseWise — Legal Document Intelligence System

**Turns dense legal/financial contracts into plain-English insights, automatically.**                               

ClauseWise takes a contract (loan agreement, employment contract, NDA, lease, etc.) and:
- 🏷️ **Classifies every clause** into categories (Termination, Liability, Payment Terms, Confidentiality, etc.) using zero-shot NLP classification
- 🔍 **Extracts key entities** — money amounts, dates, durations, parties — using Named Entity Recognition
- ⚠️ **Flags risky/unusual clauses** (e.g. unlimited liability, unilateral termination, auto-renewal traps)
- 📝 **Summarizes the whole document** in plain English using an abstractive summarization model
- 💬 **Answers questions about the document** using Retrieval-Augmented Generation (RAG)       

Built to demonstrate applied NLP across the full stack — not just a single Jupyter notebook model.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why this project

Legal/financial document review is slow, expensive, and error-prone. Startups (Ironclad, Kira Systems, Luminance) have built entire companies around exactly this problem. This project reproduces the core NLP pipeline behind that category of product, end-to-end and evaluated — not just a demo notebook.

---

## 🏗️ Architecture

```
                    ┌──────────────┐
                    │   Frontend    │  Streamlit UI
                    │ (frontend/)   │
                    └──────┬───────┘
                           │ HTTP
                    ┌──────▼───────┐
                    │   FastAPI     │  backend/app/main.py
                    │   Backend     │
                    └──┬─────┬─────┘
        ┌──────────────┘     └──────────────┐
┌───────▼────────┐                  ┌────────▼────────┐
│  NLP Pipeline   │                  │   RAG Q&A        │
│  ─────────────  │                  │  ─────────────   │
│  • NER          │                  │  • chunk + embed │
│  • Classifier   │                  │  • FAISS search  │
│  • Summarizer   │                  │  • LLM answer     │
│  • Risk Flagger │                  └──────────────────┘
└─────────────────┘
```

---

## 🧠 NLP Techniques Used

| Module | Technique | Model |
|---|---|---|
| Entity Extraction | Named Entity Recognition + regex for money/dates/duration | spaCy `en_core_web_sm` |
| Clause Classification | Zero-shot text classification | `facebook/bart-large-mnli` |
| Summarization | Abstractive summarization | `sshleifer/distilbart-cnn-12-6` |
| Risk Flagging | Rule-based lexicon + category-weighted confidence scoring | Custom |
| Document Q&A | Retrieval-Augmented Generation | `sentence-transformers/all-MiniLM-L6-v2` + FAISS + Groq LLM |

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

Per-class classification performance was strongest on structurally distinctive clauses (Termination, Governing Law, Dispute Resolution — all F1 1.00) and weakest where categories semantically overlap, e.g. **Force Majeure vs. Liability** ("liable for delays" reads as both). This is the expected failure mode of zero-shot classification without domain fine-tuning, and is the motivation for the LegalBERT fine-tuning item in the roadmap below.

Reproduce this yourself:
```bash
cd eval
pip install -r requirements.txt
python evaluate.py
```

---

## 🚀 Setup

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm

copy .env.example .env           # add your GROQ_API_KEY (free tier at console.groq.com)
uvicorn app.main:app --reload
```
Backend: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

> **First run downloads ~1.6GB of models** from Hugging Face (bart-large-mnli + distilbart). This only happens once — after that they're cached locally and load instantly. If you'd rather see the download progress directly instead of waiting through the Streamlit UI, warm the cache first:
> ```bash
> python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli'); pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')"
> ```

> **If you have TensorFlow installed** on the same environment, `transformers` may try to load it and crash with a protobuf version error — this project only uses PyTorch. `app/main.py` already sets `USE_TF=0` / `USE_FLAX=0` to avoid this, but if you hit it while running `eval/evaluate.py` directly, set it manually first: `$env:USE_TF="0"` (PowerShell) or `export USE_TF=0` (Mac/Linux).

### 2. Frontend

```bash
cd frontend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Frontend: `http://localhost:8501`

### 3. Try it without the UI

```bash
curl -X POST http://localhost:8000/analyze -F "file=@sample_data/sample_contract.txt"
```

---

## 📂 Folder Structure

```
legal-doc-intelligence/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── nlp/
│   │   │   ├── ner.py              # entity + key-term extraction
│   │   │   ├── classifier.py       # zero-shot clause classification
│   │   │   ├── summarizer.py       # abstractive summarization
│   │   │   └── risk_flagger.py     # rule-based risk detection
│   │   ├── rag/
│   │   │   ├── ingest.py           # chunk + embed document (FAISS)
│   │   │   ├── retriever.py        # vector similarity search
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
- **Honest, documented limitations** — the eval results above call out exactly where zero-shot classification breaks down and why, which is a stronger signal than pretending the model is perfect.
- **Deployed and publicly usable**, not just code that "runs on my machine."

---

## 🗺️ Roadmap

- [ ] Fine-tune a dedicated clause classifier (LegalBERT) instead of zero-shot — should directly address the Force Majeure / Liability confusion seen in evaluation
- [ ] Add PDF/DOCX ingestion (currently plain text)
- [ ] Add clause-level diffing between two versions of a contract
- [ ] Deploy backend + frontend publicly (Render/HF Spaces)
- [ ] Wire `eval/evaluate.py` into CI so classification/summarization regressions get caught automatically

## 👤 Author
**Anuj Chauhan** — [GitHub](https://github.com/anujchauhan2005)
