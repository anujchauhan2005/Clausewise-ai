"""
app.py (frontend)

Streamlit UI for ClauseWise. Went with Streamlit over building a proper
React frontend mainly because I wanted to spend my time on the NLP
pipeline itself, not fighting with frontend build tooling. Might swap
this out for React later if I have time (noted in the README roadmap).

Note: this hits a live backend URL, not localhost, when deployed - see
the BACKEND_URL logic below. Kept both paths working so this same file
runs fine locally and on Streamlit Cloud without editing anything.
"""

import os
import requests
import streamlit as st

# when running locally this defaults to localhost, but when deployed on
# Streamlit Cloud I set BACKEND_URL as a secret pointing at wherever the
# FastAPI backend is hosted. saved me from hardcoding two different URLs
# and forgetting to swap them before pushing
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="ClauseWise", page_icon="📜", layout="wide")

st.title("📜 ClauseWise")
st.caption("Upload a contract and get plain-English summaries, clause categorization, risk flags, and Q&A.")

# using session_state so the analysis result sticks around across reruns -
# without this, asking a question in the Q&A box would wipe the uploaded
# doc's results because streamlit reruns the whole script on every interaction
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

uploaded_file = st.file_uploader("Upload a contract (.txt)", type=["txt"])

col1, col2 = st.columns([1, 4])
with col1:
    analyze_clicked = st.button("Analyze Document", type="primary", disabled=uploaded_file is None)

if analyze_clicked and uploaded_file is not None:
    with st.spinner("Running NLP pipeline... first run downloads ~1.6GB of models from Hugging Face and can take several minutes depending on your connection. After that, it's fast (~15-30s)."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            resp = requests.post(f"{BACKEND_URL}/analyze", files=files, timeout=900)
            resp.raise_for_status()
            st.session_state.analysis = resp.json()
            st.session_state.doc_id = st.session_state.analysis["doc_id"]
        except requests.exceptions.HTTPError as e:
            # 400s here are usually the doc-too-short/too-large validation
            # in the backend, which has a clear message worth showing
            # directly rather than the generic connection-failure text
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            st.error(detail)
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't reach the backend - is it running? ({e})")

# --- results ---
if st.session_state.analysis:
    data = st.session_state.analysis

    st.divider()
    st.subheader("📝 Summary")
    st.write(data["summary"])

    if data.get("parties"):
        st.subheader("👥 Parties Identified")
        st.write(", ".join(data["parties"]))

    tab_clauses, tab_risks, tab_entities, tab_qa = st.tabs(
        ["Clause Breakdown", "⚠️ Risk Flags", "🔍 Entities", "💬 Ask a Question"]
    )

    with tab_clauses:
        if data.get("truncated"):
            st.warning(
                f"This document had {data['total_clauses_found']} clauses - only the first "
                f"{data['clauses_processed']} were classified for this demo, to keep processing "
                f"time reasonable on CPU. See the README for notes on scaling this further."
            )
        st.caption(f"{len(data['clauses'])} clauses detected and categorized")
        for clause in data["clauses"]:
            with st.expander(f"[{clause['category']}]  (confidence: {clause['confidence']})"):
                st.write(clause["text"])

    with tab_risks:
        risks = data.get("risks", [])
        if not risks:
            st.success("No obviously risky phrasing detected. Doesn't mean the contract is fine - still get it reviewed by a human.")
        else:
            st.caption(f"{len(risks)} clause(s) flagged - sorted by risk score, highest first")
            for r in risks:
                level = r["risk_level"]
                # just some color coding so high risk stuff actually stands out
                if level == "high":
                    st.error(f"**[{level.upper()}]** Score: {r['risk_score']}/10 — {r['category']}")
                elif level == "medium":
                    st.warning(f"**[{level.upper()}]** Score: {r['risk_score']}/10 — {r['category']}")
                else:
                    st.info(f"**[{level.upper()}]** Score: {r['risk_score']}/10 — {r['category']}")
                st.caption(f"Flagged phrases: {', '.join(r['flagged_phrases'])}")
                st.write(r["text"])
                st.markdown("---")

    with tab_entities:
        entities = data.get("entities", {})
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Organizations**")
            st.write(entities.get("organizations") or "None found")
            st.markdown("**Monetary Amounts**")
            st.write(entities.get("monetary_amounts") or "None found")
            st.markdown("**Durations**")
            st.write(entities.get("durations") or "None found")
        with e2:
            st.markdown("**Dates**")
            st.write(entities.get("dates") or "None found")
            st.markdown("**Locations**")
            st.write(entities.get("locations") or "None found")
            st.markdown("**Percentages**")
            st.write(entities.get("percentages") or "None found")

    with tab_qa:
        question = st.text_input("Ask something about this document (e.g. 'What happens if I terminate early?')")
        if st.button("Ask") and question:
            with st.spinner("Searching document..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/ask",
                        data={"doc_id": st.session_state.doc_id, "question": question},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    answer_data = resp.json()
                    st.markdown("**Answer:**")
                    st.write(answer_data["answer"])
                    if answer_data.get("mode") == "extractive":
                        st.caption("ℹ️ No LLM API key configured - showing the most relevant excerpt directly instead of a generated answer.")
                    with st.expander("Sources used"):
                        for s in answer_data.get("sources", []):
                            st.caption(s)
                except requests.exceptions.RequestException as e:
                    st.error(f"Something went wrong asking that: {e}")

else:
    st.info("Upload a .txt contract above and click 'Analyze Document' to get started. There's a sample contract in sample_data/ if you want to try it without your own file.")
