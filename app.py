import streamlit as st
import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Medical Research Copilot",
    layout="wide"
)

st.title("🧠 Medical Research Copilot")
st.caption("Evidence-based medical Q&A from uploaded research PDFs")

# -----------------------------
# LOAD MODELS (CACHED)
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        token=os.environ.get("HF_TOKEN"),
        max_length=300
    )

@st.cache_resource
def load_faiss_index():
    if not os.path.exists("medical_faiss.index"):
        st.error("❌ medical_faiss.index not found. Please build FAISS index first.")
        st.stop()
    return faiss.read_index("medical_faiss.index")

@st.cache_resource
def load_chunks():
    if not os.path.exists("chunked_docs.pkl"):
        st.error("❌ chunked_docs.pkl not found. Please save chunked docs first.")
        st.stop()
    with open("chunked_docs.pkl", "rb") as f:
        return pickle.load(f)

embedding_model = load_embedding_model()
llm = load_llm()
index = load_faiss_index()
chunked_docs = load_chunks()

# -----------------------------
# CHAT INPUT
# -----------------------------
question = st.text_input(
    "💬 Ask a medical research question:",
    placeholder="Example: What are the symptoms and causes of diabetes?"
)

# -----------------------------
# ASK BUTTON
# -----------------------------
if st.button("Ask Copilot") and question.strip():

    with st.spinner("🔍 Searching medical knowledge..."):

        # Vector search
        q_embedding = embedding_model.encode([question])
        distances, indices = index.search(np.array(q_embedding), 5)

        context = ""
        sources = []

        for idx in indices[0]:
            chunk = chunked_docs[idx]
            context += chunk["text"] + "\n"
            sources.append(
                f'{chunk["metadata"]["source"]} (page {chunk["metadata"]["page"]})'
            )

        prompt = f"""
You are a medical research assistant.
Answer the question using ONLY the context below.
Use simple English.
Do not add new information.

Question:
{question}

Context:
{context}

Answer:
"""

        response = llm(prompt)[0]["generated_text"]

    # -----------------------------
    # DISPLAY OUTPUT
    # -----------------------------
    st.subheader("✅ Medical Answer")
    st.write(response)

    st.subheader("📚 Sources")
    for s in sorted(set(sources)):
        st.write("-", s)
