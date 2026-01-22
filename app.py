# ============================================================
# MEDINTEL AI OS — Global Medical Intelligence Platform
# End-to-End Working Production Build
# Author: Veera Babu
# ============================================================

import streamlit as st
import os, json, datetime, requests
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from xml.etree import ElementTree as ET

# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "🧬 MEDINTEL AI OS — Medical Research Copilot"
DATA_DIR = "database"
UPLOAD_DIR = "uploads"

INDEX_FILE = f"{DATA_DIR}/clinical_index.faiss"
META_FILE = f"{DATA_DIR}/clinical_meta.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# UI SETUP
# ============================================================

st.set_page_config(APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Hospital | Pharma | Research | Government Medical Intelligence System")

# ============================================================
# AI MODE
# ============================================================

st.sidebar.title("⚙ AI Mode")
AI_MODE = st.sidebar.radio("Select Mode", [
    "🏥 Hospital AI Mode",
    "🌍 Global AI Mode",
    "⚡ Hybrid AI Mode"
])

# ============================================================
# LOAD AI MODEL
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
EMBEDDING_DIM = 384

# ============================================================
# LOAD VECTOR DB
# ============================================================

def load_db():
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            metadata = json.load(open(META_FILE))
        except:
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            metadata = []
    else:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = []
    return index, metadata

index, metadata = load_db()

# ============================================================
# CORE FUNCTIONS
# ============================================================

def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text
    except:
        return ""

def chunk_text(text, size=350):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def index_document(text, source):
    global index, metadata
    if not text.strip():
        return False

    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    index.add(np.array(embeddings).astype("float32"))
    for c in chunks:
        metadata.append({"text": c, "source": source})

    faiss.write_index(index, INDEX_FILE)
    json.dump(metadata, open(META_FILE, "w"), indent=2)
    return True

def search(query, k=5):
    if not metadata:
        return []
    q = model.encode([query]).astype("float32")
    _, idx = index.search(q, k)
    return [metadata[i] for i in idx[0] if i < len(metadata)]

# ============================================================
# PUBMED API
# ============================================================

def fetch_pubmed(query):
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": 3}
        res = requests.get(url, params=params, timeout=15)

        root = ET.fromstring(res.text)
        ids = [i.text for i in root.findall(".//Id")]

        articles = []
        for pmid in ids:
            f_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f_params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
            r = requests.get(f_url, params=f_params, timeout=15)
            if r.status_code == 200:
                articles.append(r.text)

        return articles
    except:
        return []

# ============================================================
# CLINICALTRIALS API
# ============================================================

def fetch_clinical_trials(query):
    try:
        api = "https://clinicaltrials.gov/api/query/study_fields"
        params = {
            "expr": query,
            "fields": "NCTId,BriefTitle,Condition,Phase,BriefSummary",
            "min_rnk": 1,
            "max_rnk": 5,
            "fmt": "json"
        }
        res = requests.get(api, params=params, timeout=15)
        if res.status_code == 200:
            return json.dumps(res.json())
    except:
        return None

# ============================================================
# FDA API
# ============================================================

def fetch_fda(query):
    try:
        api = "https://api.fda.gov/drug/label.json"
        params = {"search": query, "limit": 3}
        res = requests.get(api, params=params, timeout=15)
        if res.status_code == 200:
            return json.dumps(res.json())
    except:
        return None

# ============================================================
# AI ROUTER
# ============================================================

def ai_router(query, evidence):
    mode = "Hospital AI" if AI_MODE.startswith("🏥") else "Global AI" if AI_MODE.startswith("🌍") else "Hybrid AI"
    return f"""
{mode} Clinical Intelligence

Query:
{query}

Evidence:
{evidence}
"""

# ============================================================
# UI TABS
# ============================================================

tabs = st.tabs([
    "📄 Upload PDF",
    "🌍 PubMed",
    "🌍 ClinicalTrials",
    "🌍 FDA Drugs",
    "🧠 Clinical Copilot"
])

# ============================================================
# TAB 1 — PDF
# ============================================================

with tabs[0]:
    st.subheader("Upload Medical Research PDF")
    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file:
        path = f"{UPLOAD_DIR}/{file.name}"
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        text = read_pdf(path)
        if index_document(text, file.name):
            st.success("PDF Indexed Successfully")
        else:
            st.error("Unable to extract text from PDF")

# ============================================================
# TAB 2 — PUBMED
# ============================================================

with tabs[1]:
    q = st.text_input("Search PubMed Topic")
    if st.button("Fetch PubMed Research"):
        data = fetch_pubmed(q)
        if data:
            for i, d in enumerate(data):
                index_document(d, f"PubMed_{q}_{i}")
            st.success("PubMed Research Indexed")
        else:
            st.error("PubMed fetch failed")

# ============================================================
# TAB 3 — CLINICALTRIALS
# ============================================================

with tabs[2]:
    q = st.text_input("Search Clinical Trials")
    if st.button("Fetch Clinical Trials"):
        data = fetch_clinical_trials(q)
        if data:
            index_document(data, f"ClinicalTrials_{q}")
            st.success("ClinicalTrials Data Indexed")
        else:
            st.error("ClinicalTrials fetch failed")

# ============================================================
# TAB 4 — FDA
# ============================================================

with tabs[3]:
    q = st.text_input("Search FDA Drug")
    if st.button("Fetch FDA Drug Data"):
        data = fetch_fda(q)
        if data:
            index_document(data, f"FDA_{q}")
            st.success("FDA Drug Data Indexed")
        else:
            st.error("FDA fetch failed")

# ============================================================
# TAB 5 — COPILOT
# ============================================================

with tabs[4]:
    q = st.text_input("Ask Clinical Question")
    if st.button("Ask MEDINTEL AI"):
        results = search(q)
        if not results:
            st.warning("No research indexed yet")
        else:
            evidence = "\n".join([r["text"][:300] for r in results])
            st.text_area("AI Response", ai_router(q, evidence), height=300)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("MEDINTEL AI OS © 2026 | End-to-End Production Build | India")
