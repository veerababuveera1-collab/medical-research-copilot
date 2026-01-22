# ==========================================================
# MEDINTEL AI — Medical Research Copilot (Fresh Start)
# Author: Veera Babu
# ==========================================================

import streamlit as st
import os
import requests
from pypdf import PdfReader

# ==========================================================
# APP CONFIG
# ==========================================================

st.set_page_config(page_title="MEDINTEL AI", layout="wide")
st.title("🧬 MEDINTEL AI — Medical Research Copilot")
st.caption("Hospital | Pharma | Research Medical Intelligence System")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================================
# PDF READER
# ==========================================================

def read_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        return ""

# ==========================================================
# PUBMED SEARCH (SAFE MODE)
# ==========================================================

def search_pubmed(topic):
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": topic,
            "retmax": 10
        }
        response = requests.get(url, params=params, timeout=10)
        return response.text
    except:
        return None

# ==========================================================
# UI TABS
# ==========================================================

tab1, tab2 = st.tabs(["📄 Upload Medical Research PDF", "🌍 Search PubMed Research"])

# ==========================================================
# TAB 1 — PDF UPLOAD
# ==========================================================

with tab1:
    st.header("📄 Upload Medical Research PDF")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        text = read_pdf(uploaded_file)

        if not text.strip():
            st.error("❌ Could not extract text. This PDF may be scanned image.")
        else:
            st.success("✅ PDF processed successfully")
            st.text_area("Extracted Medical Research Text", text[:8000], height=400)

# ==========================================================
# TAB 2 — PUBMED SEARCH
# ==========================================================

with tab2:
    st.header("🌍 Search Medical Research (PubMed)")

    topic = st.text_input("Enter medical topic (e.g. diabetes, cancer, heart disease)")

    if st.button("Search PubMed"):
        if not topic.strip():
            st.warning("Please enter a medical topic")
        else:
            data = search_pubmed(topic)

            if data:
                st.success("✅ PubMed data fetched successfully")
                st.text_area("PubMed Search XML Response", data[:8000], height=400)
            else:
                st.error("❌ Failed to fetch PubMed data")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.caption("MEDINTEL AI © 2026 | Medical Research Intelligence Platform | India")
