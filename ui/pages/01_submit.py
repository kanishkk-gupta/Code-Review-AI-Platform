"""
ui/pages/01_submit.py
======================
Streamlit page: Submit a repository for review.

Allows users to:
  1. Enter a GitHub URL, OR
  2. Upload a ZIP file
  3. Configure analysis options (expandable)
  4. Submit and receive a job_id for polling
"""

from __future__ import annotations

import streamlit as st
from ui.utils.api_client import APIClient

st.set_page_config(page_title="Submit Review — CodeGuardian AI", page_icon="📤", layout="wide")
st.title("📤 Submit Repository for Review")

# ── Source Selection ───────────────────────────────────────────────────────────
source_type = st.radio("Repository Source", ["GitHub URL", "Upload ZIP"], horizontal=True)

source_url = None
source_zip_b64 = None

if source_type == "GitHub URL":
    source_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/your-org/your-repo",
    )
else:
    uploaded_file = st.file_uploader("Upload ZIP Archive", type=["zip"])
    if uploaded_file:
        import base64
        source_zip_b64 = base64.b64encode(uploaded_file.read()).decode()

repository_name = st.text_input(
    "Project Name",
    placeholder="my-service",
    help="Human-readable name shown in the review report.",
)

# ── Advanced Config ────────────────────────────────────────────────────────────
with st.expander("⚙️ Advanced Configuration"):
    col1, col2 = st.columns(2)
    with col1:
        max_chunk_lines = st.slider("Max Lines per Chunk", 20, 500, 80)
        llm_temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1, step=0.05)
    with col2:
        top_k = st.slider("FAISS Similarity Top-K", 1, 10, 3)
        st.markdown("**Enabled Analyzers**")
        enable_bugs = st.checkbox("Bug Analysis", value=True)
        enable_solid = st.checkbox("SOLID Analysis", value=True)
        enable_arch = st.checkbox("Architecture Analysis", value=True)
        enable_sec = st.checkbox("Security Analysis", value=True)
        enable_comp = st.checkbox("Complexity Analysis", value=True)

# ── Submit ────────────────────────────────────────────────────────────────────
if st.button("🚀 Start Review", type="primary", disabled=not repository_name):
    if not source_url and not source_zip_b64:
        st.error("Please provide a GitHub URL or upload a ZIP file.")
    else:
        with st.spinner("Submitting repository for review..."):
            client = APIClient()
            # TODO: Phase 2 — call real API
            # response = client.submit_review(...)
            # st.session_state["job_id"] = response.job_id
            # st.success(f"✅ Review submitted! Job ID: `{response.job_id}`")
            # st.info("Navigate to **Job Status** to monitor progress.")
            st.warning("⚠️ API integration coming in Phase 2. Backend is ready at `/review`.")
