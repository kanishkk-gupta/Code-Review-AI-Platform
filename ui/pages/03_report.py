"""
ui/pages/03_report.py
======================
Streamlit page: Display the complete ReviewResult report.

Features:
  - Overall score gauge
  - Finding counts by category
  - Severity-filtered finding cards
  - Download as JSON
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Review Report — CodeGuardian AI", page_icon="📋", layout="wide")
st.title("📋 Code Review Report")

job_id = st.text_input(
    "Job ID",
    value=st.session_state.get("job_id", ""),
    placeholder="3fa85f64-5717-4562-b3fc-2c963f66afa6",
)

col_fetch, col_download = st.columns([2, 1])
with col_fetch:
    fetch_btn = st.button("📥 Load Report", type="primary", disabled=not job_id)
with col_download:
    st.button("⬇️ Download JSON", disabled=True, help="Available after report loads.")

if fetch_btn and job_id:
    with st.spinner("Loading report..."):
        # TODO: Phase 2 — fetch real report from API
        st.info("📋 Report rendering will be active in Phase 2.")

        # Placeholder report layout
        st.markdown("---")
        st.markdown("### 📈 Overall Quality Score")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Overall Score", "—/100")
        with col2:
            st.metric("🐛 Bugs", "—")
        with col3:
            st.metric("🏗️ SOLID", "—")
        with col4:
            st.metric("🔒 Security", "—")
        with col5:
            st.metric("📊 Complexity", "—")

        st.markdown("---")
        st.markdown("### 🔴 Critical Findings")
        st.info("No report loaded yet.")
