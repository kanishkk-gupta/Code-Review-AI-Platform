"""
ui/app.py
==========
Streamlit application entry point.

Run with:
    streamlit run ui/app.py --server.port 8501

Pages are auto-discovered from ui/pages/ by Streamlit.
This file sets global configuration and renders the home/landing page.
"""

from __future__ import annotations

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeGuardian AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=CodeGuardian+AI", use_column_width=True)
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("- 📤 **Submit Review** — Upload your repository")
    st.markdown("- 📊 **Job Status** — Check analysis progress")
    st.markdown("- 📋 **Report** — View detailed findings")
    st.markdown("---")
    st.caption("CodeGuardian AI v1.0.0")

# ── Home Page ─────────────────────────────────────────────────────────────────
st.title("🛡️ CodeGuardian AI")
st.subheader("AI-powered code review — bugs, security, architecture, and more.")

st.markdown("""
Welcome to **CodeGuardian AI**. Submit your repository to receive a comprehensive
AI-driven code review across five dimensions:

| Category | What We Detect |
|----------|---------------|
| 🐛 **Bugs** | Null dereferences, race conditions, logic errors |
| 🏗️ **SOLID** | Principle violations across your codebase |
| 🏛️ **Architecture** | Coupling, God Classes, layer violations |
| 🔒 **Security** | OWASP Top 10, CWE vulnerabilities, hardcoded secrets |
| 📊 **Complexity** | Cyclomatic complexity and cognitive load hotspots |

👈 **Use the sidebar to navigate to Submit Review to get started.**
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Analysis Dimensions", "5")
with col2:
    st.metric("Supported Languages", "11")
with col3:
    st.metric("Severity Levels", "5")
