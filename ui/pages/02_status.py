"""
ui/pages/02_status.py
======================
Streamlit page: Poll and display review job status with progress.
"""

from __future__ import annotations

import time

import streamlit as st

st.set_page_config(page_title="Job Status — CodeGuardian AI", page_icon="📊", layout="wide")
st.title("📊 Review Job Status")

job_id = st.text_input(
    "Job ID",
    value=st.session_state.get("job_id", ""),
    placeholder="3fa85f64-5717-4562-b3fc-2c963f66afa6",
)

if st.button("🔄 Check Status", type="primary") and job_id:
    with st.spinner("Fetching status..."):
        # TODO: Phase 2 — call real API
        # from ui.utils.api_client import APIClient
        # client = APIClient()
        # status = client.get_status(job_id)
        # display status...

        # Placeholder display
        st.info("📡 Status polling will be active in Phase 2.")
        st.json({
            "job_id": job_id,
            "status": "PENDING",
            "progress": 0,
            "result": None,
            "error": None,
        })

# Progress bar placeholder
st.markdown("---")
st.markdown("### Progress")
progress_bar = st.progress(0)
status_text = st.empty()
status_text.text("Submit a job to see progress here.")

# Auto-refresh hint
st.caption("💡 Tip: Leave this page open — it will auto-refresh when the review completes.")
