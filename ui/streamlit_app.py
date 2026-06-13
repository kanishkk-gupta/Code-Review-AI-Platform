"""
ui/streamlit_app.py
====================
CodeGuardian AI — Streamlit frontend.

Run with:
    streamlit run ui/streamlit_app.py

Features:
- GitHub URL input + Analyze button
- 202 immediate submit → async polling every 2s
- Live progress bar with stage labels
- Score gauge + per-category breakdown
- Expandable finding cards per agent
- Markdown + PDF download buttons
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE   = "http://localhost:8000"
API_KEY    = "your-api-key-here"          # override via sidebar
POLL_SECS  = 2
MAX_POLLS  = 300   # FIX (BUG 9): abort after 300 polls (~10 min) to prevent infinite loop
HEADERS    = lambda key: {"X-API-Key": key, "Content-Type": "application/json"}

SEVERITY_COLOR = {
    "CRITICAL": "#ff2d55",
    "HIGH":     "#ff6b35",
    "MEDIUM":   "#ffcc00",
    "LOW":      "#34aadc",
    "INFO":     "#8e8e93",
}
SEVERITY_ICON = {
    "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CodeGuardian AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2d 50%, #0a1628 100%); }

/* Score circle */
.score-circle {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    width: 160px; height: 160px; border-radius: 50%;
    background: conic-gradient(var(--score-color) var(--score-deg), #1e1e3a var(--score-deg));
    margin: 0 auto; position: relative;
}
.score-inner {
    position: absolute; width: 120px; height: 120px; border-radius: 50%;
    background: #0f0f2d; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.score-number { font-size: 2rem; font-weight: 700; color: white; line-height: 1; }
.score-label  { font-size: 0.7rem; color: #8888aa; letter-spacing: 1px; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1.2rem; text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: white; }
.metric-card .lbl { font-size: 0.8rem; color: #8888aa; margin-top: 4px; }

/* Finding card */
.finding-card {
    background: rgba(255,255,255,0.04); border-left: 4px solid var(--sev-color);
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
}
.finding-title { font-weight: 600; color: white; font-size: 0.95rem; }
.finding-meta  { font-size: 0.78rem; color: #8888aa; margin-top: 4px; }
.finding-desc  { font-size: 0.85rem; color: #ccccee; margin-top: 8px; line-height: 1.5; }

/* Progress stages */
.stage-bar {
    display: flex; justify-content: space-between; margin-bottom: 0.5rem;
}
.stage { font-size: 0.75rem; color: #8888aa; }
.stage.active { color: #7c6dff; font-weight: 600; }
.stage.done   { color: #30d158; }

/* Header */
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #7c6dff, #00c2ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hero-sub { font-size: 1.1rem; color: #8888aa; margin-top: 0.5rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #080818 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🛡️ CodeGuardian AI")
    st.caption("AI-powered code review platform")
    st.markdown("---")

    api_key  = st.text_input("API Key", value=API_KEY, type="password", key="api_key_input")
    api_base = st.text_input("Backend URL", value=API_BASE, key="api_base_input")

    st.markdown("---")

    # Health check
    if st.button("Check Backend Health", use_container_width=True):
        try:
            r = requests.get(f"{api_base}/health", timeout=5)
            d = r.json()
            if d.get("status") == "ok":
                st.success(f"Backend healthy — v{d.get('version','?')}")
            else:
                st.warning(f"Degraded — model loading... v{d.get('version','?')}")
        except Exception as e:
            st.error(f"Cannot reach backend: {e}")

    st.markdown("---")
    st.markdown("**Analysis Dimensions**")
    for cat in ["🐛 Bugs (30%)", "🏗️ SOLID (25%)", "🏛️ Architecture (20%)",
                "🔒 Security (15%)", "📊 Complexity (10%)"]:
        st.caption(cat)

    st.markdown("---")
    st.caption("CodeGuardian AI v1.0.0")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "job_id":      None,
        "status":      None,
        "progress":    0,
        "result":      None,
        "error":       None,
        "md_report":   None,
        "polling":     False,
        "poll_count":  0,           # FIX (BUG 9): track poll count to abort hung jobs
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _submit(url: str, repo_name: str, key: str, base: str) -> Optional[str]:
    """POST /review. Returns job_id on success, None on error."""
    try:
        r = requests.post(
            f"{base}/review",
            json={"source_url": url, "repository_name": repo_name},
            headers=HEADERS(key),
            timeout=15,
        )
        if r.status_code == 202:
            return r.json()["job_id"]
        st.error(f"Submit failed ({r.status_code}): {r.text[:300]}")
    except Exception as e:
        st.error(f"Network error: {e}")
    return None


def _poll(job_id: str, key: str, base: str) -> dict:
    """GET /status/{job_id}. Returns status dict."""
    try:
        r = requests.get(f"{base}/status/{job_id}", headers=HEADERS(key), timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def _get_report_md(job_id: str, key: str, base: str) -> Optional[str]:
    """GET /report/{job_id} and return summary_markdown from ReviewResult."""
    try:
        r = requests.get(f"{base}/report/{job_id}", headers=HEADERS(key), timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Stage labels mapped to progress ranges
# ---------------------------------------------------------------------------

STAGES = [
    (0,  10,  "Cloning repository"),
    (10, 25,  "Chunking & embedding"),
    (25, 80,  "Running analysis agents"),
    (80, 100, "Generating report"),
]

def _stage_label(progress: int) -> str:
    for lo, hi, label in STAGES:
        if lo <= progress < hi:
            return label
    return "Complete" if progress >= 100 else "Starting..."

# ---------------------------------------------------------------------------
# Score rendering
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 80: return "#30d158"
    if score >= 60: return "#ffd60a"
    if score >= 40: return "#ff9f0a"
    return "#ff2d55"

def _score_grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"

def render_score(score: float):
    color = _score_color(score)
    grade = _score_grade(score)
    deg   = int(score / 100 * 360)
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
      <div class="score-circle" style="--score-color:{color}; --score-deg:{deg}deg;">
        <div class="score-inner">
          <div class="score-number" style="color:{color}">{score:.0f}</div>
          <div class="score-label">Grade {grade}</div>
        </div>
      </div>
      <div style="margin-top:0.8rem; font-size:0.9rem; color:#8888aa;">Quality Score / 100</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Finding card rendering
# ---------------------------------------------------------------------------

def render_finding(f: dict, category: str):
    sev   = str(f.get("severity", "INFO")).upper()
    color = SEVERITY_COLOR.get(sev, "#8e8e93")
    icon  = SEVERITY_ICON.get(sev, "⚪")
    title = f.get("title", "Finding")
    desc  = f.get("description", "")
    fp    = f.get("file_path", "")
    sl    = f.get("start_line", "")
    el    = f.get("end_line", "")
    fix   = f.get("suggested_fix", "")
    conf  = f.get("confidence", 0)

    location = f"`{fp}` : {sl}–{el}" if fp else ""

    with st.expander(f"{icon} **[{sev}]** {title}", expanded=False):
        c1, c2 = st.columns([3, 1])
        with c1:
            if location:
                st.caption(f"📁 {location}")
            st.markdown(desc)
        with c2:
            st.metric("Confidence", f"{conf*100:.0f}%")

        # Category-specific extra fields
        if category == "bugs":
            pattern = f.get("bug_pattern", "")
            repro   = f.get("reproducible", "")
            if pattern: st.caption(f"Pattern: `{pattern}` | Reproducible: {repro}")
        elif category == "security":
            cat_  = f.get("category", "")
            cwe   = f.get("cwe_id", "")
            cvss  = f.get("cvss_score", "")
            expl  = f.get("exploitability", "")
            parts = [p for p in [cat_, cwe, f"CVSS {cvss}" if cvss else "", expl] if p]
            if parts: st.caption(" | ".join(parts))
        elif category == "solid":
            principle = f.get("principle", "")
            entity    = f.get("violated_class_or_function", "")
            hint      = f.get("refactor_hint", "")
            if principle: st.caption(f"Principle: **{principle}** | Entity: `{entity}`")
            if hint:      st.caption(f"Hint: {hint}")
        elif category == "architecture":
            smell  = f.get("smell", "")
            radius = f.get("impact_radius", "")
            mods   = ", ".join(f.get("affected_modules", [])[:3])
            if smell: st.caption(f"Smell: **{smell}** | Impact: {radius}")
            if mods:  st.caption(f"Modules: `{mods}`")
        elif category == "complexity":
            cc   = f.get("cyclomatic_complexity", "")
            cog  = f.get("cognitive_complexity", "")
            nest = f.get("nesting_depth", "")
            fn   = f.get("function_name", "")
            parts = []
            if fn:   parts.append(f"`{fn}`")
            if cc:   parts.append(f"CC={cc}")
            if cog:  parts.append(f"Cog={cog}")
            if nest: parts.append(f"Depth={nest}")
            if parts: st.caption(" | ".join(parts))

        if fix:
            st.markdown("**Suggested Fix:**")
            st.code(fix, language="text")

# ---------------------------------------------------------------------------
# Findings section
# ---------------------------------------------------------------------------

def render_findings(result: dict):
    cats = [
        ("bug_findings",          "🐛 Bug Findings",          "bugs"),
        ("security_findings",     "🔒 Security Findings",     "security"),
        ("architecture_findings", "🏛️ Architecture Findings", "architecture"),
        ("solid_findings",        "🏗️ SOLID Violations",      "solid"),
        ("complexity_findings",   "📊 Complexity Findings",   "complexity"),
    ]

    # Tab layout
    labels = []
    groups = []
    for key, label, cat in cats:
        items = result.get(key, [])
        if items:
            labels.append(f"{label} ({len(items)})")
            groups.append((items, cat))

    if not labels:
        st.info("No findings detected — your repository looks clean!")
        return

    tabs = st.tabs(labels)
    for tab, (items, cat) in zip(tabs, groups):
        with tab:
            # Sort by severity
            order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
            sorted_items = sorted(items, key=lambda x: order.get(str(x.get("severity","INFO")).upper(), 5))

            # Severity summary bar
            counts = {}
            for f in sorted_items:
                s = str(f.get("severity","INFO")).upper()
                counts[s] = counts.get(s, 0) + 1

            cols = st.columns(len(counts))
            for col, (sev, cnt) in zip(cols, sorted(counts.items(), key=lambda x: order.get(x[0], 5))):
                col.metric(f"{SEVERITY_ICON.get(sev,'')} {sev}", cnt)

            st.markdown("---")
            for f in sorted_items:
                render_finding(f, cat)

# ---------------------------------------------------------------------------
# Repository info card
# ---------------------------------------------------------------------------

def render_repo_info(result: dict):
    meta = result.get("metadata", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="val">'
                    f'{meta.get("total_files", 0)}'
                    '</div><div class="lbl">FILES</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="val">'
                    f'{meta.get("total_lines", 0):,}'
                    '</div><div class="lbl">LINES</div></div>', unsafe_allow_html=True)
    with c3:
        lang = str(meta.get("primary_language", "—")).upper()
        st.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.2rem">{lang}'
                    '</div><div class="lbl">LANGUAGE</div></div>', unsafe_allow_html=True)
    with c4:
        total = result.get("total_findings", 0)
        st.markdown(f'<div class="metric-card"><div class="val">{total}'
                    '</div><div class="lbl">FINDINGS</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Download section
# ---------------------------------------------------------------------------

def render_downloads(result: dict):
    st.markdown("### 📥 Download Report")
    col1, col2 = st.columns(2)

    # Build Markdown content from summary
    md = result.get("summary_markdown", "# Report\n\nNo summary available.")
    repo = result.get("metadata", {}).get("repository_name", "repo")
    score = result.get("overall_score", 0)
    full_md = f"# CodeGuardian AI Report — {repo}\n\n**Score: {score}/100**\n\n{md}"

    with col1:
        st.download_button(
            label="⬇️ Download Markdown Report",
            data=full_md.encode("utf-8"),
            file_name=f"codeguardian_{repo}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        # PDF: attempt to convert via WeasyPrint; fallback to Markdown
        try:
            import markdown as md_lib
            from weasyprint import HTML as WH
            html_body = md_lib.markdown(full_md, extensions=["tables", "fenced_code"])
            html_doc  = f"<html><body style='font-family:sans-serif;padding:2cm'>{html_body}</body></html>"
            import io
            buf = io.BytesIO()
            WH(string=html_doc).write_pdf(buf)
            buf.seek(0)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=buf.getvalue(),
                file_name=f"codeguardian_{repo}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.download_button(
                label="⬇️ Download Report (Markdown)",
                data=full_md.encode("utf-8"),
                file_name=f"codeguardian_{repo}.md",
                mime="text/markdown",
                use_container_width=True,
                help="PDF unavailable (WeasyPrint not installed). Downloading Markdown instead.",
            )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

# ── Hero header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1.5rem 0;">
  <div class="hero-title">🛡️ CodeGuardian AI</div>
  <div class="hero-sub">AI-powered code review across bugs, security, SOLID, architecture & complexity</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Input form ───────────────────────────────────────────────────────────────
with st.container():
    c_url, c_name, c_btn = st.columns([4, 2, 1])
    with c_url:
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/owner/repository",
            label_visibility="collapsed",
            key="repo_url_input",
        )
    with c_name:
        repo_name = st.text_input(
            "Repository Name",
            value="",
            placeholder="my-project (optional)",
            label_visibility="collapsed",
            key="repo_name_input",
        )
    with c_btn:
        analyze_clicked = st.button(
            "🔍 Analyze",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.polling,
        )

# ── Submit handler ────────────────────────────────────────────────────────────
if analyze_clicked:
    if not repo_url or not repo_url.startswith("http"):
        st.error("Please enter a valid GitHub URL (must start with http).")
    else:
        # Reset state
        for k in ["job_id", "status", "progress", "result", "error", "md_report", "poll_count"]:
            st.session_state[k] = None if k != "poll_count" else 0
        st.session_state.polling = True

        name = repo_name.strip() or repo_url.rstrip("/").split("/")[-1]
        job_id = _submit(repo_url, name, api_key, api_base)
        if job_id:
            st.session_state.job_id = job_id
            st.session_state.status = "PENDING"
            st.rerun()
        else:
            st.session_state.polling = False

# ── Polling loop ──────────────────────────────────────────────────────────────
if st.session_state.polling and st.session_state.job_id:
    job_id = st.session_state.job_id

    # FIX (BUG 9): abort if polling too long (hung job / server restart)
    st.session_state.poll_count = (st.session_state.poll_count or 0) + 1
    if st.session_state.poll_count > MAX_POLLS:
        st.session_state.polling = False
        st.session_state.error   = (
            f"Analysis timed out after {MAX_POLLS * POLL_SECS // 60} minutes. "
            "The job may still be running. Check the backend logs or try again."
        )
        st.rerun()

    status_data = _poll(job_id, api_key, api_base)
    current_status   = status_data.get("status", "PENDING")
    current_progress = status_data.get("progress", st.session_state.progress or 0)

    st.session_state.status   = current_status
    st.session_state.progress = current_progress

    if current_status == "COMPLETED":
        st.session_state.result  = status_data.get("result")
        st.session_state.polling = False
    elif current_status == "FAILED":
        st.session_state.error   = status_data.get("error", "Unknown error")
        st.session_state.polling = False
    else:
        time.sleep(POLL_SECS)
        st.rerun()

# ── Progress display ──────────────────────────────────────────────────────────
if st.session_state.job_id and st.session_state.status:
    st.markdown("---")

    status_val = st.session_state.status
    progress   = st.session_state.progress or 0

    # Stage bar
    stage_label = _stage_label(progress)
    stage_html  = '<div class="stage-bar">'
    for lo, hi, label in STAGES:
        if progress >= hi:
            css = "done"
        elif lo <= progress < hi:
            css = "active"
        else:
            css = ""
        stage_html += f'<span class="stage {css}">{"✓ " if css=="done" else ""}{label}</span>'
    stage_html += "</div>"
    st.markdown(stage_html, unsafe_allow_html=True)

    status_color = {"PENDING": "🟡", "RUNNING": "🔵", "COMPLETED": "🟢", "FAILED": "🔴"}
    icon = status_color.get(status_val, "⚪")

    col_prog, col_stat = st.columns([4, 1])
    with col_prog:
        st.progress(min(progress, 100) / 100)
    with col_stat:
        st.markdown(f"**{icon} {status_val}** — {progress}%")

    if st.session_state.polling:
        st.caption(f"⏳ {stage_label}... polling every {POLL_SECS}s  •  Job ID: `{st.session_state.job_id[:8]}…`")

# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"**Analysis Failed**\n\n{st.session_state.error}")
    if st.button("🔄 Try Again"):
        for k in ["job_id", "status", "progress", "result", "error", "polling"]:
            st.session_state[k] = None if k != "polling" else False
        st.rerun()

# ── Results display ───────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    st.markdown("---")

    # Repository info
    repo_name_display = result.get("metadata", {}).get("repository_name", "Repository")
    st.markdown(f"## 📁 {repo_name_display}")
    render_repo_info(result)

    st.markdown("---")

    # Score + category breakdown
    score = result.get("overall_score", 0)
    c_score, c_breakdown = st.columns([1, 2])

    with c_score:
        render_score(score)

    with c_breakdown:
        st.markdown("### Score Breakdown")
        cats_info = [
            ("bug_findings",          "🐛 Bugs",         0.30),
            ("solid_findings",        "🏗️ SOLID",        0.25),
            ("architecture_findings", "🏛️ Architecture", 0.20),
            ("security_findings",     "🔒 Security",     0.15),
            ("complexity_findings",   "📊 Complexity",   0.10),
        ]
        for key, label, weight in cats_info:
            count = len(result.get(key, []))
            # Compute category sub-score (same formula as report_agent)
            pen = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5, "LOW": 2, "INFO": 0.5}
            total_pen = sum(pen.get(str(f.get("severity","INFO")).upper(), 0)
                           for f in result.get(key, []))
            cat_score = max(0, 100 - min(total_pen, 100))
            col_l, col_bar, col_r = st.columns([2, 5, 1])
            col_l.caption(f"{label} ({weight*100:.0f}%)")
            col_bar.progress(cat_score / 100)
            col_r.caption(f"{cat_score:.0f}")

    st.markdown("---")

    # Findings
    st.markdown("## 🔍 Findings")
    render_findings(result)

    st.markdown("---")

    # Executive summary
    if result.get("summary_markdown"):
        with st.expander("📋 Executive Summary", expanded=False):
            st.markdown(result["summary_markdown"])

    st.markdown("---")

    # Downloads
    render_downloads(result)

    # Job metadata footer
    st.markdown("---")
    st.caption(
        f"Job ID: `{result.get('job_id','?')}` · "
        f"Reviewed: {result.get('reviewed_at','?')[:19].replace('T',' ')} UTC · "
        f"Total findings: {result.get('total_findings', 0)}"
    )
