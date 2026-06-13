"""
reports/generator.py
=====================
Report generation: ReviewResult → Markdown string + PDF file.

Public API
----------
    generate_markdown_report(result: ReviewResult) -> str
    generate_pdf_report(result: ReviewResult, output_dir: Path) -> Path

Pipeline
--------
    ReviewResult
        │
        ▼
    Jinja2 (report_template.jinja2)
        │ renders Markdown
        ▼
    markdown_string
        │
        ├──► .md file (written by ReportAgent)
        │
        ▼
    markdown-it / mistune → HTML string
        │
        ▼
    WeasyPrint → PDF bytes → .pdf file

WeasyPrint is an optional dependency. If it is not installed,
generate_pdf_report raises ImportError — callers should handle gracefully.
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timezone

import structlog

from schemas import ReviewResult

logger = structlog.get_logger(__name__)

TEMPLATE_DIR  = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report_template.jinja2"
TEMPLATE_PATH = TEMPLATE_DIR / TEMPLATE_NAME


# ---------------------------------------------------------------------------
# Jinja2 helpers
# ---------------------------------------------------------------------------

def _get_jinja_env():
    """Return a configured Jinja2 Environment loading from TEMPLATE_DIR."""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise ImportError("Jinja2 is required: pip install jinja2") from exc

    env = Environment(
        loader=select_autoescape(enabled_extensions=("jinja2", "html", "md")),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # Override loader to read from template directory
    env.loader = FileSystemLoader(str(TEMPLATE_DIR))
    return env


def _severity_sort_key(finding) -> int:
    _ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    return _ORDER.get(str(getattr(finding, "severity", "INFO")), 5)


def _build_template_context(result: ReviewResult) -> dict:
    """Convert ReviewResult into a flat, Jinja2-friendly context dict."""
    return {
        "result": result,
        "reviewed_at_fmt": result.reviewed_at.strftime("%Y-%m-%d %H:%M UTC"),
        "score_grade": _grade(result.overall_score),
        "score_color": _score_color(result.overall_score),
        "bug_findings_sorted":          sorted(result.bug_findings,          key=_severity_sort_key),
        "solid_findings_sorted":        sorted(result.solid_findings,        key=_severity_sort_key),
        "architecture_findings_sorted": sorted(result.architecture_findings, key=_severity_sort_key),
        "security_findings_sorted":     sorted(result.security_findings,     key=_severity_sort_key),
        "complexity_findings_sorted":   sorted(result.complexity_findings,   key=_severity_sort_key),
    }


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def _score_color(score: float) -> str:
    if score >= 75: return "green"
    if score >= 50: return "orange"
    return "red"


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_markdown_report(result: ReviewResult) -> str:
    """
    Render the ReviewResult as a Markdown string using the Jinja2 template.

    Args:
        result: Complete ReviewResult from compile_node / ReportAgent.

    Returns:
        Markdown string.

    Raises:
        ImportError : Jinja2 is not installed.
        RuntimeError: Template rendering failed.
    """
    logger.info("generate_markdown_report", job_id=result.job_id)

    try:
        env      = _get_jinja_env()
        template = env.get_template(TEMPLATE_NAME)
        context  = _build_template_context(result)
        rendered = template.render(**context)
        logger.info(
            "generate_markdown_report_complete",
            job_id=result.job_id,
            chars=len(rendered),
        )
        return rendered
    except Exception as exc:
        logger.error("generate_markdown_report_failed", error=str(exc))
        raise RuntimeError(f"Markdown report rendering failed: {exc}") from exc


# ---------------------------------------------------------------------------
# HTML conversion helper (for PDF)
# ---------------------------------------------------------------------------

def _markdown_to_html(md: str, title: str) -> str:
    """
    Convert a Markdown string to a styled HTML document.
    Tries mistune → markdown → basic fallback in order of preference.
    """
    body_html = ""

    # Option 1 — mistune (preferred, lightweight)
    try:
        import mistune
        body_html = mistune.html(md)
    except ImportError:
        pass

    # Option 2 — markdown (stdlib-style)
    if not body_html:
        try:
            import markdown as md_lib
            body_html = md_lib.markdown(md, extensions=["tables", "fenced_code"])
        except ImportError:
            pass

    # Option 3 — bare minimum: wrap in <pre>
    if not body_html:
        escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body_html = f"<pre>{escaped}</pre>"

    css = """
    @page { margin: 2cm; }
    body  { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 12px; line-height: 1.6; color: #1a1a2e; }
    h1    { font-size: 24px; color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }
    h2    { font-size: 18px; color: #0f3460; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; margin-top: 24px; }
    h3    { font-size: 14px; color: #1a1a2e; margin-top: 16px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 11px; }
    th    { background: #0f3460; color: white; padding: 8px; text-align: left; }
    td    { padding: 6px 8px; border: 1px solid #ddd; }
    tr:nth-child(even) { background: #f9f9f9; }
    code  { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-family: monospace; font-size: 11px; }
    pre   { background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 10px; }
    hr    { border: none; border-top: 1px solid #e0e0e0; margin: 16px 0; }
    blockquote { border-left: 4px solid #0f3460; margin: 0; padding-left: 12px; color: #555; }
    .score-badge { display: inline-block; background: #0f3460; color: white;
                   padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
{body_html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_pdf_report(result: ReviewResult, output_dir: str | Path) -> Path:
    """
    Generate a PDF report from the ReviewResult and save to output_dir.

    Pipeline: ReviewResult → Markdown → HTML → PDF (WeasyPrint).

    Args:
        result     : Complete ReviewResult.
        output_dir : Directory to save the PDF file.

    Returns:
        Absolute Path to the generated PDF file.

    Raises:
        ImportError : WeasyPrint is not installed.
        RuntimeError: PDF generation failed.
    """
    try:
        from weasyprint import HTML as WeasyHTML
    except ImportError as exc:
        raise ImportError(
            "WeasyPrint is required for PDF generation: pip install weasyprint"
        ) from exc

    logger.info("generate_pdf_report", job_id=result.job_id)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Markdown
    md_content = generate_markdown_report(result)

    # Step 2: HTML
    repo_name = result.metadata.repository_name
    title     = f"CodeGuardian AI Report — {repo_name}"
    html      = _markdown_to_html(md_content, title)

    # Step 3: PDF via WeasyPrint
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", repo_name)[:40]
    filename  = f"report_{result.job_id[:8]}_{safe_name}.pdf"
    pdf_path  = output_dir / filename

    try:
        WeasyHTML(string=html).write_pdf(str(pdf_path))
    except Exception as exc:
        logger.error("generate_pdf_report_failed", error=str(exc))
        raise RuntimeError(f"PDF generation failed: {exc}") from exc

    logger.info("generate_pdf_report_complete", path=str(pdf_path), size=pdf_path.stat().st_size)
    return pdf_path
