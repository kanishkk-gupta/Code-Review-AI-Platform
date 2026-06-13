"""
agents/report_agent.py
=======================
ReportAgent — Code quality scorer and report generator.

Responsibilities
----------------
1. Compute a weighted composite quality score (0–100) from all five finding
   categories using the canonical weight distribution:

       Bugs           30%
       SOLID          25%
       Architecture   20%
       Security       15%
       Complexity     10%

2. Generate an executive summary (Markdown string) via LLM or deterministic
   fallback template when the LLM is unavailable.

3. Produce a Markdown report file and a PDF report file, saved under the
   configured output directory (defaults to reports/output/).

4. Return a ReportPaths dataclass with the absolute paths to both files.

Not a BaseAgent subclass
------------------------
ReportAgent does not analyse code; it aggregates and renders. It operates on
completed finding lists and RepositoryMetadata — not on raw CodeChunks.

Usage
-----
::

    from agents.report_agent import ReportAgent

    agent = ReportAgent()
    paths = await agent.run(
        job_id=state["job_id"],
        metadata=RepositoryMetadata(**state["metadata"]),
        bug_findings=state["bug_findings"],
        solid_findings=state["solid_findings"],
        architecture_findings=state["architecture_findings"],
        security_findings=state["security_findings"],
        complexity_findings=state["complexity_findings"],
    )
    print(paths.markdown_path)
    print(paths.pdf_path)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uuid

import structlog

from reports.generator import generate_markdown_report, generate_pdf_report
from schemas import (
    ArchitectureFinding,
    BugFinding,
    ComplexityFinding,
    RepositoryMetadata,
    ReviewResult,
    SecurityFinding,
    Severity,
    SolidFinding,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Score weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_WEIGHTS: dict[str, float] = {
    "bugs":         0.30,
    "solid":        0.25,
    "architecture": 0.20,
    "security":     0.15,
    "complexity":   0.10,
}

# Severity penalty per finding (applied within each category's sub-score)
_PENALTY: dict[str, float] = {
    Severity.CRITICAL: 20.0,
    Severity.HIGH:     10.0,
    Severity.MEDIUM:    5.0,
    Severity.LOW:       2.0,
    Severity.INFO:      0.5,
}

# Default output directory (relative to this file's package root)
_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "output"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ReportPaths:
    """Paths to generated report files."""
    markdown_path:    Path
    pdf_path:         Optional[Path]   # None when WeasyPrint is unavailable
    job_id:           str
    overall_score:    float
    summary_markdown: str              # FIX (BUG 8): expose exec summary for workflow node


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _category_score(findings: list, max_penalty: float = 100.0) -> float:
    """
    Compute a 0–100 sub-score for one finding category.

    Starts at 100 and subtracts per-severity penalties, clamped to [0, 100].
    ``max_penalty`` caps the total deduction so a single very bad category
    cannot drag the weighted total disproportionately.
    """
    total_penalty = 0.0
    for finding in findings:
        sev = str(getattr(finding, "severity", Severity.INFO))
        total_penalty += _PENALTY.get(sev, 0.5)
        if total_penalty >= max_penalty:
            break
    return max(0.0, 100.0 - min(total_penalty, max_penalty))


def compute_score(
    bug_findings:          list[BugFinding],
    solid_findings:        list[SolidFinding],
    architecture_findings: list[ArchitectureFinding],
    security_findings:     list[SecurityFinding],
    complexity_findings:   list[ComplexityFinding],
) -> float:
    """
    Compute the weighted composite quality score.

    Formula::

        score = Σ weight_i × category_score_i

    Weights: Bugs 30%, SOLID 25%, Architecture 20%, Security 15%, Complexity 10%

    Returns:
        float in [0.0, 100.0], rounded to 1 decimal place.
    """
    scores = {
        "bugs":         _category_score(bug_findings),
        "solid":        _category_score(solid_findings),
        "architecture": _category_score(architecture_findings),
        "security":     _category_score(security_findings),
        "complexity":   _category_score(complexity_findings),
    }
    weighted = sum(_WEIGHTS[k] * v for k, v in scores.items())
    result   = round(max(0.0, min(100.0, weighted)), 1)

    logger.info(
        "score_computed",
        category_scores={k: round(v, 1) for k, v in scores.items()},
        overall=result,
    )
    return result


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    if score >= 90: return "A (Excellent)"
    if score >= 75: return "B (Good)"
    if score >= 60: return "C (Fair)"
    if score >= 45: return "D (Poor)"
    return "F (Critical — immediate attention required)"


def _deterministic_summary(
    metadata: RepositoryMetadata,
    score:    float,
    bug_findings:          list,
    solid_findings:        list,
    architecture_findings: list,
    security_findings:     list,
    complexity_findings:   list,
) -> str:
    """
    Generate a deterministic Markdown executive summary without an LLM.
    Used as fallback when the LLM is unavailable.
    """
    total = (
        len(bug_findings) + len(solid_findings) + len(architecture_findings)
        + len(security_findings) + len(complexity_findings)
    )

    critical_sec = [f for f in security_findings if str(getattr(f, "severity", "")) == Severity.CRITICAL]
    high_bugs    = [f for f in bug_findings      if str(getattr(f, "severity", "")) in (Severity.CRITICAL, Severity.HIGH)]
    arch_cycles  = [f for f in architecture_findings if "Cyclic" in str(getattr(f, "smell", ""))]

    lines = [
        f"## Executive Summary",
        f"",
        f"**Repository:** `{metadata.repository_name}` | "
        f"**Language:** {metadata.primary_language} | "
        f"**Grade:** {_grade(score)}",
        f"",
        f"CodeGuardian AI analysed **{metadata.total_files} files** "
        f"({metadata.total_lines:,} lines of code) and identified **{total} total findings** "
        f"across five quality dimensions.",
        f"",
        f"### Score Breakdown",
        f"",
        f"| Category | Findings | Weight |",
        f"|----------|----------|--------|",
        f"| 🐛 Bugs | {len(bug_findings)} | 30% |",
        f"| 🏗️ SOLID Violations | {len(solid_findings)} | 25% |",
        f"| 🏛️ Architecture | {len(architecture_findings)} | 20% |",
        f"| 🔒 Security | {len(security_findings)} | 15% |",
        f"| 📊 Complexity | {len(complexity_findings)} | 10% |",
        f"| **Total** | **{total}** | — |",
        f"",
    ]

    if critical_sec:
        lines += [
            f"### ⚠️ Critical Security Issues",
            f"",
            f"{len(critical_sec)} CRITICAL security finding(s) require **immediate remediation** "
            f"before deployment.",
            f"",
        ]

    if high_bugs:
        lines += [
            f"### 🐛 High-Severity Bugs",
            f"",
            f"{len(high_bugs)} high-severity bug(s) detected that are likely to cause runtime "
            f"failures or incorrect behavior.",
            f"",
        ]

    if arch_cycles:
        lines += [
            f"### 🔄 Circular Dependencies",
            f"",
            f"{len(arch_cycles)} circular dependency cycle(s) detected. These must be broken "
            f"to allow safe modular testing and future refactoring.",
            f"",
        ]

    lines += [
        f"### Recommendation",
        f"",
        f"{'Address all CRITICAL and HIGH findings before the next release.' if score < 60 else 'Schedule a focused refactoring sprint to address the findings above.' if score < 80 else 'The codebase is in good health. Address remaining findings during normal development cycles.'}",
    ]
    return "\n".join(lines)


async def _llm_summary(
    metadata: RepositoryMetadata,
    score: float,
    all_findings_summary: str,
) -> Optional[str]:
    """Attempt to generate the executive summary via LLM. Returns None on failure."""
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from services.llm_client import get_llm_client

        tmpl = (
            "You are a senior engineering lead reviewing a code quality report.\n\n"
            "Repository: {repository_name}\n"
            "Overall quality score: {score}/100 ({grade})\n\n"
            "Finding summary:\n{findings_summary}\n\n"
            "Write a concise executive summary (3-5 paragraphs, Markdown format) for "
            "an engineering manager. Cover: what was found, biggest risks, and clear "
            "actionable next steps. Be specific, professional, and direct."
        )
        prompt = PromptTemplate(
            template=tmpl,
            input_variables=["repository_name", "score", "grade", "findings_summary"],
        )
        chain = prompt | get_llm_client() | StrOutputParser()
        result = await chain.ainvoke({
            "repository_name":  metadata.repository_name,
            "score":            score,
            "grade":            _grade(score),
            "findings_summary": all_findings_summary,
        })
        return str(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_agent_llm_summary_failed", error=str(exc))
        return None


def _findings_summary_text(
    bug_findings, solid_findings, architecture_findings,
    security_findings, complexity_findings,
) -> str:
    """Build a concise text block describing the findings for LLM context."""
    lines: list[str] = []
    for label, findings in [
        ("Bugs",         bug_findings),
        ("SOLID",        solid_findings),
        ("Architecture", architecture_findings),
        ("Security",     security_findings),
        ("Complexity",   complexity_findings),
    ]:
        if not findings:
            continue
        lines.append(f"{label} ({len(findings)} findings):")
        for f in findings[:3]:
            sev   = getattr(f, "severity", "?")
            title = getattr(f, "title", "—")
            lines.append(f"  [{sev}] {title}")
        if len(findings) > 3:
            lines.append(f"  … and {len(findings)-3} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------

class ReportAgent:
    """
    Aggregation and report generation agent.

    Not a subclass of BaseAgent — takes completed finding lists,
    not raw CodeChunks.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        job_id:                str,
        metadata:              RepositoryMetadata,
        # FIX (BUG 10): correct Optional type annotations — None is a valid default
        bug_findings:          Optional[list[BugFinding]]          = None,
        solid_findings:        Optional[list[SolidFinding]]        = None,
        architecture_findings: Optional[list[ArchitectureFinding]] = None,
        security_findings:     Optional[list[SecurityFinding]]     = None,
        complexity_findings:   Optional[list[ComplexityFinding]]   = None,
        *,
        llm_summary:           bool = True,
    ) -> ReportPaths:
        """
        Compute score, generate summary, write Markdown + PDF.

        Args:
            job_id                : UUID of the review job.
            metadata              : RepositoryMetadata from ingest_node.
            bug_findings          : Output of BugAgent.run().
            solid_findings        : Output of SolidAgent.run().
            architecture_findings : Output of ArchitectureAgent.analyse_repository().
            security_findings     : Output of SecurityAgent.run().
            complexity_findings   : Output of ComplexityAgent.run().
            llm_summary           : If True, attempt LLM-generated executive summary.

        Returns:
            ReportPaths with absolute paths to generated files.
        """
        bug_findings          = bug_findings          or []
        solid_findings        = solid_findings        or []
        architecture_findings = architecture_findings or []
        security_findings     = security_findings     or []
        complexity_findings   = complexity_findings   or []

        logger.info(
            "report_agent_run",
            job_id=job_id,
            bugs=len(bug_findings),
            solid=len(solid_findings),
            architecture=len(architecture_findings),
            security=len(security_findings),
            complexity=len(complexity_findings),
        )

        # 1. Compute score
        score = compute_score(
            bug_findings, solid_findings, architecture_findings,
            security_findings, complexity_findings,
        )

        # 2. Generate executive summary
        summary_md: Optional[str] = None
        if llm_summary:
            findings_text = _findings_summary_text(
                bug_findings, solid_findings, architecture_findings,
                security_findings, complexity_findings,
            )
            summary_md = await _llm_summary(metadata, score, findings_text)

        if not summary_md:
            summary_md = _deterministic_summary(
                metadata, score,
                bug_findings, solid_findings, architecture_findings,
                security_findings, complexity_findings,
            )

        # 3. Build ReviewResult
        result = ReviewResult(
            job_id=job_id,
            metadata=metadata,
            bug_findings=bug_findings,
            solid_findings=solid_findings,
            architecture_findings=architecture_findings,
            security_findings=security_findings,
            complexity_findings=complexity_findings,
            overall_score=score,
            summary_markdown=summary_md,
        )

        # 4. Write Markdown
        md_path = await self._write_markdown(result)

        # 5. Write PDF
        pdf_path = await self._write_pdf(result)

        logger.info(
            "report_agent_complete",
            job_id=job_id,
            score=score,
            markdown=str(md_path),
            pdf=str(pdf_path) if pdf_path else "unavailable",
        )

        return ReportPaths(
            markdown_path=md_path,
            pdf_path=pdf_path,
            job_id=job_id,
            overall_score=score,
            summary_markdown=summary_md,  # FIX (BUG 8): expose for workflow node
        )

    # ── File writers ──────────────────────────────────────────────────────

    async def _write_markdown(self, result: ReviewResult) -> Path:
        """Render and save the Markdown report. Returns the output path."""
        import asyncio

        loop = asyncio.get_event_loop()
        md_content = await loop.run_in_executor(None, generate_markdown_report, result)

        filename = f"report_{result.job_id[:8]}_{_safe_name(result.metadata.repository_name)}.md"
        path     = self.output_dir / filename
        path.write_text(md_content, encoding="utf-8")

        logger.info("report_markdown_saved", path=str(path), size=len(md_content))
        return path

    async def _write_pdf(self, result: ReviewResult) -> Optional[Path]:
        """Generate and save the PDF report. Returns None if WeasyPrint is absent."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            path = await loop.run_in_executor(
                None, generate_pdf_report, result, self.output_dir
            )
            logger.info("report_pdf_saved", path=str(path))
            return path
        except ImportError:
            logger.warning("report_pdf_skipped", reason="WeasyPrint not installed")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("report_pdf_failed", error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Sanitize repository name for use in a filename."""
    import re
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:40]
