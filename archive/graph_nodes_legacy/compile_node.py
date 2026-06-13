"""
graph/nodes/compile_node.py
============================
LangGraph node: compile_node

Reads from ReviewState:
  - All *_findings lists
  - metadata : RepositoryMetadata
  - job_id   : str

Writes to ReviewState:
  - result   : ReviewResult (complete, immutable)
  - progress : 100

Also responsible for:
  - Computing overall_score (0–100) from findings severity distribution
  - Generating summary_markdown (via LLM call or template)
  - Clearing faiss_index and chunks from state to free memory

Score formula (from data_flow.md):
  base = 100
  CRITICAL → -15 each
  HIGH     → -8  each
  MEDIUM   → -3  each
  LOW      → -1  each
  clamp to [0, 100]
"""

from __future__ import annotations

import structlog

from schemas import Severity

logger = structlog.get_logger(__name__)

# Score penalty per severity level
_SEVERITY_PENALTY: dict[str, int] = {
    Severity.CRITICAL: 15,
    Severity.HIGH: 8,
    Severity.MEDIUM: 3,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


def compute_overall_score(all_findings: list[dict]) -> float:
    """
    Compute the 0–100 quality score from all findings.
    Lower severity count = higher score.
    """
    score = 100.0
    for finding in all_findings:
        penalty = _SEVERITY_PENALTY.get(finding.get("severity", Severity.INFO), 0)
        score -= penalty
    return max(0.0, min(100.0, score))


async def compile_node(state: dict) -> dict:
    """
    Compile node: aggregate all findings into a frozen ReviewResult.

    Steps:
      1. Collect all findings from all categories.
      2. Compute overall_score.
      3. Generate summary_markdown (LLM or template).
      4. Build ReviewResult Pydantic model.
      5. Destroy FAISS index and chunk data to free memory.
      6. Set progress=100 and return result.
    """
    job_id = state.get("job_id", "unknown")
    logger.info("compile_node_start", job_id=job_id)

    try:
        # TODO: Phase 2 — full compile implementation
        # all_findings = (
        #     state["bug_findings"]
        #     + state["solid_findings"]
        #     + state["architecture_findings"]
        #     + state["security_findings"]
        #     + state["complexity_findings"]
        # )
        # score = compute_overall_score(all_findings)
        # summary = await generate_summary(state["metadata"], all_findings, score)
        # result = ReviewResult(
        #     job_id=job_id,
        #     metadata=RepositoryMetadata(**state["metadata"]),
        #     bug_findings=[BugFinding(**f) for f in state["bug_findings"]],
        #     ...
        #     overall_score=score,
        #     summary_markdown=summary,
        # )
        # return {
        #     "result": result.model_dump(),
        #     "faiss_index": None,   # free memory
        #     "chunks": [],          # free memory
        #     "progress": 100,
        # }

        raise NotImplementedError("compile_node — Phase 2 implementation pending.")

    except Exception as exc:  # noqa: BLE001
        logger.exception("compile_node_error", job_id=job_id, error=str(exc))
        return {**state, "error": str(exc)}
