"""
graph/nodes/enrich_node.py
===========================
LangGraph node: enrich_node

Reads from ReviewState:
  - bug_findings, solid_findings, architecture_findings,
    security_findings, complexity_findings
  - faiss_index : FAISSIndex (in-process)
  - config.similarity_top_k

Writes to ReviewState:
  - Updated *_findings with related_chunk_ids populated
  - progress : 80
  - error    : str (on failure)

For each finding across all 5 categories:
  1. Build a semantic query from finding.description
  2. Run FAISS similarity_search(query, k=config.similarity_top_k)
  3. Set finding.related_chunk_ids = [chunk.chunk_id for chunk in results]
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


async def enrich_node(state: dict) -> dict:
    """
    Enrich node: attach semantically similar code chunks to each finding.

    Steps:
      1. Collect all findings from state.
      2. For each finding, query FAISS with finding.description.
      3. Attach top-k chunk IDs as related_chunk_ids.
      4. Update state and set progress=80.
    """
    job_id = state.get("job_id", "unknown")

    all_findings_count = (
        len(state.get("bug_findings", []))
        + len(state.get("solid_findings", []))
        + len(state.get("architecture_findings", []))
        + len(state.get("security_findings", []))
        + len(state.get("complexity_findings", []))
    )

    logger.info("enrich_node_start", job_id=job_id, total_findings=all_findings_count)

    try:
        # TODO: Phase 2 — implement FAISS enrichment
        # from rag.retriever import similarity_search
        #
        # faiss_index = state["faiss_index"]
        # top_k = state["config"]["similarity_top_k"]
        #
        # for category in ["bug_findings", "solid_findings", ...]:
        #     for finding in state[category]:
        #         neighbors = similarity_search(faiss_index, finding["description"], k=top_k)
        #         finding["related_chunk_ids"] = [c["chunk_id"] for c in neighbors]
        #
        # return {**state, "progress": 80}

        raise NotImplementedError("enrich_node — Phase 2 implementation pending.")

    except Exception as exc:  # noqa: BLE001
        logger.exception("enrich_node_error", job_id=job_id, error=str(exc))
        return {**state, "error": str(exc)}
