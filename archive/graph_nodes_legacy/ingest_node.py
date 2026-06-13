"""
graph/nodes/ingest_node.py
===========================
LangGraph node: ingest_node

Reads from ReviewState:
  - source_url | source_zip_b64
  - config.max_chunk_lines

Writes to ReviewState:
  - metadata   : RepositoryMetadata
  - chunks     : List[CodeChunk]
  - faiss_index: FAISSIndex (in-process, excluded from serialization)
  - progress   : 20
  - error      : str (on failure)

Must NOT write to: bug_findings, solid_findings, architecture_findings,
                   security_findings, complexity_findings, result
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


async def ingest_node(state: dict) -> dict:
    """
    Ingest node: parse source, chunk files, build FAISS index.

    Steps:
      1. Clone repo (URL) or extract ZIP (base64).
      2. Filter allowed file extensions.
      3. Extract RepositoryMetadata.
      4. Chunk each file via tools.chunker.chunk_file().
      5. Embed chunks via rag.embedder.embed_chunks().
      6. Build per-job FAISS index via rag.vector_store.build_index().
      7. Update state and set progress=20.

    On any exception:
      - Set state["error"] = str(exception)
      - LangGraph conditional edge will route to error terminal.
    """
    job_id = state.get("job_id", "unknown")
    logger.info("ingest_node_start", job_id=job_id)

    try:
        # TODO: Phase 2 — implement full ingest pipeline
        # from tools.file_parser import parse_source
        # from tools.chunker import chunk_file
        # from rag.embedder import embed_chunks
        # from rag.vector_store import build_index
        #
        # file_tree = await parse_source(state["source_url"], state["source_zip_b64"])
        # metadata = extract_metadata(file_tree, state.get("repository_name", "unknown"))
        # chunks = []
        # for file_path, content, language in file_tree:
        #     chunks.extend(chunk_file(file_path, content, language, state["config"]["max_chunk_lines"]))
        # chunks = await embed_chunks(chunks)
        # faiss_index = build_index(chunks)
        #
        # return {
        #     "metadata": metadata.model_dump(),
        #     "chunks": [c.model_dump() for c in chunks],
        #     "faiss_index": faiss_index,
        #     "progress": 20,
        # }

        raise NotImplementedError("ingest_node — Phase 2 implementation pending.")

    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest_node_error", job_id=job_id, error=str(exc))
        return {**state, "error": str(exc), "progress": state.get("progress", 0)}
