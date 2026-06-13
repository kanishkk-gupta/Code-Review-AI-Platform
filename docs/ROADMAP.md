# Product Roadmap (Version 2.0)

CodeGuardian AI Version 1.0 establishes the deterministic foundation: a robust LangGraph orchestrator, resilient file ingestion, strict Pydantic data contracts, and rule-based analyzers.

Version 2.0 will introduce semantic intelligence and Retrieval-Augmented Generation (RAG).

## Phase A: Intelligence Layer (LLM Synthesis)
The primary focus of Phase A is enabling the LLM capabilities currently stubbed out in `services/llm_client.py`.

- **LLM Agent Confirmation:** Agents will submit high-confidence rule hits to an OpenAI/Ollama model for semantic verification. The LLM will assess the context of the hit and determine if it is a true positive or a benign false positive (e.g., a test file mock).
- **Report Executive Summary:** The `ReportAgent` will dynamically generate a natural language executive summary detailing the holistic health of the codebase, rather than relying on a deterministic template.

## Phase B: Context-Aware Retrieval (RAG)
While Version 1.0 actively generates `SentenceTransformer` embeddings and builds a FAISS index during the `chunk_generator` phase, this vector store is currently destroyed without being queried.

- **Wire the Retriever:** Activate `rag/retriever.py` to allow agents to perform similarity searches against the FAISS index.
- **Cross-File Context:** Agents will retrieve imported classes, inherited parents, and function definitions from distinct files to understand execution paths that span the repository.

## Phase C: UX & Stabilization
- **Aggressive Heuristics:** Implement stricter filtering in `_scan_utils.py` to completely ignore `tests/`, `docs/`, and `examples/` directories for security vulnerabilities.
- **Streamlit Dashboard:** Wire the existing Streamlit UI prototype into a fully supported, production-ready frontend for visualizing reports and interacting with findings.
