# CodeGuardian AI Architecture

This document describes the structural architecture of the Version 1.0 deterministic engine.

## 1. Top-Level Flow
CodeGuardian AI employs a completely asynchronous architecture powered by FastAPI and LangGraph.

1. **Client Submission:** The client submits a repository URL or a base64 encoded ZIP file to the `/review` endpoint.
2. **Job Queue:** The API instantly returns a HTTP 202 Accepted, generating a unique `job_id` and storing a `PENDING` state in the thread-safe `InMemoryJobStore`.
3. **Background Execution:** A FastAPI `BackgroundTask` fires off the LangGraph `ReviewState` machine.

## 2. LangGraph Workflow Topology
The orchestrator is defined in `graph/workflow.py`. It is a directed state graph.

- **`repository_processor`**: Resolves the input. If it is a GitHub URL, it clones the repository to a temporary directory using `GitPython` and filters for source code.
- **`chunk_generator`**: Reads raw source files and slices them into fixed-size `CodeChunk` objects, retaining exact line numbers. It also pre-computes semantic embeddings and builds a FAISS index (Note: retrieval is reserved for Phase 2).
- **`parallel_analysis`**: The core fan-out node. It uses `asyncio.gather` to launch 5 independent agents simultaneously.
- **`report_agent_node`**: The fan-in node. It collects all findings from the analyzers, calculates a weighted composite quality score (0-100), and generates a Markdown and PDF report via Jinja2 templates.

## 3. Analysis Agents
In Version 1.0, agents are completely deterministic and stateless. They rely on two main techniques:

- **Regex Engines:** Fast, line-by-line scanning for common anti-patterns (e.g., hardcoded secrets, weak hashes).
- **Python AST Traversal:** Advanced parsing of Python `.py` files to detect vulnerabilities like SQL injection (via Taint Analysis) and structural defects without executing the code.
- **Radon Integration:** The `ComplexityAgent` leverages the `radon` library to compute McCabe Cyclomatic Complexity and Halstead metrics.

## 4. Data Contracts (Pydantic V2)
Strict type safety is enforced at runtime. All objects (API requests, CodeChunks, Findings, JobStatus) inherit from `schemas._BaseSchema` which utilizes Pydantic V2's `ConfigDict` to strip whitespace and forbid unknown extra attributes.
