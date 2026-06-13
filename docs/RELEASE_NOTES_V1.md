# Release Notes: v1.0-rule-engine

## Overview
CodeGuardian AI Version 1.0 represents the first stable release of our asynchronous, multi-agent static analysis platform. It establishes the core engine capable of parsing, chunking, and evaluating entire repositories against strict, deterministic rulesets.

## Features
- **LangGraph Orchestration**: A resilient state-machine execution graph (`repository_processor -> chunk_generator -> parallel_analysis -> report_agent_node`).
- **Concurrent Analysis**: `asyncio` powered fan-out execution of 5 independent analyzers:
  - **BugAgent**: Regex and AST-based static analysis.
  - **SecurityAgent**: Python AST traversal and taint analysis.
  - **SolidAgent**: AST-based object-oriented heuristic analysis.
  - **ArchitectureAgent**: Cross-module import graph tracking for cyclic dependencies.
  - **ComplexityAgent**: `radon` integration for Cyclomatic and Cognitive complexity calculation.
- **FastAPI Backend**: Fully async endpoints with `InMemoryJobStore` polling and Pydantic V2 data validation.
- **Ingestion Tools**: Safely clone GitHub repositories (`GitPython`) or extract base64 encoded ZIP files, with line-preserving AST chunkers.
- **Reporting**: Automated generation of scored Markdown and PDF codebase audits.

## Deprecations & Stubs
- The LLM verification layer (`services/llm_client.py`) is explicitly stubbed out (`NotImplementedError`) to ensure strict determinism in V1.
- The `ChunkRetriever` module and FAISS semantic similarity searches are disabled in V1.

## Known Issues
- Extremely high false-positive rates on test files (flagging mock credentials as security threats).
- Regex collisions misidentifying JavaScript operators as division-by-zero vulnerabilities.
