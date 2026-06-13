"""
agents/base_agent.py
=====================
Abstract base class for all LangChain analyzer agents.

Every analyzer must inherit BaseAgent and implement:
  - `_build_chain()` → LangChain Runnable
  - `run(chunks)` → List[FindingType]

Rules:
  - Agents obtain their LLM via `services.llm_client.get_llm_client()`
  - Agents load their prompt template from `prompts/<name>.jinja2`
  - Agents use LangChain's `PydanticOutputParser` for typed outputs
  - Agents must be stateless — no instance-level mutable state
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import structlog

from schemas import CodeChunk

logger = structlog.get_logger(__name__)

F = TypeVar("F")  # Finding type (BugFinding, SolidFinding, etc.)


class BaseAgent(ABC, Generic[F]):
    """
    Abstract LangChain analyzer agent.

    Subclasses define:
      - `name`           : str — used for logging and prompt template filename
      - `finding_class`  : Type[F] — Pydantic model for output parsing
      - `_build_chain()` : Return a LangChain Runnable (prompt | llm | parser)
      - `run(chunks)`    : Invoke chain and return List[F]
    """

    name: str = "base_agent"

    @abstractmethod
    def _build_chain(self) -> Any:
        """
        Build and return the LangChain chain for this analyzer.
        Called once on first run(); cached thereafter.

        Chain structure:
            PromptTemplate | get_llm_client() | PydanticOutputParser[finding_class]
        """

    @abstractmethod
    async def run(self, chunks: list[CodeChunk]) -> list[F]:
        """
        Run the analyzer against the provided code chunks.

        Args:
            chunks: List of CodeChunk objects from ingest_node.

        Returns:
            List of typed finding objects (e.g., List[BugFinding]).

        Raises:
            RuntimeError: If the LLM call fails after all retries.
        """

    def _load_prompt_template(self) -> str:
        """
        Load the Jinja2 prompt template for this analyzer.
        Template file: prompts/{self.name}.jinja2
        """
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "prompts" / f"{self.name}.jinja2"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path.read_text(encoding="utf-8")
