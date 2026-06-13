"""Unit tests for architecture agent production-path filtering."""
from __future__ import annotations

import asyncio

from agents.architecture_agent import ArchitectureAgent, _build_file_summaries
from schemas import CodeChunk, SupportedLanguage


def _chunk(file_path: str, content: str = "import os\n") -> CodeChunk:
    lines = content.count("\n") + 1
    return CodeChunk(
        file_path=file_path,
        language=SupportedLanguage.PYTHON,
        content=content,
        start_line=1,
        end_line=lines,
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestArchitectureProductionFilter:
    def test_summaries_exclude_tests(self):
        chunks = [
            _chunk("src/app.py", "import flask\n"),
            _chunk("tests/test_basic.py", "import pytest\nimport os\n" * 20),
        ]
        summaries = _build_file_summaries(chunks)
        assert "src/app.py" in summaries
        assert "tests/test_basic.py" not in summaries

    def test_no_god_class_from_test_file(self):
        heavy_imports = "\n".join(f"import mod_{i}" for i in range(20))
        chunks = [_chunk("tests/test_cli.py", heavy_imports + "\n")]
        findings = _run(ArchitectureAgent().analyse_repository(chunks, llm_synthesize=False))
        assert not any("tests/" in f.file_path for f in findings)
        assert not any("God Class" in f.title for f in findings)
