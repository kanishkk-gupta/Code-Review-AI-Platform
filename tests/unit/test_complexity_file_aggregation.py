"""Tests for ComplexityAgent file-level aggregation and repository coverage."""
from __future__ import annotations

import ast
import asyncio
import logging
import tempfile

import pytest

from agents.complexity_agent import (
    ComplexityAgent,
    aggregate_files_from_chunks,
    _reconstruct_file_content,
)
from schemas import CodeChunk, Severity, SupportedLanguage
from tools.chunker import chunk_file
from tools.github_tools import cleanup_repo, fetch_repo, list_source_files


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestChunkContentPreservation:
    def test_indented_line_at_chunk_boundary_preserved(self):
        """Chunk 2+ must not lose leading whitespace (Pydantic strip regression)."""
        lines = ["# header\n"] + [f"    line_{i} = {i}\n" for i in range(1, 100)]
        content = "".join(lines)
        chunks = chunk_file("mod.py", content, SupportedLanguage.PYTHON, max_lines=80)
        assert len(chunks) >= 2
        assert chunks[1].content.splitlines()[0].startswith("    ")

    def test_reconstructed_file_matches_original_and_parses(self):
        lines = ["def outer():\n"] + [f"    x_{i} = {i}\n" for i in range(120)]
        content = "".join(lines)
        chunks = chunk_file("mod.py", content, SupportedLanguage.PYTHON, max_lines=80)
        rebuilt = _reconstruct_file_content(chunks)
        assert rebuilt.splitlines() == content.splitlines()
        ast.parse(rebuilt)


class TestAggregateFilesFromChunks:
    def test_groups_by_file_path(self):
        c1 = CodeChunk(
            file_path="a.py", language=SupportedLanguage.PYTHON,
            content="def a():\n    pass", start_line=1, end_line=2,
        )
        c2 = CodeChunk(
            file_path="b.py", language=SupportedLanguage.PYTHON,
            content="def b():\n    pass", start_line=1, end_line=2,
        )
        aggs = aggregate_files_from_chunks([c1, c2])
        assert set(aggs) == {"a.py", "b.py"}


@pytest.mark.integration
class TestComplexityRepoBenchmark:
    """Requires network — clones requests/flask and verifies complexity analysis."""

    @pytest.fixture
    def caplog_info(self, caplog):
        caplog.set_level(logging.WARNING, logger="agents.complexity_agent")
        return caplog

    def _chunks_for_repo(self, url: str) -> list[CodeChunk]:
        repo = fetch_repo(url, tempfile.mkdtemp())
        chunks: list[CodeChunk] = []
        try:
            for sf in list_source_files(repo):
                if not sf.rel_path.endswith(".py"):
                    continue
                chunks.extend(
                    chunk_file(sf.rel_path, sf.content, SupportedLanguage.PYTHON, 80)
                )
            return chunks
        finally:
            cleanup_repo(repo)

    def test_requests_nonzero_complexity_no_skip_logs(self, caplog_info):
        chunks = self._chunks_for_repo("https://github.com/psf/requests")
        findings = _run(
            ComplexityAgent().run(chunks, min_severity=Severity.MEDIUM)
        )
        prod = [f for f in findings if not f.file_path.startswith("tests")]
        assert len(prod) > 0, "expected production complexity findings for requests"
        assert "<chunk:" not in " ".join(f.function_name for f in findings)
        skips = [r for r in caplog_info.records if r.message == "complexity_skip_unparseable"]
        prod_skips = [r for r in skips if "tests" not in r.file_path]
        assert prod_skips == [], f"unexpected skip logs: {prod_skips[:3]}"

    def test_flask_nonzero_complexity_no_skip_logs(self, caplog_info):
        chunks = self._chunks_for_repo("https://github.com/pallets/flask")
        findings = _run(
            ComplexityAgent().run(chunks, min_severity=Severity.MEDIUM)
        )
        prod = [f for f in findings if "src/flask" in f.file_path.replace("\\", "/")]
        assert len(prod) > 0, "expected src/flask complexity findings"
        skips = [r for r in caplog_info.records if r.message == "complexity_skip_unparseable"]
        prod_skips = [
            r for r in skips
            if "src/flask" in r.file_path.replace("\\", "/")
        ]
        assert prod_skips == [], f"unexpected skip logs: {prod_skips[:3]}"
