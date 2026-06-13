"""Unit tests for SOLID agent production-path filtering."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from agents.solid_agent import SolidAgent, _scan_heuristics
from schemas import CodeChunk, Severity, SupportedLanguage


def _chunk(file_path: str, content: str) -> CodeChunk:
    lines = content.count("\n") + 1
    return CodeChunk(
        file_path=file_path,
        language=SupportedLanguage.PYTHON,
        content=content,
        start_line=1,
        end_line=lines,
    )


DIP_TEST_SRC = """\
from io import BytesIO

class Handler:
    def __init__(self):
        self.buf = BytesIO()
"""


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSolidProductionFilter:
    def test_heuristics_skip_test_files(self):
        hits = _scan_heuristics([_chunk("tests/test_io.py", DIP_TEST_SRC)])
        assert hits == []

    def test_heuristics_run_on_production_files(self):
        hits = _scan_heuristics([_chunk("src/handler.py", DIP_TEST_SRC)])
        assert any(h.rule.rule_id == "DIP-001" for h in hits)

    def test_agent_no_findings_from_tests(self):
        findings = _run(SolidAgent().run(
            [_chunk("tests/test_utils.py", DIP_TEST_SRC)],
            llm_confirm=False,
        ))
        assert findings == []

    def test_heuristics_skip_unit_tests_dir(self):
        hits = _scan_heuristics([_chunk("unit_tests/helper.py", DIP_TEST_SRC)])
        assert hits == []

    def test_heuristics_skip_static_paths(self):
        hits = _scan_heuristics([
            _chunk("static/admin/js/vendor.py", DIP_TEST_SRC),
        ])
        assert hits == []


class TestSolidLlmProductionFilter:
    def test_llm_analyze_skips_non_production_paths(self):
        agent = SolidAgent()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=[])

        chunks = [
            _chunk("tests/test_utils.py", "class A: pass"),
            _chunk("unit_tests/helper.py", "class B: pass"),
            _chunk("static/admin/js/app.js", "class C: pass"),
            _chunk("src/service.py", "class S: pass"),
        ]

        async def _run_llm() -> None:
            with patch.object(agent, "_build_chain", return_value=mock_chain):
                await agent._llm_analyze(chunks)

        _run(_run_llm())
        assert mock_chain.ainvoke.call_count == 1
        assert mock_chain.ainvoke.call_args.args[0]["file_path"] == "src/service.py"
