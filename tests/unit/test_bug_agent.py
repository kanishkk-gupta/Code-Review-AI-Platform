"""Unit tests for agents/bug_agent.py — offline rule engine."""
from __future__ import annotations

import asyncio

from agents.bug_agent import BugAgent, _scan_chunks
from schemas import CodeChunk, SupportedLanguage


def _chunk(content: str, file_path: str = "src/main.py") -> CodeChunk:
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


class TestBugAgentAst:
    def test_no_type_hint_false_positive(self):
        src = "def get(self, name: str, default: t.Any | None = None) -> t.Any:\n    return default\n"
        hits = _scan_chunks([_chunk(src)])
        assert not any(h.rule.rule_id == "BUG-001" for h in hits)

    def test_runtime_none_still_flagged(self):
        src = "def f():\n    return None.x\n"
        hits = _scan_chunks([_chunk(src)])
        assert any(h.rule.rule_id == "BUG-001" for h in hits)

    def test_client_get_not_flagged(self):
        src = 'def f(client):\n    rv = client.get("/").data\n    return rv\n'
        hits = _scan_chunks([_chunk(src, file_path="tests/test_app.py")])
        assert not any(h.rule.rule_id == "BUG-002" for h in hits)

    def test_dict_get_chain_still_flagged(self):
        src = "def f(d):\n    return d.get('k').upper()\n"
        hits = _scan_chunks([_chunk(src)])
        assert any(h.rule.rule_id == "BUG-002" for h in hits)

    def test_finding_has_quality_fields(self):
        src = "def f():\n    return None.x\n"
        findings = _run(BugAgent().run([_chunk(src)], llm_confirm=False))
        assert findings[0].evidence
        assert findings[0].reasoning
        assert findings[0].confidence_level in ("High", "Medium", "Low")
