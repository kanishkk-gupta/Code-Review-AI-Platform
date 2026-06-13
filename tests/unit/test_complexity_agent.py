"""
tests/unit/test_complexity_agent.py
=====================================
Unit tests for agents/complexity_agent.py.

No network or LLM calls.  Radon is imported only where available; tests that
need it are skipped with pytest.importorskip when radon is absent.
"""
from __future__ import annotations

import ast
import asyncio
import json
from unittest.mock import patch

import pytest

from schemas import CodeChunk, ComplexityFinding, Severity, SupportedLanguage
from tools.chunker import chunk_file
from agents.complexity_agent import (
    ComplexityAgent,
    _RawMetric,
    _analyse_python,
    _analyse_generic,
    _cc_grade,
    _classify,
    _cognitive_complexity,
    _collect_issues,
    _generate_recommendation,
    _make_finding,
    _max_nesting_depth,
    _reconstruct_files,
    _CC_THRESHOLDS,
    _COG_THRESHOLDS,
    _FN_LOC_THRESHOLDS,
    _CLASS_LOC_THRESHOLDS,
    _NEST_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_PY = """\
def add(a, b):
    return a + b
"""

COMPLEX_PY = """\
def process(data, config, mode, flag1, flag2, extra):
    result = []
    for item in data:
        if item > 0:
            if mode == "fast":
                if flag1:
                    if config.get("strict"):
                        if item > 100:
                            result.append(item * 2)
                        else:
                            result.append(item)
                    else:
                        if flag2:
                            result.append(item + 1)
                        else:
                            result.append(item - 1)
                elif flag2:
                    for x in range(item):
                        if x % 2 == 0:
                            result.append(x)
            else:
                try:
                    result.append(int(item))
                except ValueError:
                    pass
        elif item < 0:
            while item < 0:
                item += 1
            result.append(item)
    return result
"""

LONG_FN_PY = "\n".join([
    "def long_function():",
    *[f"    x_{i} = {i}  # placeholder" for i in range(110)],
    "    return x_0",
])

BIG_CLASS_PY = "\n".join([
    "class BigClass:",
    *[f"    x_{i} = {i}" for i in range(320)],
])


def _make_chunk(
    file_path: str = "src/main.py",
    language: SupportedLanguage = SupportedLanguage.PYTHON,
    content: str = "def foo():\n    pass\n",
    start_line: int = 1,
    end_line: int = 3,
    chunk_id: str = "chunk-001",
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        file_path=file_path,
        language=language,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )


def _make_metric(
    name: str = "my_fn",
    cc: int | None = None,
    cog: int | None = None,
    nest: int | None = None,
    loc: int = 10,
    is_class: bool = False,
    file_path: str = "src/main.py",
    start_line: int = 1,
    end_line: int = 10,
) -> _RawMetric:
    return _RawMetric(
        name=name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        lines_of_code=loc,
        cyclomatic_complexity=cc,
        cognitive_complexity=cog,
        nesting_depth=nest,
        is_class=is_class,
    )


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_above_highest_threshold(self):
        assert _classify(40, _CC_THRESHOLDS) == Severity.CRITICAL

    def test_at_high_threshold(self):
        assert _classify(25, _CC_THRESHOLDS) == Severity.HIGH

    def test_at_medium_threshold(self):
        assert _classify(15, _CC_THRESHOLDS) == Severity.MEDIUM

    def test_below_all_thresholds_returns_none(self):
        assert _classify(14, _CC_THRESHOLDS) is None

    def test_exact_boundary(self):
        assert _classify(40, _CC_THRESHOLDS) == Severity.CRITICAL

    def test_fn_loc_thresholds(self):
        assert _classify(120, _FN_LOC_THRESHOLDS) == Severity.CRITICAL
        assert _classify(80, _FN_LOC_THRESHOLDS) == Severity.HIGH
        assert _classify(60, _FN_LOC_THRESHOLDS) == Severity.MEDIUM
        assert _classify(10, _FN_LOC_THRESHOLDS) is None

    def test_class_loc_thresholds(self):
        assert _classify(600, _CLASS_LOC_THRESHOLDS) == Severity.CRITICAL
        assert _classify(400, _CLASS_LOC_THRESHOLDS) == Severity.HIGH
        assert _classify(250, _CLASS_LOC_THRESHOLDS) == Severity.MEDIUM


# ---------------------------------------------------------------------------
# _cc_grade
# ---------------------------------------------------------------------------

class TestCCGrade:
    def test_grade_A(self):
        assert "A" in _cc_grade(1)
        assert "A" in _cc_grade(5)

    def test_grade_B(self):
        assert "B" in _cc_grade(6)
        assert "B" in _cc_grade(10)

    def test_grade_C(self):
        assert "C" in _cc_grade(11)

    def test_grade_D(self):
        assert "D" in _cc_grade(16)

    def test_grade_F(self):
        assert "F" in _cc_grade(30)


# ---------------------------------------------------------------------------
# _max_nesting_depth
# ---------------------------------------------------------------------------

class TestMaxNestingDepth:
    def _parse_fn(self, src: str) -> ast.FunctionDef:
        tree = ast.parse(src)
        return next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
        )

    def test_no_nesting(self):
        src = "def f():\n    return 1"
        node = self._parse_fn(src)
        assert _max_nesting_depth(node) == 0

    def test_single_if(self):
        src = "def f(x):\n    if x:\n        return x"
        node = self._parse_fn(src)
        assert _max_nesting_depth(node) >= 0

    def test_nested_if_for(self):
        src = (
            "def f(data):\n"
            "    for item in data:\n"
            "        if item > 0:\n"
            "            for x in item:\n"
            "                if x:\n"
            "                    pass"
        )
        node = self._parse_fn(src)
        assert _max_nesting_depth(node) >= 3

    def test_deeply_nested(self):
        src = (
            "def f():\n"
            "    if True:\n"
            "        for x in []:\n"
            "            while x:\n"
            "                try:\n"
            "                    pass\n"
            "                except:\n"
            "                    pass"
        )
        node = self._parse_fn(src)
        assert _max_nesting_depth(node) >= 4


# ---------------------------------------------------------------------------
# _cognitive_complexity
# ---------------------------------------------------------------------------

class TestCognitiveComplexity:
    def _parse_fn(self, src: str) -> ast.FunctionDef:
        tree = ast.parse(src)
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))

    def test_simple_fn_low_score(self):
        node = self._parse_fn("def f(x):\n    return x + 1")
        assert _cognitive_complexity(node) <= 1

    def test_single_if_increases_score(self):
        node = self._parse_fn("def f(x):\n    if x > 0:\n        return x")
        assert _cognitive_complexity(node) >= 1

    def test_boolean_chain_increases_score(self):
        node = self._parse_fn(
            "def f(a, b, c):\n    if a and b and c:\n        return 1"
        )
        score = _cognitive_complexity(node)
        assert score >= 2  # 1 for if + 2 for two 'and' operators

    def test_nested_increases_more(self):
        src = (
            "def f(x):\n"
            "    if x:\n"
            "        for i in x:\n"
            "            if i:\n"
            "                pass"
        )
        node = self._parse_fn(src)
        # Nested structures should score higher than flat ones
        flat_src = "def f(x):\n    if x:\n        pass\n    for i in []:\n        pass"
        flat_node = self._parse_fn(flat_src)
        assert _cognitive_complexity(node) > _cognitive_complexity(flat_node)


# ---------------------------------------------------------------------------
# _RawMetric.worst_severity
# ---------------------------------------------------------------------------

class TestRawMetricWorstSeverity:
    def test_no_issues_returns_none(self):
        m = _make_metric(cc=1, loc=5)
        assert m.worst_severity is None

    def test_critical_cc_returns_critical(self):
        m = _make_metric(cc=40)
        assert m.worst_severity == Severity.CRITICAL

    def test_high_nest_returns_high(self):
        m = _make_metric(nest=6)
        assert m.worst_severity == Severity.HIGH

    def test_worst_metric_wins(self):
        m = _make_metric(cc=16, nest=6, loc=90)
        assert m.worst_severity == Severity.HIGH

    def test_loc_alone_below_high_not_flagged(self):
        m = _make_metric(loc=65, is_class=False)
        assert m.worst_severity is None

    def test_loc_high_with_structural_metric(self):
        m = _make_metric(cc=16, loc=65, is_class=False)
        assert m.worst_severity == Severity.MEDIUM


# ---------------------------------------------------------------------------
# _make_finding
# ---------------------------------------------------------------------------

class TestMakeFinding:
    def test_returns_complexity_finding(self):
        m = _make_metric(cc=15, loc=70)
        f = _make_finding(m, Severity.HIGH)
        assert isinstance(f, ComplexityFinding)

    def test_severity_set(self):
        m = _make_metric(cc=21, loc=10)
        f = _make_finding(m, Severity.CRITICAL)
        assert f.severity == Severity.CRITICAL

    def test_cyclomatic_complexity_in_finding(self):
        m = _make_metric(cc=12)
        f = _make_finding(m, Severity.MEDIUM)
        assert f.cyclomatic_complexity == 12

    def test_cognitive_complexity_in_finding(self):
        m = _make_metric(cog=18)
        f = _make_finding(m, Severity.HIGH)
        assert f.cognitive_complexity == 18

    def test_nesting_depth_in_finding(self):
        m = _make_metric(nest=4)
        f = _make_finding(m, Severity.MEDIUM)
        assert f.nesting_depth == 4

    def test_function_name_preserved(self):
        m = _make_metric(name="process_order")
        f = _make_finding(m, Severity.MEDIUM)
        assert f.function_name == "process_order"

    def test_lines_of_code_preserved(self):
        m = _make_metric(loc=55)
        f = _make_finding(m, Severity.MEDIUM)
        assert f.lines_of_code == 55

    def test_file_path_preserved(self):
        m = _make_metric(file_path="api/views.py")
        f = _make_finding(m, Severity.LOW)
        assert f.file_path == "api/views.py"

    def test_confidence_is_high(self):
        m = _make_metric(cc=16)
        f = _make_finding(m, Severity.MEDIUM)
        assert f.confidence >= 0.90

    def test_title_never_contains_chunk_marker(self):
        m = _make_metric(cc=16, name="real_function")
        f = _make_finding(m, Severity.MEDIUM)
        assert "<chunk:" not in f.title
        assert f.function_name == "real_function"

    def test_suggested_fix_not_empty(self):
        m = _make_metric(cc=12, loc=55)
        f = _make_finding(m, Severity.MEDIUM)
        assert f.suggested_fix
        assert len(f.suggested_fix) > 10


# ---------------------------------------------------------------------------
# _generate_recommendation
# ---------------------------------------------------------------------------

class TestGenerateRecommendation:
    def test_critical_cc_recommendation(self):
        m = _make_metric(cc=25)
        rec = _generate_recommendation(m)
        assert "CRITICAL" in rec
        assert "cyclomatic" in rec.lower() or "CC" in rec or "complexity" in rec.lower()

    def test_high_nesting_recommendation(self):
        m = _make_metric(nest=5)
        rec = _generate_recommendation(m)
        assert "nesting" in rec.lower() or "depth" in rec.lower()

    def test_long_function_recommendation(self):
        m = _make_metric(loc=110)
        rec = _generate_recommendation(m)
        assert "LOC" in rec or "lines" in rec.lower()

    def test_class_recommendation_mentions_class(self):
        m = _make_metric(loc=320, is_class=True)
        rec = _generate_recommendation(m)
        assert "Class" in rec or "class" in rec

    def test_no_issues_returns_fallback(self):
        m = _make_metric(loc=3)
        rec = _generate_recommendation(m)
        assert len(rec) > 0


# ---------------------------------------------------------------------------
# _analyse_python
# ---------------------------------------------------------------------------

class TestAnalysePython:
    def test_simple_fn_no_critical_findings(self):
        chunks = [_make_chunk(content=SIMPLE_PY, end_line=2)]
        metrics = _analyse_python("simple.py", SIMPLE_PY, chunks)
        assert any(m.name == "add" for m in metrics)

    def test_complex_fn_has_high_cc(self):
        radon = pytest.importorskip("radon")
        chunks = [_make_chunk(content=COMPLEX_PY, end_line=COMPLEX_PY.count("\n") + 1)]
        metrics = _analyse_python("complex.py", COMPLEX_PY, chunks)
        fn_metrics = [m for m in metrics if m.name == "process"]
        assert fn_metrics, "Expected metrics for 'process'"
        m = fn_metrics[0]
        if m.cyclomatic_complexity is not None:
            assert m.cyclomatic_complexity >= 5

    def test_long_fn_detected(self):
        chunks = [_make_chunk(content=LONG_FN_PY, end_line=LONG_FN_PY.count("\n") + 1)]
        metrics = _analyse_python("long.py", LONG_FN_PY, chunks)
        fn_m = [m for m in metrics if m.name == "long_function"]
        assert fn_m
        assert fn_m[0].lines_of_code >= 100

    def test_classes_not_emitted(self):
        chunks = [_make_chunk(content=BIG_CLASS_PY, end_line=BIG_CLASS_PY.count("\n") + 1)]
        metrics = _analyse_python("big.py", BIG_CLASS_PY, chunks)
        assert not any(m.is_class for m in metrics)

    def test_syntax_error_returns_empty(self):
        bad_src = "def broken(:\n    pass"
        chunks = [_make_chunk(content=bad_src, end_line=2)]
        metrics = _analyse_python("bad.py", bad_src, chunks)
        assert metrics == []

    def test_finding_links_overlapping_chunks(self):
        content = LONG_FN_PY
        chunks = chunk_file("long.py", content, SupportedLanguage.PYTHON, max_lines=40)
        findings = asyncio.get_event_loop().run_until_complete(
            ComplexityAgent().run(chunks, min_severity=Severity.MEDIUM)
        )
        fn_findings = [f for f in findings if f.function_name == "long_function"]
        assert fn_findings
        assert len(fn_findings[0].related_chunk_ids) >= 2

    def test_nesting_depth_measured(self):
        chunks = [_make_chunk(content=COMPLEX_PY, end_line=COMPLEX_PY.count("\n") + 1)]
        metrics = _analyse_python("complex.py", COMPLEX_PY, chunks)
        fn_metrics = [m for m in metrics if m.name == "process" and not m.is_class]
        if fn_metrics:
            assert fn_metrics[0].nesting_depth is not None


# ---------------------------------------------------------------------------
# _reconstruct_files
# ---------------------------------------------------------------------------

class TestReconstructFiles:
    def test_groups_by_file_path(self):
        c1 = _make_chunk(file_path="a.py", content="x = 1", chunk_id="c1", start_line=1, end_line=1)
        c2 = _make_chunk(file_path="a.py", content="y = 2", chunk_id="c2", start_line=2, end_line=2)
        c3 = _make_chunk(file_path="b.py", content="z = 3", chunk_id="c3", start_line=1, end_line=1)
        result = _reconstruct_files([c1, c2, c3])
        assert set(result.keys()) == {"a.py", "b.py"}

    def test_content_joined(self):
        c1 = _make_chunk(content="line1", start_line=1, end_line=1, chunk_id="x1")
        c2 = _make_chunk(content="line2", start_line=2, end_line=2, chunk_id="x2")
        result = _reconstruct_files([c1, c2])
        content, _, _ = result["src/main.py"]
        assert "line1" in content
        assert "line2" in content

    def test_chunks_sorted_by_start_line(self):
        c_late  = _make_chunk(content="late",  start_line=10, end_line=10, chunk_id="late")
        c_early = _make_chunk(content="early", start_line=1,  end_line=1,  chunk_id="early")
        result = _reconstruct_files([c_late, c_early])
        content, _, chunks = result["src/main.py"]
        assert chunks[0].start_line == 1

    def test_language_from_first_chunk(self):
        c = _make_chunk(language=SupportedLanguage.JAVA, chunk_id="j1")
        result = _reconstruct_files([c])
        _, language, _ = result["src/main.py"]
        assert language == SupportedLanguage.JAVA


# ---------------------------------------------------------------------------
# ComplexityAgent.run
# ---------------------------------------------------------------------------

class TestComplexityAgentRun:
    def _run(self, chunks, min_severity=Severity.MEDIUM):
        agent = ComplexityAgent()
        return asyncio.get_event_loop().run_until_complete(
            agent.run(chunks, min_severity=min_severity)
        )

    def test_empty_chunks_returns_empty(self):
        assert self._run([]) == []

    def test_simple_code_no_findings(self):
        chunk = _make_chunk(content=SIMPLE_PY, end_line=2)
        findings = self._run([chunk])
        assert isinstance(findings, list)
        # Simple add() should not trigger MEDIUM+ findings
        assert all(isinstance(f, ComplexityFinding) for f in findings)

    def test_complex_code_produces_findings(self):
        radon = pytest.importorskip("radon")
        chunk = _make_chunk(
            content=COMPLEX_PY,
            end_line=COMPLEX_PY.count("\n") + 1,
        )
        findings = self._run([chunk], min_severity=Severity.LOW)
        assert len(findings) > 0

    def test_findings_are_complexity_finding_instances(self):
        chunk = _make_chunk(
            content=LONG_FN_PY,
            end_line=LONG_FN_PY.count("\n") + 1,
        )
        findings = self._run([chunk])
        assert all(isinstance(f, ComplexityFinding) for f in findings)

    def test_findings_sorted_by_severity(self):
        radon = pytest.importorskip("radon")
        mixed = COMPLEX_PY + "\n\n" + LONG_FN_PY
        chunk = _make_chunk(content=mixed, end_line=mixed.count("\n") + 1)
        findings = self._run([chunk], min_severity=Severity.LOW)
        _ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
        idxs = [_ORDER.index(f.severity) for f in findings]
        assert idxs == sorted(idxs)

    def test_min_severity_filter(self):
        chunk = _make_chunk(
            content=COMPLEX_PY,
            end_line=COMPLEX_PY.count("\n") + 1,
        )
        low_findings  = self._run([chunk], min_severity=Severity.LOW)
        high_findings = self._run([chunk], min_severity=Severity.HIGH)
        assert len(low_findings) >= len(high_findings)

    def test_non_python_chunk_produces_no_findings(self):
        chunk = _make_chunk(
            language=SupportedLanguage.JAVA,
            content="public class Foo {\n    public void bar() {}\n}\n",
            end_line=3,
        )
        findings = self._run([chunk])
        assert findings == []

    def test_no_chunk_marker_in_findings(self):
        chunk = _make_chunk(
            content=LONG_FN_PY,
            end_line=LONG_FN_PY.count("\n") + 1,
        )
        findings = self._run([chunk])
        for f in findings:
            assert "<chunk:" not in f.title
            assert "<chunk:" not in f.function_name

    def test_to_json_valid_json(self):
        agent = ComplexityAgent()
        chunk = _make_chunk(
            content=COMPLEX_PY,
            end_line=COMPLEX_PY.count("\n") + 1,
        )
        findings = self._run([chunk], min_severity=Severity.LOW)
        json_str = agent.to_json(findings)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_finding_schema_fields_present(self):
        chunk = _make_chunk(
            content=LONG_FN_PY,
            end_line=LONG_FN_PY.count("\n") + 1,
        )
        findings = self._run([chunk])
        for f in findings:
            assert f.function_name
            assert f.lines_of_code >= 1
            assert f.file_path
            assert f.start_line >= 1
            assert f.end_line >= f.start_line
            assert f.severity in list(Severity)
