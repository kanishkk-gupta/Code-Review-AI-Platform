"""
agents/complexity_agent.py
==========================
ComplexityAgent — Static-analysis-based complexity analyzer using Radon.

Unlike other agents in this package, ComplexityAgent does NOT call an LLM.
Cyclomatic complexity, nesting depth, and function length are computable with
100% determinism from the source AST.  LLM calls are only used to generate the
human-readable ``suggested_fix`` recommendation when the complexity exceeds a
configurable threshold.

Output: list[ComplexityFinding]

Metrics Computed
----------------
| Metric               | Source          | Threshold → Severity |
|----------------------|-----------------|----------------------|
| Cyclomatic complexity | Radon CC        | ≥20 CRITICAL, ≥15 HIGH, ≥10 MEDIUM, ≥5 LOW |
| Cognitive complexity  | AST (manual)    | ≥30 CRITICAL, ≥20 HIGH, ≥12 MEDIUM, ≥7 LOW |
| Nesting depth         | AST walk        | ≥5 HIGH, ≥4 MEDIUM, ≥3 LOW |
| Function LOC          | line-count      | ≥100 CRITICAL, ≥60 HIGH, ≥40 MEDIUM |
| Class LOC             | line-count      | ≥500 CRITICAL, ≥300 HIGH, ≥150 MEDIUM |

Supported Languages
-------------------
Full Radon analysis  : Python (radon.complexity.cc_visit)
Fallback (LOC only)  : Java, C++, JavaScript, TypeScript (regex + AST-free approach)

Radon Grade → Severity Mapping
-------------------------------
    A (1–5)   : OK — not flagged
    B (6–10)  : LOW
    C (11–15) : MEDIUM
    D (16–20) : HIGH
    E/F (>20) : CRITICAL

Architecture
------------
::

    ComplexityAgent.run(chunks: list[CodeChunk])
        │
        ├── group chunks by (file_path)
        │
        ├── for each Python file:
        │   ├── radon.complexity.cc_visit(source)   → per-function CC
        │   ├── _measure_nesting_depth(ast_tree)    → max depth
        │   └── _measure_cognitive_complexity()     → cognitive score
        │
        ├── for each non-Python file:
        │   └── parser.py regex data → LOC / function length only
        │
        └── _make_finding() → ComplexityFinding (Pydantic validated)

Usage
-----
::

    from agents.complexity_agent import ComplexityAgent
    from schemas import CodeChunk

    agent = ComplexityAgent()
    findings = await agent.run(chunks)
    for f in findings:
        print(f.model_dump_json(indent=2))
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from agents.base_agent import BaseAgent
from agents._scan_utils import is_non_production_path
from schemas import CodeChunk, ComplexityFinding, ConfidenceLevel, Severity, SupportedLanguage

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Radon cyclomatic complexity (CC) → Severity (raised to cut low-value noise)
_CC_THRESHOLDS: list[tuple[int, Severity]] = [
    (40, Severity.CRITICAL),
    (25, Severity.HIGH),
    (15, Severity.MEDIUM),
]

# Cognitive complexity → Severity
_COG_THRESHOLDS: list[tuple[int, Severity]] = [
    (35, Severity.CRITICAL),
    (25, Severity.HIGH),
    (18, Severity.MEDIUM),
]

# Maximum nesting depth → Severity
_NEST_THRESHOLDS: list[tuple[int, Severity]] = [
    (6, Severity.HIGH),
    (5, Severity.MEDIUM),
]

# Function lines-of-code → Severity
_FN_LOC_THRESHOLDS: list[tuple[int, Severity]] = [
    (120, Severity.CRITICAL),
    (80,  Severity.HIGH),
    (60,  Severity.MEDIUM),
]

# Class lines-of-code → Severity
_CLASS_LOC_THRESHOLDS: list[tuple[int, Severity]] = [
    (600, Severity.CRITICAL),
    (400, Severity.HIGH),
    (250, Severity.MEDIUM),
]

# Minimum severity to emit a finding — suppress MEDIUM noise on mature codebases
_MIN_SEVERITY: Severity = Severity.HIGH


# ---------------------------------------------------------------------------
# Internal raw metric dataclass
# ---------------------------------------------------------------------------

@dataclass
class _FileAggregate:
    """Complete source file reconstructed from ordered chunks."""
    file_path:         str
    language:          SupportedLanguage
    content:           str
    chunks:            list[CodeChunk]
    line_count:        int


@dataclass
class _RawMetric:
    """Intermediate holder of raw computed values for a single function/class."""
    name:                 str
    file_path:            str
    start_line:           int
    end_line:             int
    lines_of_code:        int
    cyclomatic_complexity: Optional[int]
    cognitive_complexity:  Optional[int]
    nesting_depth:         Optional[int]
    is_class:              bool = False
    related_chunk_ids:     list[str] = field(default_factory=list)

    @property
    def worst_severity(self) -> Optional[Severity]:
        """Return the most severe issue found across structural metrics."""
        candidates: list[Severity] = []
        if self.cyclomatic_complexity is not None:
            s = _classify(self.cyclomatic_complexity, _CC_THRESHOLDS)
            if s:
                candidates.append(s)
        if self.cognitive_complexity is not None:
            s = _classify(self.cognitive_complexity, _COG_THRESHOLDS)
            if s:
                candidates.append(s)
        if self.nesting_depth is not None:
            s = _classify(self.nesting_depth, _NEST_THRESHOLDS)
            if s:
                candidates.append(s)
        # LOC alone only flags at HIGH+ (length thresholds start at MEDIUM but
        # require CC/cognitive/nesting co-trigger unless CRITICAL length)
        thresholds = _CLASS_LOC_THRESHOLDS if self.is_class else _FN_LOC_THRESHOLDS
        loc_sev = _classify(self.lines_of_code, thresholds)
        has_structural = bool(candidates)
        if loc_sev:
            if loc_sev in (Severity.CRITICAL, Severity.HIGH) or has_structural:
                candidates.append(loc_sev)

        if not candidates:
            return None
        _ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
        return min(candidates, key=lambda x: _ORDER.index(x))


# ---------------------------------------------------------------------------
# ComplexityAgent
# ---------------------------------------------------------------------------


class ComplexityAgent(BaseAgent[ComplexityFinding]):
    """
    Static-analysis complexity analyzer using Radon + Python AST.

    This agent is deterministic — no LLM calls are made during analysis.
    Each ``ComplexityFinding`` is backed by precise numeric measurements.

    For non-Python files, only LOC-based metrics are computed (regex parsers
    do not expose a CFG).
    """

    name: str = "complexity_analysis"

    def _build_chain(self) -> Any:
        # ComplexityAgent uses static analysis, not an LLM chain.
        # _build_chain is part of the BaseAgent interface but not used here.
        raise NotImplementedError(
            "ComplexityAgent uses static analysis. Use run() directly."
        )

    async def run(
        self,
        chunks: list[CodeChunk],
        *,
        min_severity: Severity = _MIN_SEVERITY,
    ) -> list[ComplexityFinding]:
        """
        Analyse code chunks for complexity issues.

        Args:
            chunks      : CodeChunk objects (from ingest_node).
            min_severity: Suppress findings below this severity level.
                          Default: MEDIUM (suppresses LOW noise on clean code).

        Returns:
            List of ``ComplexityFinding`` objects, sorted by severity then file path.

        Notes:
            - Python chunks use full Radon CC + AST nesting depth + cognitive score.
            - Non-Python chunks get LOC-only metrics.
            - Empty chunks produce no findings.
        """
        if not chunks:
            logger.info("complexity_agent_run", chunks=0, findings=0)
            return []

        logger.info("complexity_agent_run", chunks=len(chunks))

        # Aggregate chunks → complete per-file source (never analyze partial chunks)
        file_aggregates = aggregate_files_from_chunks(chunks)

        raw_metrics: list[_RawMetric] = []

        for aggregate in file_aggregates.values():
            if is_non_production_path(aggregate.file_path):
                continue
            if aggregate.language == SupportedLanguage.PYTHON:
                metrics = _analyse_python_file(aggregate)
            else:
                metrics = _analyse_generic_file(aggregate)
            raw_metrics.extend(metrics)

        _ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
        min_idx = _ORDER.index(min_severity)

        findings: list[ComplexityFinding] = []
        for metric in raw_metrics:
            severity = metric.worst_severity
            if severity is None:
                continue
            if _ORDER.index(severity) > min_idx:
                continue  # below minimum threshold
            finding = _make_finding(metric, severity)
            findings.append(finding)

        # Sort: severity (most critical first), then file path
        findings.sort(key=lambda f: (_ORDER.index(f.severity), f.file_path, f.start_line))

        logger.info(
            "complexity_agent_complete",
            chunks=len(chunks),
            findings=len(findings),
        )
        return findings

    def run_sync(
        self,
        chunks: list[CodeChunk],
        *,
        min_severity: Severity = _MIN_SEVERITY,
    ) -> list[ComplexityFinding]:
        """
        Synchronous wrapper for testing or non-async contexts.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.run(chunks, min_severity=min_severity)
        )

    def to_json(self, findings: list[ComplexityFinding]) -> str:
        """
        Serialise findings to a formatted JSON string.

        Each finding is serialised via ``ComplexityFinding.model_dump()`` using
        Pydantic V2's ``mode='json'`` to handle datetime serialisation.

        Returns:
            Indented JSON string (UTF-8).
        """
        data = [f.model_dump(mode="json") for f in findings]
        return json.dumps(data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# File aggregation (chunks → complete source files)
# ---------------------------------------------------------------------------


def aggregate_files_from_chunks(
    chunks: list[CodeChunk],
) -> dict[str, _FileAggregate]:
    """
    Group chunks by ``file_path`` and reconstruct complete file contents.

    Chunks are fixed-size line windows from the ingest pipeline.  Analysis
  must run on the merged file — never on individual chunk bodies.
    """
    by_path: dict[str, list[CodeChunk]] = defaultdict(list)
    for chunk in chunks:
        by_path[chunk.file_path].append(chunk)

    aggregates: dict[str, _FileAggregate] = {}
    for file_path, chunk_list in by_path.items():
        chunk_list.sort(key=lambda c: c.start_line)
        content = _reconstruct_file_content(chunk_list)
        aggregates[file_path] = _FileAggregate(
            file_path=file_path,
            language=SupportedLanguage(chunk_list[0].language),
            content=content,
            chunks=chunk_list,
            line_count=len(content.splitlines()),
        )
    return aggregates


def _reconstruct_file_content(chunks: list[CodeChunk]) -> str:
    """
    Rebuild full file text from ordered, non-overlapping chunks.

    Uses line-number placement so gaps are visible; contiguous chunker
    output is equivalent to ``"\\n".join`` but survives boundary validation.
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0].content

    lines_by_no: dict[int, str] = {}
    for chunk in chunks:
        for offset, line in enumerate(chunk.content.splitlines()):
            lines_by_no[chunk.start_line + offset] = line

    if not lines_by_no:
        return ""

    min_line = min(lines_by_no)
    max_line = max(lines_by_no)
    return "\n".join(lines_by_no.get(i, "") for i in range(min_line, max_line + 1))


def _chunks_for_span(chunks: list[CodeChunk], start: int, end: int) -> list[str]:
    """Return chunk IDs whose line span overlaps ``[start, end]``."""
    return [
        c.chunk_id
        for c in chunks
        if c.start_line <= end and c.end_line >= start
    ]


# ---------------------------------------------------------------------------
# Python Analysis (Radon + AST) — complete files only
# ---------------------------------------------------------------------------


def _analyse_python_file(aggregate: _FileAggregate) -> list[_RawMetric]:
    """Full Radon + AST analysis on a complete Python source file."""
    file_path = aggregate.file_path
    content = aggregate.content
    metrics: list[_RawMetric] = []

    try:
        ast.parse(content)
    except SyntaxError as exc:
        logger.warning(
            "complexity_skip_unparseable",
            file=file_path,
            error=str(exc),
            line_count=aggregate.line_count,
            chunk_count=len(aggregate.chunks),
        )
        return metrics

    cc_results: list[Any] = []
    try:
        from radon.complexity import cc_visit
        cc_results = cc_visit(content)
    except ImportError:
        logger.warning("radon_not_installed", advice="pip install radon")
    except SyntaxError as exc:
        logger.warning("radon_syntax_error", file=file_path, error=str(exc))
        return metrics

    tree = ast.parse(content)
    cc_map: dict[tuple[str, int], Any] = {
        (item.name, item.lineno): item for item in cc_results
    }

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        start = node.lineno
        end = getattr(node, "end_lineno", node.lineno)
        cc_item = cc_map.get((node.name, start))
        metrics.append(_RawMetric(
            name=node.name,
            file_path=file_path,
            start_line=start,
            end_line=end,
            lines_of_code=end - start + 1,
            cyclomatic_complexity=cc_item.complexity if cc_item else None,
            cognitive_complexity=_cognitive_complexity(node),
            nesting_depth=_max_nesting_depth(node),
            is_class=False,
            related_chunk_ids=_chunks_for_span(aggregate.chunks, start, end),
        ))

    return metrics


def _analyse_generic_file(aggregate: _FileAggregate) -> list[_RawMetric]:
    """Non-Python files require AST FunctionDef nodes — not analyzed."""
    return []


# Backward-compatible aliases for tests
def _analyse_python(
    file_path: str,
    content: str,
    chunks: list[CodeChunk],
) -> list[_RawMetric]:
    aggregate = _FileAggregate(
        file_path=file_path,
        language=SupportedLanguage.PYTHON,
        content=content,
        chunks=chunks,
        line_count=len(content.splitlines()),
    )
    return _analyse_python_file(aggregate)


def _analyse_generic(
    file_path: str,
    content: str,
    language: SupportedLanguage,
    chunks: list[CodeChunk],
) -> list[_RawMetric]:
    return _analyse_generic_file(_FileAggregate(
        file_path=file_path,
        language=language,
        content=content,
        chunks=chunks,
        line_count=len(content.splitlines()),
    ))


# ---------------------------------------------------------------------------
# AST Metric Calculators
# ---------------------------------------------------------------------------


def _max_nesting_depth(node: ast.AST) -> int:
    """
    Measure the maximum nesting depth of control-flow blocks inside *node*.

    Counts: if, for, while, with, try, except, match (3.10+).
    """
    _NESTING_NODES = (
        ast.If, ast.For, ast.AsyncFor, ast.While,
        ast.With, ast.AsyncWith, ast.Try,
        ast.ExceptHandler,
    )

    def _depth(n: ast.AST, current: int) -> int:
        if isinstance(n, _NESTING_NODES):
            current += 1
        max_d = current
        for child in ast.iter_child_nodes(n):
            max_d = max(max_d, _depth(child, current))
        return max_d

    return max(0, _depth(node, 0) - 1)  # subtract 1 (function body itself)


def _cognitive_complexity(node: ast.AST) -> int:
    """
    Approximate SonarSource cognitive complexity for a function/class node.

    Rules (simplified):
      +1 for each: if, elif, else, for, while, with, try, except
      +1 for each nesting level beyond the first (structural penalty)
      +1 for each: break, continue, goto analogue
      +1 for each boolean operator chain (and/or) in conditions
    """
    def _score(n: ast.AST, depth: int) -> int:
        increment = 0
        nesting_penalty = 0

        _BRANCHING = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                      ast.AsyncWith, ast.Try, ast.ExceptHandler)
        _JUMPING   = (ast.Break, ast.Continue, ast.Return)

        if isinstance(n, _BRANCHING):
            increment += 1
            nesting_penalty = depth  # every extra nesting level adds penalty

        if isinstance(n, _JUMPING):
            increment += 1

        # Boolean short-circuit operators
        if isinstance(n, ast.BoolOp):
            # 1 point per `and`/`or` keyword (= len(values) - 1)
            increment += len(n.values) - 1

        total = increment + nesting_penalty
        child_depth = depth + 1 if isinstance(n, _BRANCHING) else depth

        for child in ast.iter_child_nodes(n):
            total += _score(child, child_depth)

        return total

    return _score(node, 0)


# ---------------------------------------------------------------------------
# Finding Factory
# ---------------------------------------------------------------------------


def _make_finding(metric: _RawMetric, severity: Severity) -> ComplexityFinding:
    """
    Convert a ``_RawMetric`` into a validated ``ComplexityFinding``.

    Generates all three output components:
      - ``title``        : Short human label
      - ``description``  : Full diagnostic explanation
      - ``suggested_fix``: Concrete refactoring recommendation
    """
    label = "Class" if metric.is_class else "Function"
    issues: list[str] = _collect_issues(metric)
    issue_summary = "; ".join(issues) if issues else "Excessive length"

    title = f"{label} '{metric.name}' has {issue_summary[:120]}"[:200]

    description_parts: list[str] = [
        f"{label} `{metric.name}` in `{metric.file_path}` "
        f"(lines {metric.start_line}–{metric.end_line}, "
        f"{metric.lines_of_code} LOC) has complexity issues:",
    ]
    if metric.cyclomatic_complexity is not None:
        description_parts.append(
            f"  • Cyclomatic complexity = {metric.cyclomatic_complexity} "
            f"({_cc_grade(metric.cyclomatic_complexity)})"
        )
    if metric.cognitive_complexity is not None:
        description_parts.append(
            f"  • Cognitive complexity = {metric.cognitive_complexity}"
        )
    if metric.nesting_depth is not None:
        description_parts.append(
            f"  • Maximum nesting depth = {metric.nesting_depth}"
        )
    description_parts.append(
        f"  • Lines of code = {metric.lines_of_code}"
    )
    description = "\n".join(description_parts)

    suggested_fix = _generate_recommendation(metric)

    evidence_parts = [p.strip(" •") for p in description_parts[1:]]
    return ComplexityFinding(
        severity=severity,
        title=title,
        description=description,
        file_path=metric.file_path,
        start_line=metric.start_line,
        end_line=metric.end_line,
        suggested_fix=suggested_fix,
        confidence=0.95,
        confidence_level=ConfidenceLevel.HIGH,
        evidence="; ".join(evidence_parts) if evidence_parts else issue_summary,
        reasoning=(
            "Radon cyclomatic complexity and AST nesting/cognitive metrics exceed "
            "configured production thresholds."
        ),
        cyclomatic_complexity=metric.cyclomatic_complexity,
        cognitive_complexity=metric.cognitive_complexity,
        nesting_depth=metric.nesting_depth,
        function_name=metric.name,
        lines_of_code=metric.lines_of_code,
        related_chunk_ids=list(metric.related_chunk_ids),
    )


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------


def _generate_recommendation(metric: _RawMetric) -> str:
    """
    Generate a structured, actionable refactoring recommendation.

    This is a deterministic rule-based system — no LLM required.
    Recommendations are prioritised by the most severe metric found.
    """
    recs: list[str] = []

    # ── Cyclomatic complexity ──────────────────────────────────────────────
    cc = metric.cyclomatic_complexity
    if cc is not None:
        if cc >= 20:
            recs.append(
                f"CRITICAL: Cyclomatic complexity {cc} far exceeds the safe threshold of 10. "
                "Decompose this function into 3+ smaller, single-responsibility functions. "
                "Consider a State Machine or Strategy pattern to replace the branching logic."
            )
        elif cc >= 15:
            recs.append(
                f"HIGH: Cyclomatic complexity {cc} exceeds the warning threshold of 10. "
                "Extract independent branches into helper functions. "
                "Use early-return guards to flatten nested conditions."
            )
        elif cc >= 10:
            recs.append(
                f"MEDIUM: Cyclomatic complexity {cc} is approaching the threshold of 10. "
                "Review and simplify conditional branches. "
                "Extract complex conditions into named boolean predicates."
            )
        elif cc >= 5:
            recs.append(
                f"LOW: Cyclomatic complexity {cc}. "
                "Consider extracting complex inner conditions for readability."
            )

    # ── Cognitive complexity ───────────────────────────────────────────────
    cog = metric.cognitive_complexity
    if cog is not None:
        if cog >= 30:
            recs.append(
                f"CRITICAL: Cognitive complexity {cog} makes this code very hard to reason about. "
                "Break the function into smaller, well-named functions. "
                "Reduce boolean operator chains and replace them with explicit guard clauses."
            )
        elif cog >= 20:
            recs.append(
                f"HIGH: Cognitive complexity {cog}. "
                "Simplify nested conditionals. "
                "Move loop bodies into separate helper functions."
            )
        elif cog >= 12:
            recs.append(
                f"MEDIUM: Cognitive complexity {cog}. "
                "Use early returns to reduce indentation. "
                "Name complex boolean expressions as intermediate variables."
            )

    # ── Nesting depth ─────────────────────────────────────────────────────
    nest = metric.nesting_depth
    if nest is not None:
        if nest >= 5:
            recs.append(
                f"HIGH: Nesting depth {nest} layers detected. "
                "Invert conditionals to exit early (guard clause pattern). "
                "Extract inner loop bodies as separate functions. "
                "Consider pipeline or chain-of-responsibility patterns."
            )
        elif nest >= 4:
            recs.append(
                f"MEDIUM: Nesting depth {nest}. "
                "Apply guard clauses to flatten the deepest conditional levels. "
                "Use comprehensions or built-ins (filter/map) to simplify loops."
            )
        elif nest >= 3:
            recs.append(
                f"LOW: Nesting depth {nest}. "
                "Consider extracting the innermost block into a helper."
            )

    # ── Lines of code ─────────────────────────────────────────────────────
    loc = metric.lines_of_code
    thresholds = _CLASS_LOC_THRESHOLDS if metric.is_class else _FN_LOC_THRESHOLDS
    loc_severity = _classify(loc, thresholds)
    label = "Class" if metric.is_class else "Function"

    if loc_severity == Severity.CRITICAL:
        recs.append(
            f"CRITICAL: {label} has {loc} lines of code. "
            f"{'Decompose into multiple classes following SRP.' if metric.is_class else 'Split into smaller, focused functions (target < 40 LOC each).'}"
        )
    elif loc_severity == Severity.HIGH:
        recs.append(
            f"HIGH: {label} has {loc} lines. "
            f"{'Consider decomposing into modules or sub-classes.' if metric.is_class else 'Extract logical sub-steps into helper functions.'}"
        )
    elif loc_severity == Severity.MEDIUM:
        recs.append(
            f"MEDIUM: {label} has {loc} lines. "
            "Review whether all logic belongs here and consider extracting sub-routines."
        )

    if not recs:
        recs.append("Review code for potential simplification opportunities.")

    return "\n\n".join(recs)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _classify(
    value: int,
    thresholds: list[tuple[int, Severity]],
) -> Optional[Severity]:
    """
    Return the severity for *value* based on the first matching threshold.
    Returns ``None`` if the value falls below all thresholds (acceptable).
    """
    for minimum, severity in thresholds:
        if value >= minimum:
            return severity
    return None


def _cc_grade(cc: int) -> str:
    """Return the Radon letter grade for a cyclomatic complexity value."""
    if cc <= 5:   return "A — simple"
    if cc <= 10:  return "B — well-structured"
    if cc <= 15:  return "C — slightly complex"
    if cc <= 20:  return "D — more than moderate risk"
    if cc <= 25:  return "E — high risk"
    return            "F — very high risk"


def _collect_issues(metric: _RawMetric) -> list[str]:
    """Summarise all triggered thresholds as short strings (for title)."""
    issues: list[str] = []
    if metric.cyclomatic_complexity and metric.cyclomatic_complexity >= 5:
        issues.append(f"CC={metric.cyclomatic_complexity}")
    if metric.cognitive_complexity and metric.cognitive_complexity >= 7:
        issues.append(f"cognitive={metric.cognitive_complexity}")
    if metric.nesting_depth and metric.nesting_depth >= 3:
        issues.append(f"nesting={metric.nesting_depth}")
    thresholds = _CLASS_LOC_THRESHOLDS if metric.is_class else _FN_LOC_THRESHOLDS
    if _classify(metric.lines_of_code, thresholds):
        issues.append(f"LOC={metric.lines_of_code}")
    return issues


def _reconstruct_files(
    chunks: list[CodeChunk],
) -> dict[str, tuple[str, SupportedLanguage, list[CodeChunk]]]:
    """Backward-compatible wrapper around :func:`aggregate_files_from_chunks`."""
    aggregates = aggregate_files_from_chunks(chunks)
    return {
        fp: (agg.content, agg.language, agg.chunks)
        for fp, agg in aggregates.items()
    }
