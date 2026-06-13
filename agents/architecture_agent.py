"""
agents/architecture_agent.py
=============================
ArchitectureAgent — Repository-level architectural anti-pattern detector.

Fundamentally different from file-level agents: operates on ALL CodeChunks
together to reason about the system structure, not individual files.

Pipeline
--------
Phase 1 — Graph Analysis (always, zero LLM cost):
    Builds an import dependency graph from chunk content, then applies
    graph-theoretic metrics to flag:

    GOD_CLASS         : Files with far more imports/dependents than the average
    CYCLIC_DEPENDENCY : Strongly-connected components (cycles) in the import graph
    TIGHT_COUPLING    : Fan-out > threshold (module imports too many others)
    LAYER_VIOLATION   : Cross-layer imports based on directory naming conventions
    BIG_BALL_OF_MUD   : Repository with no discernible layering / very high avg coupling

Phase 2 — LLM Synthesis (one call per repository):
    Sends the repository summary + import graph statistics to the LLM to
    detect FEATURE_ENVY, ANEMIC_DOMAIN, and subtle layer violations that
    require understanding of naming intent.

Signature difference from other agents
---------------------------------------
    run(chunks, metadata=None) — takes CodeChunks + optional RepositoryMetadata
    analyse_repository(chunks, metadata) — public alias, preferred for clarity

Output: List[ArchitectureFinding]
"""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from agents.base_agent import BaseAgent
from agents._scan_utils import is_non_production_path
from schemas import (
    ArchitectureFinding,
    ArchitectureSmell,
    CodeChunk,
    RepositoryMetadata,
    Severity,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------

_SEV_ORDER = [
    Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO
]


def _sev_idx(s: Severity) -> int:
    try:
        return _SEV_ORDER.index(Severity(s))
    except ValueError:
        return len(_SEV_ORDER)


# ---------------------------------------------------------------------------
# Thresholds  (tuned to typical mid-size Python repositories)
# ---------------------------------------------------------------------------

# Composite god-module score threshold (0–100)
_GOD_MODULE_SCORE_THRESHOLD = 72.0
# Tight coupling: fan-out (direct imports) threshold per file
_TIGHT_COUPLING_FAN_OUT = 10
# Big Ball of Mud: average fan-out across the whole repository
_BIG_BALL_OF_MUD_AVG_FANOUT = 7.0
# Layer violation: layer rank ordering (lower index = lower layer)
_LAYER_KEYWORDS: list[str] = [
    "model", "entity", "domain",     # rank 0 — data / domain
    "repository", "repo", "dao",     # rank 1 — persistence
    "service", "usecase", "manager", # rank 2 — business logic
    "controller", "handler", "view", # rank 3 — presentation
    "router", "api", "endpoint",     # rank 4 — transport
]
_LAYER_RANK: dict[str, int] = {kw: i for i, kw in enumerate(_LAYER_KEYWORDS)}


# ---------------------------------------------------------------------------
# Import graph construction
# ---------------------------------------------------------------------------

@dataclass
class _FileSummary:
    """Aggregated per-file information extracted from chunks."""
    file_path:        str
    language:         str
    imports:          list[str]     = field(default_factory=list)   # raw import targets
    classes:          int           = 0
    functions:        int           = 0
    total_lines:      int           = 0
    chunks:           list[CodeChunk] = field(default_factory=list)


def _extract_python_imports(content: str) -> list[str]:
    """Parse Python imports with the stdlib ast module."""
    targets: list[str] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return targets
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                targets.append(node.module.split(".")[0])
    return targets


_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE)
_JS_IMPORT_RE   = re.compile(
    r"""(?:import\s+.*?\s+from\s+['"]([\w./\-@]+)['"]|require\s*\(\s*['"]([\w./\-@]+)['"]\s*\))""",
    re.MULTILINE,
)
_CPP_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)


def _extract_imports(chunk: CodeChunk) -> list[str]:
    lang = str(chunk.language).lower()
    if "python" in lang:
        return _extract_python_imports(chunk.content)
    if "java" in lang:
        return [m.group(1).split(".")[-2] for m in _JAVA_IMPORT_RE.finditer(chunk.content)
                if len(m.group(1).split(".")) > 1]
    if "javascript" in lang or "typescript" in lang:
        results: list[str] = []
        for m in _JS_IMPORT_RE.finditer(chunk.content):
            mod = m.group(1) or m.group(2)
            if mod and not mod.startswith("."):
                results.append(mod.split("/")[0])
        return results
    if "c" in lang or "cpp" in lang:
        return [m.group(1).split("/")[-1].replace(".h", "") for m in _CPP_INCLUDE_RE.finditer(chunk.content)]
    return []


def _count_ast_entities(content: str) -> tuple[int, int]:
    """Return (class_count, function_count) via ast; (0,0) on parse error."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 0, 0
    classes   = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    functions = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return classes, functions


def _build_file_summaries(chunks: list[CodeChunk]) -> dict[str, _FileSummary]:
    """Aggregate production chunks by file_path into _FileSummary objects."""
    summaries: dict[str, _FileSummary] = {}
    for chunk in chunks:
        fp = chunk.file_path
        if is_non_production_path(fp):
            continue
        lang = str(chunk.language)
        if fp not in summaries:
            summaries[fp] = _FileSummary(file_path=fp, language=lang)
        summary = summaries[fp]
        summary.chunks.append(chunk)
        summary.imports.extend(_extract_imports(chunk))
        summary.total_lines += (chunk.end_line - chunk.start_line + 1)
        cls, fns = _count_ast_entities(chunk.content)
        summary.classes   += cls
        summary.functions += fns

    # Deduplicate import lists per file
    for s in summaries.values():
        s.imports = list(dict.fromkeys(s.imports))

    return summaries


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------

def _build_internal_graph(
    summaries: dict[str, _FileSummary],
) -> dict[str, set[str]]:
    """
    Build a directed import graph restricted to files inside the repository.
    Edge A → B means file A imports a module whose name is a prefix of B's
    path stem.
    """
    # Build name → file_path index (last path segment without extension)
    name_index: dict[str, str] = {}
    for fp in summaries:
        stem = re.sub(r"\.\w+$", "", fp.split("/")[-1])
        name_index[stem] = fp

    graph: dict[str, set[str]] = {fp: set() for fp in summaries}
    for fp, summary in summaries.items():
        for imp in summary.imports:
            target = name_index.get(imp)
            if target and target != fp:
                graph[fp].add(target)
    return graph


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """
    Detect strongly-connected components with Tarjan's algorithm.
    Returns only SCCs with more than one node (i.e., real cycles).
    """
    index_counter = [0]
    stack: list[str] = []
    lowlink:  dict[str, int]  = {}
    index:    dict[str, int]  = {}
    on_stack: dict[str, bool] = {}
    sccs:     list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v]    = index_counter[0]
        lowlink[v]  = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in graph.get(v, set()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)

    for v in graph:
        if v not in index:
            strongconnect(v)

    return sccs


def _file_layer_rank(file_path: str) -> Optional[int]:
    """Return the lowest matching layer rank for a file path, or None."""
    path_lower = file_path.lower()
    best: Optional[int] = None
    for kw, rank in _LAYER_RANK.items():
        if kw in path_lower:
            if best is None or rank < best:
                best = rank
    return best


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _is_framework_entrypoint(file_path: str, summary: _FileSummary) -> bool:
    """
    Legitimate framework facades (main.py, core.py) are high-LOC/high-API by design.
    """
    import os
    base = os.path.basename(file_path).lower()
    if base not in ("main.py", "core.py", "__init__.py"):
        return False
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) > 4:
        return False
    if summary.total_lines >= 400 and summary.functions >= 15:
        return True
    if base == "core.py" and summary.total_lines >= 250 and summary.functions >= 8:
        return True
    return False


def _god_module_score(
    summary: _FileSummary,
    fan_out: int,
    avg_imports: float,
) -> float:
    """Composite 0–100 score: LOC, fan-out, public API, responsibilities, dependencies."""
    unique_imports = len(set(summary.imports))
    loc = summary.total_lines or 1
    api = summary.functions + summary.classes
    responsibilities = summary.classes + max(1, unique_imports // 8)

    score = 0.0
    score += min(loc / 2000, 1.0) * 22
    score += min(fan_out / 18, 1.0) * 18
    score += min(unique_imports / max(avg_imports * 2.5, 1), 1.0) * 18
    score += min(api / 45, 1.0) * 22
    score += min(responsibilities / 12, 1.0) * 20
    return score


def _detect_god_classes(
    summaries: dict[str, _FileSummary],
    graph: dict[str, set[str]],
) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    if not summaries:
        return findings

    import_counts = [len(set(s.imports)) for s in summaries.values()]
    avg_imports = sum(import_counts) / len(import_counts)

    for fp, s in summaries.items():
        if _is_framework_entrypoint(fp, s):
            continue

        fan_out = len(graph.get(fp, set()))
        score = _god_module_score(s, fan_out, avg_imports)
        if score < _GOD_MODULE_SCORE_THRESHOLD:
            continue

        unique_imports = len(set(s.imports))
        findings.append(ArchitectureFinding(
            severity=Severity.HIGH,
            title=f"God Class / God Module detected: `{fp}`",
            description=(
                f"`{fp}` scores {score:.0f}/100 on composite god-module metrics "
                f"(LOC={s.total_lines}, fan-out={fan_out}, imports={unique_imports}, "
                f"public API={s.functions + s.classes}, responsibilities≈{s.classes}). "
                f"This file likely handles too many concerns."
            ),
            file_path=fp,
            start_line=1,
            end_line=s.total_lines or 1,
            suggested_fix=(
                "Decompose this file into smaller, single-purpose modules. "
                "Move each distinct concern (data access, business logic, "
                "networking, formatting) into its own file/class."
            ),
            confidence=0.78,
            smell=ArchitectureSmell.GOD_CLASS,
            affected_modules=[fp],
            impact_radius="System-wide",
        ))
    return findings


def _detect_cycles(
    graph: dict[str, set[str]],
) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    sccs = _find_cycles(graph)
    for scc in sccs:
        cycle_display = " → ".join(sorted(scc)[:6])
        if len(scc) > 6:
            cycle_display += f" … (+{len(scc)-6} more)"
        findings.append(ArchitectureFinding(
            severity=Severity.CRITICAL,
            title=f"Circular dependency cycle detected ({len(scc)} modules)",
            description=(
                f"A circular import cycle exists between {len(scc)} module(s): {cycle_display}. "
                "Circular dependencies make modules impossible to test in isolation, "
                "cause import errors in some language runtimes, and create tight coupling "
                "that resists refactoring."
            ),
            file_path=sorted(scc)[0],
            start_line=1,
            end_line=1,
            suggested_fix=(
                "Break the cycle by:\n"
                "  1. Extracting shared types/interfaces into a separate module "
                "that both can depend on.\n"
                "  2. Using dependency injection to remove the direct import.\n"
                "  3. Applying the Mediator or Event Bus pattern to decouple modules."
            ),
            confidence=0.95,
            smell=ArchitectureSmell.CYCLIC_DEPENDENCY,
            affected_modules=sorted(scc),
            impact_radius="System-wide",
        ))
    return findings


def _detect_tight_coupling(
    summaries: dict[str, _FileSummary],
    graph: dict[str, set[str]],
) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    for fp, deps in graph.items():
        if len(deps) >= _TIGHT_COUPLING_FAN_OUT:
            findings.append(ArchitectureFinding(
                severity=Severity.HIGH,
                title=f"High coupling in `{fp}` (fan-out: {len(deps)})",
                description=(
                    f"`{fp}` has a dependency fan-out of {len(deps)} internal modules. "
                    f"High coupling makes this file a change magnet — every change to a "
                    f"dependency risks breaking it."
                ),
                file_path=fp,
                start_line=1,
                end_line=summaries[fp].total_lines or 1,
                suggested_fix=(
                    "Reduce fan-out by:\n"
                    "  - Applying the Facade pattern to hide internal sub-modules.\n"
                    "  - Moving cohesive groups of functionality into sub-modules.\n"
                    "  - Depending on interfaces rather than concrete modules."
                ),
                confidence=0.78,
                smell=ArchitectureSmell.TIGHT_COUPLING,
                affected_modules=[fp] + sorted(deps)[:5],
                impact_radius="Module-wide",
            ))
    return findings


def _detect_layer_violations(
    summaries: dict[str, _FileSummary],
    graph: dict[str, set[str]],
) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    for fp, deps in graph.items():
        src_rank = _file_layer_rank(fp)
        if src_rank is None:
            continue
        for dep_fp in deps:
            dep_rank = _file_layer_rank(dep_fp)
            if dep_rank is None:
                continue
            # Higher-rank layer calling into a layer more than one level below
            # (e.g. controller → domain: allowed; controller → model: violation)
            # OR lower-rank calling higher-rank (upward dependency): violation
            if dep_rank > src_rank + 1:
                findings.append(ArchitectureFinding(
                    severity=Severity.HIGH,
                    title=f"Layer violation: `{fp}` skips layers to reach `{dep_fp}`",
                    description=(
                        f"File `{fp}` (layer rank {src_rank}) directly imports from "
                        f"`{dep_fp}` (layer rank {dep_rank}), skipping intermediate layers. "
                        "This violates the principle of layered architecture and creates "
                        "tight coupling across layer boundaries."
                    ),
                    file_path=fp,
                    start_line=1,
                    end_line=summaries[fp].total_lines or 1,
                    suggested_fix=(
                        "Route the call through the intermediate layer.\n"
                        "If the dependency is unavoidable, consider whether the layers "
                        "are correctly named/structured — the file may need to be "
                        "relocated or refactored."
                    ),
                    confidence=0.70,
                    smell=ArchitectureSmell.LAYER_VIOLATION,
                    affected_modules=[fp, dep_fp],
                    impact_radius="Module-wide",
                ))
    return findings


def _detect_big_ball_of_mud(
    summaries: dict[str, _FileSummary],
    graph: dict[str, set[str]],
) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    if len(summaries) < 5:   # too small to diagnose
        return findings

    fan_outs = [len(deps) for deps in graph.values()]
    avg_fanout = sum(fan_outs) / max(len(fan_outs), 1)

    # Check if there is any layer structure at all
    layered_files = sum(1 for fp in summaries if _file_layer_rank(fp) is not None)
    layer_ratio   = layered_files / len(summaries)

    if avg_fanout >= _BIG_BALL_OF_MUD_AVG_FANOUT and layer_ratio < 0.2:
        findings.append(ArchitectureFinding(
            severity=Severity.CRITICAL,
            title="Big Ball of Mud: no discernible layering with high average coupling",
            description=(
                f"The repository has an average import fan-out of {avg_fanout:.1f} "
                f"across {len(summaries)} files, yet only {layered_files} file(s) "
                f"({layer_ratio*100:.0f}%) follow recognizable layer naming conventions "
                f"(model/service/controller/repository). This indicates a 'Big Ball of Mud' "
                f"architecture with no clear separation of concerns."
            ),
            file_path="(repository-wide)",
            start_line=1,
            end_line=1,
            suggested_fix=(
                "Introduce explicit architectural layers:\n"
                "  - Separate data models, business logic, and I/O into distinct packages.\n"
                "  - Enforce import rules (e.g., flake8-import-order, ArchUnit).\n"
                "  - Gradually extract cohesive modules and establish clear boundaries."
            ),
            confidence=0.75,
            smell=ArchitectureSmell.BIG_BALL_OF_MUD,
            affected_modules=list(summaries.keys())[:10],
            impact_radius="System-wide",
        ))
    return findings


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

def _build_repo_summary_for_llm(
    summaries: dict[str, _FileSummary],
    graph: dict[str, set[str]],
    metadata: Optional[RepositoryMetadata],
) -> str:
    """Build a compact text summary of the repository for LLM context."""
    lines: list[str] = []
    if metadata:
        lines.append(f"Repository: {metadata.repository_name}")
        lines.append(f"Primary language: {metadata.primary_language}")
        lines.append(f"Total files: {metadata.total_files}, Total lines: {metadata.total_lines}")
        lines.append(f"Language breakdown: {metadata.language_breakdown}")
    lines.append(f"\nFiles analysed: {len(summaries)}")
    lines.append("\nTop files by import count:")
    top = sorted(summaries.items(), key=lambda x: len(x[1].imports), reverse=True)[:10]
    for fp, s in top:
        deps = sorted(graph.get(fp, set()))[:5]
        lines.append(f"  {fp}: {len(s.imports)} imports, {s.classes} classes, {s.functions} fns")
        if deps:
            lines.append(f"    → depends on: {', '.join(deps)}")
    return "\n".join(lines)


def _parse_llm_output(raw: Any, fallback_fp: str = "(repository-wide)") -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return findings
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            item.setdefault("file_path",    fallback_fp)
            item.setdefault("start_line",   1)
            item.setdefault("end_line",     1)
            item.setdefault("confidence",   0.80)
            item.setdefault("affected_modules", [])
            item.setdefault("impact_radius", "Module-wide")
            findings.append(ArchitectureFinding.model_validate(item))
        except Exception as exc:  # noqa: BLE001
            logger.warning("architecture_agent_parse_error", error=str(exc))
    return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deduplicate(findings: list[ArchitectureFinding]) -> list[ArchitectureFinding]:
    seen: set[tuple] = set()
    result: list[ArchitectureFinding] = []
    for f in findings:
        key = (f.file_path, str(f.smell), frozenset(f.affected_modules))
        if key not in seen:
            seen.add(key)
            result.append(f)
    return result


def _sort_findings(findings: list[ArchitectureFinding]) -> list[ArchitectureFinding]:
    return sorted(findings, key=lambda f: (_sev_idx(f.severity), f.file_path))


# ---------------------------------------------------------------------------
# ArchitectureAgent
# ---------------------------------------------------------------------------

class ArchitectureAgent(BaseAgent[ArchitectureFinding]):
    """
    Repository-level architectural anti-pattern detector.

    Unlike other agents this one operates on the FULL set of chunks together.
    It builds an import dependency graph and applies graph metrics before
    optionally invoking the LLM for higher-level synthesis.

    Preferred entry point::

        agent = ArchitectureAgent()
        findings = await agent.analyse_repository(chunks, metadata)

    The standard ``run(chunks)`` interface is also supported for compatibility
    with the analyze_node dispatcher.
    """

    name: str = "architecture_analysis"

    # ── LangChain chain ───────────────────────────────────────────────────

    def _build_chain(self) -> Any:
        """
        PromptTemplate | LLM | JsonOutputParser
        One call per repository (not per chunk).
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from services.llm_client import get_llm_client

        parser = JsonOutputParser()
        tmpl   = self._load_prompt_template()
        prompt = PromptTemplate(
            template=tmpl,
            input_variables=[
                "repo_summary", "repository_name",
                "total_files", "primary_language",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | get_llm_client() | parser

    # ── Primary entry points ──────────────────────────────────────────────

    async def analyse_repository(
        self,
        chunks: list[CodeChunk],
        metadata: Optional[RepositoryMetadata] = None,
        *,
        llm_synthesize: bool = True,
    ) -> list[ArchitectureFinding]:
        """
        Full repository-level analysis.

        Args:
            chunks         : All CodeChunk objects from ingest_node.
            metadata       : RepositoryMetadata from ingest_node (optional but recommended).
            llm_synthesize : If True, invoke LLM for FEATURE_ENVY / ANEMIC_DOMAIN
                             detection and higher-level reasoning.

        Returns:
            Deduplicated, severity-sorted list of ArchitectureFinding.
        """
        if not chunks:
            logger.info("architecture_agent_run", chunks=0, findings=0)
            return []

        logger.info(
            "architecture_agent_run",
            chunks=len(chunks),
            repo=metadata.repository_name if metadata else "unknown",
        )

        # Build internal data structures
        summaries = _build_file_summaries(chunks)
        graph     = _build_internal_graph(summaries)

        logger.info(
            "architecture_agent_graph_built",
            files=len(summaries),
            edges=sum(len(v) for v in graph.values()),
        )

        # Phase 1 — graph analysis
        graph_findings: list[ArchitectureFinding] = []
        graph_findings.extend(_detect_god_classes(summaries, graph))
        graph_findings.extend(_detect_cycles(graph))
        graph_findings.extend(_detect_tight_coupling(summaries, graph))
        graph_findings.extend(_detect_layer_violations(summaries, graph))
        graph_findings.extend(_detect_big_ball_of_mud(summaries, graph))

        logger.info("architecture_agent_graph_findings", count=len(graph_findings))

        if not llm_synthesize:
            return _sort_findings(_deduplicate(graph_findings))

        # Phase 2 — LLM synthesis
        llm_findings = await self._llm_synthesize(summaries, graph, metadata)
        logger.info("architecture_agent_llm_findings", count=len(llm_findings))

        merged = _deduplicate(graph_findings + llm_findings)
        result = _sort_findings(merged)
        logger.info("architecture_agent_complete", total_findings=len(result))
        return result

    async def run(
        self,
        chunks: list[CodeChunk],
        metadata: Optional[RepositoryMetadata] = None,
    ) -> list[ArchitectureFinding]:
        """Standard BaseAgent interface — delegates to analyse_repository."""
        return await self.analyse_repository(chunks, metadata)

    # ── LLM synthesis ──────────────────────────────────────────────────────

    async def _llm_synthesize(
        self,
        summaries: dict[str, _FileSummary],
        graph: dict[str, set[str]],
        metadata: Optional[RepositoryMetadata],
    ) -> list[ArchitectureFinding]:
        findings: list[ArchitectureFinding] = []
        try:
            chain = self._build_chain()
        except Exception as exc:  # noqa: BLE001
            logger.warning("architecture_agent_llm_unavailable", error=str(exc))
            return findings

        repo_summary = _build_repo_summary_for_llm(summaries, graph, metadata)
        try:
            raw = await chain.ainvoke({
                "repo_summary":     repo_summary,
                "repository_name":  metadata.repository_name if metadata else "unknown",
                "total_files":      len(summaries),
                "primary_language": str(metadata.primary_language) if metadata else "unknown",
            })
            findings.extend(_parse_llm_output(raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("architecture_agent_llm_call_failed", error=str(exc))

        return findings

    # ── Serialization ──────────────────────────────────────────────────────

    def to_json(self, findings: list[ArchitectureFinding]) -> str:
        return json.dumps(
            [f.model_dump(mode="json") for f in findings],
            indent=2,
            ensure_ascii=False,
        )
