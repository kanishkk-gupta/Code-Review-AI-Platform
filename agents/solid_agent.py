"""
agents/solid_agent.py
======================
SolidAgent — Hybrid SOLID principle violation detector.

Phase 1 — AST Heuristics (always, zero LLM cost):
    SRP : Classes with too many methods or mixed I/O + business logic.
    OCP : isinstance/type() chains suggesting hardcoded type dispatch.
    ISP : Classes implementing large numbers of methods (fat interface proxy).
    DIP : Direct instantiation of concrete classes inside other classes.

Phase 2 — LLM Analysis (per-chunk, escalated hits + LSP):
    Confirms heuristic hits and handles LSP (Liskov Substitution Principle),
    which requires semantic reasoning the heuristics cannot provide.

Output: List[SolidFinding]  (canonical schemas.py model)
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from agents.base_agent import BaseAgent
from agents._scan_utils import classify_file_role, downgrade_severity, is_non_production_path
from schemas import CodeChunk, Severity, SolidFinding, SolidPrinciple

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Severity ordering helper
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
# Heuristic rule definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _HRule:
    rule_id:          str
    principle:        SolidPrinciple
    severity:         Severity
    title_template:   str
    description_tmpl: str
    fix_template:     str
    refactor_hint:    str


_HRULES: dict[str, _HRule] = {

    # ── SRP ──────────────────────────────────────────────────────────────────
    "SRP-001": _HRule(
        rule_id="SRP-001",
        principle=SolidPrinciple.SINGLE_RESPONSIBILITY,
        severity=Severity.MEDIUM,
        title_template="Class `{name}` has too many methods ({count}) — SRP violation",
        description_tmpl=(
            "Class `{name}` defines {count} methods, suggesting it handles multiple "
            "responsibilities. Classes with many methods are harder to test, maintain, "
            "and extend without breaking unrelated behavior."
        ),
        fix_template=(
            "Identify cohesive groups of methods that belong together and extract each "
            "group into its own focused class. Apply the Single Responsibility Principle: "
            "each class should have exactly one reason to change."
        ),
        refactor_hint="Split class into smaller, focused classes",
    ),
    "SRP-002": _HRule(
        rule_id="SRP-002",
        principle=SolidPrinciple.SINGLE_RESPONSIBILITY,
        severity=Severity.LOW,
        title_template="Function `{name}` is excessively long ({lines} lines) — SRP risk",
        description_tmpl=(
            "Function `{name}` spans {lines} lines, which usually indicates it is "
            "doing too many things. Long functions violate SRP at the method level "
            "and become difficult to unit-test in isolation."
        ),
        fix_template=(
            "Extract coherent sub-tasks into well-named helper functions. "
            "Aim for functions under 30 lines that do one thing clearly."
        ),
        refactor_hint="Extract helper functions",
    ),

    # ── OCP ──────────────────────────────────────────────────────────────────
    "OCP-001": _HRule(
        rule_id="OCP-001",
        principle=SolidPrinciple.OPEN_CLOSED,
        severity=Severity.HIGH,
        title_template="Type-dispatch chain in `{name}` is closed to extension — OCP violation",
        description_tmpl=(
            "`{name}` uses a chain of isinstance()/type() checks ({count} branches) to "
            "dispatch behavior by concrete type. Adding a new type requires modifying this "
            "function, violating the Open/Closed Principle."
        ),
        fix_template=(
            "Replace type-dispatch chains with polymorphism: define a common interface/ABC "
            "with a method each subclass overrides. The dispatching code becomes a single "
            "virtual call without modification."
        ),
        refactor_hint="Replace isinstance chain with polymorphism / strategy pattern",
    ),
    "OCP-002": _HRule(
        rule_id="OCP-002",
        principle=SolidPrinciple.OPEN_CLOSED,
        severity=Severity.MEDIUM,
        title_template="Hardcoded string-switch in `{name}` — OCP violation",
        description_tmpl=(
            "`{name}` branches on a string/enum value ({count} branches) with hardcoded "
            "if/elif logic. Extending behavior requires editing this function directly."
        ),
        fix_template=(
            "Use a registry dict mapping keys to handlers/strategies, or apply the "
            "Command / Strategy pattern so new behavior can be added without modifying "
            "existing code."
        ),
        refactor_hint="Replace string-switch with registry / strategy pattern",
    ),

    # ── ISP ──────────────────────────────────────────────────────────────────
    "ISP-001": _HRule(
        rule_id="ISP-001",
        principle=SolidPrinciple.INTERFACE_SEGREGATION,
        severity=Severity.MEDIUM,
        title_template="Class `{name}` is a fat interface ({count} abstract methods)",
        description_tmpl=(
            "`{name}` declares {count} abstract methods, forcing all implementors to "
            "provide all of them even if they only use a subset. This violates the "
            "Interface Segregation Principle."
        ),
        fix_template=(
            "Break the interface into smaller, role-specific interfaces (mixins or ABCs). "
            "Clients depend only on the interface they actually use."
        ),
        refactor_hint="Split into smaller role-specific interfaces",
    ),

    # ── DIP ──────────────────────────────────────────────────────────────────
    "DIP-001": _HRule(
        rule_id="DIP-001",
        principle=SolidPrinciple.DEPENDENCY_INVERSION,
        severity=Severity.HIGH,
        title_template="Direct instantiation of `{dep}` inside `{name}` — DIP violation",
        description_tmpl=(
            "`{name}` directly instantiates `{dep}` with `{dep}(...)` in its body. "
            "High-level classes should depend on abstractions, not on concrete "
            "implementations; this makes the code hard to test and swap."
        ),
        fix_template=(
            "Inject the dependency through the constructor or a factory:\n"
            "  def __init__(self, dep: AbstractDep):\n"
            "      self._dep = dep\n"
            "Use dependency injection or a DI container to wire implementations at "
            "application startup."
        ),
        refactor_hint="Inject dependency via constructor / factory",
    ),
}

# Thresholds
_SRP_METHOD_THRESHOLD  = 10   # methods per class
_SRP_FUNCTION_LINES    = 60   # lines per function
_OCP_ISINSTANCE_MIN    = 3    # isinstance branches to flag
_OCP_STRING_BRANCH_MIN = 4    # string if/elif branches to flag
_ISP_ABSTRACT_MIN      = 8    # abstract methods in one class


# ---------------------------------------------------------------------------
# AST Heuristic scan
# ---------------------------------------------------------------------------

@dataclass
class _HHit:
    rule:                    _HRule
    chunk:                   CodeChunk
    violated_name:           str          # class or function name
    extra:                   dict = field(default_factory=dict)


def _scan_heuristics(chunks: list[CodeChunk]) -> list[_HHit]:
    """Run all AST-based heuristics. Only works for Python production chunks."""
    hits: list[_HHit] = []
    for chunk in chunks:
        if is_non_production_path(chunk.file_path):
            continue
        if chunk.language not in ("python", "py") and str(chunk.language) != "python":
            # Heuristics are Python AST-based; skip non-Python for Phase 1
            continue
        try:
            tree = ast.parse(chunk.content)
        except SyntaxError:
            continue
        hits.extend(_check_srp(chunk, tree))
        hits.extend(_check_ocp(chunk, tree))
        hits.extend(_check_isp(chunk, tree))
        hits.extend(_check_dip(chunk, tree))
    return hits


def _check_srp(chunk: CodeChunk, tree: ast.Module) -> list[_HHit]:
    hits: list[_HHit] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(methods) >= _SRP_METHOD_THRESHOLD:
                hits.append(_HHit(
                    rule=_HRULES["SRP-001"],
                    chunk=chunk,
                    violated_name=node.name,
                    extra={"count": len(methods), "name": node.name},
                ))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            length = end - node.lineno + 1
            if length >= _SRP_FUNCTION_LINES:
                hits.append(_HHit(
                    rule=_HRULES["SRP-002"],
                    chunk=chunk,
                    violated_name=node.name,
                    extra={"lines": length, "name": node.name},
                ))
    return hits


def _check_ocp(chunk: CodeChunk, tree: ast.Module) -> list[_HHit]:
    hits: list[_HHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        isinstance_count = 0
        string_branches  = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if (isinstance(func, ast.Name) and func.id == "isinstance") or \
                   (isinstance(func, ast.Attribute) and func.attr == "isinstance"):
                    isinstance_count += 1
            if isinstance(child, ast.If):
                # look for `x == "string"` or `x == SomeEnum.VALUE`
                t = child.test
                if isinstance(t, ast.Compare):
                    for comp in t.comparators:
                        if isinstance(comp, (ast.Constant,)):
                            if isinstance(comp.value, str):
                                string_branches += 1
        if isinstance_count >= _OCP_ISINSTANCE_MIN:
            hits.append(_HHit(
                rule=_HRULES["OCP-001"],
                chunk=chunk,
                violated_name=name,
                extra={"count": isinstance_count, "name": name},
            ))
        if string_branches >= _OCP_STRING_BRANCH_MIN:
            hits.append(_HHit(
                rule=_HRULES["OCP-002"],
                chunk=chunk,
                violated_name=name,
                extra={"count": string_branches, "name": name},
            ))
    return hits


def _check_isp(chunk: CodeChunk, tree: ast.Module) -> list[_HHit]:
    hits: list[_HHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        abstract_methods = []
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in item.decorator_list:
                deco_name = ""
                if isinstance(deco, ast.Name):
                    deco_name = deco.id
                elif isinstance(deco, ast.Attribute):
                    deco_name = deco.attr
                if deco_name == "abstractmethod":
                    abstract_methods.append(item.name)
        if len(abstract_methods) >= _ISP_ABSTRACT_MIN:
            hits.append(_HHit(
                rule=_HRULES["ISP-001"],
                chunk=chunk,
                violated_name=node.name,
                extra={"count": len(abstract_methods), "name": node.name},
            ))
    return hits


def _check_dip(chunk: CodeChunk, tree: ast.Module) -> list[_HHit]:
    """Flag direct instantiation of non-builtin classes inside class methods."""
    _BUILTIN_CALLS = frozenset({
        "list", "dict", "set", "tuple", "str", "int", "float", "bool",
        "bytes", "bytearray", "frozenset", "range", "enumerate", "zip",
        "map", "filter", "reversed", "sorted", "print", "len", "open",
        "Exception", "ValueError", "TypeError", "KeyError", "RuntimeError",
        "NotImplementedError", "AttributeError", "IndexError",
    })
    hits: list[_HHit] = []
    for cls_node in ast.walk(tree):
        if not isinstance(cls_node, ast.ClassDef):
            continue
        for item in cls_node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != "__init__":
                continue
            for child in ast.walk(item):
                if isinstance(child, ast.Assign):
                    for val_node in ast.walk(child):
                        if isinstance(val_node, ast.Call):
                            func = val_node.func
                            dep_name = ""
                            if isinstance(func, ast.Name):
                                dep_name = func.id
                            elif isinstance(func, ast.Attribute):
                                dep_name = func.attr
                            if (dep_name
                                    and dep_name[0].isupper()
                                    and dep_name not in _BUILTIN_CALLS):
                                hits.append(_HHit(
                                    rule=_HRULES["DIP-001"],
                                    chunk=chunk,
                                    violated_name=cls_node.name,
                                    extra={"dep": dep_name, "name": cls_node.name},
                                ))
    return hits


# ---------------------------------------------------------------------------
# Hit → Finding
# ---------------------------------------------------------------------------

def _hit_to_finding(hit: _HHit) -> SolidFinding:
    rule   = hit.rule
    extra  = hit.extra
    desc   = rule.description_tmpl.format(**extra)
    title  = rule.title_template.format(**extra)
    fix    = rule.fix_template

    severity   = rule.severity
    confidence = 0.72
    reasoning: str | None = None
    role = classify_file_role(hit.chunk.file_path)
    if role in ('test', 'docs', 'example'):
        severity   = Severity.INFO
        confidence = 0.30
        reasoning  = (
            f"Finding in {role} file — informational only; test/example code "
            "often uses patterns that are acceptable in non-production contexts."
        )

    return SolidFinding(
        severity=severity,
        title=title,
        description=desc,
        file_path=hit.chunk.file_path,
        start_line=hit.chunk.start_line,
        end_line=hit.chunk.end_line,
        suggested_fix=fix,
        confidence=confidence,
        reasoning=reasoning,
        principle=rule.principle,
        violated_class_or_function=hit.violated_name,
        refactor_hint=rule.refactor_hint,
    )


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------

def _parse_llm_output(raw: Any, chunk: CodeChunk) -> list[SolidFinding]:
    findings: list[SolidFinding] = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        logger.warning("solid_agent_unexpected_llm_output", type=type(raw).__name__)
        return findings
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            item.setdefault("file_path",   chunk.file_path)
            item.setdefault("start_line",  chunk.start_line)
            item.setdefault("end_line",    chunk.end_line)
            item.setdefault("confidence",  0.85)
            item.setdefault("violated_class_or_function", "unknown")
            findings.append(SolidFinding.model_validate(item))
        except Exception as exc:  # noqa: BLE001
            logger.warning("solid_agent_parse_error", error=str(exc))
    return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deduplicate(findings: list[SolidFinding]) -> list[SolidFinding]:
    seen: set[tuple] = set()
    result: list[SolidFinding] = []
    for f in findings:
        key = (f.file_path, f.start_line, str(f.principle), f.violated_class_or_function)
        if key not in seen:
            seen.add(key)
            result.append(f)
    return result


def _sort_findings(findings: list[SolidFinding]) -> list[SolidFinding]:
    return sorted(findings, key=lambda f: (_sev_idx(f.severity), f.file_path))


# ---------------------------------------------------------------------------
# SolidAgent
# ---------------------------------------------------------------------------

class SolidAgent(BaseAgent[SolidFinding]):
    """
    Hybrid SOLID violation detector.

    Phase 1 — AST heuristics (SRP, OCP, ISP, DIP): always runs.
    Phase 2 — LLM analysis (LSP + confirmation of all principles): per chunk.
    """

    name: str = "solid_analysis"

    _LLM_BATCH_SIZE: int = 3

    # ── LangChain chain ───────────────────────────────────────────────────

    def _build_chain(self) -> Any:
        """
        PromptTemplate | LLM | JsonOutputParser → raw list[dict]
        The prompt is loaded from prompts/solid_analysis.jinja2.
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from services.llm_client import get_llm_client

        parser  = JsonOutputParser()
        tmpl    = self._load_prompt_template()
        prompt  = PromptTemplate(
            template=tmpl,
            input_variables=[
                "code_chunk", "file_path", "start_line", "end_line", "language",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | get_llm_client() | parser

    # ── Entry point ───────────────────────────────────────────────────────

    async def run(
        self,
        chunks: list[CodeChunk],
        *,
        llm_confirm: bool = True,
    ) -> list[SolidFinding]:
        """
        Analyse code chunks for SOLID violations.

        Args:
            chunks      : CodeChunk objects from ingest_node.
            llm_confirm : If True, also run per-chunk LLM analysis for LSP
                          detection and heuristic confirmation.

        Returns:
            Deduplicated, severity-sorted list of SolidFinding.
        """
        if not chunks:
            return []

        logger.info("solid_agent_run", chunks=len(chunks))

        # Phase 1 — AST heuristics
        hits = _scan_heuristics(chunks)
        logger.info("solid_agent_heuristic_hits", hits=len(hits))
        rule_findings: list[SolidFinding] = [_hit_to_finding(h) for h in hits]

        if not llm_confirm:
            return _sort_findings(_deduplicate(rule_findings))

        # Phase 2 — LLM analysis
        llm_findings: list[SolidFinding] = await self._llm_analyze(chunks)
        logger.info("solid_agent_llm_findings", count=len(llm_findings))

        # Merge: LLM findings preferred; heuristic findings fill the rest
        confirmed_keys = {
            (f.file_path, f.start_line, str(f.principle))
            for f in llm_findings
        }
        merged = list(llm_findings)
        for rf in rule_findings:
            if (rf.file_path, rf.start_line, str(rf.principle)) not in confirmed_keys:
                merged.append(rf)

        result = _sort_findings(_deduplicate(merged))
        logger.info("solid_agent_complete", total_findings=len(result))
        return result

    # ── LLM analysis ──────────────────────────────────────────────────────

    async def _llm_analyze(self, chunks: list[CodeChunk]) -> list[SolidFinding]:
        findings: list[SolidFinding] = []
        try:
            chain = self._build_chain()
        except Exception as exc:  # noqa: BLE001
            logger.warning("solid_agent_llm_unavailable", error=str(exc))
            return findings

        for chunk in chunks:
            if is_non_production_path(chunk.file_path):
                continue
            try:
                raw = await chain.ainvoke({
                    "code_chunk": chunk.content,
                    "file_path":  chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line":   chunk.end_line,
                    "language":   str(chunk.language),
                })
                findings.extend(_parse_llm_output(raw, chunk))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "solid_agent_llm_call_failed",
                    file=chunk.file_path,
                    error=str(exc),
                )

        return findings

    # ── Helpers ───────────────────────────────────────────────────────────

    def to_json(self, findings: list[SolidFinding]) -> str:
        return json.dumps(
            [f.model_dump(mode="json") for f in findings],
            indent=2,
            ensure_ascii=False,
        )
