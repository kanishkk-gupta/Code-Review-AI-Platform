"""
agents/bug_agent.py
====================
BugAgent — Hybrid bug analyzer: deterministic rule engine + LLM confirmation.

Pipeline
--------
Phase 1 — Rule Engine (always runs, zero LLM cost):
    Scans every CodeChunk with regex patterns covering five bug categories.
    Produces a _ScanHit per match (one hit per rule per chunk).

Phase 2 — LLM Confirmation (runs on HIGH/CRITICAL hits only):
    Batches suspicious chunks and asks the LLM to confirm, reject, or refine
    each rule hit.  Uses JsonOutputParser for structured output.
    Falls back to rule-only findings if the LLM is unavailable.

Detects
-------
  - Null / None dereference          (NullDereference)
  - Division / modulo by zero        (DivisionByZero)
  - Infinite loops                   (InfiniteLoop)
  - Resource leaks                   (ResourceLeak)
  - Dead / unreachable code          (DeadCode)

Usage
-----
::

    from agents.bug_agent import BugAgent
    from schemas import CodeChunk

    agent = BugAgent()
    findings = await agent.run(chunks)
    for f in findings:
        print(f.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from agents.base_agent import BaseAgent
from agents._ast_analysis import ast_analyze_python, confidence_to_level
from agents._scan_utils import (
    classify_file_role,
    compute_docstring_lines,
    downgrade_severity,
    is_comment_line,
    is_http_client_get,
    is_non_production_path,
    is_placeholder_secret,
    proximity_deduplicate,
    strip_inline_comment,
    strip_string_contents,
)
from agents.complexity_agent import aggregate_files_from_chunks
from schemas import BugFinding, CodeChunk, Severity

logger = structlog.get_logger(__name__)

# Rules where string literals must be stripped before matching
# (prevents route strings, URLs, and format strings from triggering)
_STRING_CONTEXT_RULES: frozenset[str] = frozenset({"BUG-010", "BUG-011"})

# After strip_string_contents, '"fmt" % var' → '"" % var'.
# This pattern detects that post-strip signature so BUG-011 can skip it.
_STRFMT_AFTER_STRIP_RE = re.compile(r'["\']["\']?\s*%')

# BUG-011 also needs the lookbehind guard applied at match time:
# after strip_string_contents, a Python "fmt" % var becomes "" % var;
# the closing quote directly precedes %, so the pattern is further
# narrowed by requiring % NOT be preceded by a closing quote —
# handled in the updated BUG-011 pattern below.

# ---------------------------------------------------------------------------
# Rule Definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    """A single regex-based bug detection rule."""
    rule_id:              str
    name:                 str
    pattern:              re.Pattern
    bug_pattern:          str          # short label for BugFinding.bug_pattern
    severity:             Severity
    reproducible:         bool
    description_template: str
    fix_template:         str


def _r(pattern: str, flags: int = re.IGNORECASE | re.MULTILINE) -> re.Pattern:
    return re.compile(pattern, flags)


_RULES: list[_Rule] = [

    # ── Null / None Dereference ───────────────────────────────────────────────
    _Rule(
        rule_id="BUG-001",
        name="Attribute access on potentially None value",
        pattern=_r(
            # Require  None  immediately followed (≤15 chars, no \n or =)
            # by a dot-and-lowercase-identifier (method/attribute call).
            # Excludes: type annotations, string literals, comments (stripped
            # upstream), conditional expressions where None is on the right.
            # Requires the identifier after the dot to look like a real attr
            # (lowercase or underscore-prefixed) to reduce annotation hits.
            r"(?<![\w'])\bNone\b[^\n=]{0,15}\.[a-z_]\w*"
        ),
        bug_pattern="NullDereference",
        severity=Severity.HIGH,
        reproducible=True,
        description_template=(
            "Possible attribute or method access on a value that may be None: `{match}`. "
            "If the expression is None at runtime this will raise AttributeError."
        ),
        fix_template=(
            "Add a None-guard before access:\n"
            "  if value is not None:\n"
            "      value.method()\n"
            "Or use a default: d.get(key) or default_obj"
        ),
    ),
    _Rule(
        rule_id="BUG-002",
        name="Unguarded dict .get() result attribute access",
        pattern=_r(
            # Match .get(key).attr but NOT HTTP client .get(url).json()
            # Negative lookbehind handled at scan time via is_http_client_get().
            r"\.\bget\b\([^)]{0,60}\)\s*\."
        ),
        bug_pattern="NullDereference",
        severity=Severity.MEDIUM,
        reproducible=True,
        description_template=(
            "`.get()` returns None when key is absent; chaining `.attribute` "
            "without a guard raises AttributeError: `{match}`."
        ),
        fix_template=(
            "Provide a default value:\n"
            "  result = d.get(key, fallback_obj)\n"
            "Or guard explicitly:\n"
            "  val = d.get(key)\n"
            "  if val is not None:\n"
            "      val.attribute"
        ),
    ),
    _Rule(
        rule_id="BUG-003",
        name="Unguarded Optional parameter dereference",
        pattern=_r(
            r"def\s+\w+\s*\([^)]*:\s*Optional\[[^\]]+\]\s*=\s*None[^)]*\)\s*(?:->[^:]+)?:\s*\n"
            r"(?!\s*if\b)"
        ),
        bug_pattern="NullDereference",
        severity=Severity.HIGH,
        reproducible=False,
        description_template=(
            "Function parameter typed `Optional[...] = None` — first line of body has no "
            "None-guard: `{match}`. Calling without that argument can trigger AttributeError."
        ),
        fix_template=(
            "Add an early guard at the top of the function:\n"
            "  if param is None:\n"
            "      raise ValueError('param must be provided') # or return early"
        ),
    ),

    # ── Division / Modulo by Zero ─────────────────────────────────────────────
    _Rule(
        rule_id="BUG-010",
        name="Division by variable without zero-check",
        pattern=_r(
            r"(?<![=<>!])/"
            r"\s*(?![\d.]+\b)"        # not a literal
            r"(?P<divisor>[a-zA-Z_]\w*)"
            r"(?!\s*=)"               # not /=
        ),
        bug_pattern="DivisionByZero",
        severity=Severity.HIGH,
        reproducible=False,
        description_template=(
            "Division by variable `{match}` with no preceding zero-guard. "
            "ZeroDivisionError will be raised if the divisor is 0."
        ),
        fix_template=(
            "Guard the divisor before dividing:\n"
            "  if denominator == 0:\n"
            "      raise ValueError('denominator cannot be zero')\n"
            "  result = numerator / denominator\n"
            "Or use a safe fallback:\n"
            "  result = numerator / denominator if denominator else default"
        ),
    ),
    _Rule(
        rule_id="BUG-011",
        name="Modulo by variable without zero-check",
        pattern=_r(
            # After strip_string_contents, '"fmt" % var' becomes '"" % var'.
            # The closing quote then directly precedes %, so this lookbehind
            # prevents string-formatting false positives.
            r'(?<!["\'])'
            r'%\s*(?![\d]+\b)(?![\'"% ])'
            r'(?P<divisor>[a-zA-Z_]\w*)'
            r'(?!\s*=)'
        ),
        bug_pattern="DivisionByZero",
        severity=Severity.MEDIUM,
        reproducible=False,
        description_template=(
            "Modulo by variable `{match}` without a preceding zero check. "
            "ZeroDivisionError if divisor is 0."
        ),
        fix_template=(
            "Check divisor before modulo:\n"
            "  if n == 0:\n"
            "      raise ValueError('modulo by zero')\n"
            "  result = value % n"
        ),
    ),

    # ── Infinite Loops ────────────────────────────────────────────────────────
    _Rule(
        rule_id="BUG-020",
        name="while True without break",
        pattern=_r(r"^\s*while\s+True\s*:"),
        bug_pattern="InfiniteLoop",
        severity=Severity.HIGH,
        reproducible=True,
        description_template=(
            "`while True:` loop detected: `{match}`. "
            "Verify a reachable `break` or `return` exists on all paths; "
            "otherwise this loop will never terminate."
        ),
        fix_template=(
            "Add an explicit exit condition:\n"
            "  while True:\n"
            "      ...\n"
            "      if exit_condition:\n"
            "          break\n"
            "Or convert to a `while condition:` loop."
        ),
    ),
    _Rule(
        rule_id="BUG-021",
        name="while 1 without break",
        pattern=_r(r"^\s*while\s+1\s*:", re.MULTILINE),
        bug_pattern="InfiniteLoop",
        severity=Severity.MEDIUM,
        reproducible=True,
        description_template=(
            "`while 1:` loop detected: `{match}`. Verify a `break` or `return` "
            "statement is reachable on all paths."
        ),
        fix_template=(
            "Use `while True:` with a clear exit condition for readability, "
            "and ensure every code path eventually breaks out of the loop."
        ),
    ),
    _Rule(
        rule_id="BUG-022",
        name="Loop variable modified inside loop body",
        pattern=_r(
            r"for\s+(?P<var>\w+)\s+in\s+range\([^)]+\)\s*:[^:]*?\n"
            r"(?:[^\n]*\n){0,15}"
            r"[^\n]*(?P=var)\s*="
        ),
        bug_pattern="InfiniteLoop",
        severity=Severity.LOW,
        reproducible=False,
        description_template=(
            "Loop variable `{match}` appears to be reassigned inside a `for` loop body. "
            "In Python this does not affect iteration order — it may indicate "
            "a misunderstanding that could cause incorrect behavior."
        ),
        fix_template=(
            "Do not rely on reassigning a `for` loop variable to control iteration. "
            "Use a `while` loop with explicit counter management if you need to skip steps, "
            "or collect indices and slice the iterable instead."
        ),
    ),

    # ── Resource Leaks ────────────────────────────────────────────────────────
    _Rule(
        rule_id="BUG-030",
        name="File opened without context manager",
        pattern=_r(
            r"^(?!\s*with\b)(?!.*\bwith\b.*\bopen\b)"
            r".*=\s*open\s*\([^)]+\)"
        ),
        bug_pattern="ResourceLeak",
        severity=Severity.HIGH,
        reproducible=False,
        description_template=(
            "File opened via assignment without a `with` statement: `{match}`. "
            "If an exception occurs before `.close()` is called the file handle leaks, "
            "potentially causing data loss or exhausting OS file descriptors."
        ),
        fix_template=(
            "Always use the context manager form:\n"
            "  with open(path, 'r') as fh:\n"
            "      data = fh.read()"
        ),
    ),
    _Rule(
        rule_id="BUG-031",
        name="Database connection not closed",
        pattern=_r(
            r"(?:conn|connection|cursor|db)\s*=\s*"
            r"(?:psycopg2|sqlite3|pymysql|cx_Oracle|pyodbc|mysql\.connector)"
            r"\.connect\s*\("
        ),
        bug_pattern="ResourceLeak",
        severity=Severity.MEDIUM,
        reproducible=False,
        description_template=(
            "Database connection opened via assignment: `{match}`. "
            "If not wrapped in a `with` block or finally-closed, the connection "
            "leaks when an exception occurs."
        ),
        fix_template=(
            "Use context manager or guarantee closure:\n"
            "  with psycopg2.connect(...) as conn:\n"
            "      ...\n"
            "Or add an explicit try/finally:\n"
            "  conn = psycopg2.connect(...)\n"
            "  try:\n"
            "      ...\n"
            "  finally:\n"
            "      conn.close()"
        ),
    ),
    _Rule(
        rule_id="BUG-032",
        name="Socket not closed",
        pattern=_r(
            r"(?:sock|socket|s)\s*=\s*socket\.socket\s*\([^)]*\)"
            r"(?!(?:[^\n]*\n){0,5}[^\n]*with\b)"
        ),
        bug_pattern="ResourceLeak",
        severity=Severity.MEDIUM,
        reproducible=False,
        description_template=(
            "Socket created without a `with` context manager: `{match}`. "
            "Network sockets are OS-level resources; leaking them causes descriptor exhaustion."
        ),
        fix_template=(
            "Wrap in a context manager:\n"
            "  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:\n"
            "      s.connect((host, port))\n"
            "      ..."
        ),
    ),

    # ── Dead / Unreachable Code ───────────────────────────────────────────────
    _Rule(
        rule_id="BUG-040",
        name="Code after return statement",
        pattern=_r(r"^\s*(return\b[^\n]*)\n(\s+)(\S[^\n]*)$", re.MULTILINE),
        bug_pattern="DeadCode",
        severity=Severity.LOW,
        reproducible=True,
        description_template=(
            "Code found after a `return` statement at the same or inner indentation: "
            "`{match}`. This code is unreachable and will never execute."
        ),
        fix_template=(
            "Remove the unreachable code, or restructure the logic so the "
            "`return` statement comes after all intended work is done."
        ),
    ),
    _Rule(
        rule_id="BUG-041",
        name="Code after raise statement",
        pattern=_r(r"^\s*(raise\b[^\n]*)\n(\s+)(\S[^\n]*)$", re.MULTILINE),
        bug_pattern="DeadCode",
        severity=Severity.LOW,
        reproducible=True,
        description_template=(
            "Statement found after a `raise` at the same indentation: `{match}`. "
            "Execution will never reach this code."
        ),
        fix_template=(
            "Remove the dead code following the `raise`, or move it before "
            "the exception is raised."
        ),
    ),
    _Rule(
        rule_id="BUG-042",
        name="Unreachable else after exhaustive if/return chain",
        pattern=_r(
            r"if\s+(?:True|1)\s*:\s*\n"
            r"(?:\s+[^\n]+\n)+"
            r"\s*else\s*:"
        ),
        bug_pattern="DeadCode",
        severity=Severity.INFO,
        reproducible=True,
        description_template=(
            "`else` block after `if True:` is unreachable: `{match}`. "
            "The else branch will never execute."
        ),
        fix_template=(
            "Remove the `else` clause or replace `if True:` with the actual condition."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Scan result dataclass
# ---------------------------------------------------------------------------


@dataclass
class _ScanHit:
    """A single rule-engine match on a chunk."""
    rule:       _Rule
    chunk:      CodeChunk
    match_text: str
    line_no:    int   # absolute line number within file
    evidence:   str = ""
    reasoning:  str = ""
    confidence: float = 0.70


# ---------------------------------------------------------------------------
# BugAgent
# ---------------------------------------------------------------------------


class BugAgent(BaseAgent[BugFinding]):
    """
    Hybrid bug analyzer.

    Phase 1 — Rule Engine (always runs, zero cost):
        Applies 12 regex rules across five bug categories to every chunk.

    Phase 2 — LLM Confirmation (HIGH/CRITICAL hits only):
        Batches flagged chunks for LLM confirmation via
        PydanticOutputParser[BugFinding]. Falls back to rule-only findings
        when the LLM is unavailable or misconfigured.
    """

    name: str = "bug_analysis"

    # Maximum chunks sent to LLM per batch (guards context overflow)
    _LLM_BATCH_SIZE: int = 4

    # Only escalate findings at or above this severity to the LLM
    _LLM_ESCALATION_THRESHOLD: Severity = Severity.HIGH

    # ── LangChain chain ───────────────────────────────────────────────────

    def _build_chain(self) -> Any:
        """
        Build the LangChain confirmation chain.

        Chain: PromptTemplate | LLM | JsonOutputParser → list[BugFinding]
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from services.llm_client import get_llm_client

        parser = JsonOutputParser()
        template_str = self._load_prompt_template()

        prompt = PromptTemplate(
            template=template_str,
            input_variables=[
                "code_chunk", "file_path", "start_line", "end_line",
                "flagged_patterns", "rule_severity", "language",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | get_llm_client() | parser

    # ── Primary entry point ───────────────────────────────────────────────

    async def run(
        self,
        chunks: list[CodeChunk],
        *,
        llm_confirm: bool = True,
    ) -> list[BugFinding]:
        """
        Run the hybrid bug analysis pipeline.

        Args:
            chunks      : CodeChunk objects from ingest_node.
            llm_confirm : If True (default), escalate HIGH/CRITICAL rule hits
                          to the LLM for confirmation and enrichment.
                          Set False for fast/offline mode.

        Returns:
            Deduplicated list of BugFinding objects sorted by severity.
        """
        if not chunks:
            logger.info("bug_agent_run", chunks=0, findings=0)
            return []

        logger.info("bug_agent_run", chunks=len(chunks))

        # Phase 1 — rule scan (always)
        hits: list[_ScanHit] = _scan_chunks(chunks)
        logger.info("bug_agent_rule_hits", hits=len(hits))

        if not hits:
            logger.info("bug_agent_complete", total_findings=0)
            return []

        # Build rule-based findings (always available as fallback)
        rule_findings: list[BugFinding] = [_hit_to_finding(hit) for hit in hits]

        if not llm_confirm:
            return _sort_findings(_deduplicate(rule_findings))

        # Phase 2 — LLM confirmation on HIGH/CRITICAL hits only
        escalated = [
            h for h in hits
            if _severity_index(h.rule.severity) <= _severity_index(self._LLM_ESCALATION_THRESHOLD)
        ]

        llm_findings: list[BugFinding] = []
        if escalated:
            llm_findings = await self._llm_confirm(escalated)
            logger.info("bug_agent_llm_findings", count=len(llm_findings))

        # Merge: LLM findings preferred; rule findings fill uncovered positions
        confirmed_positions = {(f.file_path, f.start_line) for f in llm_findings}
        merged = list(llm_findings)
        for rf in rule_findings:
            if (rf.file_path, rf.start_line) not in confirmed_positions:
                merged.append(rf)

        result = _sort_findings(_deduplicate(merged))
        logger.info("bug_agent_complete", total_findings=len(result))
        return result

    # ── LLM confirmation ──────────────────────────────────────────────────

    async def _llm_confirm(self, hits: list[_ScanHit]) -> list[BugFinding]:
        """
        Send flagged chunks to the LLM for confirmation/enrichment.
        Returns confirmed BugFinding list; false positives are dropped.
        """
        findings: list[BugFinding] = []

        try:
            chain = self._build_chain()
        except Exception as exc:  # noqa: BLE001
            logger.warning("bug_agent_llm_chain_unavailable", error=str(exc))
            return findings

        for i in range(0, len(hits), self._LLM_BATCH_SIZE):
            batch = hits[i : i + self._LLM_BATCH_SIZE]
            for hit in batch:
                try:
                    raw = await chain.ainvoke({
                        "code_chunk":      hit.chunk.content,
                        "file_path":       hit.chunk.file_path,
                        "start_line":      hit.chunk.start_line,
            "end_line":        hit.chunk.end_line,
                        "flagged_patterns": hit.rule.name,
                        "rule_severity":   hit.rule.severity,
                        "language":        hit.chunk.language,
                    })
                    findings.extend(_parse_llm_output(raw, hit))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "bug_agent_llm_call_failed",
                        rule=hit.rule.rule_id,
                        file=hit.chunk.file_path,
                        error=str(exc),
                    )

        return findings

    # ── Serialization helper ──────────────────────────────────────────────

    def to_json(self, findings: list[BugFinding]) -> str:
        """Serialize findings to an indented JSON string."""
        return json.dumps(
            [f.model_dump(mode="json") for f in findings],
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------


def _chunk_covering_line(chunks: list[CodeChunk], line_no: int) -> CodeChunk:
    """Return the chunk whose line span contains *line_no*."""
    for chunk in chunks:
        if chunk.start_line <= line_no <= chunk.end_line:
            return chunk
    return chunks[0]


def _scan_chunks(chunks: list[CodeChunk]) -> list[_ScanHit]:
    """
    Apply all rules to production code.

    Python AST rules run on complete reconstructed files (never partial chunks).
    Regex is skipped for AST-handled Python rules and for BUG-010/011 entirely.
    """
    from schemas import SupportedLanguage

    hits: list[_ScanHit] = []

    _AST_RULES = frozenset({
        "BUG-001", "BUG-002", "BUG-003", "BUG-010", "BUG-011", "BUG-020", "BUG-021",
    })

    # ── File-level AST pass (Python, production paths only) ───────────────
    for aggregate in aggregate_files_from_chunks(chunks).values():
        if is_non_production_path(aggregate.file_path):
            continue
        if aggregate.language != SupportedLanguage.PYTHON:
            continue

        file_lines = aggregate.content.splitlines()
        doc_line_indices = compute_docstring_lines(aggregate.content)
        ast_rule_hits = ast_analyze_python(
            aggregate.content,
            start_line=1,
            file_path=aggregate.file_path,
        )
        seen_ast: set[str] = set()
        for ah in ast_rule_hits:
            if ah.rule_id in seen_ast:
                continue
            rule = next((r for r in _RULES if r.rule_id == ah.rule_id), None)
            if rule is None:
                continue
            rel_idx = ah.line_no - 1
            if 0 <= rel_idx < len(file_lines) and rel_idx in doc_line_indices:
                continue
            seen_ast.add(ah.rule_id)
            chunk = _chunk_covering_line(aggregate.chunks, ah.line_no)
            hits.append(_ScanHit(
                rule=rule,
                chunk=chunk,
                match_text=ah.match_text,
                line_no=ah.line_no,
                evidence=ah.evidence,
                reasoning=ah.reasoning,
                confidence=ah.confidence,
            ))

    # ── Per-chunk regex pass (non-AST rules only) ─────────────────────────
    for chunk in chunks:
        if is_non_production_path(chunk.file_path):
            continue

        lines = chunk.content.splitlines()
        is_python = getattr(chunk, 'language', None) in (
            SupportedLanguage.PYTHON, 'python',
        )
        doc_line_indices = compute_docstring_lines(chunk.content)

        for rule in _RULES:
            if rule.rule_id in _AST_RULES:
                continue
            if rule.rule_id in ("BUG-010", "BUG-011"):
                continue

            for rel_idx, line in enumerate(lines):

                # Skip docstring lines (triple-quoted string content)
                if rel_idx in doc_line_indices:
                    continue

                # Skip full comment lines
                if is_comment_line(line):
                    continue

                # Strip inline comment suffix
                scan_line = strip_inline_comment(line)

                # String-context rules: strip string literal contents
                if rule.rule_id in _STRING_CONTEXT_RULES:
                    scan_line = strip_string_contents(scan_line)
                    # Post-strip: detect remaining "" % var (string formatting)
                    if rule.rule_id == "BUG-011" and _STRFMT_AFTER_STRIP_RE.search(scan_line):
                        continue

                m = rule.pattern.search(scan_line)
                if m:
                    match_text = m.group(0)[:120]
                    abs_line = chunk.start_line + rel_idx
                    hits.append(_ScanHit(
                        rule=rule,
                        chunk=chunk,
                        match_text=match_text,
                        line_no=abs_line,
                    ))
                    break  # one hit per (rule, chunk)
    return hits


def _hit_to_finding(hit: _ScanHit) -> BugFinding:
    """Convert a _ScanHit into a BugFinding with file-role severity adjustment."""
    desc = hit.rule.description_template.format(match=hit.match_text)

    severity = hit.rule.severity
    confidence = hit.confidence

    # P0 — downgrade findings from test / docs / example files
    role = classify_file_role(hit.chunk.file_path)
    if role in ('test', 'docs', 'example'):
        severity   = downgrade_severity(severity, steps=2)
        confidence = min(confidence, 0.25)

    from schemas import ConfidenceLevel

    return BugFinding(
        severity=severity,
        title=f"[{hit.rule.rule_id}] {hit.rule.name} in {hit.chunk.file_path}",
        description=desc,
        file_path=hit.chunk.file_path,
        start_line=hit.line_no,
        end_line=hit.line_no,
        suggested_fix=hit.rule.fix_template,
        confidence=confidence,
        confidence_level=ConfidenceLevel(confidence_to_level(confidence)),
        evidence=hit.evidence or None,
        reasoning=hit.reasoning or None,
        bug_pattern=hit.rule.bug_pattern,
        reproducible=hit.rule.reproducible,
    )


# ---------------------------------------------------------------------------
# LLM Output Parsing
# ---------------------------------------------------------------------------


def _parse_llm_output(raw: Any, hit: _ScanHit) -> list[BugFinding]:
    """
    Convert raw LLM JSON output to a list of BugFinding objects.
    Handles both array and single-object responses.
    """
    findings: list[BugFinding] = []

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list):
        logger.warning("bug_agent_unexpected_llm_output", type=type(raw).__name__)
        return findings

    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            # Ensure required fields have fallbacks from the triggering hit
            item.setdefault("file_path", hit.chunk.file_path)
            item.setdefault("start_line", hit.chunk.start_line)
            item.setdefault("end_line", hit.chunk.end_line)
            item.setdefault("confidence", 0.88)
            item.setdefault("bug_pattern", hit.rule.bug_pattern)
            item.setdefault("reproducible", hit.rule.reproducible)

            finding = BugFinding.model_validate(item)
            findings.append(finding)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "bug_agent_parse_error",
                error=str(exc),
                raw_keys=list(item.keys()) if isinstance(item, dict) else "N/A",
            )

    return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SEVERITY_ORDER = [
    Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO
]


def _severity_index(s: Severity) -> int:
    try:
        return _SEVERITY_ORDER.index(s)
    except ValueError:
        return len(_SEVERITY_ORDER)


def _sort_findings(findings: list[BugFinding]) -> list[BugFinding]:
    return sorted(findings, key=lambda f: (_severity_index(Severity(f.severity)), f.file_path))


def _deduplicate(findings: list[BugFinding]) -> list[BugFinding]:
    """
    Remove duplicates:
      - Exact key: (file_path, start_line, bug_pattern)
      - Proximity: same (file_path, bug_pattern) within 5 lines
    """
    # First pass: exact dedup
    seen: set[tuple] = set()
    exact: list[BugFinding] = []
    for f in findings:
        key = (f.file_path, f.start_line, f.bug_pattern)
        if key not in seen:
            seen.add(key)
            exact.append(f)

    # Second pass: proximity dedup (same rule within 5 lines = same issue)
    return proximity_deduplicate(
        exact,
        key_fn=lambda f: (f.file_path, f.bug_pattern),
        line_fn=lambda f: f.start_line,
        window=5,
    )
