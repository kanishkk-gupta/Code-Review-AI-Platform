"""
agents/security_agent.py
=========================
Hybrid security analyzer: rule-based scanner + LLM confirmation.

Pipeline:
  1. Rule engine scans every chunk with regex patterns (zero LLM cost).
  2. Flagged chunks are batched and sent to the LLM for confirmation.
  3. LLM output is parsed via PydanticOutputParser[SecurityFinding].
  4. Rule-only findings (LLM skipped or unavailable) are emitted directly.

Detects:
  - Hardcoded secrets / credentials      (CWE-798)
  - SQL injection                        (CWE-89)
  - Command injection / OS exec          (CWE-78)
  - Unsafe file operations               (CWE-73 / CWE-22)
  - SSRF patterns                        (CWE-918)
  - Insecure deserialization             (CWE-502)
  - Weak cryptography                    (CWE-326)
  - Path traversal                       (CWE-22)
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
    extract_secret_value,
    is_comment_line,
    is_non_production_path,
    is_placeholder_secret,
    is_test_fixture_secret,
    proximity_deduplicate,
    strip_inline_comment,
)
from agents.complexity_agent import aggregate_files_from_chunks
from schemas import (
    CodeChunk,
    SecurityCategory,
    SecurityFinding,
    Severity,
)

logger = structlog.get_logger(__name__)

# SEC-001 / SEC-003: check extracted value for placeholder / low entropy before
# adding to hits.  This prevents test fixtures, README examples, and template
# placeholders from triggering CRITICAL findings.
_PLACEHOLDER_CHECK_RULES: frozenset[str] = frozenset({"SEC-001", "SEC-003"})
_SECRET_RULES: frozenset[str] = frozenset({"SEC-001", "SEC-002", "SEC-003"})

# ---------------------------------------------------------------------------
# Rule Definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Rule:
    """A single regex-based detection rule."""
    rule_id:        str
    name:           str
    pattern:        re.Pattern
    category:       SecurityCategory
    cwe_id:         str
    severity:       Severity
    cvss_score:     float
    exploitability: str
    requires_user_interaction: bool
    description_template: str
    fix_template:          str


def _r(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
    return re.compile(pattern, flags)


_RULES: list[_Rule] = [
    # ── Hardcoded Secrets ─────────────────────────────────────────────────
    _Rule(
        rule_id="SEC-001",
        name="Hardcoded Password Assignment",
        pattern=_r(
            r'(?:password|passwd|pwd|secret(?:_key)?|api_?key|auth_?token|'
            r'db_?pass(?:word)?|database_?pass(?:word)?|secret_?key|'
            r'private_?key|access_?key|app_?secret)'
            r'\s*=\s*["\'][^"\']{4,}["\']'
        ),
        category=SecurityCategory.HARDCODED_SECRET,
        cwe_id="CWE-798",
        severity=Severity.CRITICAL,
        cvss_score=9.8,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="Hardcoded credential detected: `{match}`. Credentials embedded in source code are trivially extractable by any actor with repository access.",
        fix_template="Move the credential to an environment variable or secrets manager (e.g. AWS Secrets Manager, HashiCorp Vault). Use `os.environ['SECRET_NAME']` or a settings object.",
    ),
    _Rule(
        rule_id="SEC-002",
        name="Hardcoded Private Key / Token",
        pattern=_r(r'(?:-----BEGIN (?:RSA |EC )?PRIVATE KEY-----|sk_live_[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|xox[baprs]-[0-9a-zA-Z]{10,})'),
        category=SecurityCategory.HARDCODED_SECRET,
        cwe_id="CWE-321",
        severity=Severity.CRITICAL,
        cvss_score=9.8,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="Private key or live API token detected in source: `{match}`. This allows full account takeover or service impersonation.",
        fix_template="Revoke the exposed credential immediately. Store secrets in a vault or CI/CD secret store, never in source code.",
    ),
    _Rule(
        rule_id="SEC-003",
        name="Generic Secret Pattern",
        pattern=_r(
            r'(?:secret|token|apikey|api_key|access_key|auth_key|app_key|'
            r'client_secret|oauth_secret|signing_key|encryption_key)'
            r'\s*[=:]\s*["\'][A-Za-z0-9+/=_\-]{16,}["\']'
        ),
        category=SecurityCategory.HARDCODED_SECRET,
        cwe_id="CWE-798",
        severity=Severity.HIGH,
        cvss_score=8.2,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="Potential hardcoded secret: `{match}`. Verify this is not a live credential.",
        fix_template="Externalize to environment variables. Audit git history for past exposures (`git log -S 'secret_value'`).",
    ),

    # ── SQL Injection ─────────────────────────────────────────────────────
    _Rule(
        rule_id="SEC-010",
        name="SQL Injection via String Concatenation",
        pattern=_r(r'(?:execute|cursor\.execute|query|db\.run)\s*\(\s*[f"\'].*?\%s.*?["\']|(?:SELECT|INSERT|UPDATE|DELETE).*?\+\s*(?:request|user|input|params|data|query)'),
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-89",
        severity=Severity.CRITICAL,
        cvss_score=9.8,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="SQL query appears to concatenate user-controlled input: `{match}`. This enables SQL injection attacks allowing data exfiltration, modification, or authentication bypass.",
        fix_template="Use parameterized queries: `cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))`. Never concatenate user input into SQL strings.",
    ),
    _Rule(
        rule_id="SEC-011",
        name="Raw SQL with User Input (ORM bypass)",
        pattern=_r(r'(?:raw\(|RawSQL\(|extra\(|\.objects\.raw\().*?(?:format|%|f["\'])'),
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-89",
        severity=Severity.HIGH,
        cvss_score=8.8,
        exploitability="Moderate",
        requires_user_interaction=False,
        description_template="Raw SQL execution with potential string interpolation: `{match}`. ORM raw() calls bypass query parameter escaping.",
        fix_template="Use ORM query methods with explicit parameterization. If raw SQL is unavoidable, use `params=` argument: `Model.objects.raw(sql, params=[value])`.",
    ),

    # ── Command Injection ─────────────────────────────────────────────────
    _Rule(
        rule_id="SEC-020",
        name="OS Command Injection via shell=True",
        pattern=_r(r'subprocess\.(run|call|Popen|check_output|check_call)\s*\([^)]*shell\s*=\s*True'),
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-78",
        severity=Severity.CRITICAL,
        cvss_score=9.8,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="subprocess called with `shell=True`: `{match}`. Any user-controlled data in the command string enables arbitrary OS command execution.",
        fix_template="Pass commands as a list (shell=False): `subprocess.run(['cmd', arg1, arg2])`. Validate and allowlist any external inputs used in system calls.",
    ),
    _Rule(
        rule_id="SEC-021",
        name="Direct os.system / os.popen with Variables",
        pattern=_r(r'os\.(system|popen|execv?[ep]?)\s*\([^)]*(?:\+|%)'),
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-78",
        severity=Severity.CRITICAL,
        cvss_score=9.8,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="Dynamic OS command execution detected: `{match}`. Variables interpolated into shell commands enable command injection.",
        fix_template="Replace `os.system()` with `subprocess.run(cmd_list, shell=False)`. Sanitize all inputs with an allowlist before use in system calls.",
    ),
    _Rule(
        rule_id="SEC-022",
        name="eval() / exec() with User Input",
        pattern=_r(r'\b(?:eval|exec)\s*\(\s*(?:request|user|input|data|query|params|getattr)'),
        category=SecurityCategory.INJECTION,
        cwe_id="CWE-95",
        severity=Severity.CRITICAL,
        cvss_score=9.8,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="Dynamic code execution from user-controlled source: `{match}`. Arbitrary Python/JS code can be injected and executed.",
        fix_template="Never pass user input to `eval()` or `exec()`. Replace with safe data structures (JSON, allowlisted operations, or a sandboxed interpreter).",
    ),

    # ── Unsafe File Operations ────────────────────────────────────────────
    _Rule(
        rule_id="SEC-030",
        name="Path Traversal via User-Controlled Path",
        pattern=_r(r'(?:open|pathlib\.Path|os\.path\.join)\s*\([^)]*(?:request|user|input|data|params|query|filename)[^)]*\)'),
        category=SecurityCategory.PATH_TRAVERSAL,
        cwe_id="CWE-22",
        severity=Severity.HIGH,
        cvss_score=8.6,
        exploitability="Moderate",
        requires_user_interaction=False,
        description_template="File operation with user-supplied path component: `{match}`. Sequences like `../` allow reading or writing arbitrary files on the server.",
        fix_template="Canonicalize paths and enforce they remain within an allowed base directory:\n```python\nbase = Path('/safe/uploads').resolve()\ntarget = (base / user_filename).resolve()\nassert str(target).startswith(str(base))  # reject traversal\n```",
    ),
    _Rule(
        rule_id="SEC-031",
        name="Unsafe File Upload without Extension Check",
        pattern=_r(r'\.save\s*\(\s*(?:request|file|upload|data)(?!.*(?:allowlist|whitelist|extension|suffix))'),
        category=SecurityCategory.SECURITY_MISCONFIGURATION,
        cwe_id="CWE-434",
        severity=Severity.HIGH,
        cvss_score=7.5,
        exploitability="Moderate",
        requires_user_interaction=True,
        description_template="File save operation without apparent extension validation: `{match}`. Attackers may upload malicious executables or scripts.",
        fix_template="Validate file extension against an allowlist before saving. Store uploads outside the web root. Never execute uploaded files.",
    ),

    # ── SSRF ──────────────────────────────────────────────────────────────
    _Rule(
        rule_id="SEC-040",
        name="SSRF — HTTP Request with User-Controlled URL",
        pattern=_r(r'(?:requests|httpx|urllib|aiohttp)\s*\.(?:get|post|put|request)\s*\([^)]*(?:url|uri|endpoint|target|host|addr)\s*[=,]'),
        category=SecurityCategory.SSRF,
        cwe_id="CWE-918",
        severity=Severity.HIGH,
        cvss_score=8.8,
        exploitability="Moderate",
        requires_user_interaction=False,
        description_template="HTTP client call with potentially user-supplied URL: `{match}`. Attackers may redirect requests to internal services (metadata APIs, databases).",
        fix_template="Validate URLs against an allowlist of approved domains. Block private IP ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x). Use a dedicated HTTP proxy with egress filtering.",
    ),

    # ── Insecure Deserialization ──────────────────────────────────────────
    _Rule(
        rule_id="SEC-050",
        name="Unsafe pickle.loads / yaml.load",
        pattern=_r(r'(?:pickle\.loads?|yaml\.load\s*\([^)]*\))'),
        category=SecurityCategory.INSECURE_DESERIALIZATION,
        cwe_id="CWE-502",
        severity=Severity.CRITICAL,
        cvss_score=9.8,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="Unsafe deserialization: `{match}`. Deserializing attacker-controlled data can result in arbitrary code execution.",
        fix_template="For YAML: use `yaml.safe_load()`. For pickle: never deserialize untrusted data; use JSON or MessagePack instead.",
    ),

    # ── Weak Cryptography ─────────────────────────────────────────────────
    _Rule(
        rule_id="SEC-060",
        name="Weak Hash Algorithm (MD5 / SHA1)",
        pattern=_r(r'(?:hashlib\.md5|hashlib\.sha1|MD5\s*\(|SHA1\s*\(|DigestUtils\.md5|DigestUtils\.sha1)'),
        category=SecurityCategory.SENSITIVE_DATA_EXPOSURE,
        cwe_id="CWE-326",
        severity=Severity.MEDIUM,
        cvss_score=5.3,
        exploitability="Moderate",
        requires_user_interaction=False,
        description_template="Weak cryptographic hash function: `{match}`. MD5 and SHA-1 are cryptographically broken and unsuitable for password hashing or integrity verification.",
        fix_template="Use SHA-256+ (`hashlib.sha256`) for checksums. For passwords use `bcrypt`, `argon2`, or `hashlib.scrypt`. Never use MD5/SHA-1 for security-sensitive operations.",
    ),
    _Rule(
        rule_id="SEC-061",
        name="Hardcoded Cryptographic Key",
        pattern=_r(r'(?:AES|DES|RSA|Fernet)\s*\(["\'][A-Za-z0-9+/=]{8,}["\']'),
        category=SecurityCategory.HARDCODED_SECRET,
        cwe_id="CWE-321",
        severity=Severity.CRITICAL,
        cvss_score=9.1,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="Hardcoded cryptographic key literal: `{match}`. A fixed key negates all cryptographic protection — any holder of the source can decrypt all data.",
        fix_template="Generate keys securely at runtime and store them in a key management service (KMS). Never embed encryption keys in source code.",
    ),

    # ── Debug / Misconfiguration ──────────────────────────────────────────
    _Rule(
        rule_id="SEC-070",
        name="Debug Mode Enabled in Production Code",
        pattern=_r(r'(?:DEBUG\s*=\s*True|app\.run\s*\([^)]*debug\s*=\s*True|FLASK_DEBUG\s*=\s*["\']?1["\']?)'),
        category=SecurityCategory.SECURITY_MISCONFIGURATION,
        cwe_id="CWE-94",
        severity=Severity.HIGH,
        cvss_score=7.5,
        exploitability="Trivial",
        requires_user_interaction=False,
        description_template="Debug mode appears enabled: `{match}`. Debug mode exposes stack traces, environment variables, and interactive consoles to unauthenticated users.",
        fix_template="Set `DEBUG = False` in production. Load debug flags from environment variables: `DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'`.",
    ),
]


# ---------------------------------------------------------------------------
# Scan result dataclass
# ---------------------------------------------------------------------------

@dataclass
class _ScanHit:
    """A single rule-engine hit on a chunk."""
    rule:       _Rule
    chunk:      CodeChunk
    match_text: str
    line_no:    int   # absolute line number within file
    evidence:   str = ""
    reasoning:  str = ""
    confidence: float = 0.75


# ---------------------------------------------------------------------------
# SecurityAgent
# ---------------------------------------------------------------------------


class SecurityAgent(BaseAgent[SecurityFinding]):
    """
    Hybrid security analyzer.

    Phase 1 — Rule Engine (always runs, zero cost):
        Scans every chunk with 13 regex rules covering the four mandated
        categories plus SSRF, insecure deserialization, weak crypto.

    Phase 2 — LLM Confirmation (runs only on flagged chunks):
        Batches suspicious chunks and asks the LLM to confirm/reject each
        rule hit.  Uses PydanticOutputParser[SecurityFinding] for structured
        output.  Falls back to rule-only findings if the LLM is unavailable.
    """

    name: str = "security_analysis"

    # Maximum chunks sent to LLM in one batch (avoids context overflow)
    _LLM_BATCH_SIZE: int = 5

    # Only escalate to LLM for hits at or above this severity
    _LLM_ESCALATION_THRESHOLD: Severity = Severity.HIGH

    def _build_chain(self) -> Any:
        """
        Build the LangChain confirmation chain.

        Chain: Jinja2PromptTemplate | LLM | JsonOutputParser → list[SecurityFinding]
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
                "flagged_patterns", "rule_severity",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | get_llm_client() | parser

    async def run(
        self,
        chunks: list[CodeChunk],
        *,
        llm_confirm: bool = True,
    ) -> list[SecurityFinding]:
        """
        Run the hybrid security analysis pipeline.

        Args:
            chunks      : CodeChunk objects from ingest_node.
            llm_confirm : If True (default), send flagged chunks for LLM
                          confirmation.  Set False for fast/offline mode
                          (rule-only findings are returned directly).

        Returns:
            Deduplicated list of SecurityFinding objects, sorted by severity.
        """
        if not chunks:
            logger.info("security_agent_run", chunks=0, findings=0)
            return []

        logger.info("security_agent_run", chunks=len(chunks))

        # Phase 1 — static rule scan
        hits: list[_ScanHit] = _scan_chunks(chunks)
        logger.info("security_agent_rule_hits", hits=len(hits))

        if not hits:
            return []

        # Build rule-based findings (always available)
        rule_findings: list[SecurityFinding] = [
            _hit_to_finding(hit) for hit in hits
        ]

        if not llm_confirm:
            return _sort_findings(rule_findings)

        # Phase 2 — LLM confirmation on high/critical hits
        escalated = [
            h for h in hits
            if _severity_index(h.rule.severity) <= _severity_index(self._LLM_ESCALATION_THRESHOLD)
        ]

        llm_findings: list[SecurityFinding] = []
        if escalated:
            llm_findings = await self._llm_confirm(escalated)
            logger.info("security_agent_llm_findings", count=len(llm_findings))

        # Merge: prefer LLM findings; keep rule findings for chunks not confirmed
        confirmed_files = {(f.file_path, f.start_line) for f in llm_findings}
        merged = list(llm_findings)
        for rf in rule_findings:
            if (rf.file_path, rf.start_line) not in confirmed_files:
                merged.append(rf)

        merged = _deduplicate(merged)
        result = _sort_findings(merged)

        logger.info("security_agent_complete", total_findings=len(result))
        return result

    async def _llm_confirm(self, hits: list[_ScanHit]) -> list[SecurityFinding]:
        """
        Send flagged chunks to the LLM for confirmation.
        Returns confirmed SecurityFinding list (false positives dropped).
        """
        findings: list[SecurityFinding] = []

        try:
            chain = self._build_chain()
        except Exception as exc:  # noqa: BLE001
            logger.warning("security_agent_llm_chain_unavailable", error=str(exc))
            return findings

        # Process in batches
        for i in range(0, len(hits), self._LLM_BATCH_SIZE):
            batch = hits[i : i + self._LLM_BATCH_SIZE]
            for hit in batch:
                try:
                    raw = await chain.ainvoke({
                        "code_chunk":       hit.chunk.content,
                        "file_path":        hit.chunk.file_path,
                        "start_line":       hit.chunk.start_line,
                        "end_line":         hit.chunk.end_line,
                        "flagged_patterns": hit.rule.name,
                        "rule_severity":    hit.rule.severity.value,
                    })
                    findings.extend(_parse_llm_output(raw, hit))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "security_agent_llm_call_failed",
                        rule=hit.rule.rule_id,
                        file=hit.chunk.file_path,
                        error=str(exc),
                    )

        return findings

    def to_json(self, findings: list[SecurityFinding]) -> str:
        """Serialise findings to indented JSON string."""
        return json.dumps(
            [f.model_dump(mode="json") for f in findings],
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------


def _chunk_covering_line(chunks: list[CodeChunk], line_no: int) -> CodeChunk:
    for chunk in chunks:
        if chunk.start_line <= line_no <= chunk.end_line:
            return chunk
    return chunks[0]


def _scan_chunks(chunks: list[CodeChunk]) -> list[_ScanHit]:
    """
    Apply all rules to production code.

    Python AST rules (SEC-010/030/040/050/070) run on complete files.
    Secret rules suppress placeholders and skip test/docs/example paths.
    """
    from schemas import SupportedLanguage

    _AST_RULES = frozenset({"SEC-010", "SEC-030", "SEC-040", "SEC-050", "SEC-070"})

    hits: list[_ScanHit] = []

    # ── File-level AST pass ───────────────────────────────────────────────
    for aggregate in aggregate_files_from_chunks(chunks).values():
        if is_non_production_path(aggregate.file_path):
            continue
        if aggregate.language != SupportedLanguage.PYTHON:
            continue

        file_lines = aggregate.content.splitlines()
        doc_line_indices = compute_docstring_lines(aggregate.content)
        ast_hits = ast_analyze_python(
            aggregate.content,
            start_line=1,
            file_path=aggregate.file_path,
            rule_ids=_AST_RULES,
        )
        seen_ast: set[str] = set()
        for ah in ast_hits:
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

    for chunk in chunks:
        if is_non_production_path(chunk.file_path):
            continue

        is_python = getattr(chunk, 'language', None) in (
            SupportedLanguage.PYTHON, 'python',
        )
        # Regex rules target Python source patterns; AST pass covers Python above.
        if not is_python:
            continue

        lines = chunk.content.splitlines()
        doc_line_indices = compute_docstring_lines(chunk.content)
        file_role = classify_file_role(chunk.file_path)

        for rule in _RULES:
            if rule.rule_id in _AST_RULES:
                continue
            for rel_idx, line in enumerate(lines):

                # Skip docstring lines (API examples, code samples)
                if rel_idx in doc_line_indices:
                    continue

                # Skip full comment lines
                if is_comment_line(line):
                    continue

                # Strip inline comment suffix
                scan_line = strip_inline_comment(line)

                m = rule.pattern.search(scan_line)
                if not m:
                    continue

                # SEC-010: skip parameterized queries — execute("...%s...", (val,))
                if rule.rule_id == "SEC-010" and re.search(
                    r'["\'][^"\']*%s[^"\']*["\']\s*,\s*[\(\[]',
                    scan_line,
                ):
                    continue

                # Suppress secrets in non-production paths entirely
                if rule.rule_id in _SECRET_RULES and file_role in ('test', 'docs', 'example'):
                    break

                if rule.rule_id in _PLACEHOLDER_CHECK_RULES:
                    secret_val = extract_secret_value(m.group(0))
                    if is_placeholder_secret(secret_val):
                        break
                    if is_test_fixture_secret(secret_val):
                        break

                if rule.rule_id == "SEC-002" and file_role in ('test', 'docs', 'example'):
                    break

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


def _hit_to_finding(hit: _ScanHit) -> SecurityFinding:
    """Convert a _ScanHit into a SecurityFinding with file-role adjustment."""
    desc = hit.rule.description_template.format(match=hit.match_text)

    severity   = hit.rule.severity
    confidence = hit.confidence

    role = classify_file_role(hit.chunk.file_path)
    reasoning = hit.reasoning or None
    if role in ('test', 'docs', 'example'):
        if hit.rule.rule_id in _SECRET_RULES:
            # Residual test-file secret hits are informational only
            severity   = Severity.INFO
            confidence = min(confidence, 0.20)
            reasoning  = (
                reasoning or
                f"Secret pattern in {role} file — likely test fixture, not production credential."
            )
        else:
            severity   = downgrade_severity(severity, steps=2)
            confidence = min(confidence, 0.25)

    from schemas import ConfidenceLevel

    return SecurityFinding(
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
        reasoning=reasoning,
        category=hit.rule.category,
        cwe_id=hit.rule.cwe_id,
        cvss_score=hit.rule.cvss_score,
        exploitability=hit.rule.exploitability,
        requires_user_interaction=hit.rule.requires_user_interaction,
    )


# ---------------------------------------------------------------------------
# LLM Output Parsing
# ---------------------------------------------------------------------------


def _parse_llm_output(
    raw: Any,
    hit: _ScanHit,
) -> list[SecurityFinding]:
    """
    Convert raw LLM JSON output to a list of SecurityFinding objects.
    Handles both array and single-object responses.
    """
    findings: list[SecurityFinding] = []

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list):
        logger.warning("security_agent_unexpected_llm_output", type=type(raw).__name__)
        return findings

    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            # Coerce category string to SecurityCategory enum
            raw_cat = item.get("category", "")
            item["category"] = _resolve_category(raw_cat)

            # Ensure required base fields have fallbacks
            item.setdefault("file_path", hit.chunk.file_path)
            item.setdefault("start_line", hit.chunk.start_line)
            item.setdefault("end_line", hit.chunk.end_line)
            item.setdefault("confidence", 0.85)

            finding = SecurityFinding.model_validate(item)
            findings.append(finding)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "security_agent_parse_error",
                error=str(exc),
                raw_keys=list(item.keys()) if isinstance(item, dict) else "N/A",
            )

    return findings


def _resolve_category(raw: str) -> str:
    """Map an LLM-returned category string to a SecurityCategory value."""
    mapping = {v.value: v.value for v in SecurityCategory}
    # Exact match
    if raw in mapping:
        return raw
    # Case-insensitive fuzzy match
    raw_lower = raw.lower()
    for v in SecurityCategory:
        if raw_lower in v.value.lower() or v.value.lower() in raw_lower:
            return v.value
    return SecurityCategory.SECURITY_MISCONFIGURATION.value


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


def _sort_findings(findings: list[SecurityFinding]) -> list[SecurityFinding]:
    return sorted(findings, key=lambda f: (_severity_index(Severity(f.severity)), f.file_path))


def _deduplicate(findings: list[SecurityFinding]) -> list[SecurityFinding]:
    """
    Remove duplicates:
      - Exact key: (file_path, start_line, category)
      - Proximity: same (file_path, category) within 5 lines
    """
    seen: set[tuple] = set()
    exact: list[SecurityFinding] = []
    for f in findings:
        key = (f.file_path, f.start_line, f.category)
        if key not in seen:
            seen.add(key)
            exact.append(f)

    return proximity_deduplicate(
        exact,
        key_fn=lambda f: (f.file_path, f.category),
        line_fn=lambda f: f.start_line,
        window=5,
    )
