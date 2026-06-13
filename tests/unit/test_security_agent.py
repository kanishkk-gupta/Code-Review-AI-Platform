"""
tests/unit/test_security_agent.py
===================================
Unit tests for agents/security_agent.py — fully offline, no LLM calls.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from schemas import CodeChunk, SecurityCategory, SecurityFinding, Severity, SupportedLanguage
from agents.security_agent import (
    SecurityAgent,
    _ScanHit,
    _deduplicate,
    _hit_to_finding,
    _parse_llm_output,
    _resolve_category,
    _scan_chunks,
    _severity_index,
    _sort_findings,
    _RULES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    content: str,
    file_path: str = "src/main.py",
    language: SupportedLanguage = SupportedLanguage.PYTHON,
    start_line: int = 1,
    end_line: int | None = None,
    chunk_id: str = "test-chunk-001",
) -> CodeChunk:
    lines = content.count("\n") + 1
    return CodeChunk(
        chunk_id=chunk_id,
        file_path=file_path,
        language=language,
        content=content,
        start_line=start_line,
        end_line=end_line or (start_line + lines - 1),
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Rule Coverage — one test per vulnerability class
# ---------------------------------------------------------------------------

class TestRuleEngine:

    def _hits_for(self, content: str) -> list[_ScanHit]:
        return _scan_chunks([_chunk(content)])

    def _hit_rule_ids(self, content: str) -> set[str]:
        return {h.rule.rule_id for h in self._hits_for(content)}

    # Hardcoded secrets
    def test_detects_hardcoded_password(self):
        assert "SEC-001" in self._hit_rule_ids('password = "SuperSecret123xyz"')

    def test_suppresses_test_fixture_secret_key(self):
        code = 'SECRET_KEY = "test key"\n'
        chunk = _chunk(code, file_path="tests/test_config.py")
        hits = _scan_chunks([chunk])
        assert not any(h.rule.rule_id == "SEC-001" for h in hits)

    def test_suppresses_test_config_secret(self):
        code = 'SECRET_KEY = "config"\n'
        chunk = _chunk(code, file_path="tests/conftest.py")
        hits = _scan_chunks([chunk])
        assert not any(h.rule.rule_id == "SEC-001" for h in hits)

    def test_production_secret_still_detected(self):
        code = 'SECRET_KEY = "SuperSecret123xyz"\n'
        chunk = _chunk(code, file_path="src/settings.py")
        hits = _scan_chunks([chunk])
        assert any(h.rule.rule_id == "SEC-001" for h in hits)

    def test_regex_skips_non_python_chunks(self):
        js = 'var secret = "SuperSecret123xyz";\n'
        chunk = _chunk(
            js,
            file_path="src/app.js",
            language=SupportedLanguage.JAVASCRIPT,
        )
        hits = _scan_chunks([chunk])
        assert hits == []

    def test_regex_skips_static_javascript(self):
        js = 'eval(userInput);\n'
        chunk = _chunk(
            js,
            file_path="static/admin/js/actions.js",
            language=SupportedLanguage.JAVASCRIPT,
        )
        hits = _scan_chunks([chunk])
        assert hits == []

    def test_detects_hardcoded_api_key(self):
        assert "SEC-001" in self._hit_rule_ids('api_key = "abcdef1234567890"')

    def test_detects_private_key_header(self):
        assert "SEC-002" in self._hit_rule_ids("-----BEGIN RSA PRIVATE KEY-----")

    def test_detects_github_token(self):
        assert "SEC-002" in self._hit_rule_ids("token = 'ghp_" + "A" * 36 + "'")

    def test_detects_generic_secret(self):
        assert "SEC-003" in self._hit_rule_ids('secret: "MyLongSecretValue1234567890"')

    # SQL injection
    def test_detects_sql_concat(self):
        assert "SEC-010" in self._hit_rule_ids(
            'cursor.execute("SELECT * FROM users WHERE id=" + user_id)'
        )

    def test_detects_django_raw_format(self):
        assert "SEC-011" in self._hit_rule_ids(
            'User.objects.raw("SELECT * FROM users WHERE name=%s" % name)'
        )

    # Command injection
    def test_detects_subprocess_shell_true(self):
        assert "SEC-020" in self._hit_rule_ids(
            "subprocess.run(cmd, shell=True)"
        )

    def test_detects_os_system_format(self):
        assert "SEC-021" in self._hit_rule_ids(
            'os.system("rm -rf " + user_path)'
        )

    def test_detects_eval_user_input(self):
        assert "SEC-022" in self._hit_rule_ids(
            "result = eval(request.GET['expr'])"
        )

    # Unsafe file operations
    def test_detects_path_traversal(self):
        assert "SEC-030" in self._hit_rule_ids(
            'open(os.path.join("/uploads", request.args["filename"]))'
        )

    # SSRF
    def test_detects_ssrf_pattern(self):
        assert "SEC-040" in self._hit_rule_ids(
            "requests.get(url=user_supplied_url)"
        )

    # Insecure deserialization
    def test_detects_pickle_loads(self):
        assert "SEC-050" in self._hit_rule_ids(
            "data = pickle.loads(user_bytes)"
        )

    def test_detects_unsafe_yaml_load(self):
        assert "SEC-050" in self._hit_rule_ids(
            "config = yaml.load(request.data)"
        )

    def test_ignores_static_yaml_load(self):
        assert "SEC-050" not in self._hit_rule_ids(
            "config = yaml.load(stream)"
        )

    # Weak crypto
    def test_detects_md5(self):
        assert "SEC-060" in self._hit_rule_ids(
            "digest = hashlib.md5(data).hexdigest()"
        )

    def test_detects_sha1(self):
        assert "SEC-060" in self._hit_rule_ids(
            "sig = hashlib.sha1(payload).hexdigest()"
        )

    # Debug mode
    def test_detects_debug_true(self):
        assert "SEC-070" in self._hit_rule_ids("DEBUG = True")

    def test_detects_flask_debug(self):
        assert "SEC-070" in self._hit_rule_ids('app.run(host="0.0.0.0", debug=True)')

    # Negative — clean code
    def test_clean_code_no_hits(self):
        clean = "def add(a, b):\n    return a + b\n"
        assert self._hits_for(clean) == []

    def test_parameterized_sql_no_hit(self):
        safe = 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))'
        ids = self._hit_rule_ids(safe)
        assert "SEC-010" not in ids

    def test_one_hit_per_rule_per_chunk(self):
        # Multiple matching lines — should only produce one hit per rule
        content = (
            'password = "abc12345"\n'
            'secret   = "def67890"\n'
        )
        hits = self._hits_for(content)
        sec001_hits = [h for h in hits if h.rule.rule_id == "SEC-001"]
        assert len(sec001_hits) == 1


# ---------------------------------------------------------------------------
# _hit_to_finding
# ---------------------------------------------------------------------------

class TestHitToFinding:
    def _make_hit(self, content='password = "SuperSecret123xyz"') -> _ScanHit:
        c = _chunk(content)
        hits = _scan_chunks([c])
        return hits[0]

    def test_returns_security_finding(self):
        hit = self._make_hit()
        f = _hit_to_finding(hit)
        assert isinstance(f, SecurityFinding)

    def test_severity_matches_rule(self):
        hit = self._make_hit()
        f = _hit_to_finding(hit)
        assert f.severity == hit.rule.severity.value

    def test_cwe_id_set(self):
        hit = self._make_hit()
        f = _hit_to_finding(hit)
        assert f.cwe_id and f.cwe_id.startswith("CWE-")

    def test_confidence_is_rule_level(self):
        hit = self._make_hit()
        f = _hit_to_finding(hit)
        assert 0.0 < f.confidence <= 1.0

    def test_file_path_preserved(self):
        c = _chunk('password = "SuperSecret123xyz"', file_path="auth/login.py")
        hit = _scan_chunks([c])[0]
        f = _hit_to_finding(hit)
        assert f.file_path == "auth/login.py"

    def test_suggested_fix_not_empty(self):
        hit = self._make_hit()
        f = _hit_to_finding(hit)
        assert f.suggested_fix and len(f.suggested_fix) > 10


# ---------------------------------------------------------------------------
# _parse_llm_output
# ---------------------------------------------------------------------------

class TestParseLlmOutput:
    def _make_hit(self) -> _ScanHit:
        c = _chunk('password = "SuperSecret123xyz"')
        return _scan_chunks([c])[0]

    def _valid_payload(self) -> dict:
        return {
            "severity": "HIGH",
            "title": "Hardcoded Password",
            "description": "A password is hardcoded in the source code.",
            "file_path": "src/main.py",
            "start_line": 1,
            "end_line": 1,
            "suggested_fix": "Use environment variables.",
            "confidence": 0.9,
            "category": "Hardcoded Secret / Credential",
            "cwe_id": "CWE-798",
            "cvss_score": 9.8,
            "exploitability": "Trivial",
            "requires_user_interaction": False,
        }

    def test_parses_valid_array(self):
        hit = self._make_hit()
        result = _parse_llm_output([self._valid_payload()], hit)
        assert len(result) == 1
        assert isinstance(result[0], SecurityFinding)

    def test_parses_single_dict(self):
        hit = self._make_hit()
        result = _parse_llm_output(self._valid_payload(), hit)
        assert len(result) == 1

    def test_empty_array_returns_empty(self):
        hit = self._make_hit()
        assert _parse_llm_output([], hit) == []

    def test_missing_file_path_uses_chunk(self):
        hit = self._make_hit()
        payload = self._valid_payload()
        del payload["file_path"]
        result = _parse_llm_output([payload], hit)
        assert result[0].file_path == hit.chunk.file_path

    def test_malformed_item_skipped(self):
        hit = self._make_hit()
        # Missing required 'title'
        bad = {"severity": "HIGH", "description": "x", "file_path": "f.py",
               "start_line": 1, "end_line": 1, "category": "Injection"}
        result = _parse_llm_output([bad], hit)
        # Should not raise; malformed items are skipped
        assert isinstance(result, list)

    def test_non_list_non_dict_returns_empty(self):
        hit = self._make_hit()
        assert _parse_llm_output("not a list", hit) == []


# ---------------------------------------------------------------------------
# _resolve_category
# ---------------------------------------------------------------------------

class TestResolveCategory:
    def test_exact_match(self):
        assert _resolve_category("Injection") == "Injection"

    def test_case_insensitive(self):
        result = _resolve_category("injection")
        assert result == SecurityCategory.INJECTION.value

    def test_partial_match_hardcoded(self):
        result = _resolve_category("Hardcoded Secret")
        assert "Hardcoded" in result

    def test_unknown_falls_back(self):
        result = _resolve_category("totally unknown category xyz")
        assert result == SecurityCategory.SECURITY_MISCONFIGURATION.value


# ---------------------------------------------------------------------------
# _deduplicate / _sort_findings / _severity_index
# ---------------------------------------------------------------------------

class TestHelpers:
    def _sf(self, severity="HIGH", file_path="f.py", start_line=1, cat="Injection"):
        return SecurityFinding(
            severity=severity,
            title="Test finding title here",
            description="A test security finding description text.",
            file_path=file_path,
            start_line=start_line,
            end_line=start_line,
            category=cat,
            exploitability="Unknown",
        )

    def test_deduplicate_removes_exact_duplicate(self):
        f1 = self._sf(start_line=5)
        f2 = self._sf(start_line=5)
        result = _deduplicate([f1, f2])
        assert len(result) == 1

    def test_deduplicate_keeps_different_lines(self):
        f1 = self._sf(start_line=1)
        f2 = self._sf(start_line=20)
        assert len(_deduplicate([f1, f2])) == 2

    def test_deduplicate_keeps_different_categories(self):
        f1 = self._sf(cat="Injection")
        f2 = self._sf(cat="Hardcoded Secret / Credential")
        assert len(_deduplicate([f1, f2])) == 2

    def test_sort_findings_critical_first(self):
        findings = [self._sf("LOW"), self._sf("CRITICAL"), self._sf("HIGH")]
        sorted_f = _sort_findings(findings)
        assert sorted_f[0].severity == "CRITICAL"

    def test_severity_index_order(self):
        assert _severity_index(Severity.CRITICAL) < _severity_index(Severity.HIGH)
        assert _severity_index(Severity.HIGH) < _severity_index(Severity.MEDIUM)
        assert _severity_index(Severity.MEDIUM) < _severity_index(Severity.LOW)


# ---------------------------------------------------------------------------
# SecurityAgent.run — integration (LLM mocked)
# ---------------------------------------------------------------------------

class TestSecurityAgentRun:

    def test_empty_chunks_returns_empty(self):
        agent = SecurityAgent()
        result = _run(agent.run([], llm_confirm=False))
        assert result == []

    def test_clean_code_returns_empty(self):
        agent = SecurityAgent()
        clean = _chunk("def add(a, b):\n    return a + b\n")
        result = _run(agent.run([clean], llm_confirm=False))
        assert result == []

    def test_rule_only_mode_returns_findings(self):
        agent = SecurityAgent()
        vuln = _chunk('password = "SuperSecret123"\n')
        result = _run(agent.run([vuln], llm_confirm=False))
        assert len(result) > 0
        assert all(isinstance(f, SecurityFinding) for f in result)

    def test_findings_sorted_by_severity(self):
        agent = SecurityAgent()
        code = (
            'password = "SuperSecret123"\n'
            'hashlib.md5(data)\n'
            'DEBUG = True\n'
        )
        vuln = _chunk(code)
        result = _run(agent.run([vuln], llm_confirm=False))
        assert len(result) >= 2
        idxs = [_severity_index(Severity(f.severity)) for f in result]
        assert idxs == sorted(idxs)

    def test_llm_confirm_false_skips_chain(self):
        agent = SecurityAgent()
        with patch.object(agent, "_build_chain") as mock_chain:
            vuln = _chunk('password = "SuperSecret123"\n')
            _run(agent.run([vuln], llm_confirm=False))
            mock_chain.assert_not_called()

    def test_llm_unavailable_falls_back_to_rules(self):
        agent = SecurityAgent()
        with patch.object(agent, "_build_chain", side_effect=ImportError("no llm")):
            vuln = _chunk('password = "SuperSecret123"\n')
            result = _run(agent.run([vuln], llm_confirm=True))
            # Must still return rule-based findings
            assert len(result) > 0

    def test_to_json_returns_valid_json(self):
        agent = SecurityAgent()
        vuln = _chunk('password = "SuperSecret123"\n')
        findings = _run(agent.run([vuln], llm_confirm=False))
        js = agent.to_json(findings)
        parsed = json.loads(js)
        assert isinstance(parsed, list)
        assert all("severity" in f for f in parsed)

    def test_multiple_files_all_flagged(self):
        agent = SecurityAgent()
        chunks = [
            _chunk('password = "SuperSecret123xyz"', file_path="auth.py", chunk_id="c1"),
            _chunk('os.system("ls " + user_input)', file_path="utils.py", chunk_id="c2"),
        ]
        result = _run(agent.run(chunks, llm_confirm=False))
        files = {f.file_path for f in result}
        assert "auth.py" in files
        assert "utils.py" in files

    def test_no_duplicate_findings(self):
        agent = SecurityAgent()
        vuln = _chunk('password = "SuperSecret123"\n')
        result = _run(agent.run([vuln], llm_confirm=False))
        keys = [(f.file_path, f.start_line, f.category) for f in result]
        assert len(keys) == len(set(keys))
