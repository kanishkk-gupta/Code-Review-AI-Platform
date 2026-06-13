#!/usr/bin/env python3
"""Offline benchmark: rule-only analysis against real repositories."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.architecture_agent import ArchitectureAgent
from agents.bug_agent import BugAgent
from agents.complexity_agent import ComplexityAgent
from agents.security_agent import SecurityAgent
from agents.solid_agent import SolidAgent
from schemas import Severity, SupportedLanguage
from tools.chunker import chunk_file, detect_language, is_allowed_file
from tools.github_tools import fetch_repo, list_source_files, cleanup_repo


TARGET_RULES = {
    "BUG-001", "BUG-002", "BUG-003", "BUG-010", "BUG-011", "BUG-020",
    "SEC-001", "SEC-010", "SEC-030", "SEC-040", "SEC-050", "SEC-070",
}

BENCHMARK_REPOS = [
    ("requests", "https://github.com/psf/requests"),
    ("flask", "https://github.com/pallets/flask"),
    ("django", "https://github.com/django/django"),
    ("fastapi", "https://github.com/tiangolo/fastapi"),
    ("pydantic", "https://github.com/pydantic/pydantic"),
    ("click", "https://github.com/pallets/click"),
    ("typer", "https://github.com/fastapi/typer"),
]


def _chunks_from_repo(repo) -> list:
    chunks = []
    for sf in list_source_files(repo):
        if not is_allowed_file(sf.rel_path):
            continue
        lang = detect_language(sf.rel_path)
        if lang != SupportedLanguage.PYTHON:
            continue
        chunks.extend(chunk_file(sf.rel_path, sf.content, lang, max_lines=80))
    return chunks


def _rule_id(title: str) -> str | None:
    if title.startswith("[") and "]" in title:
        return title[1 : title.index("]")]
    return None


def _validation_issues(findings_by_agent: dict) -> list[str]:
    issues: list[str] = []
    for f in findings_by_agent.get("bug", []):
        rid = _rule_id(f.title)
        if rid == "BUG-002" and "client.get" in f.description.lower():
            issues.append(f"BUG-002 client.get FP: {f.file_path}")
        if rid == "BUG-010" and any(
            x in f.description for x in ("Division by variable: A", "/ Custom", "/ Global", "/ await", "/ chars")
        ):
            issues.append(f"BUG-010 FP: {f.file_path}")
        if rid == "BUG-001" and "None.__class__" in f.description:
            issues.append(f"BUG-001 None.__class__ FP: {f.file_path}")
    for f in findings_by_agent.get("architecture", []):
        fp = f.file_path.replace("\\", "/")
        if fp.startswith("tests/") or "/tests/" in fp:
            issues.append(f"Architecture test FP: {f.file_path}")
        if any(x in fp for x in ("pydantic/main.py", "click/core.py", "typer/main.py")):
            if "God" in f.title:
                issues.append(f"Architecture framework entrypoint FP: {f.file_path}")
    for f in findings_by_agent.get("solid", []):
        if f.file_path.replace("\\", "/").startswith("tests/"):
            issues.append(f"SOLID test FP: {f.file_path}")
    for f in findings_by_agent.get("complexity", []):
        if "<chunk:" in f.title or "<chunk:" in (f.function_name or ""):
            issues.append(f"Chunk complexity FP: {f.title}")
    for f in findings_by_agent.get("security", []):
        fp = f.file_path.replace("\\", "/")
        if any(p in fp for p in ("tests/", "docs/", "examples/")) and f.severity in ("MEDIUM", "HIGH", "CRITICAL"):
            if "SECRET" in f.title.upper() or "password" in f.description.lower():
                issues.append(f"Test secret MEDIUM+ FP: {f.file_path} {f.severity}")
        if _rule_id(f.title) == "SEC-050" and "pickle" in f.description.lower() and "tests/" not in fp:
            pass  # legitimate
    return issues


async def analyze_repo(name: str, url: str) -> dict:
    repo = fetch_repo(url, tempfile.mkdtemp(prefix=f"cg-{name}-"))
    try:
        chunks = _chunks_from_repo(repo)
        bug = await BugAgent().run(chunks, llm_confirm=False)
        sec = await SecurityAgent().run(chunks, llm_confirm=False)
        cpx = await ComplexityAgent().run(chunks, min_severity=Severity.HIGH)
        arch = await ArchitectureAgent().analyse_repository(chunks, llm_synthesize=False)
        solid = await SolidAgent().run(chunks, llm_confirm=False)

        by_agent = {
            "bug": bug, "security": sec, "complexity": cpx,
            "architecture": arch, "solid": solid,
        }

        by_rule: Counter = Counter()
        target_hits: Counter = Counter()
        agent_totals = {k: len(v) for k, v in by_agent.items()}

        for kind, findings in by_agent.items():
            for f in findings:
                rid = _rule_id(f.title)
                if rid:
                    by_rule[rid] += 1
                    if rid in TARGET_RULES:
                        target_hits[rid] += 1

        return {
            "repo": name,
            "chunks": len(chunks),
            "total": sum(agent_totals.values()),
            **agent_totals,
            "by_rule": dict(by_rule.most_common()),
            "target_rules": dict(target_hits),
            "validation_issues": _validation_issues(by_agent),
        }
    finally:
        cleanup_repo(repo)


async def main() -> None:
    results = []
    for name, url in BENCHMARK_REPOS:
        print(f"Analyzing {name}...", flush=True)
        results.append(await analyze_repo(name, url))

    summary = {
        "repos": results,
        "totals": {
            "findings": sum(r["total"] for r in results),
            "validation_issues": sum(len(r["validation_issues"]) for r in results),
        },
        "by_repo": {r["repo"]: r["total"] for r in results},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
