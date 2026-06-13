#!/usr/bin/env python3
"""Print sample false-positive hits for target rules."""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from agents.bug_agent import _scan_chunks, _RULES
from agents.security_agent import _scan_chunks as sec_scan
from schemas import SupportedLanguage
from tools.chunker import chunk_file, detect_language, is_allowed_file
from tools.github_tools import fetch_repo, list_source_files, cleanup_repo

TARGET = {"BUG-001", "BUG-003", "BUG-020", "SEC-030", "SEC-040", "SEC-070"}


async def diagnose(url: str, name: str, limit: int = 8) -> None:
    repo = fetch_repo(url, tempfile.mkdtemp(prefix=f"cg-diag-{name}-"))
    try:
        for sf in list_source_files(repo):
            if not is_allowed_file(sf.rel_path) or detect_language(sf.rel_path) != SupportedLanguage.PYTHON:
                continue
            chunks = chunk_file(sf.rel_path, sf.content, SupportedLanguage.PYTHON, 80)
            for hit in _scan_chunks(chunks):
                if hit.rule.rule_id not in TARGET:
                    continue
                print(f"\n=== {hit.rule.rule_id} {sf.rel_path}:{hit.line_no} ===")
                print(f"  match: {hit.match_text!r}")
                lines = sf.content.splitlines()
                rel = hit.line_no - chunks[0].start_line if chunks else hit.line_no - 1
                for i in range(max(0, rel - 1), min(len(lines), rel + 2)):
                    print(f"  {i+1:4d}| {lines[i]}")
            for hit in sec_scan(chunks):
                if hit.rule.rule_id not in TARGET:
                    continue
                print(f"\n=== {hit.rule.rule_id} {sf.rel_path}:{hit.line_no} ===")
                print(f"  match: {hit.match_text!r}")
                lines = sf.content.splitlines()
                rel = hit.line_no - 1
                for i in range(max(0, rel - 1), min(len(lines), rel + 2)):
                    print(f"  {i+1:4d}| {lines[i]}")
    finally:
        cleanup_repo(repo)


if __name__ == "__main__":
    asyncio.run(diagnose("https://github.com/pallets/flask", "flask"))
