# Post-Cleanup Report

This report documents the results of executing the approved Version A Release-Readiness Cleanup plan on the `codeguardian-ai-cleanup` directory.

## 1. Operations Summary

### Files Deleted
The following items were permanently deleted:
- `.pytest_cache/` (Directory)
- `reports/output/report_*.md` (Benchmark artifacts)
- `prompts/complexity_analysis.jinja2` (Unused prompt)
- `.env` (Secret configuration file)

*(Note: Duplicate scratch scripts from the parent workspace were outside the scope of `codeguardian-ai-cleanup` and were not touched).*

### Files Archived
The following files were moved to the newly created `archive/` directory:

**Moved to `archive/graph_nodes_legacy/`:**
- `graph/nodes/ingest_node.py`
- `graph/nodes/analyze_node.py`
- `graph/nodes/enrich_node.py`
- `graph/nodes/compile_node.py`
- `graph/nodes/__init__.py`

**Moved to `archive/internal_audits/`:**
- `SYSTEM_AUDIT.md`
- `scripts/benchmark_repos.py`
- `scripts/diagnose_fps.py`

**Moved to `archive/misc/`:**
- `tools/language_detector.py`
- `api/middleware/auth.py`

## 2. File Counts
- **Total Files Before Cleanup:** 81,194 (Includes `venv`)
- **Total Files After Cleanup:** 81,180
- **Total Removed/Moved:** 14 net files removed from active paths.

## 3. Verification Checks

### Python Import Check (`compileall`)
- **Status:** PASS (with warnings)
- **Notes:** Completed successfully. A `SyntaxWarning: invalid escape sequence` was emitted from an internal LangChain package (`langchain\agents\json_chat\base.py`), but this is a 3rd party dependency issue and does not affect runtime.

### Workflow Compile Check
- **Status:** PASS
- **Notes:** Running `from graph.workflow import get_workflow; get_workflow()` successfully executed and emitted `workflow_compiled`.

### FastAPI Startup Check
- **Status:** PASS
- **Notes:** Running `from api.main import app` successfully imports the FastAPI instance (Note: Requires `API_KEY` to be passed in the environment since `.env` was deleted).

### Pytest Collection Check
- **Status:** FAIL (5 Collection Errors)
- **Notes:** `pytest --collect-only` failed due to a newly broken import caused by the archiving operation.

## 4. Broken Imports Found
Archiving `tools/language_detector.py` broke `tools/__init__.py`, which still attempts to re-export it.

**Traceback:**
```python
ImportError while importing test module '...\tests\unit\test_parser.py'.
Traceback:
tools\__init__.py:4: in <module>
    from tools.language_detector import compute_language_breakdown
ModuleNotFoundError: No module named 'tools.language_detector'
```

Per the strict instructions ("Do not fix code. Only execute cleanup and report results"), this import has been left broken. You will need to remove line 4 from `tools/__init__.py` to restore green CI tests.
