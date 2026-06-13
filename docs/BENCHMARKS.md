# Benchmarks & Precision Audit

During the finalization of the Version 1.0 rule-based engine, we conducted a rigorous precision audit against several major open-source Python repositories.

### Benchmark Scope

These benchmarks measure:
* workflow stability
* parser correctness
* analysis coverage
* rule precision

They do NOT represent:
* true-positive rate
* recall
* security certification
* production security guarantees

## 1. Tested Repositories
The following repositories were subjected to full end-to-end processing to evaluate agent precision, workflow stability, and parsing reliability:
- `fastapi`
- `django`
- `flask`
- `typer`
- `pydantic`

## 2. Findings Summary
- **Complexity Accuracy (High):** The `ComplexityAgent`, powered by `radon`, exhibits highly deterministic and accurate metric reporting. It reliably surfaces long, deeply nested functions across all frameworks.
- **Architecture Accuracy (Medium):** The `ArchitectureAgent` successfully identifies cyclic dependencies and God-class symptoms via import graph construction. 
- **Security & Bug Accuracy (Low to Medium):** The rule-based engines successfully identify unsafe patterns (e.g., `pickle.loads`, `MD5` hashing, hardcoded strings). However, in Version 1.0, these agents lack semantic understanding.

## 3. Known False Positives (Limitations)
Because Version 1.0 does not utilize an LLM confirmation layer to analyze context, the system generates high noise in specific scenarios:
- **Test Suite Noise:** Test files frequently contain hardcoded secrets, mock credentials, and intentional vulnerability usage (e.g., `pickle.loads`) to test framework boundaries. These are flagged as critical vulnerabilities.
- **JavaScript Regex Collisions:** The BugAgent's division-by-zero regex occasionally misclassifies JavaScript string splits (`/ chars`) and regex literals as mathematical division operations.
- **Optional Dereferencing:** Unguarded `Optional` variables are frequently flagged in test functions where `None` is never realistically passed.

These limitations are slated to be resolved in Version 2.0 via AI-driven synthesis and aggressive test-file heuristics.
