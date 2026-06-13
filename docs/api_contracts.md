# CodeGuardian AI — API Contracts

> **Version:** 1.0.0 | **Status:** Canonical Reference  
> All request/response shapes are governed by `schemas.py`. Any deviation is a bug.

---

## Base URL

```
Development : http://localhost:8000
Production  : https://api.codeguardian.ai
```

## Global Headers

| Header | Required | Value |
|--------|----------|-------|
| `Content-Type` | Yes (POST) | `application/json` |
| `Accept` | Optional | `application/json` |
| `X-API-Key` | Yes (non-health) | API key from environment |
| `X-Request-ID` | Optional | UUID for distributed tracing |

## Global Error Response

All 4xx and 5xx responses return:

```json
{
  "code": "JOB_NOT_FOUND",
  "message": "No job exists with id '3fa85f64-5717-...'",
  "details": null
}
```

**Schema:** `ErrorDetail`

| HTTP Code | `code` value | When |
|-----------|-------------|------|
| `400` | `VALIDATION_ERROR` | Request body fails Pydantic validation |
| `400` | `INVALID_SOURCE` | Neither source_url nor source_zip_b64 provided |
| `400` | `BOTH_SOURCES_PROVIDED` | Both fields provided simultaneously |
| `400` | `PAYLOAD_TOO_LARGE` | ZIP decoded size > 50 MB |
| `401` | `UNAUTHORIZED` | Missing or invalid X-API-Key |
| `404` | `JOB_NOT_FOUND` | job_id does not exist in the job store |
| `409` | `JOB_ALREADY_EXISTS` | Idempotency key collision (future) |
| `422` | `UNPROCESSABLE_ENTITY` | FastAPI automatic body parse failure |
| `429` | `RATE_LIMITED` | Exceeds per-key rate limit |
| `500` | `INTERNAL_ERROR` | Unhandled server exception |
| `503` | `SERVICE_UNAVAILABLE` | LLM backend or vector store unavailable |

---

---

## `POST /review`

**Purpose:** Submit a repository for AI-powered code review. Returns immediately with a `job_id`; the analysis runs asynchronously.

### Request

```
POST /review
Content-Type: application/json
X-API-Key: <key>
```

**Schema:** `ReviewRequest`

```json
{
  "repository_name": "my-service",
  "source_url": "https://github.com/acme/my-service",
  "source_zip_b64": null,
  "config": {
    "max_chunk_lines": 80,
    "enable_bug_analysis": true,
    "enable_solid_analysis": true,
    "enable_architecture_analysis": true,
    "enable_security_analysis": true,
    "enable_complexity_analysis": true,
    "similarity_top_k": 3,
    "llm_temperature": 0.1
  }
}
```

**Field Rules:**

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `repository_name` | string | ✅ | `len` 1–256 |
| `source_url` | string \| null | One of | Valid HTTP/HTTPS URL |
| `source_zip_b64` | string \| null | One of | Base64; decoded ≤ 50 MB |
| `config` | object | ❌ | Defaults applied |
| `config.max_chunk_lines` | int | ❌ | 20–500, default 80 |
| `config.similarity_top_k` | int | ❌ | 1–10, default 3 |
| `config.llm_temperature` | float | ❌ | 0.0–1.0, default 0.1 |

> **Constraint:** Exactly one of `source_url` or `source_zip_b64` must be non-null.

### Response — `202 Accepted`

**Schema:** `ReviewResponse`

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "PENDING",
  "poll_url": "http://localhost:8000/status/3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "estimated_duration_seconds": 45
}
```

| Field | Type | Notes |
|-------|------|-------|
| `job_id` | UUID string | Use this for all subsequent calls |
| `status` | `"PENDING"` | Always PENDING at submission time |
| `poll_url` | string | Absolute URL to GET /status/{job_id} |
| `estimated_duration_seconds` | int \| null | Rough estimate; not guaranteed |

### Error Responses

| Status | Code | Condition |
|--------|------|-----------|
| `400` | `VALIDATION_ERROR` | `repository_name` empty, invalid config values |
| `400` | `INVALID_SOURCE` | Both `source_url` and `source_zip_b64` are null |
| `400` | `BOTH_SOURCES_PROVIDED` | Both fields are non-null |
| `400` | `PAYLOAD_TOO_LARGE` | ZIP > 50 MB |
| `401` | `UNAUTHORIZED` | Bad API key |

---

---

## `GET /status/{job_id}`

**Purpose:** Poll for current job lifecycle state and progress percentage. When `status` is `COMPLETED`, the full `ReviewResult` is embedded inline.

### Request

```
GET /status/{job_id}
X-API-Key: <key>
```

**Path Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `job_id` | UUID string | Returned from `POST /review` |

### Response — `200 OK`

**Schema:** `JobStatusResponse`

#### While Running

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "RUNNING",
  "progress": 45,
  "result": null,
  "error": null,
  "created_at": "2026-06-10T02:20:43Z",
  "started_at": "2026-06-10T02:20:44Z",
  "completed_at": null
}
```

#### On Completion

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "COMPLETED",
  "progress": 100,
  "result": { /* ReviewResult — see GET /report/{job_id} */ },
  "error": null,
  "created_at": "2026-06-10T02:20:43Z",
  "started_at": "2026-06-10T02:20:44Z",
  "completed_at": "2026-06-10T02:21:31Z"
}
```

#### On Failure

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "FAILED",
  "progress": 20,
  "result": null,
  "error": "LLM backend returned 503: Service Unavailable",
  "created_at": "2026-06-10T02:20:43Z",
  "started_at": "2026-06-10T02:20:44Z",
  "completed_at": "2026-06-10T02:20:51Z"
}
```

**`status` Transition Table:**

```
PENDING → RUNNING → COMPLETED
                  ↘ FAILED
```

**Recommended Polling Strategy:**

| Elapsed Time | Poll Interval |
|-------------|--------------|
| 0–10s | Every 2s |
| 10–60s | Every 5s |
| 60s+ | Every 15s |

### Error Responses

| Status | Code | Condition |
|--------|------|-----------|
| `404` | `JOB_NOT_FOUND` | `job_id` not in store or TTL expired |
| `401` | `UNAUTHORIZED` | Bad API key |

---

---

## `GET /report/{job_id}`

**Purpose:** Retrieve the final, complete `ReviewResult` for a completed job. Returns `409` if the job has not yet completed.

### Request

```
GET /report/{job_id}
X-API-Key: <key>
```

### Response — `200 OK`

**Schema:** `ReviewResult`

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "metadata": {
    "repository_name": "my-service",
    "source_url": "https://github.com/acme/my-service",
    "primary_language": "python",
    "language_breakdown": { "python": 78.4, "yaml": 12.1, "shell": 9.5 },
    "total_files": 42,
    "total_lines": 8430,
    "analyzed_at": "2026-06-10T02:21:31Z"
  },
  "bug_findings": [
    {
      "finding_id": "a1b2c3d4-...",
      "severity": "HIGH",
      "title": "Unchecked return value from file.read()",
      "description": "The return value of file.read() is never checked for None ...",
      "file_path": "src/parser/reader.py",
      "start_line": 47,
      "end_line": 52,
      "suggested_fix": "Add a null check: `data = file.read(); if data is None: raise IOError(...)`",
      "related_chunk_ids": ["c1d2...", "e3f4..."],
      "confidence": 0.91,
      "bug_pattern": "UncheckedReturnValue",
      "reproducible": true
    }
  ],
  "solid_findings": [ /* List[SolidFinding] */ ],
  "architecture_findings": [ /* List[ArchitectureFinding] */ ],
  "security_findings": [
    {
      "finding_id": "b2c3d4e5-...",
      "severity": "CRITICAL",
      "title": "Hardcoded database password in config.py",
      "description": "A plaintext database password is embedded directly in version-controlled source ...",
      "file_path": "config.py",
      "start_line": 12,
      "end_line": 12,
      "suggested_fix": "Use os.environ.get('DB_PASSWORD') and store secrets in a vault.",
      "related_chunk_ids": [],
      "confidence": 0.99,
      "category": "Hardcoded Secret / Credential",
      "cwe_id": "CWE-798",
      "cvss_score": 9.1,
      "exploitability": "Trivial",
      "requires_user_interaction": false
    }
  ],
  "complexity_findings": [ /* List[ComplexityFinding] */ ],
  "overall_score": 61.0,
  "summary_markdown": "## Executive Summary\n\nThe `my-service` codebase ...",
  "reviewed_at": "2026-06-10T02:21:31Z",
  "total_findings": 23,
  "critical_count": 1
}
```

**Finding Severity Distribution:** `CRITICAL > HIGH > MEDIUM > LOW > INFO`

### Error Responses

| Status | Code | Condition |
|--------|------|-----------|
| `404` | `JOB_NOT_FOUND` | `job_id` unknown or expired |
| `409` | `REPORT_NOT_READY` | Job status is PENDING or RUNNING |
| `409` | `JOB_FAILED` | Job status is FAILED; use `GET /status` for error details |
| `401` | `UNAUTHORIZED` | Bad API key |

---

---

## `GET /health`

**Purpose:** Liveness and readiness probe. No authentication required.

### Request

```
GET /health
```

### Response — `200 OK`

**Schema:** `HealthResponse`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 3847.2,
  "active_jobs": 2,
  "vector_store_ready": true
}
```

### Response — `503 Service Unavailable` (degraded)

```json
{
  "status": "degraded",
  "version": "1.0.0",
  "uptime_seconds": 3847.2,
  "active_jobs": 0,
  "vector_store_ready": false
}
```

| Field | Type | Notes |
|-------|------|-------|
| `status` | `"ok"` \| `"degraded"` | `"degraded"` triggers 503 HTTP code |
| `version` | string | Semver from `__version__` |
| `uptime_seconds` | float | Seconds since process start |
| `active_jobs` | int | Jobs in RUNNING state |
| `vector_store_ready` | bool | SentenceTransformer model loaded |

---

## OpenAPI Schema Location

When the FastAPI server is running:

```
Swagger UI : http://localhost:8000/docs
ReDoc      : http://localhost:8000/redoc
OpenAPI JSON: http://localhost:8000/openapi.json
```
