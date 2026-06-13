"""
CodeGuardian AI — Canonical Pydantic V2 Schemas
================================================
Version : 1.0.0
Status  : Canonical Reference — ALL services must import from this module.

All shared data contracts for the CodeGuardian AI platform are defined here.
No service may define its own parallel model for any concept represented below.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Optional

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    computed_field,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Universal severity scale used across all finding types."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ConfidenceLevel(str, Enum):
    """Qualitative confidence band for deterministic rule findings."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class JobStatusEnum(str, Enum):
    """Lifecycle states for a review job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SupportedLanguage(str, Enum):
    """Source languages the analyzer can process."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    UNKNOWN = "unknown"


class SolidPrinciple(str, Enum):
    """The five SOLID design principles."""
    SINGLE_RESPONSIBILITY = "Single Responsibility Principle"
    OPEN_CLOSED = "Open/Closed Principle"
    LISKOV_SUBSTITUTION = "Liskov Substitution Principle"
    INTERFACE_SEGREGATION = "Interface Segregation Principle"
    DEPENDENCY_INVERSION = "Dependency Inversion Principle"


class ArchitectureSmell(str, Enum):
    """Recognized architectural anti-patterns."""
    CYCLIC_DEPENDENCY = "Cyclic Dependency"
    GOD_CLASS = "God Class"
    FEATURE_ENVY = "Feature Envy"
    LAYER_VIOLATION = "Layer Violation"
    TIGHT_COUPLING = "Tight Coupling"
    ANEMIC_DOMAIN = "Anemic Domain Model"
    BIG_BALL_OF_MUD = "Big Ball of Mud"
    DISTRIBUTED_MONOLITH = "Distributed Monolith"


class SecurityCategory(str, Enum):
    """OWASP-aligned security vulnerability categories."""
    INJECTION = "Injection"
    BROKEN_AUTH = "Broken Authentication"
    SENSITIVE_DATA_EXPOSURE = "Sensitive Data Exposure"
    XXE = "XML External Entities"
    BROKEN_ACCESS_CONTROL = "Broken Access Control"
    SECURITY_MISCONFIGURATION = "Security Misconfiguration"
    XSS = "Cross-Site Scripting"
    INSECURE_DESERIALIZATION = "Insecure Deserialization"
    VULNERABLE_COMPONENTS = "Using Vulnerable Components"
    INSUFFICIENT_LOGGING = "Insufficient Logging & Monitoring"
    HARDCODED_SECRET = "Hardcoded Secret / Credential"
    PATH_TRAVERSAL = "Path Traversal"
    SSRF = "Server-Side Request Forgery"


# ---------------------------------------------------------------------------
# Base Config
# ---------------------------------------------------------------------------


class _BaseSchema(BaseModel):
    """Shared Pydantic V2 config for all CodeGuardian schemas."""
    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


# ---------------------------------------------------------------------------
# Core Domain Models
# ---------------------------------------------------------------------------


class RepositoryMetadata(_BaseSchema):
    """
    Top-level metadata extracted from the submitted repository.
    Populated during the `ingest_node` of the LangGraph pipeline.
    """
    repository_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Repository or project name.",
        examples=["my-service"],
    )
    source_url: Optional[str] = Field(
        None,
        description="GitHub/GitLab URL if submitted via URL.",
        examples=["https://github.com/user/repo"],
    )
    primary_language: SupportedLanguage = Field(
        ...,
        description="Dominant programming language detected.",
    )
    language_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Map of language → percentage of total lines.",
        examples=[{"python": 78.4, "yaml": 12.1, "shell": 9.5}],
    )
    total_files: int = Field(..., ge=1, description="Number of source files analyzed.")
    total_lines: int = Field(..., ge=1, description="Total lines of code analyzed.")
    analyzed_at: datetime = Field(default_factory=_utcnow)

    @field_validator("language_breakdown")
    @classmethod
    def validate_percentages(cls, v: dict[str, float]) -> dict[str, float]:
        if v and abs(sum(v.values()) - 100.0) > 1.0:
            raise ValueError("language_breakdown percentages must sum to ~100.")
        return v


class CodeChunk(_BaseSchema):
    """
    A single semantic unit of source code extracted from a file.
    Used as the atomic unit for embedding, retrieval, and analysis.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=False,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,   # required for numpy embedding
    )

    chunk_id: str = Field(default_factory=_new_uuid, description="Unique chunk identifier.")
    file_path: str = Field(..., description="Relative path within repository.", examples=["src/auth/login.py"])
    language: SupportedLanguage = Field(..., description="Language of this file.")
    content: str = Field(
        ...,
        min_length=1,
        description="Raw source code content of this chunk (must preserve indentation).",
    )
    start_line: int = Field(..., ge=1, description="First line number (1-indexed, inclusive).")
    end_line: int = Field(..., ge=1, description="Last line number (1-indexed, inclusive).")
    embedding: Optional[Any] = Field(
        None,
        exclude=True,   # never serialized to JSON
        description="numpy float32 ndarray(384,) — in-process only.",
    )
    related_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs retrieved as semantic neighbors during enrich_node.",
    )

    @model_validator(mode="after")
    def validate_line_range(self) -> "CodeChunk":
        if self.end_line < self.start_line:
            raise ValueError(f"end_line ({self.end_line}) must be >= start_line ({self.start_line}).")
        return self

    @computed_field
    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


# ---------------------------------------------------------------------------
# Finding Base
# ---------------------------------------------------------------------------


class _BaseFinding(_BaseSchema):
    """Abstract base for all finding types. Do not instantiate directly."""
    finding_id: str = Field(default_factory=_new_uuid)
    severity: Severity = Field(..., description="Impact severity of this finding.")
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10, description="Detailed explanation of the finding.")
    file_path: str = Field(..., description="Source file containing this finding.")
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    suggested_fix: Optional[str] = Field(
        None,
        description="Actionable remediation guidance or corrected code snippet.",
    )
    related_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs from FAISS similarity search (set during enrich_node).",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Numeric confidence score for this finding [0.0 – 1.0].",
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.HIGH,
        description="Qualitative confidence band derived from analysis evidence.",
    )
    evidence: Optional[str] = Field(
        None,
        description="Concrete code fact supporting the finding (AST match, taint path, etc.).",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Short explanation of why the pattern is flagged and why it matters.",
    )

    @model_validator(mode="after")
    def sync_confidence_level(self) -> "_BaseFinding":
        """Keep confidence_level aligned with numeric confidence when not explicitly set."""
        if self.confidence >= 0.80:
            object.__setattr__(self, "confidence_level", ConfidenceLevel.HIGH)
        elif self.confidence >= 0.55:
            object.__setattr__(self, "confidence_level", ConfidenceLevel.MEDIUM)
        else:
            object.__setattr__(self, "confidence_level", ConfidenceLevel.LOW)
        return self

    @model_validator(mode="after")
    def validate_line_range(self) -> "_BaseFinding":
        if self.end_line < self.start_line:
            raise ValueError(f"end_line must be >= start_line.")
        return self


# ---------------------------------------------------------------------------
# Typed Findings
# ---------------------------------------------------------------------------


class BugFinding(_BaseFinding):
    """
    A potential defect: null dereference, off-by-one, race condition, etc.
    """
    bug_pattern: str = Field(
        ...,
        description="Short label for the pattern, e.g. 'NullDereference', 'RaceCondition'.",
        examples=["NullDereference", "OffByOne", "UnhandledError"],
    )
    reproducible: bool = Field(
        default=False,
        description="True if the LLM assessed the bug as deterministically reproducible.",
    )


class SolidFinding(_BaseFinding):
    """
    A violation of one of the five SOLID design principles.
    """
    principle: SolidPrinciple = Field(..., description="Which SOLID principle is violated.")
    violated_class_or_function: str = Field(
        ...,
        description="Name of the class, function, or module where the violation occurs.",
    )
    refactor_hint: Optional[str] = Field(
        None,
        description="High-level refactoring strategy (e.g., 'Extract interface', 'Split class').",
    )


class ArchitectureFinding(_BaseFinding):
    """
    A structural issue at the module, package, or system level.
    """
    smell: ArchitectureSmell = Field(..., description="Classified architectural anti-pattern.")
    affected_modules: list[str] = Field(
        default_factory=list,
        description="Modules/packages involved in the architectural issue.",
    )
    impact_radius: str = Field(
        ...,
        description="Estimated blast radius — e.g., 'Local', 'Module-wide', 'System-wide'.",
        examples=["Local", "Module-wide", "System-wide"],
    )


class SecurityFinding(_BaseFinding):
    """
    A security vulnerability aligned to OWASP Top 10 / CWE taxonomy.
    """
    category: SecurityCategory = Field(..., description="OWASP-aligned vulnerability category.")
    cwe_id: Optional[str] = Field(
        None,
        pattern=r"^CWE-\d+$",
        description="Common Weakness Enumeration ID.",
        examples=["CWE-89", "CWE-798"],
    )
    cvss_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=10.0,
        description="CVSS v3 score if calculable.",
    )
    exploitability: str = Field(
        default="Unknown",
        description="Exploitability assessment: 'Trivial', 'Moderate', 'Complex', 'Unknown'.",
    )
    requires_user_interaction: bool = Field(default=False)


class ComplexityFinding(_BaseFinding):
    """
    A code complexity issue — high cyclomatic complexity, deeply nested logic, etc.
    """
    cyclomatic_complexity: Optional[int] = Field(
        None,
        ge=1,
        description="Calculated cyclomatic complexity of the function/method.",
    )
    cognitive_complexity: Optional[int] = Field(
        None,
        ge=0,
        description="Cognitive complexity score (SonarQube model).",
    )
    nesting_depth: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum nesting depth observed in this code block.",
    )
    function_name: str = Field(
        ...,
        description="Name of the function or method flagged.",
    )
    lines_of_code: int = Field(
        ...,
        ge=1,
        description="Number of lines in the flagged function.",
    )


# ---------------------------------------------------------------------------
# Aggregated Review Result
# ---------------------------------------------------------------------------


class ReviewResult(_BaseSchema):
    """
    The complete, immutable output of a finished code review job.
    Serialized and stored in JobStatus.result upon job completion.
    """
    job_id: str = Field(..., description="UUID of the parent review job.")
    metadata: RepositoryMetadata
    bug_findings: list[BugFinding] = Field(default_factory=list)
    solid_findings: list[SolidFinding] = Field(default_factory=list)
    architecture_findings: list[ArchitectureFinding] = Field(default_factory=list)
    security_findings: list[SecurityFinding] = Field(default_factory=list)
    complexity_findings: list[ComplexityFinding] = Field(default_factory=list)
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite quality score from 0 (worst) to 100 (best).",
    )
    summary_markdown: str = Field(
        ...,
        min_length=1,
        description="LLM-generated executive summary in Markdown format.",
    )
    reviewed_at: datetime = Field(default_factory=_utcnow)

    @computed_field
    @property
    def total_findings(self) -> int:
        return (
            len(self.bug_findings)
            + len(self.solid_findings)
            + len(self.architecture_findings)
            + len(self.security_findings)
            + len(self.complexity_findings)
        )

    @computed_field
    @property
    def critical_count(self) -> int:
        all_findings: list[_BaseFinding] = (
            self.bug_findings  # type: ignore[assignment]
            + self.solid_findings
            + self.architecture_findings
            + self.security_findings
            + self.complexity_findings
        )
        return sum(1 for f in all_findings if f.severity == Severity.CRITICAL)


# ---------------------------------------------------------------------------
# Job Status
# ---------------------------------------------------------------------------


class JobStatus(_BaseSchema):
    """
    Tracks the lifecycle of a single review job.
    Stored in the job store (memory / Redis). NOT the HTTP response model.
    """
    job_id: str = Field(default_factory=_new_uuid)
    status: JobStatusEnum = Field(default=JobStatusEnum.PENDING)
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Completion percentage [0–100].",
    )
    result: Optional[ReviewResult] = Field(
        None,
        description="Populated only when status=COMPLETED.",
    )
    error: Optional[str] = Field(
        None,
        description="Error message populated when status=FAILED.",
    )
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "JobStatus":
        if self.status == JobStatusEnum.COMPLETED and self.result is None:
            raise ValueError("result must be set when status=COMPLETED.")
        if self.status == JobStatusEnum.FAILED and self.error is None:
            raise ValueError("error must be set when status=FAILED.")
        return self


# ---------------------------------------------------------------------------
# LangGraph Review State
# ---------------------------------------------------------------------------


class ReviewConfig(_BaseSchema):
    """
    Per-request configuration options that tune the review pipeline.
    Passed in the POST /review request body and threaded through ReviewState.
    """
    max_chunk_lines: int = Field(
        default=80,
        ge=20,
        le=500,
        description="Maximum lines per CodeChunk. Smaller = finer-grained analysis.",
    )
    enable_bug_analysis: bool = True
    enable_solid_analysis: bool = True
    enable_architecture_analysis: bool = True
    enable_security_analysis: bool = True
    enable_complexity_analysis: bool = True
    similarity_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of FAISS nearest-neighbors to attach per finding.",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM sampling temperature. Keep low for deterministic findings.",
    )


class ReviewState(_BaseSchema):
    """
    LangGraph graph state — the single mutable object threaded through all nodes.

    IMPORTANT:
      - This model uses `extra='allow'` so LangGraph can attach transient fields.
      - The `faiss_index` field is excluded from serialization (in-process object only).
      - Nodes must NOT read fields they do not own; see data_flow.md for node I/O map.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="allow",              # LangGraph may inject additional keys
        arbitrary_types_allowed=True,
    )

    # ── Identity ──────────────────────────────────────────────────────────
    job_id: str = Field(default_factory=_new_uuid)
    config: ReviewConfig = Field(default_factory=ReviewConfig)

    # ── Input (set by caller before graph.invoke) ─────────────────────────
    source_url: Optional[str] = None
    source_zip_b64: Optional[str] = Field(
        None,
        description="Base64-encoded ZIP archive of the repository.",
        exclude=True,   # not stored after ingest
    )

    # ── ingest_node outputs ───────────────────────────────────────────────
    metadata: Optional[RepositoryMetadata] = None
    chunks: list[CodeChunk] = Field(default_factory=list)
    faiss_index: Optional[Any] = Field(
        None,
        exclude=True,
        description="FAISS IndexFlatL2 instance — in-process only, never serialized.",
    )

    # ── analyze_node outputs ──────────────────────────────────────────────
    bug_findings: list[BugFinding] = Field(default_factory=list)
    solid_findings: list[SolidFinding] = Field(default_factory=list)
    architecture_findings: list[ArchitectureFinding] = Field(default_factory=list)
    security_findings: list[SecurityFinding] = Field(default_factory=list)
    complexity_findings: list[ComplexityFinding] = Field(default_factory=list)

    # ── compile_node output ───────────────────────────────────────────────
    result: Optional[ReviewResult] = None

    # ── Control flow ──────────────────────────────────────────────────────
    progress: int = Field(default=0, ge=0, le=100)
    error: Optional[str] = None

    @model_validator(mode="after")
    def validate_source_provided(self) -> "ReviewState":
        if self.source_url is None and self.source_zip_b64 is None:
            # Allow partial states (nodes may clear source after ingest)
            pass
        return self


# ---------------------------------------------------------------------------
# HTTP Request / Response Models
# ---------------------------------------------------------------------------


class ReviewRequest(_BaseSchema):
    """
    Body for POST /review.
    Exactly one of source_url or source_zip_b64 must be provided.
    """
    source_url: Optional[str] = Field(
        None,
        description="Public GitHub or GitLab repository URL.",
        examples=["https://github.com/user/my-repo"],
    )
    source_zip_b64: Optional[str] = Field(
        None,
        description="Base64-encoded ZIP archive (max 50 MB decoded).",
    )
    repository_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable project name for display in the report.",
    )
    config: ReviewConfig = Field(
        default_factory=ReviewConfig,
        description="Optional pipeline configuration overrides.",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "ReviewRequest":
        if self.source_url is None and self.source_zip_b64 is None:
            raise ValueError("Exactly one of source_url or source_zip_b64 must be provided.")
        if self.source_url is not None and self.source_zip_b64 is not None:
            raise ValueError("Provide either source_url or source_zip_b64, not both.")
        return self


class ReviewResponse(_BaseSchema):
    """
    Response body for POST /review (202 Accepted).
    """
    job_id: str = Field(..., description="UUID to use for status/report polling.")
    status: JobStatusEnum = Field(default=JobStatusEnum.PENDING)
    poll_url: str = Field(..., description="Absolute URL for GET /status/{job_id}.")
    estimated_duration_seconds: Optional[int] = Field(
        None,
        description="Best-effort estimate of analysis completion time.",
    )


class JobStatusResponse(_BaseSchema):
    """
    Response body for GET /status/{job_id}.
    When status=COMPLETED, result is populated.
    When status=FAILED, error is populated.
    """
    job_id: str
    status: JobStatusEnum
    progress: int = Field(..., ge=0, le=100)
    result: Optional[ReviewResult] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class HealthResponse(_BaseSchema):
    """
    Response body for GET /health.
    """
    status: str = Field(default="ok", description="'ok' or 'degraded'.")
    version: str = Field(..., description="API semantic version string.")
    uptime_seconds: float = Field(..., ge=0)
    active_jobs: int = Field(..., ge=0, description="Jobs currently in RUNNING state.")
    vector_store_ready: bool = Field(
        ...,
        description="Whether the SentenceTransformer model is loaded and ready.",
    )


# ---------------------------------------------------------------------------
# Error Response
# ---------------------------------------------------------------------------


class ErrorDetail(_BaseSchema):
    """Structured error payload returned on 4xx / 5xx responses."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    code: str = Field(..., description="Machine-readable error code.", examples=["JOB_NOT_FOUND"])
    message: str = Field(..., description="Human-readable error message.")
    details: Optional[dict[str, Any]] = Field(
        None,
        description="Additional structured context (e.g., validation field errors).",
    )
