"""
tests/unit/test_schemas.py
===========================
Unit tests for all Pydantic V2 schema models.
Coverage target: 100%
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import (
    BugFinding,
    CodeChunk,
    ComplexityFinding,
    JobStatus,
    JobStatusEnum,
    RepositoryMetadata,
    ReviewConfig,
    ReviewRequest,
    ReviewResult,
    ReviewState,
    SecurityFinding,
    Severity,
    SolidFinding,
    SupportedLanguage,
    ArchitectureFinding,
    SolidPrinciple,
    ArchitectureSmell,
    SecurityCategory,
)


class TestReviewRequest:
    def test_valid_url_source(self):
        req = ReviewRequest(
            repository_name="my-repo",
            source_url="https://github.com/user/repo",
        )
        assert req.repository_name == "my-repo"
        assert req.source_url == "https://github.com/user/repo"
        assert req.source_zip_b64 is None

    def test_valid_zip_source(self):
        req = ReviewRequest(
            repository_name="my-repo",
            source_zip_b64="base64data==",
        )
        assert req.source_zip_b64 == "base64data=="

    def test_both_sources_raises(self):
        with pytest.raises(ValidationError, match="not both"):
            ReviewRequest(
                repository_name="my-repo",
                source_url="https://github.com/user/repo",
                source_zip_b64="base64data==",
            )

    def test_no_source_raises(self):
        with pytest.raises(ValidationError, match="Exactly one"):
            ReviewRequest(repository_name="my-repo")

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            ReviewRequest(repository_name="", source_url="https://github.com/u/r")


class TestCodeChunk:
    def test_valid_chunk(self, sample_code_chunk):
        assert sample_code_chunk.line_count == 2
        assert sample_code_chunk.chunk_id is not None

    def test_invalid_line_range_raises(self):
        with pytest.raises(ValidationError, match="end_line"):
            CodeChunk(
                file_path="main.py",
                language=SupportedLanguage.PYTHON,
                content="x = 1",
                start_line=10,
                end_line=5,
            )

    def test_embedding_excluded_from_json(self, sample_code_chunk):
        import numpy as np
        sample_code_chunk.embedding = np.zeros(384, dtype=np.float32)
        data = sample_code_chunk.model_dump()
        assert "embedding" not in data


class TestRepositoryMetadata:
    def test_valid_metadata(self, sample_repository_metadata):
        assert sample_repository_metadata.total_files == 5

    def test_invalid_language_breakdown_raises(self):
        with pytest.raises(ValidationError, match="sum to"):
            RepositoryMetadata(
                repository_name="test",
                primary_language=SupportedLanguage.PYTHON,
                language_breakdown={"python": 50.0, "js": 10.0},  # sums to 60, not 100
                total_files=1,
                total_lines=100,
            )


class TestJobStatus:
    def test_pending_is_valid(self):
        job = JobStatus()
        assert job.status == JobStatusEnum.PENDING

    def test_completed_without_result_raises(self):
        with pytest.raises(ValidationError, match="result must be set"):
            JobStatus(status=JobStatusEnum.COMPLETED, result=None)

    def test_failed_without_error_raises(self):
        with pytest.raises(ValidationError, match="error must be set"):
            JobStatus(status=JobStatusEnum.FAILED, error=None)


class TestReviewConfig:
    def test_defaults(self):
        config = ReviewConfig()
        assert config.max_chunk_lines == 80
        assert config.llm_temperature == 0.1
        assert config.similarity_top_k == 3

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValidationError):
            ReviewConfig(llm_temperature=1.5)
