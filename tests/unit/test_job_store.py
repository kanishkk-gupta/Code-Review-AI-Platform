"""
tests/unit/test_job_store.py
=============================
Unit tests for services/job_store.py (InMemoryJobStore)
"""

from __future__ import annotations

import asyncio
import pytest

from schemas import JobStatus, JobStatusEnum
from services.job_store import InMemoryJobStore


@pytest.fixture
def store():
    return InMemoryJobStore()


@pytest.fixture
def sample_job():
    return JobStatus(status=JobStatusEnum.PENDING)


class TestInMemoryJobStore:
    async def test_create_and_get(self, store, sample_job):
        await store.create(sample_job)
        retrieved = await store.get(sample_job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == sample_job.job_id

    async def test_get_missing_returns_none(self, store):
        result = await store.get("nonexistent-id")
        assert result is None

    async def test_create_duplicate_raises(self, store, sample_job):
        await store.create(sample_job)
        with pytest.raises(ValueError, match="already exists"):
            await store.create(sample_job)

    async def test_update_status(self, store, sample_job):
        await store.create(sample_job)
        await store.update(sample_job.job_id, status=JobStatusEnum.RUNNING, progress=20)
        updated = await store.get(sample_job.job_id)
        assert updated.status == JobStatusEnum.RUNNING
        assert updated.progress == 20

    async def test_update_sets_started_at_on_running(self, store, sample_job):
        await store.create(sample_job)
        await store.update(sample_job.job_id, status=JobStatusEnum.RUNNING)
        updated = await store.get(sample_job.job_id)
        assert updated.started_at is not None

    async def test_update_missing_raises(self, store):
        with pytest.raises(KeyError):
            await store.update("nonexistent", status=JobStatusEnum.RUNNING)

    async def test_delete(self, store, sample_job):
        await store.create(sample_job)
        await store.delete(sample_job.job_id)
        assert await store.get(sample_job.job_id) is None

    async def test_delete_noop_for_missing(self, store):
        """delete() must be a no-op for nonexistent IDs."""
        await store.delete("nonexistent")  # should not raise

    async def test_count_by_status(self, store):
        j1 = JobStatus(status=JobStatusEnum.PENDING)
        j2 = JobStatus(status=JobStatusEnum.PENDING)
        j3 = JobStatus(status=JobStatusEnum.RUNNING)
        for j in [j1, j2, j3]:
            await store.create(j)
        assert await store.count_by_status(JobStatusEnum.PENDING) == 2
        assert await store.count_by_status(JobStatusEnum.RUNNING) == 1
        assert await store.count_by_status(JobStatusEnum.COMPLETED) == 0
