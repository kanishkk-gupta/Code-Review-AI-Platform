"""
services/job_store.py
=====================
Async job lifecycle management service.

Interface contract (all implementations must satisfy):

    async def create(job: JobStatus) -> None
    async def get(job_id: str) -> JobStatus | None
    async def update(job_id: str, **fields) -> None
    async def delete(job_id: str) -> None
    async def count_by_status(status: JobStatusEnum) -> int

Dev implementation: asyncio.Lock-protected in-memory dict.
Prod implementation: Redis (swap InMemoryJobStore for RedisJobStore).

Usage:
    from services.job_store import get_job_store
    store = get_job_store()
    await store.create(job)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

import structlog

from schemas import JobStatus, JobStatusEnum

logger = structlog.get_logger(__name__)


# ── Abstract Interface ────────────────────────────────────────────────────────


class JobStore(ABC):
    """Abstract base class defining the job store interface contract."""

    @abstractmethod
    async def create(self, job: JobStatus) -> None:
        """Persist a new job. Raises ValueError if job_id already exists."""

    @abstractmethod
    async def get(self, job_id: str) -> Optional[JobStatus]:
        """Return the JobStatus for job_id, or None if not found / expired."""

    @abstractmethod
    async def update(self, job_id: str, **fields: Any) -> None:
        """
        Partially update a job's fields.
        Raises KeyError if job_id does not exist.
        Automatically sets started_at on first RUNNING transition.
        Automatically sets completed_at on COMPLETED/FAILED transitions.
        """

    @abstractmethod
    async def delete(self, job_id: str) -> None:
        """Remove a job from the store. No-op if not found."""

    @abstractmethod
    async def count_by_status(self, status: JobStatusEnum) -> int:
        """Return count of jobs in the given status."""


# ── In-Memory Implementation ──────────────────────────────────────────────────


class InMemoryJobStore(JobStore):
    """
    Thread-safe in-memory job store for development and testing.

    All operations are protected by asyncio.Lock.
    Jobs are stored as plain dicts for O(1) field updates.

    TTL eviction is NOT implemented here (use Redis TTL in production).
    For dev, jobs live for the lifetime of the process.
    """

    def __init__(self) -> None:
        self._store: dict[str, JobStatus] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: JobStatus) -> None:
        async with self._lock:
            if job.job_id in self._store:
                raise ValueError(f"Job '{job.job_id}' already exists.")
            self._store[job.job_id] = job
            logger.debug("job_created", job_id=job.job_id)

    async def get(self, job_id: str) -> Optional[JobStatus]:
        async with self._lock:
            return self._store.get(job_id)

    async def update(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            if job_id not in self._store:
                raise KeyError(f"Job '{job_id}' not found in store.")

            job = self._store[job_id]

            # Lifecycle timestamps
            new_status = fields.get("status")
            now = datetime.now(timezone.utc)

            if new_status == JobStatusEnum.RUNNING and job.started_at is None:
                fields.setdefault("started_at", now)

            if new_status in (JobStatusEnum.COMPLETED, JobStatusEnum.FAILED):
                fields.setdefault("completed_at", now)

            # Apply updates using model_copy for Pydantic V2 immutability
            updated = job.model_copy(update=fields)
            self._store[job_id] = updated

            logger.debug("job_updated", job_id=job_id, fields=list(fields.keys()))

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            self._store.pop(job_id, None)
            logger.debug("job_deleted", job_id=job_id)

    async def count_by_status(self, status: JobStatusEnum) -> int:
        async with self._lock:
            return sum(1 for j in self._store.values() if j.status == status)


# ── Singleton Provider ────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_in_memory_store() -> InMemoryJobStore:
    """Return the singleton in-memory store instance."""
    return InMemoryJobStore()


def get_job_store() -> JobStore:
    """
    FastAPI Depends() compatible provider.
    Returns the appropriate JobStore implementation based on settings.

    Dev:  InMemoryJobStore
    Prod: RedisJobStore (TODO: implement in Phase 3)
    """
    from config.settings import get_settings
    settings = get_settings()

    if settings.use_redis:
        # TODO: return RedisJobStore(settings.redis_url)
        raise NotImplementedError("Redis job store not yet implemented. Set REDIS_URL= in .env to use in-memory.")

    return _get_in_memory_store()
