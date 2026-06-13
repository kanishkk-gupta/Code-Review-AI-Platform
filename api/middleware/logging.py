"""
api/middleware/logging.py
=========================
Structured request/response logging middleware using structlog.

Logs:
  - HTTP method, path, status code, duration (ms)
  - X-Request-ID header (if provided)
  - Client IP
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with structured fields."""

    async def dispatch(self, request: Request, call_next: any) -> Response:  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.monotonic()

        # Bind request context to all log calls within this request
        with structlog.contextvars.bound_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        ):
            logger.info("request_received")
            response = await call_next(request)
            duration_ms = round((time.monotonic() - start) * 1000, 2)

            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

        response.headers["X-Request-ID"] = request_id
        return response
