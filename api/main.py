"""
api/main.py
===========
FastAPI application factory.

Responsibilities:
  - Create and configure the FastAPI app instance
  - Register all routers with proper prefixes and tags
  - Attach middleware (CORS, request logging, timing)
  - Define lifespan context (startup / shutdown hooks)
  - Warm up the embedding model and pre-compile the LangGraph workflow
  - Register global exception handlers

DO NOT place business logic here. This file is purely configuration.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from api.routes import health, report, review, status as status_route
from api.middleware.logging import LoggingMiddleware

logger   = structlog.get_logger(__name__)
settings = get_settings()

# Module-level start time for uptime calculation (shared with health.py)
_START_TIME: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Startup sequence:
      1. Record process start time.
      2. Pre-compile the LangGraph workflow (singleton — cached for the
         process lifetime; avoids cold-start latency on first request).
      3. Warm up the SentenceTransformer embedding model in a thread-pool
         executor so it is ready before the first review job arrives.
      4. Log readiness.

    Shutdown:
      - Flush structlog.
      - No active connections to drain (in-memory store, stateless workers).
    """
    global _START_TIME
    _START_TIME = time.monotonic()

    # Share start time with health.py for uptime reporting
    import api.routes.health as health_module
    health_module._PROCESS_START = _START_TIME

    logger.info(
        "codeguardian_startup",
        version=settings.app_version,
        env=settings.app_env,
        llm_model=settings.llm_model_name,
    )

    # ── 1. Pre-compile LangGraph workflow ─────────────────────────────────
    try:
        from graph.workflow import get_workflow
        _ = get_workflow()          # builds the StateGraph, caches via @lru_cache
        logger.info("workflow_precompiled")
    except ImportError:
        logger.warning("workflow_precompile_skipped", reason="langgraph not installed")
    except Exception as exc:        # noqa: BLE001
        logger.warning("workflow_precompile_failed", error=str(exc))

    # ── 2. Warm up embedding model ────────────────────────────────────────
    try:
        loop = asyncio.get_event_loop()
        warmed_up: bool = await loop.run_in_executor(None, _warmup_embedding_model)
        if warmed_up:
            logger.info("embedding_model_warmed_up")
        else:
            logger.warning("embedding_model_warmup_skipped",
                           reason="see embedding_warmup_skipped log for details")
    except Exception as exc:        # noqa: BLE001
        logger.warning("embedding_model_warmup_failed", error=str(exc))

    logger.info("codeguardian_ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    elapsed = time.monotonic() - _START_TIME
    logger.info("codeguardian_shutdown", uptime_seconds=round(elapsed, 1))


def _warmup_embedding_model() -> bool:
    """
    Load the SentenceTransformer model into the module-level registry.
    Runs in a thread-pool executor to avoid blocking the event loop.

    Returns:
        True  — model loaded and ready.
        False — model unavailable (dependency missing or import error).
                Caller logs a warning; application continues in degraded mode.
    """
    try:
        from rag.embeddings import load_embedding_model
        model = load_embedding_model()
        model.embed_query("warmup")   # JIT-compile the model graph
        logger.info("embedding_model_ready", model=getattr(model, "model_name", "unknown"))
        return True
    except ImportError as exc:
        # Two known causes:
        #   1. sentence-transformers not installed
        #   2. rag package __init__ has a broken import (stale function name etc.)
        logger.warning("embedding_warmup_skipped", reason=str(exc))
        return False
    except Exception as exc:        # noqa: BLE001
        logger.warning("embedding_warmup_error", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Application factory. Returns a fully configured FastAPI instance.
    Import this function in tests or alternate entry points.
    """
    app = FastAPI(
        title="CodeGuardian AI",
        description=(
            "**AI-powered code review platform.**\n\n"
            "Analyzes repositories for:\n"
            "- 🐛 **Bugs** — Null dereferences, division by zero, resource leaks, dead code\n"
            "- 🏗️ **SOLID violations** — SRP, OCP, LSP, ISP, DIP\n"
            "- 🏛️ **Architecture smells** — God classes, cyclic deps, layer violations\n"
            "- 🔒 **Security vulnerabilities** — OWASP Top 10, CWE-aligned\n"
            "- 📊 **Complexity hotspots** — Cyclomatic / cognitive complexity, deep nesting\n\n"
            "Submit a review via `POST /review`, poll via `GET /status/{job_id}`, "
            "and retrieve the full report via `GET /report/{job_id}`."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # ── Request ID + Timing middleware ────────────────────────────────────
    @app.middleware("http")
    async def request_id_and_timing(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        response.headers["X-Request-ID"]   = request_id
        response.headers["X-Process-Time"] = f"{elapsed_ms}ms"
        return response

    # ── Structured logging middleware ─────────────────────────────────────
    app.add_middleware(LoggingMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router,         tags=["Health"])
    app.include_router(review.router,         tags=["Review"])
    app.include_router(status_route.router,   tags=["Status"])
    app.include_router(report.router,         tags=["Report"])

    # ── Global Exception Handlers ─────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "unhandled_exception",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code":    "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again or contact support.",
                "details": {"request_id": request_id},
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning(
            "validation_error",
            path=request.url.path,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code":    "VALIDATION_ERROR",
                "message": str(exc),
                "details": None,
            },
        )

    # ── Root redirect ──────────────────────────────────────────────────────
    @app.get(
        "/",
        include_in_schema=False,
        summary="Root — redirects to API docs",
    )
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "service": "CodeGuardian AI",
                "version": settings.app_version,
                "docs":    "/docs",
                "health":  "/health",
                "openapi": "/openapi.json",
            }
        )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (Uvicorn entry point)
# ---------------------------------------------------------------------------

app = create_app()
