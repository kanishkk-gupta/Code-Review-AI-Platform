"""
services/llm_client.py
======================
LangChain ChatModel factory.

Responsibilities:
  - Read LLM configuration from Settings
  - Return a configured BaseChatModel instance
  - Apply retry logic via tenacity

All analyzers obtain their model through get_llm_client().
No module may instantiate ChatOpenAI or ChatOllama directly.

Usage:
    from services.llm_client import get_llm_client
    llm = get_llm_client()
    response = await llm.ainvoke([HumanMessage(content="Hello")])
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm_client() -> "BaseChatModel":
    """
    Return the singleton LangChain ChatModel.

    Priority:
      1. OpenAI (if OPENAI_API_KEY is set)
      2. Ollama (fallback for local dev)

    The model is wrapped with:
      - Temperature from settings
      - Retry logic (tenacity) for transient errors
      - Request timeout

    Returns:
        A configured BaseChatModel instance.

    Raises:
        RuntimeError: If no LLM provider is configured.
    """
    from config.settings import get_settings
    settings = get_settings()

    logger.info("llm_client_init", model=settings.llm_model_name, env=settings.app_env)

    if settings.openai_api_key:
        # TODO: Implement OpenAI client with retry wrapper
        # from langchain_openai import ChatOpenAI
        # return ChatOpenAI(
        #     model=settings.llm_model_name,
        #     temperature=settings.llm_temperature,
        #     max_retries=settings.llm_max_retries,
        #     timeout=settings.llm_request_timeout,
        #     api_key=settings.openai_api_key,
        # )
        raise NotImplementedError("OpenAI client — Phase 2 implementation.")

    # Fallback: Ollama
    # TODO: Implement Ollama client
    # from langchain_community.chat_models import ChatOllama
    # return ChatOllama(
    #     model=settings.llm_model_name,
    #     temperature=settings.llm_temperature,
    #     base_url=settings.ollama_base_url,
    # )
    raise NotImplementedError(
        "LLM client not yet implemented. Configure OPENAI_API_KEY in .env — Phase 2."
    )
