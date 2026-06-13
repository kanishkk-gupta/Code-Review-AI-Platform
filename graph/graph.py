"""
graph/graph.py
==============
LangGraph review pipeline definition.

This module is now a thin shim that delegates to graph/workflow.py,
which contains the full node and edge implementation.

Topology (defined in workflow.py):
    START → repository_processor → chunk_generator
         → parallel_analysis (all 5 agents via asyncio.gather)
         → report_agent_node → END

    Any node may short-circuit to error_node → END if state["error"] is set.

Usage:
    from graph.graph import get_review_graph
    graph  = get_review_graph()
    output = await graph.ainvoke(initial_state)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


def _route_after_node(state: dict) -> Literal["next", "error"]:
    """
    Conditional edge function.
    Routes to 'error' terminal if ReviewState.error is set, else continues.
    """
    if state.get("error"):
        logger.warning("graph_routing_to_error", error=state["error"])
        return "error"
    return "next"


@lru_cache(maxsize=1)
def get_review_graph():
    """
    Build and compile the LangGraph review graph (singleton).
    Delegates to graph.workflow.get_workflow().

    Returns:
        A compiled LangGraph CompiledStateGraph ready for .ainvoke()

    Raises:
        ImportError: langgraph is not installed.
    """
    from graph.workflow import get_workflow
    return get_workflow()


async def _error_terminal_node(state: dict) -> dict:
    """
    Terminal node reached when any preceding node sets state.error.
    Ensures the job store is updated to FAILED status.
    """
    logger.error("graph_error_terminal", error=state.get("error"))
    return state
