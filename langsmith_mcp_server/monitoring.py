"""
Optional instrumentation: log MCP tool calls to a separate LangSmith instance for monitoring.

Loads configuration from environment (e.g. from .env). When enabled, each tool call is
traced with run_type="tool" to the monitoring instance, with session_id (or thread_id)
in metadata so you can filter and analyze by session.

Environment variables (all optional; if LANGSMITH_MONITORING_API_KEY is unset, monitoring is disabled):
- LANGSMITH_MONITORING_API_KEY: API key for the LangSmith instance used for monitoring
- LANGSMITH_MONITORING_ENDPOINT: Endpoint URL (default: https://api.smith.langchain.com)
- LANGSMITH_MONITORING_WORKSPACE_ID: Workspace ID for the monitoring instance
- LANGSMITH_MONITORING_PROJECT: Project name for monitoring traces (default: mcp-server-monitoring)

For tracing to be sent, LANGSMITH_TRACING must be set to "true" in the environment
(see LangSmith custom instrumentation docs).
"""

import logging
import os
from contextvars import ContextVar
from typing import Any, Callable, Coroutine, Optional, TypeVar
from uuid import uuid4

from langsmith import Client
from langsmith.run_helpers import traceable

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Load .env from current working directory or parents (e.g. project root)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Session/thread id for the current request (set by middleware or defaulted per request)
monitoring_session_id_context: ContextVar[str] = ContextVar("monitoring_session_id", default="")

# Cached monitoring client and project (lazy init)
_monitoring_client: Optional[Client] = None
_monitoring_project: Optional[str] = None
_monitoring_initialized: bool = False


def _init_monitoring() -> None:
    global _monitoring_client, _monitoring_project, _monitoring_initialized
    if _monitoring_initialized:
        return
    _monitoring_initialized = True
    api_key = os.environ.get("LANGSMITH_MONITORING_API_KEY", "").strip()
    if not api_key:
        return
    endpoint = os.environ.get("LANGSMITH_MONITORING_ENDPOINT", "").strip() or None
    workspace_id = os.environ.get("LANGSMITH_MONITORING_WORKSPACE_ID", "").strip() or None
    project = os.environ.get("LANGSMITH_MONITORING_PROJECT", "").strip() or "mcp-server-monitoring"
    try:
        _monitoring_client = Client(
            api_key=api_key,
            api_url=endpoint,
            workspace_id=workspace_id,
        )
        _monitoring_project = project
        logger.info(
            "Monitoring enabled: tool calls will be traced to project %s",
            _monitoring_project,
        )
    except Exception as e:
        logger.warning("Monitoring client init failed: %s", e)
        _monitoring_client = None
        _monitoring_project = None


def is_monitoring_enabled() -> bool:
    """Return True if a monitoring LangSmith client is configured."""
    _init_monitoring()
    return _monitoring_client is not None


def get_monitoring_client() -> Optional[Client]:
    """Return the LangSmith client for monitoring, or None if disabled."""
    _init_monitoring()
    return _monitoring_client


def get_monitoring_project() -> str:
    """Return the project name for monitoring traces (default if disabled)."""
    _init_monitoring()
    return _monitoring_project or "mcp-server-monitoring"


def get_monitoring_session_id() -> str:
    """
    Return the current request/session id for monitoring (set by middleware or generated).
    Used to group tool calls by session in the monitoring LangSmith project.
    """
    sid = monitoring_session_id_context.get("")
    if sid:
        return sid
    return str(uuid4())


def set_monitoring_session_id(session_id: str) -> None:
    """Set the session id for the current context (e.g. from middleware)."""
    monitoring_session_id_context.set(session_id)


async def run_tool_traced_async(
    tool_name: str,
    session_id: str,
    inputs: dict,
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
) -> T:
    """
    Run an async tool and trace it to the monitoring LangSmith with run_type="tool".

    If monitoring is disabled, runs the coroutine without tracing.
    """
    if not is_monitoring_enabled():
        return await coro_fn()
    client = get_monitoring_client()
    project = get_monitoring_project()
    if not client:
        return await coro_fn()

    @traceable(
        run_type="tool",
        name=tool_name,
        client=client,
        project_name=project,
    )
    async def _traced(inputs: dict) -> T:
        return await coro_fn()

    return await _traced(
        inputs,
        langsmith_extra={"metadata": {"session_id": session_id}},
    )


def run_tool_traced_sync(
    tool_name: str,
    session_id: str,
    inputs: dict,
    fn: Callable[[], T],
) -> T:
    """
    Run a sync tool and trace it to the monitoring LangSmith with run_type="tool".

    If monitoring is disabled, runs the function without tracing.
    """
    if not is_monitoring_enabled():
        return fn()
    client = get_monitoring_client()
    project = get_monitoring_project()
    if not client:
        return fn()

    @traceable(
        run_type="tool",
        name=tool_name,
        client=client,
        project_name=project,
    )
    def _traced(inputs: dict) -> T:
        return fn()

    return _traced(inputs, langsmith_extra={"metadata": {"session_id": session_id}})
