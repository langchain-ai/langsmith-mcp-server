"""Helper functions for the LangSmith MCP server."""

import os
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, List, Optional, Union
from uuid import UUID

from fastmcp.server import Context
from langsmith import Client

# Import context variables from middleware
try:
    from langsmith_mcp_server.middleware import (
        api_key_context,
        endpoint_context,
        workspace_id_context,
    )
except ImportError:
    # Fallback if middleware not available
    from contextvars import ContextVar

    api_key_context: ContextVar[str] = ContextVar("api_key", default="")
    workspace_id_context: ContextVar[str] = ContextVar("workspace_id", default="")
    endpoint_context: ContextVar[str] = ContextVar("endpoint", default="")


def get_langsmith_client_from_api_key(
    api_key: str, workspace_id: Optional[str] = None, endpoint: Optional[str] = None
) -> Client:
    """
    Create a LangSmith client from an API key and optional configuration.

    We do not set os.environ (LANGSMITH_API_KEY, etc.) so that concurrent
    HTTP requests do not overwrite each other's credentials, which can cause
    403 Forbidden when the wrong key is used.

    Args:
        api_key: The LangSmith API key (required)
        workspace_id: Optional workspace ID for API keys scoped to multiple workspaces
        endpoint: Optional custom endpoint URL (e.g., for self-hosted installations or EU region)

    Returns:
        LangSmith Client instance
    """  # noqa: W293
    # Initialize the LangSmith client with parameters (no os.environ mutation)
    client_kwargs = {"api_key": api_key}
    if workspace_id:
        client_kwargs["workspace_id"] = workspace_id
    if endpoint:
        client_kwargs["api_url"] = endpoint

    return Client(**client_kwargs)


async def get_client_from_context(ctx: Context) -> Client:
    """
    Get LangSmith client from API key and optional config using FastMCP context.

    Supports both HTTP and STDIO transports:
    - HTTP: Config extracted from headers (LANGSMITH-API-KEY, etc.) and stored in session
    - STDIO: Config read from environment variables (LANGSMITH_API_KEY, etc.)

    On first HTTP request, config is extracted from headers and stored in session.
    On subsequent HTTP requests, config is retrieved from session state.
    For STDIO, config is always read from environment variables.

    Args:
        ctx: FastMCP context (automatically provided to tools)

    Returns:
        LangSmith Client instance

    Raises:
        ValueError: If API key is not found in headers (HTTP) or environment (STDIO)
    """  # noqa: W293
    # Try to get config from session state (set on first HTTP request)
    api_key = await ctx.get_state("api_key")
    workspace_id = (await ctx.get_state("workspace_id")) or None
    endpoint = (await ctx.get_state("endpoint")) or None

    # If not in session, try to get from context variables (set by middleware for HTTP transport)
    if not api_key:
        try:
            api_key = api_key_context.get("")
            if api_key:
                workspace_id = workspace_id_context.get("") or None
                endpoint = endpoint_context.get("") or None
                # Store in session for future requests
                await ctx.set_state("api_key", api_key)
                if workspace_id:
                    await ctx.set_state("workspace_id", workspace_id)
                if endpoint:
                    await ctx.set_state("endpoint", endpoint)
        except (RuntimeError, Exception):
            pass

    # If still not found, try to get from request headers (HTTP transport)
    if not api_key:
        try:
            request = ctx.get_http_request()
            if request:
                # HTTP transport: try request.state first (set by middleware), then headers
                if hasattr(request.state, "api_key") and request.state.api_key:
                    api_key = request.state.api_key
                    workspace_id = getattr(request.state, "workspace_id", None) or None
                    endpoint = getattr(request.state, "endpoint", None) or None
                else:
                    # Fall back to headers directly
                    api_key = request.headers.get("LANGSMITH-API-KEY")
                    workspace_id = request.headers.get("LANGSMITH-WORKSPACE-ID") or None
                    endpoint = request.headers.get("LANGSMITH-ENDPOINT") or None

                # Store in session for future requests
                if api_key:
                    await ctx.set_state("api_key", api_key)
                    if workspace_id:
                        await ctx.set_state("workspace_id", workspace_id)
                    if endpoint:
                        await ctx.set_state("endpoint", endpoint)
        except (RuntimeError, Exception):
            # STDIO transport: get_http_request() raises exception when no active HTTP request
            # Fall through to get from environment variables
            pass

        # If still no api_key (either request was None or we caught exception), try environment
        if not api_key:
            # STDIO transport: get from environment variables
            api_key = os.environ.get("LANGSMITH_API_KEY")
            workspace_id = os.environ.get("LANGSMITH_WORKSPACE_ID") or None
            endpoint = os.environ.get("LANGSMITH_ENDPOINT") or None

    if not api_key:
        raise ValueError(
            "API key not found. For HTTP transport, provide LANGSMITH-API-KEY header. "
            "For STDIO transport, set LANGSMITH_API_KEY environment variable."
        )

    # Reuse the same Client within this request to avoid creating a new one per tool
    # call (reduces memory churn and connection buildup under load).
    _cache_key = "__langsmith_client"
    cached = await ctx.get_state(_cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    client = get_langsmith_client_from_api_key(
        api_key, workspace_id=workspace_id, endpoint=endpoint
    )
    await ctx.set_state(_cache_key, client, serializable=False)
    return client


async def get_api_key_and_endpoint_from_context(ctx: Context) -> tuple[str, str]:
    """
    Get API key and endpoint from FastMCP context (same sources as get_client_from_context).
    Used by tools that call LangSmith REST APIs directly (e.g. billing/usage).

    Args:
        ctx: FastMCP context (automatically provided to tools).

    Returns:
        Tuple of (api_key, endpoint). endpoint is normalized (no trailing slash).

    Raises:
        ValueError: If API key is not found.
    """
    await get_client_from_context(ctx)  # populates ctx state (api_key, endpoint)
    api_key = await ctx.get_state("api_key")
    endpoint = (await ctx.get_state("endpoint")) or "https://api.smith.langchain.com"
    return (str(api_key), str(endpoint).rstrip("/"))


def get_langgraph_app_host_name(run_stats: dict) -> Optional[str]:
    """
    Get the langgraph app host name from the run stats

    Args:
        run_stats (dict): The run stats

    Returns:
        str | None: The langgraph app host name
    """  # noqa: W293
    if run_stats and run_stats.get("run_facets"):
        for run_facet in run_stats["run_facets"]:
            try:
                for rfk in run_facet.keys():
                    langgraph_host = re.search(r"http[s]?:\/\/(?P<langgraph_host>[^\/]+)", rfk)
                    if langgraph_host:
                        return langgraph_host.group("langgraph_host")
            except re.error:
                continue
    return None


def _parse_as_of_parameter(as_of: str) -> Union[datetime, str]:
    """
    Parse the as_of parameter, converting ISO timestamps to datetime objects
    while leaving version tags as strings.

    Args:
        as_of: Dataset version tag OR ISO timestamp string

    Returns:
        datetime object if as_of is a valid ISO timestamp, otherwise the original string
    """  # noqa: W293
    try:
        # Try to parse as ISO format datetime
        return datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        # If parsing fails, assume it's a version tag and return as string
        return as_of


def find_in_dict(data, key):
    """
    Recursively search for a key in a nested dictionary or list.

    This helper function traverses nested data structures to find a specific key,
    searching through dictionaries and lists at any depth level.

    ---
    ⚙️ PARAMETERS
    -------------
    data : dict | list | Any
        The data structure to search in. Can be a dictionary, list, or any nested
        combination of these types.

    key : str
        The key to search for in the data structure.

    ---
    📤 RETURNS
    ----------
    Any | None
        The value associated with the key if found, otherwise None.
        Returns the first occurrence found during depth-first traversal.

    ---
    🧪 EXAMPLES
    ------------
    ```python
    data = {
        "a": 1,
        "b": {
            "c": {"deployment_id": "123-456"}
        }
    }
    result = find_in_dict(data, "deployment_id")  # Returns "123-456"
    ```
    """  # noqa: W293
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            result = find_in_dict(value, key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_in_dict(item, key)
            if result is not None:
                return result
    return None


def convert_uuids_to_strings(obj: Any) -> Any:
    """
    Recursively convert UUID, datetime, and Decimal objects to strings in dictionaries, lists, and other structures.
    """  # noqa: W293
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_uuids_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_uuids_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_uuids_to_strings(item) for item in obj)
    else:
        return obj


def count_characters(obj: Any) -> int:
    """
    Recursively count the total number of characters in a data structure.
    """  # noqa: W293
    if isinstance(obj, str):
        return len(obj)
    elif isinstance(obj, dict):
        return sum(count_characters(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(count_characters(item) for item in obj)
    else:
        # For other types, convert to string and count
        return len(str(obj))


def count_fields(obj: Any) -> int:
    """
    Recursively count the total number of fields/keys in a data structure.
    """  # noqa: W293
    if isinstance(obj, dict):
        return len(obj) + sum(count_fields(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(count_fields(item) for item in obj)
    else:
        return 0


def filter_fields(run_dict: dict, select: Optional[List[str]]) -> dict:
    """
    Filter a run dictionary to only include selected fields.
    If select is None or empty, returns the full dictionary.
    """  # noqa: W293
    if not select:
        return run_dict

    filtered = {}
    for field in select:
        if field in run_dict:
            filtered[field] = run_dict[field]
    return filtered


def build_trace_tree(run_dict: dict, depth: int = 0) -> dict:
    """
    Build a simplified trace tree structure showing top-level fields with metrics for nested content.

    Args:
        run_dict: The dictionary to build a tree from
        depth: How many levels deep to show actual content before summarizing.
               0 = summarize all nested structures (default)
               1 = show one level deep, then summarize
               2 = show two levels deep, then summarize
               etc.
    """  # noqa: W293
    tree = {}
    for key, value in run_dict.items():
        if isinstance(value, dict):
            if len(value) == 0:
                # Empty dictionary - just return empty dict
                tree[key] = {}
            elif depth > 0:
                # Show one level of content, then summarize deeper
                tree[key] = build_trace_tree(value, depth - 1)
            else:
                # For dictionaries, show metrics
                field_count = count_fields(value)
                if field_count == 0:
                    # Empty dictionary - just return empty dict
                    tree[key] = {}
                else:
                    tree[key] = {
                        "_type": "dict",
                        "_field_count": field_count,
                        "_character_count": count_characters(value),
                        "_keys": list(value.keys())[:10],  # Show first 10 keys as preview
                    }
        elif isinstance(value, list):
            # For lists, show metrics
            if len(value) == 0:
                tree[key] = []
            elif depth > 0:
                # Show one level of content
                processed_items = []
                for item in value:
                    if isinstance(item, (dict, list)):
                        # Recursively process nested structures
                        if isinstance(item, dict):
                            processed_items.append(build_trace_tree(item, depth - 1))
                        else:  # list
                            processed_items.append(
                                [
                                    build_trace_tree(subitem, depth - 1)
                                    if isinstance(subitem, dict)
                                    else subitem
                                    for subitem in item
                                ]
                            )
                    else:
                        processed_items.append(item)
                tree[key] = processed_items
            else:
                # For lists, show metrics and a small preview
                preview = []
                for item in value[:2]:  # Take first 2 items
                    if isinstance(item, dict):
                        # For dict items, show just keys
                        preview.append({"_type": "dict", "_keys": list(item.keys())[:5]})
                    elif isinstance(item, list):
                        # For list items, show length
                        preview.append({"_type": "list", "_length": len(item)})
                    else:
                        # For primitive items, show the value (but limit length)
                        str_val = str(item)
                        preview.append(str_val[:100] if len(str_val) > 100 else str_val)

                tree[key] = {
                    "_type": "list",
                    "_length": len(value),
                    "_field_count": count_fields(value),
                    "_character_count": count_characters(value),
                    "_preview": preview,
                }
        else:
            # For primitive values, show directly
            tree[key] = value
    return tree
