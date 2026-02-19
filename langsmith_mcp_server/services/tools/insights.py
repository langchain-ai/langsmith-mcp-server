"""Tools for LangSmith Insights (clustering analysis). Uses REST API only."""

import json
import urllib.parse
import urllib.request
from typing import Any

from langsmith import Client

from langsmith_mcp_server.common.pagination import paginate_runs

_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"

# Hard cap for pagination character budget (matches traces.py)
_MAX_CHARS_PER_PAGE = 30000


def _request(
    api_key: str,
    endpoint: str,
    path: str,
    params: dict[str, str] | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any] | list[Any]:
    """GET request to LangSmith API. Returns JSON (dict or list)."""
    base = (endpoint or _DEFAULT_ENDPOINT).rstrip("/")
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("X-API-Key", api_key)
    if workspace_id:
        req.add_header("X-Tenant-Id", workspace_id)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {body}"}
    except OSError as e:
        return {"error": str(e)}


def _resolve_session_id(
    client: Client,
    project_name: str | None = None,
    session_id: str | None = None,
) -> str:
    """Resolve a project name or session_id to a session UUID string.

    Args:
        client: LangSmith client (used to look up project by name).
        project_name: Human-readable project name.
        session_id: Direct session UUID.

    Returns:
        Session UUID as a string.

    Raises:
        ValueError: If neither project_name nor session_id is provided.
    """
    if session_id:
        return session_id
    if not project_name:
        raise ValueError("Either project_name or session_id must be provided.")
    project = client.read_project(project_name=project_name)
    return str(project.id)


def _extract_items(
    result: dict[str, Any] | list[Any],
    *keys: str,
) -> list[Any]:
    """Extract a flat list of items from the various API response shapes.

    The insights API may return:
    - A plain list:  ``[item, ...]``
    - A list wrapping a dict: ``[{"clustering_jobs": [item, ...]}]``
    - A dict with a known key: ``{"runs": [item, ...]}``

    This helper handles all three by searching *keys* (e.g.
    ``"clustering_jobs"``, ``"runs"``, ``"items"``) in order.
    """
    # Unwrap [{"key": [...]}] → inner list
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict):
            for key in keys:
                if key in result[0] and isinstance(result[0][key], list):
                    return result[0][key]
        return result

    if isinstance(result, dict):
        for key in keys:
            if key in result and isinstance(result[key], list):
                return result[key]

    return [result] if result else []


def _normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Ensure key cross-reference fields are present as strings at the top level.

    Extracts ``id``, ``trace_id``, and ``thread_id`` so that downstream callers
    (e.g. ``fetch_runs``, ``get_thread_history``) can be invoked directly with
    the values returned here.
    """
    if "id" in run:
        run["id"] = str(run["id"])
    if "trace_id" in run:
        run["trace_id"] = str(run["trace_id"])

    # thread_id may live inside metadata under several conventional keys
    if "thread_id" not in run or not run["thread_id"]:
        metadata = run.get("metadata", {}) or {}
        for key in ("thread_id", "session_id", "conversation_id"):
            if key in metadata and metadata[key]:
                run["thread_id"] = str(metadata[key])
                break
    elif run.get("thread_id"):
        run["thread_id"] = str(run["thread_id"])

    return run


def list_insights_jobs_tool(
    api_key: str,
    endpoint: str,
    client: Client,
    project_name: str | None = None,
    session_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    List clustering (insights) jobs for a project.

    Args:
        api_key: LangSmith API key.
        endpoint: API base URL.
        client: LangSmith client (for project name resolution).
        project_name: Human-readable project name.
        session_id: Direct session UUID. Takes precedence over project_name.
        limit: Max jobs to return (capped at 100, default 20).
        offset: Number of jobs to skip (default 0).

    Returns:
        Dict with "jobs" list, or {"error": ...}.
    """
    try:
        sid = _resolve_session_id(client, project_name, session_id)
    except Exception as e:
        return {"error": str(e)}

    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    params: dict[str, str] = {"limit": str(limit), "offset": str(offset)}
    result = _request(api_key, endpoint, f"/api/v1/sessions/{sid}/insights", params, workspace_id)
    if isinstance(result, dict) and "error" in result:
        return result
    return {"jobs": _extract_items(result, "clustering_jobs", "jobs", "items")}


def get_insights_job_tool(
    api_key: str,
    endpoint: str,
    client: Client,
    job_id: str,
    project_name: str | None = None,
    session_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    Get a specific insights job with its clusters and report.

    Args:
        api_key: LangSmith API key.
        endpoint: API base URL.
        client: LangSmith client (for project name resolution).
        job_id: The clustering job UUID.
        project_name: Human-readable project name.
        session_id: Direct session UUID. Takes precedence over project_name.

    Returns:
        Dict with job details, or {"error": ...}.
    """
    try:
        sid = _resolve_session_id(client, project_name, session_id)
    except Exception as e:
        return {"error": str(e)}

    result = _request(
        api_key,
        endpoint,
        f"/api/v1/sessions/{sid}/insights/{job_id}",
        workspace_id=workspace_id,
    )
    if isinstance(result, dict):
        return result
    return {"error": "Unexpected response format"}


def get_insights_cluster_tool(
    api_key: str,
    endpoint: str,
    client: Client,
    job_id: str,
    cluster_id: str,
    project_name: str | None = None,
    session_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    Get details of a specific cluster within an insights job.

    Args:
        api_key: LangSmith API key.
        endpoint: API base URL.
        client: LangSmith client (for project name resolution).
        job_id: The clustering job UUID.
        cluster_id: The cluster UUID.
        project_name: Human-readable project name.
        session_id: Direct session UUID. Takes precedence over project_name.

    Returns:
        Dict with cluster details, or {"error": ...}.
    """
    try:
        sid = _resolve_session_id(client, project_name, session_id)
    except Exception as e:
        return {"error": str(e)}

    path = f"/api/v1/sessions/{sid}/insights/{job_id}/clusters/{cluster_id}"
    result = _request(api_key, endpoint, path, workspace_id=workspace_id)
    if isinstance(result, dict):
        return result
    return {"error": "Unexpected response format"}


def get_insights_runs_tool(
    api_key: str,
    endpoint: str,
    client: Client,
    job_id: str,
    project_name: str | None = None,
    session_id: str | None = None,
    cluster_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    sort_by: str | None = None,
    sort_direction: str | None = None,
    workspace_id: str | None = None,
    page_number: int = 1,
    max_chars_per_page: int = 25000,
    preview_chars: int = 150,
) -> dict[str, Any]:
    """
    Get runs associated with an insights job, optionally filtered by cluster.

    Results are paginated by character budget (same as ``fetch_runs``). Each run
    is normalised to include ``id``, ``trace_id``, and ``thread_id`` at the top
    level so they can be passed directly to ``fetch_runs`` or
    ``get_thread_history``.

    Args:
        api_key: LangSmith API key.
        endpoint: API base URL.
        client: LangSmith client (for project name resolution).
        job_id: The clustering job UUID.
        project_name: Human-readable project name.
        session_id: Direct session UUID. Takes precedence over project_name.
        cluster_id: Optional cluster UUID to filter runs to a single cluster.
        limit: Max runs to return from the API (capped at 100, default 20).
        offset: Number of runs to skip at the API level (default 0).
        sort_by: Optional field to sort by (e.g. "start_time", "latency").
        sort_direction: Optional sort direction ("asc" or "desc").
        page_number: 1-based page index for char-based pagination (default 1).
        max_chars_per_page: Max JSON chars per page, capped at 30000 (default 25000).
        preview_chars: Truncate long strings to this length (default 150).

    Returns:
        Paginated dict with "runs", "page_number", "total_pages",
        "max_chars_per_page", "preview_chars". May include "_truncated" keys
        if content exceeded budget. Or {"error": ...}.
    """
    try:
        sid = _resolve_session_id(client, project_name, session_id)
    except Exception as e:
        return {"error": str(e)}

    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    max_chars_per_page = min(max_chars_per_page, _MAX_CHARS_PER_PAGE)

    params: dict[str, str] = {"limit": str(limit), "offset": str(offset)}
    if cluster_id:
        params["cluster_id"] = cluster_id
    if sort_by:
        params["sort_by"] = sort_by
    if sort_direction:
        params["sort_direction"] = sort_direction

    result = _request(
        api_key,
        endpoint,
        f"/api/v1/sessions/{sid}/insights/{job_id}/runs",
        params,
        workspace_id,
    )
    if isinstance(result, dict) and "error" in result:
        return result

    # Extract the list of runs from whichever shape the API returned
    raw_runs = _extract_items(result, "runs", "items")

    # Normalize each run so id, trace_id, thread_id are top-level strings
    runs = [_normalize_run(r) for r in raw_runs]

    return paginate_runs(
        runs,
        page_number=page_number,
        max_chars_per_page=max_chars_per_page,
        preview_chars=preview_chars,
    )
