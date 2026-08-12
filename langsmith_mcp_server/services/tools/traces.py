"""Tools for interacting with LangSmith traces and conversations."""

from typing import Any, Dict, Iterable, List, Optional, Union

from langsmith import Client
from langsmith.schemas import Run

from langsmith_mcp_server.common.helpers import (
    convert_uuids_to_strings,
    find_in_dict,
)
from langsmith_mcp_server.common.pagination import (
    paginate_messages,
    paginate_runs,
)

# LangSmith API maximum for list_runs limit
LANGSMITH_LIST_RUNS_MAX_LIMIT = 100

# Hard cap for trace pagination: pages cannot exceed this character budget
MAX_CHARS_PER_PAGE_TRACE = 30000

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


def fetch_trace_tool(
    client: Client, project_name: str | None = None, trace_id: str | None = None
) -> Dict[str, Any]:
    """
    Fetch the trace content for a specific project or specify a trace ID.

    Note: Only one of the parameters (project_name or trace_id) is required.
    trace_id is preferred if both are provided.

    Args:
        client: LangSmith client instance
        project_name: The name of the project to fetch the last trace for
        trace_id: The ID of the trace to fetch (preferred parameter)

    Returns:
        Dictionary containing the last trace and metadata
    """
    # Handle None values and "null" string inputs
    if project_name == "null":
        project_name = None
    if trace_id == "null":
        trace_id = None

    if not project_name and not trace_id:
        return {"error": "Error: Either project_name or trace_id must be provided."}

    try:
        # Get the last run
        runs = client.list_runs(
            project_name=project_name if project_name else None,
            id=[trace_id] if trace_id else None,
            select=[
                "inputs",
                "outputs",
                "run_type",
                "id",
                "error",
                "total_tokens",
                "total_cost",
                "feedback_stats",
                "app_path",
                "thread_id",
            ],
            is_root=True,
            limit=1,
        )

        runs = list(runs)

        if not runs or len(runs) == 0:
            return {"error": "No runs found for project_name: {}".format(project_name)}

        run = runs[0]

        # Return just the trace ID as we can use this to open the trace view
        return {
            "trace_id": str(run.id),
            "run_type": run.run_type,
            "id": str(run.id),
            "error": run.error,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "total_tokens": run.total_tokens,
            "total_cost": str(run.total_cost),
            "feedback_stats": run.feedback_stats,
            "app_path": run.app_path,
            "thread_id": str(run.thread_id) if hasattr(run, "thread_id") else None,
        }
    except Exception as e:
        return {"error": f"Error fetching last trace: {str(e)}"}


def _messages_from_run(run: Run) -> List[Dict[str, Any]]:
    """Extract messages from a single run's inputs and outputs."""
    messages: List[Dict[str, Any]] = []
    if hasattr(run, "inputs") and run.inputs and "messages" in run.inputs:
        messages.extend(run.inputs["messages"])
    if hasattr(run, "outputs") and run.outputs:
        if isinstance(run.outputs, dict) and "choices" in run.outputs:
            if isinstance(run.outputs["choices"], list) and len(run.outputs["choices"]) > 0:
                if "message" in run.outputs["choices"][0]:
                    messages.append(run.outputs["choices"][0]["message"])
        elif isinstance(run.outputs, dict) and "message" in run.outputs:
            messages.append(run.outputs["message"])
    return messages


def get_thread_history_tool(
    client: Client,
    thread_id: str,
    project_name: str,
    page_number: int,
    max_chars_per_page: int = 25000,
    preview_chars: int = 150,
) -> Dict[str, Any]:
    """
    Get one page of message history for a specific thread (char-based pagination).

    Fetches LLM runs for the thread, sorts by start_time ascending (chronological),
    flattens messages from all runs, then paginates by character budget. Long strings
    are truncated to preview_chars. Use page_number (1-based) and total_pages to iterate.

    Args:
        client: LangSmith client instance
        thread_id: The ID of the thread to fetch history for
        project_name: The name of the project containing the thread
        page_number: 1-based page index (required)
        max_chars_per_page: Max character count per page (capped at 30000). Default 25000.
        preview_chars: Truncate long strings to this length with "… (+N chars)". Default 150.

    Returns:
        Dict with result (messages for this page), page_number, total_pages,
        max_chars_per_page, preview_chars. May include _truncated, _truncated_message,
        _truncated_preview if content was cut. Or an error dict.
    """
    try:
        max_chars_per_page = min(max_chars_per_page, MAX_CHARS_PER_PAGE_TRACE)

        filter_string = (
            f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), '
            f'eq(metadata_value, "{thread_id}"))'
        )
        runs = [
            r
            for r in client.list_runs(
                project_name=project_name,
                filter=filter_string,
                run_type="llm",
                limit=LANGSMITH_LIST_RUNS_MAX_LIMIT,
            )
        ]

        if not runs:
            return {"error": f"No runs found for thread {thread_id} in project {project_name}"}

        # Chronological order (oldest first) for history
        runs = sorted(runs, key=lambda run: run.start_time or "")

        all_messages: List[Dict[str, Any]] = []
        for run in runs:
            all_messages.extend(_messages_from_run(run))

        if not all_messages:
            return {"error": f"No messages found in the runs for thread {thread_id}"}

        return paginate_messages(
            all_messages,
            page_number=page_number,
            max_chars_per_page=max_chars_per_page,
            preview_chars=preview_chars,
        )

    except Exception as e:
        return {"error": f"Error fetching thread history: {str(e)}"}


def get_project_runs_stats_tool(
    client: Client,
    project_name: str | None = None,
    trace_id: str | None = None,
) -> Dict[str, Any]:
    """
    Get the project runs stats.

    Note: Only one of the parameters (project_name or trace_id) is required.
    trace_id is preferred if both are provided.

    Args:
        client: LangSmith client instance
        project_name: The name of the project to fetch the runs stats for
        trace_id: The ID of the trace to fetch (preferred parameter)

    Returns:
        Dictionary containing the project runs stats
    """
    # Handle None values and "null" string inputs
    if project_name == "null":
        project_name = None
    if trace_id == "null":
        trace_id = None

    if not project_name and not trace_id:
        return {"error": "Error: Either project_name or trace_id must be provided."}

    try:
        # Break down the qualified project name
        parts = project_name.split("/")
        is_qualified = len(parts) == 2
        actual_project_name = parts[1] if is_qualified else project_name

        # Get the project runs stats
        project_runs_stats = client.get_run_stats(
            project_names=[actual_project_name] if project_name else None,
            trace=trace_id if trace_id else None,
        )
        # remove the run_facets from the project_runs_stats
        project_runs_stats.pop("run_facets", None)
        # add project_name to the project_runs_stats
        project_runs_stats["project_name"] = actual_project_name
        return project_runs_stats
    except Exception as e:
        return {"error": f"Error getting project runs stats: {str(e)}"}


def list_projects_tool(
    client: Client,
    limit: int = 5,
    project_name: str | None = None,
    more_info: bool = False,
    reference_dataset_id: str | None = None,
    reference_dataset_name: str | None = None,
) -> Dict[str, Any]:
    """
    List projects from LangSmith.

    Args:
        client: LangSmith Client instance
        limit: Maximum number of projects to return (default: 5)
        project_name: Filter projects by name
        more_info: Return more detailed project information (default: False)
        reference_dataset_id: Filter projects by reference dataset ID
        reference_dataset_name: Filter projects by reference dataset name
    Returns:
        Dictionary containing a "projects" key with a list of project dictionaries
    """
    projects = []
    for project in client.list_projects(
        reference_free=True,
        name_contains=project_name,
        limit=limit,  # this can be set by the agent
        reference_dataset_id=reference_dataset_id,
        reference_dataset_name=reference_dataset_name,
    ):
        projects.append(project.dict())

    if more_info:
        return {"projects": projects}
    else:
        simple_projects = []
        for project in projects:
            deployment_id = find_in_dict(project, "deployment_id")
            project_id = project.get("id", None)
            project_dict = {
                "name": project.get("name", None),
                "project_id": str(project_id) if project_id is not None else None,
            }
            if deployment_id:
                project_dict["agent_deployment_id"] = deployment_id
            simple_projects.append(project_dict)
        return {"projects": simple_projects}


def fetch_runs_tool(
    client: Client,
    project_name: Union[str, List[str]],
    page_number: int = 1,
    max_chars_per_page: int = 25000,
    preview_chars: int = 150,
    trace_id: Optional[str] = None,
    run_type: Optional[str] = None,
    error: Optional[bool] = None,
    is_root: Optional[bool] = None,
    filter: Optional[str] = None,
    trace_filter: Optional[str] = None,
    tree_filter: Optional[str] = None,
    order_by: str = "-start_time",
    limit: int = 50,
    reference_example_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch LangSmith runs (traces, tools, chains, etc.) from one or more projects
    using flexible filters, query language expressions, and trace-level constraints.
    Results are always paginated by character budget.

    Args:
        client: LangSmith client instance
        project_name: The name of the project to fetch the runs from
        page_number: 1-based page index (default: 1)
        max_chars_per_page: Max character count per page (compact JSON). Capped at 30000. Default 25000.
        preview_chars: Truncate long strings to this length with "… (+N chars)". Default 150.
        trace_id: The ID of the trace to fetch the runs from
        run_type: The type of the run to fetch
        error: Whether to fetch errored runs
        is_root: Whether to fetch root runs
        filter: The filter to apply to the runs
        trace_filter: The filter to apply to the trace
        tree_filter: The filter to apply to the tree
        order_by: The order by to apply to the runs
        limit: The limit to apply to the runs (capped at 100 by LangSmith API)
        reference_example_id: The ID of the reference example to filter runs by
    Returns:
        Dictionary with runs, page_number, total_pages, max_chars_per_page, preview_chars.
        May include _truncated, _truncated_message, _truncated_preview if content was cut.
    """
    max_chars_per_page = min(max_chars_per_page, MAX_CHARS_PER_PAGE_TRACE)
    capped_limit = (
        min(limit, LANGSMITH_LIST_RUNS_MAX_LIMIT)
        if limit is not None
        else LANGSMITH_LIST_RUNS_MAX_LIMIT
    )
    runs_iter: Iterable[Run] = client.list_runs(
        project_name=project_name,
        trace_id=trace_id,
        run_type=run_type,
        error=error,
        is_root=is_root,
        filter=filter,
        trace_filter=trace_filter,
        tree_filter=tree_filter,
        order_by=order_by,
        limit=capped_limit,
        reference_example_id=reference_example_id,
    )
    runs_dict = []
    for run in runs_iter:
        run_dict = run.dict()
        # Convert UUID objects to strings for JSON serialization
        run_dict = convert_uuids_to_strings(run_dict)
        runs_dict.append(run_dict)

    # Always paginate the results
    return paginate_runs(
        runs_dict,
        page_number=page_number,
        max_chars_per_page=max_chars_per_page,
        preview_chars=preview_chars,
    )
