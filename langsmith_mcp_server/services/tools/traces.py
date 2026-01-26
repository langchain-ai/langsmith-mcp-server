"""Tools for interacting with LangSmith traces and conversations."""

import os
from typing import Any, Dict, Iterable, List, Optional, Union

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from fastmcp.server import Context
from langsmith import Client
from langsmith.schemas import Run

from langsmith_mcp_server.common.formatters import (
    extract_messages_from_run,
    format_messages,
)
from langsmith_mcp_server.common.helpers import (
    convert_uuids_to_strings,
    find_in_dict,
)

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


# fetch_trace_tool was removed as it's redundant with fetch_runs.
# Use fetch_runs with trace_id and limit=1 to achieve the same functionality.


def get_thread_history_tool(client: Client, thread_id: str, project_name: str) -> Dict[str, Any]:
    """
    Get the history for a specific thread.

    Args:
        client: LangSmith client instance
        thread_id: The ID of the thread to fetch history for
        project_name: The name of the project containing the thread

    Returns:
        A dictionary containing a list of messages in the thread history or an error.
    """
    try:
        # Filter runs by the specific thread and project
        filter_string = (
            f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), '
            f'eq(metadata_value, "{thread_id}"))'
        )

        # Only grab the LLM runs
        runs = [
            r
            for r in client.list_runs(
                project_name=project_name, filter=filter_string, run_type="llm"
            )
        ]

        if not runs or len(runs) == 0:
            return {"error": f"No runs found for thread {thread_id} in project {project_name}"}

        # Sort by start time to get the most recent interaction
        runs = sorted(runs, key=lambda run: run.start_time, reverse=True)

        # Get the most recent run
        latest_run = runs[0]

        # Extract messages from inputs and outputs
        messages = []

        # Add input messages if they exist
        if hasattr(latest_run, "inputs") and "messages" in latest_run.inputs:
            messages.extend(latest_run.inputs["messages"])

        # Add output message if it exists
        if hasattr(latest_run, "outputs"):
            if isinstance(latest_run.outputs, dict) and "choices" in latest_run.outputs:
                if (
                    isinstance(latest_run.outputs["choices"], list)
                    and len(latest_run.outputs["choices"]) > 0
                ):
                    if "message" in latest_run.outputs["choices"][0]:
                        messages.append(latest_run.outputs["choices"][0]["message"])
            elif isinstance(latest_run.outputs, dict) and "message" in latest_run.outputs:
                messages.append(latest_run.outputs["message"])

        if not messages or len(messages) == 0:
            return {"error": f"No messages found in the run for thread {thread_id}"}

        return {"result": messages}

    except Exception as e:
        return {"error": f"Error fetching thread history: {str(e)}"}


def get_project_runs_stats_tool(
    client: Client,
    project_name: str = None,
    trace_id: str = None,
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
        # Break down the qualified project name if project_name is provided
        actual_project_name = None
        if project_name:
            parts = project_name.split("/")
            is_qualified = len(parts) == 2
            actual_project_name = parts[1] if is_qualified else project_name

        # Get the project runs stats
        project_runs_stats = client.get_run_stats(
            project_names=[actual_project_name] if actual_project_name else None,
            trace=trace_id if trace_id else None,
        )
        # remove the run_facets from the project_runs_stats
        project_runs_stats.pop("run_facets", None)
        # add project_name to the project_runs_stats if available
        if actual_project_name:
            project_runs_stats["project_name"] = actual_project_name
        return project_runs_stats
    except Exception as e:
        return {"error": f"Error getting project runs stats: {str(e)}"}


def list_projects_tool(
    client: Client,
    limit: int = 5,
    project_name: str = None,
    more_info: bool = False,
    reference_dataset_id: str = None,
    reference_dataset_name: str = None,
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
    format_type: str = "pretty",
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Fetch LangSmith runs (traces, tools, chains, etc.) from one or more projects
    using flexible filters, query language expressions, and trace-level constraints.

    Args:
        client: LangSmith client instance
        project_name: The name of the project to fetch the runs from
        trace_id: The ID of the trace to fetch the runs from
        run_type: The type of the run to fetch
        error: Whether to fetch errored runs
        is_root: Whether to fetch root runs
        filter: The filter to apply to the runs
        trace_filter: The filter to apply to the trace
        tree_filter: The filter to apply to the tree
        order_by: The order by to apply to the runs
        limit: The limit to apply to the runs
        reference_example_id: The ID of the reference example to filter runs by
        format_type: Output format for messages ('raw', 'json', or 'pretty'). Default: 'pretty'.
                    When set, extracts and formats messages from runs for conversational AI use.
                    When format_type is set, returns only the formatted output (not the runs).
        ctx: Optional FastMCP context for getting API key and endpoint
    Returns:
        Dictionary containing:
        - If format_type is set: {"formatted": str} - formatted string representation of messages
        - If format_type is not set: {"runs": List[Dict]} - list of run dictionaries
    """
    # If format_type is set and trace_id is provided, use REST API directly
    # This avoids unnecessary list_runs call and gets all messages from trace tree
    if format_type and trace_id and HAS_REQUESTS:
        try:
            # Get API key, workspace ID, and base URL from context or environment
            api_key = None
            workspace_id = None
            base_url = None

            # Try to get from context first (supports both HTTP and STDIO)
            if ctx:
                api_key = ctx.get_state("api_key")
                workspace_id = ctx.get_state("workspace_id")
                endpoint = ctx.get_state("endpoint")
                if endpoint:
                    base_url = endpoint

                # If not in session state, try context variables (set by middleware for HTTP transport)
                if not api_key:
                    try:
                        api_key = api_key_context.get("")
                        if api_key:
                            workspace_id = workspace_id_context.get("") or None
                            endpoint = endpoint_context.get("") or None
                            if endpoint:
                                base_url = endpoint
                            # Store in session for future requests
                            ctx.set_state("api_key", api_key)
                            if workspace_id:
                                ctx.set_state("workspace_id", workspace_id)
                            if endpoint:
                                ctx.set_state("endpoint", endpoint)
                    except (RuntimeError, Exception):
                        pass

                # If still not found, try HTTP headers
                if not api_key:
                    try:
                        request = ctx.get_http_request()
                        if request:
                            # Try request.state first (set by middleware), then headers
                            if hasattr(request.state, "api_key") and request.state.api_key:
                                api_key = request.state.api_key
                                workspace_id = getattr(request.state, "workspace_id", None)
                                endpoint = getattr(request.state, "endpoint", None)
                                if endpoint:
                                    base_url = endpoint
                            else:
                                # Fall back to headers directly
                                api_key = request.headers.get("LANGSMITH-API-KEY")
                                workspace_id = request.headers.get("LANGSMITH-WORKSPACE-ID")
                                endpoint = request.headers.get("LANGSMITH-ENDPOINT")
                                if endpoint:
                                    base_url = endpoint
                    except (RuntimeError, Exception):
                        pass

            # Fall back to environment variables
            if not api_key:
                api_key = os.environ.get("LANGSMITH_API_KEY")
            if not workspace_id:
                workspace_id = os.environ.get("LANGSMITH_WORKSPACE_ID")
            if not base_url:
                base_url = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

            if not base_url.startswith("http"):
                base_url = f"https://{base_url}"

            if api_key:
                # Use REST API to get trace with messages directly
                headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
                # Add X-Tenant-Id header if workspace_id is provided
                if workspace_id:
                    headers["X-Tenant-Id"] = workspace_id

                url = f"{base_url}/runs/{trace_id}?include_messages=true"

                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                data = response.json()
                # Convert UUIDs to strings for JSON serialization
                run_dict = convert_uuids_to_strings(data)

                # Extract messages - same logic as langsmith_cli
                messages = data.get("messages")
                if not messages:
                    outputs = data.get("outputs")
                    if isinstance(outputs, dict):
                        output_messages = outputs.get("messages")
                        messages = output_messages or []
                    else:
                        messages = []

                # When format_type is set, return only the formatted output
                if messages:
                    formatted = format_messages(messages, format_type)
                    return {"formatted": formatted}
                else:
                    # No messages found, return empty formatted output
                    return {"formatted": "No messages found in this trace."}
        except Exception:
            # Fall back to list_runs if REST API fails
            pass

    # Default path: use list_runs (for non-format_type requests or when REST API fails)
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
        limit=limit,
        reference_example_id=reference_example_id,
    )
    runs_dict = []
    for run in runs_iter:
        run_dict = run.dict()
        # Convert UUID objects to strings for JSON serialization
        run_dict = convert_uuids_to_strings(run_dict)
        runs_dict.append(run_dict)

    # Extract and format messages if format_type is provided
    if format_type:
        all_messages = []
        for run_dict in runs_dict:
            try:
                run_messages = extract_messages_from_run(run_dict)
                if run_messages:
                    all_messages.extend(run_messages)
            except Exception:
                # Log the error but continue processing other runs
                # This prevents one malformed run from breaking the entire request
                # The error is likely due to unexpected data structure in inputs/outputs
                pass

        # When format_type is set, return only the formatted output
        if all_messages:
            formatted = format_messages(all_messages, format_type)
            return {"formatted": formatted}
        else:
            # No messages found, return empty formatted output
            return {"formatted": "No messages found in the runs."}

    # When format_type is not set, return runs as before
    return {"runs": runs_dict}
