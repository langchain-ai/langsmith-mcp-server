"""Tools for interacting with LangSmith traces and conversations."""

from typing import Any, Dict, List, Optional

from langsmith_mcp_server.common.helpers import get_last_run_stats


def fetch_trace_tool(client, project_name: str = None, trace_id: str = None) -> Dict[str, Any]:
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
    if not project_name and not trace_id:
        return {"error": "Error: Either project_name or trace_id must be provided."}

    try:
        # Get the last run
        runs = client.list_runs(
            project_name=project_name if project_name else None,
            id=[trace_id] if trace_id else None,
            select=["inputs", "outputs", "run_type", "id"],
            is_root=True,
            limit=1,
        )

        runs = list(runs)

        if not runs or len(runs) == 0:
            return {"error": "No runs found for project_name: {}".format(project_name)}

        run = runs[0]

        # Return just the trace ID as we can use this to open the trace view
        return {
            "trace_id": run.id,
            "run_type": run.run_type,
            "inputs": run.inputs,
            "outputs": run.outputs,
        }
    except Exception as e:
        return {"error": f"Error fetching last trace: {str(e)}"}


def get_thread_history_tool(client, thread_id: str, project_name: str) -> Dict[str, Any]:
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
    client, project_name: str, is_last_run: bool = True
) -> Dict[str, Any]:
    """
    Get the project runs stats

    Args:
        client: LangSmith client instance
        project_name (str): The name of the project
        is_last_run (bool): Whether to get only the last run stats or all stats

    Returns:
        dict: The project runs stats
    """
    try:
        if is_last_run:
            return get_last_run_stats(client, project_name)

        # Break down the qualified project name
        parts = project_name.split("/")
        is_qualified = len(parts) == 2
        actual_project_name = parts[1] if is_qualified else project_name

        # Get the project runs stats
        project_runs_stats = client.get_run_stats(project_names=[actual_project_name])
        # remove the run_facets from the project_runs_stats
        project_runs_stats.pop("run_facets", None)
        # add project_name to the project_runs_stats
        project_runs_stats["project_name"] = actual_project_name
        return project_runs_stats
    except Exception as e:
        return {"error": f"Error getting project runs stats: {str(e)}"}

def list_runs_for_trace_tool(
    client,
    project_name: Optional[str] = None,
    project_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    run_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    List the runs that belong to a specific trace and return minimal metadata.

    This utility is intended to be called first in a trace inspection flow to
    enumerate run IDs associated with a trace. You can then fetch full details
    for specific runs using `get_run_tool`.

    Args:
        client: LangSmith client instance.
        project_name: Optional project name to further scope the search.
        project_id: Optional project UUID to further scope the search.
        trace_id: The trace/run UUID to list runs for (required).
        run_count: Optional[int] — omit to return all runs (do not pass null); 0 returns none.

    Returns:
        Dict[str, Any]: A dictionary with the following keys:
            - "trace_id": The trace UUID provided.
            - "total_count": Number of runs returned in this response.
            - "runs": List of runs sorted by start time then id, where each
              entry contains minimal metadata:
                {"id", "name", "run_type", "parent_run_id"}.

        On error, returns {"error": <message>}.
    """
    try:
        if not trace_id:
            return {"error": "trace_id is required"}

        # Build base kwargs. We will rely on the SDK's native `trace` filter
        # instead of a manual filter expression to ensure compatibility with
        # the backend API requirements.
        kwargs: Dict[str, Any] = {}
        # Only use project_name if it is intentionally provided; do not
        # reinterpret UUID-looking values as project_id.
        if project_name:
            kwargs["project_name"] = project_name
        if project_id:
            kwargs["project_id"] = project_id

        # Determine requested count; None means fetch all
        # Coerce a JSON number (int or float) to an integer count deterministically
        requested_count: Optional[int] = None if run_count is None else max(0, int(run_count))

        # If an explicit request of 0 runs was made, return empty result set.
        if requested_count == 0:
            return {"trace_id": trace_id, "total_count": 0, "runs": []}

        # If a limit was requested, push that down to the API to avoid fetching all pages
        api_limit = requested_count if requested_count is not None else None
        collected = []
        generator = client.list_runs(trace=trace_id, limit=api_limit, **kwargs)

        for run in generator:
            collected.append(run)

        # Sort runs by start_time then id for determinism
        collected.sort(
            key=lambda r: (
                getattr(r, "start_time", None) or 0,
                str(getattr(r, "id", "")),
            )
        )

        # Apply run_count limitation after sorting to ensure deterministic selection
        selected = collected if requested_count is None else collected[: requested_count]

        minimal: List[Dict[str, Any]] = [
            {
                "id": str(getattr(r, "id", "")),
                "name": getattr(r, "name", None),
                "run_type": getattr(r, "run_type", None),
                "parent_run_id": str(getattr(r, "parent_run_id", "")),
            }
            for r in selected
        ]
        return {"trace_id": trace_id, "total_count": len(minimal), "runs": minimal}
    except Exception as e:
        return {"error": f"Error listing runs for trace: {str(e)}"}


def get_run_tool(client, run_id: str) -> Dict[str, Any]:
    """
    Retrieve a single run by its UUID and return a fully formatted JSON payload.

    Args:
        client: LangSmith client instance.
        run_id: The run UUID to retrieve (required).

    Returns:
        Dict[str, Any]: A dictionary under the "run" key containing a
        JSON-serializable representation of the run with fields such as:
        "id", "name", "run_type", "trace_id", "parent_run_id",
        "inputs", "outputs", "error", "start_time", "end_time",
        "extra", "metadata", "events", "feedback_stats",
        "total_tokens", "prompt_tokens", and "completion_tokens".

        On error, returns {"error": <message>}.
    """
    try:
        if not run_id:
            return {"error": "run_id is required"}

        # Use dedicated read API for a single run if available
        run = client.read_run(run_id)
        if run is None:
            return {"error": f"Run not found: {run_id}"}

        def _dt(val):
            try:
                return val.isoformat() if val is not None else None
            except Exception:
                return None

        def _str(val):
            try:
                return str(val) if val is not None else None
            except Exception:
                return None

        formatted = {
            "id": _str(getattr(run, "id", None)),
            "name": getattr(run, "name", None),
            "run_type": getattr(run, "run_type", None),
            "trace_id": _str(getattr(run, "trace_id", None)),
            "parent_run_id": _str(getattr(run, "parent_run_id", None)),
            "inputs": getattr(run, "inputs", None),
            "outputs": getattr(run, "outputs", None),
            "error": getattr(run, "error", None),
            "start_time": _dt(getattr(run, "start_time", None)),
            "end_time": _dt(getattr(run, "end_time", None)),
            "extra": getattr(run, "extra", None),
            "metadata": getattr(run, "metadata", None),
            "events": getattr(run, "events", None),
            "feedback_stats": getattr(run, "feedback_stats", None),
            "total_tokens": getattr(run, "total_tokens", None),
            "prompt_tokens": getattr(run, "prompt_tokens", None),
            "completion_tokens": getattr(run, "completion_tokens", None),
        }

        return {"run": formatted}
    except Exception as e:
        return {"error": f"Error getting run: {str(e)}"}
