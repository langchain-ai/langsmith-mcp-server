"""Tools for interacting with LangSmith traces and conversations."""

from typing import Any, Dict, List

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


def get_thread_history_tool(client, thread_id: str, project_name: str) -> List[Dict[str, Any]]:
    """
    Get the history for a specific thread.

    Args:
        client: LangSmith client instance
        thread_id: The ID of the thread to fetch history for
        project_name: The name of the project containing the thread

    Returns:
        List of messages in the thread history
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
            return [{"error": f"No runs found for thread {thread_id} in project {project_name}"}]

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
            return [{"error": f"No messages found in the run for thread {thread_id}"}]

        return messages

    except Exception as e:
        return [{"error": f"Error fetching thread history: {str(e)}"}]


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
