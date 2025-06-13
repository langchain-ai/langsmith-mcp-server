"""Registration module for LangSmith MCP tools."""

from typing import Any, Dict, List, Optional

from langsmith_mcp_server.services.tools.datasets import list_datasets_tool
from langsmith_mcp_server.services.tools.prompts import (
    get_prompt_tool,
    list_prompts_tool,
)
from langsmith_mcp_server.services.tools.traces import (
    fetch_trace_tool,
    get_project_runs_stats_tool,
    get_thread_history_tool,
)


def register_tools(mcp, langsmith_client):
    """
    Register all LangSmith tool-related functionality with the MCP server.
    This function configures and registers various tools for interacting with LangSmith,
    including prompt management, conversation history, traces, and analytics.

    Args:
        mcp: The MCP server instance to register tools with
        langsmith_client: The LangSmith client instance for API access
    """

    # Skip registration if client is not initialized
    if langsmith_client is None:
        return

    client = langsmith_client.get_client()

    @mcp.tool()
    def list_prompts(is_public: str = "false", limit: int = 20) -> Dict[str, Any]:
        """
        Fetch prompts from LangSmith with optional filtering.

        Args:
            is_public (str): Filter by prompt visibility - "true" for public prompts,
                            "false" for private prompts (default: "false")
            limit (int): Maximum number of prompts to return (default: 20)

        Returns:
            Dict[str, Any]: Dictionary containing the prompts and metadata
        """
        try:
            is_public_bool = is_public.lower() == "true"
            return list_prompts_tool(client, is_public_bool, limit)
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def get_prompt_by_name(prompt_name: str) -> Dict[str, Any]:
        """
        Get a specific prompt by its exact name.

        Args:
            prompt_name (str): The exact name of the prompt to retrieve

        Returns:
            Dict[str, Any]: Dictionary containing the prompt details and template,
                          or an error message if the prompt cannot be found
        """
        try:
            return get_prompt_tool(client, prompt_name=prompt_name)
        except Exception as e:
            return {"error": str(e)}

    # Register conversation tools
    @mcp.tool()
    def get_thread_history(thread_id: str, project_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve the message history for a specific conversation thread.

        Args:
            thread_id (str): The unique ID of the thread to fetch history for
            project_name (str): The name of the project containing the thread
                               (format: "owner/project" or just "project")

        Returns:
            List[Dict[str, Any]]: List of messages in chronological order from the thread history,
                                or an error message if the thread cannot be found
        """
        try:
            return get_thread_history_tool(client, thread_id, project_name)
        except Exception as e:
            return [{"error": str(e)}]

    # Register analytics tools
    @mcp.tool()
    def get_project_runs_stats(
        project_name: str, is_last_run: str = "true"
    ) -> Dict[str, Any]:
        """
        Get statistics about runs in a LangSmith project.

        Args:
            project_name (str): The name of the project to analyze
                              (format: "owner/project" or just "project")
            is_last_run (str): Set to "true" to get only the last run's stats,
                             set to "false" for overall project stats (default: "true")

        Returns:
            Dict[str, Any]: Dictionary containing the requested project run statistics
                          or an error message if statistics cannot be retrieved
        """
        try:
            is_last_run_bool = is_last_run.lower() == "true"
            return get_project_runs_stats_tool(client, project_name, is_last_run_bool)
        except Exception as e:
            return {"error": str(e)}

    # Register trace tools
    @mcp.tool()
    def fetch_trace(project_name: str = None, trace_id: str = None) -> Dict[str, Any]:
        """
        Fetch trace content for debugging and analyzing LangSmith runs.

        Note: Only one parameter (project_name or trace_id) is required.
        If both are provided, trace_id is preferred.
        String "null" inputs are handled as None values.

        Args:
            project_name (str, optional): The name of the project to fetch the latest trace from
            trace_id (str, optional): The specific ID of the trace to fetch (preferred parameter)

        Returns:
            Dict[str, Any]: Dictionary containing the trace data and metadata,
                          or an error message if the trace cannot be found
        """
        try:
            return fetch_trace_tool(client, project_name, trace_id)
        except Exception as e:
            return {"error": str(e)}

    # Register dataset tools
    @mcp.tool()
    def list_datasets(
        dataset_ids: Optional[List[str]] = None,
        data_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_name_contains: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        List datasets in a LangSmith project.
        """
        try:
            return list_datasets_tool(
                client,
                dataset_ids=dataset_ids,
                data_type=data_type,
                dataset_name=dataset_name,
                dataset_name_contains=dataset_name_contains,
                metadata=metadata,
                limit=20,
            )
        except Exception as e:
            return {"error": str(e)}
