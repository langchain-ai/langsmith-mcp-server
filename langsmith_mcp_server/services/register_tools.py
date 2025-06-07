"""Registration module for LangSmith MCP tools."""

from typing import Any, Dict, List

from langsmith_mcp_server.services.tools.prompts import (
    get_prompt_tool,
    list_prompts_tool,
)
from langsmith_mcp_server.services.tools.traces import (
    get_project_runs_stats_tool,
    get_thread_history_tool,
)


def register_tools(mcp, langsmith_client):
    """Register all tool-related functionality with the MCP server."""

    # Skip registration if client is not initialized
    if langsmith_client is None:
        return

    client = langsmith_client.get_client()

    @mcp.tool()
    def list_prompts(is_public: str = "false", limit: int = 20) -> Dict[str, Any]:
        """
        Fetch prompts from LangSmith with optional filtering.

        Args:
            is_public (str): Optional string ("true" or "false") to filter public/private prompts
            limit (int): Optional limit to the number of prompts to return
        Returns:
            Dictionary containing the prompts and metadata
        """
        try:
            is_public_bool = is_public.lower() == "true"
            return list_prompts_tool(client, is_public_bool, limit)
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def get_prompt_by_name(prompt_name: str) -> Dict[str, Any]:
        """
        Get a specific prompt by its name.

        Args:
            prompt_name: The name of the prompt to get

        Returns:
            Dictionary containing the prompt details and template
        """
        try:
            return get_prompt_tool(client, prompt_name=prompt_name)
        except Exception as e:
            return {"error": str(e)}

    # Register conversation tools
    @mcp.tool()
    def get_thread_history(thread_id: str, project_name: str) -> List[Dict[str, Any]]:
        """
        Get the history for a specific thread.

        Args:
            thread_id: The ID of the thread to fetch history for
            project_name: The name of the project containing the thread

        Returns:
            List of messages in the thread history
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
        Get the project runs stats
        Args:
            project_name (str): The name of the project
            is_last_run (str): "true" to get last run stats, "false" for overall project stats
        Returns:
            dict | None: The project runs stats
        """
        try:
            is_last_run_bool = is_last_run.lower() == "true"
            return get_project_runs_stats_tool(client, project_name, is_last_run_bool)
        except Exception as e:
            return {"error": str(e)}
