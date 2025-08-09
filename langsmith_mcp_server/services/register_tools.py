"""Registration module for LangSmith MCP tools."""

from typing import Any, Dict, List, Optional

from langsmith_mcp_server.services.tools.datasets import (
    list_datasets_tool,
    list_examples_tool,
    read_dataset_tool,
    read_example_tool,
)
from langsmith_mcp_server.services.tools.prompts import (
    get_prompt_tool,
    list_prompts_tool,
)
from langsmith_mcp_server.services.tools.traces import (
    fetch_trace_tool,
    get_project_runs_stats_tool,
    get_thread_history_tool,
    list_runs_for_trace_tool,
    get_run_tool,
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
    def get_thread_history(thread_id: str, project_name: str) -> Dict[str, Any]:
        """
        Retrieve the message history for a specific conversation thread.

        Args:
            thread_id (str): The unique ID of the thread to fetch history for
            project_name (str): The name of the project containing the thread
                               (format: "owner/project" or just "project")

        Returns:
            Dict[str, Any]: Dictionary containing the thread history,
                                or an error message if the thread cannot be found
        """
        try:
            return get_thread_history_tool(client, thread_id, project_name)
        except Exception as e:
            return {"error": str(e)}

    # Register analytics tools
    @mcp.tool()
    def get_project_runs_stats(project_name: str, is_last_run: str = "true") -> Dict[str, Any]:
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

    @mcp.tool(name="list runs for trace")
    def list_runs_for_trace(
        project_name: str = None,
        project_id: str = None,
        trace_id: str = None,
        run_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List runs for a specific trace and return minimal metadata.

        Usage pattern (important):
        1) Call this once to enumerate run IDs for a trace.
        2) Then call "get run" for each run_id you want full details for.
        3) Avoid re-calling this repeatedly; caching the IDs is recommended.

        Args:
            project_name (str): Optional project name to further scope the search.
            project_id (str): Optional project UUID to further scope the search.
            trace_id (str): Trace/run UUID to list runs for (required).
            run_count (Optional[int]): Omit to return all runs (do not pass null); 0 returns none.

        Returns:
            Dict[str, Any]:
                {
                    "trace_id": str,
                    "total_count": int,
                    "runs": [
                        {"id": str, "name": Optional[str], "run_type": Optional[str], "parent_run_id": str}
                    ]
                }

            On error, returns {"error": <message>}.
        """
        try:
            return list_runs_for_trace_tool(
                client,
                project_name=project_name,
                project_id=project_id,
                trace_id=trace_id,
                run_count=run_count,
            )
        except Exception as e:
            return {"error": str(e)}

    # New: get a single run by id
    @mcp.tool(name="get run")
    def get_run(run_id: str) -> Dict[str, Any]:
        """
        Fetch full details for a single run by UUID and return a JSON-serializable payload.

        Recommended flow:
        - First call "list runs for trace" to enumerate run IDs for a trace.
        - Then call this tool for each run you want to inspect.
        - Use this instead of re-calling "list runs for trace" multiple times.

        Args:
            run_id (str): The run UUID to retrieve (required).

        Returns:
            Dict[str, Any]: { "run": { ...fields... } } where the run contains:
            "id", "name", "run_type", "trace_id", "parent_run_id", "inputs",
            "outputs", "error", "start_time", "end_time", "extra", "metadata",
            "events", "feedback_stats", "total_tokens", "prompt_tokens",
            "completion_tokens".

            On error, returns {"error": <message>}.
        """
        try:
            return get_run_tool(client, run_id=run_id)
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
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Fetch LangSmith datasets.

        Note: If no arguments are provided, all datasets will be returned.

        Args:
            dataset_ids (Optional[List[str]]): List of dataset IDs to filter by
            data_type (Optional[str]): Filter by dataset data type (e.g., 'chat', 'kv')
            dataset_name (Optional[str]): Filter by exact dataset name
            dataset_name_contains (Optional[str]): Filter by substring in dataset name
            metadata (Optional[Dict[str, Any]]): Filter by metadata dict
            limit (int): Max number of datasets to return (default: 20)

        Returns:
            Dict[str, Any]: Dictionary containing the datasets and metadata,
                            or an error message if the datasets cannot be retrieved
        """
        try:
            return list_datasets_tool(
                client,
                dataset_ids=dataset_ids,
                data_type=data_type,
                dataset_name=dataset_name,
                dataset_name_contains=dataset_name_contains,
                metadata=metadata,
                limit=limit,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def list_examples(
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        example_ids: Optional[List[str]] = None,
        filter: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        splits: Optional[List[str]] = None,
        inline_s3_urls: Optional[bool] = None,
        include_attachments: Optional[bool] = None,
        as_of: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fetch examples from a LangSmith dataset with advanced filtering options.

        Note: Either dataset_id, dataset_name, or example_ids must be provided.
        If multiple are provided, they are used in order of precedence: example_ids, dataset_id, dataset_name.

        Args:
            dataset_id (Optional[str]): Dataset ID to retrieve examples from
            dataset_name (Optional[str]): Dataset name to retrieve examples from
            example_ids (Optional[List[str]]): List of specific example IDs to retrieve
            limit (Optional[int]): Maximum number of examples to return
            offset (Optional[int]): Number of examples to skip before starting to return results
            filter (Optional[str]): Filter string using LangSmith query syntax (e.g., 'has(metadata, {"key": "value"})')
            metadata (Optional[Dict[str, Any]]): Dictionary of metadata to filter by
            splits (Optional[List[str]]): List of dataset splits to include examples from
            inline_s3_urls (Optional[bool]): Whether to inline S3 URLs (default: SDK default if not specified)
            include_attachments (Optional[bool]): Whether to include attachments in response (default: SDK default if not specified)
            as_of (Optional[str]): Dataset version tag OR ISO timestamp to retrieve examples as of that version/time

        Returns:
            Dict[str, Any]: Dictionary containing the examples and metadata,
                            or an error message if the examples cannot be retrieved
        """
        try:
            return list_examples_tool(
                client,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                example_ids=example_ids,
                filter=filter,
                metadata=metadata,
                splits=splits,
                inline_s3_urls=inline_s3_urls,
                include_attachments=include_attachments,
                as_of=as_of,
                limit=limit,
                offset=offset,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def read_dataset(
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Read a specific dataset from LangSmith.

        Note: Either dataset_id or dataset_name must be provided to identify the dataset.
        If both are provided, dataset_id takes precedence.

        Args:
            dataset_id (Optional[str]): Dataset ID to retrieve
            dataset_name (Optional[str]): Dataset name to retrieve

        Returns:
            Dict[str, Any]: Dictionary containing the dataset details,
                            or an error message if the dataset cannot be retrieved
        """
        try:
            return read_dataset_tool(
                client,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def read_example(
        example_id: str,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Read a specific example from LangSmith.

        Args:
            example_id (str): Example ID to retrieve
            as_of (Optional[str]): Dataset version tag OR ISO timestamp to retrieve the example as of that version/time

        Returns:
            Dict[str, Any]: Dictionary containing the example details,
                            or an error message if the example cannot be retrieved
        """
        try:
            return read_example_tool(
                client,
                example_id=example_id,
                as_of=as_of,
            )
        except Exception as e:
            return {"error": str(e)}
