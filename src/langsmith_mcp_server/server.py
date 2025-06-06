#!/usr/bin/env python3
"""
MCP server for LangSmith SDK integration.
This server exposes methods to interact with LangSmith's observability platform:
- get_thread_history: Fetch conversation history for a specific thread
- get_prompts: Fetch prompts from LangSmith with optional filtering
- pull_prompt: Pull a specific prompt by its name
"""

import os
from typing import Any, Dict, List, Optional
import re

from langsmith import Client
from mcp.server.fastmcp import FastMCP


class LangSmithClient:
    """Client for interacting with the LangSmith API."""

    def __init__(self, api_key: str):
        """
        Initialize the LangSmith API client.

        Args:
            api_key: API key for LangSmith API
        """
        self.api_key = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key
        self.langsmith_client = Client()

    def get_conversation_thread(self, thread_id: str, project_name: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a specific thread.

        Args:
            thread_id: The ID of the thread to fetch history for
            project_name: The name of the project containing the thread

        Returns:
            List of messages in the conversation history
        """
        try:
            # Filter runs by the specific thread and project
            filter_string = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))'
            
            # Only grab the LLM runs
            runs = [r for r in self.langsmith_client.list_runs(
                project_name=project_name,
                filter=filter_string,
                run_type="llm"
            )]

            if not runs:
                return [{"error": f"No runs found for thread {thread_id} in project {project_name}"}]

            # Sort by start time to get the most recent interaction
            runs = sorted(runs, key=lambda run: run.start_time, reverse=True)
            
            # Get the most recent run
            latest_run = runs[0]
            
            # Extract messages from inputs and outputs
            messages = []
            
            # Add input messages if they exist
            if hasattr(latest_run, 'inputs') and 'messages' in latest_run.inputs:
                messages.extend(latest_run.inputs['messages'])
            
            # Add output message if it exists
            if hasattr(latest_run, 'outputs'):
                if isinstance(latest_run.outputs, dict) and 'choices' in latest_run.outputs:
                    if isinstance(latest_run.outputs['choices'], list) and len(latest_run.outputs['choices']) > 0:
                        if 'message' in latest_run.outputs['choices'][0]:
                            messages.append(latest_run.outputs['choices'][0]['message'])
                elif isinstance(latest_run.outputs, dict) and 'message' in latest_run.outputs:
                    messages.append(latest_run.outputs['message'])
            
            if not messages:
                return [{"error": f"No messages found in the run for thread {thread_id}"}]
                
            return messages
            
        except Exception as e:
            return [{"error": f"Error fetching thread history: {str(e)}"}]

    def list_prompts(self, is_public: bool = False, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch prompts from LangSmith with optional filtering.

        Args:
            is_public (bool): Optional boolean to filter public/private prompts
            limit (int): Optional limit to the number of prompts to return
        Returns:
            Dictionary containing the prompts and metadata
        """
        try:
            prompts = self.langsmith_client.list_prompts(is_public=is_public, limit=limit)
            
            # Convert prompts to a serializable format
            result = {
                "prompts": [
                    {
                        "repo_handle": prompt.repo_handle,
                        "description": prompt.description,
                        "id": prompt.id,
                        "is_public": prompt.is_public,
                        "is_archived": prompt.is_archived,
                        "tags": prompt.tags,
                        "owner": prompt.owner,
                        "full_name": prompt.full_name,
                        "num_likes": prompt.num_likes,
                        "num_downloads": prompt.num_downloads,
                        "num_views": prompt.num_views,
                        "created_at": prompt.created_at.isoformat() if prompt.created_at else None,
                        "updated_at": prompt.updated_at.isoformat() if prompt.updated_at else None
                    }
                    for prompt in prompts.repos
                ],
                "total": prompts.total
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error fetching prompts: {str(e)}"}

    def get_prompt(self, prompt_name: Optional[str] = None, prompt_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific prompt by its name or id. Use always full name if available.

        Args:
            prompt_name: The name of the prompt to get
            prompt_id: The id of the prompt to get
        Returns:
            Dictionary containing the prompt details and template
        """
        try:
            if prompt_name is not None:
                prompt = self.langsmith_client.pull_prompt(prompt_name)
            elif prompt_id is not None:
                prompt = self.langsmith_client.get_prompt(prompt_id)
            else:
                raise ValueError("Either prompt_name or prompt_id must be provided")
            
            # Convert prompt to a serializable format
            result = {
                "input_variables": prompt.input_variables,
                "input_types": prompt.input_types,
                "partial_variables": prompt.partial_variables,
                "metadata": prompt.metadata,
                "messages": [
                    {
                        "type": message.__class__.__name__,
                        "template": message.prompt.template,
                        "input_variables": message.prompt.input_variables,
                        "input_types": message.prompt.input_types,
                        "partial_variables": message.prompt.partial_variables,
                        "additional_kwargs": message.additional_kwargs
                    }
                    for message in prompt.messages
                ]
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error pulling prompt: {str(e)}"}

    def get_langgraph_app_host_name(self, run_stats: dict) -> str | None:
        """
        Get the langgraph app host name from the run stats

        Args:
            run_stats (dict): The run stats

        Returns:
            str | None: The langgraph app host name
        """
        for element in run_stats.get("run_facets", []):
            if "langgraph.app" in element.get("value", "") and "x-forwarded-host" in element.get("value", ""):
                matches = re.findall(r'"(.*?)"', element["value"])
                if matches:
                    return matches[0]
        return None

    def get_last_run_stats(self, project_name: str) -> dict | None:
        """
        Get the last run stats for a given project name

        Args:
            project_name (str): The name of the project to get the last run stats for

        Returns:
            dict: The last run stats for the given project name
        """
        client = self.langsmith_client
        project_runs_stats = client.get_run_stats(
            project_names=[project_name],
            is_root=True
        )
        start_time = project_runs_stats.get("last_run_start_time", None)
        if start_time is None:
            return None

        # get the last run stats
        last_run_stats = client.get_run_stats(
            project_names=[project_name],
            start_time=start_time,
            is_root=True
        )

        langgraph_app_host_name = self.get_langgraph_app_host_name(last_run_stats)

        # remove the run_facets from the last_run_stats
        last_run_stats.pop("run_facets", None)
        last_run_stats["langgraph_app_host_name"] = langgraph_app_host_name

        return last_run_stats

    def get_project_runs_stats(self, project_name: str, is_last_run: bool = True) -> dict | None:
        """
        Get the project runs stats
        """
        client = self.langsmith_client
        projects = client.list_projects(
            name_contains=project_name,
            limit=1,
            reference_free=True
        )
        actual_project_name = None
        for project in projects:
            actual_project_name = project.name
            break
        if actual_project_name is None:
            return None
        
        if is_last_run:
            project_last_run_stats = self.get_last_run_stats(actual_project_name)
            if project_last_run_stats is not None:
                project_last_run_stats["project_name"] = actual_project_name
            return project_last_run_stats
        else:
            project_runs_stats = client.get_run_stats(
                project_names=[actual_project_name]
            )
            # remove the run_facets from the project_runs_stats
            project_runs_stats.pop("run_facets", None)
            # add project_name to the project_runs_stats
            project_runs_stats["project_name"] = actual_project_name
            return project_runs_stats


# Create MCP server
mcp = FastMCP("LangSmith API MCP Server")

# Default API key (will be overridden in main or by direct assignment)
default_api_key = os.environ.get("LANGSMITH_API_KEY")
langsmith_client = LangSmithClient(default_api_key) if default_api_key else None


# Add tool for get_conversation_thread
@mcp.tool()
def get_conversation_thread(thread_id: str, project_name: str) -> List[Dict[str, Any]]:
    """
    Get the conversation history for a specific thread.

    Args:
        thread_id: The ID of the thread to fetch history for
        project_name: The name of the project containing the thread

    Returns:
        List of messages in the conversation history
    """
    if langsmith_client is None:
        return [{"error": "LangSmith client not initialized. Please provide an API key."}]

    try:
        return langsmith_client.get_conversation_thread(thread_id, project_name)
    except Exception as e:
        return [{"error": str(e)}]


# Add tool for list_prompts
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
    if langsmith_client is None:
        return {"error": "LangSmith client not initialized. Please provide an API key."}

    try:
        is_public_bool = is_public.lower() == "true"
        return langsmith_client.list_prompts(is_public_bool, limit)
    except Exception as e:
        return {"error": str(e)}


# Add tool for get_prompt
@mcp.tool()
def get_prompt_by_id(prompt_id: str) -> Dict[str, Any]:
    """
    Get a specific prompt by its id.

    Args:
        prompt_id: The id of the prompt to get

    Returns:
        Dictionary containing the prompt details and template
    """
    if langsmith_client is None:
        return {"error": "LangSmith client not initialized. Please provide an API key."}

    try:
        return langsmith_client.get_prompt(prompt_id)
    except Exception as e:
        return {"error": str(e)}


# Add tool for get_prompt_by_name
@mcp.tool()
def get_prompt_by_name(prompt_name: str) -> Dict[str, Any]:
    """
    Get a specific prompt by its name.

    Args:
        prompt_name: The name of the prompt to get

    Returns:
        Dictionary containing the prompt details and template
    """
    if langsmith_client is None:
        return {"error": "LangSmith client not initialized. Please provide an API key."}

    try:
        return langsmith_client.get_prompt(prompt_name)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_project_runs_stats(project_name: str, is_last_run: str = "true") -> dict | None:
    """
    Get the project runs stats
    Args:
        project_name (str): The name of the project
        is_last_run (str): "true" to get last run stats, "false" for overall project stats
    Returns:
        dict | None: The project runs stats
    """
    if langsmith_client is None:
        return {"error": "LangSmith client not initialized. Please provide an API key."}
    try:
        is_last_run_bool = is_last_run.lower() == "true"
        return langsmith_client.get_project_runs_stats(project_name, is_last_run_bool)
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    """Run the LangSmith MCP server."""
    print("Starting LangSmith MCP server!")
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main() 