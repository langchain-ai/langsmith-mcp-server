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
        os.environ["LANGCHAIN_API_KEY"] = api_key
        self.langsmith_client = Client()

    def get_thread_history(self, thread_id: str, project_name: str) -> List[Dict[str, Any]]:
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

    def get_prompts(self, query: Optional[str] = None, is_public: Optional[bool] = None) -> Dict[str, Any]:
        """
        Fetch prompts from LangSmith with optional filtering.

        Args:
            query: Optional search query to filter prompts
            is_public: Optional boolean to filter public/private prompts

        Returns:
            Dictionary containing the prompts and metadata
        """
        try:
            prompts = self.langsmith_client.list_prompts(query=query, is_public=is_public)
            
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

    def pull_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """
        Pull a specific prompt by its name. Use always full name if available.

        Args:
            prompt_name: The name of the prompt to pull

        Returns:
            Dictionary containing the prompt details and template
        """
        try:
            prompt = self.langsmith_client.pull_prompt(prompt_name)
            
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


# Create MCP server
mcp = FastMCP("LangSmith API MCP Server")

# Default API key (will be overridden in main or by direct assignment)
default_api_key = os.environ.get("LANGCHAIN_API_KEY")
langsmith_client = LangSmithClient(default_api_key) if default_api_key else None


# Add tool for get_thread_history
@mcp.tool()
def get_thread_history(thread_id: str, project_name: str) -> List[Dict[str, Any]]:
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
        return langsmith_client.get_thread_history(thread_id, project_name)
    except Exception as e:
        return [{"error": str(e)}]


# Add tool for get_prompts
@mcp.tool()
def get_prompts(query: Optional[str] = None, is_public: Optional[bool] = None) -> Dict[str, Any]:
    """
    Fetch prompts from LangSmith with optional filtering.

    Args:
        query: Optional search query to filter prompts
        is_public: Optional boolean to filter public/private prompts

    Returns:
        Dictionary containing the prompts and metadata
    """
    if langsmith_client is None:
        return {"error": "LangSmith client not initialized. Please provide an API key."}

    try:
        return langsmith_client.get_prompts(query, is_public)
    except Exception as e:
        return {"error": str(e)}


# Add tool for pull_prompt
@mcp.tool()
def pull_prompt(prompt_name: str) -> Dict[str, Any]:
    """
    Pull a specific prompt by its name.

    Args:
        prompt_name: The name of the prompt to pull

    Returns:
        Dictionary containing the prompt details and template
    """
    if langsmith_client is None:
        return {"error": "LangSmith client not initialized. Please provide an API key."}

    try:
        return langsmith_client.pull_prompt(prompt_name)
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    """Run the LangSmith MCP server."""
    print("Starting LangSmith MCP server!")
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main() 