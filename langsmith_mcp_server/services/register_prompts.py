"""Registration module for LangSmith MCP prompts."""


def register_prompts(mcp, langsmith_client):
    """Register all prompt-related functionality with the MCP server."""

    # Skip registration if client is not initialized
    if langsmith_client is None:
        return
