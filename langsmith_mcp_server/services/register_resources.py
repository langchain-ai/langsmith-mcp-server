"""Registration module for LangSmith MCP resources."""


def register_resources(mcp, langsmith_client):
    """Register all resource-related functionality with the MCP server."""

    # Skip registration if client is not initialized
    if langsmith_client is None:
        return
