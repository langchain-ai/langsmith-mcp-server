"""
MCP server for LangSmith SDK integration.
This server exposes methods to interact with LangSmith's observability platform:
- get_thread_history: Fetch conversation history for a specific thread
- get_prompts: Fetch prompts from LangSmith with optional filtering
- pull_prompt: Pull a specific prompt by its name
"""

import logging

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from langsmith_mcp_server.middleware import APIKeyMiddleware
from langsmith_mcp_server.services import (
    register_prompts,
    register_resources,
    register_tools,
)
from langsmith_mcp_server.streamable_http_patch import (
    apply_streamable_http_client_disconnect_patch,
)

logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("LangSmith API MCP Server")
logger.info("MCP server instance created: LangSmith API MCP Server")

# Register all tools with the server using simplified registration modules
# Note: Tools will use API key from request.state.api_key (set by middleware)
register_tools(mcp)
register_prompts(mcp)
register_resources(mcp)
logger.info("Tools, prompts, and resources registered")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    logger.debug(
        "Health check requested from %s", request.client.host if request.client else "unknown"
    )
    return PlainTextResponse("LangSmith MCP server is running")


# Define middleware - API key middleware must be first
middleware = [
    Middleware(APIKeyMiddleware),
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id"],
    ),
]

# Create ASGI application
app = mcp.http_app(middleware=middleware)
logger.info("ASGI app created with API key and CORS middleware")

# Handle client disconnect gracefully (avoids ClosedResourceError/BrokenResourceError
# when client or proxy closes the connection before the server finishes).
apply_streamable_http_client_disconnect_patch()
logger.info("Streamable HTTP client-disconnect patch applied")


def main() -> None:
    """Run the LangSmith MCP server."""
    logger.info("Starting LangSmith MCP server (stdio transport)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
