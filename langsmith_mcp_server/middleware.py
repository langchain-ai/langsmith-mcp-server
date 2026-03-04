"""Simple API key authentication middleware for MCP HTTP."""

from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_405_METHOD_NOT_ALLOWED

# Context variables to store LangSmith config for current request
# These are used to pass config from middleware to FastMCP's session state
api_key_context: ContextVar[str] = ContextVar("api_key", default="")
workspace_id_context: ContextVar[str] = ContextVar("workspace_id", default="")
endpoint_context: ContextVar[str] = ContextVar("endpoint", default="")


def get_api_key() -> str:
    """Get the API key from the current request context."""
    return api_key_context.get("")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts LANGSMITH-API-KEY header and optional config headers.

    FastMCP will handle session management automatically. We extract:
    1. LANGSMITH-API-KEY (required)
    2. LANGSMITH-WORKSPACE-ID (optional)
    3. LANGSMITH-ENDPOINT (optional)
    """  # noqa: W293

    # MCP endpoint only accepts POST (and DELETE for session teardown); GET returns 405
    _MCP_PATH = "/mcp"
    _MCP_ALLOWED_METHODS = frozenset({"POST", "DELETE"})

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for health check and OAuth discovery (we use API key, not OAuth)
        if request.url.path in ("/health", "/.well-known/oauth-authorization-server"):
            return await call_next(request)

        # GET (e.g. opening /mcp in a browser) → 405 Method Not Allowed (standard for MCP HTTP)
        path = request.url.path.rstrip("/") or "/"
        if path == self._MCP_PATH and request.method not in self._MCP_ALLOWED_METHODS:
            return JSONResponse(
                status_code=HTTP_405_METHOD_NOT_ALLOWED,
                content={"error": "Method Not Allowed", "allow": list(self._MCP_ALLOWED_METHODS)},
                headers={"Allow": ", ".join(sorted(self._MCP_ALLOWED_METHODS))},
            )

        # Require LANGSMITH-API-KEY header
        api_key = request.headers.get("LANGSMITH-API-KEY")
        if not api_key:
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={"error": "Missing LANGSMITH-API-KEY header"},
            )

        # Get optional headers
        workspace_id = request.headers.get("LANGSMITH-WORKSPACE-ID", "")
        endpoint = request.headers.get("LANGSMITH-ENDPOINT", "")

        # Store in request state and context variables
        # FastMCP will handle storing this in session state via the Context
        request.state.api_key = api_key
        request.state.workspace_id = workspace_id
        request.state.endpoint = endpoint

        api_key_context.set(api_key)
        workspace_id_context.set(workspace_id)
        endpoint_context.set(endpoint)

        try:
            return await call_next(request)
        finally:
            # Clear context after request
            api_key_context.set("")
            workspace_id_context.set("")
            endpoint_context.set("")
