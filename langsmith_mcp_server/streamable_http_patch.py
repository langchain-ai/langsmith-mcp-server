"""
Patch for MCP streamable HTTP transport to handle client disconnect gracefully.

When the client (or a proxy/load balancer) closes the connection before the server
finishes, writer.send() can raise anyio.ClosedResourceError or
anyio.BrokenResourceError. The MCP SDK does not catch these in _handle_post_request,
leading to ExceptionGroup and "SSE response error" crashes.

This module patches StreamableHTTPServerTransport._handle_post_request to catch
these errors and log instead of re-raising. See:
- https://github.com/modelcontextprotocol/python-sdk/issues/2064
- https://github.com/modelcontextprotocol/python-sdk/pull/2072
"""

import builtins
import logging
from typing import Any

import anyio

# BaseExceptionGroup exists in Python 3.11+; use a no-op type on 3.10 so code is uniform
BaseExceptionGroup = getattr(builtins, "BaseExceptionGroup", type("BaseExceptionGroup", (), {}))

logger = logging.getLogger(__name__)

# Types for connection-lifecycle errors we treat as "client disconnected"
_CLIENT_DISCONNECT_EXCEPTIONS = (anyio.ClosedResourceError, anyio.BrokenResourceError)


def _is_client_disconnect(exc: BaseException) -> bool:
    """True if the exception indicates the client closed the connection."""
    if isinstance(exc, _CLIENT_DISCONNECT_EXCEPTIONS):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return all(_is_client_disconnect(e) for e in exc.exceptions)
    return False


def apply_streamable_http_client_disconnect_patch() -> None:
    """
    Patch MCP StreamableHTTPServerTransport so that client disconnects during
    POST handling do not crash the server with ClosedResourceError /
    BrokenResourceError.

    Safe to call multiple times; applies the patch only once.
    """
    try:
        from mcp.server import streamable_http
    except ImportError:
        logger.debug("mcp.server.streamable_http not available; skipping client-disconnect patch")
        return

    transport_class = streamable_http.StreamableHTTPServerTransport
    original = transport_class._handle_post_request

    # Already patched (idempotent)
    if getattr(original, "_client_disconnect_patched", False):
        return

    async def patched_handle_post_request(
        self: Any, scope: Any, request: Any, receive: Any, send: Any
    ) -> None:
        try:
            await original(self, scope, request, receive, send)
        except _CLIENT_DISCONNECT_EXCEPTIONS as e:
            logger.debug(
                "Client disconnected during POST request: %s",
                type(e).__name__,
                exc_info=True,
            )
        except BaseExceptionGroup as eg:
            if _is_client_disconnect(eg):
                logger.debug(
                    "Client disconnected during POST request (ExceptionGroup)",
                    exc_info=True,
                )
            else:
                raise

    patched_handle_post_request._client_disconnect_patched = True  # type: ignore[attr-defined]
    transport_class._handle_post_request = patched_handle_post_request
    logger.debug("Applied streamable HTTP client-disconnect patch")
