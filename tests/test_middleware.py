"""Tests for middleware components."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED
from langsmith_mcp_server.middleware import APIKeyMiddleware


class TestAPIKeyMiddleware:
    """Tests for APIKeyMiddleware."""
    
    @pytest.mark.asyncio
    async def test_health_check_bypasses_auth(self):
        """Test that health check endpoint bypasses authentication."""
        # Setup
        middleware = APIKeyMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/health"
        request.headers = {}
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        # Execute
        response = await middleware.dispatch(request, call_next)
        
        # Verify
        assert call_next.called
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_missing_api_key_returns_401(self):
        """Test that missing API key returns 401."""
        # Setup
        middleware = APIKeyMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/mcp"
        request.headers = {}
        
        def get_header(key, default=None):
            return request.headers.get(key, default)
        
        request.headers.get = get_header
        
        call_next = AsyncMock()
        
        # Execute
        response = await middleware.dispatch(request, call_next)
        
        # Verify
        assert isinstance(response, JSONResponse)
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert not call_next.called
    
    @pytest.mark.asyncio
    async def test_valid_api_key_sets_state(self, sample_api_key):
        """Test that valid API key sets request state."""
        # Setup
        middleware = APIKeyMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/mcp"
        request.headers = {"LANGSMITH-API-KEY": sample_api_key}
        request.state = Mock()
        
        def get_header(key, default=None):
            return request.headers.get(key, default)
        
        request.headers.get = get_header
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        # Execute
        response = await middleware.dispatch(request, call_next)
        
        # Verify
        assert call_next.called
        assert request.state.api_key == sample_api_key
        assert hasattr(request.state, "workspace_id")
        assert hasattr(request.state, "endpoint")
    
    @pytest.mark.asyncio
    async def test_optional_headers_extracted(self, sample_api_key, sample_workspace_id):
        """Test that optional headers are properly extracted."""
        # Setup
        middleware = APIKeyMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/mcp"
        request.headers = {
            "LANGSMITH-API-KEY": sample_api_key,
            "LANGSMITH-WORKSPACE-ID": sample_workspace_id,
            "LANGSMITH-ENDPOINT": "https://custom.api.com"
        }
        request.state = Mock()
        
        def get_header(key, default=None):
            return request.headers.get(key, default)
        
        request.headers.get = get_header
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        # Execute
        response = await middleware.dispatch(request, call_next)
        
        # Verify
        assert request.state.api_key == sample_api_key
        assert request.state.workspace_id == sample_workspace_id
        assert request.state.endpoint == "https://custom.api.com"
    
    @pytest.mark.asyncio
    async def test_context_variables_set_and_cleared(self, sample_api_key):
        """Test that context variables are set during request and cleared after."""
        # Setup
        middleware = APIKeyMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/mcp"
        request.headers = {"LANGSMITH-API-KEY": sample_api_key}
        request.state = Mock()
        
        def get_header(key, default=None):
            return request.headers.get(key, default)
        
        request.headers.get = get_header
        
        # Capture context variable values during call_next
        captured_api_key = None
        
        async def call_next_with_capture(req):
            nonlocal captured_api_key
            from langsmith_mcp_server.middleware import api_key_context
            captured_api_key = api_key_context.get("")
            return JSONResponse({"status": "ok"})
        
        # Execute
        response = await middleware.dispatch(request, call_next_with_capture)
        
        # Verify
        # During request, context variable should be set
        assert captured_api_key == sample_api_key
        
        # After request, context variables should be cleared
        from langsmith_mcp_server.middleware import api_key_context
        assert api_key_context.get("") == ""

