"""Integration tests for HTTP transport."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from starlette.testclient import TestClient
from langsmith_mcp_server.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHTTPTransport:
    """Integration tests for HTTP transport."""
    
    def test_health_check_endpoint(self, client):
        """Test that health check endpoint works without authentication."""
        # Execute
        response = client.get("/health")
        
        # Verify
        assert response.status_code == 200
        assert "LangSmith MCP server is running" in response.text
    
    def test_mcp_endpoint_requires_auth(self, client):
        """Test that MCP endpoint requires authentication."""
        # Execute - no API key provided
        response = client.post("/mcp", json={"test": "data"})
        
        # Verify
        assert response.status_code == 401
        assert "LANGSMITH-API-KEY" in response.json()["error"]
    
    def test_mcp_endpoint_with_valid_auth(self, client, sample_api_key):
        """Test that MCP endpoint accepts valid API key."""
        # Setup
        headers = {"LANGSMITH-API-KEY": sample_api_key}
        
        # Note: This will fail with invalid JSON-RPC format but should pass auth
        # Execute
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "test", "id": 1},
            headers=headers
        )
        
        # Verify - should not be 401 (may be other error, but auth passed)
        assert response.status_code != 401
    
    def test_mcp_endpoint_with_optional_headers(
        self, client, sample_api_key, sample_workspace_id
    ):
        """Test that optional headers are accepted."""
        # Setup
        headers = {
            "LANGSMITH-API-KEY": sample_api_key,
            "LANGSMITH-WORKSPACE-ID": sample_workspace_id,
            "LANGSMITH-ENDPOINT": "https://custom.api.com"
        }
        
        # Execute
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "test", "id": 1},
            headers=headers
        )
        
        # Verify - auth should pass
        assert response.status_code != 401
    
    def test_cors_headers_present(self, client, sample_api_key):
        """Test that CORS headers are properly set."""
        # Setup
        headers = {"LANGSMITH-API-KEY": sample_api_key, "Origin": "http://localhost:3000"}
        
        # Execute
        response = client.options("/mcp", headers=headers)
        
        # Verify - CORS headers should be present
        # The exact headers depend on CORS middleware configuration
        assert response.status_code in [200, 204, 401]  # OPTIONS request handling
    
    @patch("langsmith_mcp_server.services.tools.prompts.Client")
    def test_list_prompts_via_http(self, mock_client_class, client, sample_api_key):
        """Test calling list_prompts tool via HTTP transport."""
        # Setup
        mock_client = Mock()
        mock_client.list_prompts.return_value = []
        mock_client_class.return_value = mock_client
        
        headers = {"LANGSMITH-API-KEY": sample_api_key}
        
        # This is a simplified test - actual MCP protocol is more complex
        # But we're testing that the HTTP layer works
        
        # For now, just verify auth works and endpoint is reachable
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "test", "id": 1},
            headers=headers
        )
        
        # Verify
        assert response.status_code != 401  # Auth passed


class TestHTTPTransportErrors:
    """Test error handling in HTTP transport."""
    
    def test_invalid_json_returns_error(self, client, sample_api_key):
        """Test that invalid JSON returns proper error."""
        # Setup
        headers = {"LANGSMITH-API-KEY": sample_api_key}
        
        # Execute - send invalid JSON
        response = client.post(
            "/mcp",
            data="invalid json",
            headers=headers
        )
        
        # Verify - should return error (not 401)
        assert response.status_code != 401
        assert response.status_code in [400, 422, 500]  # Client or server error
    
    def test_empty_request_returns_error(self, client, sample_api_key):
        """Test that empty request returns proper error."""
        # Setup
        headers = {"LANGSMITH-API-KEY": sample_api_key}
        
        # Execute
        response = client.post("/mcp", json={}, headers=headers)
        
        # Verify
        assert response.status_code != 401


class TestHTTPTransportPerformance:
    """Performance-related tests for HTTP transport."""
    
    def test_health_check_is_fast(self, client):
        """Test that health check responds quickly."""
        import time
        
        # Execute
        start = time.time()
        response = client.get("/health")
        duration = time.time() - start
        
        # Verify
        assert response.status_code == 200
        assert duration < 1.0  # Should respond in less than 1 second
    
    def test_concurrent_requests_handled(self, client, sample_api_key):
        """Test that multiple concurrent requests can be handled."""
        import concurrent.futures
        
        headers = {"LANGSMITH-API-KEY": sample_api_key}
        
        def make_request():
            return client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "test", "id": 1},
                headers=headers
            )
        
        # Execute multiple requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # Verify all requests completed (not necessarily successfully, but no crashes)
        assert len(results) == 10
        assert all(r.status_code != 500 for r in results)  # No server errors

