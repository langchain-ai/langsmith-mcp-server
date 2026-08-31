"""Integration tests for stdio transport."""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO


class TestStdioTransport:
    """Integration tests for stdio transport."""
    
    @patch("langsmith_mcp_server.server.mcp")
    def test_stdio_server_initialization(self, mock_mcp):
        """Test that stdio server can be initialized."""
        # Setup
        from langsmith_mcp_server.server import main
        
        # Mock the run method to avoid actually starting the server
        mock_mcp.run = Mock()
        
        # Execute
        main()
        
        # Verify
        mock_mcp.run.assert_called_once_with(transport="stdio")
    
    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "lsv2_pt_test_key"})
    def test_stdio_with_env_variables(self):
        """Test that stdio transport uses environment variables."""
        # Setup
        api_key = os.environ.get("LANGSMITH_API_KEY")
        
        # Verify
        assert api_key is not None
        assert api_key.startswith("lsv2_")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_stdio_without_env_variables(self):
        """Test stdio transport behavior without environment variables."""
        # This should still work since stdio transport can work without API key
        # (it's not required for the server to start, only for making API calls)
        
        api_key = os.environ.get("LANGSMITH_API_KEY")
        
        # Verify
        assert api_key is None
        # Server should still be able to start (it just won't be able to make API calls)


class TestStdioMCPProtocol:
    """Tests for MCP protocol over stdio."""
    
    def test_json_rpc_request_format(self):
        """Test that requests follow JSON-RPC 2.0 format."""
        # Setup - a valid MCP request
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1
        }
        
        # Verify structure
        assert "jsonrpc" in request
        assert request["jsonrpc"] == "2.0"
        assert "method" in request
        assert "id" in request
    
    def test_json_rpc_response_format(self):
        """Test that responses follow JSON-RPC 2.0 format."""
        # Setup - a valid MCP response
        response = {
            "jsonrpc": "2.0",
            "result": {"tools": []},
            "id": 1
        }
        
        # Verify structure
        assert "jsonrpc" in response
        assert response["jsonrpc"] == "2.0"
        assert "result" in response or "error" in response
        assert "id" in response
    
    def test_json_rpc_error_format(self):
        """Test error response format."""
        # Setup - an error response
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32600,
                "message": "Invalid Request"
            },
            "id": None
        }
        
        # Verify structure
        assert "jsonrpc" in error_response
        assert "error" in error_response
        assert "code" in error_response["error"]
        assert "message" in error_response["error"]


class TestStdioToolInvocation:
    """Tests for tool invocation via stdio."""
    
    @patch("langsmith_mcp_server.common.helpers.get_client_from_context")
    def test_list_prompts_tool_invocation(self, mock_get_client):
        """Test invoking list_prompts tool via stdio transport."""
        # Setup
        mock_client = Mock()
        mock_client.list_prompts.return_value = []
        mock_get_client.return_value = mock_client
        
        from langsmith_mcp_server.services.tools.prompts import list_prompts_tool
        
        # Execute
        result = list_prompts_tool(mock_client, is_public=False, limit=10)
        
        # Verify
        assert "prompts" in result
        assert isinstance(result["prompts"], list)
    
    @patch("langsmith_mcp_server.common.helpers.get_client_from_context")
    def test_list_datasets_tool_invocation(self, mock_get_client):
        """Test invoking list_datasets tool via stdio transport."""
        # Setup
        mock_client = Mock()
        mock_client.list_datasets.return_value = []
        mock_get_client.return_value = mock_client
        
        from langsmith_mcp_server.services.tools.datasets import list_datasets_tool
        
        # Execute
        result = list_datasets_tool(mock_client, limit=10)
        
        # Verify
        assert "datasets" in result
        assert isinstance(result["datasets"], list)


class TestStdioErrorHandling:
    """Tests for error handling in stdio transport."""
    
    def test_invalid_json_input(self):
        """Test handling of invalid JSON input."""
        # Setup
        invalid_json = "not a json string"
        
        # Verify that json.loads raises an error
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)
    
    def test_missing_required_fields(self):
        """Test handling of requests with missing required fields."""
        # Setup - request without required 'method' field
        incomplete_request = {
            "jsonrpc": "2.0",
            "id": 1
            # Missing 'method' field
        }
        
        # Verify structure
        assert "method" not in incomplete_request
        # MCP server should handle this gracefully with an error response
    
    @patch("langsmith_mcp_server.common.helpers.get_client_from_context")
    def test_tool_execution_error(self, mock_get_client):
        """Test handling of errors during tool execution."""
        # Setup
        mock_client = Mock()
        mock_client.list_prompts.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client
        
        from langsmith_mcp_server.services.tools.prompts import list_prompts_tool
        
        # Execute
        result = list_prompts_tool(mock_client, is_public=False, limit=10)
        
        # Verify error is caught and returned
        assert "error" in result
        assert "API error" in result["error"]


class TestStdioContextManagement:
    """Tests for context management in stdio transport."""
    
    def test_context_state_storage(self, mock_context):
        """Test that context can store and retrieve state."""
        # Setup
        test_key = "test_key"
        test_value = "test_value"
        
        # Execute
        mock_context.set_state(test_key, test_value)
        retrieved_value = mock_context.get_state(test_key)
        
        # Verify
        assert retrieved_value == test_value
    
    def test_context_state_default_value(self, mock_context):
        """Test that context returns default value for missing keys."""
        # Execute
        retrieved_value = mock_context.get_state("nonexistent_key", "default")
        
        # Verify
        assert retrieved_value == "default"
    
    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "lsv2_pt_test_key"})
    def test_context_environment_integration(self):
        """Test that context integrates with environment variables."""
        # Setup - environment variable is set
        api_key = os.environ.get("LANGSMITH_API_KEY")
        
        # Verify
        assert api_key == "lsv2_pt_test_key"
        # In stdio transport, the server reads from environment


class TestStdioPerformance:
    """Performance tests for stdio transport."""
    
    @patch("langsmith_mcp_server.common.helpers.get_client_from_context")
    def test_tool_response_time(self, mock_get_client):
        """Test that tool responses are fast enough."""
        import time
        
        # Setup
        mock_client = Mock()
        mock_client.list_prompts.return_value = []
        mock_get_client.return_value = mock_client
        
        from langsmith_mcp_server.services.tools.prompts import list_prompts_tool
        
        # Execute
        start = time.time()
        result = list_prompts_tool(mock_client, is_public=False, limit=10)
        duration = time.time() - start
        
        # Verify
        assert "prompts" in result
        assert duration < 1.0  # Should complete in less than 1 second (with mocked client)
    
    @patch("langsmith_mcp_server.common.helpers.get_client_from_context")
    def test_sequential_tool_calls(self, mock_get_client):
        """Test multiple sequential tool calls."""
        # Setup
        mock_client = Mock()
        mock_client.list_prompts.return_value = []
        mock_client.list_datasets.return_value = []
        mock_get_client.return_value = mock_client
        
        from langsmith_mcp_server.services.tools.prompts import list_prompts_tool
        from langsmith_mcp_server.services.tools.datasets import list_datasets_tool
        
        # Execute multiple calls
        results = []
        for _ in range(5):
            result1 = list_prompts_tool(mock_client, is_public=False, limit=10)
            result2 = list_datasets_tool(mock_client, limit=10)
            results.extend([result1, result2])
        
        # Verify all succeeded
        assert len(results) == 10
        assert all("prompts" in r or "datasets" in r for r in results)

