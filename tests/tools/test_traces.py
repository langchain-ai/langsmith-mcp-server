"""Tests for trace tools."""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4
from langsmith_mcp_server.services.tools.traces import (
    fetch_runs_tool,
    list_projects_tool,
    fetch_trace_tool,
    get_thread_history_tool,
    get_project_runs_stats_tool,
)


class TestFetchRunsTool:
    """Tests for fetch_runs_tool."""
    
    def test_fetch_runs_success(self, mock_langsmith_client, mock_run):
        """Test successful fetch of runs."""
        # Setup
        mock_langsmith_client.list_runs.return_value = [mock_run]
        
        # Execute
        result = fetch_runs_tool(
            mock_langsmith_client,
            project_name="test-project",
            limit=10
        )
        
        # Verify
        assert "runs" in result
        assert len(result["runs"]) == 1
        mock_langsmith_client.list_runs.assert_called_once()
    
    def test_fetch_runs_with_filters(self, mock_langsmith_client, mock_run):
        """Test fetch runs with various filters."""
        # Setup
        mock_langsmith_client.list_runs.return_value = [mock_run]
        
        # Execute
        result = fetch_runs_tool(
            mock_langsmith_client,
            project_name="test-project",
            run_type="chain",
            error=False,
            is_root=True,
            limit=5
        )
        
        # Verify
        assert "runs" in result
        mock_langsmith_client.list_runs.assert_called_once()
    
    def test_fetch_runs_with_trace_id(self, mock_langsmith_client, mock_run):
        """Test fetch runs with specific trace ID."""
        # Setup
        mock_langsmith_client.list_runs.return_value = [mock_run]
        trace_id = str(uuid4())
        
        # Execute
        result = fetch_runs_tool(
            mock_langsmith_client,
            project_name="test-project",
            trace_id=trace_id,
            limit=10
        )
        
        # Verify
        assert "runs" in result
        mock_langsmith_client.list_runs.assert_called_once()
    
    def test_fetch_runs_formatted_output(self, mock_langsmith_client, mock_run):
        """Test fetch runs with formatted output."""
        # Setup
        mock_run.inputs = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_run.outputs = {"messages": [{"role": "assistant", "content": "Hi"}]}
        mock_langsmith_client.list_runs.return_value = [mock_run]
        
        # Execute
        result = fetch_runs_tool(
            mock_langsmith_client,
            project_name="test-project",
            limit=10,
            format_type="pretty"
        )
        
        # Verify
        assert "formatted" in result
        assert isinstance(result["formatted"], str)


class TestListProjectsTool:
    """Tests for list_projects_tool."""
    
    def test_list_projects_success(self, mock_langsmith_client, mock_project):
        """Test successful list of projects."""
        # Setup
        mock_langsmith_client.list_projects.return_value = [mock_project]
        
        # Execute
        result = list_projects_tool(mock_langsmith_client, limit=5)
        
        # Verify
        assert "projects" in result
        assert len(result["projects"]) == 1
        assert result["projects"][0]["name"] == "test-project"
    
    def test_list_projects_with_name_filter(self, mock_langsmith_client, mock_project):
        """Test list projects with name filter."""
        # Setup
        mock_langsmith_client.list_projects.return_value = [mock_project]
        
        # Execute
        result = list_projects_tool(
            mock_langsmith_client,
            limit=10,
            project_name="test"
        )
        
        # Verify
        assert "projects" in result
        assert len(result["projects"]) == 1
    
    def test_list_projects_more_info(self, mock_langsmith_client, mock_project):
        """Test list projects with more_info flag."""
        # Setup
        mock_langsmith_client.list_projects.return_value = [mock_project]
        
        # Execute
        result = list_projects_tool(
            mock_langsmith_client,
            limit=5,
            more_info=True
        )
        
        # Verify
        assert "projects" in result
        assert len(result["projects"]) == 1
        # With more_info, we get full project dict
        assert "id" in result["projects"][0]
    
    def test_list_projects_empty(self, mock_langsmith_client):
        """Test list projects with no results."""
        # Setup
        mock_langsmith_client.list_projects.return_value = []
        
        # Execute
        result = list_projects_tool(mock_langsmith_client, limit=5)
        
        # Verify
        assert "projects" in result
        assert len(result["projects"]) == 0


class TestFetchTraceTool:
    """Tests for fetch_trace_tool."""
    
    def test_fetch_trace_by_id(self, mock_langsmith_client, mock_run):
        """Test fetch trace by trace ID."""
        # Setup
        mock_langsmith_client.list_runs.return_value = [mock_run]
        trace_id = str(uuid4())
        
        # Execute
        result = fetch_trace_tool(
            mock_langsmith_client,
            trace_id=trace_id
        )
        
        # Verify
        assert "trace_id" in result
        assert "run_type" in result
        mock_langsmith_client.list_runs.assert_called_once()
    
    def test_fetch_trace_by_project(self, mock_langsmith_client, mock_run):
        """Test fetch trace by project name."""
        # Setup
        mock_langsmith_client.list_runs.return_value = [mock_run]
        
        # Execute
        result = fetch_trace_tool(
            mock_langsmith_client,
            project_name="test-project"
        )
        
        # Verify
        assert "trace_id" in result
        assert "run_type" in result
    
    def test_fetch_trace_no_parameters(self, mock_langsmith_client):
        """Test fetch trace with no parameters returns error."""
        # Execute
        result = fetch_trace_tool(mock_langsmith_client)
        
        # Verify
        assert "error" in result
        assert "project_name or trace_id must be provided" in result["error"]
    
    def test_fetch_trace_not_found(self, mock_langsmith_client):
        """Test fetch trace when no runs found."""
        # Setup
        mock_langsmith_client.list_runs.return_value = []
        
        # Execute
        result = fetch_trace_tool(
            mock_langsmith_client,
            project_name="nonexistent-project"
        )
        
        # Verify
        assert "error" in result
        assert "No runs found" in result["error"]


class TestGetThreadHistoryTool:
    """Tests for get_thread_history_tool."""
    
    def test_get_thread_history_success(self, mock_langsmith_client, mock_run):
        """Test successful get thread history."""
        # Setup
        mock_run.inputs = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        mock_run.outputs = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hi there"}}
            ]
        }
        mock_run.run_type = "llm"
        mock_langsmith_client.list_runs.return_value = [mock_run]
        
        # Execute
        result = get_thread_history_tool(
            mock_langsmith_client,
            thread_id="test-thread-id",
            project_name="test-project"
        )
        
        # Verify
        assert "result" in result
        assert isinstance(result["result"], list)
        assert len(result["result"]) >= 1
    
    def test_get_thread_history_no_runs(self, mock_langsmith_client):
        """Test get thread history when no runs found."""
        # Setup
        mock_langsmith_client.list_runs.return_value = []
        
        # Execute
        result = get_thread_history_tool(
            mock_langsmith_client,
            thread_id="nonexistent-thread",
            project_name="test-project"
        )
        
        # Verify
        assert "error" in result
        assert "No runs found" in result["error"]
    
    def test_get_thread_history_no_messages(self, mock_langsmith_client, mock_run):
        """Test get thread history when run has no messages."""
        # Setup
        mock_run.inputs = {}
        mock_run.outputs = {}
        mock_run.run_type = "llm"
        mock_langsmith_client.list_runs.return_value = [mock_run]
        
        # Execute
        result = get_thread_history_tool(
            mock_langsmith_client,
            thread_id="test-thread-id",
            project_name="test-project"
        )
        
        # Verify
        assert "error" in result
        assert "No messages found" in result["error"]


class TestGetProjectRunsStatsTool:
    """Tests for get_project_runs_stats_tool."""
    
    def test_get_project_runs_stats_success(self, mock_langsmith_client):
        """Test successful get project runs stats."""
        # Setup
        mock_stats = {
            "run_count": 100,
            "avg_latency": 1.5,
            "error_count": 5,
            "total_tokens": 10000
        }
        mock_langsmith_client.get_run_stats.return_value = mock_stats
        
        # Execute
        result = get_project_runs_stats_tool(
            mock_langsmith_client,
            project_name="test-project"
        )
        
        # Verify
        assert "run_count" in result
        assert result["run_count"] == 100
        assert "project_name" in result
        assert result["project_name"] == "test-project"
    
    def test_get_project_runs_stats_by_trace_id(self, mock_langsmith_client):
        """Test get project runs stats by trace ID."""
        # Setup
        mock_stats = {
            "run_count": 10,
            "avg_latency": 2.0
        }
        mock_langsmith_client.get_run_stats.return_value = mock_stats
        trace_id = str(uuid4())
        
        # Execute
        result = get_project_runs_stats_tool(
            mock_langsmith_client,
            project_name="test-project",
            trace_id=trace_id
        )
        
        # Verify
        assert "run_count" in result
        assert result["run_count"] == 10
    
    def test_get_project_runs_stats_no_parameters(self, mock_langsmith_client):
        """Test get project runs stats with no parameters returns error."""
        # Execute
        result = get_project_runs_stats_tool(mock_langsmith_client)
        
        # Verify
        assert "error" in result
        assert "project_name or trace_id must be provided" in result["error"]
    
    def test_get_project_runs_stats_qualified_name(self, mock_langsmith_client):
        """Test get project runs stats with qualified project name."""
        # Setup
        mock_stats = {
            "run_count": 50,
            "avg_latency": 1.8
        }
        mock_langsmith_client.get_run_stats.return_value = mock_stats
        
        # Execute
        result = get_project_runs_stats_tool(
            mock_langsmith_client,
            project_name="owner/test-project"
        )
        
        # Verify
        assert "run_count" in result
        assert "project_name" in result
        assert result["project_name"] == "test-project"  # Should strip owner prefix

