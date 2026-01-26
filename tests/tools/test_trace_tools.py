"""Tests for trace tools."""

from datetime import datetime
from unittest.mock import Mock

import pytest

from langsmith_mcp_server.services.tools.traces import (
    get_project_runs_stats_tool,
    get_thread_history_tool,
)


class MockRun:
    """Mock run object to simulate LangSmith Run responses."""

    def __init__(
        self,
        id: str = "run-1",
        start_time: datetime = None,
        inputs: dict = None,
        outputs: dict = None,
        run_type: str = "llm",
    ):
        self.id = id
        self.start_time = start_time or datetime(2024, 1, 1, 12, 0, 0)
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.run_type = run_type


@pytest.fixture
def mock_client():
    """Create a mock LangSmith client."""
    client = Mock()
    return client


@pytest.fixture
def sample_runs_with_messages():
    """Create sample run objects with messages for thread history testing."""
    return [
        MockRun(
            id="run-1",
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
            outputs={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "How can I help you?",
                        }
                    }
                ]
            },
        ),
        MockRun(
            id="run-2",
            start_time=datetime(2024, 1, 1, 13, 0, 0),
            inputs={
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                ]
            },
            outputs={
                "message": {
                    "role": "assistant",
                    "content": "I don't have access to weather data.",
                }
            },
        ),
    ]


@pytest.fixture
def sample_stats_response():
    """Create sample stats response for project runs stats testing."""
    return {
        "total_runs": 100,
        "successful_runs": 95,
        "failed_runs": 5,
        "average_latency": 1.5,
        "run_facets": {"some": "data"},
    }


class TestGetThreadHistoryTool:
    """Test cases for get_thread_history_tool function."""

    def test_get_thread_history_success_with_choices_format(
        self, mock_client, sample_runs_with_messages
    ):
        """Test successful thread history retrieval with choices format."""
        mock_client.list_runs.return_value = iter([sample_runs_with_messages[0]])

        result = get_thread_history_tool(
            mock_client, thread_id="thread-123", project_name="test-project"
        )

        assert "result" in result
        assert "error" not in result
        assert isinstance(result["result"], list)
        assert len(result["result"]) == 3  # 2 input messages + 1 output message

        # Verify messages are correct
        messages = result["result"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "How can I help you?"

        # Verify client was called correctly
        mock_client.list_runs.assert_called_once_with(
            project_name="test-project",
            filter='and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "thread-123"))',
            run_type="llm",
        )

    def test_get_thread_history_success_with_message_format(
        self, mock_client, sample_runs_with_messages
    ):
        """Test successful thread history retrieval with message format."""
        mock_client.list_runs.return_value = iter([sample_runs_with_messages[1]])

        result = get_thread_history_tool(
            mock_client, thread_id="thread-456", project_name="test-project"
        )

        assert "result" in result
        assert "error" not in result
        assert isinstance(result["result"], list)
        assert len(result["result"]) == 2  # 1 input message + 1 output message

        messages = result["result"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What's the weather?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "I don't have access to weather data."

    def test_get_thread_history_sorts_by_start_time(self, mock_client, sample_runs_with_messages):
        """Test that runs are sorted by start_time (most recent first)."""
        # Return runs in reverse chronological order
        runs = [sample_runs_with_messages[0], sample_runs_with_messages[1]]
        mock_client.list_runs.return_value = iter(runs)

        result = get_thread_history_tool(
            mock_client, thread_id="thread-789", project_name="test-project"
        )

        # Should use the most recent run (run-2, which has later start_time)
        assert "result" in result
        messages = result["result"]
        # Should have messages from run-2 (the more recent one)
        assert messages[0]["content"] == "What's the weather?"

    def test_get_thread_history_no_runs_found(self, mock_client):
        """Test thread history when no runs are found."""
        mock_client.list_runs.return_value = iter([])

        result = get_thread_history_tool(
            mock_client, thread_id="nonexistent-thread", project_name="test-project"
        )

        assert "error" in result
        assert "No runs found" in result["error"]
        assert "nonexistent-thread" in result["error"]

    def test_get_thread_history_no_messages_in_run(self, mock_client):
        """Test thread history when run has no messages."""
        run_without_messages = MockRun(
            id="run-no-msg",
            inputs={},
            outputs={},
        )
        mock_client.list_runs.return_value = iter([run_without_messages])

        result = get_thread_history_tool(
            mock_client, thread_id="thread-no-msg", project_name="test-project"
        )

        assert "error" in result
        assert "No messages found" in result["error"]

    def test_get_thread_history_empty_inputs(self, mock_client):
        """Test thread history when inputs don't have messages key."""
        run_without_input_messages = MockRun(
            id="run-no-input-msg",
            inputs={"other": "data"},
            outputs={"message": {"role": "assistant", "content": "Response"}},
        )
        mock_client.list_runs.return_value = iter([run_without_input_messages])

        result = get_thread_history_tool(
            mock_client, thread_id="thread-partial", project_name="test-project"
        )

        # Should still work if output has message
        assert "result" in result
        assert len(result["result"]) == 1
        assert result["result"][0]["content"] == "Response"

    def test_get_thread_history_client_exception(self, mock_client):
        """Test error handling when client raises an exception."""
        mock_client.list_runs.side_effect = Exception("API Error")

        result = get_thread_history_tool(
            mock_client, thread_id="thread-error", project_name="test-project"
        )

        assert "error" in result
        assert "Error fetching thread history: API Error" in result["error"]

    def test_get_thread_history_empty_choices_list(self, mock_client):
        """Test thread history when choices list is empty."""
        run_with_empty_choices = MockRun(
            id="run-empty-choices",
            inputs={"messages": [{"role": "user", "content": "Test"}]},
            outputs={"choices": []},
        )
        mock_client.list_runs.return_value = iter([run_with_empty_choices])

        result = get_thread_history_tool(
            mock_client, thread_id="thread-empty", project_name="test-project"
        )

        # Should only have input messages
        assert "result" in result
        assert len(result["result"]) == 1
        assert result["result"][0]["role"] == "user"


class TestGetProjectRunsStatsTool:
    """Test cases for get_project_runs_stats_tool function."""

    def test_get_project_runs_stats_success_with_project_name(
        self, mock_client, sample_stats_response
    ):
        """Test successful stats retrieval with project_name."""
        stats_response = sample_stats_response.copy()
        mock_client.get_run_stats.return_value = stats_response

        result = get_project_runs_stats_tool(
            mock_client, project_name="test-project", trace_id=None
        )

        assert "error" not in result
        assert "project_name" in result
        assert result["project_name"] == "test-project"
        assert "total_runs" in result
        assert result["total_runs"] == 100
        assert "run_facets" not in result  # Should be removed

        mock_client.get_run_stats.assert_called_once_with(
            project_names=["test-project"], trace=None
        )

    def test_get_project_runs_stats_success_with_trace_id(
        self, mock_client, sample_stats_response
    ):
        """Test successful stats retrieval with trace_id."""
        stats_response = sample_stats_response.copy()
        mock_client.get_run_stats.return_value = stats_response

        result = get_project_runs_stats_tool(
            mock_client, project_name=None, trace_id="trace-123"
        )

        assert "error" not in result
        assert "total_runs" in result
        # When trace_id is used, project_name should not be in result
        assert "project_name" not in result

        mock_client.get_run_stats.assert_called_once_with(
            project_names=None, trace="trace-123"
        )

    def test_get_project_runs_stats_both_params_trace_id_preferred(
        self, mock_client, sample_stats_response
    ):
        """Test that trace_id is preferred when both parameters are provided."""
        stats_response = sample_stats_response.copy()
        mock_client.get_run_stats.return_value = stats_response

        result = get_project_runs_stats_tool(
            mock_client, project_name="test-project", trace_id="trace-456"
        )

        assert "error" not in result

        # Should use trace_id, not project_name
        mock_client.get_run_stats.assert_called_once_with(
            project_names=["test-project"], trace="trace-456"
        )

    def test_get_project_runs_stats_neither_param_provided(self, mock_client):
        """Test error when neither parameter is provided."""
        result = get_project_runs_stats_tool(mock_client, project_name=None, trace_id=None)

        assert "error" in result
        assert "Either project_name or trace_id must be provided" in result["error"]
        mock_client.get_run_stats.assert_not_called()

    def test_get_project_runs_stats_null_string_handling(self, mock_client, sample_stats_response):
        """Test that "null" strings are converted to None."""
        stats_response = sample_stats_response.copy()
        mock_client.get_run_stats.return_value = stats_response

        result = get_project_runs_stats_tool(
            mock_client, project_name="null", trace_id="null"
        )

        # Should treat "null" as None
        mock_client.get_run_stats.assert_called_once_with(
            project_names=None, trace=None
        )

    def test_get_project_runs_stats_qualified_project_name(self, mock_client, sample_stats_response):
        """Test handling of qualified project names (owner/project)."""
        stats_response = sample_stats_response.copy()
        mock_client.get_run_stats.return_value = stats_response

        result = get_project_runs_stats_tool(
            mock_client, project_name="owner/test-project", trace_id=None
        )

        assert "error" not in result
        assert result["project_name"] == "test-project"  # Should extract just the project name

        mock_client.get_run_stats.assert_called_once_with(
            project_names=["test-project"], trace=None
        )

    def test_get_project_runs_stats_removes_run_facets(
        self, mock_client, sample_stats_response
    ):
        """Test that run_facets are removed from the response."""
        stats_response = sample_stats_response.copy()
        assert "run_facets" in stats_response  # Verify it's in the original

        mock_client.get_run_stats.return_value = stats_response

        result = get_project_runs_stats_tool(
            mock_client, project_name="test-project", trace_id=None
        )

        assert "run_facets" not in result
        assert "total_runs" in result  # Other fields should remain

    def test_get_project_runs_stats_client_exception(self, mock_client):
        """Test error handling when client raises an exception."""
        mock_client.get_run_stats.side_effect = Exception("Stats API Error")

        result = get_project_runs_stats_tool(
            mock_client, project_name="test-project", trace_id=None
        )

        assert "error" in result
        assert "Error getting project runs stats: Stats API Error" in result["error"]

    def test_get_project_runs_stats_empty_stats_response(self, mock_client):
        """Test handling of empty stats response."""
        mock_client.get_run_stats.return_value = {}

        result = get_project_runs_stats_tool(
            mock_client, project_name="test-project", trace_id=None
        )

        assert "error" not in result
        assert "project_name" in result
        assert result["project_name"] == "test-project"

