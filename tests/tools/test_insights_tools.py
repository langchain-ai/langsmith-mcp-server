"""Tests for insights tools."""

from unittest.mock import Mock, patch

import pytest

from langsmith_mcp_server.services.tools.insights import (
    _extract_items,
    _normalize_run,
    _resolve_session_id,
    get_insights_cluster_tool,
    get_insights_job_tool,
    get_insights_runs_tool,
    list_insights_jobs_tool,
)

API_KEY = "lsv2_pt_test"
ENDPOINT = "https://api.smith.langchain.com"
SESSION_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
JOB_ID = "11111111-2222-3333-4444-555555555555"
CLUSTER_ID = "66666666-7777-8888-9999-000000000000"


@pytest.fixture
def mock_client():
    """Create a mock LangSmith client."""
    client = Mock()
    project = Mock()
    project.id = SESSION_ID
    client.read_project.return_value = project
    return client


class TestResolveSessionId:
    """Test cases for _resolve_session_id helper."""

    def test_session_id_takes_precedence(self, mock_client):
        """session_id is returned directly, project_name ignored."""
        result = _resolve_session_id(mock_client, project_name="my-proj", session_id=SESSION_ID)
        assert result == SESSION_ID
        mock_client.read_project.assert_not_called()

    def test_project_name_resolution(self, mock_client):
        """project_name is resolved via client.read_project."""
        result = _resolve_session_id(mock_client, project_name="my-proj")
        assert result == SESSION_ID
        mock_client.read_project.assert_called_once_with(project_name="my-proj")

    def test_neither_provided_raises(self, mock_client):
        """Raises ValueError when neither param is given."""
        with pytest.raises(ValueError, match="Either project_name or session_id"):
            _resolve_session_id(mock_client)

    def test_project_not_found(self, mock_client):
        """Propagates exception when project lookup fails."""
        mock_client.read_project.side_effect = Exception("Project not found")
        with pytest.raises(Exception, match="Project not found"):
            _resolve_session_id(mock_client, project_name="bad-proj")


class TestNormalizeRun:
    """Test cases for _normalize_run helper."""

    def test_stringifies_id_and_trace_id(self):
        """id and trace_id are converted to strings."""
        run = {"id": 123, "trace_id": 456}
        result = _normalize_run(run)
        assert result["id"] == "123"
        assert result["trace_id"] == "456"

    def test_extracts_thread_id_from_metadata(self):
        """thread_id is pulled from metadata when not at top level."""
        run = {"id": "r1", "metadata": {"thread_id": "t1"}}
        result = _normalize_run(run)
        assert result["thread_id"] == "t1"

    def test_extracts_session_id_from_metadata(self):
        """Falls back to session_id in metadata."""
        run = {"id": "r1", "metadata": {"session_id": "s1"}}
        result = _normalize_run(run)
        assert result["thread_id"] == "s1"

    def test_extracts_conversation_id_from_metadata(self):
        """Falls back to conversation_id in metadata."""
        run = {"id": "r1", "metadata": {"conversation_id": "c1"}}
        result = _normalize_run(run)
        assert result["thread_id"] == "c1"

    def test_prefers_existing_thread_id(self):
        """Existing top-level thread_id is not overwritten."""
        run = {"id": "r1", "thread_id": "top", "metadata": {"thread_id": "meta"}}
        result = _normalize_run(run)
        assert result["thread_id"] == "top"

    def test_no_metadata(self):
        """No crash when metadata is missing."""
        run = {"id": "r1"}
        result = _normalize_run(run)
        assert "thread_id" not in result

    def test_empty_metadata(self):
        """No crash when metadata is empty dict."""
        run = {"id": "r1", "metadata": {}}
        result = _normalize_run(run)
        assert "thread_id" not in result

    def test_none_metadata(self):
        """No crash when metadata is None."""
        run = {"id": "r1", "metadata": None}
        result = _normalize_run(run)
        assert "thread_id" not in result

    def test_none_thread_id_falls_through_to_metadata(self):
        """Explicit None thread_id falls through to metadata extraction."""
        run = {"id": "r1", "thread_id": None, "metadata": {"thread_id": "from-meta"}}
        result = _normalize_run(run)
        assert result["thread_id"] == "from-meta"

    def test_empty_string_thread_id_falls_through_to_metadata(self):
        """Empty string thread_id falls through to metadata extraction."""
        run = {"id": "r1", "thread_id": "", "metadata": {"session_id": "from-meta"}}
        result = _normalize_run(run)
        assert result["thread_id"] == "from-meta"

    def test_falsy_thread_id_no_metadata_match(self):
        """Falsy thread_id with no metadata match leaves thread_id as-is."""
        run = {"id": "r1", "thread_id": None, "metadata": {}}
        result = _normalize_run(run)
        assert result["thread_id"] is None


class TestExtractItems:
    """Test cases for _extract_items helper."""

    def test_plain_list(self):
        """Plain list is returned as-is."""
        assert _extract_items([{"id": "a"}, {"id": "b"}], "jobs") == [{"id": "a"}, {"id": "b"}]

    def test_nested_wrapper_dict(self):
        """[{'clustering_jobs': [...]}] is flattened to the inner list."""
        inner = [{"id": "j1"}, {"id": "j2"}]
        assert _extract_items([{"clustering_jobs": inner}], "clustering_jobs") == inner

    def test_nested_wrapper_first_key_wins(self):
        """First matching key in the wrapper dict is used."""
        inner = [{"id": "j1"}]
        result = _extract_items(
            [{"clustering_jobs": inner, "other": [1, 2]}],
            "clustering_jobs",
            "other",
        )
        assert result == inner

    def test_dict_with_known_key(self):
        """Dict response with a matching key extracts the list."""
        assert _extract_items({"runs": [1, 2], "total": 5}, "runs") == [1, 2]

    def test_dict_fallback_to_second_key(self):
        """Falls back to second key if first is absent."""
        assert _extract_items({"items": [3]}, "runs", "items") == [3]

    def test_dict_no_matching_key(self):
        """Dict without any matching key returns wrapped singleton."""
        assert _extract_items({"foo": "bar"}, "runs") == [{"foo": "bar"}]

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _extract_items([], "jobs") == []

    def test_multi_element_list_not_unwrapped(self):
        """List with >1 element is never unwrapped."""
        data = [{"clustering_jobs": [1]}, {"clustering_jobs": [2]}]
        assert _extract_items(data, "clustering_jobs") == data

    def test_wrapper_dict_with_non_list_value(self):
        """If the wrapper key's value is not a list, return the outer list."""
        assert _extract_items([{"clustering_jobs": "not-a-list"}], "clustering_jobs") == [
            {"clustering_jobs": "not-a-list"}
        ]


class TestListInsightsJobsTool:
    """Test cases for list_insights_jobs_tool."""

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_success_with_session_id(self, mock_request, mock_client):
        """Returns jobs list when API responds with a list."""
        jobs = [{"id": JOB_ID, "status": "completed"}]
        mock_request.return_value = jobs

        result = list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client, session_id=SESSION_ID)

        assert result == {"jobs": jobs}
        mock_request.assert_called_once_with(
            API_KEY,
            ENDPOINT,
            f"/api/v1/sessions/{SESSION_ID}/insights",
            {"limit": "20", "offset": "0"},
            None,
        )

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_success_with_project_name(self, mock_request, mock_client):
        """Resolves project name and returns jobs."""
        mock_request.return_value = [{"id": JOB_ID}]

        result = list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client, project_name="my-proj")

        assert result == {"jobs": [{"id": JOB_ID}]}
        mock_client.read_project.assert_called_once_with(project_name="my-proj")

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_pagination_params(self, mock_request, mock_client):
        """Passes limit and offset correctly."""
        mock_request.return_value = []

        list_insights_jobs_tool(
            API_KEY, ENDPOINT, mock_client, session_id=SESSION_ID, limit=50, offset=10
        )

        mock_request.assert_called_once_with(
            API_KEY,
            ENDPOINT,
            f"/api/v1/sessions/{SESSION_ID}/insights",
            {"limit": "50", "offset": "10"},
            None,
        )

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_limit_clamped(self, mock_request, mock_client):
        """Limit is clamped to [1, 100]."""
        mock_request.return_value = []

        list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client, session_id=SESSION_ID, limit=200)
        assert mock_request.call_args[0][3]["limit"] == "100"

        list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client, session_id=SESSION_ID, limit=-5)
        assert mock_request.call_args[0][3]["limit"] == "1"

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_offset_clamped(self, mock_request, mock_client):
        """Negative offset is clamped to 0."""
        mock_request.return_value = []

        list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client, session_id=SESSION_ID, offset=-3)
        assert mock_request.call_args[0][3]["offset"] == "0"

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_api_error(self, mock_request, mock_client):
        """Returns error dict on API failure."""
        mock_request.return_value = {"error": "HTTP 500: Internal Server Error"}

        result = list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client, session_id=SESSION_ID)

        assert "error" in result
        assert "500" in result["error"]

    def test_resolve_error(self, mock_client):
        """Returns error when neither project_name nor session_id is given."""
        result = list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client)
        assert "error" in result
        assert "Either project_name or session_id" in result["error"]

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_dict_response_with_jobs_key(self, mock_request, mock_client):
        """Handles dict response that contains a 'jobs' key."""
        mock_request.return_value = {"jobs": [{"id": JOB_ID}], "total": 1}

        result = list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client, session_id=SESSION_ID)

        assert result == {"jobs": [{"id": JOB_ID}]}

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_nested_clustering_jobs_flattened(self, mock_request, mock_client):
        """Real API shape [{'clustering_jobs': [...]}] is flattened."""
        inner_jobs = [{"id": JOB_ID, "status": "success", "name": "Pain Points"}]
        mock_request.return_value = [{"clustering_jobs": inner_jobs}]

        result = list_insights_jobs_tool(API_KEY, ENDPOINT, mock_client, session_id=SESSION_ID)

        assert result == {"jobs": inner_jobs}


class TestGetInsightsJobTool:
    """Test cases for get_insights_job_tool."""

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_success(self, mock_request, mock_client):
        """Returns job details."""
        job_data = {
            "id": JOB_ID,
            "status": "completed",
            "clusters": [{"id": CLUSTER_ID, "label": "FAQ queries"}],
        }
        mock_request.return_value = job_data

        result = get_insights_job_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID
        )

        assert result == job_data
        mock_request.assert_called_once_with(
            API_KEY,
            ENDPOINT,
            f"/api/v1/sessions/{SESSION_ID}/insights/{JOB_ID}",
            workspace_id=None,
        )

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_api_error(self, mock_request, mock_client):
        """Returns error on API failure."""
        mock_request.return_value = {"error": "HTTP 404: Not found"}

        result = get_insights_job_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID
        )

        assert "error" in result

    def test_resolve_error(self, mock_client):
        """Returns error when session cannot be resolved."""
        result = get_insights_job_tool(API_KEY, ENDPOINT, mock_client, job_id=JOB_ID)
        assert "error" in result

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_with_project_name(self, mock_request, mock_client):
        """Resolves project name before fetching job."""
        mock_request.return_value = {"id": JOB_ID}

        result = get_insights_job_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, project_name="my-proj"
        )

        assert result == {"id": JOB_ID}
        mock_client.read_project.assert_called_once_with(project_name="my-proj")


class TestGetInsightsClusterTool:
    """Test cases for get_insights_cluster_tool."""

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_success(self, mock_request, mock_client):
        """Returns cluster details."""
        cluster_data = {
            "id": CLUSTER_ID,
            "label": "FAQ queries",
            "description": "Questions about product features",
            "size": 42,
        }
        mock_request.return_value = cluster_data

        result = get_insights_cluster_tool(
            API_KEY,
            ENDPOINT,
            mock_client,
            job_id=JOB_ID,
            cluster_id=CLUSTER_ID,
            session_id=SESSION_ID,
        )

        assert result == cluster_data
        expected_path = f"/api/v1/sessions/{SESSION_ID}/insights/{JOB_ID}/clusters/{CLUSTER_ID}"
        mock_request.assert_called_once_with(API_KEY, ENDPOINT, expected_path, workspace_id=None)

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_api_error(self, mock_request, mock_client):
        """Returns error on API failure."""
        mock_request.return_value = {"error": "HTTP 404: Not found"}

        result = get_insights_cluster_tool(
            API_KEY,
            ENDPOINT,
            mock_client,
            job_id=JOB_ID,
            cluster_id=CLUSTER_ID,
            session_id=SESSION_ID,
        )

        assert "error" in result

    def test_resolve_error(self, mock_client):
        """Returns error when session cannot be resolved."""
        result = get_insights_cluster_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, cluster_id=CLUSTER_ID
        )
        assert "error" in result

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_with_project_name(self, mock_request, mock_client):
        """Resolves project name before fetching cluster."""
        mock_request.return_value = {"id": CLUSTER_ID}

        result = get_insights_cluster_tool(
            API_KEY,
            ENDPOINT,
            mock_client,
            job_id=JOB_ID,
            cluster_id=CLUSTER_ID,
            project_name="my-proj",
        )

        assert result == {"id": CLUSTER_ID}
        mock_client.read_project.assert_called_once_with(project_name="my-proj")


class TestGetInsightsRunsTool:
    """Test cases for get_insights_runs_tool."""

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_success_returns_paginated(self, mock_request, mock_client):
        """Returns paginated runs with metadata."""
        runs = [{"id": "run-1", "name": "ChatOpenAI"}]
        mock_request.return_value = runs

        result = get_insights_runs_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID
        )

        assert "runs" in result
        assert "page_number" in result
        assert "total_pages" in result
        assert "max_chars_per_page" in result
        assert "preview_chars" in result
        assert result["page_number"] == 1
        assert result["total_pages"] >= 1
        # The run should be present and normalized
        assert len(result["runs"]) == 1
        assert result["runs"][0]["id"] == "run-1"

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_normalizes_run_ids(self, mock_request, mock_client):
        """Runs are normalized so id/trace_id/thread_id are top-level strings."""
        runs = [
            {
                "id": 123,
                "trace_id": 456,
                "metadata": {"thread_id": "thread-abc"},
            }
        ]
        mock_request.return_value = runs

        result = get_insights_runs_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID
        )

        run = result["runs"][0]
        assert run["id"] == "123"
        assert run["trace_id"] == "456"
        assert run["thread_id"] == "thread-abc"

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_with_cluster_filter(self, mock_request, mock_client):
        """Passes cluster_id as query parameter."""
        mock_request.return_value = []

        get_insights_runs_tool(
            API_KEY,
            ENDPOINT,
            mock_client,
            job_id=JOB_ID,
            session_id=SESSION_ID,
            cluster_id=CLUSTER_ID,
        )

        call_params = mock_request.call_args[0][3]
        assert call_params["cluster_id"] == CLUSTER_ID

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_sorting_params(self, mock_request, mock_client):
        """Passes sort_by and sort_direction."""
        mock_request.return_value = []

        get_insights_runs_tool(
            API_KEY,
            ENDPOINT,
            mock_client,
            job_id=JOB_ID,
            session_id=SESSION_ID,
            sort_by="latency",
            sort_direction="desc",
        )

        call_params = mock_request.call_args[0][3]
        assert call_params["sort_by"] == "latency"
        assert call_params["sort_direction"] == "desc"

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_pagination_params(self, mock_request, mock_client):
        """Passes limit and offset correctly."""
        mock_request.return_value = []

        get_insights_runs_tool(
            API_KEY,
            ENDPOINT,
            mock_client,
            job_id=JOB_ID,
            session_id=SESSION_ID,
            limit=50,
            offset=10,
        )

        call_params = mock_request.call_args[0][3]
        assert call_params["limit"] == "50"
        assert call_params["offset"] == "10"

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_char_pagination_params(self, mock_request, mock_client):
        """page_number, max_chars_per_page, preview_chars are honoured."""
        mock_request.return_value = [{"id": "r1", "data": "x" * 100}]

        result = get_insights_runs_tool(
            API_KEY,
            ENDPOINT,
            mock_client,
            job_id=JOB_ID,
            session_id=SESSION_ID,
            page_number=1,
            max_chars_per_page=20000,
            preview_chars=50,
        )

        assert result["max_chars_per_page"] == 20000
        assert result["preview_chars"] == 50

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_max_chars_capped(self, mock_request, mock_client):
        """max_chars_per_page is capped at 30000."""
        mock_request.return_value = [{"id": "r1"}]

        result = get_insights_runs_tool(
            API_KEY,
            ENDPOINT,
            mock_client,
            job_id=JOB_ID,
            session_id=SESSION_ID,
            max_chars_per_page=99999,
        )

        assert result["max_chars_per_page"] == 30000

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_limit_clamped(self, mock_request, mock_client):
        """Limit is clamped to [1, 100]."""
        mock_request.return_value = []

        get_insights_runs_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID, limit=200
        )
        assert mock_request.call_args[0][3]["limit"] == "100"

        get_insights_runs_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID, limit=0
        )
        assert mock_request.call_args[0][3]["limit"] == "1"

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_api_error(self, mock_request, mock_client):
        """Returns error on API failure."""
        mock_request.return_value = {"error": "HTTP 500: Internal Server Error"}

        result = get_insights_runs_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID
        )

        assert "error" in result

    def test_resolve_error(self, mock_client):
        """Returns error when session cannot be resolved."""
        result = get_insights_runs_tool(API_KEY, ENDPOINT, mock_client, job_id=JOB_ID)
        assert "error" in result

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_dict_response_with_runs_key(self, mock_request, mock_client):
        """Handles dict response that contains a 'runs' key."""
        mock_request.return_value = {"runs": [{"id": "run-1"}], "total": 1}

        result = get_insights_runs_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID
        )

        assert result["runs"][0]["id"] == "run-1"
        assert result["page_number"] == 1

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_optional_params_not_sent_when_none(self, mock_request, mock_client):
        """Optional params like cluster_id, sort_by, sort_direction are excluded when None."""
        mock_request.return_value = []

        get_insights_runs_tool(API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID)

        call_params = mock_request.call_args[0][3]
        assert "cluster_id" not in call_params
        assert "sort_by" not in call_params
        assert "sort_direction" not in call_params

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_empty_runs_returns_zero_pages(self, mock_request, mock_client):
        """Empty run list returns total_pages=0."""
        mock_request.return_value = []

        result = get_insights_runs_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID
        )

        assert result["runs"] == []
        assert result["total_pages"] == 0

    @patch("langsmith_mcp_server.services.tools.insights._request")
    def test_nested_wrapper_runs_flattened(self, mock_request, mock_client):
        """If API returns [{'runs': [...]}] the inner list is extracted."""
        inner_runs = [{"id": "r1", "trace_id": "t1"}]
        mock_request.return_value = [{"runs": inner_runs}]

        result = get_insights_runs_tool(
            API_KEY, ENDPOINT, mock_client, job_id=JOB_ID, session_id=SESSION_ID
        )

        assert result["runs"][0]["id"] == "r1"
        assert result["page_number"] == 1
