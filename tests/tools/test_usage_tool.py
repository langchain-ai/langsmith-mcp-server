"""Tests for usage tools."""

import json
from unittest.mock import patch

from langsmith_mcp_server.services.tools.usage import (
    _request,
    get_billing_usage_tool,
)


class FakeResponse:
    """Minimal context-manager response for urllib.request.urlopen."""

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def _headers(request):
    return {key.lower(): value for key, value in request.header_items()}


def test_request_adds_tenant_header_when_workspace_id_is_provided():
    """Org-scoped LangSmith endpoints require X-Tenant-Id for workspace-scoped keys."""
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return FakeResponse({"ok": True})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = _request(
            "test-key",
            "https://api.smith.langchain.com",
            "/api/v1/orgs/current/billing/usage",
            workspace_id="workspace-123",
        )

    assert result == {"ok": True}
    assert _headers(requests[0])["x-tenant-id"] == "workspace-123"


def test_billing_usage_propagates_workspace_id_to_rest_requests():
    """The billing tool should preserve LANGSMITH_WORKSPACE_ID for raw REST calls."""
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.startswith(
            "https://api.smith.langchain.com/api/v1/orgs/current/billing/usage"
        ):
            return FakeResponse([{"name": "trace_count", "groups": {"workspace-123": 5}}])
        if request.full_url == "https://api.smith.langchain.com/api/v1/workspaces":
            return FakeResponse([{"id": "workspace-123", "display_name": "Production"}])
        return FakeResponse({"error": "unexpected url"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = get_billing_usage_tool(
            api_key="test-key",
            endpoint="https://api.smith.langchain.com",
            starting_on="2026-03-01T00:00:00Z",
            ending_before="2026-03-12T00:00:00Z",
            workspace_id="workspace-123",
        )

    assert result == [
        {
            "name": "trace_count",
            "groups": {
                "workspace-123": {"workspace_name": "Production", "value": 5},
            },
        }
    ]
    assert all(_headers(request)["x-tenant-id"] == "workspace-123" for request in requests)
