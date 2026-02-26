"""Tests for common/helpers.py utilities."""

from unittest.mock import AsyncMock, patch

import pytest

from langsmith_mcp_server.common.helpers import get_api_key_and_endpoint_from_context


class TestGetApiKeyAndEndpointFromContext:
    """Tests for get_api_key_and_endpoint_from_context."""

    @pytest.mark.asyncio
    async def test_skips_client_creation_when_state_populated(self):
        """Does not call get_client_from_context when ctx already has api_key."""
        ctx = AsyncMock()
        ctx.get_state.side_effect = lambda k: {
            "api_key": "lsv2_pt_cached",
            "endpoint": "https://custom.endpoint.com",
        }.get(k)

        with patch("langsmith_mcp_server.common.helpers.get_client_from_context") as mock_gcc:
            api_key, endpoint = await get_api_key_and_endpoint_from_context(ctx)

        mock_gcc.assert_not_called()
        assert api_key == "lsv2_pt_cached"
        assert endpoint == "https://custom.endpoint.com"

    @pytest.mark.asyncio
    async def test_calls_client_creation_when_state_empty(self):
        """Calls get_client_from_context when ctx has no api_key yet."""
        call_count = 0

        async def get_state_side_effect(k):
            # First call to get_state("api_key") returns empty (guard check),
            # after get_client_from_context populates state, subsequent calls return values.
            nonlocal call_count
            if k == "api_key":
                call_count += 1
                if call_count <= 1:
                    return ""
                return "lsv2_pt_new"
            if k == "endpoint":
                return ""
            return ""

        ctx = AsyncMock()
        ctx.get_state.side_effect = get_state_side_effect

        with patch(
            "langsmith_mcp_server.common.helpers.get_client_from_context", new_callable=AsyncMock
        ) as mock_gcc:
            api_key, endpoint = await get_api_key_and_endpoint_from_context(ctx)

        mock_gcc.assert_called_once_with(ctx)
        assert api_key == "lsv2_pt_new"

    @pytest.mark.asyncio
    async def test_default_endpoint_when_none(self):
        """Uses default LangSmith endpoint when ctx has no endpoint set."""
        ctx = AsyncMock()
        ctx.get_state.side_effect = lambda k: {
            "api_key": "lsv2_pt_test",
            "endpoint": "",
        }.get(k, "")

        with patch("langsmith_mcp_server.common.helpers.get_client_from_context"):
            _, endpoint = await get_api_key_and_endpoint_from_context(ctx)

        assert endpoint == "https://api.smith.langchain.com"

    @pytest.mark.asyncio
    async def test_endpoint_trailing_slash_stripped(self):
        """Trailing slash is stripped from endpoint."""
        ctx = AsyncMock()
        ctx.get_state.side_effect = lambda k: {
            "api_key": "lsv2_pt_test",
            "endpoint": "https://custom.endpoint.com/",
        }.get(k, "")

        with patch("langsmith_mcp_server.common.helpers.get_client_from_context"):
            _, endpoint = await get_api_key_and_endpoint_from_context(ctx)

        assert endpoint == "https://custom.endpoint.com"
