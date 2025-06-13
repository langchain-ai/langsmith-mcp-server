"""Tests for dataset tools."""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from langsmith_mcp_server.services.tools.datasets import list_datasets_tool


class MockDataset:
    """Mock dataset object to simulate LangSmith dataset responses."""

    def __init__(
        self,
        id: str,
        name: str,
        description: str = None,
        data_type: str = "kv",
        created_at: datetime = None,
        modified_at: datetime = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.data_type = data_type
        self.created_at = created_at or datetime(2024, 1, 1, 12, 0, 0)
        self.modified_at = modified_at or datetime(2024, 1, 2, 12, 0, 0)


@pytest.fixture
def mock_client():
    """Create a mock LangSmith client."""
    client = Mock()
    return client


@pytest.fixture
def sample_datasets():
    """Create sample dataset objects for testing."""
    return [
        MockDataset(
            id="dataset-1",
            name="Test Dataset 1",
            description="First test dataset",
            data_type="kv",
        ),
        MockDataset(
            id="dataset-2",
            name="Chat Dataset",
            description="Dataset for chat conversations",
            data_type="chat",
        ),
        MockDataset(
            id="dataset-3",
            name="Empty Dataset",
            description=None,
            data_type="kv",
        ),
    ]


class TestListDatasetsTool:
    """Test cases for list_datasets_tool function."""

    def test_list_datasets_success_no_filters(self, mock_client, sample_datasets):
        """Test successful dataset listing without filters."""
        mock_client.list_datasets.return_value = iter(sample_datasets)

        result = list_datasets_tool(mock_client)

        assert "datasets" in result
        assert "total_count" in result
        assert result["total_count"] == 3
        assert len(result["datasets"]) == 3

        # Check first dataset structure
        first_dataset = result["datasets"][0]
        expected_attrs = [
            "id",
            "name",
            "description",
            "data_type",
            "created_at",
            "modified_at",
        ]
        for attr in expected_attrs:
            assert attr in first_dataset

        # Verify data
        assert first_dataset["id"] == "dataset-1"
        assert first_dataset["name"] == "Test Dataset 1"
        assert first_dataset["data_type"] == "kv"
        assert first_dataset["created_at"] == "2024-01-01T12:00:00"
        assert first_dataset["modified_at"] == "2024-01-02T12:00:00"

        # Verify client was called with no filters
        mock_client.list_datasets.assert_called_once_with(limit=20)

    def test_list_datasets_with_dataset_ids_filter(self, mock_client, sample_datasets):
        """Test dataset listing with dataset_ids filter."""
        filtered_datasets = [sample_datasets[0]]
        mock_client.list_datasets.return_value = iter(filtered_datasets)

        dataset_ids = ["dataset-1"]
        result = list_datasets_tool(mock_client, dataset_ids=dataset_ids)

        assert result["total_count"] == 1
        assert result["datasets"][0]["id"] == "dataset-1"

        mock_client.list_datasets.assert_called_once_with(
            dataset_ids=dataset_ids, limit=20
        )

    def test_list_datasets_with_data_type_filter(self, mock_client, sample_datasets):
        """Test dataset listing with data_type filter."""
        chat_datasets = [sample_datasets[1]]  # Only the chat dataset
        mock_client.list_datasets.return_value = iter(chat_datasets)

        result = list_datasets_tool(mock_client, data_type="chat")

        assert result["total_count"] == 1
        assert result["datasets"][0]["data_type"] == "chat"

        mock_client.list_datasets.assert_called_once_with(data_type="chat", limit=20)

    def test_list_datasets_with_name_filter(self, mock_client, sample_datasets):
        """Test dataset listing with dataset_name filter."""
        filtered_datasets = [sample_datasets[0]]
        mock_client.list_datasets.return_value = iter(filtered_datasets)

        result = list_datasets_tool(mock_client, dataset_name="Test Dataset 1")

        assert result["total_count"] == 1
        assert result["datasets"][0]["name"] == "Test Dataset 1"

        mock_client.list_datasets.assert_called_once_with(
            dataset_name="Test Dataset 1", limit=20
        )

    def test_list_datasets_with_name_contains_filter(
        self, mock_client, sample_datasets
    ):
        """Test dataset listing with dataset_name_contains filter."""
        filtered_datasets = [sample_datasets[1]]
        mock_client.list_datasets.return_value = iter(filtered_datasets)

        result = list_datasets_tool(mock_client, dataset_name_contains="Chat")

        assert result["total_count"] == 1
        assert result["datasets"][0]["name"] == "Chat Dataset"

        mock_client.list_datasets.assert_called_once_with(
            dataset_name_contains="Chat", limit=20
        )

    def test_list_datasets_with_metadata_filter(self, mock_client, sample_datasets):
        """Test dataset listing with metadata filter."""
        filtered_datasets = [sample_datasets[0]]
        mock_client.list_datasets.return_value = iter(filtered_datasets)

        metadata_filter = {"version": "1.0"}
        result = list_datasets_tool(mock_client, metadata=metadata_filter)

        assert result["total_count"] == 1
        # Note: metadata is not returned in the output, it's only used for filtering
        assert result["datasets"][0]["id"] == "dataset-1"

        mock_client.list_datasets.assert_called_once_with(
            metadata=metadata_filter, limit=20
        )

    def test_list_datasets_with_custom_limit(self, mock_client, sample_datasets):
        """Test dataset listing with custom limit."""
        mock_client.list_datasets.return_value = iter(sample_datasets[:2])

        result = list_datasets_tool(mock_client, limit=2)

        assert result["total_count"] == 2

        mock_client.list_datasets.assert_called_once_with(limit=2)

    def test_list_datasets_with_all_filters(self, mock_client, sample_datasets):
        """Test dataset listing with all filters applied."""
        filtered_datasets = [sample_datasets[0]]
        mock_client.list_datasets.return_value = iter(filtered_datasets)

        result = list_datasets_tool(
            mock_client,
            dataset_ids=["dataset-1"],
            data_type="kv",
            dataset_name="Test Dataset 1",
            dataset_name_contains="Test",
            metadata={"version": "1.0"},
            limit=10,
        )

        assert result["total_count"] == 1

        mock_client.list_datasets.assert_called_once_with(
            dataset_ids=["dataset-1"],
            data_type="kv",
            dataset_name="Test Dataset 1",
            dataset_name_contains="Test",
            metadata={"version": "1.0"},
            limit=10,
        )

    def test_list_datasets_empty_result(self, mock_client):
        """Test dataset listing when no datasets are found."""
        mock_client.list_datasets.return_value = iter([])

        result = list_datasets_tool(mock_client)

        assert result["total_count"] == 0
        assert result["datasets"] == []

    def test_list_datasets_with_none_values(self, mock_client, sample_datasets):
        """Test that None values are properly handled and not passed to client."""
        mock_client.list_datasets.return_value = iter(sample_datasets)

        result = list_datasets_tool(
            mock_client,
            dataset_ids=None,
            data_type=None,
            dataset_name=None,
            dataset_name_contains=None,
            metadata=None,
            limit=None,
        )

        assert result["total_count"] == 3

        # Should not pass any parameters since all are None (including limit)
        mock_client.list_datasets.assert_called_once_with()

    def test_list_datasets_handles_missing_attributes(self, mock_client):
        """Test handling of datasets with missing attributes."""
        # Create a mock dataset with missing attributes
        incomplete_dataset = Mock()
        incomplete_dataset.id = "incomplete-1"
        incomplete_dataset.name = "Incomplete Dataset"
        # Configure missing attributes to return None when accessed via getattr
        incomplete_dataset.configure_mock(
            **{
                "description": None,
                "data_type": None,
                "created_at": None,
                "modified_at": None,
            }
        )

        mock_client.list_datasets.return_value = iter([incomplete_dataset])

        result = list_datasets_tool(mock_client)

        assert result["total_count"] == 1
        dataset = result["datasets"][0]
        assert dataset["id"] == "incomplete-1"
        assert dataset["name"] == "Incomplete Dataset"
        # Missing attributes should be None
        assert dataset["description"] is None
        assert dataset["data_type"] is None

    def test_list_datasets_client_exception(self, mock_client):
        """Test error handling when client raises an exception."""
        mock_client.list_datasets.side_effect = Exception("API Error")

        result = list_datasets_tool(mock_client)

        assert "error" in result
        assert "Error fetching datasets: API Error" in result["error"]
