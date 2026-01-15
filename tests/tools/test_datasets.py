"""Tests for dataset tools."""

import pytest
from unittest.mock import Mock
from langsmith_mcp_server.services.tools.datasets import (
    list_datasets_tool,
    list_examples_tool,
    read_dataset_tool,
    read_example_tool,
)


class TestListDatasetsTool:
    """Tests for list_datasets_tool."""
    
    def test_list_datasets_success(self, mock_langsmith_client, mock_dataset):
        """Test successful list of datasets."""
        # Setup
        mock_langsmith_client.list_datasets.return_value = [mock_dataset]
        
        # Execute
        result = list_datasets_tool(mock_langsmith_client, limit=10)
        
        # Verify
        assert "datasets" in result
        assert len(result["datasets"]) == 1
        assert result["datasets"][0]["name"] == "test-dataset"
        assert "total_count" in result
        assert result["total_count"] == 1
    
    def test_list_datasets_empty(self, mock_langsmith_client):
        """Test list datasets with no results."""
        # Setup
        mock_langsmith_client.list_datasets.return_value = []
        
        # Execute
        result = list_datasets_tool(mock_langsmith_client, limit=10)
        
        # Verify
        assert "datasets" in result
        assert len(result["datasets"]) == 0
        assert result["total_count"] == 0
    
    def test_list_datasets_with_filters(self, mock_langsmith_client, mock_dataset):
        """Test list datasets with various filters."""
        # Setup
        mock_langsmith_client.list_datasets.return_value = [mock_dataset]
        
        # Execute
        result = list_datasets_tool(
            mock_langsmith_client,
            data_type="kv",
            dataset_name="test-dataset",
            limit=5
        )
        
        # Verify
        assert "datasets" in result
        assert len(result["datasets"]) == 1
        mock_langsmith_client.list_datasets.assert_called_once()
    
    def test_list_datasets_error(self, mock_langsmith_client):
        """Test list datasets with error."""
        # Setup
        mock_langsmith_client.list_datasets.side_effect = Exception("API error")
        
        # Execute
        result = list_datasets_tool(mock_langsmith_client, limit=10)
        
        # Verify
        assert "error" in result
        assert "API error" in result["error"]


class TestListExamplesTool:
    """Tests for list_examples_tool."""
    
    def test_list_examples_success(self, mock_langsmith_client, mock_example):
        """Test successful list of examples."""
        # Setup
        mock_langsmith_client.list_examples.return_value = [mock_example]
        
        # Execute
        result = list_examples_tool(
            mock_langsmith_client,
            dataset_name="test-dataset",
            limit=10
        )
        
        # Verify
        assert "examples" in result
        assert len(result["examples"]) == 1
        assert result["examples"][0]["inputs"]["question"] == "What is the capital of France?"
        assert "total_count" in result
    
    def test_list_examples_with_dataset_id(self, mock_langsmith_client, mock_example):
        """Test list examples with dataset ID."""
        # Setup
        mock_langsmith_client.list_examples.return_value = [mock_example]
        dataset_id = str(mock_example.dataset_id)
        
        # Execute
        result = list_examples_tool(
            mock_langsmith_client,
            dataset_id=dataset_id,
            limit=5
        )
        
        # Verify
        assert "examples" in result
        assert len(result["examples"]) == 1
    
    def test_list_examples_empty(self, mock_langsmith_client):
        """Test list examples with no results."""
        # Setup
        mock_langsmith_client.list_examples.return_value = []
        
        # Execute
        result = list_examples_tool(
            mock_langsmith_client,
            dataset_name="empty-dataset",
            limit=10
        )
        
        # Verify
        assert "examples" in result
        assert len(result["examples"]) == 0
    
    def test_list_examples_error(self, mock_langsmith_client):
        """Test list examples with error."""
        # Setup
        mock_langsmith_client.list_examples.side_effect = Exception("Dataset not found")
        
        # Execute
        result = list_examples_tool(
            mock_langsmith_client,
            dataset_name="nonexistent",
            limit=10
        )
        
        # Verify
        assert "error" in result
        assert "Dataset not found" in result["error"]


class TestReadDatasetTool:
    """Tests for read_dataset_tool."""
    
    def test_read_dataset_by_id(self, mock_langsmith_client, mock_dataset):
        """Test reading dataset by ID."""
        # Setup
        mock_langsmith_client.read_dataset.return_value = mock_dataset
        dataset_id = str(mock_dataset.id)
        
        # Execute
        result = read_dataset_tool(mock_langsmith_client, dataset_id=dataset_id)
        
        # Verify
        assert "dataset" in result
        assert result["dataset"]["name"] == "test-dataset"
        assert result["dataset"]["data_type"] == "kv"
    
    def test_read_dataset_by_name(self, mock_langsmith_client, mock_dataset):
        """Test reading dataset by name."""
        # Setup
        mock_langsmith_client.read_dataset.return_value = mock_dataset
        
        # Execute
        result = read_dataset_tool(
            mock_langsmith_client,
            dataset_name="test-dataset"
        )
        
        # Verify
        assert "dataset" in result
        assert result["dataset"]["name"] == "test-dataset"
    
    def test_read_dataset_error(self, mock_langsmith_client):
        """Test read dataset with error."""
        # Setup
        mock_langsmith_client.read_dataset.side_effect = Exception("Dataset not found")
        
        # Execute
        result = read_dataset_tool(
            mock_langsmith_client,
            dataset_name="nonexistent"
        )
        
        # Verify
        assert "error" in result
        assert "Dataset not found" in result["error"]


class TestReadExampleTool:
    """Tests for read_example_tool."""
    
    def test_read_example_success(self, mock_langsmith_client, mock_example):
        """Test reading example by ID."""
        # Setup
        mock_langsmith_client.read_example.return_value = mock_example
        example_id = str(mock_example.id)
        
        # Execute
        result = read_example_tool(mock_langsmith_client, example_id=example_id)
        
        # Verify
        assert "example" in result
        assert result["example"]["inputs"]["question"] == "What is the capital of France?"
        assert result["example"]["outputs"]["answer"] == "Paris"
    
    def test_read_example_with_version(self, mock_langsmith_client, mock_example):
        """Test reading example with version."""
        # Setup
        mock_langsmith_client.read_example.return_value = mock_example
        example_id = str(mock_example.id)
        
        # Execute
        result = read_example_tool(
            mock_langsmith_client,
            example_id=example_id,
            as_of="v1.0"
        )
        
        # Verify
        assert "example" in result
        assert "inputs" in result["example"]
    
    def test_read_example_error(self, mock_langsmith_client):
        """Test read example with error."""
        # Setup
        mock_langsmith_client.read_example.side_effect = Exception("Example not found")
        
        # Execute
        result = read_example_tool(
            mock_langsmith_client,
            example_id="nonexistent-id"
        )
        
        # Verify
        assert "error" in result
        assert "Example not found" in result["error"]

