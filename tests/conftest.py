"""Shared test fixtures for langsmith-mcp-server tests."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import uuid4


@pytest.fixture
def mock_langsmith_client():
    """Create a mock LangSmith client for testing."""
    client = Mock()
    
    # Configure default return values
    client.list_prompts = Mock(return_value=[])
    client.list_datasets = Mock(return_value=[])
    client.list_examples = Mock(return_value=[])
    client.list_runs = Mock(return_value=[])
    client.list_projects = Mock(return_value=[])
    client.read_dataset = Mock(return_value=None)
    client.read_example = Mock(return_value=None)
    client.get_run_stats = Mock(return_value={})
    
    return client


@pytest.fixture
def mock_prompt():
    """Create a mock prompt object."""
    prompt = Mock()
    prompt.name = "test-prompt"
    prompt.id = str(uuid4())
    prompt.is_public = False
    prompt.description = "Test prompt description"
    prompt.created_at = datetime.now()
    prompt.updated_at = datetime.now()
    prompt.prompt = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{input}"}
        ]
    }
    return prompt


@pytest.fixture
def mock_dataset():
    """Create a mock dataset object."""
    dataset = Mock()
    dataset.id = uuid4()
    dataset.name = "test-dataset"
    dataset.description = "Test dataset description"
    dataset.data_type = "kv"
    dataset.example_count = 10
    dataset.session_count = 5
    dataset.created_at = datetime.now()
    dataset.modified_at = datetime.now()
    dataset.last_session_start_time = None
    dataset.inputs_schema_definition = None
    dataset.outputs_schema_definition = None
    return dataset


@pytest.fixture
def mock_example():
    """Create a mock example object."""
    example = Mock()
    example.id = uuid4()
    example.dataset_id = uuid4()
    example.inputs = {"question": "What is the capital of France?"}
    example.outputs = {"answer": "Paris"}
    example.metadata = {"source": "test"}
    example.created_at = datetime.now()
    example.modified_at = datetime.now()
    example.runs = []
    example.source_run_id = None
    example.attachments = None
    return example


@pytest.fixture
def mock_run():
    """Create a mock run object."""
    run = Mock()
    run.id = uuid4()
    run.name = "test-run"
    run.run_type = "chain"
    run.start_time = datetime.now()
    run.end_time = datetime.now()
    run.inputs = {"input": "test input"}
    run.outputs = {"output": "test output"}
    run.error = None
    run.total_tokens = 100
    run.total_cost = 0.001
    run.feedback_stats = {}
    run.app_path = "/test"
    run.thread_id = str(uuid4())
    run.trace_id = str(uuid4())
    run.metadata = {}
    run.tags = []
    
    # Add dict method for serialization
    def dict_method():
        return {
            "id": run.id,
            "name": run.name,
            "run_type": run.run_type,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "error": run.error,
            "total_tokens": run.total_tokens,
            "total_cost": run.total_cost,
            "feedback_stats": run.feedback_stats,
            "app_path": run.app_path,
            "thread_id": run.thread_id,
            "trace_id": run.trace_id,
            "metadata": run.metadata,
            "tags": run.tags,
        }
    
    run.dict = dict_method
    return run


@pytest.fixture
def mock_project():
    """Create a mock project object."""
    project = Mock()
    project.id = uuid4()
    project.name = "test-project"
    project.description = "Test project description"
    project.created_at = datetime.now()
    project.metadata = {}
    
    # Add dict method for serialization
    def dict_method():
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "metadata": project.metadata,
        }
    
    project.dict = dict_method
    return project


@pytest.fixture
def mock_context():
    """Create a mock FastMCP context."""
    context = Mock()
    
    # Mock state storage
    context._state = {}
    
    def get_state(key: str, default: Any = None) -> Any:
        return context._state.get(key, default)
    
    def set_state(key: str, value: Any) -> None:
        context._state[key] = value
    
    context.get_state = get_state
    context.set_state = set_state
    context.get_http_request = Mock(return_value=None)
    
    return context


@pytest.fixture
def sample_api_key():
    """Provide a sample API key for testing."""
    return "lsv2_pt_test_api_key_12345"


@pytest.fixture
def sample_workspace_id():
    """Provide a sample workspace ID for testing."""
    return str(uuid4())


@pytest.fixture
def mock_http_request(sample_api_key, sample_workspace_id):
    """Create a mock HTTP request with headers."""
    request = Mock()
    request.headers = {
        "LANGSMITH-API-KEY": sample_api_key,
        "LANGSMITH-WORKSPACE-ID": sample_workspace_id,
        "LANGSMITH-ENDPOINT": "https://api.smith.langchain.com",
    }
    request.state = Mock()
    request.state.api_key = sample_api_key
    request.state.workspace_id = sample_workspace_id
    request.state.endpoint = "https://api.smith.langchain.com"
    request.url = Mock()
    request.url.path = "/mcp"
    return request

