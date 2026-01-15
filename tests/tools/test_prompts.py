"""Tests for prompt tools."""

import pytest
from unittest.mock import Mock, patch
from langsmith_mcp_server.services.tools.prompts import (
    list_prompts_tool,
    get_prompt_tool,
)


class TestListPromptsTool:
    """Tests for list_prompts_tool."""
    
    def test_list_prompts_success(self, mock_langsmith_client, mock_prompt):
        """Test successful list of prompts."""
        # Setup
        mock_langsmith_client.list_prompts.return_value = [mock_prompt]
        
        # Execute
        result = list_prompts_tool(mock_langsmith_client, is_public=False, limit=10)
        
        # Verify
        assert "prompts" in result
        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["name"] == "test-prompt"
        mock_langsmith_client.list_prompts.assert_called_once()
    
    def test_list_prompts_empty(self, mock_langsmith_client):
        """Test list prompts with no results."""
        # Setup
        mock_langsmith_client.list_prompts.return_value = []
        
        # Execute
        result = list_prompts_tool(mock_langsmith_client, is_public=True, limit=20)
        
        # Verify
        assert "prompts" in result
        assert len(result["prompts"]) == 0
    
    def test_list_prompts_error(self, mock_langsmith_client):
        """Test list prompts with error."""
        # Setup
        mock_langsmith_client.list_prompts.side_effect = Exception("API error")
        
        # Execute
        result = list_prompts_tool(mock_langsmith_client, is_public=False, limit=10)
        
        # Verify
        assert "error" in result
        assert "API error" in result["error"]
    
    def test_list_prompts_with_multiple_prompts(self, mock_langsmith_client, mock_prompt):
        """Test list prompts with multiple results."""
        # Setup
        prompt1 = Mock()
        prompt1.name = "prompt-1"
        prompt1.id = "id-1"
        prompt1.is_public = False
        prompt1.description = "Description 1"
        prompt1.created_at = mock_prompt.created_at
        prompt1.updated_at = mock_prompt.updated_at
        prompt1.prompt = {"messages": []}
        
        prompt2 = Mock()
        prompt2.name = "prompt-2"
        prompt2.id = "id-2"
        prompt2.is_public = True
        prompt2.description = "Description 2"
        prompt2.created_at = mock_prompt.created_at
        prompt2.updated_at = mock_prompt.updated_at
        prompt2.prompt = {"messages": []}
        
        mock_langsmith_client.list_prompts.return_value = [prompt1, prompt2]
        
        # Execute
        result = list_prompts_tool(mock_langsmith_client, is_public=False, limit=10)
        
        # Verify
        assert len(result["prompts"]) == 2
        assert result["prompts"][0]["name"] == "prompt-1"
        assert result["prompts"][1]["name"] == "prompt-2"


class TestGetPromptTool:
    """Tests for get_prompt_tool."""
    
    def test_get_prompt_success(self, mock_langsmith_client, mock_prompt):
        """Test successful get prompt by name."""
        # Setup
        mock_langsmith_client.pull_prompt.return_value = mock_prompt
        
        # Execute
        result = get_prompt_tool(mock_langsmith_client, prompt_name="test-prompt")
        
        # Verify
        assert "prompt" in result
        assert result["prompt"]["name"] == "test-prompt"
        mock_langsmith_client.pull_prompt.assert_called_once_with("test-prompt")
    
    def test_get_prompt_not_found(self, mock_langsmith_client):
        """Test get prompt when prompt doesn't exist."""
        # Setup
        mock_langsmith_client.pull_prompt.side_effect = Exception("Prompt not found")
        
        # Execute
        result = get_prompt_tool(mock_langsmith_client, prompt_name="nonexistent")
        
        # Verify
        assert "error" in result
        assert "Prompt not found" in result["error"]
    
    def test_get_prompt_with_template(self, mock_langsmith_client, mock_prompt):
        """Test get prompt includes template information."""
        # Setup
        mock_prompt.prompt = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "{input}"}
            ]
        }
        mock_langsmith_client.pull_prompt.return_value = mock_prompt
        
        # Execute
        result = get_prompt_tool(mock_langsmith_client, prompt_name="test-prompt")
        
        # Verify
        assert "prompt" in result
        assert "prompt" in result["prompt"]
        assert len(result["prompt"]["prompt"]["messages"]) == 2

