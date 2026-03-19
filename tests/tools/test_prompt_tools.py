"""Tests for prompt tools, focusing on correct serialisation of LangChain prompt objects."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.load import dumpd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate

from langsmith_mcp_server.services.tools.prompts import get_prompt_tool


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


@pytest.fixture
def chat_prompt_template() -> ChatPromptTemplate:
    """A ChatPromptTemplate with a SystemMessagePromptTemplate and a human turn."""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )


class TestGetPromptToolLcSerializable:
    """Tests that LangChain objects (lc_serializable=True) are serialised with dumpd()."""

    def test_chat_prompt_template_uses_dumpd(
        self, mock_client: Mock, chat_prompt_template: ChatPromptTemplate
    ) -> None:
        """get_prompt_tool must use dumpd() for ChatPromptTemplate, not model_dump()."""
        mock_client.pull_prompt.return_value = chat_prompt_template

        result = get_prompt_tool(mock_client, prompt_name="owner/my-prompt")

        assert result == dumpd(chat_prompt_template)

    def test_chat_prompt_template_messages_not_empty(
        self, mock_client: Mock, chat_prompt_template: ChatPromptTemplate
    ) -> None:
        """The serialised output must preserve nested message template content.

        model_dump() would return empty dicts for SystemMessagePromptTemplate /
        HumanMessagePromptTemplate, so this test catches regressions where the
        wrong serialiser is used.
        """
        mock_client.pull_prompt.return_value = chat_prompt_template

        result = get_prompt_tool(mock_client, prompt_name="owner/my-prompt")

        assert "kwargs" in result, "dumpd() output should contain 'kwargs' key"
        messages = result["kwargs"].get("messages", [])
        assert len(messages) == 2, "Both message templates should be serialised"

        # Each message entry should carry its full prompt template, not an empty dict
        for msg in messages:
            msg_kwargs = msg.get("kwargs", {})
            prompt = msg_kwargs.get("prompt", {})
            assert prompt, f"Message template content should not be empty, got: {msg}"

    def test_system_message_template_template_text_preserved(
        self, mock_client: Mock, chat_prompt_template: ChatPromptTemplate
    ) -> None:
        """The system message template text must survive serialisation."""
        mock_client.pull_prompt.return_value = chat_prompt_template

        result = get_prompt_tool(mock_client, prompt_name="owner/my-prompt")

        messages = result["kwargs"]["messages"]
        system_msg = messages[0]
        system_template_text = system_msg["kwargs"]["prompt"]["kwargs"]["template"]
        assert system_template_text == "You are a helpful assistant."

    def test_dumpd_called_not_model_dump_for_lc_serializable(
        self, mock_client: Mock, chat_prompt_template: ChatPromptTemplate
    ) -> None:
        """model_dump() must NOT be called when lc_serializable is True."""
        mock_client.pull_prompt.return_value = chat_prompt_template

        with patch("langsmith_mcp_server.services.tools.prompts.dumpd", wraps=dumpd) as mock_dumpd:
            get_prompt_tool(mock_client, prompt_name="owner/my-prompt")

        mock_dumpd.assert_called_once_with(chat_prompt_template)


class TestGetPromptToolNonLcSerializable:
    """Tests that non-LangChain objects fall back to model_dump() / dict()."""

    def test_pydantic_model_uses_model_dump(self, mock_client: Mock) -> None:
        """A plain Pydantic v2 object without lc_serializable uses model_dump()."""
        from pydantic import BaseModel

        class SimplePydanticPrompt(BaseModel):
            template: str
            input_variables: list[str]

        plain_prompt = SimplePydanticPrompt(template="Hello {name}", input_variables=["name"])
        assert not getattr(plain_prompt, "lc_serializable", False)

        mock_client.pull_prompt.return_value = plain_prompt
        result = get_prompt_tool(mock_client, prompt_name="owner/simple-prompt")

        assert result == {"template": "Hello {name}", "input_variables": ["name"]}

    def test_missing_prompt_name_and_id_returns_error(self, mock_client: Mock) -> None:
        """Calling without prompt_name or prompt_id must return an error dict."""
        result = get_prompt_tool(mock_client)
        assert "error" in result

    def test_client_exception_returns_error(self, mock_client: Mock) -> None:
        """An exception raised by the client must be caught and returned as an error dict."""
        mock_client.pull_prompt.side_effect = Exception("network failure")
        result = get_prompt_tool(mock_client, prompt_name="owner/bad-prompt")
        assert "error" in result
        assert "network failure" in result["error"]
