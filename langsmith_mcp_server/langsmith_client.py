"""
Client for interacting with the LangSmith API.
Provides low-level API operations that can be used by higher-level tools.
"""

import os

from langsmith import Client


class LangSmithClient:
    """Client for interacting with the LangSmith API."""

    def __init__(self, api_key: str):
        """
        Initialize the LangSmith API client.

        Args:
            api_key: API key for LangSmith API
        """
        self.api_key = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key
        self.langsmith_client = Client()

    def get_client(self) -> Client:
        """Get the underlying LangSmith client."""
        return self.langsmith_client
