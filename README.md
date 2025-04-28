# LangSmith MCP Server

![LangSmith Cursor Integration](assets/cursor_mcp.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

A production-ready [Model Context Protocol](https://modelcontextprotocol.io/introduction) (MCP) server that provides seamless integration with the [LangSmith](https://smith.langchain.com) observability platform. This server enables language models to fetch conversation history and prompts from LangSmith.

## Installation and Testing

### Prerequisites

1. Install [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/langchain-ai/langsmith-mcp-server.git
   cd langsmith-mcp
   ```

### Development Setup

1. Create a virtual environment and install dependencies:
   ```bash
   uv sync
   ```

2. View available MCP commands:
   ```bash
   uv run mcp
   ```

3. For development, run the MCP inspector:
   ```bash
   uv run mcp dev src/langsmith_mcp/server.py
   ```
   - This will start the MCP inspector on a network port
   - Install any required libraries when prompted
   - The MCP inspector will be available in your browser
   - Set the `LANGSMITH_API_KEY` environment variable in the inspector
   - Connect to the server
   - Navigate to the "Tools" tab to see all available tools

### Production Setup

#### Option 1: Using uv commands

1. Install the MCP server for Claude Desktop:
   ```bash
   uv run mcp install src/langsmith_mcp/server.py
   ```

2. Run the server:
   ```bash
   uv run mcp run src/langsmith_mcp/server.py
   ```

#### Option 2: Using absolute paths (recommended)

If you encounter any issues with the above method, you can configure the MCP server using absolute paths. Add the following configuration to your Claude Desktop settings:

```json
{
    "mcpServers": {
        "LangSmith API MCP Server": {
            "command": "/path/to/uv",
            "args": [
                "--directory",
                "/path/to/langsmith-mcp/src/langsmith_mcp",
                "run",
                "server.py"
            ],
            "env": {
                "LANGSMITH_API_KEY": "your_langsmith_api_key"
            }
        }
    }
}
```

Replace the following placeholders:
- `/path/to/uv`: The absolute path to your uv installation (e.g., `/Users/username/.local/bin/uv`). You can find it running `which uv`.
- `/path/to/langsmith-mcp`: The absolute path to your langsmith-mcp project directory
- `your_langsmith_api_key`: Your LangSmith API key

Example configuration:
```json
{
    "mcpServers": {
        "LangSmith API MCP Server": {
            "command": "/Users/mperini/.local/bin/uv",
            "args": [
                "--directory",
                "/Users/mperini/Projects/langsmith-mcp-server/src/langsmith_mcp",
                "run",
                "server.py"
            ],
            "env": {
                "LANGSMITH_API_KEY": "lsv2_pt_1234"
            }
        }
    }
}
```

Copy this configuration in Cursor > MCP Settings.

## Available Tools

The server provides the following enterprise-ready tools:

- `get_thread_history(thread_id: str, project_name: str)`: Fetch conversation history for a specific thread
- `get_prompts(query: Optional[str], is_public: Optional[bool])`: Fetch prompts from LangSmith with optional filtering
- `pull_prompt(prompt_name: str)`: Pull a specific prompt by its name, including its template and metadata

## Example Use Cases

The server enables conversation history retrieval and prompt management such as:

- "Fetch the history of my conversation with the AI assistant from thread 'thread-123' in project 'my-chatbot'"
- "Get all public prompts in my workspace"
- "Find private prompts containing the word 'joke'"
- "Pull the template for the 'legal-case-summarizer' prompt"
- "Get the system message from a specific prompt template"

## Error Handling

The server implements robust error handling with detailed, actionable error messages for:

- API authentication issues
- Invalid thread or project IDs
- Invalid prompt names
- Network connectivity failures
- Rate limiting and quota management

## License

This project is distributed under the MIT License. For detailed terms and conditions, please refer to the LICENSE file.


Made with ❤️ by [LangChain](https://langchain.com) Team