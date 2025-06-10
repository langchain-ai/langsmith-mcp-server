# 🦜🛠️ LangSmith MCP Server

> [!WARNING]
> LangSmith MCP Server is under active development and many features are not yet implemented.


![LangSmith MCP Hero](https://raw.githubusercontent.com/langchain-ai/langsmith-mcp-server/refs/heads/main/docs/assets/langsmith_mcp_hero.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

A production-ready [Model Context Protocol](https://modelcontextprotocol.io/introduction) (MCP) server that provides seamless integration with the [LangSmith](https://smith.langchain.com) observability platform. This server enables language models to fetch conversation history and prompts from LangSmith.

## 📋 Overview

The LangSmith MCP Server bridges the gap between language models and the LangSmith platform, enabling advanced capabilities for conversation tracking, prompt management, and analytics integration.

## 🛠️ Installation Options

### 📝 General Prerequisites

1. Install [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/langchain-ai/langsmith-mcp-server.git
   cd langsmith-mcp-server
   ```

### 🔌 MCP Client Integration

Once you have the LangSmith MCP Server, you can integrate it with various MCP-compatible clients. You have two installation options:

#### 📦 From PyPI

1. Install the package:
   ```bash
   uv run pip install --upgrade langsmith-mcp-server
   ```

2. Add to your client MCP config:
   ```json
   {
       "mcpServers": {
           "LangSmith API MCP Server": {
               "command": "/path/to/uvx",
               "args": [
                   "langsmith-mcp-server"
               ],
               "env": {
                   "LANGSMITH_API_KEY": "your_langsmith_api_key"
               }
           }
       }
   }
   ```

#### ⚙️ From Source

Add the following configuration to your MCP client settings:

```json
{
    "mcpServers": {
        "LangSmith API MCP Server": {
            "command": "/path/to/uvx",
            "args": [
                "--directory",
                "/path/to/langsmith-mcp-server/langsmith_mcp_server",
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
- `/path/to/langsmith-mcp-server`: The absolute path to your langsmith-mcp project directory
- `your_langsmith_api_key`: Your LangSmith API key

Example configuration:
```json
{
    "mcpServers": {
        "LangSmith API MCP Server": {
            "command": "/Users/mperini/.local/bin/uvx",
            "args": [
                "langsmith-mcp-server"
            ],
            "env": {
                "LANGSMITH_API_KEY": "lsv2_pt_1234"
            }
        }
    }
}
```

Copy this configuration in Cursor > MCP Settings.

![LangSmith Cursor Integration](docs/assets/cursor_mcp.png)

## 🧪 Development and Contributing 🤝

If you want to develop or contribute to the LangSmith MCP Server, follow these steps:

1. Create a virtual environment and install dependencies:
   ```bash
   uv sync
   ```

2. To include test dependencies:
   ```bash
   uv sync --group test
   ```

3. View available MCP commands:
   ```bash
   uvx langsmith-mcp-server
   ```

4. For development, run the MCP inspector:
   ```bash
   uv run mcp dev langsmith_mcp_server/server.py
   ```
   - This will start the MCP inspector on a network port
   - Install any required libraries when prompted
   - The MCP inspector will be available in your browser
   - Set the `LANGSMITH_API_KEY` environment variable in the inspector
   - Connect to the server
   - Navigate to the "Tools" tab to see all available tools

5. Before submitting your changes, run the linting and formatting checks:
   ```bash
   make lint
   make format
   ```

## 🚀 Example Use Cases

The server enables powerful capabilities including:

- 💬 **Conversation History**: "Fetch the history of my conversation with the AI assistant from thread 'thread-123' in project 'my-chatbot'"
- 📚 **Prompt Management**: "Get all public prompts in my workspace"
- 🔍 **Smart Search**: "Find private prompts containing the word 'joke'"
- 📝 **Template Access**: "Pull the template for the 'legal-case-summarizer' prompt"
- 🔧 **Configuration**: "Get the system message from a specific prompt template"

## 📄 License

This project is distributed under the MIT License. For detailed terms and conditions, please refer to the LICENSE file.


Made with ❤️ by the [LangChain](https://langchain.com) Team
