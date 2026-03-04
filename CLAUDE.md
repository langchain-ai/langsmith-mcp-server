# LangSmith MCP Server - Technical Reference & Development Guide

## ⚠️ IMPORTANT: Keeping This Document Updated
**When you make changes to this codebase, please update this CLAUDE.md file if:**
- You add or remove MCP tools
- You modify the architecture or add new modules
- You change configuration requirements or setup procedures
- You add new dependencies or change build processes
- You discover important patterns or best practices

This document serves as the primary context for AI assistants working on this project. Keeping it accurate ensures efficient development.

---

## Project Overview

The LangSmith MCP Server is a production-ready Model Context Protocol (MCP) server that provides seamless integration with the LangSmith observability platform. This server enables AI language models to programmatically interact with LangSmith's suite of tools for conversation tracking, prompt management, dataset operations, and trace analytics.

### What is LangSmith?
LangSmith is LangChain's comprehensive observability and evaluation platform for LLM applications, providing:
- Conversation thread tracking and history
- Prompt template management and versioning
- Dataset creation and management for testing
- Trace collection and analysis for debugging
- Performance analytics and monitoring
- Billing and usage tracking

### What is MCP?
The Model Context Protocol (MCP) is an open standard that enables secure connections between AI assistants and external data sources and tools. This server implements the MCP specification to bridge AI models with LangSmith's capabilities.

---

## Architecture Deep Dive

### Core Components

#### 1. Server Architecture ([server.py](langsmith_mcp_server/server.py))
The server is built on FastMCP, a high-performance MCP framework that provides:
- **Transport Layer**: Uses stdio transport for CLI/desktop integration, HTTP for web-based access
- **Tool Registration**: Modular system for registering LangSmith tools via `register_tools()`
- **Middleware Stack**: API key authentication and CORS handling
- **Error Handling**: Comprehensive exception handling across all operations

```python
# Core server initialization (actual implementation)
mcp = FastMCP("LangSmith API MCP Server")

# Modular registration system (no client parameter - uses middleware)
register_tools(mcp)
register_prompts(mcp)      # Currently empty stub
register_resources(mcp)    # Currently empty stub
```

**Key Architecture Notes:**
- API keys are handled per-request via [middleware.py](langsmith_mcp_server/middleware.py), not at server initialization
- The server supports both stdio (CLI) and HTTP transports
- HTTP mode runs via uvicorn and is configured in the Dockerfile
- **Client disconnect patch**: [streamable_http_patch.py](langsmith_mcp_server/streamable_http_patch.py) patches the MCP streamable HTTP transport so that when the client (or proxy) closes the connection early, the server logs and continues instead of crashing with `ClosedResourceError` / `BrokenResourceError` (see [Troubleshooting](#7-http-streamable-client-disconnect-crashes)).

#### 2. Authentication & Middleware ([middleware.py](langsmith_mcp_server/middleware.py))
Request-scoped authentication system that:
- Extracts `LANGSMITH-API-KEY` header from requests
- Supports optional `LANGSMITH-WORKSPACE-ID` and `LANGSMITH-ENDPOINT` headers
- Stores credentials in context variables for access by tools
- Returns 401 for missing API keys (except /health endpoint)
- Automatically cleans up context after each request

```python
# Context variables used throughout the application
api_key_context: ContextVar[str]
workspace_id_context: ContextVar[str]
endpoint_context: ContextVar[str]
```

#### 3. Client Management ([common/helpers.py](langsmith_mcp_server/common/helpers.py))
Helper functions for LangSmith client creation and management:
- `get_langsmith_client_from_api_key()`: Creates Client instances from API keys (no `os.environ` mutation; credentials passed explicitly to avoid cross-request races)
- `get_client_from_context()`: Retrieves client from FastMCP context; reuses one Client per request via request-scoped state to reduce memory churn
- `get_api_key_and_endpoint_from_context()`: Extracts credentials from context
- No custom wrapper class - uses standard `langsmith.Client` directly

#### 4. Service Layer Architecture

**Tools ([services/tools/](langsmith_mcp_server/services/tools/))**
- [datasets.py](langsmith_mcp_server/services/tools/datasets.py): Dataset CRUD operations and example management
- [prompts.py](langsmith_mcp_server/services/tools/prompts.py): Prompt discovery, retrieval, and template access
- [traces.py](langsmith_mcp_server/services/tools/traces.py): Conversation history, run fetching, thread management
- [experiments.py](langsmith_mcp_server/services/tools/experiments.py): Experiment listing and management
- [usage.py](langsmith_mcp_server/services/tools/usage.py): Billing and usage tracking (trace counts)
- [workspaces.py](langsmith_mcp_server/services/tools/workspaces.py): **Empty stub** (future workspace management)

**Prompts ([services/prompts/](langsmith_mcp_server/services/prompts/))**
- Currently empty - only contains `__init__.py`
- [register_prompts.py](langsmith_mcp_server/services/register_prompts.py) is a stub (just `pass`)

**Resources ([services/resources/](langsmith_mcp_server/services/resources/))**
- Currently empty - only contains `__init__.py`
- [register_resources.py](langsmith_mcp_server/services/register_resources.py) is a stub (just `pass`)

**Common Utilities ([common/](langsmith_mcp_server/common/))**
- [helpers.py](langsmith_mcp_server/common/helpers.py): Client creation, UUID conversion, dictionary utilities
- [pagination.py](langsmith_mcp_server/common/pagination.py): Run and message pagination logic
- [formatters.py](langsmith_mcp_server/common/formatters.py): Output formatting utilities

---

## Available MCP Tools

All tools are registered in [register_tools.py](langsmith_mcp_server/services/register_tools.py). Here's the complete list:

### Prompt Management
- `list_prompts(is_public: str, limit: int)` - Fetch prompts with filtering
- `get_prompt_by_name(prompt_name: str)` - Get specific prompt by name
- `push_prompt()` - **Documentation tool** explaining how to push prompts

### Conversation & Thread Management
- `get_thread_history(thread_id: str, project_name: str, limit: int, offset: int)` - Retrieve conversation history with pagination
- `fetch_runs(project_name: str, limit: int, page_number: int, ...)` - Fetch runs with extensive filtering and automatic character-based pagination

### Project Management
- `list_projects(limit: int, project_name: str, ...)` - List and search projects

### Billing & Usage
- `get_billing_usage(starting_on: str, ending_before: str, workspace_id: str)` - Get billing usage and trace counts

### Experiments
- `list_experiments(reference_dataset_id: str, reference_dataset_name: str, ...)` - List experiments with filtering
- `run_experiment()` - **Documentation tool** explaining how to run experiments

### Dataset Operations
- `list_datasets(dataset_ids: str, data_type: str, dataset_name: str, ...)` - List datasets with filtering
- `list_examples(dataset_id: str, dataset_name: str, ...)` - List examples from datasets
- `read_dataset(dataset_id: str, dataset_name: str)` - Read complete dataset
- `read_example(example_id: str, as_of: str)` - Read specific example
- `create_dataset()` - **Documentation tool** explaining how to create datasets
- `update_examples()` - **Documentation tool** explaining how to update examples

### Important Notes on Tools:
- **Documentation tools** (`push_prompt`, `create_dataset`, `update_examples`, `run_experiment`) don't perform actions - they return instructions
- **All results are paginated**: `fetch_runs` always returns paginated responses with `page_number`, `total_pages`, and character budget controls
- Several tool implementations exist in [traces.py](langsmith_mcp_server/services/tools/traces.py) but are **NOT registered**:
  - `fetch_trace_tool()` - exists but not exposed as MCP tool
  - `get_project_runs_stats_tool()` - exists but not exposed as MCP tool
- All tools use `ctx: Context` parameter to access API keys from middleware

---

## Development Environment Setup

### Prerequisites
- **Python**: 3.10+ (type-checked and tested)
- **uv**: Fast Python package manager and resolver ([install guide](https://docs.astral.sh/uv/))
- **LangSmith API Key**: Get from [smith.langchain.com](https://smith.langchain.com)

### Installation

#### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/langchain-ai/langsmith-mcp-server.git
cd langsmith-mcp-server

# Create isolated environment with all dependencies
uv sync

# Include test dependencies for development
uv sync --group test

# Verify installation
uvx langsmith-mcp-server
```

#### Production Deployment
```bash
# Install from PyPI
uv pip install langsmith-mcp-server

# Or use uvx to run directly
uvx langsmith-mcp-server
```

---

## Development Best Practices

### 🚨 CRITICAL: Always Run Lint and Format Before Committing

**Every time you make code changes, you MUST run:**

```bash
make format    # Auto-format code with ruff
make lint      # Check code style and find issues
```

**Why this matters:**
- CI will fail if code doesn't pass linting
- Maintains consistent code style across the project
- Catches common bugs and issues early
- ruff is configured in [pyproject.toml](pyproject.toml) with project-specific rules

### Development Workflow

1. **Make your changes** to the code
2. **Run format and lint** (always!)
   ```bash
   make format
   make lint
   ```
3. **Run tests** to ensure nothing broke
   ```bash
   make test
   ```
4. **Test with MCP Inspector** (for tool changes)
   ```bash
   uv run mcp dev langsmith_mcp_server/server.py
   ```
5. **Update CLAUDE.md** if you added features or changed architecture
6. **Commit your changes**

### Testing Guidelines

```bash
# Run all tests
make test

# Run specific test file
make test TEST_FILE=tests/tools/test_dataset_tools.py

# Run tests in watch mode (continuous testing)
make test_watch

# Type checking
uv run mypy langsmith_mcp_server/
```

### Code Quality Standards

- **Type Hints**: All functions should have type annotations
- **Docstrings**: Public functions need docstrings explaining purpose, args, and returns
- **Error Handling**: Always return `{"error": str(e)}` rather than raising exceptions in tools
- **Line Length**: 100 characters (configured in ruff)
- **Import Sorting**: Automatic via ruff's isort integration

---

## Development Tools & Workflows

### MCP Inspector (Interactive Testing)
```bash
# Start development server with browser-based inspector
uv run mcp dev langsmith_mcp_server/server.py
```

Features:
- **Real-time Testing**: Call tools interactively through web UI
- **Environment Configuration**: Set `LANGSMITH_API_KEY` and test different configs
- **Tool Discovery**: Browse all registered tools and their schemas
- **Request/Response Debugging**: See exact inputs/outputs for each call
- **Great for**: Testing new tools, debugging issues, exploring functionality

### Configuration Files

#### [pyproject.toml](pyproject.toml)
- **Build System**: pdm-backend for modern Python packaging
- **Dependencies**: fastmcp, langsmith, langchain-core, uvicorn
- **Test Dependencies**: pytest, ruff, mypy, pytest-asyncio, pytest-socket
- **Entry Point**: `langsmith-mcp-server` command maps to `server:main()`
- **Ruff Config**: Line length 100, Python 3.10+, specific lint rules
- **Pytest Config**: Async support, socket restrictions, verbose output

#### [Makefile](Makefile)
Provides convenient commands:
- `make lint` - Check code style (ruff format --diff + ruff check --diff)
- `make format` - Auto-format code (ruff check --fix + ruff format)
- `make test` - Run pytest with socket restrictions and 10s timeout
- `make test_watch` - Continuous testing during development

#### [Dockerfile](Dockerfile)
- **Base**: Python 3.12 Alpine (minimal footprint)
- **Build Deps**: gcc, musl-dev, libffi-dev for compiled extensions
- **Package Manager**: Uses uv for fast installation
- **Security**: Runs as non-root user (uid 1000)
- **Entry Point**: `uvicorn langsmith_mcp_server.server:app` on port 8000 (HTTP mode)
- **Note**: Dockerfile uses HTTP transport, not stdio

---

## Integration Patterns

### MCP Client Configuration

#### Cursor IDE Integration
Add to `.cursorrules` or MCP settings:
```json
{
    "mcpServers": {
        "LangSmith API MCP Server": {
            "command": "uvx",
            "args": ["langsmith-mcp-server"],
            "env": {
                "LANGSMITH_API_KEY": "lsv2_pt_your_key_here"
            }
        }
    }
}
```

#### Claude Desktop Integration
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
    "mcpServers": {
        "LangSmith API MCP Server": {
            "command": "uvx",
            "args": ["langsmith-mcp-server"],
            "env": {
                "LANGSMITH_API_KEY": "lsv2_pt_your_key_here"
            }
        }
    }
}
```

#### Source-based Development Integration
For local development with your working copy:
```json
{
    "command": "uv",
    "args": [
        "--directory", "/absolute/path/to/langsmith-mcp-server",
        "run", "langsmith-mcp-server"
    ],
    "env": {
        "LANGSMITH_API_KEY": "lsv2_pt_your_key_here"
    }
}
```

---

## Common Use Cases & Examples

### 1. Conversation History Retrieval
```python
# Get conversation thread history
result = get_thread_history(
    thread_id="thread-abc123",
    project_name="my-chatbot-project",
    limit=50,
    offset=0
)
```

### 2. Prompt Library Management
```python
# List available prompts
prompts = list_prompts(is_public="true", limit=20)

# Get specific prompt
template = get_prompt_by_name("customer-support-agent")
```

### 3. Dataset Operations
```python
# List datasets
datasets = list_datasets(data_type="chat", limit=10)

# Get examples from a dataset
examples = list_examples(dataset_name="test-cases", limit=100)

# Read specific example
example = read_example(example_id="example-uuid")
```

### 4. Project and Run Management
```python
# List projects
projects = list_projects(limit=10, project_name="production")

# Fetch runs with filtering (always paginated)
result = fetch_runs(
    project_name="my-app",
    limit=20,
    page_number=1,
    run_type="chain",
    is_root="true"
)
# result includes: runs, page_number, total_pages, max_chars_per_page, preview_chars

# Paginate through all pages
for page in range(2, result["total_pages"] + 1):
    fetch_runs(project_name="my-app", limit=20, page_number=page)
```

### 5. Usage Tracking
```python
# Get billing usage
usage = get_billing_usage(
    starting_on="2025-01-01",
    ending_before="2025-02-01",
    workspace_id="optional-workspace-id"
)
```

---

## Error Handling & Reliability

### Exception Management
- **All tools return dictionaries**: Never raise exceptions to MCP clients
- **Error format**: `{"error": "description of error"}`
- **Client validation**: API key checked by middleware before tool execution
- **Input sanitization**: Parameters validated and type-coerced where appropriate
- **Graceful degradation**: Tools return partial results when possible

### Security Considerations
- **API Key Protection**: Keys passed via headers (HTTP) or environment (stdio), never logged
- **Request Isolation**: Each request gets its own context, no cross-contamination
- **Input Validation**: All user inputs sanitized before passing to LangSmith SDK
- **Network Security**: Tests run with `--disable-socket` to prevent accidental external calls
- **Minimal Attack Surface**: Non-root Docker user, minimal dependencies

### Performance Optimizations
- **Lazy Client Creation**: LangSmith client only created when needed per request
- **Request-Scoped Client Reuse**: One Client per request (cached in FastMCP request state) to avoid creating many clients under load and reduce memory churn
- **No os.environ Mutation**: Credentials are passed explicitly to the Client constructor; we do not set `LANGSMITH_API_KEY` etc. in the process environment, avoiding concurrent-request races that can cause 403 Forbidden
- **Pagination Support**: Large result sets can be paginated (runs, messages, examples)
- **Efficient Filtering**: Most tools support server-side filtering to reduce data transfer
- **Connection Reuse**: HTTP client connection pooling via httpx (used by langsmith SDK)

---

## Project Structure Reference

```
langsmith-mcp-server/
├── langsmith_mcp_server/
│   ├── server.py                 # Main MCP server, FastMCP app, HTTP/stdio setup
│   ├── streamable_http_patch.py  # Client-disconnect handling for HTTP streamable transport
│   ├── middleware.py            # API key authentication middleware
│   ├── common/
│   │   ├── helpers.py            # Client creation, UUID conversion, utilities
│   │   ├── pagination.py         # Pagination logic for runs and messages
│   │   └── formatters.py         # Output formatting
│   └── services/
│       ├── register_tools.py     # Tool registration (all @mcp.tool() definitions)
│       ├── register_prompts.py   # Stub (empty)
│       ├── register_resources.py # Stub (empty)
│       ├── tools/
│       │   ├── datasets.py       # Dataset and example operations
│       │   ├── prompts.py        # Prompt management
│       │   ├── traces.py         # Thread history, runs, traces
│       │   ├── experiments.py    # Experiment listing
│       │   ├── usage.py          # Billing/usage tracking
│       │   └── workspaces.py     # Stub (empty)
│       ├── prompts/              # Empty (future: MCP prompts)
│       └── resources/            # Empty (future: MCP resources)
├── tests/
│   ├── tools/
│   │   └── test_dataset_tools.py
│   └── test_mock.py
├── examples/                     # Example scripts showing tool usage
│   ├── fetch_runs.py
│   ├── get_thread_history.py
│   ├── fetch_trace.py
│   ├── get_run_stats.py
│   ├── datasets_annotation.py
│   └── ...
├── pyproject.toml               # Project config, dependencies, tool settings
├── Makefile                     # Development commands (lint, format, test)
├── Dockerfile                   # HTTP server deployment
└── CLAUDE.md                    # This file - keep it updated!
```

---

## Troubleshooting Guide

### Common Issues

1. **API Key Problems**
   - Error: "Missing LANGSMITH-API-KEY header" (HTTP) or client initialization failures (stdio)
   - Solution: Ensure `LANGSMITH_API_KEY` is set in environment or headers
   - Verify key format: should start with `lsv2_pt_`

2. **Import Errors**
   - Error: Module not found
   - Solution: Run `uv sync` to install dependencies
   - For tests: `uv sync --group test`

3. **Linting Failures**
   - Error: CI fails on PR, ruff errors
   - Solution: Always run `make format && make lint` before committing
   - Auto-fix most issues: `make format`

4. **Test Failures**
   - Error: `pytest.PytestUnraisableExceptionWarning` or socket errors
   - Solution: Tests use `--disable-socket` to prevent external calls
   - Mock external dependencies in tests

5. **MCP Inspector Not Working**
   - Error: Can't connect or tools not showing
   - Solution: Ensure running `uv run mcp dev langsmith_mcp_server/server.py`
   - Check that port isn't already in use
   - Set `LANGSMITH_API_KEY` in inspector UI after starting

6. **403 Forbidden or memory limit under load (e.g. hosted Render)**
   - Error: `403 Client Error: Forbidden` on valid API key, or instance hits memory limit after traffic spikes
   - Cause: Previously the server mutated `os.environ` with the request's API key, so concurrent requests could overwrite each other's credentials (leading to 403). Creating a new LangSmith Client on every tool call also increased memory use under load.
   - Solution: The codebase now (1) does not set `os.environ` for LangSmith; credentials are passed explicitly to the Client, and (2) reuses one Client per request via FastMCP request-scoped state. If you run an older version, upgrade. For hosted deployments, consider memory limits and request timeouts (see item 7).

7. **HTTP streamable client disconnect crashes**
   - Error: `anyio.BrokenResourceError` / `anyio.ClosedResourceError`, "SSE response error", or `ExceptionGroup` in `streamable_http._handle_post_request`
   - Cause: Client or proxy (e.g. load balancer) closed the connection before the server finished sending.
   - Solution: The server applies a patch at startup ([streamable_http_patch.py](langsmith_mcp_server/streamable_http_patch.py)) to catch these and log at debug level instead of crashing. If you still see crashes, ensure the patch is applied (HTTP app is created and patch runs). On Render or behind a proxy, increase request/response timeouts (e.g. 60s+ for `/mcp`) so the connection is not closed while the server is still working.
   - Upstream: [python-sdk#2064](https://github.com/modelcontextprotocol/python-sdk/issues/2064), [PR #2072](https://github.com/modelcontextprotocol/python-sdk/pull/2072). When a fix is released, upgrading `mcp`/`fastmcp` may allow removing the patch.

### Development Debugging

1. **Use MCP Inspector**: Best way to test tools interactively
2. **Check Logs**: FastMCP logs to stdout, watch for errors
3. **Test Individual Tools**: Import and call tool functions directly in Python
4. **Use Examples**: The `examples/` directory shows working usage patterns
5. **Type Checking**: Run `uv run mypy langsmith_mcp_server/` to catch type errors

---

## Contributing Guidelines

### Adding a New Tool

1. **Implement the tool function** in appropriate `services/tools/*.py` file
   ```python
   def my_new_tool(client: Client, param: str) -> Dict[str, Any]:
       """
       Tool description.

       Args:
           client: LangSmith client
           param: Parameter description

       Returns:
           Dictionary with results or error
       """
       try:
           # Implementation
           return {"result": "data"}
       except Exception as e:
           return {"error": str(e)}
   ```

2. **Register the tool** in [register_tools.py](langsmith_mcp_server/services/register_tools.py)
   ```python
   @mcp.tool()
   def my_new_tool(param: str, ctx: Context = None) -> Dict[str, Any]:
       """User-facing docstring for the tool."""
       try:
           client = get_client_from_context(ctx)
           return my_new_tool_impl(client, param)
       except Exception as e:
           return {"error": str(e)}
   ```

3. **Write tests** in `tests/tools/test_*.py`

4. **Run quality checks**
   ```bash
   make format
   make lint
   make test
   ```

5. **Test with MCP Inspector**
   ```bash
   uv run mcp dev langsmith_mcp_server/server.py
   ```

6. **Update CLAUDE.md** - Add your tool to the "Available MCP Tools" section

### Code Review Checklist

- [ ] Code is formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Tests pass (`make test`)
- [ ] Type hints are present and correct
- [ ] Docstrings explain purpose and parameters
- [ ] Error handling returns `{"error": ...}` not exceptions
- [ ] CLAUDE.md is updated if architecture or tools changed
- [ ] Examples added if introducing significant functionality

---

## Recent Architecture Changes

### Unified Pagination in fetch_runs (February 2025)
- **Removed**: `paginate_runs` tool (was redundant - required trace_id)
- **Enhanced**: `fetch_runs` now always returns paginated responses with character budget controls
- **Why**: Simplified API surface, consistent pagination across all queries
- **Key difference**: `fetch_runs` works with OR without trace_id (more flexible than old `paginate_runs`)
- **Migration examples**:
  - Old: `paginate_runs("my-project", trace_id="abc", page_number=1)`
  - New: `fetch_runs("my-project", limit=100, page_number=1, trace_id="abc")`
  - Or without trace_id: `fetch_runs("my-project", limit=50, page_number=1, is_root="true")`

## Future Development Roadmap

### Planned Features
- **Enhanced Filtering**: More query capabilities for datasets, prompts, and runs
- **Batch Operations**: Multi-item operations for efficiency
- **Real-time Streaming**: Live trace and conversation monitoring via SSE
- **Advanced Analytics**: Statistical analysis tools and aggregations
- **Workspace Management**: Implement `workspaces.py` functionality
- **MCP Resources**: Implement `register_resources()` for dynamic documentation
- **MCP Prompts**: Implement `register_prompts()` for predefined prompt templates

### Extension Points
- **Custom Tools**: Add domain-specific tools in `services/tools/`
- **Additional Transports**: WebSocket support for streaming
- **Caching Layer**: Add Redis/memcached for frequently accessed data
- **Authentication Methods**: OAuth, JWT support in middleware
- **Multi-workspace Support**: Better workspace isolation and management

---

## Additional Resources

- **LangSmith Documentation**: https://docs.smith.langchain.com
- **MCP Specification**: https://modelcontextprotocol.io
- **FastMCP Framework**: https://github.com/jlowin/fastmcp
- **LangSmith Python SDK**: https://github.com/langchain-ai/langsmith-sdk
- **Issue Tracker**: https://github.com/langchain-ai/langsmith-mcp-server/issues

---

**Remember**: This document should evolve with the codebase. When you add features, fix bugs, or discover important patterns, update this file to help future developers (including AI assistants) work more effectively.
