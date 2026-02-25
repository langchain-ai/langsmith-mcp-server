"""Registration module for LangSmith MCP tools."""

import json
from typing import Any, Dict, Optional

from fastmcp import FastMCP
from fastmcp.server import Context

from langsmith_mcp_server.common.helpers import (
    get_api_key_and_endpoint_from_context,
    get_client_from_context,
)
from langsmith_mcp_server.services.tools.datasets import (
    list_datasets_tool,
    list_examples_tool,
    read_dataset_tool,
    read_example_tool,
)
from langsmith_mcp_server.services.tools.experiments import (
    list_experiments_tool,
)
from langsmith_mcp_server.services.tools.prompts import (
    get_prompt_tool,
    list_prompts_tool,
)
from langsmith_mcp_server.services.tools.traces import (
    fetch_runs_tool,
    get_thread_history_tool,
    list_projects_tool,
)
from langsmith_mcp_server.services.tools.usage import get_billing_usage_tool


def register_tools(mcp: FastMCP) -> None:
    """
    Register all LangSmith tool-related functionality with the MCP server.
    This function configures and registers various tools for interacting with LangSmith,
    including prompt management, conversation history, traces, and analytics.

    Args:
        mcp: The MCP server instance to register tools with
    """

    @mcp.tool()
    async def list_prompts(
        is_public: str = "false", limit: int = 20, ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Fetch prompts from LangSmith with optional filtering.

        Args:
            is_public (str): Filter by prompt visibility - "true" for public prompts,
                            "false" for private prompts (default: "false")
            limit (int): Maximum number of prompts to return (default: 20)

        Returns:
            Dict[str, Any]: Dictionary containing the prompts and metadata
        """
        try:
            client = await get_client_from_context(ctx)
            is_public_bool = is_public.lower() == "true"
            return list_prompts_tool(client, is_public_bool, limit)
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def get_prompt_by_name(prompt_name: str, ctx: Context = None) -> Dict[str, Any]:
        """
        Get a specific prompt by its exact name.

        Args:
            prompt_name (str): The exact name of the prompt to retrieve
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the prompt details and template,
                          or an error message if the prompt cannot be found
        """
        try:
            client = await get_client_from_context(ctx)
            return get_prompt_tool(client, prompt_name=prompt_name)
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def push_prompt(ctx: Context = None) -> str:
        """
        Call this tool when you need to understand how to create and push prompts to LangSmith.
        """
        documentation = """
Documentation tool for understanding how to create and push prompts to LangSmith.

This tool provides comprehensive documentation on creating ChatPromptTemplate and
StructuredPrompt objects and pushing them to LangSmith using the LangSmith Client.

---
🧩 PURPOSE
----------
This is a **documentation-only tool** that explains how to:
- Create prompts using LangChain's prompt templates
- Push prompts to LangSmith for version control and management
- Handle prompt creation vs. version updates

---
📦 REQUIRED DEPENDENCIES
------------------------
To use the functionality described in this documentation, you need:
- `langsmith` - The LangSmith Python client
- `langchain-core` - Core LangChain functionality for prompt templates
- `langchain` (optional) - Required only if using `from langchain.messages` imports

Install with:
```bash
pip install langsmith langchain-core
# Optional, for message classes:
pip install langchain
```

---
🔧 HOW TO PUSH PROMPTS
-----------------------
Use the LangSmith Client's `push_prompt()` method:

```python
from langsmith import Client

client = Client()

url = client.push_prompt(
    prompt_identifier="my-prompt-name",
    object=prompt,  # Your prompt object
    description="Optional description",
    tags=["tag1", "tag2"],  # Optional tags
    is_public=False,  # Optional visibility (True/False)
)
```

**Behavior:**
- If the prompt name **doesn't exist**: Creates a new prompt in LangSmith
- If the prompt name **exists** and it's a **new version**: Creates a new commit/version
- If the prompt name **exists** and it's the **same version**: No new commit is created

---
📝 CREATING CHATPROMPTTEMPLATE PROMPTS
--------------------------------------

1️⃣ **Basic ChatPromptTemplate**
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Your name is {assistant_name}."),
    ("human", "{user_input}"),
])

client.push_prompt("my-chat-prompt", object=prompt)
```

2️⃣ **Using Message Classes**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import SystemMessage, HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a coding assistant."),
    HumanMessage(content="Write a Python function to {task}"),
    AIMessage(content="I'll help you write that function."),
    ("human", "Make it {style}"),
])

client.push_prompt("my-message-classes-prompt", object=prompt)
```

3️⃣ **With MessagesPlaceholder for Conversation History**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="conversation", optional=True),
    ("human", "{user_input}"),
])

client.push_prompt("my-conversation-prompt", object=prompt)
```

4️⃣ **Complex Prompt with Multiple Placeholders**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.messages import HumanMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are {assistant_name}, a {role} assistant."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "Current question: {question}"),
    ("ai", "Let me think about that..."),
    MessagesPlaceholder(variable_name="tool_results", optional=True),
    HumanMessage(content="Based on the above, what's your final answer?"),
])

client.push_prompt("my-complex-prompt", object=prompt)
```

---
🎯 CREATING STRUCTUREDPROMPT PROMPTS
------------------------------------

StructuredPrompt allows you to define output schemas for structured outputs.

1️⃣ **With Dictionary Schema (with title and description)**
```python
from langchain_core.prompts.structured import StructuredPrompt

schema = {
    "title": "SentimentAnalysis",
    "description": "Analyzes the sentiment of text with confidence and reasoning",
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "The sentiment of the text"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score between 0 and 1"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief reasoning for the sentiment"
        }
    },
    "required": ["sentiment", "confidence", "reasoning"],
    "strict": True
}

prompt = StructuredPrompt(
    [
        ("system", "You are a sentiment analysis expert."),
        ("human", "Analyze the sentiment of: {text}"),
    ],
    schema_=schema,
)

client.push_prompt("my-structured-prompt", object=prompt)
```

2️⃣ **With Pydantic Model (Convert to Dict Schema)**
```python
from langchain_core.prompts.structured import StructuredPrompt
from pydantic import BaseModel, Field

class UserInfo(BaseModel):
    '''User information extracted from text.'''
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")
    email: str = Field(description="The user's email address")

# Convert Pydantic model to dict schema
schema_dict = UserInfo.model_json_schema()
# Add title and description at top level if not present
if "title" not in schema_dict:
    schema_dict["title"] = UserInfo.__name__
if "description" not in schema_dict:
    schema_dict["description"] = UserInfo.__doc__ or f"Schema for {UserInfo.__name__}"

prompt = StructuredPrompt(
    [
        ("system", "You are a helpful assistant that extracts user information."),
        ("human", "Extract information from: {text}"),
    ],
    schema_=schema_dict,
)

client.push_prompt("my-pydantic-prompt", object=prompt)
```

---
🧠 HELPER FUNCTION PATTERN
---------------------------
You can create a reusable helper function:

```python
def push_prompt_to_langsmith(
    prompt,
    prompt_identifier: str,
    description: str = None,
    tags: list = None,
    is_public: bool = None,
) -> str:
    '''
    Push a prompt to LangSmith with optional metadata.

    Args:
        prompt: The prompt object (ChatPromptTemplate, StructuredPrompt, etc.)
        prompt_identifier: The name/identifier for the prompt
        description: Optional description of the prompt
        tags: Optional list of tags
        is_public: Optional visibility setting (True/False)

    Returns:
        The URL of the pushed prompt
    '''
    kwargs = {"object": prompt}
    if description:
        kwargs["description"] = description
    if tags:
        kwargs["tags"] = tags
    if is_public is not None:
        kwargs["is_public"] = is_public

    url = client.push_prompt(prompt_identifier, **kwargs)
    return url
```

---
📤 RETURNS
----------
None
    This tool is documentation-only and returns None. The documentation is in the docstring.

---
🧠 NOTES FOR AGENTS
--------------------
- This tool is **documentation-only** - it does not execute any code
- Use this tool to understand how to create and push prompts programmatically
- The `push_prompt()` method automatically handles versioning:
  - New prompt name → creates new prompt
  - Existing prompt name with changes → creates new version/commit
  - Existing prompt name with no changes → no new commit
- Always ensure you have the required dependencies installed before using these patterns
- Prompt identifiers should be unique and descriptive
- Use tags and descriptions to organize and document your prompts

---
🔐 ENVIRONMENT VARIABLES
-------------------------
Before using the LangSmith Client, make sure to set up your environment variables:

**Required:**
```bash
export LANGSMITH_API_KEY="lsv2_pt_..."
```

**Optional:**
```bash
# Only needed if using a custom endpoint (defaults to cloud if not set)
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

# Only needed if you want to specify a workspace
export LANGSMITH_WORKSPACE_ID="35e66a3b-2973-4830-83e1-352c43a660ed"
```

You can also use a `.env` file with `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env file

from langsmith import Client
client = Client()  # Will automatically use environment variables
```
"""
        return documentation

    # Register conversation tools
    @mcp.tool()
    async def get_thread_history(
        thread_id: str,
        project_name: str,
        page_number: int,
        max_chars_per_page: int = 25000,
        preview_chars: int = 150,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Retrieve one page of message history for a specific conversation thread.

        Uses char-based pagination: pages are built by character budget (max_chars_per_page).
        Long strings are truncated to preview_chars. Supply page_number (1-based) on every call;
        use the returned total_pages to request further pages.

        Args:
            thread_id (str): The unique ID of the thread to fetch history for
            project_name (str): The name of the project containing the thread
                               (format: "owner/project" or just "project")
            page_number (int): 1-based page index (required)
            max_chars_per_page (int): Max character count per page, capped at 30000 (default: 25000)
            preview_chars (int): Truncate long strings to this length with "… (+N chars)" (default: 150)

        Returns:
            Dict with result (list of messages), page_number, total_pages, max_chars_per_page,
            preview_chars; or an error message if the thread cannot be found
        """
        try:
            client = await get_client_from_context(ctx)
            return get_thread_history_tool(
                client,
                thread_id,
                project_name,
                page_number=page_number,
                max_chars_per_page=max_chars_per_page,
                preview_chars=preview_chars,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def fetch_runs(
        project_name: str,
        limit: int,
        page_number: int = 1,
        trace_id: str = None,
        run_type: str = None,
        error: str = None,
        is_root: str = None,
        filter: str = None,
        trace_filter: str = None,
        tree_filter: str = None,
        order_by: str = "-start_time",
        reference_example_id: str = None,
        max_chars_per_page: int = 25000,
        preview_chars: int = 150,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Fetch LangSmith runs from one or more projects with flexible filters and automatic pagination.

        All results are paginated by character budget to keep responses manageable. Use page_number
        and total_pages from the response to iterate through multiple pages.

        ---
        📤 RETURNS (always paginated)
        ------------------------------
        Dict with: runs, page_number, total_pages, max_chars_per_page, preview_chars.
        May include _truncated, _truncated_message, _truncated_preview if content exceeded budget.

        ---
        ⚙️ PARAMETERS (required)
        ------------------------
        project_name : str
            The project name to fetch runs from. For multiple projects, use JSON array string:
            '["project1", "project2"]'

        limit : int (required)
            Maximum number of runs to fetch from LangSmith API (capped at 100).
            These runs are then paginated by character budget into pages.

        page_number : int, default 1
            1-based page index. Use with total_pages from response to iterate through pages.

        trace_id : str, optional
            Return only runs that belong to this trace UUID.
            Example: "123e4567-e89b-12d3-a456-426614174000"

        run_type : str, optional
            Filter runs by type: "llm", "chain", "tool", "retriever".

        error : str, optional
            "true" for errored runs, "false" for successful runs.

        is_root : str, optional
            "true" for only top-level traces, "false" to exclude roots.

        filter : str, optional
            Filter Query Language (FQL) expression. Common fields:
            id, name, run_type, start_time, end_time, latency, total_tokens, error, tags,
            feedback_key, feedback_score, metadata_key, metadata_value, execution_order.

            Operators: eq, neq, gt, gte, lt, lte, has, search, and, or, not.

            Examples:
            'gt(latency, "5s")'                                # runs > 5 seconds
            'neq(error, null)'                                 # errored runs
            'has(tags, "beta")'                                # tagged "beta"
            'and(eq(name,"ChatOpenAI"), eq(run_type,"llm"))'   # name AND type
            'search("image classification")'                   # full-text search

        trace_filter : str, optional
            Filter on the root run of each trace tree.
            Example: 'and(eq(feedback_key,"user_score"), eq(feedback_score,1))'

        tree_filter : str, optional
            Filter on any run in the trace tree (siblings/children included).
            Example: 'eq(name,"ExpandQuery")'

        order_by : str, default "-start_time"
            Sort field; prefix "-" for descending. Examples: "-start_time", "latency"

        reference_example_id : str, optional
            Filter runs by dataset example ID.

        max_chars_per_page : int, default 25000
            Max JSON character count per page (capped at 30000). Pagination splits by this budget.

        preview_chars : int, default 150
            Truncate long strings to this length with "… (+N chars)". Keeps responses readable.

        ---
        🧪 EXAMPLES
        -----------
        1️⃣ Get latest 10 root runs:
        fetch_runs("my-project", limit=10, page_number=1, is_root="true")

        2️⃣ Get errored tool runs:
        fetch_runs("my-project", limit=50, page_number=1, run_type="tool", error="true")

        3️⃣ Runs > 5s with "experimental" tag:
        fetch_runs("my-project", limit=50, page_number=1,
                   filter='and(gt(latency,"5s"), has(tags,"experimental"))')

        4️⃣ All runs for a trace (paginate through pages):
        r = fetch_runs("my-project", limit=50, page_number=1, trace_id="abc-123")
        # Check r["total_pages"] and fetch subsequent pages if needed
        """  # noqa: W293
        try:
            client = await get_client_from_context(ctx)

            # Parse project name (support JSON array format)
            parsed_project_name = project_name
            if project_name and project_name.startswith("["):
                try:
                    parsed_project_name = json.loads(project_name)
                except json.JSONDecodeError:
                    pass

            # Parse boolean string parameters
            parsed_error = None
            if error is not None:
                parsed_error = (
                    error.lower() == "true" if error.lower() in ("true", "false") else None
                )

            parsed_is_root = None
            if is_root is not None:
                if is_root.lower() == "true":
                    parsed_is_root = True
                elif is_root.lower() == "false":
                    parsed_is_root = False

            # Always use paginated fetch_runs_tool (now supports pagination built-in)
            return fetch_runs_tool(
                client,
                project_name=parsed_project_name,
                page_number=page_number,
                max_chars_per_page=max_chars_per_page,
                preview_chars=preview_chars,
                trace_id=trace_id,
                run_type=run_type,
                error=parsed_error,
                is_root=parsed_is_root,
                filter=filter,
                trace_filter=trace_filter,
                tree_filter=tree_filter,
                order_by=order_by,
                limit=limit,
                reference_example_id=reference_example_id,
            )
        except Exception as e:
            return {"error": str(e)}

    # Register project tools
    @mcp.tool()
    async def list_projects(
        limit: int = 5,
        project_name: str = None,
        more_info: str = "false",
        reference_dataset_id: str = None,
        reference_dataset_name: str = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        List LangSmith projects with optional filtering and detail level control.

        Fetches projects from LangSmith, optionally filtering by name and controlling
        the level of detail returned. Can return either simplified project information
        or full project details.
        In case a dataset id or name is provided, you don't need to provide a project name.

        ---
        🧩 PURPOSE
        ----------
        This function provides a convenient way to list and explore LangSmith projects.
        It supports:
        - Filtering projects by name (partial match)
        - Limiting the number of results
        - Choosing between simplified or full project information
        - Automatically extracting deployment IDs from nested project data

        ---
        ⚙️ PARAMETERS
        -------------
        limit : int, default 5
            Maximum number of projects to return (as string, e.g., "5"). This can be adjusted by agents
            or users based on their needs.

        project_name : str, optional
            Filter projects by name using partial matching. If provided, only projects
            whose names contain this string will be returned.
            Example: `project_name="Chat"` will match "Chat-LangChain", "ChatBot", etc.

        more_info : str, default "false"
            Controls the level of detail returned:
            - `"false"` (default): Returns simplified project information with only
            essential fields: `name`, `project_id`, and `agent_deployment_id` (if available)
            - `"true"`: Returns full project details as returned by the LangSmith API

        reference_dataset_id : str, optional
            The ID of the reference dataset to filter projects by.
            Either this OR `reference_dataset_name` must be provided (but not both).

        reference_dataset_name : str, optional
            The name of the reference dataset to filter projects by.
            Either this OR `reference_dataset_id` must be provided (but not both).

        ---
        📤 RETURNS
        ----------
        List[dict]
            A list of project dictionaries. The structure depends on `more_info`:

            **When `more_info=False` (simplified):**
            ```python
            [
                {
                    "name": "Chat-LangChain",
                    "project_id": "787d5165-f110-43ff-a3fb-66ea1a70c971",
                    "agent_deployment_id": "deployment-123"  # Only if available
                },
                ...
            ]
            ```

            **When `more_info=True` (full details):**
            Returns complete project objects with all fields from the LangSmith API,
            including metadata, settings, statistics, and nested structures.

        ---
        🧪 EXAMPLES
        ------------
        1️⃣ **List first 5 projects (simplified)**
        ```python
        projects = list_projects(limit="5")
        ```

        2️⃣ **Search for projects with "Chat" in the name**
        ```python
        projects = list_projects(project_name="Chat", limit="10")
        ```

        3️⃣ **Get full project details**
        ```python
        projects = list_projects(limit="3", more_info="true")
        ```

        4️⃣ **Find a specific project with full details**
        ```python
        projects = list_projects(project_name="MyProject", more_info="true", limit="1")
        ```

        ---
        🧠 NOTES FOR AGENTS
        --------------------
        - Use `more_info="false"` for quick project discovery and listing
        - Use `more_info="true"` when you need detailed project information
        - The `agent_deployment_id` field is automatically extracted from nested
        project data when available, making it easy to identify agent deployments
        - Projects are filtered to exclude reference projects by default
        - The function uses `name_contains` for filtering, so partial matches work
        """  # noqa: W293
        try:
            client = await get_client_from_context(ctx)
            parsed_more_info = more_info.lower() == "true"
            if reference_dataset_id is not None and reference_dataset_name is not None:
                parsed_more_info = True
            return list_projects_tool(
                client,
                limit=limit,
                project_name=project_name,
                more_info=parsed_more_info,
                reference_dataset_id=reference_dataset_id,
                reference_dataset_name=reference_dataset_name,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def get_billing_usage(
        starting_on: str,
        ending_before: str,
        workspace: Optional[str] = None,
        on_current_plan: str = "true",
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Fetch organization billing usage (trace counts) with workspace names inline.

        Returns metrics from GET /api/v1/orgs/current/billing/usage. Each metric's
        `groups` is augmented to { workspace_uuid: { "workspace_name": "<name>", "value": <number> } }.
        If workspace is provided (UUID or display name), only that workspace's entries
        are included in each metric's groups.

        Args:
            starting_on: Start of date range (ISO 8601), e.g. "2025-09-01T00:00:00Z"
            ending_before: End of date range (ISO 8601), e.g. "2025-10-01T00:00:00Z"
            workspace: Optional single workspace UUID or display name to filter to
            on_current_plan: "true" to include only usage on current plan (default)

        Returns:
            List of billing metric objects with augmented groups, or dict with "error" key
        """
        try:
            api_key, endpoint = await get_api_key_and_endpoint_from_context(ctx)
            on_current = on_current_plan.lower() == "true"
            result = get_billing_usage_tool(
                api_key=api_key,
                endpoint=endpoint,
                starting_on=starting_on,
                ending_before=ending_before,
                on_current_plan=on_current,
                workspace=workspace,
            )
            if isinstance(result, dict) and "error" in result:
                return result
            return {"usage": result}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def list_experiments(
        reference_dataset_id: Optional[str] = None,
        reference_dataset_name: Optional[str] = None,
        limit: int = 5,
        project_name: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        List LangSmith experiment projects (reference projects) with mandatory dataset filtering.

        Fetches experiment projects from LangSmith that are associated with a specific dataset.
        These are projects used for model evaluation and comparison. Requires either a
        dataset ID or dataset name to filter experiments.

        ---
        🧩 PURPOSE
        ----------
        This function provides a convenient way to list and explore LangSmith experiment projects.
        It supports:
        - Filtering experiments by reference dataset (mandatory)
        - Filtering projects by name (partial match)
        - Limiting the number of results
        - Automatically extracting deployment IDs from nested project data
        - Returns simplified project information with key metrics (latency, cost, feedback stats)

        ---
        ⚙️ PARAMETERS
        -------------
        reference_dataset_id : str, optional
            The ID of the reference dataset to filter experiments by.
            Either this OR `reference_dataset_name` must be provided (but not both).

        reference_dataset_name : str, optional
            The name of the reference dataset to filter experiments by.
            Either this OR `reference_dataset_id` must be provided (but not both).

        limit : int, default 5
            Maximum number of experiments to return. This can be adjusted by agents
            or users based on their needs.

        project_name : str, optional
            Filter projects by name using partial matching. If provided, only projects
            whose names contain this string will be returned.
            Example: `project_name="Chat"` will match "Chat-LangChain", "ChatBot", etc.

        ---
        📤 RETURNS
        ----------
        Dict[str, Any]
            A dictionary containing an "experiments" key with a list of simplified experiment project dictionaries:

            ```python
            {
                "experiments": [
                    {
                        "name": "Experiment-Chat-LangChain",
                        "experiment_id": "787d5165-f110-43ff-a3fb-66ea1a70c971",
                        "feedback_stats": {...},  # Feedback statistics if available
                        "latency_p50_seconds": 1.626,  # 50th percentile latency in seconds
                        "latency_p99_seconds": 2.390,   # 99th percentile latency in seconds
                        "total_cost": 0.00013005,       # Total cost in dollars
                        "prompt_cost": 0.00002085,      # Prompt cost in dollars
                        "completion_cost": 0.0001092,   # Completion cost in dollars
                        "agent_deployment_id": "deployment-123"  # Only if available
                    },
                    ...
                ]
            }
            ```

        ---
        🧪 EXAMPLES
        ------------
        1️⃣ **List experiments for a dataset by ID**
        ```python
        experiments = list_experiments(reference_dataset_id="f5ca13c6-96ad-48ba-a432-ebb6bf94528f")
        ```

        2️⃣ **List experiments for a dataset by name**
        ```python
        experiments = list_experiments(reference_dataset_name="my-dataset", limit=10)
        ```

        3️⃣ **Find experiments with specific name pattern**
        ```python
        experiments = list_experiments(
            reference_dataset_id="f5ca13c6-96ad-48ba-a432-ebb6bf94528f",
            project_name="Chat",
            limit=1
        )
        ```

        ---
        🧠 NOTES FOR AGENTS
        --------------------
        - Returns simplified experiment information with key metrics (latency, cost, feedback stats)
        - The `agent_deployment_id` field is automatically extracted from nested
          project data when available, making it easy to identify agent deployments
        - Experiments are filtered to include only reference projects (associated with datasets)
        - The function uses `name_contains` for filtering, so partial matches work
        - You must provide either `reference_dataset_id` OR `reference_dataset_name`, but not both
        - Experiment projects are used for model evaluation and comparison across different runs
        """  # noqa: W293
        try:
            client = await get_client_from_context(ctx)
            return list_experiments_tool(
                client,
                reference_dataset_id=reference_dataset_id,
                reference_dataset_name=reference_dataset_name,
                limit=limit,
                project_name=project_name,
            )
        except Exception as e:
            return {"error": str(e)}

    # Register dataset tools
    @mcp.tool()
    async def list_datasets(
        dataset_ids: Optional[str] = None,
        data_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_name_contains: Optional[str] = None,
        metadata: Optional[str] = None,
        limit: int = 20,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Fetch LangSmith datasets.

        Note: If no arguments are provided, all datasets will be returned.

        Args:
            dataset_ids (Optional[str]): Dataset IDs to filter by as JSON array string (e.g., '["id1", "id2"]') or single ID
            data_type (Optional[str]): Filter by dataset data type (e.g., 'chat', 'kv')
            dataset_name (Optional[str]): Filter by exact dataset name
            dataset_name_contains (Optional[str]): Filter by substring in dataset name
            metadata (Optional[str]): Filter by metadata as JSON object string (e.g., '{"key": "value"}')
            limit (int): Max number of datasets to return (default: 20)
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the datasets and metadata,
                            or an error message if the datasets cannot be retrieved
        """  # noqa: W293
        try:
            client = await get_client_from_context(ctx)

            # Parse list strings (JSON arrays)
            parsed_dataset_ids = None
            if dataset_ids is not None:
                try:
                    parsed_dataset_ids = (
                        json.loads(dataset_ids) if dataset_ids.startswith("[") else [dataset_ids]
                    )
                except (json.JSONDecodeError, AttributeError):
                    parsed_dataset_ids = [dataset_ids] if dataset_ids else None

            # Parse metadata (JSON object)
            parsed_metadata = None
            if metadata is not None:
                try:
                    parsed_metadata = json.loads(metadata) if metadata.startswith("{") else None
                except (json.JSONDecodeError, AttributeError):
                    parsed_metadata = None

            return list_datasets_tool(
                client,
                dataset_ids=parsed_dataset_ids,
                data_type=data_type,
                dataset_name=dataset_name,
                dataset_name_contains=dataset_name_contains,
                metadata=parsed_metadata,
                limit=limit,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def list_examples(
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        example_ids: Optional[str] = None,
        filter: Optional[str] = None,
        metadata: Optional[str] = None,
        splits: Optional[str] = None,
        inline_s3_urls: Optional[str] = None,
        include_attachments: Optional[str] = None,
        as_of: Optional[str] = None,
        limit: int = 10,
        offset: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Fetch examples from a LangSmith dataset with advanced filtering options.

        Note: Either dataset_id, dataset_name, or example_ids must be provided.
        If multiple are provided, they are used in order of precedence: example_ids, dataset_id, dataset_name.

        Args:
            dataset_id (Optional[str]): Dataset ID to retrieve examples from
            dataset_name (Optional[str]): Dataset name to retrieve examples from
            example_ids (Optional[str]): Specific example IDs as JSON array string (e.g., '["id1", "id2"]') or single ID
            limit (int): Maximum number of examples to return (default: 10)
            offset (int): Number of examples to skip (default: 0)
            filter (Optional[str]): Filter string using LangSmith query syntax (e.g., 'has(metadata, {"key": "value"})')
            metadata (Optional[str]): Metadata to filter by as JSON object string (e.g., '{"key": "value"}')
            splits (Optional[str]): Dataset splits as JSON array string (e.g., '["train", "test"]') or single split
            inline_s3_urls (Optional[str]): Whether to inline S3 URLs: "true" or "false" (default: SDK default if not specified)
            include_attachments (Optional[str]): Whether to include attachments: "true" or "false" (default: SDK default if not specified)
            as_of (Optional[str]): Dataset version tag OR ISO timestamp to retrieve examples as of that version/time
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the examples and metadata,
                            or an error message if the examples cannot be retrieved
        """  # noqa: W293
        try:
            client = await get_client_from_context(ctx)

            # Parse list strings (JSON arrays)
            parsed_example_ids = None
            if example_ids is not None:
                try:
                    parsed_example_ids = (
                        json.loads(example_ids) if example_ids.startswith("[") else [example_ids]
                    )
                except (json.JSONDecodeError, AttributeError):
                    parsed_example_ids = [example_ids] if example_ids else None

            parsed_splits = None
            if splits is not None:
                try:
                    parsed_splits = json.loads(splits) if splits.startswith("[") else [splits]
                except (json.JSONDecodeError, AttributeError):
                    parsed_splits = [splits] if splits else None

            # Parse metadata (JSON object)
            parsed_metadata = None
            if metadata is not None:
                try:
                    parsed_metadata = json.loads(metadata) if metadata.startswith("{") else None
                except (json.JSONDecodeError, AttributeError):
                    parsed_metadata = None

            # Parse boolean strings
            parsed_inline_s3_urls = None
            if inline_s3_urls is not None:
                parsed_inline_s3_urls = inline_s3_urls.lower() == "true"

            parsed_include_attachments = None
            if include_attachments is not None:
                parsed_include_attachments = include_attachments.lower() == "true"

            # Parse integer strings
            parsed_limit = int(limit) if limit else None
            parsed_offset = int(offset) if offset else None

            return list_examples_tool(
                client,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                example_ids=parsed_example_ids,
                filter=filter,
                metadata=parsed_metadata,
                splits=parsed_splits,
                inline_s3_urls=parsed_inline_s3_urls,
                include_attachments=parsed_include_attachments,
                as_of=as_of,
                limit=parsed_limit,
                offset=parsed_offset,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def read_dataset(
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Read a specific dataset from LangSmith.

        Note: Either dataset_id or dataset_name must be provided to identify the dataset.
        If both are provided, dataset_id takes precedence.

        Args:
            dataset_id (Optional[str]): Dataset ID to retrieve
            dataset_name (Optional[str]): Dataset name to retrieve
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the dataset details,
                            or an error message if the dataset cannot be retrieved

        Example in case you need to create a separate python script to read a dataset:
            ```python
            from langsmith import Client

            client = Client()
            dataset = client.read_dataset(dataset_name="My Dataset")
            # Or by ID:
            # dataset = client.read_dataset(dataset_id="dataset-id-here")
            ```
        """
        try:
            client = await get_client_from_context(ctx)
            return read_dataset_tool(
                client,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    async def read_example(
        example_id: str,
        as_of: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Read a specific example from LangSmith.

        Args:
            example_id (str): Example ID to retrieve
            as_of (Optional[str]): Dataset version tag OR ISO timestamp to retrieve the example as of that version/time
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the example details,
                            or an error message if the example cannot be retrieved

        Example in case you need to create a separate python script to read an example:
            ```python
            from langsmith import Client

            client = Client()
            example = client.read_example(example_id="example-id-here")
            # Or with version:
            # example = client.read_example(example_id="example-id-here", as_of="v1.0")
            ```
        """
        try:
            client = await get_client_from_context(ctx)
            return read_example_tool(
                client,
                example_id=example_id,
                as_of=as_of,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def create_dataset(ctx: Context = None) -> str:
        """
        Call this tool when you need to understand how to create datasets in LangSmith.
        """
        documentation = """
Documentation tool for understanding how to create datasets in LangSmith.

This tool provides comprehensive documentation on creating datasets programmatically
using the LangSmith Python SDK, including creating datasets from lists, traces, CSV files,
and pandas DataFrames.

---
🧩 PURPOSE
----------
This is a **documentation-only tool** that explains how to:
- Create datasets from lists of examples
- Create datasets from traces/runs
- Create datasets from CSV files
- Create datasets from pandas DataFrames
- Add examples to datasets using bulk operations

---
📦 REQUIRED DEPENDENCIES
------------------------
To use the functionality described in this documentation, you need:
- `langsmith` - The LangSmith Python client
- `pandas` (optional) - Required only for DataFrame operations

Install with:
```bash
pip install langsmith
# Optional, for DataFrame operations:
pip install pandas
```

---
🔧 CREATING DATASETS
--------------------

1️⃣ **Create Dataset from List of Values**

The most flexible way to create a dataset is by creating examples from a list of inputs
and optional outputs. You can add arbitrary metadata to each example.

```python
from langsmith import Client

client = Client()

examples = [
    {
        "inputs": {"question": "What is the largest mammal?"},
        "outputs": {"answer": "The blue whale"},
        "metadata": {"source": "Wikipedia"},
    },
    {
        "inputs": {"question": "What do mammals and birds have in common?"},
        "outputs": {"answer": "They are both warm-blooded"},
        "metadata": {"source": "Wikipedia"},
    },
    {
        "inputs": {"question": "What are reptiles known for?"},
        "outputs": {"answer": "Having scales"},
        "metadata": {"source": "Wikipedia"},
    },
]

dataset_name = "Elementary Animal Questions"

# Create the dataset
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Questions and answers about animal phylogenetics.",
)

# Bulk create examples (more efficient than creating one at a time)
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)
```

**Note:** For many examples, use `create_examples()` for bulk creation. For a single
example, use `create_example()`.

2️⃣ **Create Dataset from Traces**

You can create datasets from the runs (spans) of your traces by filtering runs and
converting them to examples.

```python
from langsmith import Client

client = Client()
dataset_name = "Example Dataset"

# Filter runs to add to the dataset
runs = client.list_runs(
    project_name="my_project",
    is_root=True,
    error=False,
)

# Create the dataset
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="An example dataset"
)

# Prepare inputs and outputs for bulk creation
examples = [
    {"inputs": run.inputs, "outputs": run.outputs}
    for run in runs
]

# Use the bulk create_examples method
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)
```

3️⃣ **Create Dataset from CSV File**

You can create a dataset by uploading a CSV file. Ensure your CSV has columns that
represent your input and output keys.

```python
from langsmith import Client

client = Client()

csv_file = 'path/to/your/csvfile.csv'
input_keys = ['column1', 'column2']  # Replace with your input column names
output_keys = ['output1', 'output2']  # Replace with your output column names

dataset = client.upload_csv(
    csv_file=csv_file,
    input_keys=input_keys,
    output_keys=output_keys,
    name="My CSV Dataset",
    description="Dataset created from a CSV file",
    data_type="kv"  # "kv" or "chat"
)
```

4️⃣ **Create Dataset from Pandas DataFrame (Python only)**

The Python client offers a convenience method to upload a dataset from a pandas DataFrame.

```python
from langsmith import Client
import pandas as pd

client = Client()

# Load your data
df = pd.read_parquet('path/to/your/myfile.parquet')
# Or: df = pd.read_csv('path/to/your/myfile.csv')

input_keys = ['column1', 'column2']  # Replace with your input column names
output_keys = ['output1', 'output2']  # Replace with your output column names

dataset = client.upload_dataframe(
    df=df,
    input_keys=input_keys,
    output_keys=output_keys,
    name="My Parquet Dataset",
    description="Dataset created from a parquet file",
    data_type="kv"  # The default, can also be "chat"
)
```

---
📝 DATASET STRUCTURE
--------------------
Each example in a dataset should have:
- `inputs` (dict): The input data for the example
- `outputs` (dict, optional): The expected output data
- `metadata` (dict, optional): Arbitrary metadata (e.g., source, notes, tags)

---
📤 RETURNS
----------
None
    This tool is documentation-only and returns None. The documentation is in the docstring.

---
🧠 NOTES FOR AGENTS
--------------------
- This tool is **documentation-only** - it does not execute any code
- Use `create_examples()` for bulk operations (more efficient)
- Use `create_example()` for single example creation
- Datasets can be of type "kv" (key-value) or "chat" (conversational)
- Metadata is stored as a dictionary and can contain any key-value pairs
- Always ensure you have the required dependencies installed before using these patterns
- The dataset name should be unique and descriptive
"""
        return documentation

    @mcp.tool()
    def update_examples(ctx: Context = None) -> str:
        """
        Call this tool when you need to understand how to update dataset examples in LangSmith.
        """
        documentation = """
Documentation tool for understanding how to update dataset examples in LangSmith.

This tool provides comprehensive documentation on updating examples programmatically
using the LangSmith Python SDK, including single example updates and bulk updates.

---
🧩 PURPOSE
----------
This is a **documentation-only tool** that explains how to:
- Update a single example in a dataset
- Bulk update multiple examples in a single request
- Update inputs, outputs, metadata, and splits

---
📦 REQUIRED DEPENDENCIES
------------------------
To use the functionality described in this documentation, you need:
- `langsmith` - The LangSmith Python client

Install with:
```bash
pip install langsmith
```

---
🔧 UPDATING EXAMPLES
--------------------

1️⃣ **Update Single Example**

You can update a single example using the `update_example()` method. You can update
inputs, outputs, metadata, and split assignments.

```python
from langsmith import Client

client = Client()

# Update a single example
client.update_example(
    example_id=example.id,  # The example ID to update
    inputs={"input": "updated input"},
    outputs={"output": "updated output"},
    metadata={"foo": "bar", "source": "updated"},
    split="train"  # Can be a string or list of strings
)
```

**Parameters:**
- `example_id` (str, required): The ID of the example to update
- `inputs` (dict, optional): Updated input data
- `outputs` (dict, optional): Updated output data
- `metadata` (dict, optional): Updated metadata dictionary
- `split` (str or list, optional): Updated split assignment(s)

2️⃣ **Bulk Update Examples**

You can update multiple examples in a single request using the `update_examples()` method.
This is more efficient than updating examples one at a time.

```python
from langsmith import Client

client = Client()

# Update multiple examples at once
client.update_examples(
    example_ids=[example.id, example_2.id],
    inputs=[
        {"input": "updated input 1"},
        {"input": "updated input 2"}
    ],
    outputs=[
        {"output": "updated output 1"},
        {"output": "updated output 2"}
    ],
    metadata=[
        {"foo": "baz", "source": "source1"},
        {"foo": "qux", "source": "source2"}
    ],
    splits=[
        ["training", "foo"],  # Splits can be arrays
        "training"            # Or standalone strings
    ]
)
```

**Parameters:**
- `example_ids` (list[str], required): List of example IDs to update
- `inputs` (list[dict], optional): List of updated input dictionaries
- `outputs` (list[dict], optional): List of updated output dictionaries
- `metadata` (list[dict], optional): List of updated metadata dictionaries
- `splits` (list[str or list], optional): List of split assignments (can be strings or lists)

**Important Notes:**
- All list parameters must have the same length as `example_ids`
- Each list index corresponds to the example at the same index in `example_ids`
- Splits can be either a single string or a list of strings (for multiple splits)
- You can update any combination of fields - you don't need to provide all parameters

3️⃣ **Partial Updates**

You can update only specific fields without providing all parameters:

```python
# Update only metadata for a single example
client.update_example(
    example_id=example.id,
    metadata={"updated_at": "2024-01-01", "status": "reviewed"}
)

# Update only outputs for multiple examples
client.update_examples(
    example_ids=[example.id, example_2.id],
    outputs=[
        {"output": "new output 1"},
        {"output": "new output 2"}
    ]
)
```

---
📝 SPLIT ASSIGNMENTS
---------------------
Examples can be assigned to one or more splits (e.g., "train", "test", "validation"):

```python
# Single split
client.update_example(example_id=example.id, split="train")

# Multiple splits
client.update_example(example_id=example.id, split=["train", "validation"])

# In bulk updates, each example can have different split assignments
client.update_examples(
    example_ids=[example.id, example_2.id],
    splits=["train", ["train", "test"]]
)
```

---
📤 RETURNS
----------
None
    This tool is documentation-only and returns None. The documentation is in the docstring.

---
🧠 NOTES FOR AGENTS
--------------------
- This tool is **documentation-only** - it does not execute any code
- Use `update_examples()` for bulk operations (more efficient than single updates)
- Use `update_example()` for single example updates
- All list parameters in bulk updates must match the length of `example_ids`
- You can update any combination of fields - partial updates are supported
- Splits can be strings or lists of strings (for multiple split assignments)
- Always ensure you have the required dependencies installed before using these patterns
- Example IDs can be obtained from `list_examples()` or `read_example()` methods
"""
        return documentation

    @mcp.tool()
    def run_experiment(ctx: Context = None) -> str:
        """
        Call this tool when you need to understand how to run experiments and evaluations in LangSmith.
        """
        documentation = """
Documentation tool for understanding how to run experiments and evaluations in LangSmith.

This tool provides comprehensive documentation on running evaluations using LangSmith's
evaluate SDK, creating custom evaluators, and using the openevals library for pre-built
evaluators like LLM-as-judge and trajectory evaluators.

---
🧩 PURPOSE
----------
This is a **documentation-only tool** that explains how to:
- Run experiments using LangSmith's `evaluate()` method
- Create custom evaluator functions in Python
- Use openevals library for pre-built evaluators
- Set up LLM-as-judge evaluators
- Use trajectory evaluators for multi-turn conversations

---
📦 REQUIRED DEPENDENCIES
------------------------
To use the functionality described in this documentation, you need:
- `langsmith` - The LangSmith Python client (>=0.3.13 for evaluate)
- `openevals` (optional) - For pre-built evaluators like LLM-as-judge

Install with:
```bash
pip install langsmith
# Optional, for pre-built evaluators:
pip install openevals
```

---
🔧 RUNNING EXPERIMENTS WITH LANGSMITH
-------------------------------------

1️⃣ **Basic Evaluation Setup**

The `evaluate()` method runs your application on a dataset and scores outputs using evaluators.

```python
from langsmith import Client, traceable, wrappers
from openai import OpenAI

# Step 1: Define your application
oai_client = wrappers.wrap_openai(OpenAI())

@traceable
def toxicity_classifier(inputs: dict) -> dict:
    instructions = (
        "Please review the user query below and determine if it contains any form of "
        "toxic behavior, such as insults, threats, or highly negative comments. "
        "Respond with 'Toxic' if it does and 'Not toxic' if it doesn't."
    )
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": inputs["text"]},
    ]
    result = oai_client.chat.completions.create(
        messages=messages, model="gpt-4o-mini", temperature=0
    )
    return {"class": result.choices[0].message.content}

# Step 2: Create or select a dataset
ls_client = Client()
dataset = ls_client.create_dataset(dataset_name="Toxic Queries")

examples = [
    {"inputs": {"text": "Shut up, idiot"}, "outputs": {"label": "Toxic"}},
    {"inputs": {"text": "You're a wonderful person"}, "outputs": {"label": "Not toxic"}},
]
ls_client.create_examples(dataset_id=dataset.id, examples=examples)

# Step 3: Define an evaluator
def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return outputs["class"] == reference_outputs["label"]

# Step 4: Run the evaluation
results = ls_client.evaluate(
    toxicity_classifier,
    data=dataset.name,
    evaluators=[correct],
    experiment_prefix="gpt-4o-mini, baseline",
    description="Testing the baseline system.",
    max_concurrency=4,  # Optional: parallelize evaluation
)
```

**Key Parameters:**
- `target`: Your application function (takes inputs dict, returns outputs dict)
- `data`: Dataset name or UUID, or an iterator of examples
- `evaluators`: List of evaluator functions
- `experiment_prefix`: Optional name prefix for the experiment
- `description`: Optional experiment description
- `max_concurrency`: Optional number of parallel workers

2️⃣ **Creating Custom Evaluators**

Evaluators are functions that score your application's outputs. They receive:
- `inputs`: The example inputs
- `outputs`: Your application's actual outputs
- `reference_outputs`: Expected outputs (if available)

```python
# Simple boolean evaluator
def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return outputs["class"] == reference_outputs["label"]

# Evaluator with score and comment
def correctness_with_feedback(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> dict:
    is_correct = outputs["class"] == reference_outputs["label"]
    comment = "Match" if is_correct else "Mismatch"
    return {
        "key": "correctness",
        "score": is_correct,
        "comment": comment
    }

# Evaluator that returns a float score
def similarity_score(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> float:
    # Calculate some similarity metric (0.0 to 1.0)
    return 0.85
```

**Evaluator Return Types:**
- `bool`: Binary score (True/False)
- `float`: Numeric score (0.0 to 1.0)
- `dict`: Full result with `key`, `score`, and optional `comment`

---
🎯 USING OPENEVALS FOR PRE-BUILT EVALUATORS
-------------------------------------------

The `openevals` library provides pre-built evaluators that you can use out of the box.

1️⃣ **LLM-as-Judge Evaluators**

Use an LLM to judge your application's outputs. This is useful when you need subjective
evaluation or don't have reference outputs.

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, CONCISENESS_PROMPT

# Correctness evaluator (requires reference outputs)
correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:o3-mini",
)

# Conciseness evaluator (no reference needed)
conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    model="openai:o3-mini",
)

# Use with LangSmith evaluate
def wrapped_correctness_evaluator(inputs, outputs, reference_outputs):
    return correctness_evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )

results = ls_client.evaluate(
    toxicity_classifier,
    data=dataset.name,
    evaluators=[wrapped_correctness_evaluator],
)
```

**Pre-built Prompts Available:**
- `CORRECTNESS_PROMPT`: Evaluates correctness against reference outputs
- `CONCISENESS_PROMPT`: Evaluates how concise the output is
- `HALLUCINATION_PROMPT`: Checks for hallucinations (requires context)
- `RAG_HELPFULNESS_PROMPT`: For RAG applications
- `RAG_GROUNDEDNESS_PROMPT`: Checks if output is grounded in context
- `RAG_RETRIEVAL_RELEVANCE_PROMPT`: Evaluates retrieval quality

2️⃣ **Custom LLM-as-Judge with Custom Prompts**

You can create custom LLM-as-judge evaluators with your own prompts:

```python
from openevals.llm import create_llm_as_judge

CUSTOM_PROMPT = '''
You are an expert evaluator. Rate the output quality on a scale of 0-1.

<input>
{inputs}
</input>

<output>
{outputs}
</output>

<reference>
{reference_outputs}
</reference>
'''

custom_evaluator = create_llm_as_judge(
    prompt=CUSTOM_PROMPT,
    feedback_key="quality",
    model="openai:o3-mini",
    continuous=True,  # Returns float (0.0-1.0) instead of boolean
)
```

3️⃣ **Trajectory Evaluators for Multi-turn Conversations**

Trajectory evaluators evaluate entire conversation threads, useful for chat applications
and agents:

```python
from openevals.llm import create_llm_as_judge

# Create a trajectory evaluator that looks at the full conversation
trajectory_evaluator = create_llm_as_judge(
    model="openai:o3-mini",
    prompt="Based on the below conversation, was the user satisfied?\\n{outputs}",
    feedback_key="satisfaction",
)

# When using with evaluate, the outputs will be the full conversation trajectory
def wrapped_trajectory_evaluator(inputs, outputs, reference_outputs):
    # outputs here will be a list of messages representing the conversation
    return trajectory_evaluator(outputs=outputs)
```

4️⃣ **Other Pre-built Evaluators**

OpenEvals also provides evaluators for:
- **Exact Match**: Compare outputs exactly
- **Embedding Similarity**: Compare using embeddings
- **Levenshtein Distance**: String similarity
- **Code Evaluators**: Type checking, execution (requires additional setup)

```python
from openevals.exact import exact_match

# Simple exact match evaluator
def exact_match_evaluator(inputs, outputs, reference_outputs):
    return exact_match(outputs=outputs, reference_outputs=reference_outputs)
```

---
📝 EVALUATOR WRAPPER PATTERN
-----------------------------
When using openevals evaluators with LangSmith's `evaluate()`, you may need to wrap them
to match the expected signature:

```python
def wrap_openevals_evaluator(openevals_evaluator):
    def wrapped(inputs, outputs, reference_outputs):
        # openevals evaluators may have different parameter names
        result = openevals_evaluator(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs
        )
        return result
    return wrapped

# Usage
wrapped_evaluator = wrap_openevals_evaluator(correctness_evaluator)
results = ls_client.evaluate(
    app_function,
    data=dataset.name,
    evaluators=[wrapped_evaluator],
)
```

---
📤 RETURNS
----------
None
    This tool is documentation-only and returns None. The documentation is in the docstring.

---
🧠 NOTES FOR AGENTS
--------------------
- This tool is **documentation-only** - it does not execute any code
- Use `evaluate()` for synchronous evaluation, `aevaluate()` for async (better for large jobs)
- Set `max_concurrency` to parallelize evaluation across multiple workers
- Custom evaluators can return bool, float, or dict with `key`, `score`, `comment`
- OpenEvals evaluators are pre-built and tested - use them when possible
- LLM-as-judge evaluators are flexible but cost money (API calls to judge model)
- Trajectory evaluators are useful for multi-turn conversations and agent evaluation
- Always ensure you have the required dependencies installed before using these patterns
- For agent-specific evaluations, consider using the `agentevals` package
- Evaluation results are stored as feedback in LangSmith and can be viewed in the UI
"""
        return documentation
