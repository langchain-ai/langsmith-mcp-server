"""Tools for interacting with LangSmith datasets."""

from typing import Any, Dict

from langsmith import Client

from langsmith_mcp_server.common.helpers import _parse_as_of_parameter


def list_datasets_tool(
    client: Client,
    dataset_ids: list = None,
    data_type: str | None = None,
    dataset_name: str | None = None,
    dataset_name_contains: str | None = None,
    metadata: dict = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Fetch datasets from LangSmith with optional filtering.

    Args:
        client: LangSmith client instance
        dataset_ids: List of dataset IDs to filter by
        data_type: Filter by dataset data type (e.g., 'chat', 'kv')
        dataset_name: Filter by exact dataset name
        dataset_name_contains: Filter by substring in dataset name
        metadata: Filter by metadata dict
        limit: Max number of datasets to return

    Returns:
        Dictionary containing the datasets and metadata
    """
    try:
        # Prepare kwargs for the client call
        kwargs = {}
        if dataset_ids is not None:
            kwargs["dataset_ids"] = dataset_ids
        if data_type is not None:
            kwargs["data_type"] = data_type
        if dataset_name is not None:
            kwargs["dataset_name"] = dataset_name
        if dataset_name_contains is not None:
            kwargs["dataset_name_contains"] = dataset_name_contains
        if metadata is not None:
            kwargs["metadata"] = metadata
        if limit is not None:
            kwargs["limit"] = limit

        # Call the SDK
        datasets = list(client.list_datasets(**kwargs))

        # Attributes to return for each dataset
        attrs = [
            "id",
            "name",
            "inputs_schema_definition",
            "outputs_schema_definition",
            "description",
            "data_type",
            "example_count",
            "session_count",
            "created_at",
            "modified_at",
            "last_session_start_time",
        ]

        formatted_datasets = []
        for dataset in datasets:
            dataset_dict = {}
            for attr in attrs:
                value = getattr(dataset, attr, None)
                # Format datetimes as isoformat
                if attr in ("created_at", "modified_at") and value is not None:
                    value = value.isoformat()
                # Convert UUIDs to strings for JSON serialization
                elif attr == "id" and value is not None:
                    value = str(value)
                dataset_dict[attr] = value
            formatted_datasets.append(dataset_dict)

        return {"datasets": formatted_datasets, "total_count": len(formatted_datasets)}

    except Exception as e:
        return {"error": f"Error fetching datasets: {str(e)}"}


def list_examples_tool(
    client: Client,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    example_ids: list = None,
    filter: str | None = None,
    metadata: dict = None,
    splits: list = None,
    inline_s3_urls: bool | None = None,
    include_attachments: bool | None = None,
    as_of: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> Dict[str, Any]:
    """
    Fetch examples from a LangSmith dataset.

    Args:
        client: LangSmith Client instance
        dataset_id: Dataset ID to retrieve examples from
        dataset_name: Dataset name to retrieve examples from
        example_ids: List of specific example IDs to retrieve
        limit: Maximum number of examples to return
        offset: Number of examples to skip before starting to return results
        filter: Filter string to apply to search results (uses same syntax as list_runs)
        metadata: Dictionary of metadata to filter by
        splits: List of dataset splits to include examples from
        inline_s3_urls: Whether to inline S3 URLs (default: SDK default if not specified)
        include_attachments: Whether to include attachments in response (default: SDK default if not specified)
        as_of: Dataset version tag OR ISO timestamp to retrieve examples as of that version/time

    Returns:
        Dictionary containing the examples and metadata
    """
    try:
        # Prepare kwargs for the client call
        kwargs = {}
        if dataset_id is not None:
            kwargs["dataset_id"] = dataset_id
        if dataset_name is not None:
            kwargs["dataset_name"] = dataset_name
        if example_ids is not None:
            kwargs["example_ids"] = example_ids
        if metadata is not None:
            kwargs["metadata"] = metadata
        if splits is not None:
            kwargs["splits"] = splits
        if inline_s3_urls is not None:
            kwargs["inline_s3_urls"] = inline_s3_urls
        if include_attachments is not None:
            kwargs["include_attachments"] = include_attachments
        if as_of is not None:
            kwargs["as_of"] = _parse_as_of_parameter(as_of)
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        if filter is not None:
            kwargs["filter"] = filter

        # Call the SDK
        examples = list(client.list_examples(**kwargs))

        # Attributes to return for each example
        attrs = [
            "id",
            "dataset_id",
            "inputs",
            "outputs",
            "metadata",
            "created_at",
            "modified_at",
            "runs",
            "source_run_id",
            "attachments",
        ]

        formatted_examples = []
        for example in examples:
            example_dict = {}
            for attr in attrs:
                value = getattr(example, attr, None)
                # Format datetimes as isoformat
                if attr in ("created_at", "modified_at") and value is not None:
                    value = value.isoformat()
                # Convert UUIDs to strings for JSON serialization
                elif attr in ("id", "dataset_id", "source_run_id") and value is not None:
                    value = str(value)
                example_dict[attr] = value
            formatted_examples.append(example_dict)

        return {"examples": formatted_examples, "total_count": len(formatted_examples)}

    except Exception as e:
        return {"error": f"Error fetching examples: {str(e)}"}


def read_dataset_tool(
    client: Client,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
) -> Dict[str, Any]:
    """
    Read a specific dataset from LangSmith.

    Args:
        client: LangSmith Client instance
        dataset_id: Dataset ID to retrieve
        dataset_name: Dataset name to retrieve

    Returns:
        Dictionary containing the dataset details
    """
    try:
        # Prepare kwargs for the client call
        kwargs = {}
        if dataset_id is not None:
            kwargs["dataset_id"] = dataset_id
        if dataset_name is not None:
            kwargs["dataset_name"] = dataset_name

        # Call the SDK
        dataset = client.read_dataset(**kwargs)

        # Attributes to return for the dataset
        attrs = [
            "id",
            "name",
            "inputs_schema_definition",
            "outputs_schema_definition",
            "description",
            "data_type",
            "example_count",
            "session_count",
            "created_at",
            "modified_at",
            "last_session_start_time",
        ]

        dataset_dict = {}
        for attr in attrs:
            value = getattr(dataset, attr, None)
            # Format datetimes as isoformat
            if (
                attr in ("created_at", "modified_at", "last_session_start_time")
                and value is not None
            ):
                value = value.isoformat()
            # Convert UUIDs to strings for JSON serialization
            elif attr == "id" and value is not None:
                value = str(value)
            dataset_dict[attr] = value

        return {"dataset": dataset_dict}

    except Exception as e:
        return {"error": f"Error reading dataset: {str(e)}"}


def read_example_tool(
    client: Client,
    example_id: str,
    as_of: str | None = None,
) -> Dict[str, Any]:
    """
    Read a specific example from LangSmith.

    Args:
        client: LangSmith Client instance
        example_id: Example ID to retrieve
        as_of: Dataset version tag OR ISO timestamp to retrieve the example as of that version/time

    Returns:
        Dictionary containing the example details
    """
    try:
        # Prepare kwargs for the client call
        kwargs = {"example_id": example_id}

        if as_of is not None:
            kwargs["as_of"] = _parse_as_of_parameter(as_of)

        # Call the SDK
        example = client.read_example(**kwargs)

        # Attributes to return for the example
        attrs = [
            "id",
            "dataset_id",
            "inputs",
            "outputs",
            "metadata",
            "created_at",
            "modified_at",
            "runs",
            "source_run_id",
            "attachments",
        ]

        example_dict = {}
        for attr in attrs:
            value = getattr(example, attr, None)
            # Format datetimes as isoformat
            if attr in ("created_at", "modified_at") and value is not None:
                value = value.isoformat()
            # Convert UUIDs to strings for JSON serialization
            elif attr in ("id", "dataset_id", "source_run_id") and value is not None:
                value = str(value)
            example_dict[attr] = value

        return {"example": example_dict}

    except Exception as e:
        return {"error": f"Error reading example: {str(e)}"}
