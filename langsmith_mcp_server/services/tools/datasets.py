"""Tools for interacting with LangSmith datasets."""

from typing import Any, Dict


def list_datasets_tool(
    client,
    dataset_ids: list = None,
    data_type: str = None,
    dataset_name: str = None,
    dataset_name_contains: str = None,
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
                dataset_dict[attr] = value
            formatted_datasets.append(dataset_dict)

        return {"datasets": formatted_datasets, "total_count": len(formatted_datasets)}

    except Exception as e:
        return {"error": f"Error fetching datasets: {str(e)}"}
