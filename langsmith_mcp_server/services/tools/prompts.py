"""Tools for interacting with LangSmith prompts."""

from typing import Any, Dict

from langchain_core.load import dumpd
from langchain_core.load.serializable import Serializable
from langsmith import Client


def list_prompts_tool(client: Client, is_public: bool = False, limit: int = 20) -> Dict[str, Any]:
    """
    Fetch prompts from LangSmith with optional filtering.

    Args:
        client: LangSmith Client instance
        is_public (bool): Optional boolean to filter public/private prompts
        limit (int): Optional limit to the number of prompts to return

    Returns:
        Dictionary containing the prompts and metadata
    """
    try:
        response = client.list_prompts(is_public=is_public, limit=limit)

        # Convert list of tuples to a dictionary
        response_dict = dict(response)

        prompts_list = response_dict.get("repos", [])
        total_count = response_dict.get("total", len(prompts_list))

        formatted_prompts = []

        for prompt in prompts_list:
            prompt_dict = {}
            for attr in [
                "repo_handle",
                "description",
                "id",
                "is_public",
                "tags",
                "owner",
                "full_name",
                "num_likes",
                "num_downloads",
                "num_views",
            ]:
                prompt_dict[attr] = getattr(prompt, attr, None)

            if hasattr(prompt, "created_at"):
                prompt_dict["created_at"] = prompt.created_at.isoformat()
            if hasattr(prompt, "updated_at"):
                prompt_dict["updated_at"] = prompt.updated_at.isoformat()

            formatted_prompts.append(prompt_dict)

        return {"prompts": formatted_prompts, "total_count": total_count}

    except Exception as e:
        return {"error": f"Error fetching prompts: {str(e)}"}


def get_prompt_tool(client: Client, prompt_name: str = None, prompt_id: str = None) -> dict:
    """
    Get a specific prompt (including model/tool bindings) by its name or ID, and return its full
    string representation.

    Args:
        client: LangSmith Client instance
        prompt_name: The full name of the prompt (e.g., 'owner/repo')
        prompt_id: The UUID of the prompt

    Returns:
        String representation of the full prompt object (including tools and config)
    """
    try:
        if prompt_name:
            prompt = client.pull_prompt(prompt_identifier=prompt_name)
        elif prompt_id:
            prompt = client.pull_prompt(prompt_identifier=prompt_id)
        else:
            return {"error": "Error: Either prompt_name or prompt_id must be provided."}
        # Convert prompt object to JSON-serializable dictionary
        try:
            # LangChain objects (subclasses of Serializable) must be serialized
            # with dumpd() — model_dump() loses nested template content (e.g.
            # SystemMessagePromptTemplate becomes {}). For non-LangChain objects
            # the existing model_dump / dict / dumpd fallback chain still applies.
            if isinstance(prompt, Serializable):
                return dumpd(prompt)
            # Try using model_dump() if available (Pydantic v2)
            if hasattr(prompt, "model_dump"):
                return prompt.model_dump()
            # Try using dict() method if available
            elif hasattr(prompt, "dict"):
                return prompt.dict()
            # Fall back to dumpd() from langchain_core.load
            else:
                return dumpd(prompt)
        except Exception as conv_error:
            # If conversion fails, return a basic representation
            return {
                "type": type(prompt).__name__,
                "repr": repr(prompt),
                "error": f"Could not fully serialize prompt: {str(conv_error)}",
            }

    except Exception as e:
        return {"error": f"Error fetching prompt: {str(e)}"}
