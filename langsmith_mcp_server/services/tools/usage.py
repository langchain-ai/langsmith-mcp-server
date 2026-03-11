"""Tools for LangSmith billing/usage (trace counts). Uses REST API only."""

import copy
import json
import urllib.parse
import urllib.request
from typing import Any

_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


def _request(
    api_key: str,
    endpoint: str,
    path: str,
    params: dict[str, str] | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any] | list[Any]:
    """GET request to LangSmith API. Returns JSON (dict or list)."""
    base = (endpoint or _DEFAULT_ENDPOINT).rstrip("/")
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("X-API-Key", api_key)
    if tenant_id:
        req.add_header("X-Tenant-Id", tenant_id)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {body}"}
    except OSError as e:
        return {"error": str(e)}


def _list_workspaces(
    api_key: str, endpoint: str, tenant_id: str | None = None
) -> dict[str, Any] | list[Any]:
    """GET /api/v1/workspaces."""
    return _request(api_key, endpoint, "/api/v1/workspaces", tenant_id=tenant_id)


def _get_workspace_by_id(
    api_key: str,
    endpoint: str,
    workspace_id: str,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """GET /api/v1/workspaces/{id}."""
    out = _request(api_key, endpoint, f"/api/v1/workspaces/{workspace_id}", tenant_id=tenant_id)
    return out if isinstance(out, dict) else {"error": "Unexpected response"}


def _build_workspace_id_to_name(
    api_key: str,
    endpoint: str,
    single_workspace: str | None,
    tenant_id: str | None = None,
) -> dict[str, str]:
    """Build workspace_id -> name. If single_workspace set, fetch only that one."""
    id_to_name: dict[str, str] = {}
    if single_workspace:
        single_workspace = single_workspace.strip()
        if len(single_workspace) == 36 and single_workspace.count("-") == 4:
            resp = _get_workspace_by_id(api_key, endpoint, single_workspace, tenant_id=tenant_id)
            if isinstance(resp, dict) and "error" not in resp and resp.get("id"):
                name = resp.get("display_name") or resp.get("name") or single_workspace
                id_to_name[str(resp["id"])] = name
                return id_to_name
        ws_resp = _list_workspaces(api_key, endpoint, tenant_id=tenant_id)
        workspaces: list[dict] = []
        if isinstance(ws_resp, list):
            workspaces = [w for w in ws_resp if isinstance(w, dict)]
        elif isinstance(ws_resp, dict) and "error" not in ws_resp:
            workspaces = list(ws_resp.get("workspaces") or ws_resp.get("items") or [])
        single_lower = single_workspace.lower()
        for w in workspaces:
            if not w.get("id"):
                continue
            wid = str(w["id"])
            name = w.get("display_name") or w.get("name") or wid
            if wid == single_workspace or (name or "").lower() == single_lower:
                id_to_name[wid] = name
                return id_to_name
        return id_to_name
    ws_resp = _list_workspaces(api_key, endpoint, tenant_id=tenant_id)
    if isinstance(ws_resp, list):
        for w in ws_resp:
            if isinstance(w, dict) and w.get("id"):
                id_to_name[str(w["id"])] = w.get("display_name") or w.get("name") or str(w["id"])
    elif isinstance(ws_resp, dict) and "error" not in ws_resp:
        for w in ws_resp.get("workspaces") or ws_resp.get("items") or []:
            if isinstance(w, dict) and w.get("id"):
                id_to_name[str(w["id"])] = w.get("display_name") or w.get("name") or str(w["id"])
    return id_to_name


def _augment_usage_groups_with_names(
    usage: list[dict],
    workspace_id_to_name: dict[str, str],
    only_workspace_id: str | None = None,
) -> list[dict]:
    """Put workspace_name next to each group value; optionally filter to one workspace.

    All workspace UUIDs from the billing API are preserved. If a UUID is not in
    workspace_id_to_name (e.g. workspace not in /api/v1/workspaces list), it is
    still included with workspace_name set to the raw UUID.
    """
    result = copy.deepcopy(usage)
    for item in result:
        groups = (item or {}).get("groups")
        if not isinstance(groups, dict):
            continue
        new_groups: dict[str, Any] = {}
        for uid, val in groups.items():
            if only_workspace_id and uid != only_workspace_id:
                continue
            # Keep every UUID; use id-to-name map or fall back to UUID as name
            name = workspace_id_to_name.get(uid) or uid
            new_groups[uid] = {"workspace_name": name, "value": val}
        item["groups"] = new_groups
    return result


def get_billing_usage_tool(
    api_key: str,
    endpoint: str,
    starting_on: str,
    ending_before: str,
    on_current_plan: bool = True,
    workspace: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any] | list[dict]:
    """
    Fetch org billing usage (trace counts) with workspace names inline.

    Uses GET /api/v1/orgs/current/billing/usage and optionally
    GET /api/v1/workspaces (or single workspace) to resolve names.
    When workspace is provided (UUID or name), only that workspace's
    entries are included in each metric's groups.

    Args:
        api_key: LangSmith API key.
        endpoint: API base URL (e.g. https://api.smith.langchain.com).
        starting_on: Start of range (ISO 8601).
        ending_before: End of range (ISO 8601).
        on_current_plan: If true, only usage on current plan.
        workspace: Optional single workspace UUID or name to filter to.
        tenant_id: Optional workspace/tenant ID for org-scoped API keys (X-Tenant-Id header).

    Returns:
        List of billing metrics with groups as
        { uuid: { "workspace_name": "<name>", "value": <number> } },
        or error dict with "error" key.
    """
    params = {
        "starting_on": starting_on,
        "ending_before": ending_before,
        "on_current_plan": "true" if on_current_plan else "false",
    }
    raw = _request(
        api_key, endpoint, "/api/v1/orgs/current/billing/usage", params, tenant_id=tenant_id
    )
    if isinstance(raw, dict) and "error" in raw:
        return raw
    if not isinstance(raw, list) or not raw:
        return {"error": "Unexpected billing usage response"}
    workspace_id_to_name = _build_workspace_id_to_name(
        api_key, endpoint, workspace, tenant_id=tenant_id
    )
    only_workspace_id: str | None = None
    if workspace and workspace_id_to_name:
        only_workspace_id = next(iter(workspace_id_to_name.keys()), None)
    return _augment_usage_groups_with_names(raw, workspace_id_to_name, only_workspace_id)
