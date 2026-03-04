"""
Session-based load test for the LangSmith MCP server (HTTP streamable).

Opens N concurrent MCP sessions; each session initializes, lists tools (via
langchain-mcp-adapters), and calls the list_prompts tool several times.
Uses the same pattern as real clients: one session per "user", multiple calls per session.

Usage:
  uv sync --group load
  uv run python tests/load_test_sessions.py [--url URL] [--sessions N] [--calls-per-session K]
  uv run python tests/load_test_sessions.py --report load_test_report   # writes load_test_report.json

Environment:
  Loads .env from project root. LANGSMITH_API_KEY is required (or set --api-key).
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# .env at project root (parent of tests/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    print("Install MCP client: uv sync --group load", file=sys.stderr)
    sys.exit(1)

try:
    from langchain_mcp_adapters.tools import load_mcp_tools
except ImportError:
    print("Install langchain-mcp-adapters: uv sync --group load", file=sys.stderr)
    sys.exit(1)


async def run_one_session(
    url: str,
    headers: dict[str, str],
    calls_per_session: int,
    session_id: int,
    debug: bool = False,
) -> tuple[int, int, float, Exception | None, str]:
    """Open one MCP session, initialize, list tools, call list_prompts `calls_per_session` times."""
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    successes = 0
    failures = 0
    first_error: Exception | None = None
    try:
        if debug:
            print(f"  [session {session_id}] connecting to {url} ...", flush=True)
        async with streamablehttp_client(url, headers=headers) as (read, write, _):
            if debug:
                print(f"  [session {session_id}] connected, initializing ...", flush=True)
            async with ClientSession(read, write) as session:
                await session.initialize()
                if debug:
                    print(f"  [session {session_id}] loading tools ...", flush=True)
                tools = await load_mcp_tools(session)
                list_prompts_tool = next(
                    (t for t in tools if getattr(t, "name", None) == "list_prompts"), None
                )
                if not list_prompts_tool:
                    first_error = RuntimeError(
                        f"list_prompts not found; got {[getattr(t, 'name', None) for t in tools]}"
                    )
                    if debug:
                        print(f"  [session {session_id}] {first_error}", flush=True)
                    return (
                        0,
                        calls_per_session,
                        time.perf_counter() - start,
                        first_error,
                        started_at,
                    )
                if debug:
                    print(
                        f"  [session {session_id}] calling list_prompts x{calls_per_session} ...",
                        flush=True,
                    )
                for _ in range(calls_per_session):
                    try:
                        await list_prompts_tool.ainvoke({"limit": 5, "is_public": "false"})
                        successes += 1
                    except Exception as e:
                        if first_error is None:
                            first_error = e
                        failures += 1
    except Exception as e:
        first_error = e
        failures += calls_per_session
        if debug:
            print(f"  [session {session_id}] error: {e}", flush=True)
            traceback.print_exc()
    elapsed = time.perf_counter() - start
    return (successes, failures, elapsed, first_error, started_at)


def write_report(
    report_path: Path,
    *,
    url: str,
    sessions: int,
    calls_per_session: int,
    total_ok: int,
    total_fail: int,
    total_elapsed: float,
    session_results: list[dict],
    first_error_str: str | None,
) -> None:
    """Write a JSON report to the given path (stem used for .json)."""
    stem = report_path.stem
    dir_path = report_path.parent
    json_path = dir_path / f"{stem}.json"

    throughput = (total_ok + total_fail) / total_elapsed if total_elapsed > 0 else 0.0

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {"url": url, "sessions": sessions, "calls_per_session": calls_per_session},
        "summary": {
            "total_ok": total_ok,
            "total_fail": total_fail,
            "wall_time_sec": round(total_elapsed, 2),
            "throughput_calls_per_sec": round(throughput, 1),
        },
        "sessions": session_results,
        "first_error": first_error_str,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report written to {json_path}")


async def main(
    url: str,
    api_key: str,
    sessions: int,
    calls_per_session: int,
    debug: bool = False,
    report_path: Path | None = None,
) -> None:
    headers = {"LANGSMITH-API-KEY": api_key}
    print(f"Starting load test: {sessions} sessions, {calls_per_session} list_prompts calls each")
    print(f"URL: {url}")
    if debug:
        print("Debug: on")
    if report_path:
        print(f"Report: {report_path.stem}.json")
    print()
    run_started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    results = await asyncio.gather(
        *[
            run_one_session(url, headers, calls_per_session, i, debug=debug)
            for i in range(sessions)
        ],
        return_exceptions=True,
    )
    total_elapsed = time.perf_counter() - start
    total_ok = 0
    total_fail = 0
    first_error_shown: Exception | None = None
    session_results: list[dict] = []

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            if first_error_shown is None:
                first_error_shown = r
            print(f"Session {i}: error {r}")
            total_fail += calls_per_session
            session_results.append(
                {
                    "session_id": i,
                    "started_at": run_started_at,
                    "ok": 0,
                    "fail": calls_per_session,
                    "elapsed_sec": 0.0,
                    "error": str(r),
                }
            )
        else:
            ok, fail, el, err, started_at = r
            total_ok += ok
            total_fail += fail
            if first_error_shown is None and err is not None:
                first_error_shown = err
            session_results.append(
                {
                    "session_id": i,
                    "started_at": started_at,
                    "ok": ok,
                    "fail": fail,
                    "elapsed_sec": el,
                    "error": str(err) if err else None,
                }
            )
            if fail:
                print(f"Session {i}: {ok} ok, {fail} fail, {el:.2f}s")

    if first_error_shown is not None and total_fail > 0:
        print()
        print("First error (for debugging):")
        print(f"  {type(first_error_shown).__name__}: {first_error_shown}")
        if debug:
            traceback.print_exception(
                type(first_error_shown), first_error_shown, first_error_shown.__traceback__
            )

    print()
    print(f"Total: {total_ok} ok, {total_fail} fail, {total_elapsed:.2f}s wall")
    if total_elapsed > 0:
        print(f"Throughput: {(total_ok + total_fail) / total_elapsed:.1f} calls/s")

    if report_path is not None:
        first_error_str = (
            f"{type(first_error_shown).__name__}: {first_error_shown}"
            if first_error_shown is not None
            else None
        )
        write_report(
            report_path,
            url=url,
            sessions=sessions,
            calls_per_session=calls_per_session,
            total_ok=total_ok,
            total_fail=total_fail,
            total_elapsed=total_elapsed,
            session_results=session_results,
            first_error_str=first_error_str,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Session-based MCP load test")
    parser.add_argument("--url", default="http://localhost:8000/mcp", help="MCP endpoint URL")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LANGSMITH_API_KEY"),
        help="LANGSMITH_API_KEY (default: from .env)",
    )
    parser.add_argument("--sessions", type=int, default=10, help="Number of concurrent sessions")
    parser.add_argument(
        "--calls-per-session", type=int, default=3, help="list_prompts calls per session"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print step-by-step and first error traceback"
    )
    parser.add_argument(
        "--report",
        type=Path,
        metavar="PATH",
        help="Write JSON report after run (PATH stem used for .json, e.g. load_test_report)",
    )
    args = parser.parse_args()
    if not args.api_key:
        print("Set LANGSMITH_API_KEY in .env or pass --api-key", file=sys.stderr)
        sys.exit(1)
    asyncio.run(
        main(
            args.url,
            args.api_key,
            args.sessions,
            args.calls_per_session,
            debug=args.debug,
            report_path=args.report,
        )
    )
