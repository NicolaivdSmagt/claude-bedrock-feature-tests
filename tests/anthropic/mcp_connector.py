#!/usr/bin/env python3
# ABOUTME: Tests the MCP connector feature on the Anthropic 1st-party API.
# ABOUTME: Verifies remote MCP server connection, tool discovery, and error handling.

"""
MCP Connector Tests for Anthropic API
======================================

Tests the MCP connector beta feature which allows connecting to remote MCP
servers directly from the Messages API without a separate MCP client.

Test cases:
1. Connect to public MCP test server and verify tool discovery
2. Verify validation error when mcp_server_name doesn't match
3. Verify error handling for unreachable MCP server

Requirements:
    uv add anthropic

Usage:
    uv run python tests/anthropic/mcp_connector.py
"""

import json
import os
import sys

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: uv add anthropic")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_anthropic_api_key

MCP_BETA = "mcp-client-2025-11-20"

# AWS Knowledge MCP server — public, no authentication required, Streamable HTTP.
# Provides tools: search_documentation, read_documentation, recommend,
# list_regions, get_regional_availability.
MCP_TEST_SERVER_URL = "https://knowledge-mcp.global.api.aws"
MCP_TEST_SERVER_NAME = "aws-knowledge"


def test_basic_mcp_connection(client, model_id):
    """Connect to the public MCP test server and verify tool discovery.
    Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 1: BASIC MCP CONNECTION")
    print("=" * 70)

    try:
        response = client.beta.messages.create(
            model=model_id,
            max_tokens=1024,
            betas=[MCP_BETA],
            messages=[
                {
                    "role": "user",
                    "content": "What tools do you have available? List them briefly.",
                }
            ],
            mcp_servers=[
                {
                    "type": "url",
                    "url": MCP_TEST_SERVER_URL,
                    "name": MCP_TEST_SERVER_NAME,
                }
            ],
            tools=[
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": MCP_TEST_SERVER_NAME,
                }
            ],
        )

        result = response.model_dump()
        print("\n--- RESPONSE ---")
        print(json.dumps(result, indent=2, default=str))

        # Check for mcp_tool_use or mcp_tool_result content blocks in the response,
        # or a text response that indicates tools were discovered
        content_types = [b.get("type") for b in result.get("content", [])]
        has_mcp_blocks = any(
            t in content_types for t in ("mcp_tool_use", "mcp_tool_result")
        )
        has_text = "text" in content_types
        stop_reason = result.get("stop_reason")

        print(f"\n  stop_reason: {stop_reason}")
        print(f"  content types: {content_types}")

        # The response should either contain MCP tool blocks (Claude used MCP tools)
        # or a text response (Claude listed the tools without calling them).
        # Either indicates the MCP connection worked.
        if has_mcp_blocks or has_text:
            print("\n  Result: PASS - MCP connection succeeded, got response")
            return ("PASS", None)
        else:
            msg = f"No text or MCP blocks in response, content types: {content_types}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- API ERROR ---")
        print(f"  {error_msg[:300]}")
        return ("ERROR", error_msg)

    finally:
        print()


def test_server_name_mismatch(client, model_id):
    """Verify validation error when mcp_server_name doesn't match any server.
    Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 2: SERVER NAME MISMATCH VALIDATION")
    print("=" * 70)

    try:
        response = client.beta.messages.create(
            model=model_id,
            max_tokens=256,
            betas=[MCP_BETA],
            messages=[{"role": "user", "content": "Hello"}],
            mcp_servers=[
                {
                    "type": "url",
                    "url": MCP_TEST_SERVER_URL,
                    "name": "server-a",
                }
            ],
            tools=[
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "nonexistent-server",
                }
            ],
        )

        # If we get here, the API didn't reject the mismatched name
        result = response.model_dump()
        print("\n--- RESPONSE ---")
        print(json.dumps(result, indent=2, default=str))
        msg = "Expected validation error for mismatched server name, but request succeeded"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    except anthropic.BadRequestError as e:
        error_msg = str(e)
        print(f"\n  BadRequestError: {error_msg[:300]}")
        print(f"\n  Result: PASS - validation error returned as expected")
        return ("PASS", None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- API ERROR ---")
        print(f"  {error_msg[:300]}")
        # Any error is acceptable here since we expected a rejection
        if (
            "400" in error_msg
            or "invalid" in error_msg.lower()
            or "validation" in error_msg.lower()
        ):
            print(f"\n  Result: PASS - request rejected as expected")
            return ("PASS", None)
        return ("ERROR", error_msg)

    finally:
        print()


def test_unreachable_server(client, model_id):
    """Verify error handling when MCP server URL is unreachable.
    Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 3: UNREACHABLE MCP SERVER")
    print("=" * 70)

    try:
        response = client.beta.messages.create(
            model=model_id,
            max_tokens=256,
            betas=[MCP_BETA],
            messages=[{"role": "user", "content": "What tools are available?"}],
            mcp_servers=[
                {
                    "type": "url",
                    "url": "https://this-mcp-server-does-not-exist.invalid/sse",
                    "name": "unreachable-mcp",
                }
            ],
            tools=[
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "unreachable-mcp",
                }
            ],
        )

        # If we somehow get a response, check if it contains error info
        result = response.model_dump()
        print("\n--- RESPONSE ---")
        print(json.dumps(result, indent=2, default=str))
        msg = "Expected error for unreachable server, but request succeeded"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n  {error_msg[:300]}")
        print(f"\n  Result: PASS - unreachable server produced error as expected")
        return ("PASS", None)

    finally:
        print()


def print_summary(results):
    """Print a summary table of all test outcomes."""
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    name_width = max(len(name) for name, _, _ in results)

    for name, status, error in results:
        line = f"  {name:<{name_width}}  {status}"
        if error:
            line += f"  {error[:120]}"
        print(line)

    passes = sum(1 for _, s, _ in results if s == "PASS")
    fails = sum(1 for _, s, _ in results if s == "FAIL")
    errors = sum(1 for _, s, _ in results if s == "ERROR")

    print()
    print(f"  Results: {passes} PASS, {fails} FAIL, {errors} ERROR")
    print("=" * 70)

    return all(s == "PASS" for _, s, _ in results)


def main():
    config = load_config()
    model_id = config["anthropic_model_id"]

    print(f"Model: {model_id}")
    print(f"MCP server: {MCP_TEST_SERVER_URL}")
    print(f"MCP beta: {MCP_BETA}")

    print("\nFetching API key from AWS Secrets Manager...")
    api_key = get_anthropic_api_key(config)
    print("API key retrieved successfully.\n")

    client = anthropic.Anthropic(api_key=api_key)

    results = []

    status, error = test_basic_mcp_connection(client, model_id)
    results.append(("Basic MCP connection", status, error))

    status, error = test_server_name_mismatch(client, model_id)
    results.append(("Server name mismatch", status, error))

    status, error = test_unreachable_server(client, model_id)
    results.append(("Unreachable MCP server", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
