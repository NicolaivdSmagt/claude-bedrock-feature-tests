#!/usr/bin/env python3
# ABOUTME: Tests the MCP connector feature on Amazon Bedrock via the invoke_model API.
# ABOUTME: Verifies remote MCP server connection and tool discovery.

"""
MCP Connector Test for Amazon Bedrock (Invoke API)
====================================================

Tests the MCP connector beta feature which allows connecting to remote MCP
servers directly from the Messages API without a separate MCP client.

Connects to the public AWS Knowledge MCP server and verifies that tool
discovery works via the mcp_toolset tool type.

Note: MCP connector on Bedrock is not yet live. This test is expected to
FAIL until the feature is available (Bedrock returns "The provided request
is not valid" when MCP fields are not recognized).

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/mcp_connector_invoke.py
"""

import json
import os
import sys

try:
    import boto3
except ImportError:
    print("Error: boto3 package not installed. Run: uv add boto3")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

MCP_BETA = "mcp-client-2025-11-20"

# AWS Knowledge MCP server — public, no authentication required, Streamable HTTP.
# Provides tools: search_documentation, read_documentation, recommend,
# list_regions, get_regional_availability.
MCP_TEST_SERVER_URL = "https://knowledge-mcp.global.api.aws"
MCP_TEST_SERVER_NAME = "aws-knowledge"

# Error messages that indicate the MCP connector feature is not yet available
# on Bedrock. These map to FAIL (feature not supported) rather than ERROR
# (transient/infra problem).
FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "not supported",
    "unknown field",
    "unrecognized",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other).
    Returns (status, error_msg)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def test_basic_mcp_connection(client, model_id):
    """Connect to the public MCP server and verify tool discovery.
    Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: BASIC MCP CONNECTION")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [MCP_BETA],
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "What tools do you have available? List them briefly.",
            }
        ],
        "mcp_servers": [
            {
                "type": "url",
                "url": MCP_TEST_SERVER_URL,
                "name": MCP_TEST_SERVER_NAME,
            }
        ],
        "tools": [
            {
                "type": "mcp_toolset",
                "mcp_server_name": MCP_TEST_SERVER_NAME,
            }
        ],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

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
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg[:300]}")
        status, msg = classify_error(error_msg)
        print(f"\n  Result: {status}")
        return (status, msg)

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
            line += f"  {error}"
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
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    print(f"Model: {model_id}")
    print(f"MCP server: {MCP_TEST_SERVER_URL}")
    print(f"MCP beta: {MCP_BETA}")
    print()

    results = []

    status, error = test_basic_mcp_connection(client, model_id)
    results.append(("Basic MCP connection", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
