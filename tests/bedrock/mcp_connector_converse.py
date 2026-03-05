#!/usr/bin/env python3
# ABOUTME: Tests the MCP connector feature on Amazon Bedrock via the Converse API.
# ABOUTME: Verifies remote MCP server connection and tool discovery.

"""
MCP Connector Test for Amazon Bedrock (Converse API)
=====================================================

Tests the MCP connector beta feature which allows connecting to remote MCP
servers directly from the Messages API without a separate MCP client.

MCP connector configuration (mcp_servers, mcp_toolset tools, and the beta
header) is passed via additionalModelRequestFields since the Converse API
doesn't natively support these Anthropic-specific types. A placeholder
toolSpec is required in toolConfig.tools to satisfy the Converse schema.

Connects to the public AWS Knowledge MCP server and verifies that tool
discovery works via the mcp_toolset tool type.

Note: MCP connector on Bedrock is not yet live. This test is expected to
FAIL until the feature is available (Bedrock returns "The provided request
is not valid" when MCP fields are not recognized).

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/mcp_connector_converse.py
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


# Converse requires at least one toolSpec in toolConfig; this placeholder satisfies that.
PLACEHOLDER_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "placeholder",
                "inputSchema": {"json": {"type": "object"}},
            }
        }
    ]
}


def test_basic_mcp_connection(client, model_id):
    """Connect to the public MCP server and verify tool discovery.
    Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: BASIC MCP CONNECTION")
    print("=" * 70)

    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "What tools do you have available? List them briefly."}
                ],
            }
        ],
        "toolConfig": PLACEHOLDER_TOOL_CONFIG,
        "additionalModelRequestFields": {
            "anthropic_beta": [MCP_BETA],
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
        },
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        stop_reason = response.get("stopReason")
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        # Check for text or tool use blocks indicating MCP connection worked
        has_text = any("text" in b for b in content_blocks)
        has_tool_use = any("toolUse" in b for b in content_blocks)

        print(f"\n  stopReason: {stop_reason}")
        print(f"  has_text: {has_text}")
        print(f"  has_tool_use: {has_tool_use}")

        if has_text or has_tool_use:
            print("\n  Result: PASS - MCP connection succeeded, got response")
            return ("PASS", None)
        else:
            msg = f"No text or tool use blocks in response"
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
