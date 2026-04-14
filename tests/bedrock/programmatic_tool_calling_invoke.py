#!/usr/bin/env python3
# ABOUTME: Tests programmatic tool calling on Amazon Bedrock via the invoke_model API.
# ABOUTME: Validates that code_execution_20260120 can call tools with allowed_callers configuration.

"""
Programmatic Tool Calling Test for Amazon Bedrock (Invoke API)
===============================================================

Tests programmatic tool calling, a feature of code_execution_20260120 that
allows Claude to write code that calls user-defined tools from within the
sandboxed execution environment.

This reduces latency and token consumption for multi-tool workflows by allowing
Claude to loop, filter, and process data programmatically before returning
results to the model context.

Key API elements:
- code_execution_20260120 tool (not older versions)
- Custom tools with allowed_callers: ["code_execution_20260120"]
- Response includes tool_use blocks with caller.type = "code_execution_20260120"

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/programmatic_tool_calling_invoke.py
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

CODE_EXECUTION_TOOL_TYPE = "code_execution_20260120"

# Error messages that indicate the programmatic tool calling feature is not
# available on Bedrock. These map to FAIL (feature not supported) rather than
# ERROR (transient/infra problem).
FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "not supported",
    "unknown field",
    "unrecognized",
    "unknown tool type",
    "does not match any of the expected tags",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other).
    Returns (status, error_msg)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def test_programmatic_tool_calling(client, model_id):
    """Test basic programmatic tool calling. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: PROGRAMMATIC TOOL CALLING")
    print("=" * 70)

    # Define a mock database query tool that will be called programmatically
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Query sales data for regions West, East, and Central. "
                    "For each region, call get_sales(region_name) and tell me "
                    "which region had the highest revenue."
                ),
            }
        ],
        "tools": [
            {
                "type": CODE_EXECUTION_TOOL_TYPE,
                "name": "code_execution",
            },
            {
                "name": "get_sales",
                "description": (
                    "Get sales data for a specific region. Returns JSON with "
                    "fields: region (string), revenue (number), orders (number)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "string",
                            "description": "Region name (West, East, Central, etc.)",
                        }
                    },
                    "required": ["region"],
                },
                "allowed_callers": [CODE_EXECUTION_TOOL_TYPE],
            },
        ],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        # Validate the response structure
        content = result.get("content", [])

        # Look for server_tool_use (code_execution)
        server_tool_uses = [
            block for block in content if block.get("type") == "server_tool_use"
        ]

        # Look for tool_use with programmatic caller
        programmatic_tool_uses = [
            block for block in content
            if block.get("type") == "tool_use"
            and block.get("caller", {}).get("type") == CODE_EXECUTION_TOOL_TYPE
        ]

        if not server_tool_uses:
            msg = "No server_tool_use block found (expected code_execution)"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        if not programmatic_tool_uses:
            msg = (
                "No programmatic tool_use blocks found (expected get_sales calls "
                f"with caller.type={CODE_EXECUTION_TOOL_TYPE})"
            )
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        # Validate the caller structure
        for tool_use in programmatic_tool_uses:
            caller = tool_use.get("caller", {})
            if caller.get("type") != CODE_EXECUTION_TOOL_TYPE:
                msg = f"Invalid caller type: {caller.get('type')}"
                print(f"\n  Result: FAIL - {msg}")
                return ("FAIL", msg)

            if "tool_id" not in caller:
                msg = "Missing tool_id in caller field"
                print(f"\n  Result: FAIL - {msg}")
                return ("FAIL", msg)

            # Verify the tool_id references a code_execution server_tool_use
            tool_id = caller["tool_id"]
            matching_code_exec = [
                block for block in server_tool_uses
                if block.get("id") == tool_id
            ]
            if not matching_code_exec:
                msg = f"caller.tool_id {tool_id} doesn't match any server_tool_use"
                print(f"\n  Result: FAIL - {msg}")
                return ("FAIL", msg)

        print(
            f"\n  Result: PASS - Found {len(programmatic_tool_uses)} programmatic "
            f"tool call(s) from code execution"
        )
        return ("PASS", None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
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
    print(f"\n  Results: {passes} PASS, {fails} FAIL, {errors} ERROR")
    print("=" * 70)
    return all(s == "PASS" for _, s, _ in results)


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    print(f"Model: {model_id}")
    print()

    results = []

    status, error = test_programmatic_tool_calling(client, model_id)
    results.append(("Programmatic tool calling", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
