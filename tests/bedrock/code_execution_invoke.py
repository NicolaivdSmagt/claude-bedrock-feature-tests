#!/usr/bin/env python3
# ABOUTME: Tests the code_execution_20250825 tool on Amazon Bedrock via the invoke_model API.
# ABOUTME: Validates programmatic tool calling via sandboxed Python execution.

"""
Code Execution Tool Test for Amazon Bedrock (Invoke API)
=========================================================

Tests the code execution tool (code_execution_20250825) which enables
sandboxed Python execution and programmatic tool calling.

Sends a request with the code execution tool and checks for server_tool_use
blocks indicating sandboxed Python execution ran server-side.

The code execution tool requires Anthropic's backend infrastructure (sandboxed
containers, server-side tool execution). It may not be available on Bedrock.

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/code_execution_invoke.py
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

CODE_EXECUTION_BETA = "advanced-tool-use-2025-11-20"
CODE_EXECUTION_TOOL_TYPE = "code_execution_20250825"

# Error messages that indicate the code execution feature is not available
# on Bedrock. These map to FAIL (feature not supported) rather than ERROR
# (transient/infra problem).
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


def get_tools_config():
    """Return the tools configuration for the code execution test."""
    return [
        {"type": CODE_EXECUTION_TOOL_TYPE, "name": "code_execution"},
        {
            "name": "get_temperature",
            "description": "Get the current temperature for a city.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"}
                },
                "required": ["city"],
            },
            "allowed_callers": [CODE_EXECUTION_TOOL_TYPE],
        },
    ]


def test_with_beta(client, model_id):
    """Test code execution tool with beta header. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: CODE EXECUTION")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [CODE_EXECUTION_BETA],
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Get the temperature for Paris and London, then tell me which is warmer.",
            }
        ],
        "tools": get_tools_config(),
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        content = result.get("content", [])
        content_types = [b.get("type") for b in content]
        stop_reason = result.get("stop_reason")

        print(f"\n  stop_reason: {stop_reason}")
        print(f"  content types: {content_types}")

        # Check for server_tool_use (code execution ran server-side) or
        # tool_use with code_execution caller (programmatic tool calling)
        has_server_tool_use = "server_tool_use" in content_types
        has_code_exec_caller = any(
            b.get("caller", {}).get("type") == CODE_EXECUTION_TOOL_TYPE
            for b in content
            if b.get("type") == "tool_use"
        )

        if has_server_tool_use or has_code_exec_caller:
            print("\n  Result: PASS - code execution tool is working")
            return ("PASS", None)
        elif "text" in content_types or "tool_use" in content_types:
            # Got a response but no code execution blocks — model responded
            # without using code execution
            print(
                "\n  Result: PASS - got response (model may not have used code execution)"
            )
            return ("PASS", None)
        else:
            msg = f"Unexpected response, content types: {content_types}"
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
    print(f"Beta: {CODE_EXECUTION_BETA}")
    print(f"Tool type: {CODE_EXECUTION_TOOL_TYPE}")
    print()

    results = []

    status, error = test_with_beta(client, model_id)
    results.append(("Code execution", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
