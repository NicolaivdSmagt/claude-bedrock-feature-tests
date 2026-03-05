#!/usr/bin/env python3
# ABOUTME: Tests the code_execution_20250825 tool on Amazon Bedrock via the Converse API.
# ABOUTME: Validates programmatic tool calling with and without beta headers.

"""
Code Execution Tool Test for Amazon Bedrock (Converse API)
===========================================================

Tests the code execution tool (code_execution_20250825) which enables
sandboxed Python execution and programmatic tool calling.

Code execution tool configuration and beta headers are passed via
additionalModelRequestFields since the Converse API doesn't natively
support these Anthropic-specific tool types. A placeholder toolSpec is
required in toolConfig.tools to satisfy the Converse schema.

Test cases:
1. Code execution with beta header — should produce server_tool_use blocks
2. Code execution without beta header — validates behavior without the beta

The code execution tool requires Anthropic's backend infrastructure (sandboxed
containers, server-side tool execution). It may not be available on Bedrock.

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/code_execution_converse.py
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
]

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


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other).
    Returns (status, error_msg)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def get_anthropic_tools_config():
    """Return the Anthropic-native tools configuration for code execution."""
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
    print("TEST 1: CODE EXECUTION WITH BETA HEADER")
    print("=" * 70)

    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Get the temperature for Paris and London, then tell me which is warmer."
                    }
                ],
            }
        ],
        "toolConfig": PLACEHOLDER_TOOL_CONFIG,
        "additionalModelRequestFields": {
            "anthropic_beta": [CODE_EXECUTION_BETA],
            "tools": get_anthropic_tools_config(),
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

        has_text = any("text" in b for b in content_blocks)
        has_tool_use = any("toolUse" in b for b in content_blocks)

        print(f"\n  stopReason: {stop_reason}")
        print(f"  has_text: {has_text}")
        print(f"  has_tool_use: {has_tool_use}")

        if has_text or has_tool_use:
            print("\n  Result: PASS - got response with beta header")
            return ("PASS", None)
        else:
            msg = "No text or tool use blocks in response"
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


def test_without_beta(client, model_id):
    """Test code execution tool without beta header. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 2: CODE EXECUTION WITHOUT BETA HEADER")
    print("=" * 70)

    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Get the temperature for Paris."}],
            }
        ],
        "toolConfig": PLACEHOLDER_TOOL_CONFIG,
        "additionalModelRequestFields": {
            # No anthropic_beta header
            "tools": get_anthropic_tools_config(),
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

        has_text = any("text" in b for b in content_blocks)
        has_tool_use = any("toolUse" in b for b in content_blocks)

        print(f"\n  stopReason: {stop_reason}")
        print(f"  has_text: {has_text}")
        print(f"  has_tool_use: {has_tool_use}")

        if has_text or has_tool_use:
            print("\n  Result: PASS - got response without beta header")
            return ("PASS", None)
        else:
            msg = "No text or tool use blocks in response"
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
    results.append(("With beta header", status, error))

    status, error = test_without_beta(client, model_id)
    results.append(("Without beta header", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
