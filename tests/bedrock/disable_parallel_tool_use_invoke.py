#!/usr/bin/env python3
# ABOUTME: Tests disabling parallel tool use on Amazon Bedrock via the invoke_model API.
# ABOUTME: Validates tool_choice.disable_parallel_tool_use limits responses to one tool call.

"""
Disable Parallel Tool Use Test for Amazon Bedrock (Invoke API)
================================================================

Tests the `disable_parallel_tool_use` flag in tool_choice. When set to true,
it ensures Claude uses at most one tool per response (when tool_choice type
is "auto") or exactly one tool (when type is "any" or "tool").

Test cases:
1. tool_choice.auto + disable_parallel_tool_use=true — at most 1 tool_use block
2. tool_choice.any + disable_parallel_tool_use=true — exactly 1 tool_use block

Both use a prompt that would normally trigger parallel calls (weather in
multiple cities plus time). The existing parallel_tool_use_invoke.py test
confirms the baseline produces multiple parallel calls for this prompt, so
observing ≤ 1 call here validates the disable flag is taking effect.

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/disable_parallel_tool_use_invoke.py
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

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current time in a given timezone",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone, e.g. America/New_York",
                }
            },
            "required": ["timezone"],
        },
    },
]

PROMPT = (
    "What's the weather in San Francisco and New York City, and what time "
    "is it there?"
)

FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "not supported",
    "unknown field",
    "unrecognized",
    "does not match any of the expected tags",
    "disable_parallel_tool_use",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def count_tool_uses(content):
    """Count tool_use blocks in response content."""
    return sum(1 for b in content if b.get("type") == "tool_use")


def test_auto_with_disable(client, model_id):
    """tool_choice.auto + disable_parallel_tool_use=true → at most 1 tool_use."""
    print("=" * 70)
    print("TEST: AUTO + disable_parallel_tool_use=true")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "tools": TOOLS,
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": [{"role": "user", "content": PROMPT}],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        content = result.get("content", [])
        stop_reason = result.get("stop_reason")
        tool_use_count = count_tool_uses(content)

        print(f"\n  stop_reason: {stop_reason}")
        print(f"  tool_use blocks: {tool_use_count}")

        if tool_use_count <= 1:
            print(
                f"\n  Result: PASS - got {tool_use_count} tool_use block(s) "
                f"(≤ 1 as required by disable_parallel_tool_use)"
            )
            return ("PASS", None)
        else:
            msg = (
                f"Expected ≤ 1 tool_use block with disable_parallel_tool_use=true, "
                f"got {tool_use_count}"
            )
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        status, msg = classify_error(error_msg)
        print(f"\n  Result: {status}")
        return (status, msg)

    finally:
        print()


def test_any_with_disable(client, model_id):
    """tool_choice.any + disable_parallel_tool_use=true → exactly 1 tool_use."""
    print("=" * 70)
    print("TEST: ANY + disable_parallel_tool_use=true")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "tools": TOOLS,
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        "messages": [{"role": "user", "content": PROMPT}],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        content = result.get("content", [])
        stop_reason = result.get("stop_reason")
        tool_use_count = count_tool_uses(content)

        print(f"\n  stop_reason: {stop_reason}")
        print(f"  tool_use blocks: {tool_use_count}")

        if tool_use_count == 1:
            print(
                "\n  Result: PASS - got exactly 1 tool_use block "
                "(as required by type=any + disable_parallel_tool_use)"
            )
            return ("PASS", None)
        else:
            msg = (
                f"Expected exactly 1 tool_use block with type=any + "
                f"disable_parallel_tool_use=true, got {tool_use_count}"
            )
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

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

    print()
    print(f"  Results: {passes} PASS, {fails} FAIL, {errors} ERROR")
    print("=" * 70)

    return all(s == "PASS" for _, s, _ in results)


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    print(f"Model: {model_id}")
    print(f"Prompt: {PROMPT}")
    print()

    results = []

    status, error = test_auto_with_disable(client, model_id)
    results.append(("auto + disable_parallel_tool_use", status, error))

    status, error = test_any_with_disable(client, model_id)
    results.append(("any + disable_parallel_tool_use", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
