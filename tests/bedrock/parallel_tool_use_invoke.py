#!/usr/bin/env python3
# ABOUTME: Tests parallel tool use on Amazon Bedrock via the invoke_model API.
# ABOUTME: Verifies that Claude returns multiple tool_use blocks in a single response.

"""
Parallel Tool Use Test for Amazon Bedrock (Invoke API)
=======================================================

Tests that Claude can make multiple tool calls in a single response via
invoke_model. Presents a prompt that naturally requires parallel lookups
(weather + time in multiple cities) and validates:

1. Multiple tool_use blocks in one response (parallel calls)
2. Tool results can be sent back in a single user message
3. Claude synthesizes a coherent final response from all results

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/parallel_tool_use_invoke.py
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


def simulate_tool_result(tool_use):
    """Return a simulated tool result based on the tool call."""
    if tool_use["name"] == "get_weather":
        location = str(tool_use["input"].get("location", ""))
        if "san francisco" in location.lower() or "sf" in location.lower():
            return "San Francisco: 68F, partly cloudy"
        elif "new york" in location.lower() or "nyc" in location.lower():
            return "New York: 45F, clear skies"
        else:
            return f"{location}: 72F, sunny"
    elif tool_use["name"] == "get_time":
        timezone = str(tool_use["input"].get("timezone", ""))
        if "los_angeles" in timezone.lower() or "pacific" in timezone.lower():
            return "2:30 PM PST"
        elif "new_york" in timezone.lower() or "eastern" in timezone.lower():
            return "5:30 PM EST"
        else:
            return "3:30 PM"
    return "result unavailable"


def test_parallel_tool_use(client, model_id):
    """Test parallel tool use in a single response. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: PARALLEL TOOL USE")
    print("=" * 70)

    messages = [
        {
            "role": "user",
            "content": "What's the weather in SF and NYC, and what time is it there?",
        }
    ]

    # Request 1: expect multiple tool_use blocks
    print("\n--- REQUEST 1: Initial prompt (expecting parallel tool calls) ---")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": messages,
        "tools": TOOLS,
    }

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    content = result.get("content", [])
    stop_reason = result.get("stop_reason")
    tool_uses = [b for b in content if b.get("type") == "tool_use"]

    print(f"  stop_reason: {stop_reason}")
    print(f"  content blocks: {len(content)}")
    print(f"  tool_use blocks: {len(tool_uses)}")
    for tu in tool_uses:
        print(f"    - {tu['name']}({json.dumps(tu['input'])})")

    if len(tool_uses) < 2:
        msg = f"Expected >= 2 parallel tool calls, got {len(tool_uses)}"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    # Request 2: send all tool results back in one message
    print("\n--- REQUEST 2: Sending tool results (expecting synthesized response) ---")

    tool_results = []
    for tu in tool_uses:
        result_text = simulate_tool_result(tu)
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_text,
            }
        )
        print(f"  {tu['name']} -> {result_text}")

    messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": tool_results})

    body2 = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": messages,
        "tools": TOOLS,
    }

    try:
        response2 = client.invoke_model(modelId=model_id, body=json.dumps(body2))
        result2 = json.loads(response2["body"].read())
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    content2 = result2.get("content", [])
    stop_reason2 = result2.get("stop_reason")
    text_blocks = [b for b in content2 if b.get("type") == "text"]

    print(f"\n  stop_reason: {stop_reason2}")
    if text_blocks:
        print(f"  Response: {text_blocks[0]['text'][:200]}")

    if stop_reason2 != "end_turn":
        msg = (
            f"Expected stop_reason='end_turn' for final response, got '{stop_reason2}'"
        )
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if not text_blocks:
        msg = "No text in final response after tool results"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - {len(tool_uses)} parallel tool calls, synthesized response received"
    )
    return ("PASS", None)


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
    print()

    results = []

    status, error = test_parallel_tool_use(client, model_id)
    results.append(("Parallel tool use", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
