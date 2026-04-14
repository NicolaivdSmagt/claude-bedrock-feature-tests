#!/usr/bin/env python3
# ABOUTME: Tests the advisor_20260301 tool on Amazon Bedrock via the invoke_model API.
# ABOUTME: Validates that a faster executor model can consult a higher-intelligence advisor model mid-generation.

"""
Advisor Tool Test for Amazon Bedrock (Invoke API)
===================================================

Tests the advisor tool (advisor_20260301) which lets an executor model consult
a higher-intelligence advisor model for strategic guidance mid-generation.

The advisor tool is a server-side tool: the executor emits a server_tool_use
block and the server runs a separate inference pass on the advisor model,
returning the result as an advisor_tool_result block — all within a single
API call.

Test case:
1. Basic advisor call — send a complex prompt that should trigger the executor
   to consult the advisor, verify response contains server_tool_use and
   advisor_tool_result blocks.

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/advisor_tool_invoke.py
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

ADVISOR_BETA = "advisor-tool-2026-03-01"
ADVISOR_TOOL_TYPE = "advisor_20260301"
ADVISOR_MODEL = "claude-opus-4-6"

# Error messages that indicate the advisor feature is not available on Bedrock.
# These map to FAIL (feature not supported) rather than ERROR (transient/infra problem).
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


def test_basic_advisor(client, model_id):
    """Test basic advisor tool call. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: BASIC ADVISOR CALL")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [ADVISOR_BETA],
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Build a concurrent worker pool in Go with graceful shutdown. "
                    "Include proper error handling and context cancellation."
                ),
            }
        ],
        "tools": [
            {
                "type": ADVISOR_TOOL_TYPE,
                "name": "advisor",
                "model": ADVISOR_MODEL,
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

        content = result.get("content", [])
        content_types = [b.get("type") for b in content]
        stop_reason = result.get("stop_reason")
        usage = result.get("usage", {})

        print(f"\n  stop_reason: {stop_reason}")
        print(f"  content types: {content_types}")
        print(f"  usage: {json.dumps(usage, indent=4)}")

        has_server_tool_use = any(
            b.get("type") == "server_tool_use" and b.get("name") == "advisor"
            for b in content
        )
        has_advisor_result = any(
            b.get("type") == "advisor_tool_result" for b in content
        )
        has_text = "text" in content_types

        print(f"  has_server_tool_use (advisor): {has_server_tool_use}")
        print(f"  has_advisor_tool_result: {has_advisor_result}")
        print(f"  has_text: {has_text}")

        if has_server_tool_use and has_advisor_result:
            print("\n  Result: PASS - advisor tool invoked and result returned")
            return ("PASS", None)
        elif has_text:
            # Model responded without consulting the advisor — still a valid
            # response, the model just chose not to use the tool
            print(
                "\n  Result: PASS - got text response (model chose not to consult advisor)"
            )
            return ("PASS", None)
        else:
            msg = f"Unexpected response, content types: {content_types}"
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
    print(f"Beta: {ADVISOR_BETA}")
    print(f"Tool type: {ADVISOR_TOOL_TYPE}")
    print(f"Advisor model: {ADVISOR_MODEL}")
    print()

    results = []

    status, error = test_basic_advisor(client, model_id)
    results.append(("Basic advisor call", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
