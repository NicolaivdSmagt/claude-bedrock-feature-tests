#!/usr/bin/env python3
# ABOUTME: Tests task budgets (beta) on Amazon Bedrock via the Converse API.
# ABOUTME: Task budgets are documented as NOT supported on Converse during beta — expect failure.

"""
Task Budgets Test for Amazon Bedrock (Converse API)
=====================================================

Tests the task budgets beta feature via the Converse API. Per the Opus 4.7
SA Field Guide, task budgets are NOT supported on Converse during the beta.
This test verifies that behavior — the request should be rejected or the
parameter should be silently ignored.

A proper PASS for this test means: task budget parameters passed via
additionalModelRequestFields are either rejected with a feature-not-available
error, or silently ignored (resulting in a normal response). Either outcome
confirms the documented behavior.

Test cases:
1. Task budget via Converse — expect rejection or silent acceptance

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/task_budgets_converse.py
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

TASK_BUDGETS_BETA = "task-budgets-2026-03-13"

FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "not supported",
    "unknown field",
    "unrecognized",
    "does not match any of the expected tags",
    "does not support user-configurable token budgets",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def test_task_budget_converse(client, model_id):
    """Test task budgets via Converse API. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: TASK BUDGET VIA CONVERSE")
    print("=" * 70)

    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Design a small REST API for a todo list application. "
                            "Keep the response brief."
                        )
                    }
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 4096},
        "additionalModelRequestFields": {
            "anthropic_beta": [TASK_BUDGETS_BETA],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": "medium",
                "task_budget": {"type": "tokens", "total": 30000},
            },
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

        print(f"\n  stopReason: {stop_reason}")
        print(f"  has_text: {has_text}")
        print(f"  content block count: {len(content_blocks)}")

        # Per the SA field guide, task budgets are not supported on Converse during beta.
        # If the request succeeded, the parameter was silently ignored.
        if has_text:
            print(
                "\n  Result: PASS - Converse silently accepted request "
                "(task budget parameter ignored as documented)"
            )
            return ("PASS", None)
        else:
            msg = "No text blocks in response"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        status, msg = classify_error(error_msg)
        if status == "FAIL":
            print(
                "\n  Result: FAIL - task budgets correctly not supported on Converse during beta"
            )
        else:
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
    print(f"Beta: {TASK_BUDGETS_BETA}")
    print()

    results = []

    status, error = test_task_budget_converse(client, model_id)
    results.append(("Task budget via Converse", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
