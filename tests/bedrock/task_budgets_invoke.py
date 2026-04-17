#!/usr/bin/env python3
# ABOUTME: Tests task budgets (beta) on Amazon Bedrock via the invoke_model API.
# ABOUTME: Validates output_config.task_budget with task-budgets-2026-03-13 beta header.

"""
Task Budgets Test for Amazon Bedrock (Invoke API)
===================================================

Tests the task budgets beta feature in Opus 4.7, which lets developers set a
total token spending limit for an agentic turn. Claude sees a live countdown
and uses it to prioritize work, plan ahead, and wrap up gracefully.

Key facts:
- Beta header required: anthropic-beta: task-budgets-2026-03-13
- Parameter: output_config.task_budget = {"type": "tokens", "total": N}
- Minimum: 20,000 tokens (requests below this are rejected)
- Advisory, not enforced
- Only available on Opus 4.7 and future models
- Available on Messages API and InvokeModel (not Converse during beta)

Test cases:
1. Valid task budget (30K tokens) — should accept and respond
2. Below minimum (10K tokens) — should be rejected

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/task_budgets_invoke.py
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

SIZE_LIMIT_ERROR_MARKERS = [
    "too small",
    "minimum",
    "below",
    "at least",
    "20000",
    "20,000",
]


def classify_error(error_msg, below_minimum=False):
    """Classify an error as FAIL (feature not available / expected rejection) or ERROR (other)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    if below_minimum:
        for marker in SIZE_LIMIT_ERROR_MARKERS:
            if marker in lower:
                return ("PASS", None)
    return ("ERROR", error_msg)


def test_task_budget_valid(client, model_id):
    """Test a valid task budget (30K tokens). Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: VALID TASK BUDGET (30000 tokens)")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [TASK_BUDGETS_BETA],
        "max_tokens": 4096,
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": "medium",
            "task_budget": {"type": "tokens", "total": 30000},
        },
        "messages": [
            {
                "role": "user",
                "content": (
                    "Design a small REST API for a todo list application. "
                    "Include three endpoints with their HTTP methods and paths. "
                    "Keep the response brief."
                ),
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

        stop_reason = result.get("stop_reason")
        usage = result.get("usage", {})
        content = result.get("content", [])
        has_text = any(b.get("type") == "text" for b in content)

        print(f"\n  stop_reason: {stop_reason}")
        print(f"  usage: {json.dumps(usage, indent=4)}")
        print(f"  has_text: {has_text}")

        if has_text:
            print("\n  Result: PASS - task budget accepted, got text response")
            return ("PASS", None)
        else:
            msg = f"No text in response, stop_reason: {stop_reason}"
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


def test_task_budget_below_minimum(client, model_id):
    """Test a task budget below the 20K minimum. Should be rejected."""
    print("=" * 70)
    print("TEST: TASK BUDGET BELOW MINIMUM (10000 tokens)")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [TASK_BUDGETS_BETA],
        "max_tokens": 2048,
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": "medium",
            "task_budget": {"type": "tokens", "total": 10000},
        },
        "messages": [{"role": "user", "content": "Hello"}],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        # Should have been rejected; if we got here, that's a FAIL for the expected behavior
        msg = "Expected rejection for budget below 20K, but request succeeded"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        status, msg = classify_error(error_msg, below_minimum=True)
        if status == "PASS":
            print(
                "\n  Result: PASS - request correctly rejected for below-minimum budget"
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

    status, error = test_task_budget_valid(client, model_id)
    results.append(("Valid task budget (30K)", status, error))

    status, error = test_task_budget_below_minimum(client, model_id)
    results.append(("Below minimum (10K)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
