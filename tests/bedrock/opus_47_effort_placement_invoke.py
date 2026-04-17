#!/usr/bin/env python3
# ABOUTME: Tests Opus 4.7 effort parameter placement via the invoke_model API.
# ABOUTME: effort must go in output_config, NOT inside the thinking dict.

"""
Opus 4.7 Effort Placement Test (Invoke API)
=============================================

On Opus 4.7, the `effort` parameter must be placed in `output_config`, NOT
inside the `thinking` dict. Putting `effort` inside `thinking` returns HTTP 400.

Test cases:
1. Correct placement: thinking={type: adaptive}, output_config={effort: high}
   — should succeed
2. Wrong placement: thinking={type: adaptive, effort: high}, no output_config
   — should return HTTP 400

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/opus_47_effort_placement_invoke.py
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

OPUS_47_MODEL_ID = "global.anthropic.claude-opus-4-7"

FEATURE_NOT_AVAILABLE_MARKERS = [
    "does not match any of the expected tags",
    "don't have access",
    "model is not available",
    "access denied",
    "unknownserviceerror",
]

WRONG_PLACEMENT_MARKERS = [
    "extra inputs are not permitted",
    "thinking.adaptive.effort",
    "thinking.effort",
    "unexpected",
    "unknown field",
    "unknown property",
    "not permitted",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def is_wrong_placement_rejection(error_msg):
    """Check if the error indicates rejection of effort-inside-thinking."""
    lower = error_msg.lower()
    return any(marker in lower for marker in WRONG_PLACEMENT_MARKERS)


def test_correct_placement(client, model_id):
    """Test correct effort placement in output_config. Should succeed."""
    print("=" * 70)
    print("TEST: CORRECT PLACEMENT — output_config.effort")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        content = result.get("content", [])
        has_text = any(b.get("type") == "text" for b in content)

        if has_text:
            print("\n  Result: PASS - correct placement accepted")
            return ("PASS", None)
        else:
            msg = "No text in response"
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


def test_wrong_placement(client, model_id):
    """Test effort inside thinking dict. Should return HTTP 400."""
    print("=" * 70)
    print("TEST: WRONG PLACEMENT — thinking.effort")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "thinking": {"type": "adaptive", "effort": "high"},
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        msg = "Expected HTTP 400 for effort inside thinking dict, but request succeeded"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")

        # Check placement rejection first — it must take priority over generic
        # feature-not-available markers.
        if is_wrong_placement_rejection(error_msg):
            print("\n  Result: PASS - effort inside thinking correctly rejected")
            return ("PASS", None)

        feature_status, _ = classify_error(error_msg)
        if feature_status == "FAIL":
            print(
                "\n  Result: FAIL - Opus 4.7 not accessible in this region "
                "(can't verify effort placement rule)"
            )
            return ("FAIL", error_msg)

        print(f"\n  Result: ERROR - unexpected error type")
        return ("ERROR", error_msg)

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
    client = get_bedrock_client(config)
    model_id = OPUS_47_MODEL_ID

    print(f"Model (hardcoded): {model_id}")
    print(f"Config region: {config['region']}")
    print()

    results = []

    status, error = test_correct_placement(client, model_id)
    results.append(("Correct placement (output_config.effort)", status, error))

    status, error = test_wrong_placement(client, model_id)
    results.append(("Wrong placement (thinking.effort)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
