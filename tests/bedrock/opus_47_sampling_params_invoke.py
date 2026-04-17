#!/usr/bin/env python3
# ABOUTME: Tests Opus 4.7 deprecation of temperature/top_p/top_k via invoke_model API.
# ABOUTME: Setting non-default sampling parameters should return HTTP 400 on Opus 4.7.

"""
Opus 4.7 Sampling Parameters Deprecation Test (Invoke API)
============================================================

On Opus 4.7, sampling parameters (temperature, top_p, top_k) are deprecated
and will return HTTP 400 if set to non-default values. The safest path is to
omit these parameters entirely.

This test specifically targets the Opus 4.7 model (hardcoded model ID). If
Opus 4.7 is unavailable in the configured region, the test returns FAIL with
a classification indicating feature-not-available.

Test cases:
1. No sampling parameters — should succeed
2. temperature=0.7 — should return HTTP 400
3. top_p=0.9 — should return HTTP 400
4. top_k=50 — should return HTTP 400

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/opus_47_sampling_params_invoke.py
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

# Hardcoded Opus 4.7 model ID — test targets this model specifically
OPUS_47_MODEL_ID = "global.anthropic.claude-opus-4-7"

FEATURE_NOT_AVAILABLE_MARKERS = [
    "does not match any of the expected tags",
    "don't have access",
    "model is not available",
    "access denied",
    "unknownserviceerror",
]

# Error markers indicating the parameter was correctly rejected as deprecated
DEPRECATED_PARAM_MARKERS = [
    "is deprecated",
    "deprecated for this model",
    "not supported",
    "must not be set",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def is_deprecated_param_rejection(error_msg, param_name):
    """Check if the error indicates the deprecated parameter was rejected."""
    lower = error_msg.lower()
    # Require both the parameter name AND a deprecation indicator.
    has_param_mention = param_name.lower() in lower
    has_deprecation_indicator = any(
        marker in lower for marker in DEPRECATED_PARAM_MARKERS
    )
    return has_param_mention and has_deprecation_indicator


def test_no_sampling_params(client, model_id):
    """Test a baseline request with no sampling parameters. Should succeed."""
    print("=" * 70)
    print("TEST: BASELINE — NO SAMPLING PARAMETERS")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Say hello."}],
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
            print("\n  Result: PASS - baseline request succeeded")
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


def test_deprecated_param(client, model_id, param_name, param_value):
    """Test that a deprecated sampling param returns HTTP 400. Returns (status, error_msg)."""
    print("=" * 70)
    print(f"TEST: DEPRECATED PARAM — {param_name}={param_value}")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Say hello."}],
        param_name: param_value,
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        # Request succeeded — either the parameter was silently accepted (acceptable for older
        # models) or Opus 4.7 isn't enforcing the deprecation. For Opus 4.7, we expect rejection.
        msg = f"Expected HTTP 400 for {param_name}, but request succeeded"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")

        # Check first whether it was rejected for the expected reason (deprecation).
        # This takes priority so we don't false-positive on generic AWS error strings
        # that might also match the feature-not-available markers.
        if is_deprecated_param_rejection(error_msg, param_name):
            print(f"\n  Result: PASS - {param_name} correctly rejected as deprecated")
            return ("PASS", None)

        # Otherwise check if Opus 4.7 is accessible at all
        feature_status, _ = classify_error(error_msg)
        if feature_status == "FAIL":
            print(
                "\n  Result: FAIL - Opus 4.7 not accessible in this region "
                "(can't verify sampling parameter deprecation)"
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

    status, error = test_no_sampling_params(client, model_id)
    results.append(("Baseline (no sampling params)", status, error))

    status, error = test_deprecated_param(client, model_id, "temperature", 0.7)
    results.append(("temperature=0.7", status, error))

    status, error = test_deprecated_param(client, model_id, "top_p", 0.9)
    results.append(("top_p=0.9", status, error))

    status, error = test_deprecated_param(client, model_id, "top_k", 50)
    results.append(("top_k=50", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
