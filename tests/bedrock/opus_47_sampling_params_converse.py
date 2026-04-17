#!/usr/bin/env python3
# ABOUTME: Tests Opus 4.7 deprecation of temperature/top_p/top_k via the Converse API.
# ABOUTME: Setting non-default sampling parameters should return HTTP 400 on Opus 4.7.

"""
Opus 4.7 Sampling Parameters Deprecation Test (Converse API)
==============================================================

On Opus 4.7, sampling parameters (temperature, top_p, top_k) are deprecated
and will return HTTP 400 if set to non-default values. In the Converse API,
`temperature` and `topP` appear in `inferenceConfig`, while `top_k` (snake
case) is passed via `additionalModelRequestFields`.

This test specifically targets the Opus 4.7 model (hardcoded model ID).

Test cases:
1. No sampling parameters — should succeed
2. inferenceConfig.temperature=0.7 — should return HTTP 400
3. inferenceConfig.topP=0.9 — should return HTTP 400
4. additionalModelRequestFields.top_k=50 — should return HTTP 400

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/opus_47_sampling_params_converse.py
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
    """Check if the error indicates the deprecated parameter was rejected.

    Converse parameter names may differ from the API-level name (e.g. `topP`
    vs `top_p`), so we normalize underscores and case before matching.
    """
    norm = lambda s: s.lower().replace("_", "")
    has_param_mention = norm(param_name) in norm(error_msg)
    has_deprecation_indicator = any(
        marker in error_msg.lower() for marker in DEPRECATED_PARAM_MARKERS
    )
    return has_param_mention and has_deprecation_indicator


def test_no_sampling_params(client, model_id):
    """Test a baseline request with no sampling parameters. Should succeed."""
    print("=" * 70)
    print("TEST: BASELINE — NO SAMPLING PARAMETERS")
    print("=" * 70)

    request = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 256},
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        has_text = any("text" in b for b in content_blocks)

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


def test_inference_config_param(client, model_id, param_name, param_value):
    """Test a deprecated param in inferenceConfig. Returns (status, error_msg)."""
    print("=" * 70)
    print(f"TEST: DEPRECATED PARAM — inferenceConfig.{param_name}={param_value}")
    print("=" * 70)

    inference_config = {"maxTokens": 256, param_name: param_value}

    request = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": inference_config,
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        msg = f"Expected HTTP 400 for {param_name}, but request succeeded"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")

        # Check deprecation first — it must take priority over the generic
        # feature-not-available markers.
        if is_deprecated_param_rejection(error_msg, param_name):
            print(f"\n  Result: PASS - {param_name} correctly rejected as deprecated")
            return ("PASS", None)

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


def test_additional_fields_param(client, model_id, param_name, param_value):
    """Test a deprecated param in additionalModelRequestFields. Returns (status, error_msg)."""
    print("=" * 70)
    print(
        f"TEST: DEPRECATED PARAM — additionalModelRequestFields.{param_name}={param_value}"
    )
    print("=" * 70)

    request = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 256},
        "additionalModelRequestFields": {param_name: param_value},
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        msg = f"Expected HTTP 400 for {param_name}, but request succeeded"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")

        # Check deprecation first — it must take priority over the generic
        # feature-not-available markers.
        if is_deprecated_param_rejection(error_msg, param_name):
            print(f"\n  Result: PASS - {param_name} correctly rejected as deprecated")
            return ("PASS", None)

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

    status, error = test_inference_config_param(client, model_id, "temperature", 0.7)
    results.append(("temperature=0.7", status, error))

    status, error = test_inference_config_param(client, model_id, "topP", 0.9)
    results.append(("topP=0.9", status, error))

    status, error = test_additional_fields_param(client, model_id, "top_k", 50)
    results.append(("top_k=50", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
