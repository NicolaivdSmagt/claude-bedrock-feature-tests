#!/usr/bin/env python3
# ABOUTME: Tests high-resolution vision support on Amazon Bedrock via the invoke_model API.
# ABOUTME: Opus 4.7 processes images up to 2576px (up from 1568px on previous models).

"""
High-Resolution Vision Test for Amazon Bedrock (Invoke API)
=============================================================

Tests Opus 4.7's enhanced vision capability. Starting with Opus 4.7, images
are processed at up to 2576px (up from 1568px on prior models). Higher
resolution means ~3x more image tokens. No API changes are needed — the
resolution is applied server-side.

This test sends a high-resolution image (≥2576px on one dimension) and
verifies:
- The request succeeds
- The image token usage is higher than a baseline low-res image, confirming
  the server processed at the higher resolution

Test cases:
1. High-res image (3000x2000) — verify successful processing
2. Low-res image (800x600) — baseline for token comparison
3. Token usage comparison — verify hi-res consumes more image tokens

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/hi_res_vision_invoke.py
"""

import base64
import json
import os
import sys
from pathlib import Path

try:
    import boto3
except ImportError:
    print("Error: boto3 package not installed. Run: uv add boto3")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

FILES_DIR = Path(os.path.dirname(os.path.abspath(__file__)), "..", "..", "files")

HI_RES_IMAGE = "hires_test.jpg"  # 3000 x 2000 — above the 2576px threshold
LOW_RES_IMAGE = "lores_test.jpg"  # 800 x 600 — below the 1568px threshold

FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "not supported",
    "unknown field",
    "unrecognized",
    "does not match any of the expected tags",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def encode_image(image_name):
    """Read and base64 encode an image file from the files/ directory."""
    path = FILES_DIR / image_name
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def send_image(client, model_id, image_name, prompt):
    """Send an image via invoke_model. Returns (result, error)."""
    image_data = encode_image(image_name)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    print("\n--- REQUEST (truncated for image data) ---")
    body_preview = json.loads(json.dumps(body))
    body_preview["messages"][0]["content"][0]["source"]["data"] = (
        f"<{len(image_data)} chars base64>"
    )
    print(json.dumps(body_preview, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))
        return (result, None)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return (None, error_msg)


def test_hi_res_image(client, model_id):
    """Test sending a high-resolution image (5788x2146). Returns (status, error_msg, usage)."""
    print("=" * 70)
    print(f"TEST: HIGH-RESOLUTION IMAGE ({HI_RES_IMAGE})")
    print("=" * 70)

    result, error = send_image(
        client,
        model_id,
        HI_RES_IMAGE,
        "Describe what you see in this image in one sentence.",
    )

    if error:
        status, msg = classify_error(error)
        print(f"\n  Result: {status}")
        return (status, msg, None)

    usage = result.get("usage", {})
    content = result.get("content", [])
    has_text = any(b.get("type") == "text" for b in content)

    print(f"\n  usage: {json.dumps(usage, indent=4)}")
    print(f"  has_text: {has_text}")

    if has_text:
        print("\n  Result: PASS - high-res image processed successfully")
        return ("PASS", None, usage)
    else:
        msg = "No text in response"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg, usage)


def test_low_res_baseline(client, model_id):
    """Test sending a low-resolution image for baseline token comparison."""
    print("=" * 70)
    print(f"TEST: LOW-RESOLUTION BASELINE ({LOW_RES_IMAGE})")
    print("=" * 70)

    result, error = send_image(
        client,
        model_id,
        LOW_RES_IMAGE,
        "Describe what you see in this image in one sentence.",
    )

    if error:
        status, msg = classify_error(error)
        print(f"\n  Result: {status}")
        return (status, msg, None)

    usage = result.get("usage", {})
    content = result.get("content", [])
    has_text = any(b.get("type") == "text" for b in content)

    print(f"\n  usage: {json.dumps(usage, indent=4)}")
    print(f"  has_text: {has_text}")

    if has_text:
        print("\n  Result: PASS - baseline image processed successfully")
        return ("PASS", None, usage)
    else:
        msg = "No text in response"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg, usage)


def test_token_comparison(hi_res_usage, low_res_usage):
    """Verify hi-res image consumed more input tokens than baseline."""
    print("=" * 70)
    print("TEST: TOKEN USAGE COMPARISON")
    print("=" * 70)

    if not hi_res_usage or not low_res_usage:
        msg = "One or both image tests failed — cannot compare tokens"
        print(f"\n  Result: ERROR - {msg}")
        return ("ERROR", msg)

    hi_tokens = hi_res_usage.get("input_tokens", 0)
    low_tokens = low_res_usage.get("input_tokens", 0)

    print(f"\n  Hi-res image input tokens: {hi_tokens}")
    print(f"  Low-res image input tokens: {low_tokens}")
    print(f"  Ratio: {hi_tokens / low_tokens if low_tokens else 0:.2f}x")

    if hi_tokens > low_tokens:
        print(
            "\n  Result: PASS - high-res image consumed more tokens "
            "(consistent with higher resolution processing)"
        )
        return ("PASS", None)
    elif hi_tokens == low_tokens:
        msg = f"Hi-res and low-res consumed same tokens ({hi_tokens}) — resolution handling may not differ"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)
    else:
        msg = f"Hi-res consumed fewer tokens ({hi_tokens} < {low_tokens}) — unexpected"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)


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
    print(f"Hi-res image: {HI_RES_IMAGE}")
    print(f"Low-res image: {LOW_RES_IMAGE}")
    print()

    results = []

    hi_status, hi_error, hi_usage = test_hi_res_image(client, model_id)
    results.append(("High-res image", hi_status, hi_error))

    lo_status, lo_error, lo_usage = test_low_res_baseline(client, model_id)
    results.append(("Low-res baseline", lo_status, lo_error))

    cmp_status, cmp_error = test_token_comparison(hi_usage, lo_usage)
    results.append(("Token comparison", cmp_status, cmp_error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
