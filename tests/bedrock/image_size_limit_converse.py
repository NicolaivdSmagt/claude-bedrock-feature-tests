#!/usr/bin/env python3
# ABOUTME: Tests image size and total payload limits on Bedrock via the Converse API.
# ABOUTME: Tests single 3MB and 5MB images, and a ~22MB multi-image payload.

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

# Error markers that indicate the API rejected the request due to size limits
SIZE_LIMIT_ERROR_MARKERS = [
    "too large",
    "too long",
    "size limit",
    "maximum size",
    "payload",
    "exceeds",
    "request size",
    "content length",
    "image size",
    "too many",
    "input is too long",
]


def is_size_limit_error(error_str):
    """Check if an error message indicates a file or request size rejection."""
    lower = error_str.lower()
    return any(marker in lower for marker in SIZE_LIMIT_ERROR_MARKERS)


def get_file_size_mb(file_path):
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def send_single_image(client, model_id, image_path):
    """Send a single image via Converse. Returns (succeeded, detail)."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    content = [
        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
        {"text": "Describe this image in one sentence."},
    ]

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 50},
        )

        stop_reason = response.get("stopReason")
        usage = response.get("usage", {})
        detail = (
            f"stopReason={stop_reason}, "
            f"inputTokens={usage.get('inputTokens', 0)}, "
            f"outputTokens={usage.get('outputTokens', 0)}"
        )
        return (True, detail)

    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


def send_multi_image(client, model_id, image_path, count):
    """Send multiple copies of an image via Converse with 1M context beta.
    Returns (succeeded, detail)."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    content = [
        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}}
        for _ in range(count)
    ]
    content.append(
        {
            "text": f"I sent you {count} images. Confirm you can see them in one sentence.",
        }
    )

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 50},
            additionalModelRequestFields={"anthropic_beta": ["context-1m-2025-08-07"]},
        )

        stop_reason = response.get("stopReason")
        usage = response.get("usage", {})
        detail = (
            f"stopReason={stop_reason}, "
            f"inputTokens={usage.get('inputTokens', 0)}, "
            f"outputTokens={usage.get('outputTokens', 0)}"
        )
        return (True, detail)

    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    image_3mb = FILES_DIR / "3mb.jpg"
    image_5mb = FILES_DIR / "5mb.jpg"

    for img in [image_3mb, image_5mb]:
        if not img.exists():
            print(f"ERROR: {img} not found")
            sys.exit(1)

    size_3mb = get_file_size_mb(image_3mb)
    size_5mb = get_file_size_mb(image_5mb)

    multi_7_total_mb = size_3mb * 7
    multi_8_total_mb = size_3mb * 8

    # All tests expected to succeed
    tests = [
        (
            f"Single 3MB image ({size_3mb:.1f} MB)",
            lambda: send_single_image(client, model_id, image_3mb),
        ),
        (
            f"Single 5MB image ({size_5mb:.1f} MB)",
            lambda: send_single_image(client, model_id, image_5mb),
        ),
        (
            f"7x 3MB images (~{multi_7_total_mb:.0f} MB payload)",
            lambda: send_multi_image(client, model_id, image_3mb, 7),
        ),
        (
            f"8x 3MB images (~{multi_8_total_mb:.0f} MB payload)",
            lambda: send_multi_image(client, model_id, image_3mb, 8),
        ),
    ]

    outcomes = []
    for label, run_test in tests:
        print("=" * 70)
        print(f"  {label}")
        print("=" * 70)

        succeeded, detail = run_test()
        outcomes.append((label, succeeded, detail))

        status_label = "accepted" if succeeded else "rejected"
        print(f"  {status_label}")
        print(f"  {detail[:300]}")
        print()

    # Single overall verdict
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    name_width = max(len(label) for label, _, _ in outcomes)
    for label, succeeded, detail in outcomes:
        status_label = "accepted" if succeeded else "rejected"
        print(f"  {label:<{name_width}}  {status_label}")

    # Check for unrelated errors
    unrelated_errors = [
        (label, detail)
        for label, succeeded, detail in outcomes
        if not succeeded and not is_size_limit_error(detail)
    ]
    if unrelated_errors:
        print()
        for label, detail in unrelated_errors:
            print(f"  ERROR: {label} - {detail[:200]}")
        print()
        print("  Result: ERROR")
        print("=" * 70)
        sys.exit(1)

    # Check for size-limit rejections (all should have been accepted)
    size_rejections = [
        (label, detail)
        for label, succeeded, detail in outcomes
        if not succeeded and is_size_limit_error(detail)
    ]
    if size_rejections:
        print()
        for label, detail in size_rejections:
            print(f"  REJECTED: {label} - {detail[:200]}")
        print()
        print("  Result: FAIL")
        print("=" * 70)
        sys.exit(1)

    print()
    print("  All payloads accepted")
    print()
    print("  Result: PASS")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
