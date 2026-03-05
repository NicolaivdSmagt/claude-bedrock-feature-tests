#!/usr/bin/env python3
# ABOUTME: Tests the maximum image count limit on Bedrock via the invoke_model API.
# ABOUTME: Verifies 20, 21, and 100 images succeed, and 101 images is rejected.

import base64
import io
import json
import os
import sys

try:
    import boto3
    from PIL import Image
except ImportError:
    print("Error: required packages not installed. Run: uv add boto3 pillow")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# Generate a small test image in memory
_buffer = io.BytesIO()
Image.new("RGB", (50, 50), "blue").save(_buffer, format="JPEG")
IMAGE_B64 = base64.b64encode(_buffer.getvalue()).decode("utf-8")

# Counts at or below this limit should be accepted; above should be rejected
ACCEPTED_COUNTS = [20, 21, 100]
REJECTED_COUNTS = [101]


def is_image_limit_error(error_str):
    """Check if an error message indicates an image/document count rejection."""
    lower = error_str.lower()
    return "too many images" in lower or "too many documents" in lower


def send_images(client, model_id, count):
    """Send `count` images via invoke_model. Returns (succeeded: bool, detail: str)."""
    content = [{"type": "text", "text": f"Count {count} images"}] + [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": IMAGE_B64,
            },
        }
        for _ in range(count)
    ]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["context-1m-2025-08-07"],
        "max_tokens": 20,
        "messages": [{"role": "user", "content": content}],
    }

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())

        stop_reason = result.get("stop_reason")
        usage = result.get("usage", {})
        n_blocks = len(result.get("content", []))
        detail = (
            f"stop_reason={stop_reason}, "
            f"usage={json.dumps(usage, default=str)}, "
            f"{n_blocks} content block(s)"
        )
        return (True, detail)

    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        return (False, detail)


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    # Run all counts and collect outcomes
    outcomes = {}
    for count in ACCEPTED_COUNTS + REJECTED_COUNTS:
        print("=" * 70)
        print(f"  Sending {count} images...")
        succeeded, detail = send_images(client, model_id, count)
        outcomes[count] = (succeeded, detail)
        status_label = "accepted" if succeeded else "rejected"
        print(f"  {count} images: {status_label}")
        print(f"  {detail[:300]}")
        print()

    # Evaluate: single overall verdict
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    name_width = max(len(str(c)) for c in outcomes) + len(" images")
    for count, (succeeded, detail) in outcomes.items():
        label = f"{count} images"
        status_label = "accepted" if succeeded else "rejected"
        print(f"  {label:<{name_width}}  {status_label:<10}  {detail[:200]}")

    # Check for unrelated errors: any rejection that isn't an image limit error
    unrelated_errors = []
    for count in ACCEPTED_COUNTS + REJECTED_COUNTS:
        succeeded, detail = outcomes[count]
        if not succeeded and not is_image_limit_error(detail):
            unrelated_errors.append((count, detail))

    if unrelated_errors:
        print()
        for count, detail in unrelated_errors:
            print(
                f"  ERROR: {count} images failed with unrelated error: {detail[:200]}"
            )
        print()
        print(f"  Result: ERROR")
        print("=" * 70)
        sys.exit(1)

    # Check accepted counts all succeeded
    rejected_below_limit = [c for c in ACCEPTED_COUNTS if not outcomes[c][0]]
    # Check rejected counts all failed
    accepted_above_limit = [c for c in REJECTED_COUNTS if outcomes[c][0]]

    failures = []
    if rejected_below_limit:
        failures.append(
            f"{rejected_below_limit} rejected but should have been accepted"
        )
    if accepted_above_limit:
        failures.append(
            f"{accepted_above_limit} accepted but should have been rejected"
        )

    print()
    if failures:
        for f in failures:
            print(f"  {f}")
        print()
        print(f"  Result: FAIL")
        print("=" * 70)
        sys.exit(1)

    print(f"  20, 21, 100 accepted and 101 rejected")
    print()
    print(f"  Result: PASS")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
