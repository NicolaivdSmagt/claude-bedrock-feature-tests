#!/usr/bin/env python3
# ABOUTME: Tests PDF size and count limits on Bedrock via the Converse API.
# ABOUTME: Tests single large PDFs and multiple PDF combinations with 1M context beta.

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

_FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "files"
)
PDF_31MB = os.path.join(_FILES_DIR, "test_pdf_31MB.pdf")
PDF_1MB = os.path.join(_FILES_DIR, "example.pdf")

PROMPT = "How many PDF documents did I send? Answer with just the number."

# Error markers that indicate the API rejected the request due to size/payload limits
SIZE_LIMIT_ERROR_MARKERS = [
    "too large",
    "too long",
    "size limit",
    "maximum size",
    "payload",
    "exceeds",
    "request size",
    "content length",
    "too many",
    "input is too long",
]


def is_size_limit_error(error_str):
    """Check if an error message indicates a size or payload limit rejection."""
    lower = error_str.lower()
    return any(marker in lower for marker in SIZE_LIMIT_ERROR_MARKERS)


def load_pdf(path):
    """Load a PDF file from disk."""
    with open(path, "rb") as f:
        return f.read()


def send_pdfs(client, model_id, pdf_list):
    """Send PDFs via Converse with 1M context beta. Returns (succeeded, detail)."""
    content = []
    for i, pdf_data in enumerate(pdf_list):
        content.append(
            {
                "document": {
                    "name": f"document_{i}",
                    "format": "pdf",
                    "source": {"bytes": pdf_data},
                }
            }
        )
    content.append({"text": PROMPT})

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 50},
            additionalModelRequestFields={"anthropic_beta": ["context-1m-2025-08-07"]},
        )

        stop_reason = response.get("stopReason")
        usage = response.get("usage", {})
        output_message = response.get("output", {}).get("message", {})
        text_blocks = [
            b["text"] for b in output_message.get("content", []) if "text" in b
        ]
        text = text_blocks[0] if text_blocks else ""
        detail = (
            f"stopReason={stop_reason}, "
            f"inputTokens={usage.get('inputTokens', 0)}, "
            f"response={text[:80]}"
        )
        return (True, detail)

    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    pdf_1mb = load_pdf(PDF_1MB)
    pdf_31mb = load_pdf(PDF_31MB)

    pdf_1mb_size = len(pdf_1mb) / (1024 * 1024)
    print(f"1MB PDF actual size: {pdf_1mb_size:.2f} MB")

    tests = [
        ("~31MB single PDF", [pdf_31mb]),
        ("~10MB (7x 1.4MB text PDFs)", [pdf_1mb] * 7),
        ("~20MB (14x 1.4MB text PDFs)", [pdf_1mb] * 14),
        ("~30MB (21x 1.4MB text PDFs)", [pdf_1mb] * 21),
    ]

    # Run all tests, all expected to succeed
    outcomes = []
    for desc, pdf_list in tests:
        total_mb = sum(len(p) for p in pdf_list) / (1024 * 1024)

        print("=" * 70)
        print(f"  {desc}")
        print(f"  raw: {total_mb:.1f} MB, {len(pdf_list)} PDF(s)")
        print("=" * 70)

        succeeded, detail = send_pdfs(client, model_id, pdf_list)
        outcomes.append((desc, succeeded, detail))

        status_label = "accepted" if succeeded else "rejected"
        print(f"  {status_label}")
        print(f"  {detail[:300]}")
        print()

    # Single overall verdict
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    name_width = max(len(desc) for desc, _, _ in outcomes)
    for desc, succeeded, detail in outcomes:
        status_label = "accepted" if succeeded else "rejected"
        print(f"  {desc:<{name_width}}  {status_label}")

    # Check for unrelated errors
    unrelated_errors = [
        (desc, detail)
        for desc, succeeded, detail in outcomes
        if not succeeded and not is_size_limit_error(detail)
    ]
    if unrelated_errors:
        print()
        for desc, detail in unrelated_errors:
            print(f"  ERROR: {desc} - {detail[:200]}")
        print()
        print("  Result: ERROR")
        print("=" * 70)
        sys.exit(1)

    # Check for size-limit rejections (all should have been accepted)
    size_rejections = [
        (desc, detail)
        for desc, succeeded, detail in outcomes
        if not succeeded and is_size_limit_error(detail)
    ]
    if size_rejections:
        print()
        for desc, detail in size_rejections:
            print(f"  REJECTED: {desc} - {detail[:200]}")
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
