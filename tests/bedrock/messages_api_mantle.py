#!/usr/bin/env python3
# ABOUTME: Tests the Messages API via Bedrock Mantle using the AnthropicBedrockMantle client.
# ABOUTME: This is a new API path introduced with Opus 4.7, distinct from InvokeModel/Converse.

"""
Messages API via Bedrock Mantle — Test
========================================

Tests the new Anthropic Messages API path on Amazon Bedrock Mantle. Starting
with Opus 4.7, customers can reach Claude through three API paths:
  1. Messages API via Bedrock Mantle (new) — this test
  2. InvokeModel via bedrock-runtime (existing)
  3. Converse via bedrock-runtime (existing)

The Messages API uses:
  - Endpoint: https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages
  - Python client: AnthropicBedrockMantle from the anthropic SDK
  - Authentication: SigV4 against bedrock-mantle:CreateInference IAM action
  - Model IDs: bare format like "anthropic.claude-opus-4-7" (no us./eu./global. prefix)

This test is NOT a pair — it exercises a single new API path and does not
have an invoke/converse variant.

Test cases:
1. Basic message — verify successful round-trip with Mantle client

Requirements:
    uv add anthropic
    AnthropicBedrockMantle requires a recent anthropic SDK version that
    includes Mantle support.

Usage:
    uv run python tests/bedrock/messages_api_mantle.py
"""

import os
import sys

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config

try:
    from anthropic import AnthropicBedrockMantle

    MANTLE_CLIENT_AVAILABLE = True
    MANTLE_IMPORT_ERROR = None
except ImportError as e:
    MANTLE_CLIENT_AVAILABLE = False
    MANTLE_IMPORT_ERROR = str(e)

FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "not supported",
    "unknown field",
    "unrecognized",
    "does not match any of the expected tags",
    "does not exist",
    "don't have access",
    "not found",
    "no such host",
    "nameresolutionerror",
    "name or service not known",
    "accessdenied",
    "unauthorized",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def test_mantle_client_import():
    """Verify the AnthropicBedrockMantle client is importable from the anthropic SDK."""
    print("=" * 70)
    print("TEST: AnthropicBedrockMantle CLIENT IMPORT")
    print("=" * 70)

    if MANTLE_CLIENT_AVAILABLE:
        print("\n  AnthropicBedrockMantle imported successfully")
        print("\n  Result: PASS")
        return ("PASS", None)

    print(f"\n  Import error: {MANTLE_IMPORT_ERROR}")
    print(
        "\n  Result: FAIL - AnthropicBedrockMantle not available in installed "
        "anthropic SDK version (feature not released yet)"
    )
    return ("FAIL", MANTLE_IMPORT_ERROR)


def test_basic_message(config):
    """Send a basic message via the Mantle Messages API. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: BASIC MESSAGE VIA MANTLE MESSAGES API")
    print("=" * 70)

    if not MANTLE_CLIENT_AVAILABLE:
        print("\n  Skipped — AnthropicBedrockMantle not available in installed SDK")
        print("\n  Result: FAIL - SDK support not yet available")
        return ("FAIL", "AnthropicBedrockMantle class not importable")

    # Opus 4.7 launches in US East (N. Virginia) and US West (Oregon) only.
    # The config `region` may be an EU region for other tests — use a dedicated
    # Mantle region that's guaranteed to have Opus 4.7. Allow override via config.
    region = config.get("mantle_region", "us-east-1")
    mantle_model = config.get("bedrock_mantle_model_id", "us.anthropic.claude-opus-4-7")
    aws_profile = config.get("aws_profile")

    print(f"\n  Region: {region}")
    print(f"  Model: {mantle_model}")
    print(f"  AWS profile: {aws_profile}")

    try:
        # Configure the Mantle client. It uses the default boto3 credential
        # chain, which picks up AWS_PROFILE from the environment.
        client = AnthropicBedrockMantle(aws_region=region)

        print("\n  Sending message.create request...")
        message = client.messages.create(
            model=mantle_model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": "Say hello in exactly one short sentence."}
            ],
        )

        print("\n--- MESSAGE RESPONSE ---")
        print(f"  id: {message.id}")
        print(f"  model: {message.model}")
        print(f"  stop_reason: {message.stop_reason}")
        print(f"  usage.input_tokens: {message.usage.input_tokens}")
        print(f"  usage.output_tokens: {message.usage.output_tokens}")
        print(f"  content blocks: {len(message.content)}")

        for i, block in enumerate(message.content):
            block_type = getattr(block, "type", "?")
            print(f"  content[{i}].type: {block_type}")
            if block_type == "text":
                text = getattr(block, "text", "")
                print(f"  content[{i}].text: {text!r}")

        has_text = any(getattr(b, "type", None) == "text" for b in message.content)
        if has_text:
            print("\n  Result: PASS - Mantle Messages API round-trip successful")
            return ("PASS", None)

        msg = "No text block in response"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- ERROR ---")
        print(f"  {error_msg}")
        status, msg = classify_error(error_msg)
        if status == "FAIL":
            print(
                "\n  Result: FAIL - Mantle endpoint not reachable or model "
                "not available in this region/account"
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

    print("Messages API via Bedrock Mantle")
    print(f"Mantle region: {config.get('mantle_region', 'us-east-1')}")
    print(
        f"Model: {config.get('bedrock_mantle_model_id', 'us.anthropic.claude-opus-4-7')}"
    )
    print()

    results = []

    status, error = test_mantle_client_import()
    results.append(("AnthropicBedrockMantle import", status, error))

    status, error = test_basic_message(config)
    results.append(("Basic message (Mantle)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
