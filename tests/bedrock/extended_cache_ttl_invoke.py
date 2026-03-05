#!/usr/bin/env python3
# ABOUTME: Tests extended 1-hour cache TTL on Amazon Bedrock via the invoke_model API.
# ABOUTME: Verifies that the ttl parameter in cache_control is accepted and caching works.

"""
Extended Cache TTL Test for Amazon Bedrock (Invoke API)
========================================================

Tests the extended cache TTL feature via invoke_model. Sends two requests
with a cached system prompt using ttl="1h":

1. First request — should create a cache entry
2. Second request (different user message) — should read from cache

Validates that the ttl parameter is accepted and that cache creation/read
tokens appear in the usage response.

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/extended_cache_ttl_invoke.py
"""

import json
import os
import sys
import time

try:
    import boto3
except ImportError:
    print("Error: boto3 package not installed. Run: uv add boto3")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client


CONVERSATION_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "50000_token_conversation.json"
)


def load_system_content():
    """Load system content from the 50K-token conversation file.

    Combines the system prompt, tools definitions, and a portion of the
    conversation history from the file to produce content well above the
    2048-token caching threshold.
    """
    with open(CONVERSATION_FILE, "r") as f:
        data = json.load(f)

    system_text = data["system"]
    tools_text = json.dumps(data["tools"], indent=2)
    # Include first 8 messages as reference context to exceed the caching threshold
    messages_text = json.dumps(data["messages"][:8], indent=2)

    return (
        f"{system_text}\n\n"
        f"# Available Tools Reference\n\n"
        f"The following tool definitions are available in this session:\n\n"
        f"{tools_text}\n\n"
        f"# Conversation Context\n\n"
        f"Here is relevant prior conversation context for reference:\n\n"
        f"{messages_text}"
    )


def test_extended_cache_ttl(client, model_id):
    """Test extended cache TTL with two requests. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: EXTENDED CACHE TTL (1h)")
    print("=" * 70)

    system_content = load_system_content()

    # Request 1: should create cache
    print("\n--- REQUEST 1: Cache creation ---")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "system": [
            {
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What is the Single Responsibility Principle? Answer in one sentence.",
            }
        ],
    }

    print(f"  System content length: {len(system_content)} chars")
    print(f"  cache_control: {{type: ephemeral, ttl: 1h}}")

    try:
        response1 = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result1 = json.loads(response1["body"].read())
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    usage1 = result1.get("usage", {})
    content1 = result1.get("content", [{}])[0].get("text", "N/A")

    print(f"\n  Response: {content1[:200]}")
    print(f"  input_tokens:                {usage1.get('input_tokens', 'N/A')}")
    print(
        f"  cache_creation_input_tokens: {usage1.get('cache_creation_input_tokens', 'N/A')}"
    )
    print(
        f"  cache_read_input_tokens:     {usage1.get('cache_read_input_tokens', 'N/A')}"
    )
    if "cache_creation" in usage1:
        print(f"  cache_creation details:      {usage1['cache_creation']}")

    # Request 2: should read from cache
    print("\n  Waiting 2 seconds...")
    time.sleep(2)

    print("\n--- REQUEST 2: Cache read ---")

    body["messages"][0]["content"] = (
        "What is the Open/Closed Principle? Answer in one sentence."
    )

    try:
        response2 = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result2 = json.loads(response2["body"].read())
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    usage2 = result2.get("usage", {})
    content2 = result2.get("content", [{}])[0].get("text", "N/A")

    print(f"\n  Response: {content2[:200]}")
    print(f"  input_tokens:                {usage2.get('input_tokens', 'N/A')}")
    print(
        f"  cache_creation_input_tokens: {usage2.get('cache_creation_input_tokens', 'N/A')}"
    )
    print(
        f"  cache_read_input_tokens:     {usage2.get('cache_read_input_tokens', 'N/A')}"
    )
    if "cache_creation" in usage2:
        print(f"  cache_creation details:      {usage2['cache_creation']}")

    # Validate
    cache_created = usage1.get("cache_creation_input_tokens", 0) or 0
    cache_read_1 = usage1.get("cache_read_input_tokens", 0) or 0
    cache_read_2 = usage2.get("cache_read_input_tokens", 0) or 0

    if cache_created == 0 and cache_read_1 == 0:
        msg = f"Request 1 had no cache activity (creation={cache_created}, read={cache_read_1})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if cache_read_2 == 0:
        msg = f"Request 2 did not read from cache (cache_read_input_tokens={cache_read_2})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - cache created ({cache_created} tokens), cache read ({cache_read_2} tokens)"
    )
    return ("PASS", None)


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
    print()

    results = []

    status, error = test_extended_cache_ttl(client, model_id)
    results.append(("Extended cache TTL (1h)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
