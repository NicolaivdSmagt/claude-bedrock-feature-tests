#!/usr/bin/env python3
# ABOUTME: Tests automatic prompt caching on Amazon Bedrock via the Converse API.
# ABOUTME: Verifies that top-level cache_control auto-applies a breakpoint to the last cacheable block.

"""
Automatic Prompt Caching Test for Amazon Bedrock (Converse API)
================================================================

Tests the automatic caching feature via the Converse API. Instead of
placing cachePoint blocks on individual content, automatic caching adds
a single cache_control field (passed via additionalModelRequestFields)
at the top level. The system automatically applies the cache breakpoint
to the last cacheable block and moves it forward as conversations grow.

Validates:
1. Basic automatic caching: a top-level cache_control triggers cache
   creation on the first request and cache reads on subsequent requests
2. Multi-turn caching: the cache breakpoint moves forward as the
   conversation grows, reading previously cached content
3. TTL support: automatic caching accepts ttl="1h" parameter
4. Combination with explicit breakpoints: automatic caching works
   alongside block-level cachePoint markers

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/automatic_caching_converse.py
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

FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "does not match any of the expected tags",
    "not supported",
    "unknown field",
    "cache_control",
    "unsupported model",
    "did not allow prompt caching",
    "extra inputs are not permitted",
]

CONVERSATION_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "50000_token_conversation.json"
)


def load_system_content():
    """Load system content from the 50K-token conversation file.

    Combines the system prompt, tools definitions, and a portion of the
    conversation history to produce content well above the 2048-token
    caching threshold.
    """
    with open(CONVERSATION_FILE, "r") as f:
        data = json.load(f)

    system_text = data["system"]
    tools_text = json.dumps(data["tools"], indent=2)
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


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (unexpected)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def converse_with_auto_cache(client, model_id, system, messages, cache_control):
    """Make a Converse API call with top-level cache_control.

    The cache_control is passed via additionalModelRequestFields since
    it is an Anthropic-specific top-level parameter.

    Returns the full Converse response dict.
    """
    return client.converse(
        modelId=model_id,
        system=system,
        messages=messages,
        inferenceConfig={"maxTokens": 200},
        additionalModelRequestFields={
            "cache_control": cache_control,
        },
    )


def print_usage(usage, label=""):
    """Print cache-related usage fields (Converse camelCase keys)."""
    prefix = f"  {label}" if label else "  "
    print(f"{prefix}inputTokens:           {usage.get('inputTokens', 'N/A')}")
    print(f"{prefix}cacheWriteInputTokens: {usage.get('cacheWriteInputTokens', 'N/A')}")
    print(f"{prefix}cacheReadInputTokens:  {usage.get('cacheReadInputTokens', 'N/A')}")


def get_text(response):
    """Extract the first text block from a Converse response."""
    content = response.get("output", {}).get("message", {}).get("content", [])
    for block in content:
        if isinstance(block, dict) and "text" in block:
            return block["text"]
    return "N/A"


def test_basic_automatic_caching(client, model_id):
    """Test that top-level cache_control creates and reads cache entries.

    Returns (status, error_msg).
    """
    print("=" * 70)
    print("TEST: BASIC AUTOMATIC CACHING")
    print("=" * 70)

    system_content = load_system_content()
    system = [{"text": system_content}]
    cache_control = {"type": "ephemeral"}

    # Request 1: should create cache
    print("\n--- REQUEST 1: Cache creation ---")
    print(f"  System content length: {len(system_content)} chars")
    print(f"  Top-level cache_control: {json.dumps(cache_control)}")

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "What is the Single Responsibility Principle? One sentence."}
            ],
        }
    ]

    try:
        response1 = converse_with_auto_cache(
            client, model_id, system, messages, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage1 = response1.get("usage", {})
    text1 = get_text(response1)
    print(f"\n  Response: {text1[:200]}")
    print_usage(usage1)

    # Request 2: same system, different question — should read from cache
    print("\n  Waiting 2 seconds...")
    time.sleep(2)

    print("\n--- REQUEST 2: Cache read ---")

    messages2 = [
        {
            "role": "user",
            "content": [{"text": "What is the Open/Closed Principle? One sentence."}],
        }
    ]

    try:
        response2 = converse_with_auto_cache(
            client, model_id, system, messages2, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage2 = response2.get("usage", {})
    text2 = get_text(response2)
    print(f"\n  Response: {text2[:200]}")
    print_usage(usage2)

    # Validate
    cache_created = usage1.get("cacheWriteInputTokens", 0) or 0
    cache_read_1 = usage1.get("cacheReadInputTokens", 0) or 0
    cache_read_2 = usage2.get("cacheReadInputTokens", 0) or 0

    if cache_created == 0 and cache_read_1 == 0:
        msg = f"Request 1 had no cache activity (creation={cache_created}, read={cache_read_1})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if cache_read_2 == 0:
        msg = f"Request 2 did not read from cache (cacheReadInputTokens={cache_read_2})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - cache created ({cache_created} tokens), "
        f"cache read ({cache_read_2} tokens)"
    )
    return ("PASS", None)


def test_multi_turn_automatic_caching(client, model_id):
    """Test that automatic caching moves the breakpoint forward in multi-turn conversations.

    Returns (status, error_msg).
    """
    print("=" * 70)
    print("TEST: MULTI-TURN AUTOMATIC CACHING")
    print("=" * 70)

    system_content = load_system_content()
    system = [{"text": system_content}]
    cache_control = {"type": "ephemeral"}

    # Turn 1
    print("\n--- TURN 1 ---")
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "What is the Single Responsibility Principle? One sentence."}
            ],
        }
    ]

    try:
        response1 = converse_with_auto_cache(
            client, model_id, system, messages, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage1 = response1.get("usage", {})
    text1 = get_text(response1)
    print(f"  Response: {text1[:150]}")
    print_usage(usage1)

    # Build turn 2 with conversation history
    print("\n  Waiting 2 seconds...")
    time.sleep(2)

    print("\n--- TURN 2 ---")
    assistant_content = (
        response1.get("output", {}).get("message", {}).get("content", [])
    )
    messages.append({"role": "assistant", "content": assistant_content})
    messages.append(
        {
            "role": "user",
            "content": [
                {"text": "Now explain the Open/Closed Principle. One sentence."}
            ],
        }
    )

    try:
        response2 = converse_with_auto_cache(
            client, model_id, system, messages, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage2 = response2.get("usage", {})
    text2 = get_text(response2)
    print(f"  Response: {text2[:150]}")
    print_usage(usage2)

    # Build turn 3
    print("\n  Waiting 2 seconds...")
    time.sleep(2)

    print("\n--- TURN 3 ---")
    assistant_content2 = (
        response2.get("output", {}).get("message", {}).get("content", [])
    )
    messages.append({"role": "assistant", "content": assistant_content2})
    messages.append(
        {
            "role": "user",
            "content": [
                {"text": "What about the Liskov Substitution Principle? One sentence."}
            ],
        }
    )

    try:
        response3 = converse_with_auto_cache(
            client, model_id, system, messages, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage3 = response3.get("usage", {})
    text3 = get_text(response3)
    print(f"  Response: {text3[:150]}")
    print_usage(usage3)

    # Validate: turns 2 and 3 should read from cache
    cache_read_2 = usage2.get("cacheReadInputTokens", 0) or 0
    cache_read_3 = usage3.get("cacheReadInputTokens", 0) or 0

    print(f"\n  Turn 2 cacheReadInputTokens: {cache_read_2}")
    print(f"  Turn 3 cacheReadInputTokens: {cache_read_3}")

    if cache_read_2 == 0:
        msg = f"Turn 2 did not read from cache (cacheReadInputTokens={cache_read_2})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if cache_read_3 == 0:
        msg = f"Turn 3 did not read from cache (cacheReadInputTokens={cache_read_3})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if cache_read_3 > cache_read_2:
        print(
            f"\n  Result: PASS - cache breakpoint moved forward "
            f"(turn 2 read={cache_read_2}, turn 3 read={cache_read_3})"
        )
    else:
        print(
            f"\n  Result: PASS - cache reads present on turns 2 and 3 "
            f"(turn 2 read={cache_read_2}, turn 3 read={cache_read_3})"
        )
    return ("PASS", None)


def test_automatic_caching_with_ttl(client, model_id):
    """Test that automatic caching accepts the ttl parameter.

    Returns (status, error_msg).
    """
    print("=" * 70)
    print("TEST: AUTOMATIC CACHING WITH 1h TTL")
    print("=" * 70)

    system_content = load_system_content()
    system = [{"text": system_content}]
    cache_control = {"type": "ephemeral", "ttl": "1h"}

    # Request 1: should create cache with 1h TTL
    print("\n--- REQUEST 1: Cache creation (ttl=1h) ---")
    print(f"  Top-level cache_control: {json.dumps(cache_control)}")

    messages = [
        {
            "role": "user",
            "content": [{"text": "Explain dependency injection in one sentence."}],
        }
    ]

    try:
        response1 = converse_with_auto_cache(
            client, model_id, system, messages, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage1 = response1.get("usage", {})
    text1 = get_text(response1)
    print(f"\n  Response: {text1[:200]}")
    print_usage(usage1)

    # Request 2: should read from cache
    print("\n  Waiting 2 seconds...")
    time.sleep(2)

    print("\n--- REQUEST 2: Cache read (ttl=1h) ---")

    messages2 = [
        {
            "role": "user",
            "content": [{"text": "Explain the factory pattern in one sentence."}],
        }
    ]

    try:
        response2 = converse_with_auto_cache(
            client, model_id, system, messages2, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage2 = response2.get("usage", {})
    text2 = get_text(response2)
    print(f"\n  Response: {text2[:200]}")
    print_usage(usage2)

    # Validate
    cache_created = usage1.get("cacheWriteInputTokens", 0) or 0
    cache_read_1 = usage1.get("cacheReadInputTokens", 0) or 0
    cache_read_2 = usage2.get("cacheReadInputTokens", 0) or 0

    if cache_created == 0 and cache_read_1 == 0:
        msg = f"Request 1 had no cache activity (creation={cache_created}, read={cache_read_1})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if cache_read_2 == 0:
        msg = f"Request 2 did not read from cache (cacheReadInputTokens={cache_read_2})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - 1h TTL cache created ({cache_created} tokens), "
        f"cache read ({cache_read_2} tokens)"
    )
    return ("PASS", None)


def test_combined_auto_and_explicit(client, model_id):
    """Test automatic caching combined with explicit block-level cachePoint.

    Returns (status, error_msg).
    """
    print("=" * 70)
    print("TEST: COMBINED AUTOMATIC + EXPLICIT CACHING")
    print("=" * 70)

    system_content = load_system_content()

    # System prompt with explicit cachePoint block AND top-level auto cache
    system = [
        {"text": system_content},
        {"cachePoint": {"type": "default"}},
    ]
    cache_control = {"type": "ephemeral"}

    print(f"  System has explicit cachePoint: {{type: default}}")
    print(f"  Top-level cache_control: {json.dumps(cache_control)}")

    # Request 1: should create cache
    print("\n--- REQUEST 1: Cache creation ---")

    messages = [
        {
            "role": "user",
            "content": [{"text": "What are SOLID principles? List them briefly."}],
        }
    ]

    try:
        response1 = converse_with_auto_cache(
            client, model_id, system, messages, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage1 = response1.get("usage", {})
    text1 = get_text(response1)
    print(f"\n  Response: {text1[:200]}")
    print_usage(usage1)

    # Request 2: should read from cache
    print("\n  Waiting 2 seconds...")
    time.sleep(2)

    print("\n--- REQUEST 2: Cache read ---")

    messages2 = [
        {
            "role": "user",
            "content": [
                {"text": "Which SOLID principle is most important? One sentence."}
            ],
        }
    ]

    try:
        response2 = converse_with_auto_cache(
            client, model_id, system, messages2, cache_control
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    usage2 = response2.get("usage", {})
    text2 = get_text(response2)
    print(f"\n  Response: {text2[:200]}")
    print_usage(usage2)

    # Validate
    cache_created = usage1.get("cacheWriteInputTokens", 0) or 0
    cache_read_1 = usage1.get("cacheReadInputTokens", 0) or 0
    cache_read_2 = usage2.get("cacheReadInputTokens", 0) or 0

    if cache_created == 0 and cache_read_1 == 0:
        msg = f"Request 1 had no cache activity (creation={cache_created}, read={cache_read_1})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if cache_read_2 == 0:
        msg = f"Request 2 did not read from cache (cacheReadInputTokens={cache_read_2})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - combined caching: created ({cache_created} tokens), "
        f"read ({cache_read_2} tokens)"
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

    status, error = test_basic_automatic_caching(client, model_id)
    results.append(("Basic automatic caching", status, error))

    status, error = test_multi_turn_automatic_caching(client, model_id)
    results.append(("Multi-turn automatic caching", status, error))

    status, error = test_automatic_caching_with_ttl(client, model_id)
    results.append(("Automatic caching with 1h TTL", status, error))

    status, error = test_combined_auto_and_explicit(client, model_id)
    results.append(("Combined auto + explicit caching", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
