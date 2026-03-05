#!/usr/bin/env python3
# ABOUTME: Tests Bedrock message compaction via the invoke_model API.
# ABOUTME: Loads a pre-built conversation and verifies compaction triggers correctly.

"""
Loads a static conversation history from 50000_token_conversation.json (generated once
by generate_compaction_history.py) and sends it to Bedrock with compaction enabled.

Prints the raw API request body and response body as JSON, replacing only the large
messages array with a placeholder.

Usage:
    uv run python tests/bedrock/compaction_invoke.py
"""

import json
import os
import sys
import time

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("Error: boto3 package not installed. Run: uv add boto3")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config

COMPACTION_TRIGGER_TOKENS = 50_000
COMPACTION_BETA = "compact-2026-01-12"

CUSTOM_COMPACTION_PROMPT = (
    "You have written a partial transcript for the initial task above. "
    "Please write a summary of the transcript. The purpose of this summary "
    "is to provide continuity so you can continue to make progress towards "
    "solving the task in a future context, where the raw history above may "
    "not be accessible and will be replaced with this summary. Write down "
    "anything that would be helpful, including the state, next steps, "
    "learnings etc. You must wrap your summary in a <summary></summary> block. "
)

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "files",
    "50000_token_conversation.json",
)


def setup_client(region):
    config = Config(region_name=region, read_timeout=600, retries=dict(max_attempts=3))
    return boto3.client("bedrock-runtime", config=config)


def test_compaction(client, model_id, messages, system, tools, stored_token_count):
    """Send a compaction request via invoke_model. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: MESSAGE COMPACTION (invoke_model)")
    print("=" * 70)

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [COMPACTION_BETA],
        "max_tokens": 4096,
        "system": system,
        "tools": tools,
        "messages": messages,
        "context_management": {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {
                        "type": "input_tokens",
                        "value": COMPACTION_TRIGGER_TOKENS,
                    },
                    "pause_after_compaction": False,
                    "instructions": CUSTOM_COMPACTION_PROMPT,
                }
            ]
        },
    }

    # Print request with messages replaced by placeholder
    request_display = dict(request_body)
    request_display["messages"] = (
        f"[LARGE_CONVERSATION_HISTORY: {len(messages)} turns, {stored_token_count} tokens]"
    )
    print("\n--- REQUEST BODY ---")
    print(json.dumps(request_display, indent=2))

    print(f"\nSending to {model_id}...")
    start = time.time()

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(request_body))
        response_body = json.loads(response["body"].read())
        elapsed = time.time() - start

        print(f"Response received in {elapsed:.1f}s\n")
        print("--- RESPONSE BODY ---")
        print(json.dumps(response_body, indent=2, default=str))

        # Validate: response should have content and a valid stop_reason
        stop_reason = response_body.get("stop_reason")
        content = response_body.get("content", [])
        if not content:
            msg = "Response has no content blocks"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        if stop_reason not in ("end_turn", "tool_use"):
            msg = f"Unexpected stop_reason={stop_reason!r}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        print(
            f"\n  Result: PASS - stop_reason={stop_reason!r}, {len(content)} content block(s), {elapsed:.1f}s"
        )
        return ("PASS", None)

    except Exception as e:
        elapsed = time.time() - start
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\nResponse failed after {elapsed:.1f}s")
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
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
    model_id = config["bedrock_model_id"]
    region = config["region"]

    if not os.path.exists(HISTORY_FILE):
        print(
            f"ERROR: {HISTORY_FILE} not found. Run generate_compaction_history.py first."
        )
        sys.exit(1)

    with open(HISTORY_FILE) as f:
        data = json.load(f)

    system = data["system"]
    tools = data["tools"]
    messages = data["messages"]
    stored_token_count = data["token_count"]

    # Append final user message that pushes over the trigger threshold
    messages.append(
        {
            "role": "user",
            "content": (
                "Now that we've completed the full migration, let's move on to the next phase. "
                "Can you set up the data migration pipeline to move existing orders from the "
                "MySQL monolith to the new PostgreSQL microservice? We need to handle UUID mapping, "
                "data validation, and a rollback strategy."
            ),
        }
    )

    client = setup_client(region)

    results = []
    status, error = test_compaction(
        client, model_id, messages, system, tools, stored_token_count
    )
    results.append(("Compaction (invoke_model)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
