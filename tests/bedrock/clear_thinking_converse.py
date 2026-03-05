#!/usr/bin/env python3
# ABOUTME: Tests context management clear_thinking on Amazon Bedrock via the Converse API.
# ABOUTME: Verifies that old thinking blocks are automatically cleared to save tokens.

"""
Clear Thinking Test for Amazon Bedrock (Converse API)
=======================================================

Tests the clear_thinking_20251015 context management edit via the
Converse API. Runs a multi-turn conversation with extended thinking
enabled, configured to keep only the most recent thinking turn.

The thinking config, beta header, and context_management config are
passed via additionalModelRequestFields.

Validates that:
1. Extended thinking works (thinking blocks appear in responses)
2. Context management clears old thinking turns when the threshold is reached
3. The conversation continues to work after clearing

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/clear_thinking_converse.py
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

CONTEXT_MGMT_BETA = "context-management-2025-06-27"


def run_conversation(client, model_id, prompts, context_management=None):
    """Run a multi-turn thinking conversation. Returns list of (turn, input_tokens, thinking_seen)."""
    messages = []
    results = []

    additional_fields = {
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    if context_management:
        additional_fields["anthropic_beta"] = [CONTEXT_MGMT_BETA]
        additional_fields["context_management"] = context_management

    for turn, prompt in enumerate(prompts, 1):
        messages.append({"role": "user", "content": [{"text": prompt}]})

        response = client.converse(
            modelId=model_id,
            messages=messages,
            additionalModelRequestFields=additional_fields,
        )

        output_message = response.get("output", {}).get("message", {})
        content = output_message.get("content", [])
        usage = response.get("usage", {})

        messages.append({"role": "assistant", "content": content})

        input_tokens = usage.get("inputTokens", 0)
        thinking_seen = any(
            isinstance(b, dict) and "reasoningContent" in b for b in content
        )

        results.append((turn, input_tokens, thinking_seen))

    return results


def test_clear_thinking(client, model_id):
    """Test that context management clears old thinking blocks. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: CLEAR THINKING (clear_thinking_20251015)")
    print("=" * 70)

    context_management = {
        "edits": [
            {
                "type": "clear_thinking_20251015",
                "keep": {"type": "thinking_turns", "value": 1},
            }
        ]
    }

    user_prompts = [
        "If I have 3 apples and eat one, how many do I have? Explain the logic.",
        "Now I buy 5 more. How many do I have total? Double check your math.",
        "If I share them equally with a friend, how many do we each get? Show your work.",
    ]

    print(f"\n  Context management config:")
    print(f"    keep: 1 thinking turn")

    # The Converse API does not return a context_management field in
    # responses, so we verify clearing by comparing input token counts
    # between a run WITH clearing and a run WITHOUT clearing. If clearing
    # works, later turns should use fewer input tokens.

    try:
        print("\n--- Run 1: WITH clear_thinking ---")
        with_clearing = run_conversation(
            client, model_id, user_prompts, context_management
        )
        for turn, tokens, thinking in with_clearing:
            print(f"  Turn {turn}: inputTokens={tokens}, thinking={thinking}")

        print("\n--- Run 2: WITHOUT clear_thinking (baseline) ---")
        without_clearing = run_conversation(client, model_id, user_prompts)
        for turn, tokens, thinking in without_clearing:
            print(f"  Turn {turn}: inputTokens={tokens}, thinking={thinking}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    # Check that thinking blocks were produced
    any_thinking = any(thinking for _, _, thinking in with_clearing)
    if not any_thinking:
        msg = "No thinking blocks were produced"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    # Compare token counts on turns 2+ (turn 1 has nothing to clear)
    print("\n  Token comparison (turn 2+):")
    total_saved = 0
    clearing_observed = False
    for (turn_w, tokens_w, _), (turn_b, tokens_b, _) in zip(
        with_clearing[1:], without_clearing[1:]
    ):
        saved = tokens_b - tokens_w
        total_saved += saved
        print(f"    Turn {turn_w}: with={tokens_w}, without={tokens_b}, saved={saved}")
        if saved > 0:
            clearing_observed = True

    if not clearing_observed:
        msg = "No token savings observed from clearing (with-clearing tokens >= without-clearing tokens)"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - clearing reduced input tokens by {total_saved} across turns 2+"
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
    print(f"Beta: {CONTEXT_MGMT_BETA}")
    print()

    results = []

    status, error = test_clear_thinking(client, model_id)
    results.append(("Clear thinking", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
