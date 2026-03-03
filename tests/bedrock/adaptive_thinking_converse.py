#!/usr/bin/env python3
# ABOUTME: Tests adaptive thinking on Bedrock Converse API with Claude Sonnet 4.6 across all effort levels
# ABOUTME: Runs low/medium/high/max effort and reports thinking blocks, text output, and token usage

import boto3
import json
import os
import sys
import time
from datetime import datetime

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

EFFORT_LEVELS = ["low", "medium", "high", "max"]

PROMPT = (
    "You are about to leave for holiday, but you forgot socks! You race back to your "
    "room, but the power is off so you can't see sock colors. Never mind, because you "
    "remember that in your drawer there are ten pairs of identical green socks, ten "
    "pairs of identical black socks, and eleven pairs of identical blue socks, but they "
    "are all mixed up. How many of your socks do you need to take before you can be "
    "sure to have at least one pair matching in color? Give your answer as a single "
    "number first, then explain your reasoning."
)

EXPECTED_ANSWER = 4


def run_adaptive_thinking(client, effort_level, model_id):
    """Run a single adaptive thinking request via the Converse API at the given effort level.

    Returns a dict with the result details: thinking text, response text,
    token usage, timing, and any error encountered.
    """
    request_params = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [{"text": PROMPT}],
            }
        ],
        "inferenceConfig": {
            "maxTokens": 8192,
        },
        "additionalModelRequestFields": {
            "thinking": {
                "type": "adaptive",
            },
            "output_config": {
                "effort": effort_level,
            },
        },
    }

    result = {
        "effort": effort_level,
        "model_id": model_id,
        "request_params": request_params,
        "thinking_blocks": [],
        "text_blocks": [],
        "usage": {},
        "stop_reason": None,
        "error": None,
        "elapsed_ms": 0,
    }

    start = time.time()
    try:
        response = client.converse(**request_params)
        elapsed_ms = int((time.time() - start) * 1000)
        result["elapsed_ms"] = elapsed_ms

        usage = response.get("usage", {})
        result["usage"] = {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        }
        result["stop_reason"] = response.get("stopReason")

        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        for block in content:
            if "reasoningContent" in block:
                reasoning = block["reasoningContent"]
                reasoning_text = reasoning.get("reasoningText", {})
                result["thinking_blocks"].append(reasoning_text.get("text", ""))
            elif "text" in block:
                result["text_blocks"].append(block["text"])
            else:
                # Capture any unexpected block types
                block_type = list(block.keys())[0] if block else "unknown"
                result["text_blocks"].append(
                    f"[{block_type}]: {json.dumps(block, default=str)[:300]}"
                )

    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        result["elapsed_ms"] = elapsed_ms
        result["error"] = str(e)

    return result


def print_result(result):
    """Pretty-print the result of a single effort-level run."""
    effort = result["effort"]
    print(f"\n{'=' * 70}")
    print(f"  EFFORT: {effort.upper()}")
    print(f"{'=' * 70}")

    print(f"\n  Model ID: {result['model_id']}")
    print(f"\n  Request Params:")
    print(f"  {json.dumps(result['request_params'], indent=4, default=str)}")

    if result["error"]:
        print(f"\n  STATUS: ERROR")
        print(f"  Error:  {result['error']}")
        print(f"  Time:   {result['elapsed_ms']}ms")
        return

    print(f"\n  STATUS: OK")
    print(f"  Time:   {result['elapsed_ms']}ms")
    print(f"  Stop:   {result['stop_reason']}")

    usage = result["usage"]
    print(f"\n  Token Usage:")
    print(f"    Input tokens:  {usage.get('input_tokens', 0)}")
    print(f"    Output tokens: {usage.get('output_tokens', 0)}")
    print(f"    Total tokens:  {usage.get('total_tokens', 0)}")

    # Thinking blocks
    if result["thinking_blocks"]:
        print(f"\n  Thinking Blocks ({len(result['thinking_blocks'])}):")
        for i, thinking in enumerate(result["thinking_blocks"]):
            print(f"  --- thinking block {i + 1} ({len(thinking)} chars) ---")
            if len(thinking) > 2000:
                print(f"  {thinking[:2000]}")
                print(f"  ... [truncated, {len(thinking)} chars total]")
            else:
                print(f"  {thinking}")
    else:
        print(f"\n  Thinking Blocks: NONE (Claude chose not to think)")

    # Text response
    if result["text_blocks"]:
        print(f"\n  Text Response:")
        for i, text in enumerate(result["text_blocks"]):
            print(f"  --- text block {i + 1} ---")
            print(f"  {text}")
    else:
        print(f"\n  Text Response: NONE")


def print_comparison(results):
    """Print a side-by-side comparison table of all effort levels."""
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"  {'Effort':<10} {'Status':<10} {'Output Tok':<12} "
        f"{'Think Chars':<14} {'Time (ms)':<12} {'Answer'}"
    )
    print(f"  {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 14} {'-' * 12} {'-' * 10}")

    for r in results:
        status = "ERROR" if r["error"] else "OK"
        output_tok = r["usage"].get("output_tokens", "-") if not r["error"] else "-"
        think_chars = (
            sum(len(t) for t in r["thinking_blocks"]) if r["thinking_blocks"] else 0
        )
        think_str = str(think_chars) if not r["error"] else "-"

        all_text = " ".join(r["text_blocks"])
        if r["error"]:
            answer = "-"
        elif str(EXPECTED_ANSWER) in all_text:
            answer = f"correct ({EXPECTED_ANSWER})"
        else:
            answer = "check output"

        print(
            f"  {r['effort']:<10} {status:<10} {str(output_tok):<12} "
            f"{think_str:<14} {r['elapsed_ms']:<12} {answer}"
        )


def main():
    config = load_config()
    MODEL_ID = config["bedrock_model_id"]
    REGION = config["region"]

    print(f"\n{'=' * 70}")
    print(f"  ADAPTIVE THINKING TEST - Claude Sonnet 4.6 on Bedrock Converse API")
    print(f"{'=' * 70}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model:  {MODEL_ID}")
    print(f"  Region: {REGION}")
    print(f"  API:    Converse")
    print(f"  Effort: {', '.join(EFFORT_LEVELS)}")
    print(f"\n  Prompt: {PROMPT[:80]}...")
    print(f"  Expected answer: {EXPECTED_ANSWER}")

    client = get_bedrock_client(config)

    results = []
    for i, effort in enumerate(EFFORT_LEVELS, 1):
        print(f"\n[{i}/{len(EFFORT_LEVELS)}] Running effort={effort}...")
        result = run_adaptive_thinking(client, effort, MODEL_ID)
        results.append(result)
        print_result(result)

    print_comparison(results)

    # Final verdict
    errors = [r for r in results if r["error"]]
    successes = [r for r in results if not r["error"]]
    print(f"\n{'=' * 70}")
    print(f"  VERDICT: {len(successes)} succeeded, {len(errors)} errored")
    if errors:
        print(f"  Errored levels: {', '.join(r['effort'] for r in errors)}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
