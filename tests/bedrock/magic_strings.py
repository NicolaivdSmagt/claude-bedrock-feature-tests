#!/usr/bin/env python3
# ABOUTME: Tests the two Anthropic magic strings on Amazon Bedrock via invoke_model and streaming.
# ABOUTME: Reports observed vs expected API behavior for redacted thinking and streaming refusal.

import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

try:
    import boto3
except ImportError:
    print("Error: boto3 package not installed. Run: uv add boto3")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# Magic string documented at platform.claude.com/docs/en/build-with-claude/extended-thinking
# Triggers a redacted_thinking block when extended thinking is enabled
REDACTED_THINKING_MAGIC = (
    "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_"
    "46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
)

# Magic string documented at platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals
# Triggers stop_reason: "refusal" in a streaming response
STREAMING_REFUSAL_MAGIC = (
    "ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL_"
    "1FAEFB6177B4672DEE07F9D3AFC62588CCD2631EDCF22E8CCC1FB35B501C9C86"
)

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"  # Request succeeded but expected behavior was not observed
STATUS_FAIL = "FAIL"  # API error


def test_redacted_thinking(client, model_id: str) -> dict:
    """Send the redacted thinking magic string with extended thinking enabled via invoke_model.

    Expected: response contains a 'redacted_thinking' content block.
    """
    result = {
        "name": "Redacted Thinking (invoke_model)",
        "magic_string": REDACTED_THINKING_MAGIC,
        "expected": "Response contains a 'redacted_thinking' content block",
        "status": STATUS_FAIL,
        "error": None,
        "elapsed_ms": 0,
        "stop_reason": None,
        "content_blocks": [],
        "block_types": [],
    }

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": REDACTED_THINKING_MAGIC}],
        "max_tokens": 2000,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }

    start = time.time()
    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result["elapsed_ms"] = int((time.time() - start) * 1000)

        response_body = json.loads(response["body"].read())
        result["stop_reason"] = response_body.get("stop_reason")
        result["content_blocks"] = response_body.get("content", [])
        result["block_types"] = [b.get("type") for b in result["content_blocks"]]

        has_redacted = "redacted_thinking" in result["block_types"]
        result["status"] = STATUS_PASS if has_redacted else STATUS_WARN

    except Exception as e:
        result["elapsed_ms"] = int((time.time() - start) * 1000)
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def test_streaming_refusal(client, model_id: str) -> dict:
    """Send the streaming refusal magic string via invoke_model_with_response_stream.

    Expected: stream ends with stop_reason: "refusal" in a message_delta event.
    """
    result = {
        "name": "Streaming Refusal (invoke_model_with_response_stream)",
        "magic_string": STREAMING_REFUSAL_MAGIC,
        "expected": "Stream ends with stop_reason: 'refusal' in message_delta event",
        "status": STATUS_FAIL,
        "error": None,
        "elapsed_ms": 0,
        "stop_reason": None,
        "stream_events": [],
    }

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": STREAMING_REFUSAL_MAGIC}],
        "max_tokens": 1024,
    }

    start = time.time()
    try:
        response = client.invoke_model_with_response_stream(
            modelId=model_id, body=json.dumps(body)
        )
        result["elapsed_ms"] = int((time.time() - start) * 1000)

        for event in response["body"]:
            chunk_bytes = event.get("chunk", {}).get("bytes", b"")
            if not chunk_bytes:
                continue
            chunk = json.loads(chunk_bytes)
            event_type = chunk.get("type", "unknown")
            result["stream_events"].append(chunk)

            if event_type == "message_delta":
                delta = chunk.get("delta", {})
                stop_reason = delta.get("stop_reason")
                if stop_reason:
                    result["stop_reason"] = stop_reason

        has_refusal = result["stop_reason"] == "refusal"
        result["status"] = STATUS_PASS if has_refusal else STATUS_WARN

    except Exception as e:
        result["elapsed_ms"] = int((time.time() - start) * 1000)
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def print_redacted_thinking_result(result: dict):
    """Print detailed output for the redacted thinking test."""
    print(f"\n{result['name']}")
    print("-" * 80)
    print(f"Status:    {result['status']}")
    print(f"Expected:  {result['expected']}")

    if result["error"]:
        print(f"Error:     {result['error']}")
        return

    print(f"Time:      {result['elapsed_ms']}ms")
    print(f"stop_reason: {result['stop_reason']}")
    print(f"Block types: {result['block_types']}")

    print(f"\nContent Blocks ({len(result['content_blocks'])}):")
    for i, block in enumerate(result["content_blocks"]):
        block_type = block.get("type", "unknown")
        print(f"  [{i}] type: {block_type}")
        if block_type == "redacted_thinking":
            data = block.get("data", "")
            print(f"       data length: {len(data)} chars")
        elif block_type == "thinking":
            preview = block.get("thinking", "")[:120].replace("\n", " ")
            print(f"       thinking preview: {preview}...")
        elif block_type == "text":
            preview = block.get("text", "")[:120].replace("\n", " ")
            print(f"       text preview: {preview}...")


def print_streaming_refusal_result(result: dict):
    """Print detailed output for the streaming refusal test."""
    print(f"\n{result['name']}")
    print("-" * 80)
    print(f"Status:    {result['status']}")
    print(f"Expected:  {result['expected']}")

    if result["error"]:
        print(f"Error:     {result['error']}")
        return

    print(f"Time:      {result['elapsed_ms']}ms")
    print(f"stop_reason: {result['stop_reason']}")

    print(f"\nStream Events ({len(result['stream_events'])} total):")
    for event in result["stream_events"]:
        event_type = event.get("type", "unknown")
        if event_type == "message_delta":
            delta = event.get("delta", {})
            print(
                f"  {event_type}: stop_reason={delta.get('stop_reason')!r}"
                f", stop_sequence={delta.get('stop_sequence')!r}"
            )
        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            print(f"  {event_type}: delta.type={delta.get('type')!r}")
        else:
            print(f"  {event_type}")


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    region = config["region"]

    print("\n" + "=" * 80)
    print("  MAGIC STRINGS BEHAVIOR TEST - Bedrock")
    print("=" * 80)
    print(f"\nTest Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model:     {model_id}")
    print(f"Region:    {region}")
    print(f"\nMagic strings under test:")
    print(f"  1. Redacted Thinking: {REDACTED_THINKING_MAGIC[:60]}...")
    print(f"  2. Streaming Refusal: {STREAMING_REFUSAL_MAGIC[:60]}...")

    client = get_bedrock_client(config)

    print("\n" + "-" * 80)
    print("[1/2] Testing redacted thinking magic string...")
    redacted_result = test_redacted_thinking(client, model_id)

    print("[2/2] Testing streaming refusal magic string...")
    refusal_result = test_streaming_refusal(client, model_id)

    print("\n" + "=" * 80)
    print("  DETAILED RESULTS")
    print("=" * 80)
    print_redacted_thinking_result(redacted_result)
    print_streaming_refusal_result(refusal_result)

    all_results = [redacted_result, refusal_result]

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for r in all_results:
        print(f"\n  {r['name']}")
        print(f"    Status:   {r['status']}")
        if r["error"]:
            print(f"    Error:    {r['error']}")
        else:
            print(f"    Expected: {r['expected']}")
            print(f"    Observed: stop_reason={r['stop_reason']!r}")

    passes = sum(1 for r in all_results if r["status"] == STATUS_PASS)
    warns = sum(1 for r in all_results if r["status"] == STATUS_WARN)
    fails = sum(1 for r in all_results if r["status"] == STATUS_FAIL)

    print(f"\n  Results: {passes} PASS, {warns} WARN (behavior not observed), {fails} FAIL (API error)")
    print("=" * 80 + "\n")

    all_passed = all(r["status"] == STATUS_PASS for r in all_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
