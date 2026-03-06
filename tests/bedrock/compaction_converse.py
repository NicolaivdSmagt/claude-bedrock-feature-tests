#!/usr/bin/env python3
# ABOUTME: Tests Bedrock message compaction via the Converse API.
# ABOUTME: Loads a pre-built conversation, converts it to Converse format, and verifies compaction.

"""
Loads a static conversation history from 50000_token_conversation.json (generated once
by generate_compaction_history.py) and sends it to Bedrock via the Converse API with
compaction enabled via additionalModelRequestFields.

The conversation history is stored in Anthropic-native format (text/tool_use/tool_result
blocks) and is converted to Converse format (text/toolUse/toolResult blocks) at runtime.

Usage:
    uv run python tests/bedrock/compaction_converse.py
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

# Error markers indicating that a feature is not available on Bedrock. These
# map to FAIL (feature not supported) rather than ERROR (transient/infra).
FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "not supported",
    "not currently supported",
    "unknown field",
    "unrecognized",
    "unknown tool type",
    "does not match any of the expected tags",
]


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other).
    Returns (status, error_msg)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


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


def convert_tools_to_converse(tools):
    """Convert Anthropic-native tool definitions to Converse toolConfig format."""
    converse_tools = []
    for tool in tools:
        converse_tools.append(
            {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": {
                        "json": tool["input_schema"],
                    },
                }
            }
        )
    return {"tools": converse_tools}


def convert_content_block(block):
    """Convert an Anthropic-native content block to Converse format."""
    block_type = block.get("type")

    if block_type == "text":
        return {"text": block["text"]}

    if block_type == "tool_use":
        return {
            "toolUse": {
                "toolUseId": block["id"],
                "name": block["name"],
                "input": block["input"],
            }
        }

    if block_type == "tool_result":
        content_value = block.get("content", "")
        # tool_result content can be a string or a list of blocks
        if isinstance(content_value, str):
            result_content = [{"text": content_value}]
        elif isinstance(content_value, list):
            result_content = []
            for sub_block in content_value:
                if isinstance(sub_block, str):
                    result_content.append({"text": sub_block})
                elif isinstance(sub_block, dict) and sub_block.get("type") == "text":
                    result_content.append({"text": sub_block["text"]})
                else:
                    result_content.append({"text": json.dumps(sub_block)})
        else:
            result_content = [{"text": str(content_value)}]

        return {
            "toolResult": {
                "toolUseId": block["tool_use_id"],
                "content": result_content,
            }
        }

    # Fallback: treat as text
    return {"text": json.dumps(block)}


def convert_messages_to_converse(messages):
    """Convert Anthropic-native messages list to Converse format.

    Anthropic format has:
      - string content: "content": "some text"
      - list content: "content": [{"type": "text", ...}, {"type": "tool_use", ...}, ...]

    Converse format has:
      - list content: "content": [{"text": "..."}, {"toolUse": {...}}, {"toolResult": {...}}]

    In Anthropic format, tool_result blocks appear in user messages alongside other content.
    In Converse format, toolResult blocks also appear in user message content.
    """
    converse_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            converse_messages.append(
                {
                    "role": role,
                    "content": [{"text": content}],
                }
            )
        elif isinstance(content, list):
            converse_content = [convert_content_block(block) for block in content]
            converse_messages.append(
                {
                    "role": role,
                    "content": converse_content,
                }
            )
        else:
            converse_messages.append(
                {
                    "role": role,
                    "content": [{"text": str(content)}],
                }
            )

    return converse_messages


def test_compaction(client, model_id, messages, system, tools, stored_token_count):
    """Send a compaction request via Converse. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: MESSAGE COMPACTION (Converse)")
    print("=" * 70)

    converse_messages = convert_messages_to_converse(messages)
    tool_config = convert_tools_to_converse(tools)

    request = {
        "modelId": model_id,
        "system": [{"text": system}],
        "messages": converse_messages,
        "toolConfig": tool_config,
        "additionalModelRequestFields": {
            "anthropic_beta": [COMPACTION_BETA],
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
        },
    }

    # Print request with messages replaced by placeholder
    request_display = dict(request)
    request_display["messages"] = (
        f"[LARGE_CONVERSATION_HISTORY: {len(messages)} turns, {stored_token_count} tokens]"
    )
    print("\n--- REQUEST ---")
    print(json.dumps(request_display, indent=2, default=str))

    print(f"\nSending to {model_id}...")
    start = time.time()

    try:
        response = client.converse(**request)
        elapsed = time.time() - start

        print(f"Response received in {elapsed:.1f}s\n")
        print("--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        # Validate: response should have output content and a valid stopReason
        stop_reason = response.get("stopReason")
        output_message = response.get("output", {}).get("message", {})
        content = output_message.get("content", [])

        if not content:
            msg = "Response has no content blocks"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        if stop_reason not in ("end_turn", "tool_use"):
            msg = f"Unexpected stopReason={stop_reason!r}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        print(
            f"\n  Result: PASS - stopReason={stop_reason!r}, {len(content)} content block(s), {elapsed:.1f}s"
        )
        return ("PASS", None)

    except Exception as e:
        elapsed = time.time() - start
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\nResponse failed after {elapsed:.1f}s")
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

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
    results.append(("Compaction (Converse)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
