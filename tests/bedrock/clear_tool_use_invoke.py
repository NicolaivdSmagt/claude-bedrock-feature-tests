#!/usr/bin/env python3
# ABOUTME: Tests context management clear_tool_uses on Amazon Bedrock via the invoke_model API.
# ABOUTME: Verifies that old tool use/result pairs are automatically cleared when a token threshold is reached.

"""
Clear Tool Uses Test for Amazon Bedrock (Invoke API)
=====================================================

Tests the clear_tool_uses_20250919 context management edit via
invoke_model with streaming. Runs a multi-turn conversation with the
memory tool, configured with a low token trigger so old tool use/result
pairs are cleared quickly.

Validates that:
1. The memory tool works (view/create operations succeed)
2. Context management clears old tool uses when the threshold is reached
3. The conversation continues to work after clearing

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/clear_tool_use_invoke.py
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

MEMORY_BETA = "context-management-2025-06-27"
MEMORY_TOOL_TYPE = "memory_20250818"


def handle_memory_tool(tool_input, memory_store):
    """Handle Claude's memory tool operations using in-memory storage."""
    command = tool_input.get("command")
    path = tool_input.get("path", "")

    if command == "view":
        if path == "/memories":
            memories = [
                {"path": f"/memories/{name}", "content": content}
                for name, content in memory_store.items()
            ]
            return {"memories": memories}
        elif path.startswith("/memories/"):
            filename = path.split("/")[-1]
            if filename in memory_store:
                return {"memory": {"path": path, "content": memory_store[filename]}}
            return {"memory": {"error": "not found"}}

    elif command == "create":
        filename = path.split("/")[-1] if "/" in path else "memory.md"
        file_text = tool_input.get("file_text", "")
        memory_store[filename] = file_text
        return {
            "success": True,
            "operation": "create",
            "created": filename,
            "path": f"/memories/{filename}",
            "size_bytes": len(file_text),
            "message": f"Created '{filename}' with {len(file_text)} characters",
        }

    elif command == "str_replace":
        filename = path.split("/")[-1]
        if filename in memory_store:
            content = memory_store[filename]
            old_str = tool_input.get("old_str", "")
            new_str = tool_input.get("new_str", "")
            updated = content.replace(old_str, new_str)
            memory_store[filename] = updated
            return {
                "success": True,
                "operation": "str_replace",
                "file": filename,
                "message": f"Updated '{filename}'",
            }
        return {"error": "File not found", "path": path}

    return {"status": "handled"}


def call_claude_streaming(
    client, model_id, messages, system_prompt, context_management
):
    """Make streaming API call to Claude with context management enabled.

    Returns (content_blocks, usage, context_mgmt).
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": messages,
        "tools": [{"type": MEMORY_TOOL_TYPE, "name": "memory"}],
        "anthropic_beta": [MEMORY_BETA],
        "context_management": context_management,
    }

    response = client.invoke_model_with_response_stream(
        modelId=model_id, body=json.dumps(body)
    )

    content_blocks = []
    usage = {}
    context_mgmt = None

    for event in response["body"]:
        if "chunk" in event:
            chunk = json.loads(event["chunk"]["bytes"].decode())

            if chunk["type"] == "message_start" and "message" in chunk:
                usage = chunk["message"].get("usage", {})
            elif chunk["type"] == "message_delta":
                if "usage" in chunk:
                    usage.update(chunk["usage"])
                if "context_management" in chunk:
                    context_mgmt = chunk["context_management"]
            elif chunk["type"] == "content_block_start":
                block = chunk["content_block"]
                if block["type"] == "text":
                    content_blocks.append({"type": "text", "text": ""})
                elif block["type"] == "tool_use":
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": "",
                        }
                    )
            elif chunk["type"] == "content_block_delta":
                delta = chunk["delta"]
                idx = chunk["index"]
                if delta["type"] == "text_delta":
                    content_blocks[idx]["text"] += delta["text"]
                elif delta["type"] == "input_json_delta":
                    content_blocks[idx]["input"] += delta["partial_json"]

    # Parse tool inputs from accumulated JSON strings
    for block in content_blocks:
        if block["type"] == "tool_use":
            block["input"] = json.loads(block["input"])

    return content_blocks, usage, context_mgmt


def test_clear_tool_uses(client, model_id):
    """Test that context management clears old tool uses. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: CLEAR TOOL USES (clear_tool_uses_20250919)")
    print("=" * 70)

    memory_store = {}

    context_management = {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 500},
                "keep": {"type": "tool_uses", "value": 0},
                "clear_at_least": {"type": "input_tokens", "value": 0},
            }
        ]
    }

    system_prompt = (
        "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE. "
        "MEMORY PROTOCOL: "
        "1. Store all memories in .md format at /memories/ "
        "2. Use the view command to check for earlier progress. "
        "3. WRITE DETAILED MEMORIES with context and explanations. "
        "ASSUME INTERRUPTION: Your context window might be reset at any moment."
    )

    user_prompts = [
        "Store my tech stack: Python, Go, TypeScript, and Rust.",
        "Add my preferred tools: VSCode, Docker, Git, and Kubernetes.",
        "Remember my cloud platforms: AWS, GCP, and Azure.",
    ]

    messages = []
    total_cleared = 0
    total_saved = 0

    print(f"\n  Context management config:")
    print(f"    trigger:       500 input tokens")
    print(f"    keep:          0 tool uses")
    print(f"    clear_at_least: 0 tokens")

    try:
        for turn, prompt in enumerate(user_prompts, 1):
            print(f"\n--- Turn {turn}: {prompt[:60]} ---")

            messages.append({"role": "user", "content": prompt})

            content_blocks, usage, context_mgmt = call_claude_streaming(
                client, model_id, messages, system_prompt, context_management
            )

            messages.append({"role": "assistant", "content": content_blocks})

            tokens = usage.get("input_tokens", 0)
            tool_count = sum(
                1
                for m in messages
                if m["role"] == "assistant"
                for b in m["content"]
                if isinstance(b, dict) and b.get("type") == "tool_use"
            )
            print(
                f"  Input tokens: {tokens} | Total tool uses in history: {tool_count}"
            )

            if context_mgmt:
                print(f"  context_management: {json.dumps(context_mgmt)}")
                edits = context_mgmt.get("applied_edits", [])
                for edit in edits:
                    if edit.get("type") == "clear_tool_uses_20250919":
                        cleared = edit.get("cleared_tool_uses", 0)
                        saved = edit.get("cleared_input_tokens", 0)
                        if cleared > 0:
                            total_cleared += cleared
                            total_saved += saved
                            print(
                                f"  Cleared {cleared} tool uses, saved {saved} tokens"
                            )

            # Execute tool uses and continue the conversation loop
            tool_uses = [b for b in content_blocks if b["type"] == "tool_use"]
            while tool_uses:
                results = []
                for tool_use in tool_uses:
                    result = handle_memory_tool(tool_use["input"], memory_store)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": json.dumps(result),
                        }
                    )
                messages.append({"role": "user", "content": results})

                # Continue conversation if Claude wants to use more tools
                content_blocks, usage, context_mgmt = call_claude_streaming(
                    client, model_id, messages, system_prompt, context_management
                )
                messages.append({"role": "assistant", "content": content_blocks})

                if context_mgmt:
                    print(f"  context_management: {json.dumps(context_mgmt)}")
                    edits = context_mgmt.get("applied_edits", [])
                    for edit in edits:
                        if edit.get("type") == "clear_tool_uses_20250919":
                            cleared = edit.get("cleared_tool_uses", 0)
                            saved = edit.get("cleared_input_tokens", 0)
                            if cleared > 0:
                                total_cleared += cleared
                                total_saved += saved
                                print(
                                    f"  Cleared {cleared} tool uses, saved {saved} tokens"
                                )

                tool_uses = [b for b in content_blocks if b["type"] == "tool_use"]

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    print(f"\n  Total tool uses cleared: {total_cleared}")
    print(f"  Total tokens saved:      {total_saved}")
    print(f"  Memories stored:         {len(memory_store)}")

    if total_cleared == 0:
        msg = "No tool uses were cleared by context management"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if len(memory_store) == 0:
        msg = "No memories were stored during the conversation"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - cleared {total_cleared} tool uses, saved {total_saved} tokens"
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
    print(f"Beta: {MEMORY_BETA}")
    print()

    results = []

    status, error = test_clear_tool_uses(client, model_id)
    results.append(("Clear tool uses", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
