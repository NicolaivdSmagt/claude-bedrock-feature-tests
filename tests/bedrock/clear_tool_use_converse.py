#!/usr/bin/env python3
# ABOUTME: Tests context management clear_tool_uses on Amazon Bedrock via the Converse API.
# ABOUTME: Verifies that old tool use/result pairs are automatically cleared when a token threshold is reached.

"""
Clear Tool Uses Test for Amazon Bedrock (Converse API)
=======================================================

Tests the clear_tool_uses_20250919 context management edit via the
Converse API. Runs a multi-turn conversation with the memory tool,
configured with a low token trigger so old tool use/result pairs are
cleared quickly.

The memory tool type, beta header, and context_management config are
passed via additionalModelRequestFields. A placeholder toolSpec is
required in toolConfig.tools to satisfy the Converse schema.

Validates that:
1. The memory tool works (view/create operations succeed)
2. Context management clears old tool uses when the threshold is reached
3. The conversation continues to work after clearing

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/clear_tool_use_converse.py
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

# Converse requires at least one toolSpec in toolConfig; this placeholder satisfies that.
PLACEHOLDER_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "placeholder",
                "inputSchema": {"json": {"type": "object"}},
            }
        }
    ]
}


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


SYSTEM_TEXT = (
    "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE. "
    "MEMORY PROTOCOL: "
    "1. Store all memories in .md format at /memories/ "
    "2. Use the view command to check for earlier progress. "
    "3. WRITE DETAILED MEMORIES with context and explanations. "
    "ASSUME INTERRUPTION: Your context window might be reset at any moment."
)

CONTEXT_MANAGEMENT = {
    "edits": [
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 500},
            "keep": {"type": "tool_uses", "value": 0},
            "clear_at_least": {"type": "input_tokens", "value": 0},
        }
    ]
}

USER_PROMPTS = [
    "Store my tech stack: Python, Go, TypeScript, and Rust.",
    "Add my preferred tools: VSCode, Docker, Git, and Kubernetes.",
    "Remember my cloud platforms: AWS, GCP, and Azure.",
]


def call_claude_converse(client, model_id, messages, context_management=None):
    """Make Converse API call to Claude, optionally with context management.

    Returns (content, usage).
    """
    additional = {
        "anthropic_beta": [MEMORY_BETA],
        "tools": [{"type": MEMORY_TOOL_TYPE, "name": "memory"}],
    }
    if context_management:
        additional["context_management"] = context_management

    response = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_TEXT}],
        messages=messages,
        toolConfig=PLACEHOLDER_TOOL_CONFIG,
        additionalModelRequestFields=additional,
    )

    output_message = response.get("output", {}).get("message", {})
    content = output_message.get("content", [])
    usage = response.get("usage", {})

    return content, usage


def run_conversation(client, model_id, label, context_management=None):
    """Run a full multi-turn conversation with memory tool.

    Returns (last_turn_input_tokens, memory_count, per_turn_tokens).
    The Converse API does not return context_management stats, so we
    track input token counts per API call to compare with/without clearing.
    """
    memory_store = {}
    messages = []
    per_turn_tokens = []

    print(f"\n--- {label} ---")

    for turn, prompt in enumerate(USER_PROMPTS, 1):
        messages.append({"role": "user", "content": [{"text": prompt}]})

        content, usage = call_claude_converse(
            client, model_id, messages, context_management
        )
        messages.append({"role": "assistant", "content": content})
        per_turn_tokens.append(usage.get("inputTokens", 0))

        # Execute tool uses in a loop until no more tool calls
        tool_uses = [
            b["toolUse"] for b in content if isinstance(b, dict) and "toolUse" in b
        ]
        while tool_uses:
            results = []
            for tool_use in tool_uses:
                result = handle_memory_tool(tool_use["input"], memory_store)
                results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"text": json.dumps(result)}],
                        }
                    }
                )
            messages.append({"role": "user", "content": results})

            content, usage = call_claude_converse(
                client, model_id, messages, context_management
            )
            messages.append({"role": "assistant", "content": content})
            per_turn_tokens.append(usage.get("inputTokens", 0))

            tool_uses = [
                b["toolUse"] for b in content if isinstance(b, dict) and "toolUse" in b
            ]

    # Print token progression
    for i, tokens in enumerate(per_turn_tokens):
        print(f"    API call {i + 1}: inputTokens={tokens}")

    last_tokens = per_turn_tokens[-1] if per_turn_tokens else 0
    return last_tokens, len(memory_store), per_turn_tokens


def test_clear_tool_uses(client, model_id):
    """Test that context management clears old tool uses. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: CLEAR TOOL USES (clear_tool_uses_20250919)")
    print("=" * 70)

    print(f"\n  Context management config:")
    print(f"    trigger:       500 input tokens")
    print(f"    keep:          0 tool uses")
    print(f"    clear_at_least: 0 tokens")

    # The Converse API does not return a context_management field in
    # responses. To verify clearing, we compare input token counts between
    # a run WITH clearing and a run WITHOUT clearing.

    try:
        last_with, mem_with, tokens_with = run_conversation(
            client, model_id, "Run 1: WITH clear_tool_uses", CONTEXT_MANAGEMENT
        )
        last_without, mem_without, tokens_without = run_conversation(
            client, model_id, "Run 2: WITHOUT clear_tool_uses (baseline)", None
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    if mem_with == 0:
        msg = "No memories were stored during the conversation"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    # Compare the last API call's input tokens
    saved = last_without - last_with
    print(
        f"\n  Last API call tokens: with={last_with}, without={last_without}, saved={saved}"
    )
    print(f"  Memories stored: {mem_with}")

    if saved <= 0:
        msg = (
            f"No token savings from clearing (with={last_with}, without={last_without})"
        )
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(f"\n  Result: PASS - clearing reduced last-call input tokens by {saved}")
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
