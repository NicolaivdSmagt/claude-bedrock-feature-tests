#!/usr/bin/env python3
# ABOUTME: Tests the memory tool on Amazon Bedrock via the Converse API.
# ABOUTME: Verifies memory creation and retrieval across separate conversation sessions.

"""
Memory Tool Test for Amazon Bedrock (Converse API)
====================================================

Tests the memory tool (memory_20250818) which allows Claude to persist
information across conversation sessions.

The memory tool type and beta header are passed via
additionalModelRequestFields since the Converse API doesn't natively
support these Anthropic-specific tool types. A placeholder toolSpec is
required in toolConfig.tools to satisfy the Converse schema.

Test cases:
1. Session 1: Ask Claude to remember user preferences — verify memory is created
2. Session 2: Ask Claude to recall preferences — verify memory is read

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/memory_test_converse.py
"""

import json
import os
import sys
from pathlib import Path

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

# Memory storage directory
MEMORY_DIR = Path("./memories")
MEMORY_DIR.mkdir(exist_ok=True)


def get_memory_file_path():
    """Get the path to the memory file."""
    return MEMORY_DIR / "claude_memories_converse.md"


def process_memory_command(tool_input):
    """Process a memory tool command and return the result text."""
    command = tool_input.get("command")
    memory_file = get_memory_file_path()

    if command == "view":
        if memory_file.exists():
            content = memory_file.read_text()
            print(f"    [MEMORY] Read {len(content)} chars")
            return content
        else:
            print(f"    [MEMORY] No memories found")
            return ""
    elif command == "create":
        file_text = tool_input.get("file_text", "")
        memory_file.write_text(file_text)
        print(f"    [MEMORY] Wrote {len(file_text)} chars")
        return "Memory created successfully"
    elif command == "str_replace":
        old_str = tool_input.get("old_str", "")
        new_str = tool_input.get("new_str", "")
        current = memory_file.read_text() if memory_file.exists() else ""
        updated = current.replace(old_str, new_str)
        memory_file.write_text(updated)
        print(f"    [MEMORY] str_replace done")
        return "Memory updated successfully"
    else:
        return f"Unknown memory command: {command}"


def run_session(client, model_id, user_message, session_name):
    """Run a conversation session with the memory tool. Returns final response text or None."""
    print(f"\n--- {session_name} ---")
    print(f"  User: {user_message}")

    messages = [{"role": "user", "content": [{"text": user_message}]}]
    max_turns = 10

    for turn in range(1, max_turns + 1):
        request = {
            "modelId": model_id,
            "messages": messages,
            "toolConfig": PLACEHOLDER_TOOL_CONFIG,
            "additionalModelRequestFields": {
                "anthropic_beta": [MEMORY_BETA],
                "tools": [{"type": MEMORY_TOOL_TYPE, "name": "memory"}],
            },
        }

        response = client.converse(**request)

        output_message = response.get("output", {}).get("message", {})
        content = output_message.get("content", [])

        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": content})

        # Collect text and tool use blocks
        text_parts = []
        tool_uses = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(block["text"])
                if "toolUse" in block:
                    tool_uses.append(block["toolUse"])

        if text_parts:
            print(f"  Claude (turn {turn}): {' '.join(text_parts)[:200]}")

        # If no tool use, conversation is done
        if not tool_uses:
            return " ".join(text_parts) if text_parts else None

        # Process tool uses and add results
        tool_results = []
        for tool_use in tool_uses:
            print(f"  [Turn {turn}] memory command: {tool_use['input'].get('command')}")
            result_text = process_memory_command(tool_use["input"])
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"text": result_text}],
                    }
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return None


def test_memory_create_and_recall(client, model_id):
    """Test memory creation and retrieval across sessions. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: MEMORY CREATE AND RECALL")
    print("=" * 70)

    # Clean up any prior memory file
    memory_file = get_memory_file_path()
    if memory_file.exists():
        memory_file.unlink()

    try:
        # Session 1: Create memory
        run_session(
            client,
            model_id,
            (
                "Hi! My name is Nicolai. I'm a software engineer who loves Python and AWS. "
                "I prefer clean, maintainable code over clever solutions. "
                "I use uv for Python package management. "
                "Please remember these preferences."
            ),
            "SESSION 1: Create Memory",
        )

        # Verify memory was written
        if not memory_file.exists() or len(memory_file.read_text().strip()) == 0:
            msg = "Memory file was not created after session 1"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        memory_content = memory_file.read_text()
        print(
            f"\n  Memory file ({len(memory_content)} chars): {memory_content[:200]}..."
        )

        # Session 2: Recall memory (fresh conversation, no prior context)
        response_text = run_session(
            client,
            model_id,
            "Hi! Can you remind me what package manager I prefer to use?",
            "SESSION 2: Recall Memory",
        )

        # Verify Claude mentioned uv in the response
        if response_text and "uv" in response_text.lower():
            print(f"\n  Result: PASS - memory created and recalled successfully")
            return ("PASS", None)
        elif response_text:
            msg = f"Response didn't mention 'uv': {response_text[:200]}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)
        else:
            msg = "No text response from session 2"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
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
    client = get_bedrock_client(config)

    print(f"Model: {model_id}")
    print(f"Beta: {MEMORY_BETA}")
    print(f"Memory file: {get_memory_file_path()}")
    print()

    results = []

    status, error = test_memory_create_and_recall(client, model_id)
    results.append(("Memory create and recall", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
