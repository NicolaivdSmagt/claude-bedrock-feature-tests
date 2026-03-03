#!/usr/bin/env python3
# ABOUTME: Tests Claude Sonnet 4.5 memory tool feature on AWS Bedrock with Converse API
# ABOUTME: Demonstrates how Claude autonomously stores, reads, and updates memories across conversation sessions

import boto3
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)

# Memory storage directory
MEMORY_DIR = Path("./memories")
MEMORY_DIR.mkdir(exist_ok=True)


def get_memory_file_path() -> Path:
    """Get the path to the memory file."""
    return MEMORY_DIR / "claude_memories.md"


def view_memories() -> str:
    """View all stored memories."""
    memory_file = get_memory_file_path()
    if memory_file.exists():
        content = memory_file.read_text()
        print(f"\n[MEMORY SYSTEM] Reading memory ({len(content)} chars)")
        return content
    else:
        print(f"\n[MEMORY SYSTEM] No memories found")
        return ""


def create_memory(content: str) -> str:
    """Create or replace memory content."""
    memory_file = get_memory_file_path()
    memory_file.write_text(content)
    print(f"\n[MEMORY SYSTEM] Writing memory ({len(content)} chars)")
    return "Memory created successfully"


def update_memory(content: str) -> str:
    """Update existing memory content (overwrites current content)."""
    return create_memory(content)


def process_memory_tool_use(tool_use: Dict[str, Any]) -> Dict[str, Any]:
    """Process a memory tool call from Claude."""
    tool_input = tool_use.get("input", {})
    command = tool_input.get("command")

    print(f"\n[TOOL] memory command: {command}")
    print(f"  Input: {json.dumps(tool_input, indent=2)}")

    result = None

    if command == "view":
        result = view_memories()
    elif command == "create":
        file_text = tool_input.get("file_text", "")
        result = create_memory(file_text)
    elif command == "str_replace":
        old_str = tool_input.get("old_str", "")
        new_str = tool_input.get("new_str", "")
        print(f"  Replacing: {old_str[:50]}... -> {new_str[:50]}...")
        current_content = view_memories()
        updated_content = current_content.replace(old_str, new_str)
        result = update_memory(updated_content)
    else:
        result = f"Unknown memory command: {command}"

    return {"text": result}


def run_conversation(
    bedrock_runtime, model_id: str, user_message: str, conversation_name: str
):
    """Run a single conversation with memory enabled."""

    print(f"\n{'=' * 70}")
    print(f"{conversation_name}")
    print(f"{'=' * 70}")
    print(f"USER: {user_message}\n")

    messages = [{"role": "user", "content": [{"text": user_message}]}]

    # Conversation loop
    max_turns = 10
    turn = 0

    while turn < max_turns:
        turn += 1

        # Prepare request with memory tool
        request = {
            "modelId": model_id,
            "messages": messages,
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "placeholder",
                            "inputSchema": {"json": {"type": "object"}},
                        }
                    }
                ]
            },
            "additionalModelRequestFields": {
                "anthropic_beta": ["context-management-2025-06-27"],
                "tools": [{"type": "memory_20250818", "name": "memory"}],
            },
        }

        print(f"\n[API REQUEST - Turn {turn}]")
        print(
            f"  Beta headers: {request['additionalModelRequestFields']['anthropic_beta']}"
        )
        print(
            f"  Tools: {[t['name'] for t in request['additionalModelRequestFields']['tools']]}"
        )

        # Call Claude
        response = bedrock_runtime.converse(**request)

        # Extract response content
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": content})

        # Process response
        tool_uses = []

        for content_item in content:
            if isinstance(content_item, dict):
                if "text" in content_item:
                    print(f"\nCLAUDE: {content_item['text']}")

                if "toolUse" in content_item:
                    tool_uses.append(content_item["toolUse"])

        # If no tool use, conversation is complete
        if not tool_uses:
            break

        # Process tool uses
        tool_results = []
        for tool_use in tool_uses:
            tool_use_id = tool_use.get("toolUseId")
            tool_name = tool_use.get("name")

            # Handle memory tool
            if tool_name == "memory":
                result = process_memory_tool_use(tool_use)
            else:
                result = {"text": f"Unknown tool: {tool_name}"}

            tool_results.append(
                {"toolResult": {"toolUseId": tool_use_id, "content": [result]}}
            )

        # Add tool results to conversation
        messages.append({"role": "user", "content": tool_results})

    # Print usage statistics
    usage = response.get("usage", {})
    print(
        f"\nTokens: in={usage.get('inputTokens', 0)} out={usage.get('outputTokens', 0)}"
    )


def main():
    # Initialize Bedrock Runtime client
    config = load_config()
    bedrock_runtime = get_bedrock_client(config)

    model_id = config["bedrock_model_id"]

    print("\n" + "=" * 70)
    print("MEMORY DEMONSTRATION")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Memory file: {get_memory_file_path()}")

    # Conversation 1: Create memory
    run_conversation(
        bedrock_runtime,
        model_id,
        "Hi! My name is Nicolai. I'm a software engineer who loves Python and AWS. "
        "I prefer clean, maintainable code over clever solutions. "
        "I use uv for Python package management. "
        "Please remember these preferences.",
        "Conversation 1: Create Memory",
    )

    print("\n" + "-" * 70)
    print("Memory file after creation:")
    print("-" * 70)
    memory_file = get_memory_file_path()
    if memory_file.exists():
        print(memory_file.read_text())
    else:
        print("No memory file found")
    print("-" * 70)

    # Conversation 2: Retrieve memory
    print("\n" + "=" * 70)
    print("NEW SESSION (no context from previous conversation)")
    print("=" * 70)

    run_conversation(
        bedrock_runtime,
        model_id,
        "Hi! Can you remind me what package manager I prefer to use?",
        "Conversation 2: Retrieve Memory",
    )

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("\nClaude Memory Tool Test on AWS Bedrock")
    print("Testing: Create and retrieve memory across conversation sessions\n")
    main()
