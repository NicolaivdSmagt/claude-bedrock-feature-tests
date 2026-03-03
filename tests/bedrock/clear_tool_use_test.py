#!/usr/bin/env python3
# ABOUTME: Educational demo of Claude Sonnet 4.5's clear_tool_uses feature on AWS Bedrock
# ABOUTME: Shows how context management automatically clears old tool uses to manage token limits

import boto3
import json
import os
import sys

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)


def handle_memory_tool(tool_input, memory_store):
    """Handle Claude's built-in memory tool operations using in-memory storage"""
    command = tool_input.get("command")
    path = tool_input.get("path", "")

    if command == "view":
        if path == "/memories":
            # Return all memories as a list
            memories = [
                {"path": f"/memories/{name}", "content": content}
                for name, content in memory_store.items()
            ]
            return {"memories": memories}
        elif path.startswith("/memories/"):
            # Return specific memory
            filename = path.split("/")[-1]
            if filename in memory_store:
                return {"memory": {"path": path, "content": memory_store[filename]}}
            return {"memory": {"error": "not found"}}

    elif command == "create":
        # Create new memory in dictionary
        filename = path.split("/")[-1] if "/" in path else "memory.md"
        file_text = tool_input.get("file_text", "")
        memory_store[filename] = file_text
        # Return verbose result with detailed information
        return {
            "success": True,
            "operation": "create",
            "created": filename,
            "path": f"/memories/{filename}",
            "size_bytes": len(file_text),
            "size_characters": len(file_text),
            "line_count": len(file_text.split("\n")),
            "word_count": len(file_text.split()),
            "content_preview": file_text[:200] if len(file_text) > 200 else file_text,
            "message": f"Successfully created memory file '{filename}' with {len(file_text)} characters",
            "timestamp": "2025-01-21T12:00:00Z",
            "storage_location": "in-memory dictionary",
            "encoding": "utf-8",
            "status": "persisted",
        }

    elif command == "str_replace":
        # Update existing memory
        filename = path.split("/")[-1]
        if filename in memory_store:
            content = memory_store[filename]
            old_str = tool_input.get("old_str", "")
            new_str = tool_input.get("new_str", "")
            updated = content.replace(old_str, new_str)
            memory_store[filename] = updated
            # Return verbose result with detailed information
            return {
                "success": True,
                "operation": "str_replace",
                "file": filename,
                "path": f"/memories/{filename}",
                "replacements_made": content.count(old_str),
                "old_length": len(content),
                "new_length": len(updated),
                "old_word_count": len(content.split()),
                "new_word_count": len(updated.split()),
                "delta_chars": len(updated) - len(content),
                "replaced_text_preview": old_str[:100]
                if len(old_str) > 100
                else old_str,
                "new_text_preview": new_str[:100] if len(new_str) > 100 else new_str,
                "message": f"Successfully updated '{filename}' with {content.count(old_str)} replacement(s)",
                "timestamp": "2025-01-21T12:00:00Z",
                "status": "updated",
            }
        return {"error": "File not found", "path": path, "attempted_file": filename}

    return {"status": "handled"}


def call_claude(client, model_id, messages, system_prompt, context_management):
    """Make streaming API call to Claude with context management enabled"""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": messages,
        "tools": [{"type": "memory_20250818", "name": "memory"}],
        "anthropic_beta": ["context-management-2025-06-27"],
        "context_management": context_management,
    }

    response = client.invoke_model_with_response_stream(
        modelId=model_id, body=json.dumps(body)
    )

    # Process streaming response
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

    # Parse tool inputs
    for block in content_blocks:
        if block["type"] == "tool_use":
            block["input"] = json.loads(block["input"])

    return content_blocks, usage, context_mgmt


def main():
    """
    Demonstrates Claude's context management feature (clear_tool_uses_20250919).

    As conversations grow, old tool use/result pairs accumulate tokens. Context management
    automatically clears old tool uses when token threshold is reached, keeping recent ones.

    Configuration:
    - trigger: 500 tokens - start clearing when input exceeds this
    - keep: 0 tool uses - clear all old tool uses (keeps none)
    - clear_at_least: 0 tokens - clear regardless of amount saved
    """
    config = load_config()
    client = get_bedrock_client(config)
    model_id = config["bedrock_model_id"]

    # In-memory storage for memories (simulates file system)
    memory_store = {}

    # Context management configuration - lower trigger for faster demo
    context_management = {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {
                    "type": "input_tokens",
                    "value": 500,
                },  # When to start clearing
                "keep": {"type": "tool_uses", "value": 0},  # How many to preserve
                "clear_at_least": {
                    "type": "input_tokens",
                    "value": 0,
                },  # Minimum to clear
            }
        ]
    }

    # System prompt instructs Claude to actively use memory with VERBOSE, DETAILED content
    system_prompt = """IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.
MEMORY PROTOCOL:
1. Store all memories in .md format at /memories/
2. Use the `view` command of your `memory` tool to check for earlier progress.
3. Use memory_{topic}.md format to store general memories based on their topic
4. WRITE EXTREMELY DETAILED, VERBOSE MEMORIES with extensive context, explanations, and examples
5. Include multiple paragraphs, lists, and comprehensive information in each memory
6. Add background context, reasoning, and detailed descriptions for everything
ASSUME INTERRUPTION: Your context window might be reset at any moment, so record progress in memory."""

    # Generic user prompts demonstrating typical memory operations (only 3 turns)
    user_prompts = [
        "Store my tech stack: Python, Go, TypeScript, and Rust.",
        "Add my preferred tools: VSCode, Docker, Git, and Kubernetes.",
        "Remember my cloud platforms: AWS, GCP, and Azure.",
    ]

    messages = []
    total_cleared = 0
    total_saved = 0

    print("\n" + "=" * 70)
    print("Context Management Demo: clear_tool_uses_20250919")
    print("=" * 70)
    print(
        f"Config: Trigger={context_management['edits'][0]['trigger']['value']} tokens | "
        f"Keep={context_management['edits'][0]['keep']['value']} tools | "
        f"Clear≥{context_management['edits'][0]['clear_at_least']['value']} tokens\n"
    )

    for turn, prompt in enumerate(user_prompts, 1):
        print(f"Turn {turn}: {prompt[:60]}...")

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Call Claude with streaming
        content_blocks, usage, context_mgmt = call_claude(
            client, model_id, messages, system_prompt, context_management
        )

        # Add assistant response
        messages.append({"role": "assistant", "content": content_blocks})

        # Display token usage and clearing stats
        tokens = usage.get("input_tokens", 0)
        tool_count = sum(
            1
            for m in messages
            if m["role"] == "assistant"
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        )
        print(f"  Input: {tokens} tokens | Total tool uses: {tool_count}", end="")

        # Check if any tool uses were cleared and show API response details
        if context_mgmt:
            print("\n  📋 API Response - context_management field:")
            print(f"     {json.dumps(context_mgmt, indent=6)}")

            edits = context_mgmt.get("applied_edits", [])
            for edit in edits:
                if edit.get("type") == "clear_tool_uses_20250919":
                    cleared = edit.get("cleared_tool_uses", 0)
                    saved = edit.get("cleared_input_tokens", 0)
                    if cleared > 0:
                        total_cleared += cleared
                        total_saved += saved
                        print(
                            f"  ✅ CLEARED {cleared} tool uses, saved {saved} tokens!"
                        )
        print()

        # Execute tool uses
        tool_uses = [b for b in content_blocks if b["type"] == "tool_use"]
        if tool_uses:
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

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Conversation turns: {len(user_prompts)}")
    print(f"Tool uses cleared: {total_cleared}")
    print(f"Tokens saved: {total_saved}")
    print(f"Memories stored: {len(memory_store)}")

    # Show stored memories
    if memory_store:
        print(f"\n📝 In-Memory Storage Contents:")
        for name, content in memory_store.items():
            print(f"   • {name}: {len(content)} chars")

    print(
        f"\nKey Insight: Context management cleared {total_cleared} old tool use/result"
    )
    print(
        f"pairs, saving {total_saved} tokens. This keeps conversations efficient while"
    )
    print(f"preserving the actual memory data in storage for future retrieval.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
