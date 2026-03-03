# ABOUTME: Demonstrates the Tool Search Tool on the Anthropic API with Claude.
# ABOUTME: Shows server-side regex and BM25 tool search patterns.

"""
Tool Search Tool Demo for Anthropic API
=======================================

This script demonstrates how to use the Tool Search Tool to efficiently work with
large tool catalogs on the Anthropic API.

Key Concepts:
- Regex-based server-side tool search: Claude constructs regex patterns
- BM25-based server-side tool search: Claude uses natural language queries
- Tools with defer_loading: true are discovered via search
- Tools without defer_loading are always visible to Claude

Requirements:
    uv add anthropic

Usage:
    uv run python tool_search_demo_ant.py
"""

import json
import os
import sys

import anthropic

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_anthropic_api_key

TOOL_SEARCH_BETA = "advanced-tool-use-2025-11-20"


def demo_regex_search(api_key, model_id):
    """Demo 1: Server-side regex tool search."""
    print("\n" + "=" * 70)
    print("DEMO 1: SERVER-SIDE REGEX SEARCH")
    print("=" * 70)

    client = anthropic.Anthropic(api_key=api_key)

    request_params = {
        "model": model_id,
        "betas": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": "You have access to many tools. Use tool search to find them.",
        "messages": [{"role": "user", "content": "What's the weather in Seattle?"}],
        "tools": [
            {
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search_tool_regex",
            },
            {
                "name": "get_weather",
                "description": "Get current weather for a location.",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "defer_loading": True,
            },
            {
                "name": "send_message",
                "description": "Send a message to a Slack channel.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["channel", "message"],
                },
                "defer_loading": True,
            },
            {
                "name": "search_documents",
                "description": "Search through documents and files.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ],
    }

    print("\n--- REQUEST PARAMS ---")
    print(json.dumps(request_params, indent=2))

    response = client.beta.messages.create(**request_params)

    print("\n--- RAW RESPONSE ---")
    print(json.dumps(response.model_dump(), indent=2))


def demo_bm25_search(api_key, model_id):
    """Demo 2: Server-side BM25 search."""
    print("\n" + "=" * 70)
    print("DEMO 2: SERVER-SIDE BM25 SEARCH")
    print("=" * 70)
    print("\nBM25 uses natural language queries instead of regex patterns.")

    client = anthropic.Anthropic(api_key=api_key)

    request_params = {
        "model": model_id,
        "betas": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": "You have access to many tools. Use tool search to find them.",
        "messages": [{"role": "user", "content": "What's the weather in Seattle?"}],
        "tools": [
            {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
            {
                "name": "search_documents",
                "description": "Search through documents and files.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_weather",
                "description": "Get current weather conditions, temperature, humidity, and forecast for a city or location. Returns real-time meteorological data.",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "defer_loading": True,
            },
            {
                "name": "send_message",
                "description": "Send a message to a Slack channel.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["channel", "message"],
                },
                "defer_loading": True,
            },
        ],
    }

    print("\n--- REQUEST PARAMS ---")
    print(json.dumps(request_params, indent=2))

    response = client.beta.messages.create(**request_params)

    print("\n--- RAW RESPONSE ---")
    print(json.dumps(response.model_dump(), indent=2))


def demo_custom_search(api_key, model_id):
    """Demo 3: Custom client-side tool search."""
    print("\n" + "=" * 70)
    print("DEMO 3: CUSTOM CLIENT-SIDE TOOL SEARCH")
    print("=" * 70)
    print("\nYou implement search, return tool_reference blocks, API expands them.")

    client = anthropic.Anthropic(api_key=api_key)

    # Step 1: Claude calls our custom search tool
    print("\n--- STEP 1: INITIAL REQUEST ---")

    request_params = {
        "model": model_id,
        "betas": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": "Use my_search_tool to find tools.",
        "messages": [{"role": "user", "content": "What's the weather in Seattle?"}],
        "tools": [
            {
                "name": "my_search_tool",
                "description": "Search for tools by keyword.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_weather",
                "description": "Get current weather for a location.",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "defer_loading": True,
            },
            {
                "name": "send_message",
                "description": "Send a message to a Slack channel.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["channel", "message"],
                },
                "defer_loading": True,
            },
        ],
    }

    print("\n--- REQUEST PARAMS ---")
    print(json.dumps(request_params, indent=2))

    response = client.beta.messages.create(**request_params)
    result = response.model_dump()

    print("\n--- RAW RESPONSE ---")
    print(json.dumps(result, indent=2))

    # Extract search tool call
    tool_call = next(b for b in result["content"] if b["type"] == "tool_use")

    # Step 2: Return tool_reference blocks in a regular tool_result
    print("\n--- STEP 2: RETURN TOOL_REFERENCES ---")

    tool_references = [{"type": "tool_reference", "tool_name": "get_weather"}]

    print("Returning:")
    print(json.dumps(tool_references, indent=2))

    # Step 3: Continue with tool_references in tool_result
    print("\n--- STEP 3: CONTINUE ---")

    request_params = {
        "model": model_id,
        "betas": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather in Seattle?"},
            {"role": "assistant", "content": result["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": tool_references,
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "my_search_tool",
                "description": "Search for tools by keyword.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_weather",
                "description": "Get current weather for a location.",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "defer_loading": True,
            },
            {
                "name": "send_message",
                "description": "Send a message to a Slack channel.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["channel", "message"],
                },
                "defer_loading": True,
            },
        ],
    }

    print("\n--- REQUEST PARAMS ---")
    print(json.dumps(request_params, indent=2))

    try:
        response = client.beta.messages.create(**request_params)
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(response.model_dump(), indent=2))
    except Exception as e:
        print("\n--- API ERROR ---")
        print(f"{type(e).__name__}: {e}")


def main():
    config = load_config()
    model_id = config["anthropic_model_id"]

    print("""
========================================================================
  TOOL SEARCH TOOL DEMO - Anthropic API
========================================================================
    """)
    print(f"Model: {model_id}")

    print("Fetching API key from AWS Secrets Manager...")
    api_key = get_anthropic_api_key(config)
    print("API key retrieved successfully.\n")

    demo_regex_search(api_key, model_id)
    demo_bm25_search(api_key, model_id)
    demo_custom_search(api_key, model_id)


if __name__ == "__main__":
    main()
