#!/usr/bin/env python3
# ABOUTME: Tests tool search on Amazon Bedrock via the Converse API.
# ABOUTME: Covers server-side regex, server-side BM25, and custom client-side tool search.

"""
Tool Search Tests for Amazon Bedrock (Converse API)
====================================================

Tests three tool search patterns via the Converse API:

1. Regex search  – server-side regex-based tool discovery
2. BM25 search   – server-side BM25-based tool discovery
3. Custom search – client-side search with tool_reference responses

Tool search tools and beta headers are passed via additionalModelRequestFields
since the Converse API doesn't natively support these Anthropic-specific types.
A placeholder toolSpec is required in toolConfig.tools to satisfy the Converse
schema.

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/tool_search_converse.py
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

TOOL_SEARCH_BETA = "tool-search-tool-2025-10-19"

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


def test_regex_search(client, model_id):
    """Server-side regex tool search via Converse. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 1: SERVER-SIDE REGEX SEARCH")
    print("=" * 70)

    request = {
        "modelId": model_id,
        "system": [
            {"text": "You have access to many tools. Use tool search to find them."}
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"text": "What's the weather in Seattle?"}],
            }
        ],
        "toolConfig": PLACEHOLDER_TOOL_CONFIG,
        "additionalModelRequestFields": {
            "anthropic_beta": [TOOL_SEARCH_BETA],
            "tools": [
                # The search tool itself
                {"type": "tool_search_tool_regex", "name": "tool_search_tool_regex"},
                # Deferred tools - discovered via search
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
                # Non-deferred tool - always visible to Claude
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
        },
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        stop_reason = response.get("stopReason")
        if stop_reason == "tool_use":
            print("\n  Result: PASS - got tool_use stopReason")
            return ("PASS", None)
        else:
            msg = f"Expected stopReason=tool_use, got {stop_reason!r}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print("\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    finally:
        print()


def test_bm25_search(client, model_id):
    """Server-side BM25 tool search via Converse. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 2: SERVER-SIDE BM25 SEARCH")
    print("=" * 70)

    request = {
        "modelId": model_id,
        "system": [
            {"text": "You have access to many tools. Use tool search to find them."}
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"text": "What's the weather in Seattle?"}],
            }
        ],
        "toolConfig": PLACEHOLDER_TOOL_CONFIG,
        "additionalModelRequestFields": {
            "anthropic_beta": [TOOL_SEARCH_BETA],
            "tools": [
                # BM25 search tool - not currently supported on Bedrock
                {"type": "tool_search_tool_bm25", "name": "tool_search_tool_bm25"},
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
        },
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        stop_reason = response.get("stopReason")
        if stop_reason == "tool_use":
            print("\n  Result: PASS - got tool_use stopReason")
            return ("PASS", None)
        else:
            msg = f"Expected stopReason=tool_use, got {stop_reason!r}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print("\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    finally:
        print()


def test_custom_search(client, model_id):
    """Custom client-side tool search with tool_reference responses via Converse.
    Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 3: CUSTOM CLIENT-SIDE TOOL SEARCH")
    print("=" * 70)

    # Step 1: Initial request - Claude should call our search tool
    print("\n--- STEP 1: INITIAL REQUEST ---")

    anthropic_tools = [
        {
            "name": "semantic_tool_search",
            "description": "Search for available tools by describing what you need. Returns tools that match your search query.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the tool you're looking for",
                    }
                },
                "required": ["query"],
            },
            "defer_loading": False,
        },
        {
            "name": "get_weather",
            "description": "Get current weather conditions, temperature, humidity, and forecast for a city or location. Returns real-time meteorological data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["location"],
            },
            "defer_loading": True,
        },
        {
            "name": "search_flights",
            "description": "Search for available flights between airports.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
            "defer_loading": True,
        },
    ]

    system_text = "You have access to many tools. Use semantic_tool_search to find relevant tools before attempting to answer questions."
    user_text = "What's the weather forecast in Seattle for the next 3 days?"

    request = {
        "modelId": model_id,
        "system": [{"text": system_text}],
        "messages": [
            {"role": "user", "content": [{"text": user_text}]},
        ],
        "toolConfig": PLACEHOLDER_TOOL_CONFIG,
        "additionalModelRequestFields": {
            "anthropic_beta": [TOOL_SEARCH_BETA],
            "tools": anthropic_tools,
        },
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print("\n--- BEDROCK ERROR (step 1) ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    print("\n--- RESPONSE ---")
    print(json.dumps(response, indent=2, default=str))

    # Extract tool call from Converse response
    output_message = response.get("output", {}).get("message", {})
    content_blocks = output_message.get("content", [])
    tool_call = next((b["toolUse"] for b in content_blocks if "toolUse" in b), None)

    if not tool_call:
        msg = "No tool call in step 1 response"
        print(f"\n  Result: FAIL - {msg}")
        print()
        return ("FAIL", msg)

    print(f"\nClaude called: {tool_call['name']}")
    print(f"   Query: {tool_call['input'].get('query', 'N/A')}")

    # Step 2: Return tool references via tool_result with tool_reference content blocks.
    # For custom client-side search, use a plain tool_result with tool_reference blocks
    # passed through additionalModelRequestFields since Converse doesn't natively support
    # the tool_reference content type.
    print("\n--- STEP 2: RETURN TOOL REFERENCES ---")

    # Build the assistant message in Converse format for the multi-turn conversation
    assistant_content = content_blocks

    # The tool_reference block goes through the Anthropic-native message format
    # in additionalModelRequestFields, while the Converse-native messages carry
    # a standard toolResult.
    tool_result_content = [
        {
            "toolResult": {
                "toolUseId": tool_call["toolUseId"],
                "content": [{"text": "Found matching tool: get_weather"}],
            }
        }
    ]

    print("\n--- TOOL RESULT (Converse format) ---")
    print(json.dumps(tool_result_content, indent=2))

    # The tool_reference block is passed in additionalModelRequestFields
    # to tell the API to expand the deferred tool for Claude
    anthropic_tool_result = {
        "type": "tool_result",
        "tool_use_id": tool_call["toolUseId"],
        "content": [{"type": "tool_reference", "tool_name": "get_weather"}],
    }

    print("\n--- TOOL REFERENCE (additionalModelRequestFields) ---")
    print(json.dumps(anthropic_tool_result, indent=2))

    # Step 3: Continue conversation with tool result
    print("\n--- STEP 3: CONTINUE WITH TOOL REFERENCES ---")

    request = {
        "modelId": model_id,
        "system": [{"text": system_text}],
        "messages": [
            {"role": "user", "content": [{"text": user_text}]},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_result_content},
        ],
        "toolConfig": PLACEHOLDER_TOOL_CONFIG,
        "additionalModelRequestFields": {
            "anthropic_beta": [TOOL_SEARCH_BETA],
            "tools": anthropic_tools,
        },
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)

        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        # Check if Claude now calls get_weather
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        tool_call = next((b["toolUse"] for b in content_blocks if "toolUse" in b), None)
        if tool_call:
            print(f"\nClaude called: {tool_call['name']}")
            print(f"   Input: {json.dumps(tool_call['input'])}")
            if tool_call["name"] == "get_weather":
                print(
                    "\n  Result: PASS - Claude called get_weather after tool reference"
                )
                return ("PASS", None)
            else:
                msg = f"Expected get_weather call, got {tool_call['name']!r}"
                print(f"\n  Result: FAIL - {msg}")
                return ("FAIL", msg)
        else:
            msg = "No tool call in step 3 response after providing tool reference"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print("\n--- BEDROCK ERROR (step 3) ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    finally:
        print()


def print_summary(results):
    """Print a summary table of all test outcomes."""
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Find the longest test name for alignment
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

    results = []

    status, error = test_regex_search(client, model_id)
    results.append(("Regex search (server-side)", status, error))

    status, error = test_bm25_search(client, model_id)
    results.append(("BM25 search (server-side)", status, error))

    status, error = test_custom_search(client, model_id)
    results.append(("Custom search (client-side)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
