#!/usr/bin/env python3
# ABOUTME: Tests tool search on Amazon Bedrock via the invoke_model API.
# ABOUTME: Covers server-side regex, server-side BM25, and custom client-side tool search.

"""
Tool Search Tests for Amazon Bedrock (Invoke API)
==================================================

Tests three tool search patterns via invoke_model:

1. Regex search  – server-side regex-based tool discovery
2. BM25 search   – server-side BM25-based tool discovery
3. Custom search – client-side search with tool_reference responses

Key Concepts:
- Tools with defer_loading: true are invisible until discovered via search
- Tools without defer_loading are always visible to Claude
- Custom search lets you implement your own search logic and return
  tool_reference blocks so the API expands them for Claude

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/tool_search_invoke.py
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


def test_regex_search(client, model_id):
    """Server-side regex tool search. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 1: SERVER-SIDE REGEX SEARCH")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": "You have access to many tools. Use tool search to find them.",
        "messages": [{"role": "user", "content": "What's the weather in Seattle?"}],
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
                # This will cause the tool to be invisible to Claude until it is discovered via search
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
                # This will cause the tool to be invisible to Claude until it is discovered via search
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
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        if result.get("stop_reason") == "tool_use":
            print("\n  Result: PASS - got tool_use stop_reason")
            return ("PASS", None)
        else:
            msg = f"Expected stop_reason=tool_use, got {result.get('stop_reason')!r}"
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
    """Server-side BM25 tool search (natural language queries instead of regex).
    Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 2: SERVER-SIDE BM25 SEARCH")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": "You have access to many tools. Use tool search to find them.",
        "messages": [{"role": "user", "content": "What's the weather in Seattle?"}],
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
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        if result.get("stop_reason") == "tool_use":
            print("\n  Result: PASS - got tool_use stop_reason")
            return ("PASS", None)
        else:
            msg = f"Expected stop_reason=tool_use, got {result.get('stop_reason')!r}"
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
    """Custom client-side tool search with tool_reference responses.
    Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 3: CUSTOM CLIENT-SIDE TOOL SEARCH")
    print("=" * 70)

    # Step 1: Initial request - Claude should call our search tool
    print("\n--- STEP 1: INITIAL REQUEST ---")

    tools = [
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

    system_prompt = "You have access to many tools. Use semantic_tool_search to find relevant tools before attempting to answer questions."
    user_msg = "What's the weather forecast in Seattle for the next 3 days?"

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}],
        "tools": tools,
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print("\n--- BEDROCK ERROR (step 1) ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    print("\n--- RAW RESPONSE ---")
    print(json.dumps(result, indent=2))

    # Extract the tool call
    tool_call = next((b for b in result["content"] if b["type"] == "tool_use"), None)
    if not tool_call:
        msg = "No tool call in step 1 response"
        print(f"\n  Result: FAIL - {msg}")
        print()
        return ("FAIL", msg)

    print(f"\nClaude called: {tool_call['name']}")
    print(f"   Query: {tool_call['input'].get('query', 'N/A')}")

    # Step 2: Return tool references via a regular tool_result with tool_reference content blocks.
    # Note: tool_search_tool_result is only for server-side search (srvtoolu_ IDs).
    # For custom client-side search, use a plain tool_result with tool_reference blocks.
    print("\n--- STEP 2: RETURN TOOL REFERENCES ---")

    tool_result_block = {
        "type": "tool_result",
        "tool_use_id": tool_call["id"],
        "content": [{"type": "tool_reference", "tool_name": "get_weather"}],
    }

    print("\n--- TOOL RESULT BLOCK ---")
    print(json.dumps(tool_result_block, indent=2))

    # Step 3: Continue conversation with tool result
    print("\n--- STEP 3: CONTINUE WITH TOOL REFERENCES ---")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": result["content"]},
            {"role": "user", "content": [tool_result_block]},
        ],
        "tools": tools,
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())

        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        # Check if Claude now calls get_weather
        tool_call = next(
            (b for b in result["content"] if b["type"] == "tool_use"), None
        )
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
