# ABOUTME: Tests custom client-side tool search on Amazon Bedrock.
# ABOUTME: Demonstrates the tool_search_tool_result format for returning tool references.

"""
Custom Tool Search Demo for Amazon Bedrock
==========================================

This script tests custom client-side tool search on Bedrock.
You implement the search logic, return tool_reference blocks via tool_search_tool_result,
and the API expands them for Claude.

Requirements:
    uv add boto3

Usage:
    uv run python tool_search_custom_bedrock.py
"""

import json
import os
import sys
import boto3

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

_cfg = load_config()
REGION = _cfg["region"]
MODEL_ID = _cfg["bedrock_model_id"]
TOOL_SEARCH_BETA = "tool-search-tool-2025-10-19"


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              CUSTOM TOOL SEARCH DEMO - BEDROCK                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    client = boto3.client("bedrock-runtime", region_name=REGION)

    # Step 1: Initial request - Claude should call our search tool
    print("\n" + "=" * 70)
    print("STEP 1: INITIAL REQUEST")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": "You have access to many tools. Use semantic_tool_search to find relevant tools before attempting to answer questions.",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather forecast in Seattle for the next 3 days?",
            }
        ],
        "tools": [
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
        ],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    result = json.loads(response["body"].read())

    print("\n--- RAW RESPONSE ---")
    print(json.dumps(result, indent=2))

    # Extract the tool call
    tool_call = next((b for b in result["content"] if b["type"] == "tool_use"), None)
    if not tool_call:
        print("\n❌ ERROR: No tool call in response")
        return

    print(f"\n✅ Claude called: {tool_call['name']}")
    print(f"   Query: {tool_call['input'].get('query', 'N/A')}")

    # Step 2: Return tool references via a regular tool_result with tool_reference content blocks.
    # Note: tool_search_tool_result is only for server-side search (srvtoolu_ IDs).
    # For custom client-side search, use a plain tool_result with tool_reference blocks.
    print("\n" + "=" * 70)
    print("STEP 2: RETURN TOOL REFERENCES")
    print("=" * 70)

    tool_result_block = {
        "type": "tool_result",
        "tool_use_id": tool_call["id"],
        "content": [{"type": "tool_reference", "tool_name": "get_weather"}],
    }

    print("\n--- TOOL RESULT BLOCK ---")
    print(json.dumps(tool_result_block, indent=2))

    # Step 3: Continue conversation with tool result
    print("\n" + "=" * 70)
    print("STEP 3: CONTINUE WITH TOOL REFERENCES")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [TOOL_SEARCH_BETA],
        "max_tokens": 1024,
        "system": "You have access to many tools. Use semantic_tool_search to find relevant tools before attempting to answer questions.",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather forecast in Seattle for the next 3 days?",
            },
            {"role": "assistant", "content": result["content"]},
            {"role": "user", "content": [tool_result_block]},
        ],
        "tools": [
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
        ],
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        result = json.loads(response["body"].read())

        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        # Check if Claude now calls get_weather
        tool_call = next(
            (b for b in result["content"] if b["type"] == "tool_use"), None
        )
        if tool_call:
            print(f"\n✅ Claude called: {tool_call['name']}")
            print(f"   Input: {json.dumps(tool_call['input'])}")
        else:
            print("\n⚠️ No tool call in final response")

    except Exception as e:
        print("\n--- BEDROCK ERROR ---")
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
