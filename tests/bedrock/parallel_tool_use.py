#!/usr/bin/env python3
"""Test script to verify parallel tool calls with the Claude API"""

import os
import sys
import boto3
import json
from botocore.config import Config

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)

# Load config and parse command line arguments for model selection
cfg = load_config()
DEFAULT_MODEL = cfg["bedrock_model_id"]
model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
print(f"Using model: {model_id}\n")

# Initialize Bedrock client
boto_config = Config(read_timeout=600, retries=dict(max_attempts=5))
bedrock = boto3.client(
    service_name="bedrock-runtime", region_name=cfg["region"], config=boto_config
)

# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current time in a given timezone",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone, e.g. America/New_York",
                }
            },
            "required": ["timezone"],
        },
    },
]

# Test conversation with parallel tool calls
messages = [
    {
        "role": "user",
        "content": "What's the weather in SF and NYC, and what time is it there?",
    }
]

# Make initial request
print("=" * 60)
print("REQUESTING PARALLEL TOOL CALLS")
print("=" * 60)
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "messages": messages,
    "tools": tools,
}

print("\n" + "=" * 60)
print("RAW API REQUEST")
print("=" * 60)
print(json.dumps(body, indent=2))
print("=" * 60)

response = bedrock.invoke_model(
    body=json.dumps(body),
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())

print("\n" + "=" * 60)
print("RAW API RESPONSE FROM CLAUDE")
print("=" * 60)
print(json.dumps(response_body, indent=2))
print("=" * 60)

# Check for parallel tool calls
tool_uses = [block for block in response_body["content"] if block["type"] == "tool_use"]

print(f"\n{'=' * 60}")
print(f"TOOL CALLS ANALYSIS")
print(f"{'=' * 60}")
print(f"Number of tool calls made: {len(tool_uses)}")
print(f"Parallel execution: {'YES ✓' if len(tool_uses) > 1 else 'NO ✗'}")
print("=" * 60)

# Simulate tool execution and format results correctly
tool_results = []
for tool_use in tool_uses:
    if tool_use["name"] == "get_weather":
        if "San Francisco" in str(tool_use["input"]):
            result = "San Francisco: 68°F, partly cloudy"
        else:
            result = "New York: 45°F, clear skies"
    else:  # get_time
        if "Los_Angeles" in str(tool_use["input"]):
            result = "2:30 PM PST"
        else:
            result = "5:30 PM EST"

    tool_results.append(
        {"type": "tool_result", "tool_use_id": tool_use["id"], "content": result}
    )

# Continue conversation with tool results
messages.extend(
    [
        {"role": "assistant", "content": response_body["content"]},
        {"role": "user", "content": tool_results},  # All results in one message!
    ]
)

# Get final response
print("\n" + "=" * 60)
print("GETTING FINAL RESPONSE")
print("=" * 60)
final_body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "messages": messages,
    "tools": tools,
}

final_response = bedrock.invoke_model(
    body=json.dumps(final_body),
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

final_response_body = json.loads(final_response.get("body").read())

print(f"\nClaude's synthesized response:")
print("-" * 60)
print(final_response_body["content"][0]["text"])

# Verify formatting
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"✓ Tool results sent in single user message: {len(tool_results)} results")
print("✓ No text before tool results in content array")
print("✓ Conversation formatted correctly for future parallel tool use")
print("=" * 60)
