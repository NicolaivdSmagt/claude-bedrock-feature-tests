#!/usr/bin/env python3
# ABOUTME: Tests the web_search tool on Amazon Bedrock via the invoke_model API.
# ABOUTME: Tries both web_search_20250305 and web_search_20260209 tool versions.

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

TOOL_VERSIONS = ["web_search_20250305", "web_search_20260209"]


def test_web_search(client, model_id, tool_version):
    """Send a single invoke_model request with the given web_search tool version."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Who won the 2025 F1 grand prix in Las Vegas, and what happened to the driver that finished second?",
            }
        ],
        "tools": [
            {
                "type": tool_version,
                "name": "web_search",
                "max_uses": 1,
            }
        ],
    }

    print(f"--- REQUEST [{tool_version}] ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print(f"\n--- RESPONSE [{tool_version}] ---")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n--- ERROR [{tool_version}] ---")
        print(f"{type(e).__name__}: {e}")

    print()


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    for tool_version in TOOL_VERSIONS:
        test_web_search(client, model_id, tool_version)


if __name__ == "__main__":
    main()
