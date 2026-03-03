#!/usr/bin/env python3
# ABOUTME: Tests the text_editor_20250728 tool on Amazon Bedrock via the invoke_model API.
# ABOUTME: Detects when Bedrock enables the text editor tool on invoke_model.

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

TOOL_TYPE = "text_editor_20250728"
TOOL_NAME = "str_replace_based_edit_tool"


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "There's a syntax error in my primes.py file. Can you help me fix it?",
            }
        ],
        "tools": [
            {
                "type": TOOL_TYPE,
                "name": TOOL_NAME,
            }
        ],
    }

    print(f"--- REQUEST [{TOOL_TYPE}] ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print(f"\n--- RESPONSE [{TOOL_TYPE}] ---")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n--- ERROR [{TOOL_TYPE}] ---")
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
