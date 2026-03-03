#!/usr/bin/env python3
# ABOUTME: Tests the bash_20250124 tool on Amazon Bedrock via the Converse API.
# ABOUTME: Detects when Bedrock enables the bash tool on Converse.

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

TOOL_TYPE = "bash_20250124"


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "List all Python files in the current directory."}
                ],
            }
        ],
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
            "tools": [
                {
                    "type": TOOL_TYPE,
                    "name": "bash",
                }
            ]
        },
    }

    print(f"--- REQUEST [{TOOL_TYPE}] ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print(f"\n--- RESPONSE [{TOOL_TYPE}] ---")
        print(json.dumps(response, indent=2, default=str))
    except Exception as e:
        print(f"\n--- ERROR [{TOOL_TYPE}] ---")
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
