#!/usr/bin/env python3

import json
import os
import sys

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

config = load_config()
client = get_bedrock_client(config)

response = client.invoke_model(
    modelId=config["bedrock_model_id"],
    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "thinking": {
                "type": "adaptive",
            },
            "output_config": {
                "effort": "low",
            },
            "messages": [
                {
                    "role": "user",
                    "content": "How many socks must you blindly grab from a drawer containing 10 pairs of green, 10 pairs of black, and 11 pairs of blue socks to guarantee a matching pair? Answer with a number first, then explain.",
                }
            ],
        }
    ),
)

body = json.loads(response["body"].read())
print(body)
print(f"Model: {body.get('model')}")
print(f"Stop:  {body.get('stop_reason')}")
print(f"Usage: {json.dumps(body.get('usage', {}))}")

for block in body.get("content", []):
    if block["type"] == "thinking":
        print(f"\n[THINKING] ({len(block['thinking'])} chars)")
        print(block["thinking"])
    elif block["type"] == "text":
        print(f"\n[RESPONSE]")
        print(block["text"])
