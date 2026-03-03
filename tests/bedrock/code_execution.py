#!/usr/bin/env python3
# ABOUTME: Tests the code_execution_20250825 tool on Amazon Bedrock (invoke_model API).
# ABOUTME: Validates programmatic tool calling with and without beta headers.

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


def get_tools_config_bedrock():
    """
    Return the tools configuration for Bedrock's invoke_model API.

    Bedrock uses a slightly different schema for tools compared to
    the direct Anthropic API.
    """
    return [
        {"type": "code_execution_20250825", "name": "code_execution"},
        {
            "name": "get_temperature",
            "description": "Get the current temperature for a city.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"}
                },
                "required": ["city"],
            },
            "allowed_callers": ["code_execution_20250825"],
        },
    ]


def test_bedrock_with_beta(client, model_id):
    """Test code execution tool on Bedrock with beta header."""
    print("\n" + "=" * 70)
    print("TEST 1: BEDROCK - code_execution_20250825 tool (with beta header)")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["advanced-tool-use-2025-11-20"],
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Get the temperature for Paris and London, then tell me which is warmer.",
            }
        ],
        "tools": get_tools_config_bedrock(),
    }

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())

        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        # Check for code execution or tool use blocks
        content = result.get("content", [])
        for block in content:
            block_type = block.get("type")
            print(f"\nContent block type: {block_type}")
            if block_type == "server_tool_use":
                print("   Server tool use detected - code execution is working!")
            elif block_type == "tool_use":
                caller = block.get("caller", {})
                print(f"   Tool name: {block.get('name')}")
                print(f"   Caller: {caller}")
                if caller.get("type") == "code_execution_20250825":
                    print("   Programmatic tool calling is working!")

        return True, result

    except client.exceptions.ValidationException as e:
        print(f"\nBEDROCK VALIDATION ERROR:")
        print(f"   {e}")
        return False, str(e)

    except Exception as e:
        print(f"\nBEDROCK ERROR:")
        print(f"   {type(e).__name__}: {e}")
        return False, str(e)


def test_bedrock_without_beta(client, model_id):
    """Test what happens when we omit the beta header on Bedrock."""
    print("\n" + "=" * 70)
    print("TEST 2: BEDROCK WITHOUT BETA HEADER")
    print("=" * 70)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        # No anthropic_beta header
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Get the temperature for Paris."}],
        "tools": get_tools_config_bedrock(),
    }

    print("\n--- REQUEST BODY (no beta header) ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())

        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        return True, result

    except client.exceptions.ValidationException as e:
        print(f"\nBEDROCK VALIDATION ERROR (expected):")
        print(f"   {e}")
        return False, str(e)

    except Exception as e:
        print(f"\nBEDROCK ERROR:")
        print(f"   {type(e).__name__}: {e}")
        return False, str(e)


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    region = config["region"]

    print("""
========================================================================
  CODE EXECUTION TOOL TEST - BEDROCK
  Testing code_execution_20250825 (programmatic tool calling)
========================================================================
    """)
    print(f"Model: {model_id}")
    print(f"Region: {region}")

    client = get_bedrock_client(config)

    # Test 1: Bedrock with beta header
    bedrock_success, bedrock_result = test_bedrock_with_beta(client, model_id)

    # Test 2: Bedrock without beta header
    bedrock_no_beta_success, bedrock_no_beta_result = test_bedrock_without_beta(
        client, model_id
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\nBedrock (with beta header):    {'SUCCESS' if bedrock_success else 'FAILED'}"
    )
    print(
        f"Bedrock (without beta header): {'SUCCESS' if bedrock_no_beta_success else 'FAILED'}"
    )

    print("\n" + "-" * 70)
    if not bedrock_success:
        print("""
CONCLUSION: The code_execution_20250825 tool does NOT work on Bedrock.

The code execution tool requires Anthropic's backend infrastructure:
- Sandboxed Python execution containers
- Container lifecycle management
- Server-side tool execution (server_tool_use blocks)

These are features of Anthropic's direct API, not part of the model
capabilities that Bedrock licenses.
""")
    else:
        print("""
The code_execution_20250825 tool appears to work on Bedrock!
Please verify the response contains actual server_tool_use blocks.
""")


if __name__ == "__main__":
    main()
