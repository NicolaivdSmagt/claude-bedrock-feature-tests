#!/usr/bin/env python3
# ABOUTME: Tests the code_execution_20250825 tool on the Anthropic 1st-party API.
# ABOUTME: Validates programmatic tool calling with the beta header.

import json
import os
import sys

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: uv add anthropic")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_anthropic_client


def test_anthropic_direct(client, model_id):
    """Test code execution tool on direct Anthropic API."""
    print("\n" + "=" * 70)
    print("TEST: ANTHROPIC API - code_execution_20250825 tool")
    print("=" * 70)

    tools = [
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

    print(f"\n--- REQUEST ---")
    print(f"Model: {model_id}")
    print(f"Tools: {json.dumps(tools, indent=2)}")

    try:
        response = client.beta.messages.create(
            model=model_id,
            max_tokens=1024,
            betas=["advanced-tool-use-2025-11-20"],
            messages=[
                {
                    "role": "user",
                    "content": "Get the temperature for Paris and London, then tell me which is warmer.",
                }
            ],
            tools=tools,
        )

        print("\n--- RAW RESPONSE ---")
        print(response.model_dump_json(indent=2))

        # Check for code execution blocks
        for block in response.content:
            print(f"\nContent block type: {block.type}")
            if block.type == "server_tool_use":
                print("   Server tool use detected - code execution is working!")
            elif block.type == "tool_use":
                caller = getattr(block, "caller", None)
                print(f"   Tool name: {block.name}")
                print(f"   Caller: {caller}")
                if (
                    caller
                    and getattr(caller, "type", None) == "code_execution_20250825"
                ):
                    print("   Programmatic tool calling is working!")

        return True, response

    except Exception as e:
        print(f"\nANTHROPIC API ERROR:")
        print(f"   {type(e).__name__}: {e}")
        return False, str(e)


def main():
    config = load_config()
    model_id = config["anthropic_model_id"]

    print("""
========================================================================
  CODE EXECUTION TOOL TEST - ANTHROPIC API
  Testing code_execution_20250825 (programmatic tool calling)
========================================================================
    """)
    print(f"Model: {model_id}")

    print("Fetching API key from AWS Secrets Manager...")
    client = get_anthropic_client(config)
    print("Client ready.\n")

    success, result = test_anthropic_direct(client, model_id)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAnthropic API: {'SUCCESS' if success else 'FAILED'}")

    if success:
        print("\nCode execution tool works on the Anthropic API.")
    else:
        print("\nCode execution tool failed on the Anthropic API.")


if __name__ == "__main__":
    main()
