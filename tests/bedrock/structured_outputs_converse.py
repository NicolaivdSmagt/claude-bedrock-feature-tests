#!/usr/bin/env python3
# ABOUTME: Tests structured outputs on Amazon Bedrock via the Converse API.
# ABOUTME: Validates JSON schema output and strict tool use via additionalModelRequestFields.

"""
Structured Outputs Test for Amazon Bedrock (Converse API)
==========================================================

Tests two structured output mechanisms via the Converse API:

1. JSON schema output — uses output_config.format (passed via
   additionalModelRequestFields) with a json_schema to constrain the
   model's response to match a specific JSON structure.
2. Strict tool use — uses strict: true on a tool definition (passed via
   additionalModelRequestFields) to guarantee schema-compliant tool inputs.

Both features are Anthropic-specific and must be passed through
additionalModelRequestFields since the Converse API doesn't natively
support them. A placeholder toolSpec is required in toolConfig.tools for
the strict tool use test.

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/structured_outputs_converse.py
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


def test_json_schema_output(client, model_id):
    """Test JSON schema output via output_config.format. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 1: JSON SCHEMA OUTPUT")
    print("=" * 70)

    output_schema = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "plan_interest": {"type": "string"},
                "demo_requested": {"type": "boolean"},
            },
            "required": ["name", "email", "plan_interest", "demo_requested"],
            "additionalProperties": False,
        },
    }

    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Extract the key information from this email: "
                            "John Smith (john@example.com) is interested in our Enterprise plan "
                            "and wants to schedule a demo for next Tuesday at 2pm."
                        )
                    }
                ],
            }
        ],
        "additionalModelRequestFields": {
            "output_config": {
                "format": output_schema,
            },
        },
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        stop_reason = response.get("stopReason")
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        print(f"\n  stopReason: {stop_reason}")

        # Find the text block
        text_block = next((b for b in content_blocks if "text" in b), None)
        if not text_block:
            msg = "No text block in response"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        text = text_block["text"]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            msg = f"Response is not valid JSON: {text[:200]}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        print(f"  Parsed JSON: {json.dumps(parsed, indent=2)}")

        # Validate all required fields are present
        required = ["name", "email", "plan_interest", "demo_requested"]
        missing = [f for f in required if f not in parsed]
        if missing:
            msg = f"Missing fields: {missing}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        # Validate demo_requested is boolean
        if not isinstance(parsed.get("demo_requested"), bool):
            msg = f"demo_requested should be bool, got {type(parsed['demo_requested']).__name__}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        print("\n  Result: PASS - valid JSON output with all required fields")
        return ("PASS", None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    finally:
        print()


def test_strict_tool_use(client, model_id):
    """Test strict tool use with schema validation. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST 2: STRICT TOOL USE")
    print("=" * 70)

    # The strict flag and Anthropic-native tool definition go via
    # additionalModelRequestFields since Converse's toolSpec doesn't
    # support the strict parameter.
    anthropic_tools = [
        {
            "name": "search_flights",
            "description": "Search for available flights",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin airport code (e.g., SFO)",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination airport code (e.g., JFK)",
                    },
                    "departure_date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format",
                    },
                    "passengers": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5, 6],
                        "description": "Number of passengers (1-6)",
                    },
                },
                "required": [
                    "origin",
                    "destination",
                    "departure_date",
                    "passengers",
                ],
                "additionalProperties": False,
            },
        }
    ]

    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Find me a flight from San Francisco to New York for 2 people on February 15, 2025"
                    }
                ],
            }
        ],
        "toolConfig": PLACEHOLDER_TOOL_CONFIG,
        "additionalModelRequestFields": {
            "tools": anthropic_tools,
        },
    }

    print("\n--- REQUEST ---")
    print(json.dumps(request, indent=2, default=str))

    try:
        response = client.converse(**request)
        print("\n--- RESPONSE ---")
        print(json.dumps(response, indent=2, default=str))

        stop_reason = response.get("stopReason")
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        print(f"\n  stopReason: {stop_reason}")

        # Find tool use block
        tool_use_block = next((b for b in content_blocks if "toolUse" in b), None)
        if not tool_use_block:
            msg = "No toolUse block in response"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        tool_use = tool_use_block["toolUse"]
        tool_input = tool_use.get("input", {})
        print(f"  Tool: {tool_use.get('name')}")
        print(f"  Input: {json.dumps(tool_input, indent=2)}")

        # Validate passengers is an integer from the enum
        passengers = tool_input.get("passengers")
        if not isinstance(passengers, int):
            msg = f"passengers should be int, got {type(passengers).__name__}: {passengers}"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        if passengers not in [1, 2, 3, 4, 5, 6]:
            msg = f"passengers {passengers} not in enum [1-6]"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

        print(f"\n  Result: PASS - valid strict tool call with passengers={passengers}")
        return ("PASS", None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    finally:
        print()


def print_summary(results):
    """Print a summary table of all test outcomes."""
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

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

    print(f"Model: {model_id}")
    print()

    results = []

    status, error = test_json_schema_output(client, model_id)
    results.append(("JSON schema output", status, error))

    status, error = test_strict_tool_use(client, model_id)
    results.append(("Strict tool use", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
