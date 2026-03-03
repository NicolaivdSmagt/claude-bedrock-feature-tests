#!/usr/bin/env python3
# ABOUTME: Tests structured outputs feature on AWS Bedrock with Claude Sonnet 4.6
# ABOUTME: Validates JSON schema outputs (output_config.format) and strict tool use via invoke_model API

import boto3
import json
import os
import sys
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Any, Optional, Tuple

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)

cfg = load_config()
model_id = cfg["bedrock_model_id"]


def print_section(title: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{char * 70}")
    print(title)
    print(f"{char * 70}")


def test_json_schema_output(
    client: Any, model_id: str
) -> Tuple[bool, str, Optional[dict]]:
    """
    Test JSON schema output via output_config.format.

    This is the GA parameter for structured JSON outputs on Bedrock.
    """
    print("\n  Testing JSON schema output via output_config.format...")

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

    messages = [
        {
            "role": "user",
            "content": (
                "Extract the key information from this email: "
                "John Smith (john@example.com) is interested in our Enterprise plan "
                "and wants to schedule a demo for next Tuesday at 2pm."
            ),
        }
    ]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": messages,
        "output_config": {
            "format": output_schema,
        },
    }

    try:
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response.get("body").read())

        content = result.get("content", [])
        print(content)
        if content and content[0].get("type") == "text":
            text = content[0].get("text", "")
            try:
                parsed = json.loads(text)
                # Validate structure
                required = ["name", "email", "plan_interest", "demo_requested"]
                missing = [f for f in required if f not in parsed]
                if missing:
                    return False, f"Missing fields: {missing}", result
                if not isinstance(parsed.get("demo_requested"), bool):
                    return False, "demo_requested not boolean", result
                return True, "Valid JSON output", {"parsed": parsed, "raw": result}
            except json.JSONDecodeError:
                return False, f"Response not valid JSON: {text[:200]}", result
        return False, "Unexpected response structure", result

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        return False, f"{error_code}: {error_msg}", None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None


def test_strict_tool_use(
    client: Any, model_id: str
) -> Tuple[bool, str, Optional[dict]]:
    """
    Test strict tool use validation.

    Uses strict: true on tools to guarantee schema-compliant tool inputs.
    """
    print("\n  Testing strict tool use (strict: true)...")

    tools = [
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
                "required": ["origin", "destination", "departure_date", "passengers"],
                "additionalProperties": False,
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "Find me a flight from San Francisco to New York for 2 people on February 15, 2025",
        }
    ]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": messages,
        "tools": tools,
    }

    try:
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response.get("body").read())

        # Check for tool use
        content = result.get("content", [])
        print(content)
        tool_uses = [c for c in content if c.get("type") == "tool_use"]

        if not tool_uses:
            return False, "No tool use in response", result

        tool_use = tool_uses[0]
        tool_input = tool_use.get("input", {})

        # Validate passengers is integer
        passengers = tool_input.get("passengers")
        if not isinstance(passengers, int):
            return (
                False,
                f"passengers should be int, got {type(passengers).__name__}: {passengers}",
                result,
            )

        if passengers not in [1, 2, 3, 4, 5, 6]:
            return False, f"passengers {passengers} not in enum [1-6]", result

        return True, f"Valid tool call with passengers={passengers}", result

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        return False, f"{error_code}: {error_msg}", None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None


def main():
    """Run structured output tests."""
    # Initialize Bedrock client
    region = cfg["region"]
    boto_config = Config(
        read_timeout=600, retries=dict(max_attempts=3), region_name=region
    )
    client = boto3.client(
        service_name="bedrock-runtime",
        config=boto_config,
    )

    print_section("STRUCTURED OUTPUTS TEST")
    print(f"Model: {model_id}")
    print(f"Region: {region}")
    print(f"\nTesting structured outputs on Bedrock:")
    print("  1. JSON schema output (output_config.format)")
    print("  2. Strict tool use (strict: true)")

    results = {}

    # Test 1: JSON schema output
    print_section("TEST 1: JSON Schema Output", "-")
    print("\nSchema: {name, email, plan_interest, demo_requested}")

    success, msg, data = test_json_schema_output(client, model_id)
    results["json_schema"] = (success, msg)
    print(f"  Result: {'PASS' if success else 'FAIL'} - {msg}")
    if data and isinstance(data, dict) and "parsed" in data:
        print(f"  Parsed output: {json.dumps(data['parsed'], indent=4)}")

    # Test 2: Strict tool use
    print_section("TEST 2: Strict Tool Use", "-")
    print("\nTool: search_flights with passengers as integer enum [1-6]")

    success, msg, data = test_strict_tool_use(client, model_id)
    results["strict_tool"] = (success, msg)
    print(f"  Result: {'PASS' if success else 'FAIL'} - {msg}")

    # Summary
    print_section("SUMMARY")

    s, m = results["json_schema"]
    print(f"  JSON Schema Output: {'PASS' if s else 'FAIL'} - {m}")

    s, m = results["strict_tool"]
    print(f"  Strict Tool Use:    {'PASS' if s else 'FAIL'} - {m}")

    all_passed = all(s for s, _ in results.values())
    print(
        f"\nTotal: {sum(1 for s, _ in results.values() if s)}/{len(results)} tests passed"
    )

    if all_passed:
        print("\nStructured outputs are working on Bedrock!")
    else:
        print("\nSome tests failed.")

    print_section("END OF TEST")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
