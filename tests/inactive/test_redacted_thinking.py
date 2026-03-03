#!/usr/bin/env python3
# ABOUTME: Tests redacted_thinking feature on Amazon Bedrock using invoke_model API.
# ABOUTME: Sends a magic string that triggers redacted thinking and validates the response.

import json
import sys
import time
import os
from typing import Optional
from datetime import datetime

try:
    import boto3
except ImportError:
    print("Error: boto3 package not installed. Run: uv add boto3")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# Magic string to trigger redacted thinking
MAGIC_STRING = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"


class TestResult:
    """Store results from a single test"""

    def __init__(self, api_name: str):
        self.api_name = api_name
        self.success = False
        self.error: Optional[str] = None
        self.has_redacted_thinking = False
        self.has_text_response = False
        self.redacted_data_length = 0
        self.text_content = ""
        self.response_time_ms = 0
        self.raw_content_blocks: list = []

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"{self.api_name}: {status}"


def test_bedrock_api(client, model_id, region) -> TestResult:
    """Test redacted thinking on AWS Bedrock"""
    result = TestResult("Bedrock API")

    try:
        # Prepare request body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": MAGIC_STRING}],
            "max_tokens": 2000,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }

        # Make request
        start_time = time.time()

        response = client.invoke_model(modelId=model_id, body=json.dumps(body))

        result.response_time_ms = int((time.time() - start_time) * 1000)

        # Parse response
        response_body = json.loads(response["body"].read())

        # Analyze response
        result.raw_content_blocks = response_body.get("content", [])

        for block in response_body.get("content", []):
            if block.get("type") == "redacted_thinking":
                result.has_redacted_thinking = True
                result.redacted_data_length = len(block.get("data", ""))
            elif block.get("type") == "text":
                result.has_text_response = True
                result.text_content += block.get("text", "")

        result.success = True

    except Exception as e:
        result.error = str(e)

    return result


def print_result_details(result: TestResult):
    """Print detailed results for a single test"""
    print(f"\n{result.api_name}")
    print("-" * 80)

    if not result.success:
        print(f"Status: FAILED")
        print(f"Error: {result.error}")
        return

    print(f"Status: SUCCESS")
    print(f"Response Time: {result.response_time_ms}ms")
    print(f"\nContent Blocks Found:")
    print(f"  - Has redacted_thinking: {result.has_redacted_thinking}")
    print(f"  - Has text response: {result.has_text_response}")

    if result.has_redacted_thinking:
        print(f"\nRedacted Thinking Details:")
        print(f"  - Data Length: {result.redacted_data_length} characters")

    if result.has_text_response:
        print(f"\nText Response:")
        print(f"  Length: {len(result.text_content)} characters")
        if len(result.text_content) > 200:
            print(f"  Preview: {result.text_content[:200]}...")
        else:
            print(f"  Content: {result.text_content}")

    print(f"\nRaw Content Blocks Structure:")
    for i, block in enumerate(result.raw_content_blocks):
        block_type = block.get("type", "unknown")
        print(f"  [{i}] Type: {block_type}")

    print(f"\nRaw Response Output:")
    print("-" * 80)
    for block in result.raw_content_blocks:
        print(json.dumps(block, indent=2))
    print("-" * 80)


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    region = config["region"]

    print("\n" + "=" * 80)
    print("  REDACTED THINKING TEST - Bedrock")
    print("=" * 80)
    print(f"\nTest Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Magic String: {MAGIC_STRING[:50]}...")
    print(f"Bedrock Model: {model_id}")
    print(f"Bedrock Region: {region}")
    print("\n" + "-" * 80)

    client = get_bedrock_client(config)

    print("\nTesting Bedrock API...")
    result = test_bedrock_api(client, model_id, region)
    print(f"      {result}")

    print("\n" + "=" * 80)
    print("  DETAILED RESULTS")
    print("=" * 80)
    print_result_details(result)

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    if result.success and result.has_redacted_thinking:
        print("\nALL TESTS PASSED")
        print("  - Bedrock API returned successful response")
        print("  - Redacted thinking block was present")
    elif result.success:
        print("\nTESTS COMPLETED WITH WARNINGS")
        print("  - Bedrock API returned successful response")
        print("  - No redacted_thinking block found")
    else:
        print("\nTESTS FAILED")
        print(f"  - Bedrock API: {result.error}")
    print("=" * 80 + "\n")

    success = result.success and result.has_redacted_thinking
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
