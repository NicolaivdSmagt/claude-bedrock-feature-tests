#!/usr/bin/env python3
"""
ABOUTME: Script to count tokens in an input string using AWS Bedrock's native token counting.
ABOUTME: Takes input text and model ID as command line arguments.
"""

import boto3
import sys
import os
from botocore.config import Config
from botocore.exceptions import ClientError

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)

_cfg = load_config()


def setup_bedrock_client(region=None):
    """Setup Bedrock client"""
    if region is None:
        region = _cfg["region"]
    config = Config(region_name=region, retries=dict(max_attempts=1000))
    return boto3.client("bedrock-runtime", config=config)


def count_text_tokens(text, model_id):
    """Count tokens in a text string using Bedrock's native count_tokens"""
    bedrock = setup_bedrock_client()

    # Create message structure for token counting
    messages = [{"role": "user", "content": [{"text": text}]}]

    try:
        response = bedrock.count_tokens(
            modelId=model_id, input={"converse": {"messages": messages}}
        )

        input_tokens = response.get("inputTokens", 0)
        print(f"Token count result: {response}")
        return input_tokens

    except ClientError as e:
        print(f"Error counting tokens: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main():
    config = load_config()
    # count_tokens API does not support CRIS prefixes (global., eu., us.)
    import re

    model_id = re.sub(r"^(global|eu|us)\.", "", config["bedrock_model_id"])

    input_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for token counting."

    print(f"Model: {model_id}")
    print(f"Region: {config['region']}")
    print(f"Input text: {input_text}")

    token_count = count_text_tokens(input_text, model_id)
    char_count = len(input_text)

    print(f"Character count: {char_count:,}")

    if token_count is None:
        print("FAIL: Token counting returned no result")
        sys.exit(1)

    if token_count <= 0:
        print(f"FAIL: Token count should be positive, got {token_count}")
        sys.exit(1)

    print(f"Token count: {token_count:,}")
    print(f"Char-to-token ratio: {char_count / token_count:.2f}:1")
    print(f"\nPASS: count_tokens API returned {token_count} tokens")


if __name__ == "__main__":
    main()
