# ABOUTME: Lightweight script to invoke Claude Sonnet 4 via Amazon Bedrock with configurable region and model ID.
# ABOUTME: Demonstrates basic usage of the Bedrock invoke_model API with 1M context support.
import boto3, json, argparse
import os
import sys

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client


def invoke_claude_sonnet4_1m(prompt_text, region, model_id):
    # Configure Bedrock client
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ],
        "anthropic_beta": ["context-1m-2025-08-07"],  # Header for 1M context
    }

    response = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))

    return json.loads(response["body"].read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Invoke Claude Sonnet 4 via Amazon Bedrock"
    )
    config = load_config()
    parser.add_argument(
        "--region",
        type=str,
        default=config["region"],
        help=f"AWS region (default: {config['region']})",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=config["bedrock_model_id"],
        help=f"Model ID (default: {config['bedrock_model_id']})",
    )

    args = parser.parse_args()

    # Example usage (this will cause ~205K input tokens to be sent to Claude)
    files_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "files"
    )
    with open(
        os.path.join(files_dir, "input_205000.txt"), "r", encoding="utf-8"
    ) as file:
        large_document = file.read()
    result = invoke_claude_sonnet4_1m(
        f"Provide a numbered list of Darwin's main arguments about natural selection, without further explanation: {large_document}",
        args.region,
        args.model_id,
    )
    print(result["content"][0]["text"])
