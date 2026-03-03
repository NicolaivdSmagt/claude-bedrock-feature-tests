# ABOUTME: Script to connect to Amazon Bedrock using the converse API
# ABOUTME: Sends content from input_205000.txt and asks about word count

import boto3
import json
import os
import sys
from botocore.config import Config

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)

# Configure Bedrock client with timeout and retries
_cfg = load_config()
boto_config = Config(read_timeout=60, retries=dict(max_attempts=1))

bedrock = boto3.client(
    service_name="bedrock-runtime", region_name=_cfg["region"], config=boto_config
)


def read_input_file():
    """Read the content of input_205000.txt from the files/ directory"""
    files_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "files"
    )
    path = os.path.join(files_dir, "input_205000.txt")
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: {path} not found")
        return None


def invoke_model_converse(content):
    """Use the converse API to send the content and question"""
    if not content:
        return

    # Prepare the prompt with context tags and question
    prompt = f"<context>\n{content}\n</context>\n\nHow many times do you see the phrase 'technical expert' in the context? Hint: it should be close to the name 'Richard Harris'."

    try:
        response = bedrock.converse(
            modelId=_cfg["bedrock_model_id"],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 4000, "temperature": 0.1},
            additionalModelRequestFields={"anthropic_beta": ["context-1m-2025-08-07"]},
        )

        # Extract and print the response
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)

    except Exception as e:
        print(f"Error invoking model: {e}")


if __name__ == "__main__":
    content = read_input_file()
    invoke_model_converse(content)
