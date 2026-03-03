# ABOUTME: Tests Bedrock message compaction by loading a pre-built enterprise workflow conversation.
# ABOUTME: Prints raw API request and response JSON with the conversation history replaced by a placeholder.

"""
Loads a static conversation history from 50000_token_conversation.json (generated once
by generate_compaction_history.py) and sends it to Bedrock with compaction enabled.

Prints the raw API request body and response body as JSON, replacing only the large
messages array with a placeholder.

Usage:
    uv run python manual/test_compaction.py
"""

import json
import os
import sys
import time

import boto3
from botocore.config import Config

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config

COMPACTION_TRIGGER_TOKENS = 50_000
COMPACTION_BETA = "compact-2026-01-12"

CUSTOM_COMPACTION_PROMPT = (
    "You have written a partial transcript for the initial task above. "
    "Please write a summary of the transcript. The purpose of this summary "
    "is to provide continuity so you can continue to make progress towards "
    "solving the task in a future context, where the raw history above may "
    "not be accessible and will be replaced with this summary. Write down "
    "anything that would be helpful, including the state, next steps, "
    "learnings etc. You must wrap your summary in a <summary></summary> block. "
)

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "files",
    "50000_token_conversation.json",
)


def setup_client(region):
    config = Config(region_name=region, read_timeout=600, retries=dict(max_attempts=3))
    return boto3.client("bedrock-runtime", config=config)


def main():
    config = load_config()
    MODEL_ID = config["bedrock_model_id"]
    REGION = config["region"]

    if not os.path.exists(HISTORY_FILE):
        print(
            f"ERROR: {HISTORY_FILE} not found. Run generate_compaction_history.py first."
        )
        return

    with open(HISTORY_FILE) as f:
        data = json.load(f)

    system = data["system"]
    tools = data["tools"]
    messages = data["messages"]
    stored_token_count = data["token_count"]

    # Append final user message that pushes over the trigger threshold
    messages.append(
        {
            "role": "user",
            "content": (
                "Now that we've completed the full migration, let's move on to the next phase. "
                "Can you set up the data migration pipeline to move existing orders from the "
                "MySQL monolith to the new PostgreSQL microservice? We need to handle UUID mapping, "
                "data validation, and a rollback strategy."
            ),
        }
    )

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": [COMPACTION_BETA],
        "max_tokens": 4096,
        "system": system,
        "tools": tools,
        "messages": messages,
        "context_management": {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {
                        "type": "input_tokens",
                        "value": COMPACTION_TRIGGER_TOKENS,
                    },
                    "pause_after_compaction": False,
                    "instructions": CUSTOM_COMPACTION_PROMPT,
                }
            ]
        },
    }

    # Print request with messages replaced by placeholder
    request_display = dict(request_body)
    request_display["messages"] = (
        f"[LARGE_CONVERSATION_HISTORY: {len(messages)} turns, {stored_token_count} tokens]"
    )
    print(">>> REQUEST BODY")
    print(json.dumps(request_display, indent=2))

    client = setup_client(REGION)
    print(f"\nSending to {MODEL_ID} in {REGION}...")
    start = time.time()

    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(request_body))
    response_body = json.loads(response["body"].read())
    elapsed = time.time() - start

    print(f"Response received in {elapsed:.1f}s\n")
    print("<<< RESPONSE BODY")
    print(json.dumps(response_body, indent=2, default=str))


if __name__ == "__main__":
    main()
