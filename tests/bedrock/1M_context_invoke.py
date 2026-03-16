# ABOUTME: Tests 1M context window support on Bedrock via the invoke_model API.
# ABOUTME: Sends ~205K tokens from input_205000.txt and validates a successful response.

import copy
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

FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "files"
)

PROMPT_PREFIX = (
    "Provide a numbered list of Darwin's main arguments about "
    "natural selection, without further explanation: "
)

# Models that still require the beta header for 1M context.
# Sonnet 4 (sonnet-4-20250514) and Sonnet 4.5 (sonnet-4-5) need it;
# all other models (Sonnet 4.6+, Opus, Haiku, etc.) have 1M context GA.
BETA_REQUIRED_MARKERS = ["sonnet-4-20", "sonnet-4-5"]


def needs_1m_beta(model_id):
    """Return True if the model requires the context-1m beta header."""
    return any(marker in model_id for marker in BETA_REQUIRED_MARKERS)


def test_1m_context(client, model_id):
    """Send ~205K tokens via invoke_model, using the 1M context beta header
    only for older models that require it. Returns (status, error_msg)."""
    use_beta = needs_1m_beta(model_id)
    print(f"  model: {model_id}")
    print(f"  1M context beta header: {'yes' if use_beta else 'no (GA)'}")

    input_path = os.path.join(FILES_DIR, "input_205000.txt")
    with open(input_path, "r", encoding="utf-8") as f:
        large_document = f.read()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": PROMPT_PREFIX + large_document}],
            }
        ],
    }
    if use_beta:
        body["anthropic_beta"] = ["context-1m-2025-08-07"]

    # Print request body with large document replaced for readability
    display_body = copy.deepcopy(body)
    display_body["messages"][0]["content"][0]["text"] = (
        PROMPT_PREFIX + "[LARGE_DOCUMENT]"
    )
    print("\n--- REQUEST BODY ---")
    print(json.dumps(display_body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{type(e).__name__}: {e}")
        return ("ERROR", f"{type(e).__name__}: {e}")

    stop_reason = result.get("stop_reason")
    usage = result.get("usage", {})
    text = result.get("content", [{}])[0].get("text", "")

    # Print relevant response fields
    print("\n--- RESPONSE ---")
    print(
        json.dumps(
            {
                "stop_reason": stop_reason,
                "usage": usage,
                "content_preview": text[:200],
            },
            indent=2,
        )
    )

    if stop_reason != "end_turn":
        return ("FAIL", f"unexpected stop_reason: {stop_reason}")

    if not text.strip():
        return ("FAIL", "empty response text")

    return ("PASS", None)


def print_summary(results):
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
    print(f"\n  Results: {passes} PASS, {fails} FAIL, {errors} ERROR")
    print("=" * 70)
    return all(s == "PASS" for _, s, _ in results)


def main():
    config = load_config()
    # Allow overriding the model ID via command-line argument
    model_id = sys.argv[1] if len(sys.argv) > 1 else config["bedrock_model_id"]
    client = get_bedrock_client(config)

    results = []

    print("=" * 70)
    print("  1M context invoke_model (~205K tokens)")
    print("=" * 70)
    status, error = test_1m_context(client, model_id)
    results.append(("1M context invoke_model (~205K tokens)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
