---
name: bedrock-test-writer
description: >
  Write Bedrock API feature test scripts in both invoke_model and Converse
  variants. Use this skill whenever you need to create a new test for a Claude
  API feature on Amazon Bedrock, write invoke/converse test pairs, or generate
  test scripts that follow the project's conventions (ABOUTME headers, error
  classification, PASS/FAIL summary tables, structured output). Also use it when
  asked to validate or test a new Anthropic feature on Bedrock.
---

# Bedrock Test Writer

Create test scripts that validate Claude API features on Amazon Bedrock.
Every feature needs two files: an `_invoke.py` (invoke_model API) and a
`_converse.py` (Converse API). Both go in `tests/bedrock/` in the project root.

## Workflow

1. Read the Anthropic documentation for the feature (use WebFetch)
2. Read 1-2 existing tests in `tests/bedrock/` that test similar features
3. Write the `_invoke.py` variant
4. Write the `_converse.py` variant
5. Run both with `uv run python tests/bedrock/<name>.py` from the project root
6. Fix any issues through incremental test-and-fix cycles
7. Add both tests to the table in `README.md`

## File anatomy

Every test script follows this exact structure, in order:

```
1. Shebang:        #!/usr/bin/env python3
2. ABOUTME:        Two comment lines starting with "# ABOUTME: "
3. Docstring:      Triple-quoted description of what's being tested
4. Imports:        stdlib → third-party (with try/except) → sys.path hack → local
5. Constants:      UPPER_SNAKE_CASE module-level values
6. classify_error: Error classification helper (when testing features that may not be available)
7. test_*():       One function per test case, returns (status, error_msg)
8. print_summary:  Summary table printer
9. main():         Load config, create client, run tests, collect results, print summary
10. Guard:         if __name__ == "__main__": main()
```

## Import preamble

This exact boilerplate goes at the top of every test (after the docstring):

```python
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
```

## Config usage

Never hardcode model IDs, regions, or profiles. Always use config values:

```python
config = load_config()
model_id = config["bedrock_model_id"]
client = get_bedrock_client(config)
```

## invoke_model pattern

The invoke API uses the Anthropic-native JSON body format. Key conventions:
- `"anthropic_version": "bedrock-2023-05-31"` is required
- Beta features go in `"anthropic_beta": ["beta-name-here"]`
- snake_case keys throughout: `stop_reason`, `cache_control`, `tool_use`, `tool_result`
- Tools with Anthropic-specific types use `"type": "tool_type_date"` (e.g., `"type": "web_search_20250305"`)
- Standard tools use `"name"` and `"input_schema"`
- Response parsed from `json.loads(response["body"].read())`

```python
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "anthropic_beta": ["some-beta-2025-01-01"],  # only if feature requires a beta
    "max_tokens": 4096,
    "messages": [
        {"role": "user", "content": "Your test prompt here"}
    ],
    "tools": [
        {"type": "some_tool_20250101", "name": "tool_name"}
    ],
}

response = client.invoke_model(modelId=model_id, body=json.dumps(body))
result = json.loads(response["body"].read())
```

## Converse pattern

The Converse API has structural differences that matter. Key conventions:
- camelCase keys: `stopReason`, `inputTokens`, `outputTokens`, `toolUseId`, `maxTokens`
- Messages use `[{"text": "..."}]` content blocks
- Anthropic-specific tools and beta headers go in `additionalModelRequestFields`
- A placeholder toolSpec is required when using Anthropic-specific tools:

```python
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

request = {
    "modelId": model_id,
    "messages": [
        {
            "role": "user",
            "content": [{"text": "Your test prompt here"}],
        }
    ],
    "toolConfig": PLACEHOLDER_TOOL_CONFIG,
    "additionalModelRequestFields": {
        "anthropic_beta": ["some-beta-2025-01-01"],
        "tools": [
            {"type": "some_tool_20250101", "name": "tool_name"}
        ],
    },
    "inferenceConfig": {"maxTokens": 4096},
}

response = client.converse(**request)
```

## Error classification

Features that may not be available on Bedrock need error classification.
Bedrock returns specific error messages when a feature isn't supported, and
these should map to FAIL (not ERROR) since they tell us the feature isn't
available rather than indicating an infrastructure problem.

```python
FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "not supported",
    "unknown field",
    "unrecognized",
    "unknown tool type",
    "does not match any of the expected tags",
]

def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (other)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)
```

## Test function pattern

Each test function accepts `(client, model_id)` and returns `(status, error_msg)`:

```python
def test_feature_name(client, model_id):
    """Test description. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: FEATURE NAME")
    print("=" * 70)

    # ... build request body ...

    print("\n--- REQUEST BODY ---")
    print(json.dumps(body, indent=2))

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        print("\n--- RAW RESPONSE ---")
        print(json.dumps(result, indent=2))

        # ... validate response ...

        if validation_passed:
            print("\n  Result: PASS - description of what succeeded")
            return ("PASS", None)
        else:
            msg = "Description of what went wrong"
            print(f"\n  Result: FAIL - {msg}")
            return ("FAIL", msg)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        status, msg = classify_error(error_msg)
        print(f"\n  Result: {status}")
        return (status, msg)

    finally:
        print()
```

## Summary table

Every test must print a summary table and exit 0 only when all tests pass.
Use this exact function:

```python
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
    print(f"\n  Results: {passes} PASS, {fails} FAIL, {errors} ERROR")
    print("=" * 70)
    return all(s == "PASS" for _, s, _ in results)
```

Never truncate error messages in the summary — show the full message so
failures can be diagnosed from test output.

## main() pattern

```python
def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    print(f"Model: {model_id}")
    print()

    results = []

    status, error = test_feature_name(client, model_id)
    results.append(("Feature name", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
```

## Rules

- All tests make real API calls. Never implement mock modes or fake responses.
- Always create both `_invoke.py` and `_converse.py` for each feature.
- Print raw request/response JSON so failures can be diagnosed from output.
- Exit code 0 on all-pass, non-zero otherwise.
- Use f-strings for formatting, parenthesized concatenation for long strings.
- ABOUTME comments: two lines, each starting with `# ABOUTME: `.
- After writing tests, run them to verify they work. Fix issues incrementally.
- Add new tests to the table in `README.md` (Bedrock Tests section).
