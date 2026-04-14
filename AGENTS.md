# AGENTS.md — Claude Bedrock Feature Tests

## Project Overview

Standalone Python test scripts for validating Claude API features and parity
between Amazon Bedrock and the Anthropic 1st-party API. Each test script is
self-contained and makes real API calls. An automated test runner executes all
tests via subprocess, then uses Claude (via Bedrock) to classify the results
and produce a markdown report.

## Directory Structure

- `config.yaml` — Centralized config (region, model IDs, AWS profile)
- `load_config.py` — Shared config loader + Bedrock/Anthropic client helpers
- `run_tests.py` — Automated test runner (subprocess + LLM classification)
- `tests/bedrock/` — Bedrock API tests (invoke_model + Converse)
- `tests/anthropic/` — Anthropic 1st-party API tests
- `tests/inactive/` — Tests for features not available on current models
- `files/` — Test assets (images, PDFs, large text files)
- `support/` — Helper scripts for generating test data
- `reports/` — Generated test reports (gitignored)

## Build & Run Commands

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/) (no pip/poetry),
AWS credentials (`AWS_PROFILE=work`), Anthropic API key in AWS Secrets Manager.

```bash
uv sync                                                  # Install dependencies

export AWS_PROFILE=work CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=eu-west-1
uv run python run_tests.py --suite bedrock               # Run all Bedrock tests
uv run python run_tests.py --suite anthropic              # Anthropic API tests
uv run python run_tests.py --suite all                    # All tests
uv run python run_tests.py --timeout 600                  # Custom timeout per test
uv run python run_tests.py -v                             # Include full test output
uv run python run_tests.py -o auto                        # Save report to reports/
uv run python run_tests.py -o results.md                  # Save report to specific file
```

### Run a single test

```bash
export AWS_PROFILE=work
uv run python tests/bedrock/adaptive_thinking_invoke.py
uv run python tests/anthropic/caching_min_prefix.py
```

### No formal lint/format/typecheck tooling

No ruff, flake8, mypy, black, etc. Follow the conventions documented below.

### Git Push

See `AGENTS_PRIVATE.md` for remote URLs and push instructions.

## Configuration

All test scripts read from `config.yaml` in the project root via `load_config.py`.
Never hardcode model IDs, regions, or AWS profiles — always use config values:

```python
from load_config import load_config, get_bedrock_client, get_anthropic_client
config = load_config()
model_id = config["bedrock_model_id"]
client = get_bedrock_client(config)
```

Key config fields: `aws_profile`, `region`, `bedrock_model_id`,
`anthropic_model_id`, `secrets_manager_region`, `secrets_manager_secret_name`.

## Code Style Guidelines

### Import Order

Imports follow this order with a blank line between each group:
1. Standard library (`import json`, `import os`, `import sys`, `import time`)
2. Third-party packages (`import boto3`, `import anthropic`, `import yaml`)
3. `sys.path` manipulation for local imports
4. Local imports (`from load_config import load_config, get_bedrock_client`)

Guard optional third-party imports with try/except:
`try: import anthropic / except ImportError: print("...Run: uv add anthropic"); sys.exit(1)`

### Type Hints

Used selectively — primarily in function signatures for helpers and shared
utilities. Use `typing` imports (`Any`, `Optional`, `Tuple`) and built-in
generics (`list[Path]`, `dict`). Follow surrounding code.

### Naming Conventions

- **Functions**: `snake_case` — `run_adaptive_thinking`, `print_comparison`
- **Constants**: `UPPER_SNAKE_CASE` — `EFFORT_LEVELS`, `MODEL_ID`, `REGION`
- **Variables**: `snake_case` — `model_id`, `response_body`, `elapsed_ms`
- **Module-level config**: `snake_case` or `UPPER_SNAKE_CASE` depending on
  whether they are true constants or derived from config

### String Formatting

Use f-strings consistently. Multi-line strings use parenthesized concatenation:

```python
PROMPT = (
    "First part of the prompt "
    "second part continues here."
)
```

### Error Handling

- Wrap API calls in try/except blocks
- Catch specific exceptions first (`ClientError`, `ValidationException`),
  then broad `Exception` as fallback
- Store errors in result dicts as strings (`result["error"] = str(e)`)
- Print errors with type info: `f"{type(e).__name__}: {e}"`
- Never silently swallow exceptions

### Test Script Structure

Every test script follows this pattern:
1. Shebang + ABOUTME header
2. Imports (stdlib, third-party, local)
3. Module-level constants (`MODEL_ID`, `PROMPT`, etc.)
4. Helper functions for individual test cases (return result dicts or tuples)
5. Pretty-print / summary functions
6. `main()` function: load config, create client, run tests, print results
7. `if __name__ == "__main__": main()` guard

### Output & Return Values

Tests produce structured console output: section headers (`"=" * 70`), labeled
fields, JSON dumps for request/response bodies, PASS/FAIL verdicts, and summary
comparison tables. Test functions return `(status, error_msg)` tuples as
described in the PASS/FAIL Summary Table section below.

### PASS/FAIL Summary Table

Every test script must print a summary table at the end and exit with code 0
only when all tests pass. Each test function returns `(status, error_msg)`
where status is `"PASS"`, `"FAIL"`, or `"ERROR"` and error_msg is `None` on
success or a descriptive string otherwise. `main()` collects these into a
results list and calls a `print_summary()` function:

```python
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
```

### Multi-Test Scripts

When a script covers multiple related features (e.g. regex search, BM25
search, custom search), put each in its own `test_*()` function that accepts
`(client, model_id)` and returns `(status, error_msg)`. `main()` calls them
sequentially, collects results with descriptive labels, and delegates to
`print_summary()`.

### No Mock Mode

All tests make real API calls. Never implement mock modes or fake responses.

### Invoke / Converse Pairs

Most Bedrock tests must have both an `_invoke.py` and `_converse.py` variant:

- **Invoke** (`client.invoke_model()`) uses the Anthropic-native JSON body
  format: `stop_reason`, `cache_control`, `tool_use` / `tool_result` content
  blocks, betas in `anthropic_beta`, etc.
- **Converse** (`client.converse()`) uses the Converse-native format:
  `stopReason`, `toolUse` / `toolResult`, content as `[{"text": "..."}]`,
  and Anthropic-specific features passed via `additionalModelRequestFields`.

Certain tests (such as magic_strings.py) test functionality that is API agnostic.
Those do not require separate variants for Converse and invoke_model APIs.


### Converse API Patterns

The Converse API has several structural differences from invoke_model:

- **Placeholder toolSpec**: When using Anthropic-specific tools (memory,
  code execution, etc.) via `additionalModelRequestFields`, Converse still
  requires at least one `toolSpec` in `toolConfig.tools`:
  ```python
  PLACEHOLDER_TOOL_CONFIG = {
      "tools": [{"toolSpec": {"name": "placeholder",
                              "inputSchema": {"json": {"type": "object"}}}}]
  }
  ```

- **Beta headers and tool definitions** go in `additionalModelRequestFields`:
  ```python
  additionalModelRequestFields={
      "anthropic_beta": ["context-management-2025-06-27"],
      "tools": [{"type": "memory_20250818", "name": "memory"}],
  }
  ```

- **Cache points** use `cachePoint` as a separate content block (not inline
  like invoke's `cache_control`):
  ```python
  system=[{"text": content}, {"cachePoint": {"type": "default", "ttl": "1h"}}]
  ```

- **camelCase keys** throughout: `inputTokens`, `outputTokens`,
  `cacheWriteInputTokens`, `cacheReadInputTokens`, `stopReason`, `toolUseId`,
  `maxTokens`, etc.

- **context_management not returned in responses**: The Converse API accepts
  `context_management` parameters without error, and clearing does happen
  server-side, but the response does NOT include a `context_management` field
  with clearing stats (unlike invoke_model which returns `applied_edits`).
  To verify clearing in Converse tests, compare input token counts between
  runs with and without `context_management` enabled.

- **Some beta features not supported on Converse**: Tool search
  (`tool-search-tool-2025-10-19`) and compaction (`compact-2026-01-12`)
  are not supported on the Converse API. Scripts will ERROR when run,
  which is expected and intentional for validation.

### Error Classification

For tests that validate features which may not be available on Bedrock,
use error markers and a `classify_error()` helper:

- **Feature not available** (classify as `FAIL`): When Bedrock returns
  "The provided request is not valid" or "does not match any of the expected
  tags", this means the feature isn't supported. Use
  `FEATURE_NOT_AVAILABLE_MARKERS` list to detect these.
- **Size/limit errors** (classify as `FAIL`): When testing size limits,
  match against `SIZE_LIMIT_ERROR_MARKERS` which includes `"too large"`,
  `"too long"`, `"input is too long"`, `"size limit"`, `"maximum size"`,
  `"payload"`, `"exceeds"`, `"request size"`, `"content length"`,
  `"image size"`, `"too many"`.
- **Other errors** (classify as `ERROR`): Auth failures, network problems,
  or unexpected exceptions that aren't related to the feature under test.

### Limit Test Pattern

Tests that validate API limits (image count, image size, PDF limits) use a
single overall verdict pattern:

1. Run multiple test cases (e.g. 20/21/100/101 images)
2. Show raw API output per case (stop_reason, usage, content on success;
   full exception on failure) — don't use canned strings like "Expected
   rejection but succeeded"
3. Print a summary table
4. Compute a single PASS/FAIL/ERROR verdict at the end

### Don't Truncate Error Messages

The `print_summary()` function must NOT truncate error strings (no `[:120]`
etc.) — show the full message so failures can be diagnosed from test output.

### Caching Threshold

Bedrock prompt caching requires a minimum of ~2048 tokens in the cached
content. For tests that need large system prompts (e.g. extended cache TTL),
use `files/50000_token_conversation.json` as a content source — load its
`system`, `tools`, and/or first N `messages` to build content above the
threshold.

### Adding New Tests

1. Create a `.py` file in `tests/bedrock/` or `tests/anthropic/`
2. For Bedrock: create both `_invoke.py` and `_converse.py` variants
3. Start with ABOUTME comments and the standard import preamble
4. Use `load_config()` and client helpers from `load_config.py`
5. Use `config["bedrock_model_id"]` / `config["anthropic_model_id"]`
6. The automated runner discovers tests via `tests/<suite>/*.py` glob
7. Exit with code 0 on success, non-zero on failure
8. Add the test to the appropriate table in `README.md` (Bedrock or Anthropic)
9. **Always run new test scripts after writing them** to verify they execute
   correctly, produce proper output, and classify results as expected. Fix
   any issues through incremental test-and-fix cycles before considering
   the work complete. Do not submit test scripts that have not been executed
   at least once.
