# AGENTS.md — Claude Bedrock Feature Tests

## Project Overview

Standalone Python test scripts for validating Claude API features on Amazon Bedrock
and the Anthropic 1st-party API. Each test script is self-contained and makes real
API calls. An automated test runner (powered by the Claude Agent SDK) can execute
all tests and produce a markdown report.

## Directory Structure

- `config.yaml` — Centralized config (region, model IDs, AWS profile)
- `load_config.py` — Shared config loader + Bedrock/Anthropic client helpers
- `run_tests.py` — Automated test runner (Claude Agent SDK)
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
```

### Run a single test

```bash
export AWS_PROFILE=work
uv run python tests/bedrock/adaptive_thinking_invoke.py
uv run python tests/anthropic/caching_min_prefix.py
```

### No formal lint/format/typecheck tooling

No ruff, flake8, mypy, black, etc. Follow the conventions documented below.

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
comparison tables. Test functions return either `(bool, str, Optional[dict])` or
`dict` with structured result fields (`error`, `usage`, `elapsed_ms`, etc.).

### No Mock Mode

All tests make real API calls. Never implement mock modes or fake responses.

### Adding New Tests

1. Create a `.py` file in `tests/bedrock/` or `tests/anthropic/`
2. Start with ABOUTME comments and the standard import preamble
3. Use `load_config()` and client helpers from `load_config.py`
4. Use `config["bedrock_model_id"]` / `config["anthropic_model_id"]`
5. The automated runner discovers tests via `tests/<suite>/*.py` glob
6. Exit with code 0 on success, non-zero on failure
7. Add the test to the appropriate table in `README.md` (Bedrock or Anthropic)
