# Claude Bedrock Feature Tests

Standalone test scripts for validating Claude API features on Amazon Bedrock and the Anthropic 1st-party API. Covers adaptive thinking, caching, structured outputs, tool use, vision, PDFs, context windows, and more.

An automated test runner executes all tests via subprocess, then uses Claude (via Bedrock) to classify the results and produce a markdown report. This gives accurate, context-aware classification of test outcomes without brittle regex parsing.

## Quick Start

See `AGENTS_PRIVATE.md` for remote URLs and push instructions.bash
# Install dependencies
uv sync

# Configure (edit config.yaml with your region, model, and profile)
vim config.yaml

# Run all Bedrock tests
export AWS_PROFILE=work
uv run python run_tests.py --suite bedrock
See `AGENTS_PRIVATE.md` for remote URLs and push instructions.

The report is saved to `reports/report_<model>_<region>_<timestamp>.md`.

## Configuration

All test scripts read from `config.yaml`:

See `AGENTS_PRIVATE.md` for remote URLs and push instructions.yaml
aws_profile: "work"
region: "eu-west-1"
bedrock_model_id: "global.anthropic.claude-sonnet-4-6"
anthropic_model_id: "claude-sonnet-4-6"
secrets_manager_region: "eu-west-1"
secrets_manager_secret_name: "anthropic_api_key"
See `AGENTS_PRIVATE.md` for remote URLs and push instructions.

To test a different model, change `bedrock_model_id` and/or `anthropic_model_id` and rerun.

## Directory Structure

See `AGENTS_PRIVATE.md` for remote URLs and push instructions.
.
├── config.yaml              # Centralized configuration
├── load_config.py           # Shared config loader + client helpers
├── run_tests.py             # Automated test runner (subprocess + LLM classification)
├── tests/
│   ├── bedrock/             # Bedrock API tests (invoke_model + Converse)
│   ├── anthropic/           # Anthropic 1st-party API tests
│   └── inactive/            # Tests for features not available on current models
├── files/                   # Test assets (images, PDFs, large text files)
├── support/                 # Helper scripts for generating test data
└── reports/                 # Generated test reports (gitignored)
See `AGENTS_PRIVATE.md` for remote URLs and push instructions.

## Test Suites

### Bedrock Tests (`tests/bedrock/`)

| Test | Feature |
|------|---------|
| `adaptive_thinking_invoke.py` | Extended thinking with effort levels (invoke_model) |
| `adaptive_thinking_converse.py` | Extended thinking with effort levels (Converse) |
| `advisor_tool_invoke.py` | Advisor tool: executor consults advisor model (invoke_model) |
| `advisor_tool_converse.py` | Advisor tool: executor consults advisor model (Converse) |
| `automatic_caching_invoke.py` | Automatic prompt caching with top-level cache_control (invoke_model) |
| `automatic_caching_converse.py` | Automatic prompt caching with top-level cache_control (Converse) |
| `bash_tool_invoke.py` | Bash tool (invoke_model) |
| `bash_tool_converse.py` | Bash tool (Converse) |
| `text_editor_invoke.py` | Text editor tool (invoke_model) |
| `text_editor_converse.py` | Text editor tool (Converse) |
| `caching_min_prefix.py` | Minimum prefix size for prompt caching (~2048 tokens) |
| `cache_with_structured_outputs.py` | Caching interaction with structured JSON output |
| `extended_cache_ttl_invoke.py` | Extended 1-hour cache TTL (invoke_model) |
| `extended_cache_ttl_converse.py` | Extended 1-hour cache TTL (Converse) |
| `fine_grained_tool_streaming_invoke.py` | Fine-grained tool streaming with eager_input_streaming (invoke_model) |
| `fine_grained_tool_streaming_converse.py` | Fine-grained tool streaming with eager_input_streaming (Converse) |
| `structured_outputs_invoke.py` | JSON schema output + strict tool use (invoke_model) |
| `structured_outputs_converse.py` | JSON schema output + strict tool use (Converse) |
| `count_tokens.py` | count_tokens API |
| `code_execution_invoke.py` | Code execution tool (invoke_model) |
| `code_execution_converse.py` | Code execution tool (Converse) |
| `tool_search_invoke.py` | Tool search: regex, BM25, custom client-side (invoke_model) |
| `tool_search_converse.py` | Tool search: regex, BM25, custom client-side (Converse) |
| `web_search_invoke.py` | Web search (invoke_model) |
| `web_search_converse.py` | Web search (Converse) |
| `parallel_tool_use_invoke.py` | Parallel tool calls in a single response (invoke_model) |
| `parallel_tool_use_converse.py` | Parallel tool calls in a single response (Converse) |
| `clear_thinking_invoke.py` | Clear old thinking blocks to save tokens (invoke_model) |
| `clear_thinking_converse.py` | Clear old thinking blocks to save tokens (Converse) |
| `clear_tool_use_invoke.py` | Clear old tool use/result pairs (invoke_model) |
| `clear_tool_use_converse.py` | Clear old tool use/result pairs (Converse) |
| `compaction_invoke.py` | Message compaction (invoke_model) |
| `compaction_converse.py` | Message compaction (Converse) |
| `mcp_connector_invoke.py` | MCP connector: remote MCP server connection (invoke_model) |
| `mcp_connector_converse.py` | MCP connector: remote MCP server connection (Converse) |
| `memory_test_invoke.py` | Memory tool: create + retrieve across sessions (invoke_model) |
| `memory_test_converse.py` | Memory tool: create + retrieve across sessions (Converse) |
| `image_limit_invoke.py` | Image count limit per request (invoke_model) |
| `image_limit_converse.py` | Image count limit per request (Converse) |
| `image_size_limit_invoke.py` | Maximum image size, single + multi (invoke_model) |
| `image_size_limit_converse.py` | Maximum image size, single + multi (Converse) |
| `pdf_limits_invoke.py` | PDF count and payload size limits (invoke_model) |
| `pdf_limits_converse.py` | PDF count and payload size limits (Converse) |
| `1M_context_invoke.py` | 1M context window (invoke_model) |
| `1M_context_converse.py` | 1M context window (Converse) |
| `magic_strings.py` | Redacted thinking + streaming refusal magic string behavior |

### Anthropic API Tests (`tests/anthropic/`)

| Test | Feature |
|------|---------|
| `adaptive_thinking.py` | Extended thinking with effort levels |
| `caching_min_prefix.py` | Minimum prefix size for prompt caching |
| `code_execution.py` | Code execution tool |
| `image_size_limit.py` | Maximum image size limits |
| `mcp_connector.py` | MCP connector: remote MCP server connection and tool discovery |
| `tool_search.py` | Server-side tool search (regex + BM25) |

### Inactive Tests (`tests/inactive/`)

| Test | Reason |
|------|--------|
| `test_redacted_thinking.py` | `redacted_thinking` blocks not observable on Claude 4+ (summarized thinking) |
| `test_redacted_thinking_anthropic.py` | Same - Anthropic API version |

## Running Tests

### Automated (recommended)

The test runner executes all tests via subprocess, printing live status as each completes. After all tests finish, it sends the captured output to Claude (via Bedrock) for accurate classification and report generation:

See `AGENTS_PRIVATE.md` for remote URLs and push instructions.bash
# Bedrock tests only (default)
uv run python run_tests.py

# Anthropic API tests
uv run python run_tests.py --suite anthropic

# All tests
uv run python run_tests.py --suite all

# Custom timeout per test (default: 300s)
uv run python run_tests.py --timeout 600

# Save report to file (use 'auto' for auto-generated filename)
uv run python run_tests.py -o auto

# Include full raw output from each test
uv run python run_tests.py -v

# Skip LLM classification (use regex-based heuristics only)
uv run python run_tests.py --no-llm
See `AGENTS_PRIVATE.md` for remote URLs and push instructions.

Environment variables:
- `AWS_PROFILE` - AWS credentials profile

### Manual (individual tests)

Each test is a standalone script:

See `AGENTS_PRIVATE.md` for remote URLs and push instructions.bash
export AWS_PROFILE=work
uv run python tests/bedrock/adaptive_thinking_invoke.py
uv run python tests/anthropic/caching_min_prefix.py
See `AGENTS_PRIVATE.md` for remote URLs and push instructions.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for package management
- AWS credentials with Bedrock access
- For Anthropic API tests: API key stored in AWS Secrets Manager


## Test Assets (`files/`)

| File | Size | Used By |
|------|------|---------|
| `3mb.jpg`, `4mb.jpg`, `5mb.jpg`, `6mb.jpg` | 3-6 MB | `image_size_limit.py` |
| `example.pdf` | 1.4 MB | `pdf_limits.py` |
| `example_106.pdf` | 9.1 MB | `pdf_limits.py` |
| `test_pdf_31MB.pdf` | 31 MB | `pdf_limits.py` |
| `input_205000.txt` | 888 KB | `1M_context_invoke.py`, `1M_context_converse.py` |
| `50000_token_conversation.json` | 212 KB | `compaction.py` |

## Adding Tests

1. Create a new `.py` file in `tests/bedrock/` or `tests/anthropic/`
2. Import config via:
   See `AGENTS_PRIVATE.md` for remote URLs and push instructions.python
   import os, sys
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
   from load_config import load_config, get_bedrock_client
   See `AGENTS_PRIVATE.md` for remote URLs and push instructions.
3. Use `config["bedrock_model_id"]` and `config["region"]` instead of hardcoded values
4. The test runner will automatically discover and run the new test
