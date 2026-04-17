# Agent Context — Claude Bedrock Feature Tests

This agent runs in the `agent/` subdirectory of the Claude Bedrock Feature
Tests project. The parent directory contains the test scripts, configuration,
and shared utilities.

## Project Layout (relative to parent)

- `../config.yaml` — Model IDs, AWS profile, region, SNS settings
- `../load_config.py` — Shared config loader + Bedrock/Anthropic client helpers
- `../tests/bedrock/` — Bedrock API test scripts (invoke_model + Converse pairs)
- `../tests/anthropic/` — Anthropic 1st-party API tests
- `../files/` — Test assets (images, PDFs, large text files)
- `../README.md` — Test listing tables (update when adding tests)
- `../run_tests.py` — Automated test runner

## Running the Agent

```bash
export AWS_PROFILE=work
uv run python agent/run_agent.py                # daily mode (default)
uv run python agent/run_agent.py --mode daily   # explicit daily
uv run python agent/run_agent.py --mode weekly  # full audit
```

### Daily mode

Searches Anthropic news, AWS blogs, and X for new Claude API features
announced in the past 30 days. Writes tests for uncovered features.
Budget: $5.

### Weekly mode

Pulls the latest `claude-api` skill from GitHub, reads the full API
documentation, compares against the test suite, writes tests for gaps,
and flags deprecated features. Budget: $10.

### Notifications

Both modes send a report via AWS SNS. Configure in `../config.yaml`:
- `sns_topic_arn` — ARN of the SNS topic
- `sns_region` — region of the SNS topic

Use the Terraform in `infra/` to create the topic:
```bash
cd agent/infra
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your region, profile, and email
terraform init
terraform apply
```
Copy the output `sns_topic_arn` to `config.yaml`. Confirm the email
subscription via the link AWS sends.

## Running Tests

From the project root (parent directory):

```
uv run python tests/bedrock/<test_name>.py
```

Set `AWS_PROFILE=work` before running.

## Adding Tests

1. Write both `_invoke.py` and `_converse.py` variants in `../tests/bedrock/`
2. Use the `bedrock-test-writer` skill for the exact patterns and boilerplate
3. Run the tests to verify they work — fix issues incrementally
4. Add both tests to the Bedrock Tests table in `../README.md`

## Git Commit and Push

After creating new tests, the agent must commit and push:

1. Read `../AGENTS_PRIVATE.md` for remote URLs and push flags.
2. Stage only new/modified files (test scripts + README.md).
3. Commit with a descriptive message.
4. Push to BOTH remotes as described in `../AGENTS_PRIVATE.md`.

Skip this step if no new tests were created.

## Safety Rules

The agent operates under strict safety guardrails:

- **NEVER delete or remove existing test files** — even for deprecated features
- **NEVER modify existing test files** — only create new files
- **Flag deprecated functionality** in the report but take no action on files
- Only write to `../tests/bedrock/` (new scripts) and `../README.md` (new rows)

Humans handle all removals and modifications to existing tests.

## Code Style

- Two-line `# ABOUTME:` header at the top of every file
- f-strings for formatting
- snake_case functions/variables, UPPER_SNAKE_CASE constants
- No mock modes — all tests make real API calls
- Wrap API calls in try/except, print errors with type info
- Never silently swallow exceptions

## Available Bedrock Model IDs

These are in `../config.yaml` but for reference:

### Claude 4.7 (on Bedrock Mantle)
- `global.anthropic.claude-opus-4-7`
- `us.anthropic.claude-opus-4-7`
- `eu.anthropic.claude-opus-4-7`
- `anthropic.claude-opus-4-7` — bare format, used with `AnthropicBedrockMantle` client via the Messages API on the `bedrock-mantle` endpoint

### Claude 4.6 and earlier
- `global.anthropic.claude-opus-4-6-v1`
- `global.anthropic.claude-sonnet-4-6`
- `global.anthropic.claude-haiku-4-5-20251001-v1:0`

## Bedrock Mantle — The New Inference Path

Starting with Opus 4.7, Anthropic models run on AWS's next-generation inference
engine called **Bedrock Mantle**. Three API paths reach Opus 4.7 today:

1. **Messages API (via Bedrock Mantle)** — Endpoint:
   `https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages`. Uses the
   `AnthropicBedrockMantle` client from the `anthropic` Python SDK (install
   with `uv add "anthropic[bedrock]"`). Most feature-complete path. Uses
   bare model IDs REQUIRED (e.g., `anthropic.claude-opus-4-7` — CRIS IDs
   like `us.*` return 404). Requires `bedrock-mantle:CreateInference` IAM
   action. Opus 4.7 launches in `us-east-1` and `us-west-2` only.
2. **InvokeModel (on bedrock-runtime)** — Existing path, fully supported for
   Opus 4.7. Uses CRIS model IDs (`us.`, `eu.`, `global.`).
3. **Converse (on bedrock-runtime)** — Existing path, fully supported for
   Opus 4.7. Note: compaction and task budgets are NOT supported on Converse
   during their beta periods.

Coming soon on Bedrock Mantle for Claude: OpenAI-compatible Responses API
and Chat Completions API.

### Opus 4.7 Breaking Changes (vs 4.6)

Tests targeting Opus 4.7 specifically should account for:
- **Sampling parameters deprecated** — `temperature`, `top_p`, `top_k` return
  HTTP 400 if set to non-default values. Safest: omit them entirely.
- **`budget_tokens` removed** — Extended thinking with `budget_tokens` is no
  longer supported. Migrate to adaptive thinking with `effort` in `output_config`.
- **Effort placement** — `effort` goes in `output_config`, NOT in `thinking`
  dict. `{"thinking": {"type": "adaptive", "effort": "high"}}` returns 400.
- **Vision resolution increase** — Images processed up to 2576px (was 1568px).
  ~3x more image tokens; max images per request may drop from ~100 to ~40.

### Opus 4.7 New Features

- **Task Budgets (beta)** — `anthropic-beta: task-budgets-2026-03-13`,
  `output_config.task_budget = {"type": "tokens", "total": N}`. Minimum 20K
  tokens. Advisory (not hard-enforced). Available on Messages API and
  InvokeModel; NOT on Converse during beta.
- **1M context, no pricing jump above 200K** (was a premium pricing tier
  on 4.6).
- **Max output: 128K tokens.**

## Skills

The agent has access to two skills:

### bedrock-test-writer (project skill)
How to write invoke_model and Converse test pairs. Contains the exact
boilerplate, patterns, and conventions for test scripts in this project.

### claude-api (refreshed from GitHub weekly)
Comprehensive reference for the Claude API surface — tools, features,
parameters, beta headers. Pulled fresh from `anthropics/skills` on GitHub
before each weekly audit run.
