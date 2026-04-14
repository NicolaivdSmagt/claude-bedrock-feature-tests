#!/usr/bin/env python3
# ABOUTME: Feature scout agent that searches for new Claude API features and writes Bedrock tests.
# ABOUTME: Supports daily (news search) and weekly (full audit) modes with SNS notifications.

"""
Anthropic Feature Scout Agent
==============================

Daily mode: Searches Anthropic and AWS news for new Claude API features,
then writes invoke_model + Converse test scripts for uncovered features.

Weekly mode: Pulls the latest claude-api skill from GitHub, reads the full
API documentation, compares against existing test coverage, writes tests
for any gaps, and flags deprecated functionality.

Both modes send a report via AWS SNS when complete.

Usage:
    export AWS_PROFILE=work
    uv run python agent/run_agent.py                # daily (default)
    uv run python agent/run_agent.py --mode daily
    uv run python agent/run_agent.py --mode weekly
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Error: boto3 not installed. Run: uv add boto3")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: pyyaml not installed. Run: uv add pyyaml")
    sys.exit(1)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
except ImportError:
    print("Error: claude-agent-sdk not installed. Run: uv add claude-agent-sdk")
    sys.exit(1)

# Resolve paths relative to this script's location
AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
CLAUDE_API_SKILL_DIR = AGENT_DIR / ".claude" / "skills" / "claude-api"

SKILLS_REPO_URL = "https://github.com/anthropics/skills.git"
SKILLS_REPO_SKILL_PATH = "skills/claude-api"

# Git commit/push instructions injected into every prompt
GIT_INSTRUCTIONS = """
## Git: Commit and Push

After all tests are written, run, and added to README.md, commit and push:

1. Read `AGENTS_PRIVATE.md` in the project root for remote URLs and push
   instructions (it contains required flags for specific remotes).
2. Stage only the new/modified files: new test scripts in `tests/bedrock/`
   and `README.md`. Do NOT stage `config.yaml`, `load_config.py`, or any
   other pre-existing files.
3. Commit with a descriptive message, e.g.:
   `Add tests for <feature>: invoke + converse variants`
4. Push to BOTH remotes as described in `AGENTS_PRIVATE.md`.

If no new tests were created, skip the commit/push step.
"""

# Safety guardrails injected into every prompt
SAFETY_RULES = """
## CRITICAL SAFETY RULES

These rules override all other instructions. Violating them is never acceptable.

1. NEVER delete or remove any existing test file. Not even if a feature is
   deprecated, removed, or no longer functional. Only humans remove tests.

2. NEVER modify existing test files. You may only create new files.

3. If you discover that a feature tested by an existing script has been
   deprecated or removed, include it in your report under "Deprecated
   features" — but take NO action on the file.

4. Only write files inside `tests/bedrock/` (new test scripts) and
   `README.md` (adding rows to the test table). Do not write anywhere else
   in the project.
"""

DAILY_PROMPT = f"""\
You are a feature scout for the Claude Bedrock Feature Tests project. Your job
is to find new Claude API features announced in the past 30 days and write
Bedrock test scripts for any that are not already covered.

{SAFETY_RULES}

## Step 1: Discover what's already tested

Read the existing test coverage:
- Glob for `tests/bedrock/*.py` in the project root to see all current tests.
- Read the ABOUTME headers of each test to understand what features are covered.
- Read `README.md` to see the test table.

## Step 2: Search for new Claude API features

Search these sources for recent Claude API announcements (last 30 days):

1. **Anthropic official**: Search for "Anthropic Claude API new features"
   and check https://docs.anthropic.com/en/docs/about-claude/models and
   https://www.anthropic.com/news
2. **AWS Bedrock**: Search for "AWS Bedrock Claude new features" and check
   https://aws.amazon.com/about-aws/whats-new/ (filter for Bedrock/Claude)
3. **X/Twitter**: Search for recent posts from @AnthropicAI about API features

Focus specifically on:
- New API tools (like web_search, code_execution, bash, text_editor, memory, advisor)
- New API features (like caching, structured outputs, context management, compaction)
- New beta capabilities and beta headers
- New model parameters or options
- Changes to existing API behavior (new tool versions, new parameters)

Ignore: pricing changes, model availability in new regions, non-API product
updates (Claude.ai UI changes, etc.), and features already covered by existing tests.

## Step 3: Write tests for new features

For each genuinely new feature that isn't already tested:

1. Fetch the official Anthropic documentation for the feature (use WebFetch)
   to understand the exact API format (parameters, request/response structure,
   beta headers, tool type strings)
2. Use the bedrock-test-writer skill to create both `_invoke.py` and
   `_converse.py` test variants
3. Write the files to `tests/bedrock/` in the project root
4. Run each test from the project root to verify it works:
   `cd <project_root> && uv run python tests/bedrock/<name>.py`
5. If a test fails due to a bug in the script (not a feature-not-available
   error), fix the script and rerun. Iterate until the test runs cleanly.
6. Add the new tests to the Bedrock Tests table in `README.md`

{GIT_INSTRUCTIONS}

## Step 4: Report

Print a structured report with these sections:
- **Features discovered**: name, brief description, source URL
- **Tests created**: file names and test results (PASS/FAIL/ERROR)
- **Features skipped**: already tested or not API-related (brief reason)
- **Git**: commit hash and push status (or "no changes" if nothing was created)
- **No new features**: if nothing was found, say so — that's a valid outcome

If you find no new features to test, that's fine — report it clearly.
"""

WEEKLY_PROMPT = f"""\
You are running the weekly audit of the Claude Bedrock Feature Tests project.
Your job is to comprehensively compare the full Claude API surface against the
existing test suite, write tests for any gaps, and flag deprecated features.

{SAFETY_RULES}

## Step 1: Read the latest API documentation

You have access to the claude-api skill which contains comprehensive
documentation of the Claude API surface. Load it and read:
- The main SKILL.md for an overview of all API features
- shared/tool-use-concepts.md for the full tool reference (all server-side
  tools, client tools, structured outputs, context editing, memory, etc.)
- shared/live-sources.md for URLs to the latest official documentation

Then use WebFetch to read these key documentation pages for the absolute
latest information:
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- https://platform.claude.com/docs/en/build-with-claude/context-editing
- https://platform.claude.com/docs/en/build-with-claude/compaction

Also use WebSearch to find any features announced in the past 7 days that
may not yet be in the documentation.

## Step 2: Inventory existing test coverage

For every `*.py` file in `tests/bedrock/`:
- Read the first 5 lines to get the ABOUTME description
- Note what API feature, tool, or capability it tests
- Note whether it's an invoke or converse variant

Build a complete map: feature → [invoke test, converse test].

## Step 3: Gap analysis

Compare the documented API features against the test inventory:

**Features to check for test coverage:**
- All server-side tools (web_search, code_execution, web_fetch, tool_search, etc.)
- All client-side Anthropic tools (bash, text_editor, memory, advisor, etc.)
- Extended/adaptive thinking
- Prompt caching (automatic, manual, extended TTL)
- Structured outputs (JSON schema, strict tool use)
- Context management (clearing thinking, clearing tool use)
- Compaction
- Parallel tool use
- Streaming (fine-grained tool streaming)
- Image handling (limits, formats)
- PDF handling (limits)
- Context window limits
- MCP connector
- Any other documented features

For each feature:
- If both invoke and converse tests exist → covered, skip
- If only one variant exists → gap, write the missing variant
- If no test exists → gap, write both variants
- If a test exists for a feature that appears deprecated → flag it

## Step 4: Write missing tests

For each gap found:
1. Read the official documentation for the feature (WebFetch if needed)
2. Use the bedrock-test-writer skill to create the test file(s)
3. Write to `tests/bedrock/` in the project root
4. Run from the project root: `cd <project_root> && uv run python tests/bedrock/<name>.py`
5. Fix any script bugs and rerun until the test executes cleanly
6. Add new tests to the Bedrock Tests table in `README.md`

{GIT_INSTRUCTIONS}

## Step 5: Report

Print a structured report with these sections:

- **API features audited**: total count of features checked
- **Coverage summary**: N features covered, M gaps found, K deprecated
- **Tests created**: file names and test results (PASS/FAIL/ERROR)
- **Git**: commit hash and push status (or "no changes" if nothing was created)
- **Gaps remaining**: features that couldn't be tested (explain why)
- **Deprecated features**: existing tests for features that appear to be
  deprecated or removed (file names + reason for flagging). REMINDER:
  do NOT delete or modify these files — only report them.
- **No gaps found**: if coverage is complete, say so
"""


def load_config():
    """Load config.yaml from the project root."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def refresh_claude_api_skill():
    """Pull the latest claude-api skill from the Anthropic skills repo on GitHub."""
    print("Refreshing claude-api skill from GitHub...")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    SKILLS_REPO_URL,
                    tmpdir,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

            subprocess.run(
                ["git", "sparse-checkout", "set", SKILLS_REPO_SKILL_PATH],
                check=True,
                capture_output=True,
                text=True,
                cwd=tmpdir,
                timeout=30,
            )

            source = Path(tmpdir) / SKILLS_REPO_SKILL_PATH
            if not source.exists():
                print(f"  WARNING: {SKILLS_REPO_SKILL_PATH} not found in repo")
                return False

            # Remove old copy and replace with fresh one
            if CLAUDE_API_SKILL_DIR.exists():
                shutil.rmtree(CLAUDE_API_SKILL_DIR)

            shutil.copytree(source, CLAUDE_API_SKILL_DIR)
            print(f"  Updated claude-api skill at {CLAUDE_API_SKILL_DIR}")
            return True

        except subprocess.TimeoutExpired:
            print("  WARNING: Git clone timed out — using existing skill if available")
            return False
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Git clone failed: {e.stderr.strip()}")
            return False
        except Exception as e:
            print(f"  WARNING: Failed to refresh skill: {type(e).__name__}: {e}")
            return False


def send_sns_notification(config, mode, report):
    """Publish the agent report to the configured SNS topic."""
    topic_arn = config.get("sns_topic_arn", "")
    sns_region = config.get("sns_region", config.get("region", "eu-west-1"))

    if not topic_arn:
        print("  SNS topic ARN not configured — skipping notification")
        return

    mode_label = "Daily Scout" if mode == "daily" else "Weekly Audit"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    subject = f"[Feature Scout] {mode_label} — {timestamp}"

    # SNS subjects are limited to 100 characters
    if len(subject) > 100:
        subject = subject[:97] + "..."

    # SNS messages are limited to 256KB
    message = report
    if len(message.encode("utf-8")) > 256 * 1024:
        message = message[:250000] + "\n\n[Report truncated due to SNS size limit]"

    try:
        session = boto3.session.Session(profile_name=config.get("aws_profile", "work"))
        client = session.client("sns", region_name=sns_region)
        client.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message,
        )
        print(f"  Notification sent to {topic_arn}")
    except ClientError as e:
        print(f"  WARNING: Failed to send SNS notification: {e}")
    except Exception as e:
        print(f"  WARNING: SNS error: {type(e).__name__}: {e}")


async def run_agent(mode, config):
    """Run the feature scout agent in the specified mode."""
    prompt = DAILY_PROMPT if mode == "daily" else WEEKLY_PROMPT

    # For weekly mode, refresh the claude-api skill from GitHub first
    if mode == "weekly":
        refresh_claude_api_skill()

    options = ClaudeAgentOptions(
        cwd=str(AGENT_DIR),
        add_dirs=[str(PROJECT_ROOT)],
        setting_sources=["project"],
        tools={"type": "preset", "preset": "claude_code"},
        allowed_tools=[
            "WebSearch",
            "WebFetch",
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "Skill",
        ],
        permission_mode="acceptEdits",
        max_budget_usd=5.0 if mode == "daily" else 10.0,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
        },
    )

    result_text = None

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result_text = message.result
            print("\n--- AGENT RESULT ---")
            print(result_text)
        elif hasattr(message, "type"):
            msg_type = getattr(message, "type", "unknown")
            if msg_type == "assistant":
                content = getattr(message, "content", [])
                for block in content:
                    if hasattr(block, "text"):
                        print(block.text)

    return result_text


def main():
    parser = argparse.ArgumentParser(description="Anthropic Feature Scout Agent")
    parser.add_argument(
        "--mode",
        choices=["daily", "weekly"],
        default="daily",
        help="Run mode: daily (news search) or weekly (full audit)",
    )
    args = parser.parse_args()

    config = load_config()
    mode_label = "DAILY SCOUT" if args.mode == "daily" else "WEEKLY AUDIT"

    print("=" * 70)
    print(f"  ANTHROPIC FEATURE SCOUT — {mode_label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project root: {PROJECT_ROOT}")
    print("=" * 70)
    print()

    result_text = asyncio.run(run_agent(args.mode, config))

    # Send notification
    if result_text:
        print("\n--- SENDING NOTIFICATION ---")
        send_sns_notification(config, args.mode, result_text)
    else:
        print("\n  Agent produced no result — skipping notification")

    print()
    print("=" * 70)
    print("  AGENT COMPLETE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return 0 if result_text else 1


if __name__ == "__main__":
    sys.exit(main())
