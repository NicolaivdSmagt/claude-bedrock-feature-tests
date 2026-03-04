#!/usr/bin/env python3
# ABOUTME: Test runner that uses the Claude Agent SDK to execute all Bedrock and Anthropic tests.
# ABOUTME: Generates a markdown report with result tables and summary statistics.

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

try:
    from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock
except ImportError:
    print("Error: claude-code-sdk not installed. Run: uv add claude-code-sdk")
    sys.exit(1)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_tests(suite: str) -> list[Path]:
    """Discover test scripts for the given suite (bedrock, anthropic, or all)."""
    base = Path(__file__).parent / "tests"
    tests = []

    suites = []
    if suite in ("bedrock", "all"):
        suites.append("bedrock")
    if suite in ("anthropic", "all"):
        suites.append("anthropic")

    for s in suites:
        suite_dir = base / s
        if suite_dir.exists():
            tests.extend(sorted(suite_dir.glob("*.py")))

    return tests


def build_prompt(
    suite: str, tests: list[Path], config: dict, timeout: int, verbose: bool
) -> str:
    """Build the prompt for the Claude agent."""
    test_list = "\n".join(f"  - {t.parent.name}/{t.name}" for t in tests)

    verbose_capture = ""
    if verbose:
        verbose_capture = (
            "\n2. For each test, capture the FULL stdout/stderr output (do NOT truncate with tail)."
            "\n   Save the complete output — it will be included in the report."
        )
    else:
        verbose_capture = (
            "\n2. For each test, capture:"
            "\n   - The exit code (0 = success, non-zero = failure)"
            "\n   - The last 30 lines of output (use `tail -30` via piping)"
            "\n   - Whether the test passed, failed, or had an error"
        )

    verbose_section = ""
    if verbose:
        verbose_section = """

## Raw Output

For each test, include a section with the full output:

### <test_name.py>

See `AGENTS_PRIVATE.md` for remote URLs and push instructions.
<full stdout/stderr output>
See `AGENTS_PRIVATE.md` for remote URLs and push instructions.

(Repeat for each test)
"""

    return f"""You are a test runner. Your job is to execute Python test scripts and produce a structured markdown report.

## Configuration
- Region: {config["region"]}
- Bedrock Model: {config["bedrock_model_id"]}
- Anthropic Model: {config.get("anthropic_model_id", "N/A")}
- AWS Profile: {config.get("aws_profile", "default")}
- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Scripts to Run
{test_list}

## Instructions

1. Run each test script listed above using: `AWS_PROFILE={config.get("aws_profile", "default")} timeout {timeout} uv run python <script_path>`
   Run them from the working directory: {Path(__file__).parent}
{verbose_capture}

3. Classify each test result as one of:
   - **PASS**: Exit code 0 and output indicates success (look for "PASS", "SUCCESS", "PASSED", correct results, or no errors)
   - **FAIL**: Exit code 0 but output shows test failures (look for "FAIL", "FAILED", expected failures like "does NOT work on Bedrock")
   - **FAIL (expected)**: The test demonstrates a known limitation (e.g., code_execution not working on Bedrock). Mark as "FAIL (expected)" and note why.
   - **ERROR**: Exit code non-zero, or Python traceback/exception in output

4. Write a concise "Notes" column (max ~80 chars) summarizing what happened.
   For errors, include the key error message in an "Errors" column.

5. After running ALL tests, produce EXACTLY the following markdown output (and nothing else before or after it):

See `AGENTS_PRIVATE.md` for remote URLs and push instructions.
# Test Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Suite:** {suite}
**Region:** {config["region"]}
**Bedrock Model:** {config["bedrock_model_id"]}
**Anthropic Model:** {config.get("anthropic_model_id", "N/A")}

## Results

| # | Test | Result | Notes | Errors |
|---|------|--------|-------|--------|
| 1 | test_name.py | PASS | Description of what happened | |
...

## Summary

| Status | Count |
|--------|-------|
| PASS | N |
| FAIL | N |
| FAIL (expected) | N |
| ERROR | N |
| **Total** | **N** |
{verbose_section}See `AGENTS_PRIVATE.md` for remote URLs and push instructions.

IMPORTANT RULES:
- Run ALL {len(tests)} tests. Do not skip any.
- Run tests ONE AT A TIME sequentially (they make API calls that could conflict).
- Use `timeout {timeout}` for each test to prevent hangs.
- The final output must be ONLY the markdown report, nothing else.
- Do not ask any questions. Just run the tests and produce the report.
"""


async def run_agent(prompt: str, cwd: str) -> str:
    """Run the Claude agent and collect its final text output."""
    options = ClaudeCodeOptions(
        allowed_tools=["Bash"],
        permission_mode="acceptEdits",
        cwd=cwd,
        max_turns=100,
    )

    final_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    final_text = block.text

    return final_text


def extract_markdown_report(text: str) -> str:
    """Extract the markdown report from the agent's output.

    The agent may produce intermediate text before the final report.
    We look for the report starting with '# Test Report'.
    """
    # Look for the report header
    marker = "# Test Report"
    idx = text.rfind(marker)
    if idx >= 0:
        return text[idx:]

    # Fallback: return everything
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Run API feature tests via Claude Agent SDK"
    )
    parser.add_argument(
        "--suite",
        choices=["bedrock", "anthropic", "all"],
        default="bedrock",
        help="Which test suite to run (default: bedrock)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds per test (default: 300)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save report to file (default: screen only). Use 'auto' for auto-generated filename.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include full raw output from each test in the report",
    )
    args = parser.parse_args()

    config = load_config()
    tests = discover_tests(args.suite)

    if not tests:
        print(f"No tests found for suite '{args.suite}'")
        sys.exit(1)

    print(f"Discovered {len(tests)} tests for suite '{args.suite}'")
    print(f"Region: {config['region']}")
    print(f"Bedrock Model: {config['bedrock_model_id']}")
    print(f"Timeout per test: {args.timeout}s")
    print()

    cwd = str(Path(__file__).parent)
    prompt = build_prompt(args.suite, tests, config, args.timeout, args.verbose)

    print("Starting Claude agent to run tests...")
    print("=" * 60)

    result_text = asyncio.run(run_agent(prompt, cwd))
    report = extract_markdown_report(result_text)

    print()
    print("=" * 60)

    # Save to file only if -o was specified
    if args.output:
        if args.output == "auto":
            reports_dir = Path(__file__).parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_slug = config["bedrock_model_id"].replace(".", "_").replace(":", "_")
            output_path = (
                reports_dir / f"report_{model_slug}_{config['region']}_{timestamp}.md"
            )
        else:
            output_path = Path(args.output)

        output_path.write_text(report)
        print(f"Report written to: {output_path}")

    print()
    print(report)


if __name__ == "__main__":
    main()
