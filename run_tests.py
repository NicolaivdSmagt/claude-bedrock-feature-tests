#!/usr/bin/env python3
# ABOUTME: Test runner that executes tests via subprocess and uses Claude to classify results.
# ABOUTME: Generates a markdown report with result tables and summary statistics.

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from load_config import load_config, get_bedrock_client

PROJECT_ROOT = Path(__file__).parent

# Maximum characters of stdout/stderr to include per test in the LLM prompt.
# Keeps the total prompt size manageable while preserving the most relevant output
# (the end of test output contains the summary/verdict lines).
MAX_OUTPUT_CHARS_PER_TEST = 8000


def discover_tests(suite: str) -> list[Path]:
    """Discover test scripts for the given suite (bedrock, anthropic, or all)."""
    base = PROJECT_ROOT / "tests"
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


def classify_result(exit_code: int, stdout: str, stderr: str):
    """Classify a test result using regex-based heuristics.

    Returns (status, notes, error_detail). Used for live status output during
    test execution and as a fallback when LLM classification is unavailable.

    Recognizes two output formats:
    - Multi-test summary:  "Results: N PASS, N FAIL, N ERROR"
    - Single-verdict:      "Result: PASS" / "Result: FAIL" / "Result: ERROR"
    Falls back to exit code if neither is found.
    """
    summary_line = None
    single_verdict = None

    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        # Multi-test summary: "Results: N PASS, N FAIL, N ERROR"
        if (
            not summary_line
            and "Results:" in stripped
            and ("PASS" in stripped or "FAIL" in stripped)
        ):
            summary_line = stripped
        # Single-verdict: "Result: PASS - ..." or "Result: FAIL - ..."
        if not single_verdict and stripped.startswith("Result:"):
            single_verdict = stripped

    # Use multi-test summary if available, otherwise single verdict
    status_line = summary_line or single_verdict

    # Extract error detail from stderr or last non-trivial stdout line
    error_detail = ""
    if exit_code != 0 or (
        status_line and ("FAIL" in status_line or "ERROR" in status_line)
    ):
        for source in (stderr, stdout):
            for line in reversed(source.splitlines()):
                stripped = line.strip()
                if stripped and not stripped.startswith("="):
                    error_detail = stripped
                    break
            if error_detail:
                break

    if status_line:
        if summary_line:
            # Multi-test format: "Results: N PASS, N FAIL, N ERROR"
            has_fail = "FAIL" in summary_line and "0 FAIL" not in summary_line
            has_error = "ERROR" in summary_line and "0 ERROR" not in summary_line
            if has_fail:
                return ("FAIL", summary_line, error_detail)
            elif has_error:
                return ("ERROR", summary_line, error_detail)
            else:
                return ("PASS", summary_line, "")
        else:
            # Single-verdict format: "Result: PASS - ..." / "Result: FAIL - ..."
            if "FAIL" in single_verdict:
                return ("FAIL", single_verdict, error_detail)
            elif "ERROR" in single_verdict:
                return ("ERROR", single_verdict, error_detail)
            else:
                return ("PASS", single_verdict, "")

    # No status line found
    if exit_code != 0:
        return ("ERROR", "non-zero exit code", error_detail)

    return ("PASS", "exit code 0 (no summary line)", "")


def run_test(test_path: Path, timeout: int) -> dict:
    """Run a single test script and return structured results."""
    name = test_path.name
    suite = test_path.parent.name

    start = time.monotonic()
    try:
        result = subprocess.run(
            ["uv", "run", "python", str(test_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.monotonic() - start
        status, notes, error_detail = classify_result(
            result.returncode, result.stdout, result.stderr
        )
        return {
            "name": name,
            "suite": suite,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed": elapsed,
            "status": status,
            "notes": notes,
            "error_detail": error_detail,
        }
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        return {
            "name": name,
            "suite": suite,
            "exit_code": -1,
            "stdout": (e.stdout or b"").decode(errors="replace")
            if isinstance(e.stdout, bytes)
            else (e.stdout or ""),
            "stderr": (e.stderr or b"").decode(errors="replace")
            if isinstance(e.stderr, bytes)
            else (e.stderr or ""),
            "elapsed": elapsed,
            "status": "ERROR",
            "notes": f"timeout after {timeout}s",
            "error_detail": f"Process killed after {timeout}s timeout",
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "name": name,
            "suite": suite,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "elapsed": elapsed,
            "status": "ERROR",
            "notes": "failed to execute",
            "error_detail": f"{type(e).__name__}: {e}",
        }


def truncate_output(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, keeping the end (where verdicts are).

    If truncation occurs, prepends a marker showing how many characters were
    omitted so the reader knows context was removed.
    """
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"[...{omitted} chars omitted...]\n" + text[-max_chars:]


def build_llm_prompt(results: list[dict]) -> str:
    """Build the classification prompt containing all test outputs."""
    test_blocks = []
    for r in results:
        stdout = truncate_output(r["stdout"], MAX_OUTPUT_CHARS_PER_TEST)
        stderr = (
            truncate_output(r["stderr"], MAX_OUTPUT_CHARS_PER_TEST)
            if r["stderr"]
            else ""
        )

        block = f"### {r['name']}\n"
        block += f"Exit code: {r['exit_code']}\n"
        block += f"Elapsed: {r['elapsed']:.1f}s\n"
        block += f"\n--- stdout ---\n{stdout}\n"
        if stderr:
            block += f"\n--- stderr ---\n{stderr}\n"
        test_blocks.append(block)

    test_output_section = "\n".join(test_blocks)

    return f"""You are a test result classifier. Below are the outputs from {len(results)} API feature test scripts. Each test validates a specific Claude API feature on Amazon Bedrock or the Anthropic API.

Your job is to read each test's output carefully and classify it.

## Classification Rules

- **PASS**: The test ran successfully and the feature works as expected. Look for "PASS" verdicts, successful API responses, correct behavior, and exit code 0.
- **FAIL**: The test ran to completion but the feature did not work as expected. This includes:
  - Tests that print "FAIL" verdicts
  - Features that are "not supported" or "not currently supported" on Bedrock (this is a feature availability failure, not an infrastructure error)
  - API validation errors indicating the feature is not available (e.g., "The provided request is not valid", "does not match any of the expected tags")
  - Tests where expected behavior was not observed
- **ERROR**: The test could not execute properly due to infrastructure issues. This includes:
  - Timeouts (process killed)
  - Authentication failures
  - Network errors
  - Python crashes / unhandled exceptions unrelated to the feature under test
  - Import errors

Important: If the API returns an error saying a feature is "not supported" or "not available", that is a FAIL (feature doesn't work), NOT an ERROR (infrastructure problem). ERROR is reserved for cases where the test itself couldn't run.

## Notes Guidelines

Write a concise (max 100 chars) summary of what happened. For PASS results, briefly describe what was verified. For FAIL results, explain what didn't work. For ERROR results, explain what went wrong with execution.

## Errors Guidelines

For FAIL and ERROR results, include the key error message or reason. Leave empty for PASS results.

## Test Outputs

{test_output_section}

## Response Format

Respond with ONLY a JSON array. No other text before or after. Each element must have exactly these fields:
- "test": the test filename (string)
- "status": "PASS", "FAIL", or "ERROR" (string)
- "notes": concise description of what happened (string, max 100 chars)
- "errors": key error message for FAIL/ERROR, empty string for PASS (string)

Example:
[
  {{"test": "example_test.py", "status": "PASS", "notes": "Feature X works correctly, 3 sub-tests passed", "errors": ""}},
  {{"test": "another_test.py", "status": "FAIL", "notes": "Feature Y not supported on Bedrock", "errors": "ValidationException: not currently supported"}}
]"""


def llm_classify(results: list[dict], config: dict) -> list[dict] | None:
    """Send test outputs to Claude via Bedrock for classification.

    Returns a list of dicts with keys: test, status, notes, errors.
    Returns None if the API call fails.
    """
    try:
        client = get_bedrock_client(config)
        model_id = config["bedrock_model_id"]
        prompt = build_llm_prompt(results)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        response_body = json.loads(response["body"].read())

        # Extract text content from response
        text = ""
        for block in response_body.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        # Parse JSON from response — strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
            # Remove closing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()

        classifications = json.loads(text)

        # Validate structure
        if not isinstance(classifications, list):
            print("  WARNING: LLM returned non-list JSON, falling back to regex")
            return None

        for item in classifications:
            if not all(k in item for k in ("test", "status", "notes", "errors")):
                print("  WARNING: LLM returned malformed item, falling back to regex")
                return None
            if item["status"] not in ("PASS", "FAIL", "ERROR"):
                print(
                    f"  WARNING: LLM returned invalid status '{item['status']}', falling back to regex"
                )
                return None

        return classifications

    except json.JSONDecodeError as e:
        print(f"  WARNING: Failed to parse LLM JSON response: {e}")
        return None
    except Exception as e:
        print(f"  WARNING: LLM classification failed: {type(e).__name__}: {e}")
        return None


def apply_llm_classifications(results: list[dict], classifications: list[dict]):
    """Update results list with LLM classifications.

    Matches by test filename. If a test is missing from the LLM output,
    its regex-based classification is kept.
    """
    llm_map = {c["test"]: c for c in classifications}

    for r in results:
        if r["name"] in llm_map:
            c = llm_map[r["name"]]
            r["status"] = c["status"]
            r["notes"] = c["notes"]
            r["error_detail"] = c["errors"]


def generate_report(
    results: list[dict],
    config: dict,
    suite: str,
    verbose: bool,
    classifier: str = "regex",
) -> str:
    """Generate a markdown report from test results."""
    lines = []
    lines.append("# Test Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Suite:** {suite}")
    lines.append(f"**Region:** {config['region']}")
    lines.append(f"**Bedrock Model:** {config['bedrock_model_id']}")
    lines.append(f"**Anthropic Model:** {config.get('anthropic_model_id', 'N/A')}")
    lines.append(f"**Classifier:** {classifier}")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("| # | Test | Result | Time | Notes | Errors |")
    lines.append("|---|------|--------|------|-------|--------|")

    for i, r in enumerate(results, 1):
        elapsed_str = f"{r['elapsed']:.1f}s"
        notes = r["notes"].replace("|", "\\|")
        error = r["error_detail"].replace("|", "\\|") if r["error_detail"] else ""
        lines.append(
            f"| {i} | {r['name']} | {r['status']} | {elapsed_str} | {notes} | {error} |"
        )

    lines.append("")

    # Summary table
    passes = sum(1 for r in results if r["status"] == "PASS")
    fails = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    total = len(results)

    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| PASS | {passes} |")
    lines.append(f"| FAIL | {fails} |")
    lines.append(f"| ERROR | {errors} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    total_time = sum(r["elapsed"] for r in results)
    lines.append(f"**Total time:** {total_time:.1f}s")
    lines.append("")

    # Verbose raw output
    if verbose:
        lines.append("## Raw Output")
        lines.append("")
        for r in results:
            lines.append(f"### {r['name']}")
            lines.append("")
            lines.append("```")
            output = r["stdout"]
            if r["stderr"]:
                output += "\n--- stderr ---\n" + r["stderr"]
            lines.append(output.rstrip())
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run API feature tests via subprocess")
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
        help="Save report to file. Use 'auto' for auto-generated filename.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include full raw output from each test in the report",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM classification; use only regex-based heuristics",
    )
    args = parser.parse_args()

    config = load_config()
    tests = discover_tests(args.suite)

    if not tests:
        print(f"No tests found for suite '{args.suite}'")
        sys.exit(1)

    total = len(tests)
    print(f"Region: {config['region']}")
    print(f"Model:  {config['bedrock_model_id']}")
    print(f"Timeout: {args.timeout}s per test")
    print()
    print(f"Running {total} {args.suite} tests...")
    print()

    # Phase 1: Run tests sequentially, print live status using regex classifier
    results = []
    name_width = max(len(t.name) for t in tests)

    for i, test_path in enumerate(tests, 1):
        r = run_test(test_path, args.timeout)
        results.append(r)

        elapsed_str = f"{r['elapsed']:.1f}s"
        line = f"  [{i:>{len(str(total))}}/{total}]  {r['status']:<5}  {r['name']:<{name_width}}  {elapsed_str:>7}"
        if r["status"] == "ERROR" and "timeout" in r["notes"]:
            line += "  (timeout)"
        print(line)

    # Print interim summary from regex classification
    passes = sum(1 for r in results if r["status"] == "PASS")
    fails = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    total_time = sum(r["elapsed"] for r in results)

    print()
    print("=" * 60)
    print(
        f"{total} tests: {passes} PASS, {fails} FAIL, {errors} ERROR  "
        f"({total_time:.1f}s)"
    )
    print("=" * 60)

    # Phase 2: LLM classification (unless --no-llm)
    classifier = "regex"
    if not args.no_llm:
        print()
        print("Classifying results with Claude...")
        classifications = llm_classify(results, config)
        if classifications:
            apply_llm_classifications(results, classifications)
            classifier = "llm"

            # Recount after LLM classification
            passes = sum(1 for r in results if r["status"] == "PASS")
            fails = sum(1 for r in results if r["status"] == "FAIL")
            errors = sum(1 for r in results if r["status"] == "ERROR")

            print(f"  Done — {passes} PASS, {fails} FAIL, {errors} ERROR")
        else:
            print("  Falling back to regex classification")

    # Phase 3: Generate report
    report = generate_report(results, config, args.suite, args.verbose, classifier)

    # Save to file if requested
    if args.output:
        if args.output == "auto":
            reports_dir = PROJECT_ROOT / "reports"
            reports_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_slug = config["bedrock_model_id"].replace(".", "_").replace(":", "_")
            output_path = (
                reports_dir / f"report_{model_slug}_{config['region']}_{timestamp}.md"
            )
        else:
            output_path = Path(args.output)

        output_path.write_text(report)
        print(f"\nReport written to: {output_path}")

    # Print full report if not saving to file
    if not args.output:
        print()
        print(report)

    sys.exit(0 if (fails == 0 and errors == 0) else 1)


if __name__ == "__main__":
    main()
