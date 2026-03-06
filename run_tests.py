#!/usr/bin/env python3
# ABOUTME: Test runner that executes all test scripts via subprocess with live status output.
# ABOUTME: Generates a markdown report with result tables and summary statistics.

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from load_config import load_config

PROJECT_ROOT = Path(__file__).parent


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
    """Classify a test result into (status, notes, error_detail).

    Parses the test output for status information. Recognizes two formats:
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


def generate_report(
    results: list[dict], config: dict, suite: str, verbose: bool
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
            lines.append("See `AGENTS_PRIVATE.md` for remote URLs and push instructions.")
            output = r["stdout"]
            if r["stderr"]:
                output += "\n--- stderr ---\n" + r["stderr"]
            lines.append(output.rstrip())
            lines.append("See `AGENTS_PRIVATE.md` for remote URLs and push instructions.")
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

    # Run tests sequentially, print status as each completes
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

    # Print summary line
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

    # Generate report
    report = generate_report(results, config, args.suite, args.verbose)

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

    # Print full report if saving to file (user can review both)
    # If not saving, print the report to stdout
    if not args.output:
        print()
        print(report)

    sys.exit(0 if (fails == 0 and errors == 0) else 1)


if __name__ == "__main__":
    main()
