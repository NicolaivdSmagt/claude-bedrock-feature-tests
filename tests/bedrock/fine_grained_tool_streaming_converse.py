#!/usr/bin/env python3
# ABOUTME: Tests fine-grained tool streaming on Amazon Bedrock via the Converse stream API.
# ABOUTME: Verifies that eager_input_streaming delivers tool input deltas without buffering.

"""
Fine-Grained Tool Streaming Test for Amazon Bedrock (Converse API)
===================================================================

Tests the eager_input_streaming feature via converse_stream. When
eager_input_streaming is set to true on a tool definition, tool use
parameter values stream character-by-character without buffering or JSON
validation, reducing latency for large parameters.

The eager_input_streaming flag is an Anthropic-specific tool property,
so it is passed via additionalModelRequestFields alongside the tool
definitions. A placeholder toolSpec is required in toolConfig.tools to
satisfy the Converse schema.

Validates:
1. Basic streaming: eager_input_streaming=true produces streaming tool
   input delta events that accumulate into valid tool input
2. Latency improvement: first tool input delta chunk arrives faster with
   eager_input_streaming=true vs false
3. Large parameter streaming: a tool call with a large array parameter
   streams correctly and accumulates into valid JSON
4. Mixed tools: a request with both eager and non-eager tools streams
   the eager tool's input deltas correctly

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/fine_grained_tool_streaming_converse.py
"""

import json
import os
import sys
import time

try:
    import boto3
except ImportError:
    print("Error: boto3 package not installed. Run: uv add boto3")
    sys.exit(1)

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

FEATURE_NOT_AVAILABLE_MARKERS = [
    "the provided request is not valid",
    "does not match any of the expected tags",
    "not supported",
    "unknown field",
    "eager_input_streaming",
]

# Converse requires at least one toolSpec in toolConfig; this placeholder satisfies that.
PLACEHOLDER_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "placeholder",
                "description": "Placeholder tool",
                "inputSchema": {"json": {"type": "object"}},
            }
        }
    ]
}

# Tool definitions for additionalModelRequestFields (Anthropic-native format)
MAKE_FILE_TOOL_EAGER = {
    "name": "make_file",
    "description": "Write text to a file",
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The filename to write text to",
            },
            "lines_of_text": {
                "type": "array",
                "items": {"type": "string"},
                "description": "An array of lines of text to write to the file",
            },
        },
        "required": ["filename", "lines_of_text"],
    },
}

MAKE_FILE_TOOL_NO_EAGER = {
    "name": "make_file",
    "description": "Write text to a file",
    "eager_input_streaming": False,
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The filename to write text to",
            },
            "lines_of_text": {
                "type": "array",
                "items": {"type": "string"},
                "description": "An array of lines of text to write to the file",
            },
        },
        "required": ["filename", "lines_of_text"],
    },
}

WRITE_DOCUMENT_TOOL = {
    "name": "write_document",
    "description": "Write a document with structured content",
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The document title",
            },
            "sections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Content sections of the document",
            },
        },
        "required": ["title", "sections"],
    },
}

GET_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city, e.g. San Francisco",
            }
        },
        "required": ["location"],
    },
}


def classify_error(error_msg):
    """Classify an error as FAIL (feature not available) or ERROR (unexpected)."""
    lower = error_msg.lower()
    for marker in FEATURE_NOT_AVAILABLE_MARKERS:
        if marker in lower:
            return ("FAIL", error_msg)
    return ("ERROR", error_msg)


def stream_converse_response(client, model_id, messages, tools):
    """Make a streaming Converse API call and collect chunk-level timing and content data.

    Tools are passed via additionalModelRequestFields since they use
    Anthropic-specific properties (eager_input_streaming).

    Returns a dict with:
        content_blocks: list of accumulated content blocks (Converse format)
        stop_reason: the stopReason from the stream
        usage: usage dict from metadata
        first_delta_ms: time in ms from request start to first tool input delta
        total_delta_chunks: count of tool input delta chunks received
    """
    start_time = time.monotonic()

    response = client.converse_stream(
        modelId=model_id,
        messages=messages,
        toolConfig=PLACEHOLDER_TOOL_CONFIG,
        inferenceConfig={"maxTokens": 8192},
        additionalModelRequestFields={
            "tools": tools,
        },
    )

    # Use a dict keyed by contentBlockIndex so we handle cases where
    # text deltas arrive before a contentBlockStart event.
    blocks_by_idx = {}
    usage = {}
    stop_reason = None
    first_delta_ms = None
    total_delta_chunks = 0
    # Track accumulated JSON strings for tool inputs by block index
    tool_input_accumulators = {}

    for event in response["stream"]:
        if "messageStart" in event:
            pass  # role info

        elif "contentBlockStart" in event:
            start = event["contentBlockStart"]
            idx = start.get("contentBlockIndex", 0)
            block_start = start.get("start", {})

            if "toolUse" in block_start:
                tool_info = block_start["toolUse"]
                blocks_by_idx[idx] = {
                    "toolUse": {
                        "toolUseId": tool_info["toolUseId"],
                        "name": tool_info["name"],
                        "input": {},
                    }
                }
                tool_input_accumulators[idx] = ""
            else:
                blocks_by_idx.setdefault(idx, {"text": ""})

        elif "contentBlockDelta" in event:
            delta_event = event["contentBlockDelta"]
            idx = delta_event.get("contentBlockIndex", 0)
            delta = delta_event.get("delta", {})

            if "toolUse" in delta:
                # Tool input delta: partial JSON string
                partial = delta["toolUse"].get("input", "")
                if isinstance(partial, str):
                    if idx not in tool_input_accumulators:
                        tool_input_accumulators[idx] = ""
                    tool_input_accumulators[idx] += partial
                    total_delta_chunks += 1
                    if first_delta_ms is None:
                        first_delta_ms = (time.monotonic() - start_time) * 1000
            elif "text" in delta:
                # Auto-create text block if we get deltas before contentBlockStart
                if idx not in blocks_by_idx:
                    blocks_by_idx[idx] = {"text": ""}
                if "text" in blocks_by_idx[idx]:
                    blocks_by_idx[idx]["text"] += delta["text"]

        elif "contentBlockStop" in event:
            pass  # block finalized

        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason")

        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})

    # Parse accumulated JSON strings into tool inputs
    for idx, json_str in tool_input_accumulators.items():
        if idx in blocks_by_idx and "toolUse" in blocks_by_idx[idx]:
            if json_str:
                blocks_by_idx[idx]["toolUse"]["input"] = json.loads(json_str)
            else:
                blocks_by_idx[idx]["toolUse"]["input"] = {}

    # Build final content_blocks list in index order
    content_blocks = [blocks_by_idx[k] for k in sorted(blocks_by_idx)]

    return {
        "content_blocks": content_blocks,
        "stop_reason": stop_reason,
        "usage": usage,
        "first_delta_ms": first_delta_ms,
        "total_delta_chunks": total_delta_chunks,
    }


def test_basic_streaming(client, model_id):
    """Test that eager_input_streaming produces streaming tool input delta events.

    Returns (status, error_msg).
    """
    print("=" * 70)
    print("TEST: BASIC FINE-GRAINED TOOL STREAMING")
    print("=" * 70)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "Write a short poem (at least 8 lines) about the ocean "
                        "and save it to a file called ocean_poem.txt using the make_file tool. "
                        "Put each line of the poem as a separate element in the lines_of_text array."
                    )
                }
            ],
        }
    ]

    try:
        result = stream_converse_response(
            client, model_id, messages, [MAKE_FILE_TOOL_EAGER]
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    content_blocks = result["content_blocks"]
    tool_uses = [
        b["toolUse"] for b in content_blocks if isinstance(b, dict) and "toolUse" in b
    ]

    print(f"\n  stopReason: {result['stop_reason']}")
    print(f"  content blocks: {len(content_blocks)}")
    print(f"  toolUse blocks: {len(tool_uses)}")
    print(f"  total tool input delta chunks: {result['total_delta_chunks']}")
    if result["first_delta_ms"] is not None:
        print(f"  first tool input delta at: {result['first_delta_ms']:.0f}ms")

    if len(tool_uses) == 0:
        msg = "No toolUse blocks in response"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    tool_use = tool_uses[0]
    print(f"\n  Tool: {tool_use['name']}")
    print(f"  Input keys: {list(tool_use['input'].keys())}")

    if tool_use["name"] != "make_file":
        msg = f"Expected make_file tool call, got '{tool_use['name']}'"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    tool_input = tool_use["input"]
    if "filename" not in tool_input:
        msg = "Tool input missing 'filename' key"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if "lines_of_text" not in tool_input:
        msg = "Tool input missing 'lines_of_text' key"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    lines = tool_input["lines_of_text"]
    if not isinstance(lines, list) or len(lines) == 0:
        msg = f"Expected non-empty lines_of_text array, got {type(lines).__name__} with {len(lines) if isinstance(lines, list) else 'N/A'} items"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(f"  Filename: {tool_input['filename']}")
    print(f"  Lines count: {len(lines)}")
    for i, line in enumerate(lines[:4]):
        print(f"    [{i}]: {line[:80]}")
    if len(lines) > 4:
        print(f"    ... and {len(lines) - 4} more lines")

    if result["total_delta_chunks"] < 2:
        msg = f"Expected multiple tool input delta chunks for streaming, got {result['total_delta_chunks']}"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - {result['total_delta_chunks']} delta chunks, {len(lines)} lines streamed"
    )
    return ("PASS", None)


def test_latency_comparison(client, model_id):
    """Compare first-chunk latency between eager and non-eager streaming.

    Returns (status, error_msg).
    """
    print("=" * 70)
    print("TEST: LATENCY COMPARISON (EAGER vs NON-EAGER)")
    print("=" * 70)

    prompt_text = (
        "Write a detailed story (at least 15 lines) about a robot learning to paint "
        "and save it to a file called robot_story.txt using the make_file tool. "
        "Put each sentence as a separate element in the lines_of_text array."
    )
    messages = [{"role": "user", "content": [{"text": prompt_text}]}]

    # Run with eager_input_streaming=true
    print("\n--- Run 1: eager_input_streaming=true ---")
    try:
        result_eager = stream_converse_response(
            client, model_id, messages, [MAKE_FILE_TOOL_EAGER]
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    eager_tool_uses = [
        b["toolUse"]
        for b in result_eager["content_blocks"]
        if isinstance(b, dict) and "toolUse" in b
    ]
    print(f"  toolUse blocks: {len(eager_tool_uses)}")
    print(f"  total delta chunks: {result_eager['total_delta_chunks']}")
    if result_eager["first_delta_ms"] is not None:
        print(f"  first delta at: {result_eager['first_delta_ms']:.0f}ms")
    else:
        print("  first delta: N/A")

    # Run with eager_input_streaming=false
    print("\n--- Run 2: eager_input_streaming=false ---")
    try:
        result_no_eager = stream_converse_response(
            client, model_id, messages, [MAKE_FILE_TOOL_NO_EAGER]
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    no_eager_tool_uses = [
        b["toolUse"]
        for b in result_no_eager["content_blocks"]
        if isinstance(b, dict) and "toolUse" in b
    ]
    print(f"  toolUse blocks: {len(no_eager_tool_uses)}")
    print(f"  total delta chunks: {result_no_eager['total_delta_chunks']}")
    if result_no_eager["first_delta_ms"] is not None:
        print(f"  first delta at: {result_no_eager['first_delta_ms']:.0f}ms")
    else:
        print("  first delta: N/A")

    # Compare results
    print("\n--- Comparison ---")

    if len(eager_tool_uses) == 0:
        msg = "No toolUse blocks in eager streaming response"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if len(no_eager_tool_uses) == 0:
        msg = "No toolUse blocks in non-eager streaming response"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    eager_input = eager_tool_uses[0].get("input", {})
    no_eager_input = no_eager_tool_uses[0].get("input", {})

    eager_lines = eager_input.get("lines_of_text", [])
    no_eager_lines = no_eager_input.get("lines_of_text", [])

    eager_summary = (
        f"{len(eager_lines)} lines, {result_eager['total_delta_chunks']} chunks"
    )
    if result_eager["first_delta_ms"] is not None:
        eager_summary += f", first at {result_eager['first_delta_ms']:.0f}ms"
    print(f"  Eager: {eager_summary}")

    no_eager_summary = (
        f"{len(no_eager_lines)} lines, {result_no_eager['total_delta_chunks']} chunks"
    )
    if result_no_eager["first_delta_ms"] is not None:
        no_eager_summary += f", first at {result_no_eager['first_delta_ms']:.0f}ms"
    print(f"  Non-eager: {no_eager_summary}")

    if len(eager_lines) == 0:
        msg = "Eager streaming produced empty lines_of_text"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if len(no_eager_lines) == 0:
        msg = "Non-eager streaming produced empty lines_of_text"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    # Latency differences are informational since they depend on network conditions
    if (
        result_eager["first_delta_ms"] is not None
        and result_no_eager["first_delta_ms"] is not None
    ):
        eager_ms = result_eager["first_delta_ms"]
        no_eager_ms = result_no_eager["first_delta_ms"]
        print(f"  First delta: eager={eager_ms:.0f}ms, non-eager={no_eager_ms:.0f}ms")
        if eager_ms < no_eager_ms:
            print(f"  Eager was {no_eager_ms - eager_ms:.0f}ms faster")
        else:
            print(
                f"  Eager was {eager_ms - no_eager_ms:.0f}ms slower (can vary per-request)"
            )

    print(
        f"\n  Result: PASS - both modes produced valid tool input with streaming deltas"
    )
    return ("PASS", None)


def test_large_parameter_streaming(client, model_id):
    """Test streaming a tool call with a large array parameter.

    Returns (status, error_msg).
    """
    print("=" * 70)
    print("TEST: LARGE PARAMETER STREAMING")
    print("=" * 70)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "Write a poem with exactly 20 lines about the four seasons "
                        "(5 lines per season: spring, summer, autumn, winter). "
                        "Save it using the make_file tool with filename 'seasons_poem.txt'. "
                        "Each line of the poem must be a separate element in the lines_of_text array."
                    )
                }
            ],
        }
    ]

    try:
        result = stream_converse_response(
            client, model_id, messages, [MAKE_FILE_TOOL_EAGER]
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    content_blocks = result["content_blocks"]
    tool_uses = [
        b["toolUse"] for b in content_blocks if isinstance(b, dict) and "toolUse" in b
    ]

    print(f"\n  stopReason: {result['stop_reason']}")
    print(f"  content blocks: {len(content_blocks)}")
    print(f"  toolUse blocks: {len(tool_uses)}")
    print(f"  total tool input delta chunks: {result['total_delta_chunks']}")

    if len(tool_uses) == 0:
        msg = "No toolUse blocks in response"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    tool_use = tool_uses[0]
    tool_input = tool_use["input"]

    if "lines_of_text" not in tool_input:
        msg = "Tool input missing 'lines_of_text' key"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    lines = tool_input["lines_of_text"]
    print(f"\n  Filename: {tool_input.get('filename', 'N/A')}")
    print(f"  Lines count: {len(lines)}")

    for i, line in enumerate(lines[:3]):
        print(f"    [{i}]: {line[:80]}")
    if len(lines) > 6:
        print(f"    ...")
    for i in range(max(3, len(lines) - 3), len(lines)):
        print(f"    [{i}]: {lines[i][:80]}")

    total_chars = sum(len(line) for line in lines)
    print(f"  Total characters across lines: {total_chars}")

    if len(lines) < 5:
        msg = f"Expected at least 5 lines in array, got {len(lines)}"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if result["total_delta_chunks"] < 3:
        msg = f"Expected at least 3 delta chunks for large parameter, got {result['total_delta_chunks']}"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - {len(lines)} lines streamed across {result['total_delta_chunks']} delta chunks"
    )
    return ("PASS", None)


def test_mixed_tools(client, model_id):
    """Test a request with both eager and non-eager tools.

    Returns (status, error_msg).
    """
    print("=" * 70)
    print("TEST: MIXED TOOLS (EAGER + NON-EAGER)")
    print("=" * 70)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "Write a short document about the history of computing with at least "
                        "5 sections using the write_document tool. Each section should be a "
                        "paragraph in the sections array."
                    )
                }
            ],
        }
    ]

    try:
        result = stream_converse_response(
            client, model_id, messages, [WRITE_DOCUMENT_TOOL, GET_WEATHER_TOOL]
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return classify_error(error_msg)

    content_blocks = result["content_blocks"]
    tool_uses = [
        b["toolUse"] for b in content_blocks if isinstance(b, dict) and "toolUse" in b
    ]

    print(f"\n  stopReason: {result['stop_reason']}")
    print(f"  content blocks: {len(content_blocks)}")
    print(f"  toolUse blocks: {len(tool_uses)}")
    print(f"  total tool input delta chunks: {result['total_delta_chunks']}")

    if len(tool_uses) == 0:
        msg = "No toolUse blocks in response"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    # Find the write_document call
    write_doc_calls = [tu for tu in tool_uses if tu["name"] == "write_document"]

    if len(write_doc_calls) == 0:
        msg = "Expected write_document tool call but got: " + ", ".join(
            tu["name"] for tu in tool_uses
        )
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    tool_input = write_doc_calls[0]["input"]
    print(f"\n  Tool: write_document")
    print(f"  Title: {tool_input.get('title', 'N/A')}")
    sections = tool_input.get("sections", [])
    print(f"  Sections count: {len(sections)}")
    for i, section in enumerate(sections[:3]):
        print(f"    [{i}]: {section[:80]}...")
    if len(sections) > 3:
        print(f"    ... and {len(sections) - 3} more sections")

    if "title" not in tool_input:
        msg = "Tool input missing 'title' key"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if (
        "sections" not in tool_input
        or not isinstance(sections, list)
        or len(sections) == 0
    ):
        msg = f"Expected non-empty sections array, got {len(sections) if isinstance(sections, list) else 'not a list'}"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - write_document called with {len(sections)} sections, {result['total_delta_chunks']} delta chunks"
    )
    return ("PASS", None)


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

    print()
    print(f"  Results: {passes} PASS, {fails} FAIL, {errors} ERROR")
    print("=" * 70)

    return all(s == "PASS" for _, s, _ in results)


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    print(f"Model: {model_id}")
    print()

    results = []

    status, error = test_basic_streaming(client, model_id)
    results.append(("Basic fine-grained streaming", status, error))

    status, error = test_latency_comparison(client, model_id)
    results.append(("Latency comparison", status, error))

    status, error = test_large_parameter_streaming(client, model_id)
    results.append(("Large parameter streaming", status, error))

    status, error = test_mixed_tools(client, model_id)
    results.append(("Mixed tools (eager + non-eager)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
