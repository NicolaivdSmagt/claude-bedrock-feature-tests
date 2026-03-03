#!/usr/bin/env python3
# ABOUTME: A dummy MCP server with many tools to trigger tool search in Claude Code.
# ABOUTME: Exposes ~100 tools with verbose descriptions to exceed 20K token threshold.

import json
import sys


def generate_tools():
    """Generate ~100 tools with verbose descriptions to consume context window."""
    tools = []
    categories = [
        ("database", "database management and SQL operations"),
        ("filesystem", "file system operations and path manipulation"),
        ("network", "network analysis and HTTP requests"),
        ("math", "mathematical computations and statistical analysis"),
        ("string", "string manipulation and text processing"),
        ("date", "date and time operations"),
        ("api", "external API integrations and data fetching"),
        ("crypto", "cryptography and security operations"),
        ("image", "image processing and analysis"),
        ("audio", "audio processing and transcription"),
    ]

    operations = [
        (
            "query",
            "Performs complex parameterized queries with extensive validation and result formatting. Supports pagination, filtering, sorting, and aggregate operations. Returns structured data with metadata.",
        ),
        (
            "create",
            "Creates new resources with comprehensive input validation, schema enforcement, and error handling. Supports batch operations and transaction management with rollback capabilities.",
        ),
        (
            "update",
            "Updates existing resources with optimistic locking, conflict detection, and audit trail generation. Supports partial updates and field-level modifications with version control.",
        ),
        (
            "delete",
            "Permanently removes resources with soft-delete options, cascading deletions, and referential integrity checks. Includes undo functionality within a configurable time window.",
        ),
        (
            "validate",
            "Validates data against complex schemas with custom validation rules, custom error messages, and detailed validation reports. Supports nested object validation and cross-field dependencies.",
        ),
        (
            "search",
            "Performs full-text search with fuzzy matching, relevance scoring, and result highlighting. Supports advanced query syntax, filters, faceted search, and personalization based on user context.",
        ),
        (
            "analyze",
            "Conducts deep analysis using statistical methods, machine learning models, and pattern recognition. Returns comprehensive reports with visualizations, confidence scores, and actionable insights.",
        ),
        (
            "transform",
            "Transforms data formats with schema mapping, data type conversion, and encoding changes. Supports complex transformations with conditional logic, data enrichment, and quality checks.",
        ),
        (
            "export",
            "Exports data to various formats including JSON, CSV, XML, Parquet, and custom formats. Supports streaming for large datasets, compression, encryption, and incremental exports.",
        ),
        (
            "import",
            "Imports data from external sources with format detection, schema inference, and data cleansing. Validates against target schemas, handles errors gracefully, and provides detailed import reports.",
        ),
    ]

    for cat_idx, (category, cat_desc) in enumerate(categories):
        for op_idx, (operation, op_desc) in enumerate(operations):
            tool_num = cat_idx * 10 + op_idx + 1

            # Generate verbose tool definition
            description = (
                f"Comprehensive {operation} operation for {cat_desc}. This tool provides enterprise-grade functionality "
                f"with extensive configurability. {op_desc} Designed for production use with proper error handling, "
                f"logging, and monitoring. Supports rate limiting, authentication, and authorization checks. "
                f"Includes comprehensive documentation and examples for all parameters. Thread-safe and async-compatible. "
                f"Supports distributed tracing and OpenTelemetry integration. Compatible with cloud-native architectures "
                f"and containerized deployments. Tool number {tool_num} of 100 in the comprehensive toolkit."
            )

            tool = {
                "name": f"{category}_{operation}",
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": f"Primary input data for the {operation} operation on {category}. Accepts various formats including raw data, file paths, or resource identifiers. Minimum length: 1. Maximum length: 1000000 characters. Supports unicode encoding and special characters.",
                        },
                        "options": {
                            "type": "object",
                            "description": f"Configuration options for customizing the {operation} behavior. Supports fine-grained control over processing parameters, output formats, and operational modes.",
                            "properties": {
                                "mode": {
                                    "type": "string",
                                    "enum": [
                                        "standard",
                                        "strict",
                                        "lenient",
                                        "performance",
                                        "accuracy",
                                    ],
                                    "description": "Operational mode selection: standard (balanced), strict (maximum validation), lenient (permissive), performance (optimized for speed), or accuracy (optimized for precision).",
                                },
                                "timeout": {
                                    "type": "number",
                                    "description": "Maximum execution time in seconds. Range: 1-3600. Default varies by operation complexity.",
                                    "minimum": 1,
                                    "maximum": 3600,
                                },
                                "retries": {
                                    "type": "integer",
                                    "description": "Number of retry attempts on transient failures. Range: 0-10. Default: 3.",
                                    "minimum": 0,
                                    "maximum": 10,
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["json", "xml", "yaml", "csv", "binary"],
                                    "description": "Output format specification. Options include structured formats (JSON, XML, YAML) and tabular formats (CSV). Binary format available for specific operations.",
                                },
                            },
                        },
                        "metadata": {
                            "type": "object",
                            "description": f"Additional metadata for the {operation} operation including request tracing, user context, and audit information.",
                            "properties": {
                                "request_id": {
                                    "type": "string",
                                    "description": "Unique identifier for tracking this request through the system. Used for logging, monitoring, and debugging purposes.",
                                },
                                "user_id": {
                                    "type": "string",
                                    "description": "Identifier of the user initiating this operation. Used for access control, rate limiting, and audit trail generation.",
                                },
                                "timestamp": {
                                    "type": "string",
                                    "description": "ISO 8601 timestamp indicating when this operation was requested. Used for temporal ordering and scheduling.",
                                },
                            },
                        },
                    },
                    "required": ["input"],
                },
            }
            tools.append(tool)

    return tools


def handle_request(request):
    """Handle incoming MCP requests."""
    method = request.get("method")

    if method == "initialize":
        # Negotiate protocol version - return what client supports
        client_info = request.get("params", {})
        client_version = client_info.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": client_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "dummy-tool-server", "version": "1.0.0"},
            },
        }

    elif method == "tools/list":
        tools = generate_tools()
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": {"tools": tools}}

    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "unknown")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": f"Tool {tool_name} executed successfully (this is a dummy implementation)",
                    }
                ]
            },
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def main():
    """Main entry point for the MCP server."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
            if response:
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError as e:
            err = {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            print(json.dumps(err), flush=True)


if __name__ == "__main__":
    main()
