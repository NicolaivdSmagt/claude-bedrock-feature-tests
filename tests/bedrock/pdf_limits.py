#!/usr/bin/env python3
# ABOUTME: Test script for discovering PDF limits on Amazon Bedrock Claude models
# ABOUTME: Tests PDF count limits, page limits, and payload size limits using invoke_model API

import base64
import json
import os
import sys

import boto3

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)

# Paths to existing large PDF files (relative to this script via files/ directory)
_FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "files"
)
PDF_31MB = os.path.join(_FILES_DIR, "test_pdf_31MB.pdf")
PDF_9MB = os.path.join(_FILES_DIR, "example_106.pdf")
PDF_1MB = os.path.join(_FILES_DIR, "example.pdf")


def load_pdf(path: str) -> bytes:
    """Load a PDF file from disk"""
    with open(path, "rb") as f:
        return f.read()


def invoke_model_with_pdfs(client, model_id: str, pdf_list: list, prompt: str) -> dict:
    """Send a request with multiple PDFs using invoke_model API with 1M context beta"""
    content = []

    for pdf_data in pdf_list:
        pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
        content.append(
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_base64,
                },
            }
        )

    content.append({"type": "text", "text": prompt})

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["context-1m-2025-08-07"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    response = client.invoke_model(modelId=model_id, body=json.dumps(body))
    return json.loads(response["body"].read())


def main():
    print("PDF SIZE LIMIT TESTS - 1M CONTEXT")
    print("=" * 60)

    config = load_config()
    model_id = config["bedrock_model_id"]
    client = get_bedrock_client(config)

    # Load PDFs
    pdf_1mb = load_pdf(PDF_1MB)
    pdf_31mb = load_pdf(PDF_31MB)

    pdf_1mb_size = len(pdf_1mb) / (1024 * 1024)
    print(f"1MB PDF actual size: {pdf_1mb_size:.2f}MB")

    # Test: single 31MB PDF, then combinations
    tests = [
        ("~31MB single PDF", [pdf_31mb]),
        ("~10MB (7x 1.4MB text PDFs)", [pdf_1mb] * 7),
        ("~20MB (14x 1.4MB text PDFs)", [pdf_1mb] * 14),
        ("~30MB (21x 1.4MB text PDFs)", [pdf_1mb] * 21),
    ]

    for desc, pdf_list in tests:
        total_mb = sum(len(p) for p in pdf_list) / (1024 * 1024)
        encoded_mb = total_mb * 1.33
        print(
            f"\nTesting {desc} (raw: {total_mb:.1f}MB, encoded: ~{encoded_mb:.1f}MB)...",
            end=" ",
            flush=True,
        )

        try:
            response = invoke_model_with_pdfs(
                client,
                model_id,
                pdf_list,
                "How many PDF documents did I send? Answer with just the number.",
            )
            text = response.get("content", [{}])[0].get("text", "")
            print(f"✅ SUCCESS - Response: {text[:50]}")
        except Exception as e:
            print(f"❌ FAILED - {str(e)[:150]}")


if __name__ == "__main__":
    main()
