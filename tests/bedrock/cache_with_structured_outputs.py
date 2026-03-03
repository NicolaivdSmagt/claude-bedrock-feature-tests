#!/usr/bin/env python3
# ABOUTME: Tests prompt caching combined with structured outputs on AWS Bedrock invoke_model API.
# ABOUTME: Verifies cache hit with same schema, then observes cache behavior when schema changes (extra field added).

import boto3
import json
import os
import sys
import time
import argparse
from botocore.config import Config

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)

_cfg = load_config()
DEFAULT_MODEL_ID = _cfg["bedrock_model_id"]
DEFAULT_REGION = _cfg["region"]


def print_section(title: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{char * 70}")
    print(title)
    print(f"{char * 70}")


def create_large_system_content():
    """Create system content large enough to be cached (min 1024 tokens for Haiku/Sonnet)."""
    base_text = """You are an expert data analyst specializing in extracting structured information from text.

Here is extensive background knowledge you should use when analyzing data:

# Data Analysis Principles

## Data Quality Dimensions
1. Accuracy: Data correctly represents the real-world entity or event.
2. Completeness: All required data is present without missing values.
3. Consistency: Data does not contradict itself across different sources.
4. Timeliness: Data is up-to-date and available when needed.
5. Validity: Data conforms to defined business rules and constraints.
6. Uniqueness: No duplicate records exist in the dataset.

## Statistical Concepts
- Descriptive Statistics: Mean, median, mode, standard deviation, variance
- Inferential Statistics: Hypothesis testing, confidence intervals, p-values
- Regression Analysis: Linear regression, logistic regression, polynomial regression
- Classification Methods: Decision trees, random forests, support vector machines
- Clustering Techniques: K-means, hierarchical clustering, DBSCAN

## Data Transformation Techniques
- Normalization: Min-max scaling, z-score standardization
- Encoding: One-hot encoding, label encoding, target encoding
- Feature Engineering: Polynomial features, interaction terms, binning
- Dimensionality Reduction: PCA, t-SNE, UMAP
- Missing Data Handling: Imputation, deletion, interpolation

## Common Data Formats
- Tabular: CSV, TSV, Excel, Parquet, ORC
- Semi-structured: JSON, XML, YAML, TOML
- Graph: RDF, Neo4j, GraphML
- Time Series: InfluxDB line protocol, OpenTSDB format
- Geospatial: GeoJSON, Shapefile, KML, WKT

## Best Practices for Data Extraction
- Always validate extracted data against the source
- Handle edge cases like missing fields gracefully
- Use consistent data types for each field
- Normalize text fields (trim whitespace, consistent casing)
- Parse dates into ISO 8601 format when possible
- Validate email addresses against RFC 5322
- Ensure numeric fields are properly typed (integer vs float)

"""
    # Repeat to ensure we have enough tokens
    repeated_content = base_text * 5
    return repeated_content


def make_request(client, model_id, system_content, user_message, output_schema):
    """Make a single invoke_model request with caching and structured output."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": [
            {
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_message}]}
        ],
        "output_config": {
            "format": output_schema,
        },
    }

    response = client.invoke_model(
        body=json.dumps(body),
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
    result = json.loads(response.get("body").read())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test prompt caching with structured outputs on Amazon Bedrock"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    args = parser.parse_args()
    model_id = args.model_id
    region = args.region

    config = Config(read_timeout=600, retries=dict(max_attempts=3))
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region,
        config=config,
    )

    system_content = create_large_system_content()

    # JSON schema for requests 1 & 2: extract person info from text
    output_schema_base = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "company": {"type": "string"},
                "years_experience": {"type": "integer"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "role", "company", "years_experience", "skills"],
            "additionalProperties": False,
        },
    }

    # JSON schema for request 3: same fields plus a "location" field
    output_schema_extended = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "company": {"type": "string"},
                "years_experience": {"type": "integer"},
                "skills": {"type": "array", "items": {"type": "string"}},
                "location": {"type": "string"},
            },
            "required": [
                "name",
                "role",
                "company",
                "years_experience",
                "skills",
                "location",
            ],
            "additionalProperties": False,
        },
    }

    print_section("PROMPT CACHING + STRUCTURED OUTPUTS TEST")
    print(f"Model: {model_id}")
    print(f"Region: {region}")
    print(f"\nThis test sends three requests with:")
    print(f"  - Cached system prompt (cache_control: ephemeral)")
    print(f"  - Structured JSON output (output_config.format)")
    print(f"  - Requests 1 & 2: same schema (verify cache hit)")
    print(f"  - Request 3: modified schema with extra 'location' field")
    print(f"\nExpected:")
    print(f"  - Request 1 creates cache")
    print(f"  - Request 2 reads from cache (same schema)")
    print(f"  - Request 3 may miss cache (schema change affects cache key)")

    # --- Request 1: should create cache ---
    print_section("REQUEST 1: Cache creation", "-")
    user_message_1 = (
        "Extract the person's information from this bio: "
        "Alice Chen is a Senior Data Engineer at DataFlow Inc with 8 years of experience. "
        "She specializes in Python, Apache Spark, SQL, and Kafka."
    )
    print(f"User message: {user_message_1[:100]}...")
    print(f"Schema fields: name, role, company, years_experience, skills")

    try:
        result1 = make_request(
            client, model_id, system_content, user_message_1, output_schema_base
        )
    except Exception as e:
        print(f"  FAIL - Request 1 failed: {e}")
        return False

    usage1 = result1.get("usage", {})
    content1 = result1.get("content", [])
    text1 = content1[0].get("text", "") if content1 else ""

    print(f"\nResponse text: {text1[:300]}")
    print(f"\nUsage:")
    print(f"  input_tokens:                {usage1.get('input_tokens', 'N/A')}")
    print(f"  output_tokens:               {usage1.get('output_tokens', 'N/A')}")
    print(
        f"  cache_creation_input_tokens: {usage1.get('cache_creation_input_tokens', 'N/A')}"
    )
    print(
        f"  cache_read_input_tokens:     {usage1.get('cache_read_input_tokens', 'N/A')}"
    )

    try:
        parsed1 = json.loads(text1)
        print(f"\nParsed JSON: {json.dumps(parsed1, indent=2)}")
    except json.JSONDecodeError:
        print(f"\n  WARNING: Response is not valid JSON: {text1[:200]}")
        parsed1 = None

    # --- Wait before second request ---
    print("\nWaiting 2 seconds before second request...")
    time.sleep(2)

    # --- Request 2: same schema, should read from cache ---
    print_section("REQUEST 2: Cache read (same schema)", "-")
    user_message_2 = (
        "Extract the person's information from this bio: "
        "Bob Martinez is a Lead Machine Learning Engineer at NeuralTech Corp "
        "with 12 years of experience. "
        "He is skilled in TensorFlow, PyTorch, Kubernetes, and Go."
    )
    print(f"User message: {user_message_2[:100]}...")
    print(f"Schema fields: name, role, company, years_experience, skills")

    try:
        result2 = make_request(
            client, model_id, system_content, user_message_2, output_schema_base
        )
    except Exception as e:
        print(f"  FAIL - Request 2 failed: {e}")
        return False

    usage2 = result2.get("usage", {})
    content2 = result2.get("content", [])
    text2 = content2[0].get("text", "") if content2 else ""

    print(f"\nResponse text: {text2[:300]}")
    print(f"\nUsage:")
    print(f"  input_tokens:                {usage2.get('input_tokens', 'N/A')}")
    print(f"  output_tokens:               {usage2.get('output_tokens', 'N/A')}")
    print(
        f"  cache_creation_input_tokens: {usage2.get('cache_creation_input_tokens', 'N/A')}"
    )
    print(
        f"  cache_read_input_tokens:     {usage2.get('cache_read_input_tokens', 'N/A')}"
    )

    try:
        parsed2 = json.loads(text2)
        print(f"\nParsed JSON: {json.dumps(parsed2, indent=2)}")
    except json.JSONDecodeError:
        print(f"\n  WARNING: Response is not valid JSON: {text2[:200]}")
        parsed2 = None

    # --- Wait before third request ---
    print("\nWaiting 2 seconds before third request...")
    time.sleep(2)

    # --- Request 3: modified schema with location field ---
    print_section("REQUEST 3: Modified schema (extra location field)", "-")
    user_message_3 = (
        "Extract the person's information from this bio: "
        "Carol Davis is a Principal Software Architect at CloudScale Systems "
        "based in Seattle, Washington with 15 years of experience. "
        "She is proficient in Rust, C++, distributed systems, and AWS."
    )
    print(f"User message: {user_message_3[:100]}...")
    print(f"Schema fields: name, role, company, years_experience, skills, location")

    try:
        result3 = make_request(
            client, model_id, system_content, user_message_3, output_schema_extended
        )
    except Exception as e:
        print(f"  FAIL - Request 3 failed: {e}")
        return False

    usage3 = result3.get("usage", {})
    content3 = result3.get("content", [])
    text3 = content3[0].get("text", "") if content3 else ""

    print(f"\nResponse text: {text3[:300]}")
    print(f"\nUsage:")
    print(f"  input_tokens:                {usage3.get('input_tokens', 'N/A')}")
    print(f"  output_tokens:               {usage3.get('output_tokens', 'N/A')}")
    print(
        f"  cache_creation_input_tokens: {usage3.get('cache_creation_input_tokens', 'N/A')}"
    )
    print(
        f"  cache_read_input_tokens:     {usage3.get('cache_read_input_tokens', 'N/A')}"
    )

    try:
        parsed3 = json.loads(text3)
        print(f"\nParsed JSON: {json.dumps(parsed3, indent=2)}")
    except json.JSONDecodeError:
        print(f"\n  WARNING: Response is not valid JSON: {text3[:200]}")
        parsed3 = None

    # --- Analysis ---
    print_section("ANALYSIS")

    all_pass = True

    # Check cache creation or read on request 1
    cache_created_1 = usage1.get("cache_creation_input_tokens", 0) or 0
    cache_read_1 = usage1.get("cache_read_input_tokens", 0) or 0
    if cache_created_1 > 0:
        print(f"  PASS - Request 1 created cache with {cache_created_1} tokens")
    elif cache_read_1 > 0:
        print(
            f"  PASS - Request 1 read from existing cache with {cache_read_1} tokens (from prior run)"
        )
    else:
        print(
            f"  FAIL - Request 1 had no cache activity (creation={cache_created_1}, read={cache_read_1})"
        )
        all_pass = False

    # Check cache read on request 2 (same schema as request 1)
    cache_read_2 = usage2.get("cache_read_input_tokens", 0) or 0
    if cache_read_2 > 0:
        print(
            f"  PASS - Request 2 read from cache with {cache_read_2} tokens (same schema)"
        )
    else:
        print(
            f"  FAIL - Request 2 did not read from cache (cache_read_input_tokens={cache_read_2})"
        )
        all_pass = False

    # Check request 3 behavior (modified schema)
    cache_created_3 = usage3.get("cache_creation_input_tokens", 0) or 0
    cache_read_3 = usage3.get("cache_read_input_tokens", 0) or 0
    if cache_created_3 > 0 and cache_read_3 == 0:
        print(
            f"  INFO - Request 3 created a new cache with {cache_created_3} tokens (schema change invalidated cache)"
        )
    elif cache_read_3 > 0:
        print(
            f"  INFO - Request 3 read from cache with {cache_read_3} tokens (schema change did NOT invalidate cache)"
        )
    else:
        print(
            f"  INFO - Request 3 cache: creation={cache_created_3}, read={cache_read_3}"
        )

    # Check structured output validity for requests 1 & 2
    required_fields_base = ["name", "role", "company", "years_experience", "skills"]
    for i, parsed in enumerate([parsed1, parsed2], start=1):
        if parsed and all(f in parsed for f in required_fields_base):
            print(
                f"  PASS - Request {i} returned valid structured JSON with all required fields"
            )
        else:
            print(
                f"  FAIL - Request {i} structured output missing fields or invalid JSON"
            )
            all_pass = False

    # Check structured output validity for request 3 (includes location)
    required_fields_extended = required_fields_base + ["location"]
    if parsed3 and all(f in parsed3 for f in required_fields_extended):
        print(
            f"  PASS - Request 3 returned valid structured JSON with all required fields (including location)"
        )
    else:
        print(
            f"  FAIL - Request 3 structured output missing fields or invalid JSON (expected location field)"
        )
        all_pass = False

    # Summary
    print_section("SUMMARY")
    if all_pass:
        print("  All checks passed: caching and structured outputs work together.")
    else:
        print("  Some checks failed. See details above.")

    print_section("END OF TEST")
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
