#!/usr/bin/env python3
# ABOUTME: Tests extended 1-hour cache TTL on Amazon Bedrock via the Converse API.
# ABOUTME: Verifies that the ttl parameter in cachePoint is accepted and caching works.

"""
Extended Cache TTL Test for Amazon Bedrock (Converse API)
==========================================================

Tests the extended cache TTL feature via the Converse API. Sends two
requests with a cached system prompt using cachePoint with ttl="1h":

1. First request — should create a cache entry
2. Second request (different user message) — should read from cache

The Converse API uses cachePoint blocks (separate from text blocks) with
a ttl parameter to control cache duration.

Requirements:
    uv add boto3

Usage:
    uv run python tests/bedrock/extended_cache_ttl_converse.py
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


def create_large_system_content():
    """Create system content large enough to be cached (min 2048 tokens for caching threshold)."""
    base_text = """You are an expert assistant specializing in software development.

Here is extensive background knowledge you should use when answering questions:

# Software Engineering Principles

## SOLID Principles
1. Single Responsibility Principle (SRP): A class should have only one reason to change. This means that a class should only have one job.
2. Open/Closed Principle (OCP): Software entities should be open for extension but closed for modification. This is typically achieved through abstraction and polymorphism.
3. Liskov Substitution Principle (LSP): Objects of a superclass should be replaceable with objects of subclasses without affecting the correctness of the program.
4. Interface Segregation Principle (ISP): Many client-specific interfaces are better than one general-purpose interface. Clients should not be forced to depend on methods they do not use.
5. Dependency Inversion Principle (DIP): Depend upon abstractions, not concretions. High-level modules should not depend on low-level modules. Both should depend on abstractions.

## Design Patterns

### Creational Patterns
- Singleton: Ensures a class has only one instance and provides a global point of access to it.
- Factory Method: Defines an interface for creating an object but lets subclasses decide which class to instantiate.
- Abstract Factory: Provides an interface for creating families of related or dependent objects without specifying their concrete classes.
- Builder: Separates the construction of a complex object from its representation.
- Prototype: Creates new objects by copying an existing object, known as the prototype.

### Structural Patterns
- Adapter: Converts the interface of a class into another interface clients expect.
- Bridge: Decouples an abstraction from its implementation so that the two can vary independently.
- Composite: Composes objects into tree structures to represent part-whole hierarchies.
- Decorator: Attaches additional responsibilities to an object dynamically.
- Facade: Provides a unified interface to a set of interfaces in a subsystem.
- Flyweight: Uses sharing to support large numbers of fine-grained objects efficiently.
- Proxy: Provides a surrogate or placeholder for another object to control access to it.

### Behavioral Patterns
- Chain of Responsibility: Passes a request along a chain of handlers.
- Command: Encapsulates a request as an object.
- Iterator: Provides a way to access the elements of an aggregate object sequentially.
- Mediator: Defines an object that encapsulates how a set of objects interact.
- Memento: Captures and externalizes an object's internal state.
- Observer: Defines a one-to-many dependency between objects.
- State: Allows an object to alter its behavior when its internal state changes.
- Strategy: Defines a family of algorithms, encapsulates each one, and makes them interchangeable.
- Template Method: Defines the skeleton of an algorithm in an operation.
- Visitor: Represents an operation to be performed on the elements of an object structure.

## Clean Code Practices
- Meaningful names for variables, functions, and classes
- Functions should be small and do one thing
- Comments should explain why, not what
- Error handling should be clean and predictable
- DRY (Don't Repeat Yourself)
- KISS (Keep It Simple, Stupid)
- YAGNI (You Aren't Gonna Need It)

## Testing Best Practices
- TDD: Red-Green-Refactor cycle
- Unit Tests: Test individual components in isolation
- Integration Tests: Test interaction between components
- End-to-End Tests: Test the entire system from the user's perspective

## Distributed Systems
- CAP Theorem: Consistency, Availability, Partition tolerance
- Consensus Algorithms: Paxos, Raft, PBFT
- Event Sourcing and CQRS

"""
    return base_text


def test_extended_cache_ttl(client, model_id):
    """Test extended cache TTL with two requests. Returns (status, error_msg)."""
    print("=" * 70)
    print("TEST: EXTENDED CACHE TTL (1h)")
    print("=" * 70)

    system_content = create_large_system_content()

    # Converse API uses cachePoint as a separate system content block
    system = [
        {"text": system_content},
        {"cachePoint": {"type": "default", "ttl": "1h"}},
    ]

    # Request 1: should create cache
    print("\n--- REQUEST 1: Cache creation ---")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "What is the Single Responsibility Principle? Answer in one sentence."
                }
            ],
        }
    ]

    print(f"  System content length: {len(system_content)} chars")
    print(f"  cachePoint: {{type: default, ttl: 1h}}")

    try:
        response1 = client.converse(
            modelId=model_id,
            system=system,
            messages=messages,
            inferenceConfig={"maxTokens": 100},
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    usage1 = response1.get("usage", {})
    content1 = (
        response1.get("output", {})
        .get("message", {})
        .get("content", [{}])[0]
        .get("text", "N/A")
    )

    print(f"\n  Response: {content1[:200]}")
    print(f"  inputTokens:           {usage1.get('inputTokens', 'N/A')}")
    print(f"  cacheWriteInputTokens: {usage1.get('cacheWriteInputTokens', 'N/A')}")
    print(f"  cacheReadInputTokens:  {usage1.get('cacheReadInputTokens', 'N/A')}")

    # Request 2: should read from cache
    print("\n  Waiting 2 seconds...")
    time.sleep(2)

    print("\n--- REQUEST 2: Cache read ---")

    messages[0]["content"][0]["text"] = (
        "What is the Open/Closed Principle? Answer in one sentence."
    )

    try:
        response2 = client.converse(
            modelId=model_id,
            system=system,
            messages=messages,
            inferenceConfig={"maxTokens": 100},
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n--- BEDROCK ERROR ---")
        print(f"  {error_msg}")
        return ("ERROR", error_msg)

    usage2 = response2.get("usage", {})
    content2 = (
        response2.get("output", {})
        .get("message", {})
        .get("content", [{}])[0]
        .get("text", "N/A")
    )

    print(f"\n  Response: {content2[:200]}")
    print(f"  inputTokens:           {usage2.get('inputTokens', 'N/A')}")
    print(f"  cacheWriteInputTokens: {usage2.get('cacheWriteInputTokens', 'N/A')}")
    print(f"  cacheReadInputTokens:  {usage2.get('cacheReadInputTokens', 'N/A')}")

    # Validate
    cache_created = usage1.get("cacheWriteInputTokens", 0) or 0
    cache_read_1 = usage1.get("cacheReadInputTokens", 0) or 0
    cache_read_2 = usage2.get("cacheReadInputTokens", 0) or 0

    if cache_created == 0 and cache_read_1 == 0:
        msg = f"Request 1 had no cache activity (creation={cache_created}, read={cache_read_1})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    if cache_read_2 == 0:
        msg = f"Request 2 did not read from cache (cacheReadInputTokens={cache_read_2})"
        print(f"\n  Result: FAIL - {msg}")
        return ("FAIL", msg)

    print(
        f"\n  Result: PASS - cache created ({cache_created} tokens), cache read ({cache_read_2} tokens)"
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

    status, error = test_extended_cache_ttl(client, model_id)
    results.append(("Extended cache TTL (1h)", status, error))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
