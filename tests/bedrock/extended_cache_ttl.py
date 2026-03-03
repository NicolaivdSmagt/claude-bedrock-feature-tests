# ABOUTME: Tests extended 1-hour cache TTL feature on Amazon Bedrock invoke_model and converse APIs.
# ABOUTME: Verifies if Bedrock supports the ttl parameter in cache_control blocks.
import boto3
import json
import argparse
import os
import sys
import time

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client


def create_large_system_content(min_tokens=2100):
    """Create system content large enough to be cached (min 2048 tokens for caching threshold)."""
    base_text = """You are an expert assistant specializing in software development.

Here is extensive background knowledge you should use when answering questions:

# Software Engineering Principles

## SOLID Principles
1. Single Responsibility Principle (SRP): A class should have only one reason to change. This means that a class should only have one job. For example, a class that compiles and prints a report should be separated into two classes: one for compiling and one for printing.
2. Open/Closed Principle (OCP): Software entities should be open for extension but closed for modification. This means you should be able to add new functionality without changing existing code. This is typically achieved through abstraction and polymorphism.
3. Liskov Substitution Principle (LSP): Objects of a superclass should be replaceable with objects of subclasses without affecting the correctness of the program. If class B is a subclass of class A, we should be able to replace A with B without disrupting the behavior of the program.
4. Interface Segregation Principle (ISP): Many client-specific interfaces are better than one general-purpose interface. Clients should not be forced to depend on methods they do not use. This means splitting large interfaces into smaller, more specific ones.
5. Dependency Inversion Principle (DIP): Depend upon abstractions, not concretions. High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details. Details should depend on abstractions.

## Design Patterns

### Creational Patterns
- Singleton: Ensures a class has only one instance and provides a global point of access to it. Used for logging, driver objects, caching, thread pools, and database connections.
- Factory Method: Defines an interface for creating an object but lets subclasses decide which class to instantiate. It lets a class defer instantiation to subclasses.
- Abstract Factory: Provides an interface for creating families of related or dependent objects without specifying their concrete classes. It encapsulates a group of individual factories.
- Builder: Separates the construction of a complex object from its representation, allowing the same construction process to create various representations.
- Prototype: Creates new objects by copying an existing object, known as the prototype. This is used when the cost of creating a new object is expensive or complex.

### Structural Patterns
- Adapter: Converts the interface of a class into another interface clients expect. It lets classes work together that could not otherwise because of incompatible interfaces.
- Bridge: Decouples an abstraction from its implementation so that the two can vary independently. It is used when both the class and what it does vary often.
- Composite: Composes objects into tree structures to represent part-whole hierarchies. It lets clients treat individual objects and compositions of objects uniformly.
- Decorator: Attaches additional responsibilities to an object dynamically. It provides a flexible alternative to subclassing for extending functionality.
- Facade: Provides a unified interface to a set of interfaces in a subsystem. It defines a higher-level interface that makes the subsystem easier to use.
- Flyweight: Uses sharing to support large numbers of fine-grained objects efficiently. A flyweight is a shared object that can be used in multiple contexts simultaneously.
- Proxy: Provides a surrogate or placeholder for another object to control access to it. This can be useful for lazy initialization, logging, access control, and caching.

### Behavioral Patterns
- Chain of Responsibility: Passes a request along a chain of handlers. Upon receiving a request, each handler decides either to process the request or to pass it to the next handler in the chain.
- Command: Encapsulates a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.
- Iterator: Provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
- Mediator: Defines an object that encapsulates how a set of objects interact. It promotes loose coupling by keeping objects from referring to each other explicitly.
- Memento: Without violating encapsulation, captures and externalizes an object's internal state so that the object can be restored to this state later.
- Observer: Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
- State: Allows an object to alter its behavior when its internal state changes. The object will appear to change its class.
- Strategy: Defines a family of algorithms, encapsulates each one, and makes them interchangeable. It lets the algorithm vary independently from clients that use it.
- Template Method: Defines the skeleton of an algorithm in an operation, deferring some steps to subclasses. It lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.
- Visitor: Represents an operation to be performed on the elements of an object structure. It lets you define a new operation without changing the classes of the elements on which it operates.

## Clean Code Practices
- Meaningful names for variables, functions, and classes: Names should reveal intent. A name should tell you why it exists, what it does, and how it is used.
- Functions should be small and do one thing: Functions should hardly ever be 20 lines long. They should do one thing, do it well, and do it only.
- Comments should explain why, not what: The proper use of comments is to compensate for our failure to express ourselves in code. Comments should explain intent and rationale.
- Error handling should be clean and predictable: Use exceptions rather than return codes. Create informative error messages. Do not return null. Do not pass null.
- Tests should be fast, independent, repeatable, self-validating, and timely (FIRST principles). Tests should be written before the production code (TDD).
- The Boy Scout Rule: Leave the campground cleaner than you found it. Always leave the code better than you found it.
- DRY (Don't Repeat Yourself): Every piece of knowledge must have a single, unambiguous, authoritative representation within a system.
- KISS (Keep It Simple, Stupid): Most systems work best if they are kept simple rather than made complicated.
- YAGNI (You Aren't Gonna Need It): Don't add functionality until it is necessary. Implement things when you actually need them, never when you just foresee that you might need them.

## Testing Best Practices

### Test-Driven Development (TDD)
TDD is a software development approach where tests are written before the production code. The cycle follows Red-Green-Refactor:
1. Red: Write a failing test that defines the desired behavior
2. Green: Write the minimal code to make the test pass
3. Refactor: Clean up the code while keeping tests green

### Types of Tests
- Unit Tests: Test individual components in isolation. They should be fast, isolated, and focused on a single unit of behavior. Mock external dependencies.
- Integration Tests: Test the interaction between multiple components. They verify that different parts of the system work together correctly. Use real dependencies where practical.
- End-to-End Tests: Test the entire system from the user's perspective. They simulate real user scenarios and verify the system behaves correctly as a whole.
- Contract Tests: Verify that the communication between services adheres to a defined contract. They ensure API compatibility between producer and consumer services.
- Performance Tests: Measure response times, throughput, and resource utilization under various load conditions. Include load tests, stress tests, and soak tests.

### Test Organization
- Arrange-Act-Assert (AAA) pattern: Structure tests into setup, execution, and verification phases
- Given-When-Then: BDD-style test organization that describes preconditions, actions, and expected outcomes
- Test fixtures: Reusable test data and setup code that can be shared across multiple tests
- Test factories: Create test objects with sensible defaults that can be customized per test

## Distributed Systems Concepts

### CAP Theorem
The CAP theorem states that a distributed data store cannot simultaneously provide more than two of the following three guarantees: Consistency (every read receives the most recent write), Availability (every request receives a non-error response), and Partition tolerance (the system continues to operate despite network partitions).

### Consensus Algorithms
- Paxos: A family of protocols for solving consensus in a network of unreliable processors. It ensures that a single value is agreed upon by a majority of participants.
- Raft: Designed as an understandable alternative to Paxos. It separates the key elements of consensus into leader election, log replication, and safety.
- PBFT (Practical Byzantine Fault Tolerance): Handles Byzantine faults where nodes may behave arbitrarily, including malicious behavior.

### Event Sourcing and CQRS
Event Sourcing stores the state of an application as a sequence of events rather than current state. Combined with CQRS (Command Query Responsibility Segregation), it separates read and write models for optimized performance and scalability.

"""
    return base_text


def test_extended_cache_ttl(region: str, model_id: str, ttl: str = "1h"):
    """
    Test if Bedrock supports extended cache TTL.

    Args:
        region: AWS region
        model_id: Bedrock model ID
        ttl: Cache TTL - "5m" for 5 minutes or "1h" for 1 hour
    """
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)

    system_content = create_large_system_content()

    # Request body with cache_control including ttl parameter
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "system": [
            {
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral", "ttl": ttl},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the Single Responsibility Principle? Answer in one sentence.",
                    }
                ],
            }
        ],
    }

    print(f"Testing cache TTL: {ttl}")
    print(f"Region: {region}")
    print(f"Model: {model_id}")
    print("-" * 50)

    # First request - should create cache
    print("\n[Request 1] Creating cache entry...")
    try:
        response1 = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
        result1 = json.loads(response1["body"].read())

        print(
            f"Response: {result1.get('content', [{}])[0].get('text', 'N/A')[:200]}..."
        )
        print(f"\nUsage:")
        usage1 = result1.get("usage", {})
        print(f"  input_tokens: {usage1.get('input_tokens', 'N/A')}")
        print(f"  output_tokens: {usage1.get('output_tokens', 'N/A')}")
        print(
            f"  cache_creation_input_tokens: {usage1.get('cache_creation_input_tokens', 'N/A')}"
        )
        print(
            f"  cache_read_input_tokens: {usage1.get('cache_read_input_tokens', 'N/A')}"
        )

        # Check for detailed cache_creation breakdown (1h vs 5m)
        if "cache_creation" in usage1:
            print(f"  cache_creation details: {usage1['cache_creation']}")

    except Exception as e:
        print(f"Request 1 failed: {e}")
        return False

    # Short delay before second request
    print("\nWaiting 2 seconds before second request...")
    time.sleep(2)

    # Second request - should read from cache
    print("\n[Request 2] Reading from cache...")
    body["messages"][0]["content"][0]["text"] = (
        "What is the Open/Closed Principle? Answer in one sentence."
    )

    try:
        response2 = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
        result2 = json.loads(response2["body"].read())

        print(
            f"Response: {result2.get('content', [{}])[0].get('text', 'N/A')[:200]}..."
        )
        print(f"\nUsage:")
        usage2 = result2.get("usage", {})
        print(f"  input_tokens: {usage2.get('input_tokens', 'N/A')}")
        print(f"  output_tokens: {usage2.get('output_tokens', 'N/A')}")
        print(
            f"  cache_creation_input_tokens: {usage2.get('cache_creation_input_tokens', 'N/A')}"
        )
        print(
            f"  cache_read_input_tokens: {usage2.get('cache_read_input_tokens', 'N/A')}"
        )

        if "cache_creation" in usage2:
            print(f"  cache_creation details: {usage2['cache_creation']}")

    except Exception as e:
        print(f"Request 2 failed: {e}")
        return False

    # Analysis
    print("\n" + "=" * 50)
    print("ANALYSIS")
    print("=" * 50)

    cache_created = usage1.get("cache_creation_input_tokens", 0)
    cache_read = usage2.get("cache_read_input_tokens", 0)

    if cache_created and cache_created > 0:
        print(f"✓ Cache was created with {cache_created} tokens")
    else:
        print("✗ No cache creation detected in first request")

    if cache_read and cache_read > 0:
        print(f"✓ Cache was read with {cache_read} tokens")
    else:
        print("✗ No cache read detected in second request")

    # Check if ttl parameter was accepted (no error = likely accepted)
    print(f"\n✓ TTL parameter '{ttl}' was accepted (no validation error)")

    return True


def test_extended_cache_ttl_converse(region: str, model_id: str, ttl: str = "1h"):
    """
    Test if Bedrock Converse API supports extended cache TTL.

    Args:
        region: AWS region
        model_id: Bedrock model ID
        ttl: Cache TTL - "5m" for 5 minutes or "1h" for 1 hour
    """
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)

    system_content = create_large_system_content()

    # Converse API system format with cache_control
    # cachePoint is a separate content block that follows the text block
    # NOTE: ttl parameter may not be supported yet - deployment may be ongoing
    system = [{"text": system_content}, {"cachePoint": {"type": "default", "ttl": ttl}}]

    # Converse API message format
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

    print(f"\nTesting cache TTL with Converse API: {ttl}")
    print(f"Region: {region}")
    print(f"Model: {model_id}")
    print("-" * 50)

    # First request - should create cache
    print("\n[Request 1] Creating cache entry...")
    try:
        response1 = bedrock.converse(
            modelId=model_id,
            system=system,
            messages=messages,
            inferenceConfig={"maxTokens": 100},
        )

        output_text = (
            response1.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "N/A")
        )
        print(f"Response: {output_text[:200]}...")
        print(f"\nUsage:")
        usage1 = response1.get("usage", {})
        print(f"  inputTokens: {usage1.get('inputTokens', 'N/A')}")
        print(f"  outputTokens: {usage1.get('outputTokens', 'N/A')}")
        print(f"  cacheReadInputTokens: {usage1.get('cacheReadInputTokens', 'N/A')}")
        print(f"  cacheWriteInputTokens: {usage1.get('cacheWriteInputTokens', 'N/A')}")

    except Exception as e:
        print(f"Request 1 failed: {e}")
        return False

    # Short delay before second request
    print("\nWaiting 2 seconds before second request...")
    time.sleep(2)

    # Second request - should read from cache
    print("\n[Request 2] Reading from cache...")
    messages[0]["content"][0]["text"] = (
        "What is the Open/Closed Principle? Answer in one sentence."
    )

    try:
        response2 = bedrock.converse(
            modelId=model_id,
            system=system,
            messages=messages,
            inferenceConfig={"maxTokens": 100},
        )

        output_text = (
            response2.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "N/A")
        )
        print(f"Response: {output_text[:200]}...")
        print(f"\nUsage:")
        usage2 = response2.get("usage", {})
        print(f"  inputTokens: {usage2.get('inputTokens', 'N/A')}")
        print(f"  outputTokens: {usage2.get('outputTokens', 'N/A')}")
        print(f"  cacheReadInputTokens: {usage2.get('cacheReadInputTokens', 'N/A')}")
        print(f"  cacheWriteInputTokens: {usage2.get('cacheWriteInputTokens', 'N/A')}")

    except Exception as e:
        print(f"Request 2 failed: {e}")
        return False

    # Analysis
    print("\n" + "=" * 50)
    print("ANALYSIS (Converse API)")
    print("=" * 50)

    cache_created = usage1.get("cacheWriteInputTokens", 0)
    cache_read = usage2.get("cacheReadInputTokens", 0)

    if cache_created and cache_created > 0:
        print(f"✓ Cache was created with {cache_created} tokens")
    else:
        print("✗ No cache creation detected in first request")

    if cache_read and cache_read > 0:
        print(f"✓ Cache was read with {cache_read} tokens")
    else:
        print("✗ No cache read detected in second request")

    # Check if ttl parameter was accepted (no error = likely accepted)
    print(f"\n✓ TTL parameter '{ttl}' was accepted (no validation error)")

    return True


def test_compare_ttls(region: str, model_id: str, api: str = "both"):
    """Compare 5m and 1h TTL behavior."""
    if api in ("invoke_model", "both"):
        print("\n" + "=" * 60)
        print("TESTING 5-MINUTE TTL (invoke_model API)")
        print("=" * 60)
        test_extended_cache_ttl(region, model_id, ttl="5m")

        print("\n\n" + "=" * 60)
        print("TESTING 1-HOUR TTL (invoke_model API)")
        print("=" * 60)
        test_extended_cache_ttl(region, model_id, ttl="1h")

    if api in ("converse", "both"):
        print("\n\n" + "=" * 60)
        print("TESTING 5-MINUTE TTL (Converse API)")
        print("=" * 60)
        test_extended_cache_ttl_converse(region, model_id, ttl="5m")

        print("\n\n" + "=" * 60)
        print("TESTING 1-HOUR TTL (Converse API)")
        print("=" * 60)
        test_extended_cache_ttl_converse(region, model_id, ttl="1h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test extended cache TTL on Amazon Bedrock"
    )
    config = load_config()
    parser.add_argument(
        "--region",
        type=str,
        default=config["region"],
        help=f"AWS region (default: {config['region']})",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=config["bedrock_model_id"],
        help=f"Model ID (default: {config['bedrock_model_id']})",
    )
    parser.add_argument(
        "--ttl",
        type=str,
        default="1h",
        choices=["5m", "1h"],
        help="Cache TTL to test (default: 1h)",
    )
    parser.add_argument(
        "--api",
        type=str,
        default="both",
        choices=["invoke_model", "converse", "both"],
        help="API to test (default: both)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Test both 5m and 1h TTLs"
    )

    args = parser.parse_args()

    if args.compare:
        test_compare_ttls(args.region, args.model_id, args.api)
    else:
        if args.api in ("invoke_model", "both"):
            test_extended_cache_ttl(args.region, args.model_id, args.ttl)
        if args.api in ("converse", "both"):
            test_extended_cache_ttl_converse(args.region, args.model_id, args.ttl)
