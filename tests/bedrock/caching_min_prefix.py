#!/usr/bin/env python3
# ABOUTME: Tests minimum prefix size required for prompt caching on Amazon Bedrock.
# ABOUTME: Runs two prefix sizes (~1940 and ~2051 tokens) to bracket the 2048-token threshold.

import json
import os
import sys
import random
import string
import time

import boto3

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client


# The shorter prefix produces ~1940 tokens, which is below the 2048 cache threshold.
# The longer prefix adds extra Failure Detection content to reach ~2051 tokens.
SHARED_PREFIX = """\
You are an expert distributed systems architect.

# Comprehensive Guide to Distributed Systems

## CAP Theorem
The CAP theorem states that a distributed data store cannot simultaneously provide
more than two of the following three guarantees: Consistency (every read receives
the most recent write or an error), Availability (every request receives a non-error
response, without the guarantee that it contains the most recent write), and
Partition tolerance (the system continues to operate despite an arbitrary number
of messages being dropped or delayed by the network between nodes).

In practice, since network partitions are unavoidable in distributed systems, the
real choice is between consistency and availability during a partition event. Systems
like ZooKeeper and etcd choose consistency (CP), while systems like Cassandra and
DynamoDB can be configured to favor availability (AP). Understanding this tradeoff
is essential when designing any distributed data storage solution. Google Spanner
attempts to provide both consistency and availability by using TrueTime, a globally
synchronized clock based on GPS and atomic clocks, to implement external consistency
(a stronger property than linearizability) across geographically distributed data
centers. However, Spanner still cannot violate the CAP theorem during an actual
network partition; it chooses consistency over availability in that scenario.

## Consensus Algorithms
Paxos is a family of protocols for solving consensus in a network of unreliable
processors. Raft is a consensus algorithm designed as an alternative to Paxos,
intended to be more understandable. Both algorithms ensure that a cluster of servers
can agree on a sequence of values even if some servers fail. Key concepts include
leader election, log replication, and safety guarantees under network partitions.

Multi-Paxos extends basic Paxos by electing a stable leader to reduce message
complexity for successive consensus rounds. Raft decomposes consensus into leader
election, log replication, and safety subproblems. Viewstamped Replication is
another protocol that predates Raft but shares its emphasis on understandability.
PBFT (Practical Byzantine Fault Tolerance) extends consensus to tolerate arbitrary
(Byzantine) failures, requiring 3f+1 nodes to tolerate f faulty nodes. EPaxos
(Egalitarian Paxos) eliminates the need for a stable leader by allowing any replica
to propose commands, achieving optimal commit latency in the common case when
commands do not conflict. Mencius is another leaderless variant that partitions
the sequence number space among replicas to achieve balanced load without a single
leader bottleneck.

## Consistent Hashing
Consistent hashing is a distributed hashing scheme that operates independently of
the number of servers or objects in a distributed hash table. It minimizes the
number of keys that need to be remapped when the hash table is resized. This
technique is fundamental to distributed caching systems and load balancers.

The basic idea is to hash both keys and nodes onto a circular ring. Each key is
assigned to the first node encountered when walking clockwise around the ring.
Virtual nodes improve load distribution by mapping each physical node to multiple
positions on the ring. Systems like Amazon DynamoDB, Apache Cassandra, and Akamai
CDN all rely on consistent hashing for data partitioning and request routing. Jump
consistent hashing is a more recent algorithm that achieves perfect balance without
virtual nodes by using a pseudorandom function seeded with the key, requiring only
O(ln n) time and O(1) memory. Maglev hashing, developed at Google, is another
alternative designed for load balancing that provides both consistent hashing
properties and minimal disruption when backend servers are added or removed from
a pool. Rendezvous hashing (also known as highest random weight hashing) assigns
each key to the server that produces the highest hash value for that key-server
pair, providing an elegant alternative to ring-based approaches.

## Vector Clocks and Logical Time
Vector clocks are a mechanism for ordering events in a distributed system without
relying on synchronized physical clocks. Each process maintains a vector of logical
timestamps, one per process. When a process sends a message, it increments its own
entry and attaches the full vector. The recipient merges the received vector with
its own by taking element-wise maximums, then increments its own entry.

Lamport clocks provide a simpler but weaker ordering: if event A happened before
event B, then the Lamport timestamp of A is less than that of B, but the converse
is not necessarily true. Vector clocks fix this by capturing causality precisely.
Dotted version vectors extend vector clocks to handle sibling resolution in
eventually consistent systems like Riak. Hybrid logical clocks (HLC) combine
physical timestamps with logical counters to provide a practical ordering mechanism
that stays close to real time while still capturing causality. Interval tree clocks
generalize vector clocks to support dynamic creation and retirement of processes
without requiring a fixed set of process identifiers. The trade-off with vector
clocks is that their size grows linearly with the number of processes, which can
become prohibitive in large-scale systems with thousands of nodes.

## Replication Strategies
Single-leader replication routes all writes through one node. Multi-leader
replication allows writes on multiple nodes and merges conflicts. Leaderless
replication sends writes to several replicas in parallel and uses quorum reads
to resolve inconsistencies. Each approach makes different tradeoffs between
consistency, availability, latency, and operational complexity.

Chain replication is a variant of single-leader replication where writes propagate
through a chain of nodes, offering strong consistency with high throughput. CRDT
(Conflict-free Replicated Data Types) enable automatic conflict resolution in
multi-leader and leaderless setups by restricting data structures to those with
mathematically guaranteed merge properties. State-based CRDTs (CvRDTs) propagate
their full state and merge using a join semilattice, while operation-based CRDTs
(CmRDTs) propagate individual operations and require exactly-once delivery. Delta
state CRDTs combine the advantages of both approaches by propagating only the
state changes since the last synchronization. Anti-entropy protocols like Merkle
trees enable efficient detection and repair of replica divergence by comparing
hierarchical hashes of data ranges. Read-repair and hinted handoff are additional
mechanisms used in leaderless systems to ensure eventual convergence after
transient failures or network partitions have been resolved.

## Distributed Transactions
Two-phase commit (2PC) is the classic protocol for atomic distributed transactions.
A coordinator sends a prepare message to all participants, waits for votes, then
sends either a commit or abort decision. The protocol blocks if the coordinator
crashes after prepare but before the decision, leaving participants uncertain.
Three-phase commit (3PC) adds a pre-commit phase to reduce blocking at the cost
of additional message rounds and the assumption of bounded network delays.

Saga patterns decompose long-running transactions into a sequence of local
transactions, each with a compensating action that undoes its effects if a later
step fails. Choreography-based sagas let each service publish events that trigger
the next step, while orchestration-based sagas use a central coordinator to direct
the sequence. The choice between these approaches affects coupling, observability,
and error handling complexity.

Percolator, developed at Google, implements distributed transactions over Bigtable
using a timestamp oracle and a two-phase commit protocol with lazy cleanup of
failed transactions. Calvin is a deterministic database system that achieves
distributed transactions without traditional two-phase commit by pre-ordering all
transactions through a sequencing layer, eliminating the need for distributed
coordination at execution time. Spanner uses TrueTime to assign globally meaningful
timestamps to transactions, enabling external consistency across data centers.

## Failure Detection
Failure detection in distributed systems relies on heartbeat mechanisms and timeout
thresholds. The phi accrual failure detector provides a continuous suspicion level
rather than a binary alive-or-dead classification, allowing applications to make
nuanced decisions based on their specific requirements. SWIM (Scalable Weakly-
consistent Infection-style Membership) protocol achieves scalable failure detection
by combining random probing with protocol piggybacking, reducing the bandwidth
overhead to O(1) per member while maintaining logarithmic detection time. Gossip-
based failure detectors propagate membership information epidemically, providing
eventual consistency of the cluster view across all nodes."""

# Extra content appended to SHARED_PREFIX to push past the 2048 token threshold.
ABOVE_THRESHOLD_SUFFIX = """ Lifeguard enhances SWIM
with local health awareness and suspicion subgroups that reduce false positive
failure detections in cloud environments with variable network latency. Accurate
and timely failure detection is the foundation on which all higher-level distributed
protocols (consensus, replication, transactions) build their correctness and
liveness guarantees. In large-scale deployments spanning multiple data centers,
hierarchical failure detection organizes nodes into groups monitored by local
detectors, with cross-group communication handled by designated gateway nodes,
reducing overall network traffic while maintaining acceptable detection latency."""


def run_cache_pair_bedrock(bedrock, model_id: str, system_prefix: str, label: str):
    """Send two Bedrock requests with the same system prefix and check caching.

    Returns a dict with token counts, or None on error.
    """
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print(f"System prefix length: {len(system_prefix)} characters")
    print(f"{'=' * 60}")

    body1 = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 150,
        "system": [
            {
                "type": "text",
                "text": system_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain the CAP theorem in two sentences.",
                    }
                ],
            }
        ],
    }

    # --- Request 1 ---
    print("\n[Request 1] Sending first request...")
    try:
        response1 = bedrock.invoke_model(modelId=model_id, body=json.dumps(body1))
        result1 = json.loads(response1["body"].read())
    except Exception as e:
        print(f"FAIL: Request 1 raised an exception: {e}")
        return None

    usage1 = result1.get("usage", {})
    text1 = result1.get("content", [{}])[0].get("text", "")
    cache_creation = usage1.get("cache_creation_input_tokens", 0) or 0
    cache_read_r1 = usage1.get("cache_read_input_tokens", 0) or 0

    print(f"  Response: {text1[:200]}...")
    print(f"  input_tokens:                {usage1.get('input_tokens', 'N/A')}")
    print(f"  output_tokens:               {usage1.get('output_tokens', 'N/A')}")
    print(f"  cache_creation_input_tokens: {cache_creation}")
    print(f"  cache_read_input_tokens:     {cache_read_r1}")

    # --- Brief pause ---
    print("\nWaiting 2 seconds before second request...")
    time.sleep(2)

    # --- Request 2: same prefix, different user question ---
    body2 = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 150,
        "system": [
            {
                "type": "text",
                "text": system_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the difference between Paxos and Raft?",
                    }
                ],
            }
        ],
    }

    print("\n[Request 2] Sending second request...")
    try:
        response2 = bedrock.invoke_model(modelId=model_id, body=json.dumps(body2))
        result2 = json.loads(response2["body"].read())
    except Exception as e:
        print(f"FAIL: Request 2 raised an exception: {e}")
        return None

    usage2 = result2.get("usage", {})
    text2 = result2.get("content", [{}])[0].get("text", "")
    cache_creation_r2 = usage2.get("cache_creation_input_tokens", 0) or 0
    cache_read = usage2.get("cache_read_input_tokens", 0) or 0

    print(f"  Response: {text2[:200]}...")
    print(f"  input_tokens:                {usage2.get('input_tokens', 'N/A')}")
    print(f"  output_tokens:               {usage2.get('output_tokens', 'N/A')}")
    print(f"  cache_creation_input_tokens: {cache_creation_r2}")
    print(f"  cache_read_input_tokens:     {cache_read}")

    return {
        "cache_creation": cache_creation,
        "cache_read": cache_read,
        "cache_creation_r2": cache_creation_r2,
        "input_tokens_r1": usage1.get("input_tokens", 0) or 0,
    }


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    region = config["region"]

    # Generate a random nonce to bust any warm caches from previous runs.
    nonce = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    prefix_below = f"[{nonce}] " + SHARED_PREFIX
    prefix_above = f"[{nonce}] " + SHARED_PREFIX + ABOVE_THRESHOLD_SUFFIX

    print(f"Region: {region}")
    print(f"Bedrock model: {model_id}")
    print(f"Cache-bust nonce: {nonce}")

    bedrock = get_bedrock_client(config)

    # --- Bedrock: below threshold (~1940 tokens) ---
    bedrock_below = run_cache_pair_bedrock(
        bedrock,
        model_id,
        prefix_below,
        "Bedrock - below threshold (~1940 tokens)",
    )

    # --- Bedrock: above threshold (~2051 tokens) ---
    bedrock_above = run_cache_pair_bedrock(
        bedrock,
        model_id,
        prefix_above,
        "Bedrock - above threshold (~2051 tokens)",
    )

    # --- Summary ---
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True

    print(f"\n--- Bedrock ---")

    if bedrock_below is None:
        print("FAIL: Below-threshold test encountered an error")
        all_passed = False
    else:
        total_tokens = (
            bedrock_below["input_tokens_r1"] + bedrock_below["cache_creation"]
        )
        print(f"Below threshold (~1940 target, observed ~{total_tokens} tokens):")
        if bedrock_below["cache_creation"] == 0 and bedrock_below["cache_read"] == 0:
            print("  PASS: No caching occurred (prefix below threshold)")
        else:
            print(
                f"  INFO: Caching WAS observed (cache_creation={bedrock_below['cache_creation']}, "
                f"cache_read={bedrock_below['cache_read']})"
            )

    if bedrock_above is None:
        print("FAIL: Above-threshold test encountered an error")
        all_passed = False
    else:
        total_tokens = (
            bedrock_above["input_tokens_r1"] + bedrock_above["cache_creation"]
        )
        print(f"Above threshold (~2051 target, observed ~{total_tokens} tokens):")

        if bedrock_above["cache_creation"] > 0:
            print(
                f"  PASS: Cache created with {bedrock_above['cache_creation']} tokens"
            )
        else:
            print("  FAIL: No cache creation on first request")
            all_passed = False

        if bedrock_above["cache_read"] > 0:
            print(f"  PASS: Cache read with {bedrock_above['cache_read']} tokens")
        else:
            print("  FAIL: No cache read on second request")
            all_passed = False

        if bedrock_above["cache_creation_r2"] == 0:
            print("  PASS: Second request did not create a new cache entry")
        else:
            print(
                f"  WARN: Second request created cache with "
                f"{bedrock_above['cache_creation_r2']} tokens"
            )

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
