# ABOUTME: One-time script to generate and save a static conversation history for compaction testing.
# ABOUTME: Uses Bedrock count_tokens API to calibrate the conversation to just over 50,000 tokens.

"""
Run this once to produce 50000_token_conversation.json.
It builds a realistic enterprise agentic workflow conversation, then uses
the Bedrock count_tokens API to grow it to just over 50,000 tokens.

Usage:
    uv run python manual/generate_compaction_history.py
"""

import json
import os
import re
import sys

import boto3
from botocore.config import Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_config import load_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enterprise_workflow_history import (
    build_system_prompt,
    build_tools,
    build_workflow_turns,
)

_cfg = load_config()
REGION = _cfg["region"]
# count_tokens API does not support CRIS prefixes (global., eu., us.)
COUNT_TOKENS_MODEL_ID = re.sub(r"^(global|eu|us)\.", "", _cfg["bedrock_model_id"])
TARGET_TOKENS = 50_000
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "50000_token_conversation.json"
)


def setup_client():
    config = Config(region_name=REGION, read_timeout=60, retries=dict(max_attempts=3))
    return boto3.client("bedrock-runtime", config=config)


def count_tokens(client, messages, system, tools):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 4096,
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    response = client.count_tokens(
        modelId=COUNT_TOKENS_MODEL_ID,
        input={"invokeModel": {"body": json.dumps(body)}},
    )
    return response["inputTokens"]


# Padding material: additional realistic enterprise workflow turns
PADDING_TOPICS = [
    (
        "We also need to add rate limiting to the API. Can you add a Redis-based rate limiter "
        "that limits each customer to 100 orders per hour? Include both the Spring configuration "
        "and a custom filter that checks the rate limit before the request hits the controller.",
        "I'll implement a Redis-based sliding window rate limiter. The approach uses a sorted set "
        "in Redis where each element is a request timestamp. For each incoming request, we:\n\n"
        "1. Remove all entries older than the window (1 hour)\n"
        "2. Count remaining entries\n"
        "3. If count < limit (100), add the new timestamp and allow the request\n"
        "4. Otherwise, return 429 Too Many Requests with a Retry-After header\n\n"
        "Here's the implementation:\n\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.java\n"
        "@Component\n"
        "public class RateLimitFilter extends OncePerRequestFilter {\n\n"
        "    private static final int MAX_REQUESTS_PER_HOUR = 100;\n"
        "    private static final Duration WINDOW = Duration.ofHours(1);\n"
        "    private final StringRedisTemplate redisTemplate;\n\n"
        "    @Override\n"
        "    protected void doFilterInternal(HttpServletRequest request,\n"
        "                                    HttpServletResponse response,\n"
        "                                    FilterChain filterChain)\n"
        "            throws ServletException, IOException {\n"
        '        String customerId = request.getHeader("X-Customer-Id");\n'
        "        if (customerId == null) { filterChain.doFilter(request, response); return; }\n"
        '        String key = "rate_limit:orders:" + customerId;\n'
        "        long now = Instant.now().toEpochMilli();\n"
        "        long windowStart = now - WINDOW.toMillis();\n"
        "        List<Object> results = redisTemplate.executePipelined((RedisCallback<Object>) conn -> {\n"
        "            byte[] rawKey = key.getBytes();\n"
        "            conn.zRemRangeByScore(rawKey, 0, windowStart);\n"
        "            conn.zCard(rawKey);\n"
        "            conn.zAdd(rawKey, now, String.valueOf(now).getBytes());\n"
        "            conn.expire(rawKey, WINDOW.getSeconds());\n"
        "            return null;\n"
        "        });\n"
        "        long currentCount = (Long) results.get(1);\n"
        "        if (currentCount >= MAX_REQUESTS_PER_HOUR) {\n"
        "            response.setStatus(429);\n"
        '            response.setHeader("Retry-After", "60");\n'
        '            response.getWriter().write("{\\"error\\": \\"Rate limit exceeded\\"}");\n'
        "            return;\n"
        "        }\n"
        '        response.setHeader("X-RateLimit-Remaining",\n'
        "            String.valueOf(MAX_REQUESTS_PER_HOUR - currentCount - 1));\n"
        "        filterChain.doFilter(request, response);\n"
        "    }\n"
        "}\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.\n\n"
        "And the Redis configuration:\n\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.java\n"
        "@Configuration\n"
        "public class RedisConfig {\n"
        "    @Bean\n"
        "    public LettuceConnectionFactory redisConnectionFactory(\n"
        '            @Value("${spring.redis.host:localhost}") String host,\n'
        '            @Value("${spring.redis.port:6379}") int port) {\n'
        "        return new LettuceConnectionFactory(new RedisStandaloneConfiguration(host, port));\n"
        "    }\n"
        "    @Bean\n"
        "    public StringRedisTemplate stringRedisTemplate(LettuceConnectionFactory cf) {\n"
        "        return new StringRedisTemplate(cf);\n"
        "    }\n"
        "}\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.",
    ),
    (
        "We need distributed tracing with OpenTelemetry. Set up the configuration so that "
        "all REST calls, Kafka messages, and database queries are automatically traced. Also add "
        "custom spans for the order creation workflow with relevant attributes.",
        "I'll configure OpenTelemetry with the Java Agent for auto-instrumentation plus manual "
        "spans for business operations.\n\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.java\n"
        "@Configuration\n"
        "public class ObservabilityConfig {\n"
        "    @Bean\n"
        "    public OpenTelemetry openTelemetry() {\n"
        "        Resource resource = Resource.getDefault().merge(Resource.create(Attributes.of(\n"
        '            ResourceAttributes.SERVICE_NAME, "order-service",\n'
        '            ResourceAttributes.SERVICE_VERSION, "1.0.0")));\n'
        "        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()\n"
        "            .addSpanProcessor(BatchSpanProcessor.builder(\n"
        "                OtlpGrpcSpanExporter.builder()\n"
        '                    .setEndpoint("http://otel-collector:4317").build()).build())\n'
        "            .setResource(resource)\n"
        "            .setSampler(Sampler.traceIdRatioBased(0.1))\n"
        "            .build();\n"
        "        return OpenTelemetrySdk.builder()\n"
        "            .setTracerProvider(tracerProvider)\n"
        "            .setPropagators(ContextPropagators.create(\n"
        "                W3CTraceContextPropagator.getInstance()))\n"
        "            .buildAndRegisterGlobal();\n"
        "    }\n"
        "}\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.\n\n"
        "Custom spans in OrderService:\n\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.java\n"
        'Span span = tracer.spanBuilder("order.create")\n'
        '    .setAttribute("order.customer_id", request.getCustomerId().toString())\n'
        '    .setAttribute("order.item_count", request.getItems().size())\n'
        "    .startSpan();\n"
        "try (Scope scope = span.makeCurrent()) {\n"
        "    // ... business logic ...\n"
        '    span.setAttribute("order.id", order.getId().toString());\n'
        '    span.setAttribute("order.total_amount", order.getTotalAmount().doubleValue());\n'
        "    span.setStatus(StatusCode.OK);\n"
        "} catch (Exception e) {\n"
        "    span.setStatus(StatusCode.ERROR, e.getMessage());\n"
        "    span.recordException(e);\n"
        "    throw e;\n"
        "} finally { span.end(); }\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.\n\n"
        "Application config:\nSee `AGENTS_PRIVATE.md` for remote URLs and push instructions.yaml\notel:\n  traces.exporter: otlp\n  metrics.exporter: otlp\n"
        "  exporter.otlp.endpoint: http://otel-collector:4317\n"
        "  instrumentation:\n    spring-web.enabled: true\n    kafka.enabled: true\n"
        "    jdbc.enabled: true\n    redis.enabled: true\nSee `AGENTS_PRIVATE.md` for remote URLs and push instructions.",
    ),
    (
        "Add custom health check endpoints that verify connectivity to PostgreSQL, Kafka, and Redis.",
        "Custom health indicators for each dependency:\n\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.java\n"
        "@Component\n"
        "public class KafkaHealthIndicator implements HealthIndicator {\n"
        "    private final AdminClient adminClient;\n"
        "    @Override\n"
        "    public Health health() {\n"
        "        try {\n"
        "            int nodeCount = adminClient.describeCluster().nodes()\n"
        "                .get(5, TimeUnit.SECONDS).size();\n"
        "            String clusterId = adminClient.describeCluster().clusterId()\n"
        "                .get(5, TimeUnit.SECONDS);\n"
        "            return Health.up()\n"
        '                .withDetail("clusterId", clusterId)\n'
        '                .withDetail("nodeCount", nodeCount).build();\n'
        "        } catch (Exception e) { return Health.down().withException(e).build(); }\n"
        "    }\n"
        "}\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.\n\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.java\n"
        "@Component\n"
        "public class RedisHealthIndicator implements HealthIndicator {\n"
        "    private final StringRedisTemplate redisTemplate;\n"
        "    @Override\n"
        "    public Health health() {\n"
        "        try {\n"
        "            String pong = redisTemplate.getConnectionFactory()\n"
        "                .getConnection().ping();\n"
        '            return Health.up().withDetail("ping", pong).build();\n'
        "        } catch (Exception e) { return Health.down().withException(e).build(); }\n"
        "    }\n"
        "}\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.\n\n"
        "Config:\nSee `AGENTS_PRIVATE.md` for remote URLs and push instructions.yaml\nmanagement:\n  endpoints.web.exposure.include: health,info,metrics,prometheus\n"
        "  endpoint.health:\n    show-details: when_authorized\n    group:\n"
        "      readiness.include: db,kafka,redis\n      liveness.include: ping\nSee `AGENTS_PRIVATE.md` for remote URLs and push instructions.",
    ),
    (
        "Add a dead letter queue handler for Kafka messages that fail processing after retries, "
        "with an admin endpoint to view and retry DLQ messages.",
        "DLQ solution with database-backed store and admin API:\n\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.java\n"
        "@Component\n"
        "public class DeadLetterQueueHandler {\n"
        "    private final DeadLetterRepository dlqRepository;\n"
        "    private final KafkaTemplate<String, String> kafkaTemplate;\n\n"
        '    @KafkaListener(topics = {"payment.results.DLQ", "inventory.updates.DLQ"},\n'
        '                   groupId = "order-service-dlq")\n'
        "    @Transactional\n"
        "    public void handleDeadLetter(ConsumerRecord<String, String> record, Acknowledgment ack) {\n"
        "        DeadLetterEntry entry = new DeadLetterEntry();\n"
        '        entry.setOriginalTopic(record.topic().replace(".DLQ", ""));\n'
        "        entry.setPayload(record.value());\n"
        "        entry.setStatus(DLQStatus.PENDING);\n"
        "        record.headers().forEach(h -> {\n"
        '            if ("kafka_dlt-exception-message".equals(h.key()))\n'
        "                entry.setErrorMessage(new String(h.value()));\n"
        "        });\n"
        "        dlqRepository.save(entry);\n"
        "        ack.acknowledge();\n"
        "    }\n\n"
        "    public void retryMessage(UUID entryId) {\n"
        "        DeadLetterEntry entry = dlqRepository.findById(entryId).orElseThrow();\n"
        "        kafkaTemplate.send(entry.getOriginalTopic(), entry.getPayload())\n"
        "            .whenComplete((r, ex) -> {\n"
        "                entry.setStatus(ex != null ? DLQStatus.PENDING : DLQStatus.RETRIED);\n"
        "                entry.setRetryCount(entry.getRetryCount() + 1);\n"
        "                dlqRepository.save(entry);\n"
        "            });\n"
        "    }\n"
        "}\n"
        "See `AGENTS_PRIVATE.md` for remote URLs and push instructions.\n\n"
        "Admin controller:\nSee `AGENTS_PRIVATE.md` for remote URLs and push instructions.java\n"
        '@RestController @RequestMapping("/admin/dlq")\n'
        "public class DeadLetterController {\n"
        "    @GetMapping\n"
        "    public Page<DeadLetterEntry> list(@RequestParam(required=false) DLQStatus status,\n"
        '            @RequestParam(defaultValue="0") int page) {\n'
        "        return status != null\n"
        "            ? dlqRepo.findByStatus(status, PageRequest.of(page, 20))\n"
        '            : dlqRepo.findAll(PageRequest.of(page, 20, Sort.by(DESC, "failedAt")));\n'
        "    }\n"
        '    @PostMapping("/{id}/retry")\n'
        "    public ResponseEntity<String> retry(@PathVariable UUID id) {\n"
        '        dlqHandler.retryMessage(id); return ResponseEntity.ok("Retry initiated");\n'
        "    }\n"
        '    @GetMapping("/stats")\n'
        "    public Map<String,Object> stats() {\n"
        '        return Map.of("pending", dlqRepo.countByStatus(PENDING),\n'
        '                      "retried", dlqRepo.countByStatus(RETRIED));\n'
        "    }\n"
        "}\nSee `AGENTS_PRIVATE.md` for remote URLs and push instructions.",
    ),
]


def make_filler_turn(n):
    """Generate a filler review turn for padding to the token target."""
    user = (
        f"Review iteration {n}: Check the error handling in the OrderService. "
        f"Make sure all exceptions are properly logged with correlation IDs, and that the "
        f"circuit breaker configuration uses appropriate timeout and retry values for our "
        f"production traffic patterns (p99 latency target of 200ms for order creation). "
        f"Also verify that the Kafka producer configuration has proper acknowledgment settings "
        f"(acks=all) and that the consumer has correct offset management."
    )
    assistant = (
        f"Review pass {n} — I've audited the error handling and infrastructure configuration:\n\n"
        f"**Error Handling Improvements:**\n"
        f"- Added MDC for correlation ID propagation across all log statements\n"
        f"- All catch blocks now log the full exception chain\n"
        f"- Custom exceptions carry correlation IDs via a base OrderDomainException class\n"
        f"- Added @ControllerAdvice global exception handler\n\n"
        f"**Circuit Breaker Configuration (application.yml):**\n"
        f"See `AGENTS_PRIVATE.md` for remote URLs and push instructions.yaml\n"
        f"resilience4j:\n"
        f"  circuitbreaker:\n"
        f"    instances:\n"
        f"      orderCreation:\n"
        f"        slidingWindowSize: 10\n"
        f"        minimumNumberOfCalls: 5\n"
        f"        waitDurationInOpenState: 30s\n"
        f"        failureRateThreshold: 50\n"
        f"        slowCallDurationThreshold: 200ms\n"
        f"  retry:\n"
        f"    instances:\n"
        f"      orderCreation:\n"
        f"        maxAttempts: 3\n"
        f"        waitDuration: 100ms\n"
        f"        exponentialBackoffMultiplier: 2\n"
        f"See `AGENTS_PRIVATE.md` for remote URLs and push instructions.\n\n"
        f"**Kafka Producer Configuration:**\n"
        f"See `AGENTS_PRIVATE.md` for remote URLs and push instructions.yaml\n"
        f"spring:\n"
        f"  kafka:\n"
        f"    producer:\n"
        f"      acks: all\n"
        f"      retries: 3\n"
        f"      properties:\n"
        f"        enable.idempotence: true\n"
        f"        max.in.flight.requests.per.connection: 5\n"
        f"    consumer:\n"
        f"      enable-auto-commit: false\n"
        f"      auto-offset-reset: earliest\n"
        f"See `AGENTS_PRIVATE.md` for remote URLs and push instructions.\n\n"
        f"All configurations verified and aligned with production requirements."
    )
    return user, assistant


def main():
    client = setup_client()

    system = build_system_prompt()
    tools = build_tools()
    messages = build_workflow_turns()

    token_count = count_tokens(client, messages, system, tools)
    print(f"Base conversation: {len(messages)} turns, {token_count:,} tokens")

    # Add padding topics
    for i, (user_msg, assistant_msg) in enumerate(PADDING_TOPICS):
        if token_count >= TARGET_TOKENS:
            break
        messages.append({"role": "user", "content": user_msg})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]}
        )
        token_count = count_tokens(client, messages, system, tools)
        print(f"  After padding turn {i + 1}: {token_count:,} tokens")

    # Add filler turns until over target
    filler = 0
    while token_count < TARGET_TOKENS:
        filler += 1
        user_msg, assistant_msg = make_filler_turn(filler)
        messages.append({"role": "user", "content": user_msg})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]}
        )
        token_count = count_tokens(client, messages, system, tools)
        print(f"  After filler turn {filler}: {token_count:,} tokens")

    print(
        f"\nFinal: {len(messages)} turns, {token_count:,} tokens (target was {TARGET_TOKENS:,})"
    )

    # Save the static conversation
    output = {
        "description": (
            "Static conversation history for compaction testing. "
            "Simulates an enterprise coding agent migrating FinCorp's OrderService "
            "from a Java monolith to a Spring Boot microservice with Kafka, PostgreSQL, "
            "and Kubernetes."
        ),
        "token_count": token_count,
        "system": system,
        "tools": tools,
        "messages": messages,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
