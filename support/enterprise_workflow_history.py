# ABOUTME: Generates a realistic enterprise agentic workflow conversation for compaction testing.
# ABOUTME: Simulates a coding agent migrating a Java monolith to Spring Boot microservices.

"""
This module builds a multi-turn conversation history that looks like a real enterprise
coding agent session. The scenario: an AI agent is helping migrate "OrderService" from
a legacy Java monolith (FinCorp ERP) into a standalone Spring Boot microservice with
Kafka event streaming, PostgreSQL, and Kubernetes deployment.

Each turn includes realistic tool_use / tool_result pairs for file reads, code edits,
shell commands, database queries, test runs, etc.
"""

import uuid


def _tool_id():
    """Generate a realistic tool_use id."""
    return f"toolu_{uuid.uuid4().hex[:24]}"


def build_system_prompt():
    """Return the system prompt for the enterprise coding agent."""
    return (
        "You are an expert enterprise software architect and coding agent. "
        "You have access to tools for reading files, editing code, running shell commands, "
        "querying databases, and searching codebases. You are helping migrate the OrderService "
        "module from the FinCorp ERP monolith into a standalone Spring Boot microservice. "
        "The target architecture uses Spring Boot 3.2, PostgreSQL 16, Apache Kafka for event "
        "streaming, Redis for caching, and Kubernetes for deployment. Follow enterprise coding "
        "standards: comprehensive error handling, audit logging, circuit breakers, and full "
        "test coverage including unit, integration, and contract tests."
    )


def build_tools():
    """Return the tool definitions for the enterprise agent."""
    return [
        {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path to read",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "edit_file",
            "description": "Apply an edit to a file. Replaces oldText with newText.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_text": {
                        "type": "string",
                        "description": "Text to find and replace",
                    },
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
        {
            "name": "run_command",
            "description": "Execute a shell command and return stdout/stderr.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory",
                    },
                },
                "required": ["command"],
            },
        },
        {
            "name": "search_codebase",
            "description": "Search the codebase for files matching a pattern or containing text.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or regex pattern",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern for files",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "query_database",
            "description": "Execute a read-only SQL query against the development database.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Database name"},
                },
                "required": ["sql"],
            },
        },
        {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    ]


def build_workflow_turns():
    """
    Build the full enterprise agentic workflow as a list of message turns.

    The scenario follows these phases:
    1. Discovery - analyze the monolith's OrderService
    2. Schema design - design the new microservice database schema
    3. Entity & repository layer - create JPA entities and Spring Data repos
    4. Service layer - business logic with Kafka event publishing
    5. API layer - REST controllers with OpenAPI docs
    6. Testing - unit tests, integration tests
    7. Kafka consumer - event-driven inventory sync
    8. Docker & K8s - containerization and deployment configs
    9. CI/CD pipeline - GitHub Actions workflow
    10. Final validation - end-to-end test run
    """
    turns = []

    # =========================================================================
    # PHASE 1: DISCOVERY — Analyze the monolith
    # =========================================================================

    # Turn 1: User kicks off the migration
    turns.append(
        {
            "role": "user",
            "content": (
                "We need to extract the OrderService from our FinCorp ERP monolith into a "
                "standalone microservice. The monolith is a Java 11 application using Spring "
                "Boot 2.7, Hibernate, and MySQL. The new service should use Spring Boot 3.2, "
                "PostgreSQL 16, and Kafka for event streaming. Start by analyzing the existing "
                "OrderService code to understand all dependencies."
            ),
        }
    )

    # Turn 2: Agent reads the main OrderService class
    tid1 = _tool_id()
    tid2 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll start by analyzing the existing OrderService in the monolith to understand "
                        "the full dependency graph. Let me read the main service class and search for all "
                        "related components."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid1,
                    "name": "read_file",
                    "input": {
                        "path": "/src/main/java/com/fincorp/erp/order/OrderService.java"
                    },
                },
                {
                    "type": "tool_use",
                    "id": tid2,
                    "name": "search_codebase",
                    "input": {
                        "query": "import com.fincorp.erp.order",
                        "file_pattern": "**/*.java",
                    },
                },
            ],
        }
    )

    # Turn 3: Tool results — the legacy OrderService code
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid1,
                    "content": """package com.fincorp.erp.order;

import com.fincorp.erp.customer.CustomerRepository;
import com.fincorp.erp.inventory.InventoryService;
import com.fincorp.erp.payment.PaymentGateway;
import com.fincorp.erp.shipping.ShippingCalculator;
import com.fincorp.erp.audit.AuditLogger;
import com.fincorp.erp.notification.NotificationService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import javax.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class OrderService {

    private final OrderRepository orderRepository;
    private final CustomerRepository customerRepository;
    private final InventoryService inventoryService;
    private final PaymentGateway paymentGateway;
    private final ShippingCalculator shippingCalculator;
    private final AuditLogger auditLogger;
    private final NotificationService notificationService;
    private final OrderValidator orderValidator;

    public OrderService(OrderRepository orderRepository,
                        CustomerRepository customerRepository,
                        InventoryService inventoryService,
                        PaymentGateway paymentGateway,
                        ShippingCalculator shippingCalculator,
                        AuditLogger auditLogger,
                        NotificationService notificationService,
                        OrderValidator orderValidator) {
        this.orderRepository = orderRepository;
        this.customerRepository = customerRepository;
        this.inventoryService = inventoryService;
        this.paymentGateway = paymentGateway;
        this.shippingCalculator = shippingCalculator;
        this.auditLogger = auditLogger;
        this.notificationService = notificationService;
        this.orderValidator = orderValidator;
    }

    @Transactional
    public Order createOrder(CreateOrderRequest request) {
        // Validate the order request
        orderValidator.validate(request);

        // Check customer exists and is active
        Customer customer = customerRepository.findById(request.getCustomerId())
            .orElseThrow(() -> new OrderException("Customer not found: " + request.getCustomerId()));
        if (!customer.isActive()) {
            throw new OrderException("Customer account is suspended: " + customer.getId());
        }

        // Check inventory availability for all items
        for (OrderItemRequest item : request.getItems()) {
            InventoryStatus status = inventoryService.checkAvailability(
                item.getProductId(), item.getQuantity());
            if (!status.isAvailable()) {
                throw new InsufficientInventoryException(
                    "Insufficient inventory for product " + item.getProductId()
                    + ": requested=" + item.getQuantity()
                    + ", available=" + status.getAvailableQuantity());
            }
        }

        // Calculate order totals
        BigDecimal subtotal = BigDecimal.ZERO;
        List<OrderItem> orderItems = new ArrayList<>();
        for (OrderItemRequest item : request.getItems()) {
            BigDecimal itemTotal = item.getUnitPrice().multiply(BigDecimal.valueOf(item.getQuantity()));
            subtotal = subtotal.add(itemTotal);
            orderItems.add(new OrderItem(item.getProductId(), item.getQuantity(),
                item.getUnitPrice(), itemTotal));
        }

        // Apply discount if applicable
        BigDecimal discount = calculateDiscount(customer, subtotal);
        BigDecimal taxableAmount = subtotal.subtract(discount);

        // Calculate tax based on shipping address
        BigDecimal tax = calculateTax(taxableAmount, request.getShippingAddress());

        // Calculate shipping cost
        BigDecimal shippingCost = shippingCalculator.calculate(
            request.getShippingAddress(), orderItems, request.getShippingMethod());

        BigDecimal totalAmount = taxableAmount.add(tax).add(shippingCost);

        // Create the order entity
        Order order = Order.builder()
            .customerId(customer.getId())
            .status(OrderStatus.PENDING_PAYMENT)
            .items(orderItems)
            .subtotal(subtotal)
            .discount(discount)
            .tax(tax)
            .shippingCost(shippingCost)
            .totalAmount(totalAmount)
            .shippingAddress(request.getShippingAddress())
            .billingAddress(request.getBillingAddress())
            .shippingMethod(request.getShippingMethod())
            .createdAt(LocalDateTime.now())
            .updatedAt(LocalDateTime.now())
            .build();

        order = orderRepository.save(order);

        // Process payment
        PaymentResult paymentResult = paymentGateway.processPayment(
            new PaymentRequest(order.getId(), totalAmount, request.getPaymentMethod()));

        if (paymentResult.isSuccessful()) {
            order.setStatus(OrderStatus.PAID);
            order.setPaymentTransactionId(paymentResult.getTransactionId());

            // Reserve inventory
            for (OrderItem item : orderItems) {
                inventoryService.reserveStock(item.getProductId(), item.getQuantity(), order.getId());
            }

            order.setStatus(OrderStatus.CONFIRMED);
        } else {
            order.setStatus(OrderStatus.PAYMENT_FAILED);
            order.setPaymentFailureReason(paymentResult.getFailureReason());
        }

        order.setUpdatedAt(LocalDateTime.now());
        order = orderRepository.save(order);

        // Audit log
        auditLogger.log("ORDER_CREATED", order.getId(), customer.getId(),
            Map.of("total", totalAmount, "items", orderItems.size()));

        // Send notifications
        if (order.getStatus() == OrderStatus.CONFIRMED) {
            notificationService.sendOrderConfirmation(customer.getEmail(), order);
        } else {
            notificationService.sendPaymentFailure(customer.getEmail(), order);
        }

        return order;
    }

    @Transactional
    public Order cancelOrder(Long orderId, String reason) {
        Order order = orderRepository.findById(orderId)
            .orElseThrow(() -> new OrderNotFoundException("Order not found: " + orderId));

        if (!order.getStatus().isCancellable()) {
            throw new OrderException("Order cannot be cancelled in status: " + order.getStatus());
        }

        // Release inventory reservations
        for (OrderItem item : order.getItems()) {
            inventoryService.releaseReservation(item.getProductId(), item.getQuantity(), orderId);
        }

        // Process refund if payment was made
        if (order.getPaymentTransactionId() != null) {
            paymentGateway.processRefund(order.getPaymentTransactionId(), order.getTotalAmount());
        }

        order.setStatus(OrderStatus.CANCELLED);
        order.setCancellationReason(reason);
        order.setUpdatedAt(LocalDateTime.now());
        order = orderRepository.save(order);

        auditLogger.log("ORDER_CANCELLED", orderId, order.getCustomerId(),
            Map.of("reason", reason));

        Customer customer = customerRepository.findById(order.getCustomerId()).orElse(null);
        if (customer != null) {
            notificationService.sendOrderCancellation(customer.getEmail(), order, reason);
        }

        return order;
    }

    public List<Order> getOrdersByCustomer(Long customerId, int page, int size) {
        return orderRepository.findByCustomerIdOrderByCreatedAtDesc(customerId,
            PageRequest.of(page, size));
    }

    public OrderSummary getOrderSummary(Long customerId) {
        List<Order> orders = orderRepository.findByCustomerId(customerId);
        BigDecimal totalSpent = orders.stream()
            .filter(o -> o.getStatus() == OrderStatus.CONFIRMED || o.getStatus() == OrderStatus.SHIPPED)
            .map(Order::getTotalAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        return new OrderSummary(customerId, orders.size(), totalSpent);
    }

    private BigDecimal calculateDiscount(Customer customer, BigDecimal subtotal) {
        if (customer.getLoyaltyTier() == LoyaltyTier.PLATINUM) {
            return subtotal.multiply(new BigDecimal("0.10"));
        } else if (customer.getLoyaltyTier() == LoyaltyTier.GOLD) {
            return subtotal.multiply(new BigDecimal("0.05"));
        }
        return BigDecimal.ZERO;
    }

    private BigDecimal calculateTax(BigDecimal amount, Address shippingAddress) {
        BigDecimal taxRate = TaxRateProvider.getTaxRate(shippingAddress.getState());
        return amount.multiply(taxRate);
    }
}""",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": tid2,
                    "content": """Found 23 files importing com.fincorp.erp.order:
  src/main/java/com/fincorp/erp/order/OrderController.java
  src/main/java/com/fincorp/erp/order/OrderRepository.java
  src/main/java/com/fincorp/erp/order/OrderValidator.java
  src/main/java/com/fincorp/erp/order/Order.java
  src/main/java/com/fincorp/erp/order/OrderItem.java
  src/main/java/com/fincorp/erp/order/OrderStatus.java
  src/main/java/com/fincorp/erp/order/CreateOrderRequest.java
  src/main/java/com/fincorp/erp/order/OrderException.java
  src/main/java/com/fincorp/erp/order/OrderNotFoundException.java
  src/main/java/com/fincorp/erp/order/InsufficientInventoryException.java
  src/main/java/com/fincorp/erp/order/OrderSummary.java
  src/main/java/com/fincorp/erp/reporting/SalesReportService.java
  src/main/java/com/fincorp/erp/reporting/RevenueAggregator.java
  src/main/java/com/fincorp/erp/fulfillment/FulfillmentService.java
  src/main/java/com/fincorp/erp/fulfillment/PickListGenerator.java
  src/main/java/com/fincorp/erp/returns/ReturnService.java
  src/main/java/com/fincorp/erp/analytics/OrderAnalyticsService.java
  src/main/java/com/fincorp/erp/dashboard/DashboardController.java
  src/main/java/com/fincorp/erp/export/OrderExportService.java
  src/main/java/com/fincorp/erp/integration/ERPSyncService.java
  src/main/java/com/fincorp/erp/webhook/OrderWebhookHandler.java
  src/test/java/com/fincorp/erp/order/OrderServiceTest.java
  src/test/java/com/fincorp/erp/order/OrderControllerTest.java""",
                },
            ],
        }
    )

    # Turn 4: Agent analyzes and reads the entity + DB schema
    tid3 = _tool_id()
    tid4 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Good — the OrderService has significant coupling to 6 other domain services "
                        "(Customer, Inventory, Payment, Shipping, Audit, Notification). For the microservice, "
                        "we'll replace direct dependencies with:\n\n"
                        "- **CustomerRepository** → REST client call to Customer service (or Kafka lookup)\n"
                        "- **InventoryService** → Kafka events for stock checks/reservations\n"
                        "- **PaymentGateway** → Async payment via Kafka + webhook callback\n"
                        "- **ShippingCalculator** → REST call to Shipping service\n"
                        "- **AuditLogger** → Kafka audit events\n"
                        "- **NotificationService** → Kafka notification events\n\n"
                        "Let me examine the Order entity and existing database schema to design the "
                        "PostgreSQL migration."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid3,
                    "name": "read_file",
                    "input": {
                        "path": "/src/main/java/com/fincorp/erp/order/Order.java"
                    },
                },
                {
                    "type": "tool_use",
                    "id": tid4,
                    "name": "query_database",
                    "input": {
                        "sql": "DESCRIBE orders; DESCRIBE order_items;",
                        "database": "fincorp_erp",
                    },
                },
            ],
        }
    )

    # Turn 5: Tool results — entity and DB schema
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid3,
                    "content": """package com.fincorp.erp.order;

import javax.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import lombok.*;

@Entity
@Table(name = "orders")
@Getter @Setter @Builder
@NoArgsConstructor @AllArgsConstructor
public class Order {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "customer_id", nullable = false)
    private Long customerId;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private OrderStatus status;

    @OneToMany(cascade = CascadeType.ALL, fetch = FetchType.EAGER, orphanRemoval = true)
    @JoinColumn(name = "order_id")
    private List<OrderItem> items;

    @Column(precision = 12, scale = 2)
    private BigDecimal subtotal;

    @Column(precision = 12, scale = 2)
    private BigDecimal discount;

    @Column(precision = 12, scale = 2)
    private BigDecimal tax;

    @Column(name = "shipping_cost", precision = 12, scale = 2)
    private BigDecimal shippingCost;

    @Column(name = "total_amount", precision = 12, scale = 2)
    private BigDecimal totalAmount;

    @Embedded
    @AttributeOverrides({
        @AttributeOverride(name = "street", column = @Column(name = "shipping_street")),
        @AttributeOverride(name = "city", column = @Column(name = "shipping_city")),
        @AttributeOverride(name = "state", column = @Column(name = "shipping_state")),
        @AttributeOverride(name = "zipCode", column = @Column(name = "shipping_zip")),
        @AttributeOverride(name = "country", column = @Column(name = "shipping_country"))
    })
    private Address shippingAddress;

    @Embedded
    @AttributeOverrides({
        @AttributeOverride(name = "street", column = @Column(name = "billing_street")),
        @AttributeOverride(name = "city", column = @Column(name = "billing_city")),
        @AttributeOverride(name = "state", column = @Column(name = "billing_state")),
        @AttributeOverride(name = "zipCode", column = @Column(name = "billing_zip")),
        @AttributeOverride(name = "country", column = @Column(name = "billing_country"))
    })
    private Address billingAddress;

    @Enumerated(EnumType.STRING)
    @Column(name = "shipping_method")
    private ShippingMethod shippingMethod;

    @Column(name = "payment_transaction_id")
    private String paymentTransactionId;

    @Column(name = "payment_failure_reason")
    private String paymentFailureReason;

    @Column(name = "cancellation_reason")
    private String cancellationReason;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;
}""",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": tid4,
                    "content": """Table: orders
+-------------------------+--------------+------+-----+---------+----------------+
| Field                   | Type         | Null | Key | Default | Extra          |
+-------------------------+--------------+------+-----+---------+----------------+
| id                      | bigint       | NO   | PRI | NULL    | auto_increment |
| customer_id             | bigint       | NO   | MUL | NULL    |                |
| status                  | varchar(50)  | NO   | MUL | NULL    |                |
| subtotal                | decimal(12,2)| YES  |     | NULL    |                |
| discount                | decimal(12,2)| YES  |     | NULL    |                |
| tax                     | decimal(12,2)| YES  |     | NULL    |                |
| shipping_cost           | decimal(12,2)| YES  |     | NULL    |                |
| total_amount            | decimal(12,2)| YES  |     | NULL    |                |
| shipping_street         | varchar(255) | YES  |     | NULL    |                |
| shipping_city           | varchar(100) | YES  |     | NULL    |                |
| shipping_state          | varchar(50)  | YES  |     | NULL    |                |
| shipping_zip            | varchar(20)  | YES  |     | NULL    |                |
| shipping_country        | varchar(50)  | YES  |     | NULL    |                |
| billing_street          | varchar(255) | YES  |     | NULL    |                |
| billing_city            | varchar(100) | YES  |     | NULL    |                |
| billing_state           | varchar(50)  | YES  |     | NULL    |                |
| billing_zip             | varchar(20)  | YES  |     | NULL    |                |
| billing_country         | varchar(50)  | YES  |     | NULL    |                |
| shipping_method         | varchar(50)  | YES  |     | NULL    |                |
| payment_transaction_id  | varchar(100) | YES  |     | NULL    |                |
| payment_failure_reason  | varchar(255) | YES  |     | NULL    |                |
| cancellation_reason     | varchar(500) | YES  |     | NULL    |                |
| created_at              | datetime     | NO   |     | NULL    |                |
| updated_at              | datetime     | NO   |     | NULL    |                |
+-------------------------+--------------+------+-----+---------+----------------+

Table: order_items
+----------------+--------------+------+-----+---------+----------------+
| Field          | Type         | Null | Key | Default | Extra          |
+----------------+--------------+------+-----+---------+----------------+
| id             | bigint       | NO   | PRI | NULL    | auto_increment |
| order_id       | bigint       | NO   | MUL | NULL    |                |
| product_id     | bigint       | NO   |     | NULL    |                |
| quantity        | int          | NO   |     | NULL    |                |
| unit_price     | decimal(12,2)| NO   |     | NULL    |                |
| total_price    | decimal(12,2)| NO   |     | NULL    |                |
+----------------+--------------+------+-----+---------+----------------+""",
                },
            ],
        }
    )

    # =========================================================================
    # PHASE 2: CREATE THE NEW MICROSERVICE PROJECT STRUCTURE
    # =========================================================================

    # Turn 6: Agent creates the new project structure
    tid5 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Now I have a clear picture of the existing schema. Let me create the new "
                        "microservice project structure and the Flyway migration for PostgreSQL. Key "
                        "changes from the monolith schema:\n\n"
                        "1. Use UUID primary keys instead of auto-increment bigints (better for distributed systems)\n"
                        "2. Add `version` column for optimistic locking\n"
                        "3. Add `idempotency_key` for safe retries\n"
                        "4. Normalize addresses into a separate table\n"
                        "5. Add event outbox table for reliable Kafka publishing\n\n"
                        "Let me create the Flyway migration first."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid5,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/src/main/resources/db/migration/V1__initial_schema.sql",
                        "content": """-- Order Service PostgreSQL Schema
-- Flyway migration V1: Initial schema for order-service microservice

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Orders table with UUID PKs and optimistic locking
CREATE TABLE orders (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idempotency_key         VARCHAR(64) UNIQUE,
    customer_id             UUID NOT NULL,
    status                  VARCHAR(50) NOT NULL DEFAULT 'PENDING_PAYMENT',
    subtotal                NUMERIC(12,2) NOT NULL,
    discount                NUMERIC(12,2) NOT NULL DEFAULT 0,
    tax                     NUMERIC(12,2) NOT NULL DEFAULT 0,
    shipping_cost           NUMERIC(12,2) NOT NULL DEFAULT 0,
    total_amount            NUMERIC(12,2) NOT NULL,
    shipping_method         VARCHAR(50),
    payment_transaction_id  VARCHAR(100),
    payment_failure_reason  VARCHAR(500),
    cancellation_reason     VARCHAR(500),
    version                 INTEGER NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              VARCHAR(100),
    updated_by              VARCHAR(100)
);

-- Order items
CREATE TABLE order_items (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id        UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id      UUID NOT NULL,
    product_name    VARCHAR(255) NOT NULL,
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    unit_price      NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    total_price     NUMERIC(12,2) NOT NULL CHECK (total_price >= 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Addresses (normalized)
CREATE TABLE order_addresses (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id        UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    address_type    VARCHAR(20) NOT NULL CHECK (address_type IN ('SHIPPING', 'BILLING')),
    street          VARCHAR(255) NOT NULL,
    city            VARCHAR(100) NOT NULL,
    state           VARCHAR(50) NOT NULL,
    zip_code        VARCHAR(20) NOT NULL,
    country         VARCHAR(50) NOT NULL DEFAULT 'US',
    UNIQUE(order_id, address_type)
);

-- Event outbox for reliable Kafka publishing (transactional outbox pattern)
CREATE TABLE event_outbox (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    aggregate_type  VARCHAR(100) NOT NULL,
    aggregate_id    UUID NOT NULL,
    event_type      VARCHAR(100) NOT NULL,
    payload         JSONB NOT NULL,
    published       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at    TIMESTAMPTZ
);

-- Indexes
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX idx_orders_idempotency ON orders(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_event_outbox_unpublished ON event_outbox(created_at) WHERE published = FALSE;
CREATE INDEX idx_event_outbox_aggregate ON event_outbox(aggregate_type, aggregate_id);""",
                    },
                },
            ],
        }
    )

    # Turn 7: Tool result and user asks to continue
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid5,
                    "content": "File written successfully: /order-service/src/main/resources/db/migration/V1__initial_schema.sql",
                },
                {
                    "type": "text",
                    "text": "Good. Now create the JPA entities, the Spring Data repositories, and the configuration.",
                },
            ],
        }
    )

    # =========================================================================
    # PHASE 3: ENTITY & REPOSITORY LAYER
    # =========================================================================

    # Turn 8: Agent creates the Order entity
    tid6 = _tool_id()
    tid7 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll create the JPA entities for the new microservice. Key differences from the monolith:\n"
                        "- UUID primary keys\n"
                        "- `@Version` for optimistic locking\n"
                        "- Auditable base entity with `@CreatedDate`/`@LastModifiedDate`\n"
                        "- Jakarta Persistence (not javax — Spring Boot 3.2)\n"
                        "- Immutable value objects for money amounts using `@Embeddable`"
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid6,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/src/main/java/com/fincorp/order/entity/OrderEntity.java",
                        "content": """package com.fincorp.order.entity;

import jakarta.persistence.*;
import org.springframework.data.annotation.CreatedBy;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedBy;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;
import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "orders")
@EntityListeners(AuditingEntityListener.class)
public class OrderEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "idempotency_key", unique = true, length = 64)
    private String idempotencyKey;

    @Column(name = "customer_id", nullable = false)
    private UUID customerId;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 50)
    private OrderStatus status = OrderStatus.PENDING_PAYMENT;

    @Column(precision = 12, scale = 2, nullable = false)
    private BigDecimal subtotal;

    @Column(precision = 12, scale = 2, nullable = false)
    private BigDecimal discount = BigDecimal.ZERO;

    @Column(precision = 12, scale = 2, nullable = false)
    private BigDecimal tax = BigDecimal.ZERO;

    @Column(name = "shipping_cost", precision = 12, scale = 2, nullable = false)
    private BigDecimal shippingCost = BigDecimal.ZERO;

    @Column(name = "total_amount", precision = 12, scale = 2, nullable = false)
    private BigDecimal totalAmount;

    @Enumerated(EnumType.STRING)
    @Column(name = "shipping_method", length = 50)
    private ShippingMethod shippingMethod;

    @Column(name = "payment_transaction_id", length = 100)
    private String paymentTransactionId;

    @Column(name = "payment_failure_reason", length = 500)
    private String paymentFailureReason;

    @Column(name = "cancellation_reason", length = 500)
    private String cancellationReason;

    @Version
    private Integer version = 0;

    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<OrderItemEntity> items = new ArrayList<>();

    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<OrderAddressEntity> addresses = new ArrayList<>();

    @CreatedDate
    @Column(name = "created_at", nullable = false, updatable = false)
    private OffsetDateTime createdAt;

    @LastModifiedDate
    @Column(name = "updated_at", nullable = false)
    private OffsetDateTime updatedAt;

    @CreatedBy
    @Column(name = "created_by", length = 100)
    private String createdBy;

    @LastModifiedBy
    @Column(name = "updated_by", length = 100)
    private String updatedBy;

    // Business methods
    public void addItem(OrderItemEntity item) {
        items.add(item);
        item.setOrder(this);
    }

    public void addAddress(OrderAddressEntity address) {
        addresses.add(address);
        address.setOrder(this);
    }

    public void confirm(String transactionId) {
        if (this.status != OrderStatus.PENDING_PAYMENT) {
            throw new IllegalStateException("Cannot confirm order in status: " + this.status);
        }
        this.paymentTransactionId = transactionId;
        this.status = OrderStatus.CONFIRMED;
    }

    public void cancel(String reason) {
        if (!this.status.isCancellable()) {
            throw new IllegalStateException("Cannot cancel order in status: " + this.status);
        }
        this.cancellationReason = reason;
        this.status = OrderStatus.CANCELLED;
    }

    public void markPaymentFailed(String reason) {
        this.paymentFailureReason = reason;
        this.status = OrderStatus.PAYMENT_FAILED;
    }

    // Getters and setters omitted for brevity — use Lombok @Data in real impl
    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public UUID getCustomerId() { return customerId; }
    public void setCustomerId(UUID customerId) { this.customerId = customerId; }
    public OrderStatus getStatus() { return status; }
    public void setStatus(OrderStatus status) { this.status = status; }
    public BigDecimal getTotalAmount() { return totalAmount; }
    public void setTotalAmount(BigDecimal totalAmount) { this.totalAmount = totalAmount; }
    public List<OrderItemEntity> getItems() { return items; }
    public String getIdempotencyKey() { return idempotencyKey; }
    public void setIdempotencyKey(String key) { this.idempotencyKey = key; }
    public BigDecimal getSubtotal() { return subtotal; }
    public void setSubtotal(BigDecimal subtotal) { this.subtotal = subtotal; }
    public void setDiscount(BigDecimal discount) { this.discount = discount; }
    public void setTax(BigDecimal tax) { this.tax = tax; }
    public void setShippingCost(BigDecimal cost) { this.shippingCost = cost; }
    public void setShippingMethod(ShippingMethod method) { this.shippingMethod = method; }
    public Integer getVersion() { return version; }
    public OffsetDateTime getCreatedAt() { return createdAt; }
    public String getPaymentTransactionId() { return paymentTransactionId; }
}""",
                    },
                },
                {
                    "type": "tool_use",
                    "id": tid7,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/src/main/java/com/fincorp/order/repository/OrderRepository.java",
                        "content": """package com.fincorp.order.repository;

import com.fincorp.order.entity.OrderEntity;
import com.fincorp.order.entity.OrderStatus;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface OrderRepository extends JpaRepository<OrderEntity, UUID> {

    Page<OrderEntity> findByCustomerIdOrderByCreatedAtDesc(UUID customerId, Pageable pageable);

    List<OrderEntity> findByCustomerId(UUID customerId);

    Optional<OrderEntity> findByIdempotencyKey(String idempotencyKey);

    List<OrderEntity> findByStatusAndCreatedAtBefore(OrderStatus status, OffsetDateTime cutoff);

    @Query("SELECT COUNT(o) FROM OrderEntity o WHERE o.customerId = :customerId AND o.status IN :statuses")
    long countByCustomerIdAndStatusIn(
        @Param("customerId") UUID customerId,
        @Param("statuses") List<OrderStatus> statuses);

    @Query("SELECT COALESCE(SUM(o.totalAmount), 0) FROM OrderEntity o " +
           "WHERE o.customerId = :customerId AND o.status IN ('CONFIRMED', 'SHIPPED', 'DELIVERED')")
    BigDecimal calculateTotalSpentByCustomer(@Param("customerId") UUID customerId);

    @Query("SELECT o FROM OrderEntity o LEFT JOIN FETCH o.items WHERE o.id = :id")
    Optional<OrderEntity> findByIdWithItems(@Param("id") UUID id);
}""",
                    },
                },
            ],
        }
    )

    # Turn 9: Tool results
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid6,
                    "content": "File written successfully: /order-service/src/main/java/com/fincorp/order/entity/OrderEntity.java",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": tid7,
                    "content": "File written successfully: /order-service/src/main/java/com/fincorp/order/repository/OrderRepository.java",
                },
                {
                    "type": "text",
                    "text": "Now create the service layer with Kafka event publishing.",
                },
            ],
        }
    )

    # =========================================================================
    # PHASE 4: SERVICE LAYER WITH KAFKA
    # =========================================================================

    # Turn 10: Agent creates the OrderService with Kafka events
    tid8 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Now I'll create the core service layer. The key architectural decision here is "
                        "using the Transactional Outbox Pattern for reliable Kafka event publishing. Instead "
                        "of publishing directly to Kafka (which risks data loss if Kafka is down), we:\n\n"
                        "1. Write the event to the `event_outbox` table in the same transaction as the order\n"
                        "2. A separate outbox poller reads unpublished events and sends them to Kafka\n"
                        "3. This guarantees at-least-once delivery\n\n"
                        "The service also uses circuit breakers (Resilience4j) for external service calls."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid8,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/src/main/java/com/fincorp/order/service/OrderService.java",
                        "content": """package com.fincorp.order.service;

import com.fincorp.order.entity.*;
import com.fincorp.order.repository.OrderRepository;
import com.fincorp.order.event.EventOutboxService;
import com.fincorp.order.event.OrderEvent;
import com.fincorp.order.client.CustomerServiceClient;
import com.fincorp.order.client.InventoryServiceClient;
import com.fincorp.order.client.ShippingServiceClient;
import com.fincorp.order.dto.*;
import com.fincorp.order.exception.*;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.math.BigDecimal;
import java.util.List;
import java.util.UUID;

@Service
public class OrderService {

    private static final Logger log = LoggerFactory.getLogger(OrderService.class);

    private final OrderRepository orderRepository;
    private final EventOutboxService eventOutboxService;
    private final CustomerServiceClient customerClient;
    private final InventoryServiceClient inventoryClient;
    private final ShippingServiceClient shippingClient;
    private final TaxCalculationService taxService;
    private final DiscountService discountService;

    public OrderService(OrderRepository orderRepository,
                        EventOutboxService eventOutboxService,
                        CustomerServiceClient customerClient,
                        InventoryServiceClient inventoryClient,
                        ShippingServiceClient shippingClient,
                        TaxCalculationService taxService,
                        DiscountService discountService) {
        this.orderRepository = orderRepository;
        this.eventOutboxService = eventOutboxService;
        this.customerClient = customerClient;
        this.inventoryClient = inventoryClient;
        this.shippingClient = shippingClient;
        this.taxService = taxService;
        this.discountService = discountService;
    }

    @Transactional
    @CircuitBreaker(name = "orderCreation", fallbackMethod = "createOrderFallback")
    @Retry(name = "orderCreation")
    public OrderResponse createOrder(CreateOrderRequest request) {
        log.info("Creating order for customer: {}", request.getCustomerId());

        // Idempotency check
        if (request.getIdempotencyKey() != null) {
            var existing = orderRepository.findByIdempotencyKey(request.getIdempotencyKey());
            if (existing.isPresent()) {
                log.info("Idempotent request detected, returning existing order: {}",
                    existing.get().getId());
                return OrderResponse.from(existing.get());
            }
        }

        // Validate customer exists via REST call to Customer service
        CustomerInfo customer = customerClient.getCustomer(request.getCustomerId());
        if (!customer.isActive()) {
            throw new OrderValidationException("Customer account is suspended");
        }

        // Check inventory availability via REST call to Inventory service
        for (OrderItemRequest item : request.getItems()) {
            InventoryStatus invStatus = inventoryClient.checkAvailability(
                item.getProductId(), item.getQuantity());
            if (!invStatus.isAvailable()) {
                throw new InsufficientInventoryException(
                    item.getProductId(), item.getQuantity(), invStatus.getAvailableQuantity());
            }
        }

        // Build order entity
        OrderEntity order = new OrderEntity();
        order.setCustomerId(request.getCustomerId());
        order.setIdempotencyKey(request.getIdempotencyKey());
        order.setShippingMethod(request.getShippingMethod());

        BigDecimal subtotal = BigDecimal.ZERO;
        for (OrderItemRequest item : request.getItems()) {
            OrderItemEntity orderItem = new OrderItemEntity();
            orderItem.setProductId(item.getProductId());
            orderItem.setProductName(item.getProductName());
            orderItem.setQuantity(item.getQuantity());
            orderItem.setUnitPrice(item.getUnitPrice());
            orderItem.setTotalPrice(item.getUnitPrice().multiply(BigDecimal.valueOf(item.getQuantity())));
            order.addItem(orderItem);
            subtotal = subtotal.add(orderItem.getTotalPrice());
        }

        // Calculate financials
        BigDecimal discount = discountService.calculateDiscount(customer.getLoyaltyTier(), subtotal);
        BigDecimal taxableAmount = subtotal.subtract(discount);
        BigDecimal tax = taxService.calculateTax(taxableAmount, request.getShippingAddress());
        BigDecimal shippingCost = shippingClient.calculateShipping(
            request.getShippingAddress(), order.getItems().size(), request.getShippingMethod());
        BigDecimal totalAmount = taxableAmount.add(tax).add(shippingCost);

        order.setSubtotal(subtotal);
        order.setDiscount(discount);
        order.setTax(tax);
        order.setShippingCost(shippingCost);
        order.setTotalAmount(totalAmount);

        // Save order
        order = orderRepository.save(order);
        log.info("Order created: {} with total: {}", order.getId(), totalAmount);

        // Publish OrderCreated event via outbox
        eventOutboxService.publishEvent(
            "Order", order.getId(),
            OrderEvent.ORDER_CREATED,
            new OrderCreatedPayload(order.getId(), order.getCustomerId(),
                order.getTotalAmount(), order.getItems().size()));

        // Publish InventoryReservationRequested events
        for (OrderItemEntity item : order.getItems()) {
            eventOutboxService.publishEvent(
                "Order", order.getId(),
                OrderEvent.INVENTORY_RESERVATION_REQUESTED,
                new InventoryReservationPayload(order.getId(), item.getProductId(),
                    item.getQuantity()));
        }

        // Publish PaymentRequested event
        eventOutboxService.publishEvent(
            "Order", order.getId(),
            OrderEvent.PAYMENT_REQUESTED,
            new PaymentRequestPayload(order.getId(), totalAmount,
                request.getPaymentMethod()));

        return OrderResponse.from(order);
    }

    @Transactional
    public OrderResponse cancelOrder(UUID orderId, String reason) {
        log.info("Cancelling order: {} reason: {}", orderId, reason);

        OrderEntity order = orderRepository.findByIdWithItems(orderId)
            .orElseThrow(() -> new OrderNotFoundException(orderId));

        order.cancel(reason);
        order = orderRepository.save(order);

        // Publish cancellation events
        eventOutboxService.publishEvent(
            "Order", orderId,
            OrderEvent.ORDER_CANCELLED,
            new OrderCancelledPayload(orderId, reason, order.getCustomerId()));

        // Request inventory release
        for (OrderItemEntity item : order.getItems()) {
            eventOutboxService.publishEvent(
                "Order", orderId,
                OrderEvent.INVENTORY_RELEASE_REQUESTED,
                new InventoryReleasePayload(orderId, item.getProductId(), item.getQuantity()));
        }

        // Request refund if payment was processed
        if (order.getPaymentTransactionId() != null) {
            eventOutboxService.publishEvent(
                "Order", orderId,
                OrderEvent.REFUND_REQUESTED,
                new RefundPayload(orderId, order.getPaymentTransactionId(),
                    order.getTotalAmount()));
        }

        return OrderResponse.from(order);
    }

    public Page<OrderResponse> getOrdersByCustomer(UUID customerId, int page, int size) {
        return orderRepository.findByCustomerIdOrderByCreatedAtDesc(
            customerId, PageRequest.of(page, size))
            .map(OrderResponse::from);
    }

    public OrderSummaryResponse getOrderSummary(UUID customerId) {
        BigDecimal totalSpent = orderRepository.calculateTotalSpentByCustomer(customerId);
        long orderCount = orderRepository.countByCustomerIdAndStatusIn(customerId,
            List.of(OrderStatus.CONFIRMED, OrderStatus.SHIPPED, OrderStatus.DELIVERED));
        return new OrderSummaryResponse(customerId, orderCount, totalSpent);
    }

    // Circuit breaker fallback
    private OrderResponse createOrderFallback(CreateOrderRequest request, Throwable t) {
        log.error("Order creation failed, circuit breaker activated: {}", t.getMessage());
        throw new ServiceUnavailableException("Order service is temporarily unavailable. " +
            "Please retry in a few moments.", t);
    }
}""",
                    },
                },
            ],
        }
    )

    # Turn 11: Tool result and user asks for REST controller
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid8,
                    "content": "File written successfully: /order-service/src/main/java/com/fincorp/order/service/OrderService.java",
                },
                {
                    "type": "text",
                    "text": "Now create the REST controller with OpenAPI annotations and proper error handling.",
                },
            ],
        }
    )

    # =========================================================================
    # PHASE 5: REST CONTROLLER
    # =========================================================================

    # Turn 12: Agent creates the REST controller
    tid9 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll create the REST controller with comprehensive OpenAPI/Swagger annotations, "
                        "request validation, proper HTTP status codes, and a global exception handler."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid9,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/src/main/java/com/fincorp/order/controller/OrderController.java",
                        "content": """package com.fincorp.order.controller;

import com.fincorp.order.dto.*;
import com.fincorp.order.service.OrderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/orders")
@Tag(name = "Orders", description = "Order management endpoints")
public class OrderController {

    private static final Logger log = LoggerFactory.getLogger(OrderController.class);
    private final OrderService orderService;

    public OrderController(OrderService orderService) {
        this.orderService = orderService;
    }

    @PostMapping
    @Operation(summary = "Create a new order", description = "Creates a new order and initiates payment processing")
    @ApiResponses({
        @ApiResponse(responseCode = "201", description = "Order created successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid order request"),
        @ApiResponse(responseCode = "409", description = "Duplicate idempotency key"),
        @ApiResponse(responseCode = "422", description = "Insufficient inventory"),
        @ApiResponse(responseCode = "503", description = "Service temporarily unavailable")
    })
    public ResponseEntity<OrderResponse> createOrder(
            @Valid @RequestBody CreateOrderRequest request,
            @RequestHeader(value = "X-Idempotency-Key", required = false) String idempotencyKey,
            @RequestHeader(value = "X-Correlation-Id", required = false) String correlationId) {
        log.info("POST /api/v1/orders correlationId={} idempotencyKey={}", correlationId, idempotencyKey);
        if (idempotencyKey != null) {
            request.setIdempotencyKey(idempotencyKey);
        }
        OrderResponse response = orderService.createOrder(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping("/{orderId}")
    @Operation(summary = "Get order by ID")
    @ApiResponses({
        @ApiResponse(responseCode = "200", description = "Order found"),
        @ApiResponse(responseCode = "404", description = "Order not found")
    })
    public ResponseEntity<OrderResponse> getOrder(
            @Parameter(description = "Order UUID") @PathVariable UUID orderId) {
        log.info("GET /api/v1/orders/{}", orderId);
        return ResponseEntity.ok(orderService.getOrder(orderId));
    }

    @PostMapping("/{orderId}/cancel")
    @Operation(summary = "Cancel an order")
    @ApiResponses({
        @ApiResponse(responseCode = "200", description = "Order cancelled"),
        @ApiResponse(responseCode = "404", description = "Order not found"),
        @ApiResponse(responseCode = "409", description = "Order cannot be cancelled in current status")
    })
    public ResponseEntity<OrderResponse> cancelOrder(
            @PathVariable UUID orderId,
            @Valid @RequestBody CancelOrderRequest request) {
        log.info("POST /api/v1/orders/{}/cancel", orderId);
        return ResponseEntity.ok(orderService.cancelOrder(orderId, request.getReason()));
    }

    @GetMapping("/customer/{customerId}")
    @Operation(summary = "Get orders by customer")
    public ResponseEntity<Page<OrderResponse>> getOrdersByCustomer(
            @PathVariable UUID customerId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        return ResponseEntity.ok(orderService.getOrdersByCustomer(customerId, page, size));
    }

    @GetMapping("/customer/{customerId}/summary")
    @Operation(summary = "Get order summary for customer")
    public ResponseEntity<OrderSummaryResponse> getOrderSummary(@PathVariable UUID customerId) {
        return ResponseEntity.ok(orderService.getOrderSummary(customerId));
    }
}""",
                    },
                },
            ],
        }
    )

    # Turn 13: Tool result and user asks for tests
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid9,
                    "content": "File written successfully: /order-service/src/main/java/com/fincorp/order/controller/OrderController.java",
                },
                {
                    "type": "text",
                    "text": "Write the unit tests for OrderService and integration tests for the API.",
                },
            ],
        }
    )

    # =========================================================================
    # PHASE 6: TESTING
    # =========================================================================

    # Turn 14: Agent creates unit tests
    tid10 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll write comprehensive tests. The unit tests will mock all external dependencies "
                        "(Kafka, external services), and the integration tests will use Testcontainers for "
                        "PostgreSQL and an embedded Kafka broker."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid10,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/src/test/java/com/fincorp/order/service/OrderServiceTest.java",
                        "content": """package com.fincorp.order.service;

import com.fincorp.order.client.CustomerServiceClient;
import com.fincorp.order.client.InventoryServiceClient;
import com.fincorp.order.client.ShippingServiceClient;
import com.fincorp.order.dto.*;
import com.fincorp.order.entity.*;
import com.fincorp.order.event.EventOutboxService;
import com.fincorp.order.exception.*;
import com.fincorp.order.repository.OrderRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class OrderServiceTest {

    @Mock private OrderRepository orderRepository;
    @Mock private EventOutboxService eventOutboxService;
    @Mock private CustomerServiceClient customerClient;
    @Mock private InventoryServiceClient inventoryClient;
    @Mock private ShippingServiceClient shippingClient;
    @Mock private TaxCalculationService taxService;
    @Mock private DiscountService discountService;

    @InjectMocks private OrderService orderService;

    private UUID customerId;
    private UUID productId;
    private CreateOrderRequest validRequest;

    @BeforeEach
    void setUp() {
        customerId = UUID.randomUUID();
        productId = UUID.randomUUID();
        validRequest = CreateOrderRequest.builder()
            .customerId(customerId)
            .items(List.of(OrderItemRequest.builder()
                .productId(productId)
                .productName("Enterprise Widget")
                .quantity(2)
                .unitPrice(new BigDecimal("49.99"))
                .build()))
            .shippingAddress(new AddressDto("123 Main St", "Springfield", "IL", "62701", "US"))
            .billingAddress(new AddressDto("123 Main St", "Springfield", "IL", "62701", "US"))
            .shippingMethod(ShippingMethod.STANDARD)
            .paymentMethod("CREDIT_CARD")
            .idempotencyKey("idem-" + UUID.randomUUID())
            .build();
    }

    @Nested
    @DisplayName("createOrder")
    class CreateOrderTests {

        @Test
        @DisplayName("should create order successfully with valid request")
        void shouldCreateOrderSuccessfully() {
            // Arrange
            when(orderRepository.findByIdempotencyKey(anyString())).thenReturn(Optional.empty());
            when(customerClient.getCustomer(customerId))
                .thenReturn(new CustomerInfo(customerId, "ACME Corp", true, "GOLD"));
            when(inventoryClient.checkAvailability(productId, 2))
                .thenReturn(new InventoryStatus(true, 100));
            when(discountService.calculateDiscount(eq("GOLD"), any(BigDecimal.class)))
                .thenReturn(new BigDecimal("5.00"));
            when(taxService.calculateTax(any(), any())).thenReturn(new BigDecimal("8.55"));
            when(shippingClient.calculateShipping(any(), anyInt(), any()))
                .thenReturn(new BigDecimal("9.99"));
            when(orderRepository.save(any(OrderEntity.class))).thenAnswer(invocation -> {
                OrderEntity saved = invocation.getArgument(0);
                saved.setId(UUID.randomUUID());
                return saved;
            });

            // Act
            OrderResponse response = orderService.createOrder(validRequest);

            // Assert
            assertThat(response).isNotNull();
            assertThat(response.getStatus()).isEqualTo(OrderStatus.PENDING_PAYMENT);
            verify(orderRepository).save(any(OrderEntity.class));
            verify(eventOutboxService, times(3)).publishEvent(
                eq("Order"), any(), any(), any());
        }

        @Test
        @DisplayName("should return existing order for duplicate idempotency key")
        void shouldReturnExistingForDuplicateIdempotencyKey() {
            OrderEntity existing = new OrderEntity();
            existing.setId(UUID.randomUUID());
            existing.setCustomerId(customerId);
            existing.setStatus(OrderStatus.CONFIRMED);
            existing.setTotalAmount(new BigDecimal("100.00"));
            when(orderRepository.findByIdempotencyKey(validRequest.getIdempotencyKey()))
                .thenReturn(Optional.of(existing));

            OrderResponse response = orderService.createOrder(validRequest);

            assertThat(response.getId()).isEqualTo(existing.getId());
            verify(customerClient, never()).getCustomer(any());
            verify(orderRepository, never()).save(any());
        }

        @Test
        @DisplayName("should throw when customer is suspended")
        void shouldThrowWhenCustomerSuspended() {
            when(orderRepository.findByIdempotencyKey(anyString())).thenReturn(Optional.empty());
            when(customerClient.getCustomer(customerId))
                .thenReturn(new CustomerInfo(customerId, "Suspended Corp", false, "NONE"));

            assertThatThrownBy(() -> orderService.createOrder(validRequest))
                .isInstanceOf(OrderValidationException.class)
                .hasMessageContaining("suspended");
        }

        @Test
        @DisplayName("should throw when inventory is insufficient")
        void shouldThrowWhenInventoryInsufficient() {
            when(orderRepository.findByIdempotencyKey(anyString())).thenReturn(Optional.empty());
            when(customerClient.getCustomer(customerId))
                .thenReturn(new CustomerInfo(customerId, "ACME Corp", true, "GOLD"));
            when(inventoryClient.checkAvailability(productId, 2))
                .thenReturn(new InventoryStatus(false, 1));

            assertThatThrownBy(() -> orderService.createOrder(validRequest))
                .isInstanceOf(InsufficientInventoryException.class);
        }
    }

    @Nested
    @DisplayName("cancelOrder")
    class CancelOrderTests {

        @Test
        @DisplayName("should cancel order and publish events")
        void shouldCancelAndPublishEvents() {
            UUID orderId = UUID.randomUUID();
            OrderEntity order = new OrderEntity();
            order.setId(orderId);
            order.setCustomerId(customerId);
            order.setStatus(OrderStatus.CONFIRMED);
            order.setTotalAmount(new BigDecimal("100.00"));
            order.setPaymentTransactionId("txn_123");
            when(orderRepository.findByIdWithItems(orderId)).thenReturn(Optional.of(order));
            when(orderRepository.save(any())).thenAnswer(i -> i.getArgument(0));

            OrderResponse response = orderService.cancelOrder(orderId, "Customer requested");

            assertThat(response.getStatus()).isEqualTo(OrderStatus.CANCELLED);
            // Verify: cancellation event + refund event (no items so no inventory release)
            verify(eventOutboxService, atLeast(2)).publishEvent(
                eq("Order"), eq(orderId), any(), any());
        }

        @Test
        @DisplayName("should throw when order not found")
        void shouldThrowWhenOrderNotFound() {
            UUID orderId = UUID.randomUUID();
            when(orderRepository.findByIdWithItems(orderId)).thenReturn(Optional.empty());

            assertThatThrownBy(() -> orderService.cancelOrder(orderId, "reason"))
                .isInstanceOf(OrderNotFoundException.class);
        }
    }
}""",
                    },
                },
            ],
        }
    )

    # Turn 15: Tool result and user asks to run the tests
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid10,
                    "content": "File written successfully: /order-service/src/test/java/com/fincorp/order/service/OrderServiceTest.java",
                },
                {
                    "type": "text",
                    "text": "Run the tests to see if they compile and pass.",
                },
            ],
        }
    )

    # Turn 16: Agent runs tests
    tid11 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Let me run the test suite to verify compilation and test results.",
                },
                {
                    "type": "tool_use",
                    "id": tid11,
                    "name": "run_command",
                    "input": {
                        "command": "./gradlew test --info 2>&1 | tail -80",
                        "working_dir": "/order-service",
                    },
                },
            ],
        }
    )

    # Turn 17: Test results (with some failures to make it realistic)
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid11,
                    "content": """> Task :compileJava
> Task :compileTestJava
> Task :test

OrderServiceTest > CreateOrderTests > should create order successfully with valid request PASSED
OrderServiceTest > CreateOrderTests > should return existing order for duplicate idempotency key PASSED
OrderServiceTest > CreateOrderTests > should throw when customer is suspended PASSED
OrderServiceTest > CreateOrderTests > should throw when inventory is insufficient PASSED
OrderServiceTest > CancelOrderTests > should cancel order and publish events FAILED
    org.mockito.exceptions.misusing.UnnecessaryStubbingException:
    Unnecessary stubbings detected in test class:
      - orderRepository.save(...)
    at com.fincorp.order.service.OrderServiceTest$CancelOrderTests.shouldCancelAndPublishEvents

OrderServiceTest > CancelOrderTests > should throw when order not found PASSED

5 tests completed, 1 failed

FAILURE: Build failed with an exception.
* What went wrong:
Execution failed for task ':test'.
> There were failing tests. See the report at: file:///order-service/build/reports/tests/test/index.html""",
                }
            ],
        }
    )

    # Turn 18: Agent fixes the failing test
    tid12 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The test failure is due to Mockito's strict stubbing — the `orderRepository.save()` "
                        "stub isn't needed because `cancel()` modifies the entity in-place and the real "
                        "save happens inside the `@Transactional` boundary. Let me fix the test."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid12,
                    "name": "edit_file",
                    "input": {
                        "path": "/order-service/src/test/java/com/fincorp/order/service/OrderServiceTest.java",
                        "old_text": """            when(orderRepository.findByIdWithItems(orderId)).thenReturn(Optional.of(order));
            when(orderRepository.save(any())).thenAnswer(i -> i.getArgument(0));""",
                        "new_text": """            when(orderRepository.findByIdWithItems(orderId)).thenReturn(Optional.of(order));
            // save() is called in the transactional method; use lenient stubbing
            lenient().when(orderRepository.save(any())).thenAnswer(i -> i.getArgument(0));""",
                    },
                },
            ],
        }
    )

    # Turn 19: Tool result and user asks to re-run
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid12,
                    "content": "Edit applied successfully.",
                },
                {"type": "text", "text": "Re-run the tests."},
            ],
        }
    )

    # Turn 20: Agent re-runs tests
    tid13 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Running tests again after the fix."},
                {
                    "type": "tool_use",
                    "id": tid13,
                    "name": "run_command",
                    "input": {
                        "command": "./gradlew test --info 2>&1 | tail -40",
                        "working_dir": "/order-service",
                    },
                },
            ],
        }
    )

    # Turn 21: All tests pass
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid13,
                    "content": """> Task :compileJava UP-TO-DATE
> Task :compileTestJava
> Task :test

OrderServiceTest > CreateOrderTests > should create order successfully with valid request PASSED
OrderServiceTest > CreateOrderTests > should return existing order for duplicate idempotency key PASSED
OrderServiceTest > CreateOrderTests > should throw when customer is suspended PASSED
OrderServiceTest > CreateOrderTests > should throw when inventory is insufficient PASSED
OrderServiceTest > CancelOrderTests > should cancel order and publish events PASSED
OrderServiceTest > CancelOrderTests > should throw when order not found PASSED

6 tests completed, 0 failed

BUILD SUCCESSFUL in 8s""",
                },
                {
                    "type": "text",
                    "text": "All passing. Now create the Kafka consumer for inventory sync, plus the Docker and Kubernetes configs.",
                },
            ],
        }
    )

    # =========================================================================
    # PHASE 7: KAFKA CONSUMER + DOCKER + K8S
    # =========================================================================

    # Turn 22: Agent creates Kafka consumer and Docker/K8s configs
    tid14 = _tool_id()
    tid15 = _tool_id()
    tid16 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll create the Kafka event consumer for handling payment confirmations and "
                        "inventory updates, plus the Docker and Kubernetes deployment configurations."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid14,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/src/main/java/com/fincorp/order/consumer/OrderEventConsumer.java",
                        "content": """package com.fincorp.order.consumer;

import com.fincorp.order.entity.OrderEntity;
import com.fincorp.order.entity.OrderStatus;
import com.fincorp.order.event.EventOutboxService;
import com.fincorp.order.event.OrderEvent;
import com.fincorp.order.repository.OrderRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import java.util.UUID;

@Component
public class OrderEventConsumer {

    private static final Logger log = LoggerFactory.getLogger(OrderEventConsumer.class);
    private final OrderRepository orderRepository;
    private final EventOutboxService eventOutboxService;
    private final ObjectMapper objectMapper;

    public OrderEventConsumer(OrderRepository orderRepository,
                              EventOutboxService eventOutboxService,
                              ObjectMapper objectMapper) {
        this.orderRepository = orderRepository;
        this.eventOutboxService = eventOutboxService;
        this.objectMapper = objectMapper;
    }

    @KafkaListener(topics = "payment.results", groupId = "order-service",
                    containerFactory = "kafkaListenerContainerFactory")
    @Transactional
    public void handlePaymentResult(ConsumerRecord<String, String> record, Acknowledgment ack) {
        try {
            JsonNode payload = objectMapper.readTree(record.value());
            UUID orderId = UUID.fromString(payload.get("orderId").asText());
            String status = payload.get("status").asText();
            String transactionId = payload.path("transactionId").asText(null);

            log.info("Received payment result for order {}: status={}", orderId, status);

            OrderEntity order = orderRepository.findById(orderId)
                .orElseThrow(() -> new RuntimeException("Order not found: " + orderId));

            if ("SUCCESS".equals(status)) {
                order.confirm(transactionId);
                orderRepository.save(order);

                eventOutboxService.publishEvent("Order", orderId,
                    OrderEvent.ORDER_CONFIRMED,
                    new OrderConfirmedPayload(orderId, transactionId));

                log.info("Order {} confirmed with transaction {}", orderId, transactionId);
            } else {
                String reason = payload.path("failureReason").asText("Unknown payment failure");
                order.markPaymentFailed(reason);
                orderRepository.save(order);

                eventOutboxService.publishEvent("Order", orderId,
                    OrderEvent.PAYMENT_FAILED,
                    new PaymentFailedPayload(orderId, reason));

                log.warn("Order {} payment failed: {}", orderId, reason);
            }

            ack.acknowledge();
        } catch (Exception e) {
            log.error("Error processing payment result: {}", e.getMessage(), e);
            // Don't acknowledge — message will be retried
        }
    }

    @KafkaListener(topics = "inventory.updates", groupId = "order-service")
    @Transactional
    public void handleInventoryUpdate(ConsumerRecord<String, String> record, Acknowledgment ack) {
        try {
            JsonNode payload = objectMapper.readTree(record.value());
            UUID orderId = UUID.fromString(payload.get("orderId").asText());
            String status = payload.get("status").asText();

            log.info("Received inventory update for order {}: {}", orderId, status);

            if ("RESERVED".equals(status)) {
                log.info("Inventory reserved for order {}", orderId);
            } else if ("INSUFFICIENT".equals(status)) {
                OrderEntity order = orderRepository.findById(orderId).orElse(null);
                if (order != null && order.getStatus() == OrderStatus.PENDING_PAYMENT) {
                    order.cancel("Insufficient inventory");
                    orderRepository.save(order);
                    log.warn("Order {} cancelled due to insufficient inventory", orderId);
                }
            }

            ack.acknowledge();
        } catch (Exception e) {
            log.error("Error processing inventory update: {}", e.getMessage(), e);
        }
    }
}""",
                    },
                },
                {
                    "type": "tool_use",
                    "id": tid15,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/Dockerfile",
                        "content": """# Multi-stage build for order-service
FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /app
COPY gradle gradle
COPY gradlew .
COPY build.gradle.kts settings.gradle.kts ./
RUN ./gradlew dependencies --no-daemon || true
COPY src src
RUN ./gradlew bootJar --no-daemon -x test

FROM eclipse-temurin:21-jre-alpine
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
WORKDIR /app
COPY --from=builder /app/build/libs/order-service-*.jar app.jar
RUN chown -R appuser:appgroup /app
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1
ENTRYPOINT ["java", "-XX:+UseG1GC", "-XX:MaxRAMPercentage=75.0", \
            "-Djava.security.egd=file:/dev/./urandom", "-jar", "app.jar"]""",
                    },
                },
                {
                    "type": "tool_use",
                    "id": tid16,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/k8s/deployment.yaml",
                        "content": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: fincorp
  labels:
    app: order-service
    team: order-platform
    environment: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/actuator/prometheus"
    spec:
      serviceAccountName: order-service
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: order-service
        image: registry.fincorp.internal/order-service:latest
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: "production"
        - name: SPRING_DATASOURCE_URL
          valueFrom:
            secretKeyRef:
              name: order-service-db
              key: jdbc-url
        - name: SPRING_DATASOURCE_USERNAME
          valueFrom:
            secretKeyRef:
              name: order-service-db
              key: username
        - name: SPRING_DATASOURCE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: order-service-db
              key: password
        - name: SPRING_KAFKA_BOOTSTRAP_SERVERS
          valueFrom:
            configMapKeyRef:
              name: kafka-config
              key: bootstrap-servers
        - name: JAVA_OPTS
          value: "-XX:+UseG1GC -XX:MaxRAMPercentage=75.0 -Xms512m"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        readinessProbe:
          httpGet:
            path: /actuator/health/readiness
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /actuator/health/liveness
            port: http
          initialDelaySeconds: 60
          periodSeconds: 30
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /actuator/health
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 30
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: order-service
---
apiVersion: v1
kind: Service
metadata:
  name: order-service
  namespace: fincorp
spec:
  selector:
    app: order-service
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service
  namespace: fincorp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60""",
                    },
                },
            ],
        }
    )

    # Turn 23: Tool results
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid14,
                    "content": "File written: /order-service/src/main/java/com/fincorp/order/consumer/OrderEventConsumer.java",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": tid15,
                    "content": "File written: /order-service/Dockerfile",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": tid16,
                    "content": "File written: /order-service/k8s/deployment.yaml",
                },
                {
                    "type": "text",
                    "text": "Now set up the CI/CD pipeline with GitHub Actions, then do a final integration test.",
                },
            ],
        }
    )

    # =========================================================================
    # PHASE 8: CI/CD PIPELINE
    # =========================================================================

    # Turn 24: Agent creates GitHub Actions workflow
    tid17 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll create a comprehensive GitHub Actions CI/CD pipeline with:\n"
                        "- Build and unit test stage\n"
                        "- Integration test stage with Testcontainers\n"
                        "- Container image build and push\n"
                        "- Security scanning with Trivy\n"
                        "- Deployment to staging with smoke tests\n"
                        "- Production deployment with approval gate"
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid17,
                    "name": "write_file",
                    "input": {
                        "path": "/order-service/.github/workflows/ci-cd.yml",
                        "content": """name: Order Service CI/CD

on:
  push:
    branches: [main, develop]
    paths:
      - 'order-service/**'
  pull_request:
    branches: [main]
    paths:
      - 'order-service/**'

env:
  REGISTRY: registry.fincorp.internal
  IMAGE_NAME: order-service
  JAVA_VERSION: '21'

jobs:
  build-and-test:
    name: Build & Unit Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          java-version: ${{ env.JAVA_VERSION }}
          distribution: temurin
          cache: gradle
      - name: Build and test
        working-directory: order-service
        run: ./gradlew build test jacocoTestReport
      - name: Check code coverage
        working-directory: order-service
        run: |
          COVERAGE=$(grep -oP 'Total.*?([0-9]+)%' build/reports/jacoco/test/html/index.html | grep -oP '[0-9]+' | tail -1)
          echo "Code coverage: ${COVERAGE}%"
          if [ "$COVERAGE" -lt 80 ]; then
            echo "::error::Code coverage ${COVERAGE}% is below 80% threshold"
            exit 1
          fi
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: order-service/build/reports/

  integration-test:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: build-and-test
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: order_service_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          java-version: ${{ env.JAVA_VERSION }}
          distribution: temurin
          cache: gradle
      - name: Run integration tests
        working-directory: order-service
        env:
          SPRING_DATASOURCE_URL: jdbc:postgresql://localhost:5432/order_service_test
          SPRING_DATASOURCE_USERNAME: test
          SPRING_DATASOURCE_PASSWORD: test
        run: ./gradlew integrationTest

  container-build:
    name: Build & Push Container
    runs-on: ubuntu-latest
    needs: [build-and-test, integration-test]
    if: github.ref == 'refs/heads/main'
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Login to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ secrets.REGISTRY_USERNAME }}
          password: ${{ secrets.REGISTRY_PASSWORD }}
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: order-service
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: container-build
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: sarif
          output: trivy-results.sarif
          severity: CRITICAL,HIGH
      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [container-build, security-scan]
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to staging
        run: |
          kubectl set image deployment/order-service \
            order-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n fincorp-staging
          kubectl rollout status deployment/order-service -n fincorp-staging --timeout=300s
      - name: Smoke test
        run: |
          STAGING_URL="https://staging-api.fincorp.internal"
          HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${STAGING_URL}/actuator/health")
          if [ "$HTTP_CODE" != "200" ]; then
            echo "::error::Smoke test failed with HTTP $HTTP_CODE"
            exit 1
          fi

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to production
        run: |
          kubectl set image deployment/order-service \
            order-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n fincorp
          kubectl rollout status deployment/order-service -n fincorp --timeout=600s""",
                    },
                },
            ],
        }
    )

    # Turn 25: Tool result and final integration test
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid17,
                    "content": "File written: /order-service/.github/workflows/ci-cd.yml",
                },
                {
                    "type": "text",
                    "text": "Run the full integration test suite with Docker to make sure everything works end-to-end.",
                },
            ],
        }
    )

    # =========================================================================
    # PHASE 9: FINAL VALIDATION
    # =========================================================================

    # Turn 26: Agent runs full integration tests
    tid18 = _tool_id()
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Let me run the full integration test suite using Docker Compose to spin up "
                        "PostgreSQL, Kafka, and Redis, then run the complete test pipeline."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": tid18,
                    "name": "run_command",
                    "input": {
                        "command": "docker compose -f docker-compose.test.yml up -d && sleep 10 && ./gradlew integrationTest --info 2>&1 | tail -60",
                        "working_dir": "/order-service",
                    },
                },
            ],
        }
    )

    # Turn 27: Integration test results
    turns.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid18,
                    "content": """[+] Running 3/3
 ✔ Container postgres-test    Started   1.2s
 ✔ Container kafka-test       Started   2.8s
 ✔ Container redis-test       Started   0.9s

> Task :integrationTest

OrderControllerIntegrationTest > should create order via REST API PASSED
OrderControllerIntegrationTest > should return 404 for non-existent order PASSED
OrderControllerIntegrationTest > should cancel order via REST API PASSED
OrderControllerIntegrationTest > should return paginated orders for customer PASSED
OrderControllerIntegrationTest > should handle idempotent order creation PASSED

OrderEventConsumerIntegrationTest > should process successful payment PASSED
OrderEventConsumerIntegrationTest > should handle payment failure PASSED
OrderEventConsumerIntegrationTest > should handle insufficient inventory PASSED

EventOutboxIntegrationTest > should publish events to Kafka PASSED
EventOutboxIntegrationTest > should mark events as published PASSED
EventOutboxIntegrationTest > should handle duplicate events idempotently PASSED

FlywayMigrationTest > should apply all migrations successfully PASSED
FlywayMigrationTest > should create expected tables PASSED

13 tests completed, 0 failed

BUILD SUCCESSFUL in 45s""",
                },
                {
                    "type": "text",
                    "text": (
                        "All integration tests pass. Great work! Can you give me a summary of everything "
                        "we've built and what the next steps would be for production readiness?"
                    ),
                },
            ],
        }
    )

    # Turn 28: Agent provides final summary (this is the last assistant turn
    # before compaction kicks in — the user's next message triggers it)
    turns.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Here's a complete summary of the OrderService microservice migration:\n\n"
                        "## What We Built\n\n"
                        "### Database Layer\n"
                        "- PostgreSQL 16 schema with UUID primary keys, optimistic locking, and audit columns\n"
                        "- Flyway migrations for schema versioning\n"
                        "- Transactional outbox table for reliable Kafka event publishing\n"
                        "- Normalized address table (shipping/billing separated from orders)\n\n"
                        "### Entity & Repository Layer\n"
                        "- JPA entities using Jakarta Persistence (Spring Boot 3.2 compatible)\n"
                        "- Spring Data JPA repositories with custom queries\n"
                        "- Auditing support via @CreatedDate/@LastModifiedDate\n\n"
                        "### Service Layer\n"
                        "- OrderService with full CRUD operations\n"
                        "- Idempotency support via idempotency keys\n"
                        "- Circuit breakers (Resilience4j) for external service calls\n"
                        "- Transactional outbox pattern for Kafka event publishing\n"
                        "- Events: ORDER_CREATED, ORDER_CONFIRMED, ORDER_CANCELLED, PAYMENT_REQUESTED, "
                        "INVENTORY_RESERVATION_REQUESTED, REFUND_REQUESTED\n\n"
                        "### API Layer\n"
                        "- REST controller with OpenAPI/Swagger annotations\n"
                        "- Endpoints: POST /orders, GET /orders/{id}, POST /orders/{id}/cancel, "
                        "GET /orders/customer/{id}, GET /orders/customer/{id}/summary\n"
                        "- Proper HTTP status codes and correlation ID tracking\n\n"
                        "### Event Consumers\n"
                        "- Kafka consumer for payment results (confirms or fails orders)\n"
                        "- Kafka consumer for inventory updates (cancels on insufficient stock)\n"
                        "- Manual acknowledgment for at-least-once processing\n\n"
                        "### Infrastructure\n"
                        "- Multi-stage Docker build with security hardening\n"
                        "- Kubernetes deployment with 3 replicas, HPA, topology spread\n"
                        "- Health checks (readiness, liveness, startup probes)\n"
                        "- Prometheus metrics exposure\n\n"
                        "### CI/CD\n"
                        "- GitHub Actions pipeline: build → unit test → integration test → "
                        "container build → security scan → staging → production\n"
                        "- 80% code coverage gate\n"
                        "- Trivy security scanning\n\n"
                        "### Testing\n"
                        "- 6 unit tests covering order creation, idempotency, validation, and cancellation\n"
                        "- 13 integration tests covering REST API, Kafka consumers, outbox, and migrations\n"
                        "- All tests passing\n\n"
                        "## Remaining Work for Production Readiness\n\n"
                        "1. **Observability**: Add distributed tracing (OpenTelemetry), structured logging, "
                        "and Grafana dashboards\n"
                        "2. **Security**: Implement OAuth2/JWT authentication, API rate limiting, "
                        "and input sanitization\n"
                        "3. **Data Migration**: Write the data migration script to move existing MySQL "
                        "orders to PostgreSQL with UUID mapping\n"
                        "4. **Contract Tests**: Add Pact consumer/provider tests for the Customer, "
                        "Inventory, and Shipping service integrations\n"
                        "5. **Performance Testing**: Run load tests with k6 or Gatling to validate "
                        "throughput under production traffic patterns\n"
                        "6. **Runbook**: Create operational runbook covering failover, scaling, "
                        "and incident response procedures\n"
                        "7. **Feature Flags**: Add feature flag support (LaunchDarkly or similar) "
                        "for gradual traffic migration from monolith"
                    ),
                }
            ],
        }
    )

    return turns
