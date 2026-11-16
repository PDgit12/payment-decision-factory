# Multi-Agent Payment Decision Orchestration Engine

The `payment_decision_factory.orchestration` package provides a high-performance, asynchronous DAG/pipeline orchestrator designed for mission-critical payment decisioning under sub-45ms latency budgets.

---

## Architectural Highlights

1. **Sequential & Parallel Stage Execution**:
   - Run computationally independent models in parallel (e.g., `FraudScreenerAgent` and `Dynamic3DSAgent` running concurrently via `asyncio.gather` within a 15ms stage budget).
   - Feed enriched intermediate outputs into downstream sequential stages (e.g., `SmartRoutingAgent` consuming upstream `fraud_score` and `three_ds_choice`).
   - Conditional stages (e.g., `SmartRetryAgent` executing only when a transaction suffers a soft decline).

2. **Strict Latency Budgeting**:
   - **Global Latency Deadline**: Configurable end-to-end deadline (default: 45.0ms) enforced using monotonic clocks (`time.perf_counter()`).
   - **Per-Agent SLA Enforcement**: Dynamic remaining budget allocation (`effective_timeout = min(agent.sla_ms, remaining_budget_ms)`).
   - **Emergency Global Fallback**: If the global budget is exhausted, execution terminates immediately with deterministic fallback routing without holding checkout threads.

3. **Resilience & Circuit Breakers**:
   - **State Machine**: `CLOSED` -> `OPEN` -> `HALF_OPEN` -> `CLOSED`.
   - **Fast-Fail Semantics**: When a circuit opens due to consecutive failures or timeouts, subsequent calls bypass the agent in `<0.1ms` directly to domain-specific deterministic backup rules.
   - **Low-Confidence Fallback**: Automatically invokes deterministic backup if an agent's confidence score drops below its configured `confidence_floor` (e.g., < 0.40).

4. **Short-Circuit Termination**:
   - Early detection of high fraud risk (e.g., score >= 0.70 or automated card-testing velocity burst) immediately terminates the pipeline and marks the transaction declined, preventing unnecessary downstream processor dispatch and fees.

5. **Transaction Context Accumulator (`TransactionContext`)**:
   - Preserves raw input features, intermediate agent decisions, confidence distributions, and latency metrics throughout the payment lifecycle.
   - Thread-safe, hierarchical accessor: `context.get("amount")`, `context.get("fraud_screener.score")`, `context.get("fraud_score")`.
   - Complete, immutable chronological audit trail (`AuditEvent`) logging every initialization, stage start, agent invocation, fallback, and completion event.

---

## Quickstart & Usage

### 1. Default Production Payment Pipeline

```python
from payment_decision_factory.agents import (
    FraudScreenerAgent,
    Dynamic3DSAgent,
    SmartRoutingAgent,
    SmartRetryAgent,
)
from payment_decision_factory.orchestration import build_default_payment_pipeline

# 1. Instantiate domain agents
fraud = FraudScreenerAgent()
three_ds = Dynamic3DSAgent()
routing = SmartRoutingAgent()
retry = SmartRetryAgent()

# 2. Build production pipeline
pipeline = build_default_payment_pipeline(
    fraud_agent=fraud,
    three_ds_agent=three_ds,
    routing_agent=routing,
    retry_agent=retry,
    global_timeout_ms=45.0,
)

# 3. Execute transaction
transaction_payload = {
    "amount": 49.99,
    "currency": "USD",
    "card_brand": "visa",
    "bin_country": "US",
    "mcc": "5411",
    "velocity_1h": 1,
}

context = await pipeline.execute(transaction_payload)

print("Final Routing Choice:", context.final_decision["choice"]) # e.g. 'stripe_us'
print("Total Latency (ms):", context.total_latency_ms)           # e.g. 1.8ms
print("Short Circuited:", context.is_short_circuited)            # False
print("Audit Events:", len(context.audit_trail))
```

### 2. Custom DAG Compilation

```python
from payment_decision_factory.orchestration import DAGPipelineBuilder, MockAgent

builder = (
    DAGPipelineBuilder(name="custom_dag", global_timeout_ms=45.0, primary_decision_agent="routing")
    .add_node("fraud", fraud_agent, dependencies=[])
    .add_node("compliance", three_ds_agent, dependencies=[])
    .add_node("routing", routing_agent, dependencies=["fraud", "compliance"])
)

pipeline = builder.compile()
context = await pipeline.execute(payload)
```

### 3. Closed-Loop Lifecycle Orchestrator & Canary Deployment

The `ClosedLoopOrchestrator` automates the entire payment intelligence lifecycle:

```python
from payment_decision_factory.orchestration import (
    ClosedLoopOrchestrator,
    RolloutMode,
)

orchestrator = ClosedLoopOrchestrator(client_id="enterprise_merchant", output_dir="./models")

# Execute end-to-end autonomous lifecycle
result = orchestrator.execute_lifecycle(
    records=historical_payment_data,
    spec=routing_spec,
    license_key="lic_ent_secret_key_999",
    deploy_pipeline=production_pipeline,
    deploy_target_agent="routing",
    rollout_mode=RolloutMode.CANARY,
    canary_percentage=15.0,
)

print(f"Ingested: {result.ingestion.total_records} txs")
print(f"Simulation Auth Rate: {result.simulation.simulated_approval_rate:.2%}")
print(f"Self-Adjustment Uplift: +{result.self_adjustment.approval_rate_uplift_pct:.2f}%")
print(f"Sealed Package: {result.sealed_package.package_path} (AES-256-GCM)")
print(f"Deployment Status: {result.deployment_status}")
```

#### Lifecycle Stages:
1. **Data Ingestion (`ingest_records`)**: Automated schema profiling, type inference, missing value imputation, and validation.
2. **Specialist SDM Training (`train_specialist_sdm`)**: In-process training of compact gated residual networks with temperature calibration (<1s CPU execution).
3. **Digital Twin Simulation (`run_digital_twin_simulation`)**: Counterfactual replay against historical baselines measuring approval rate, Brier calibration, and fee economics.
4. **Policy Self-Adjustment (`optimize_and_self_adjust`)**: Autonomous tuning of probability thresholds, temperature parameters, and NOUL boundary relaxations based on simulator feedback.
5. **Sealed Packaging (`seal_package`)**: Compiles model into encrypted `.sdmpkg` using AES-256-GCM, PBKDF2-HMAC-SHA256 (100k iterations), and tamper-evident AAD manifest binding.
6. **Canary / Shadow Rollout (`CanaryDeploymentAgent`)**: Safe live deployment supporting 0-downtime shadow mirroring, canary percentage splits (e.g. 10%), and instant automated rollback if candidate error rate exceeds 5%.

---

## Verification & Tests

Run the test suite:
```bash
# Orchestration DAG, Circuit Breakers, Context & Latency tests (23 tests)
pytest payment_decision_factory/tests/test_orchestration.py -v

# Closed-Loop Lifecycle & Canary Deployment tests (7 tests)
pytest payment_decision_factory/tests/test_closed_loop_orchestrator.py -v
```

All 30 orchestration and closed-loop tests pass with 100% coverage of concurrency, timeouts, circuit breaking, fallback rules, closed-loop execution, and domain agent integration.
