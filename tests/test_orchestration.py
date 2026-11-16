"""
Comprehensive Test Suite for Multi-Agent Orchestration Engine.
==============================================================
Validates:
1. TransactionContext accumulation, hierarchical retrieval, and audit trail preservation.
2. Per-agent SLA budgeting and strict latency ceiling (<45ms).
3. Parallel stage concurrency and error isolation.
4. Sequential stage dependency chains and context enrichment.
5. CircuitBreaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).
6. Deterministic backup rules & low-confidence fallback triggers.
7. Conditional short-circuiting on fraud risk / card testing attacks.
8. DAG pipeline compilation from dependency graph.
9. End-to-end integration with Core DecisionEngine and domain agents.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

# Ensure root directory is appended to sys.path (after standard library modules)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import pytest

from payment_decision_factory.core.engine import DecisionEngine
from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.agents import (
    Dynamic3DSAgent,
    FraudScreenerAgent,
    SmartRetryAgent,
    SmartRoutingAgent,
)
from payment_decision_factory.orchestration.agent_base import (
    CallableAgent,
    EngineAgentAdapter,
    MockAgent,
    PaymentAgent,
    SpecialistSDMAgentAdapter,
)
from payment_decision_factory.orchestration.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from payment_decision_factory.orchestration.context import (
    AgentExecutionResult,
    ExecutionStatus,
    TransactionContext,
)
from payment_decision_factory.orchestration.pipeline import (
    DAGPipelineBuilder,
    PaymentOrchestrationPipeline,
    build_default_payment_pipeline,
    build_specialist_sdm_pipeline,
)
from payment_decision_factory.orchestration.stages import (
    ConditionalStage,
    ParallelStage,
    SequentialStage,
)

CORE_SPEC_PATH = os.path.join(
    os.path.dirname(__file__), "../specs/smart_routing.yaml"
)


# =============================================================================
# 1. TRANSACTION CONTEXT TESTS
# =============================================================================

class TestTransactionContext:
    """Tests for TransactionContext accumulator and audit trail."""

    def test_context_initialization_and_inputs(self):
        ctx = TransactionContext(
            inputs={"amount": 100.0, "currency": "USD", "card_brand": "visa"},
            metadata={"merchant_id": "m_123"},
        )
        assert ctx.transaction_id.startswith("tx_")
        assert ctx.get("amount") == 100.0
        assert ctx.get("currency") == "USD"
        assert ctx.get("merchant_id") == "m_123"
        assert ctx.get("unknown_key", "default_val") == "default_val"
        assert len(ctx.audit_trail) >= 1
        assert ctx.audit_trail[0].event_type == "PIPELINE_INIT"

    def test_record_agent_result_and_hierarchical_get(self):
        ctx = TransactionContext(inputs={"amount": 250.0})

        # Record fraud agent result
        ctx.record_agent_result(
            agent_name="fraud_screener",
            choice="review",
            score=0.42,
            confidence=0.88,
            probabilities={"allow": 0.3, "review": 0.58, "decline": 0.12},
            latency_ms=5.4,
            status=ExecutionStatus.SUCCESS,
        )

        # Direct dotted retrieval
        assert ctx.get("fraud_screener.choice") == "review"
        assert ctx.get("fraud_screener.score") == 0.42
        assert ctx.get("fraud_screener.confidence") == 0.88

        # Semantic alias retrieval
        assert ctx.get("fraud_score") == 0.42
        assert ctx.get("fraud_decision") == "review"

        # Direct input access injected by record_agent_result
        assert ctx.get("fraud_screener_choice") == "review"
        assert ctx.get("fraud_screener_score") == 0.42

    def test_audit_trail_chronology_and_serialization(self):
        ctx = TransactionContext(inputs={"amount": 50.0})
        ctx.record_audit("CUSTOM_EVENT", stage="risk", details={"info": "ok"})
        ctx.short_circuit(reason="velocity_breach", final_choice="decline")

        data = ctx.to_dict()
        assert data["is_short_circuited"] is True
        assert data["short_circuit_reason"] == "velocity_breach"
        assert len(data["audit_trail"]) >= 3

        json_str = ctx.to_json()
        assert "velocity_breach" in json_str

        summary = ctx.summary()
        assert summary["short_circuited"] is True
        assert summary["choice"] == "decline"


# =============================================================================
# 2. CIRCUIT BREAKER & DETERMINISTIC FALLBACK TESTS
# =============================================================================

class TestCircuitBreaker:
    """Tests for CircuitBreaker states, SLA enforcement, and fallback policies."""

    @pytest.mark.asyncio
    async def test_circuit_state_transition_and_fast_fail(self):
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_sec=0.2,
            half_open_successes=1,
            sla_timeout_ms=10.0,
        )
        cb = CircuitBreaker("mock_agent", config=config)
        assert cb.get_state() == CircuitState.CLOSED

        # First failure
        await cb.record_failure("error_1")
        assert cb.get_state() == CircuitState.CLOSED

        # Second failure -> triggers OPEN
        await cb.record_failure("error_2")
        assert cb.get_state() == CircuitState.OPEN

        # While OPEN: execute fallback executes fast in <1ms without calling agent
        ctx = TransactionContext(inputs={"bin_country": "US", "amount": 100.0})
        fb_res = await cb.execute_fallback(ctx, "circuit_open_test")
        assert fb_res.is_fallback is True
        assert fb_res.choice == "stripe_us"  # deterministic rule for US

        # Wait for recovery timeout -> transitions to HALF_OPEN
        await asyncio.sleep(0.25)
        assert cb.get_state() == CircuitState.HALF_OPEN

        # Successful probe call -> closes circuit
        await cb.record_success(latency_ms=2.0, confidence=0.9)
        assert cb.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_deterministic_backup_by_domain(self):
        cb_fraud = CircuitBreaker("fraud_screener")
        cb_3ds = CircuitBreaker("dynamic_3ds")
        cb_routing = CircuitBreaker("smart_routing")

        # Fraud fallback on high ticket
        ctx_high = TransactionContext(inputs={"amount": 900.0, "velocity_1h": 8})
        res_fraud = await cb_fraud.execute_fallback(ctx_high, "timeout")
        assert res_fraud.choice == "challenge"
        assert res_fraud.score == 0.75

        # 3DS fallback in EU
        ctx_eu = TransactionContext(inputs={"bin_country": "DE", "currency": "EUR", "amount": 100.0})
        res_3ds = await cb_3ds.execute_fallback(ctx_eu, "timeout")
        assert res_3ds.choice == "challenge"

        # 3DS low-value exemption under €30
        ctx_low = TransactionContext(inputs={"bin_country": "FR", "currency": "EUR", "amount": 15.0, "velocity_1h": 1})
        res_low_3ds = await cb_3ds.execute_fallback(ctx_low, "timeout")
        assert res_low_3ds.choice == "exemption_low_value"

        # Routing fallback in EU
        res_routing = await cb_routing.execute_fallback(ctx_eu, "timeout")
        assert res_routing.choice == "adyen_eu"


# =============================================================================
# 3. PARALLEL & SEQUENTIAL STAGE TESTS
# =============================================================================

class TestPipelineStages:
    """Tests for stage concurrency, error isolation, and sequence ordering."""

    @pytest.mark.asyncio
    async def test_parallel_stage_concurrency_and_timing(self):
        """Two 15ms agents running in parallel must complete in ~15-25ms, not 30ms+."""
        agent1 = MockAgent(name="agent_alpha", choice="alpha_done", sleep_ms=15.0, sla_ms=30.0)
        agent2 = MockAgent(name="agent_beta", choice="beta_done", sleep_ms=15.0, sla_ms=30.0)

        stage = ParallelStage(name="parallel_test", agents=[agent1, agent2], sla_ms=40.0)
        ctx = TransactionContext()
        registry = CircuitBreakerRegistry()

        start = time.perf_counter()
        stage_res = await stage.execute(ctx, remaining_budget_ms=50.0, registry=registry)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert stage_res.status == ExecutionStatus.SUCCESS
        assert ctx.get("agent_alpha.choice") == "alpha_done"
        assert ctx.get("agent_beta.choice") == "beta_done"
        # Concurrency verification: elapsed time should be well below 30ms sequential time
        assert elapsed_ms < 30.0

    @pytest.mark.asyncio
    async def test_parallel_stage_error_isolation(self):
        """Failure or timeout in one agent must NOT crash or cancel healthy parallel agents."""
        failing_agent = MockAgent(name="failing_agent", fail=True, sla_ms=60.0)
        healthy_agent = MockAgent(name="healthy_agent", choice="healthy_choice", sleep_ms=5.0, sla_ms=60.0)

        stage = ParallelStage(name="isolation_test", agents=[failing_agent, healthy_agent], sla_ms=80.0)
        ctx = TransactionContext(inputs={"bin_country": "US"})
        registry = CircuitBreakerRegistry()

        await stage.execute(ctx, remaining_budget_ms=100.0, registry=registry)

        # Healthy agent succeeded
        assert ctx.get("healthy_agent.choice") == "healthy_choice"
        assert ctx.get_agent_result("healthy_agent").status == ExecutionStatus.SUCCESS

        # Failing agent fell back gracefully
        failing_res = ctx.get_agent_result("failing_agent")
        assert failing_res.is_fallback is True
        assert failing_res.status == ExecutionStatus.ERROR

    @pytest.mark.asyncio
    async def test_sequential_stage_context_enrichment(self):
        """Sequential agent reads upstream context written by prior agent."""
        agent_upstream = MockAgent(name="fraud_screener", score=0.15, confidence=0.92, sleep_ms=2.0)

        def downstream_decide(context: TransactionContext) -> dict:
            upstream_score = context.get("fraud_score")
            if upstream_score is not None and upstream_score < 0.20:
                return {"choice": "express_routing", "confidence": 0.95}
            return {"choice": "standard_routing", "confidence": 0.80}

        agent_downstream = CallableAgent(name="routing", decide_fn=downstream_decide)

        seq_stage = SequentialStage(name="seq_chain", agents=[agent_upstream, agent_downstream], sla_ms=30.0)
        ctx = TransactionContext()
        registry = CircuitBreakerRegistry()

        await seq_stage.execute(ctx, remaining_budget_ms=40.0, registry=registry)
        assert ctx.get("fraud_screener.score") == 0.15
        assert ctx.get("routing.choice") == "express_routing"


# =============================================================================
# 4. LATENCY BUDGETING, TIMEOUTS & CIRCUIT BREAKING IN PIPELINE
# =============================================================================

class TestPipelineLatencyAndFallbacks:
    """Tests for strict SLA budgeting (<45ms) and circuit breaker activation."""

    @pytest.mark.asyncio
    async def test_agent_sla_timeout_triggers_fallback(self):
        """Agent that hangs (50ms) exceeds its 10ms SLA, triggering deterministic fallback."""
        slow_agent = MockAgent(name="smart_routing", choice="ideal_gateway", sleep_ms=50.0, sla_ms=10.0)

        pipeline = PaymentOrchestrationPipeline(global_timeout_ms=30.0)
        pipeline.add_sequential_stage("routing_stage", [slow_agent], sla_ms=15.0)

        ctx = TransactionContext(inputs={"bin_country": "US", "currency": "USD"})
        result_ctx = await pipeline.execute(ctx)

        agent_res = result_ctx.get_agent_result("smart_routing")
        assert agent_res.is_fallback is True
        assert agent_res.status == ExecutionStatus.TIMEOUT
        assert agent_res.choice == "stripe_us"  # deterministic US fallback
        assert result_ctx.total_latency_ms < 30.0

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_fallback(self):
        """Agent returning confidence below floor triggers fallback."""
        shaky_agent = MockAgent(
            name="smart_routing",
            choice="unstable_choice",
            confidence=0.25,  # floor is 0.40
            confidence_floor=0.40,
        )

        pipeline = PaymentOrchestrationPipeline(global_timeout_ms=30.0)
        pipeline.add_sequential_stage("routing_stage", [shaky_agent], sla_ms=20.0)

        ctx = TransactionContext(inputs={"bin_country": "DE", "currency": "EUR"})
        result_ctx = await pipeline.execute(ctx)

        agent_res = result_ctx.get_agent_result("smart_routing")
        assert agent_res.is_fallback is True
        assert agent_res.status == ExecutionStatus.LOW_CONFIDENCE_FALLBACK
        assert agent_res.choice == "adyen_eu"

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_subsequent_latency(self):
        """After threshold failures, circuit opens and subsequent calls return in <1ms."""
        flaky_agent = MockAgent(name="flaky_agent", fail=True, sla_ms=10.0)
        pipeline = PaymentOrchestrationPipeline(global_timeout_ms=50.0)
        pipeline.add_sequential_stage("flaky_stage", [flaky_agent], sla_ms=15.0)

        # Call 1 & 2 & 3: failures accumulate
        for _ in range(3):
            ctx = TransactionContext(inputs={"bin_country": "US"})
            await pipeline.execute(ctx)

        # Breaker should now be OPEN
        breaker = pipeline.registry.get_or_create("flaky_agent")
        assert breaker.get_state() == CircuitState.OPEN

        # Call 4: Fast-fails immediately without running agent
        start = time.perf_counter()
        ctx4 = TransactionContext(inputs={"bin_country": "US"})
        res_ctx4 = await pipeline.execute(ctx4)
        fast_elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert res_ctx4.get_agent_result("flaky_agent").status == ExecutionStatus.CIRCUIT_OPEN
        assert fast_elapsed_ms < 5.0  # Fast-fail under 5ms


# =============================================================================
# 5. SHORT-CIRCUIT & CONDITIONAL STAGES
# =============================================================================

class TestShortCircuitAndConditions:
    """Tests for early short-circuiting on fraud / attack detection."""

    @pytest.mark.asyncio
    async def test_fraud_decline_short_circuits_before_routing(self):
        """If fraud screener declines, pipeline immediately halts without calling routing."""
        fraud_agent = MockAgent(name="fraud_screener", choice="decline", score=0.89, sleep_ms=2.0)
        routing_agent = MockAgent(name="smart_routing", choice="stripe_us", sleep_ms=2.0)

        pipeline = build_default_payment_pipeline(
            fraud_agent=fraud_agent,
            three_ds_agent=MockAgent(name="dynamic_3ds", choice="challenge"),
            routing_agent=routing_agent,
            global_timeout_ms=45.0,
        )

        ctx = TransactionContext(inputs={"amount": 100.0, "currency": "USD"})
        result_ctx = await pipeline.execute(ctx)

        # Must be short circuited
        assert result_ctx.is_short_circuited is True
        assert result_ctx.final_decision["choice"] == "decline"
        # Smart routing must NOT have been executed
        assert "smart_routing" not in result_ctx.agent_results

    @pytest.mark.asyncio
    async def test_conditional_stage_runs_only_on_decline(self):
        """Smart retry runs only when transaction has decline code."""
        retry_agent = MockAgent(name="smart_retry", choice="retry_smart_window", score=24.0)

        stage = ConditionalStage(
            name="retry_check",
            condition=lambda ctx: ctx.get("decline_code") is not None,
            inner_stage=SequentialStage(name="retry_run", agents=[retry_agent]),
        )

        # Case A: Approved transaction -> retry skipped
        ctx_ok = TransactionContext(inputs={"decline_code": None})
        res_ok = await stage.execute(ctx_ok, remaining_budget_ms=30.0, registry=CircuitBreakerRegistry())
        assert res_ok.status == ExecutionStatus.SKIPPED
        assert "smart_retry" not in ctx_ok.agent_results

        # Case B: Declined transaction -> retry executes
        ctx_declined = TransactionContext(inputs={"decline_code": "insufficient_funds"})
        res_declined = await stage.execute(ctx_declined, remaining_budget_ms=30.0, registry=CircuitBreakerRegistry())
        assert res_declined.status == ExecutionStatus.SUCCESS
        assert ctx_declined.get("smart_retry.choice") == "retry_smart_window"


# =============================================================================
# 6. DAG PIPELINE BUILDER TESTS
# =============================================================================

class TestDAGPipelineBuilder:
    """Tests for DAG dependency graph compilation."""

    @pytest.mark.asyncio
    async def test_dag_compilation_and_execution(self):
        agent_fraud = MockAgent(name="fraud", score=0.10, sleep_ms=2.0)
        agent_3ds = MockAgent(name="three_ds", choice="frictionless", sleep_ms=2.0)
        agent_routing = MockAgent(name="routing", choice="adyen_eu", sleep_ms=2.0)

        # Build DAG: routing depends on fraud and 3ds; fraud and 3ds have no dependencies
        builder = (
            DAGPipelineBuilder(name="dag_test", global_timeout_ms=45.0, primary_decision_agent="routing")
            .add_node("fraud", agent_fraud, dependencies=[])
            .add_node("three_ds", agent_3ds, dependencies=[])
            .add_node("routing", agent_routing, dependencies=["fraud", "three_ds"])
        )
        compiled_pipeline = builder.compile()

        assert len(compiled_pipeline.stages) == 2
        assert isinstance(compiled_pipeline.stages[0], ParallelStage)
        assert isinstance(compiled_pipeline.stages[1], SequentialStage)

        ctx = TransactionContext(inputs={"amount": 40.0, "currency": "EUR"})
        res = await compiled_pipeline.execute(ctx)

        assert res.get("fraud.score") == 0.10
        assert res.get("three_ds.choice") == "frictionless"
        assert res.final_decision["choice"] == "adyen_eu"

    def test_dag_cycle_detection(self):
        builder = (
            DAGPipelineBuilder()
            .add_node("A", MockAgent("A"), dependencies=["B"])
            .add_node("B", MockAgent("B"), dependencies=["A"])
        )
        with pytest.raises(ValueError, match="Cycle detected"):
            builder.compile()


# =============================================================================
# 7. INTEGRATION WITH CORE DECISION ENGINE
# =============================================================================

class TestCoreEngineIntegration:
    """Tests integration between Core DecisionEngine and the Orchestrator."""

    @pytest.mark.asyncio
    async def test_engine_adapter_in_pipeline(self):
        spec = DecisionSpec.from_yaml(CORE_SPEC_PATH)
        engine = DecisionEngine(spec)
        routing_agent = EngineAgentAdapter(name="smart_routing", engine=engine, sla_ms=20.0)

        fraud_agent = MockAgent(name="fraud_screener", score=0.08, choice="allow", sleep_ms=1.0)
        three_ds_agent = MockAgent(name="dynamic_3ds", choice="tra_exemption", sleep_ms=1.0)

        pipeline = build_default_payment_pipeline(
            fraud_agent=fraud_agent,
            three_ds_agent=three_ds_agent,
            routing_agent=routing_agent,
            global_timeout_ms=45.0,
        )

        # Happy path US transaction
        us_tx = {
            "amount": 75.50,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
        }
        res_us = await pipeline.execute(us_tx)

        assert res_us.final_decision["choice"] == "stripe_us"
        assert res_us.final_decision["confidence"] > 0.50
        assert res_us.total_latency_ms < 45.0  # Well within 45ms SLA

        # Happy path EU transaction
        eu_tx = {
            "amount": 120.00,
            "currency": "EUR",
            "card_brand": "mastercard",
            "bin_country": "DE",
            "mcc": "5732",
            "velocity_1h": 2,
        }
        res_eu = await pipeline.execute(eu_tx)

        assert res_eu.final_decision["choice"] == "adyen_eu"
        assert res_eu.final_decision["confidence"] > 0.50
        assert res_eu.total_latency_ms < 45.0


# =============================================================================
# 8. FULL MULTI-AGENT DOMAIN PIPELINE INTEGRATION
# =============================================================================

class TestEndToEndMultiAgentDomainPipeline:
    """
    Tests complete real-world multi-agent pipeline executing all 4 concrete domain agents:
    FraudScreenerAgent, Dynamic3DSAgent, SmartRoutingAgent, and SmartRetryAgent.
    """

    @pytest.fixture
    def domain_pipeline(self) -> PaymentOrchestrationPipeline:
        fraud = FraudScreenerAgent()
        three_ds = Dynamic3DSAgent()
        routing = SmartRoutingAgent()
        retry = SmartRetryAgent()

        return build_default_payment_pipeline(
            fraud_agent=fraud,
            three_ds_agent=three_ds,
            routing_agent=routing,
            retry_agent=retry,
            global_timeout_ms=45.0,
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_domestic_us(self, domain_pipeline):
        tx = {
            "amount": 49.99,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
            "ip_country": "US",
            "cvv_result": "match",
            "avs_code": "match",
        }
        res_ctx = await domain_pipeline.execute(tx)

        # Stage 1: Fraud Screener + Dynamic 3DS in parallel
        assert "fraud_screener" in res_ctx.agent_results
        assert "dynamic_3ds" in res_ctx.agent_results
        assert res_ctx.get("fraud_screener.choice") == "allow"
        assert res_ctx.get("fraud_screener.score") < 0.25

        # Stage 2: Smart Routing receives enriched inputs
        assert "smart_routing" in res_ctx.agent_results
        assert res_ctx.final_decision["choice"] == "stripe_us"
        assert res_ctx.final_decision["confidence"] > 0.50
        assert res_ctx.is_short_circuited is False
        assert res_ctx.total_latency_ms < 45.0

        # Complete audit trail check
        event_types = [a.event_type for a in res_ctx.audit_trail]
        assert "PIPELINE_INIT" in event_types
        assert "PIPELINE_EXECUTION_START" in event_types
        assert "STAGE_START" in event_types
        assert "AGENT_EXECUTION_RECORDED" in event_types
        assert "STAGE_COMPLETED" in event_types
        assert "PIPELINE_FINALIZE" in event_types

    @pytest.mark.asyncio
    async def test_full_pipeline_european_psd2_low_value(self, domain_pipeline):
        tx = {
            "amount": 25.00,
            "currency": "EUR",
            "card_brand": "mastercard",
            "bin_country": "DE",
            "mcc": "5411",
            "velocity_1h": 1,
            "ip_country": "DE",
        }
        res_ctx = await domain_pipeline.execute(tx)

        assert res_ctx.get("dynamic_3ds.choice") == "low_value"
        assert res_ctx.final_decision["choice"] == "adyen_eu"
        assert res_ctx.total_latency_ms < 45.0

    @pytest.mark.asyncio
    async def test_full_pipeline_card_testing_attack_short_circuit(self, domain_pipeline):
        """Card testing velocity burst triggers instant fraud decline and short circuits."""
        attack_tx = {
            "amount": 10.00,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 18,  # >12 velocity threshold
            "ip_country": "US",
        }
        res_ctx = await domain_pipeline.execute(attack_tx)

        # Fraud Screener must decline
        assert res_ctx.get("fraud_screener.choice") == "decline"
        # Must be short circuited!
        assert res_ctx.is_short_circuited is True
        assert res_ctx.short_circuit_reason == "fraud_screener_pre_auth_decline"
        assert res_ctx.final_decision["choice"] == "decline"
        # Smart routing must NOT have been called, saving critical processor latency
        assert "smart_routing" not in res_ctx.agent_results
        assert res_ctx.total_latency_ms < 45.0

    @pytest.mark.asyncio
    async def test_full_pipeline_post_decline_smart_retry(self, domain_pipeline):
        """Declined transaction triggers post-decline retry optimization stage."""
        declined_tx = {
            "amount": 99.00,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
            "is_declined": True,
            "decline_code": "insufficient_funds",
            "decline_category": "soft_decline",
            "retry_attempt": 1,
        }
        res_ctx = await domain_pipeline.execute(declined_tx)

        # Smart Retry should execute in Stage 3
        assert "smart_retry" in res_ctx.agent_results
        retry_res = res_ctx.get_agent_result("smart_retry")
        assert retry_res.choice in ("retry_smart_window", "retry_payday_window")
        assert retry_res.score > 0  # recommended backoff hours


# =============================================================================
# 9. SPECIALIST SDM SUITE ORCHESTRATION TESTS
# =============================================================================

class TestSpecialistSDMOrchestration:
    """
    Tests native integration of compact Specialist Small Decision Models (SDMs)
    into the async DAG orchestration pipeline.
    """

    @pytest.fixture
    def routing_sdm_adapter(self) -> SpecialistSDMAgentAdapter:
        from payment_decision_factory.core.sdm import SpecialistPipeline, SpecialistSDM
        spec = DecisionSpec.from_yaml(CORE_SPEC_PATH)
        pipe = SpecialistPipeline(spec, specialist_type="routing")
        sdm = SpecialistSDM(
            specialist_type="routing",
            num_continuous_features=30,
            num_classes=len(spec.choices),
            hidden_dim=32,
            num_blocks=1,
        )
        return SpecialistSDMAgentAdapter(
            name="smart_routing_specialist",
            sdm_model=sdm,
            sdm_pipeline=pipe,
            sla_ms=15.0,
        )

    @pytest.mark.asyncio
    async def test_specialist_sdm_adapter_decide(self, routing_sdm_adapter):
        ctx = TransactionContext(
            inputs={
                "amount": 95.0,
                "currency": "USD",
                "card_brand": "visa",
                "bin_country": "US",
                "mcc": "5411",
                "velocity_1h": 1,
            }
        )
        res = await routing_sdm_adapter.decide(ctx)
        assert res.agent_name == "smart_routing_specialist"
        assert res.choice is not None
        assert res.confidence > 0.0
        assert len(res.probabilities) > 0
        assert res.latency_ms < 15.0

    @pytest.mark.asyncio
    async def test_build_specialist_sdm_pipeline(self, routing_sdm_adapter):
        fraud = FraudScreenerAgent(name="fraud_specialist")
        three_ds = Dynamic3DSAgent(name="dynamic_3ds_specialist")
        retry = SmartRetryAgent(name="smart_retry_specialist")

        pipeline = build_specialist_sdm_pipeline(
            routing_specialist=routing_sdm_adapter,
            fraud_specialist=fraud,
            three_ds_specialist=three_ds,
            retry_specialist=retry,
            global_timeout_ms=45.0,
        )

        tx = {
            "amount": 49.99,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
            "ip_country": "US",
            "cvv_result": "match",
            "avs_code": "match",
        }
        res_ctx = await pipeline.execute(tx)

        # Stage 1: Parallel Fraud + 3DS
        assert "fraud_specialist" in res_ctx.agent_results
        assert "dynamic_3ds_specialist" in res_ctx.agent_results

        # Stage 2: Sequential Routing Specialist SDM
        assert "smart_routing_specialist" in res_ctx.agent_results
        assert res_ctx.final_decision["choice"] is not None
        assert res_ctx.is_short_circuited is False
        assert res_ctx.total_latency_ms < 45.0

        # Audit trail checks
        stage_names = [s.stage_name for s in res_ctx.stage_results]
        assert "parallel_risk_and_compliance_specialists" in stage_names
        assert "sequential_smart_routing_specialist" in stage_names

    @pytest.mark.asyncio
    async def test_specialist_sdm_pipeline_short_circuit(self, routing_sdm_adapter):
        fraud = FraudScreenerAgent(name="fraud_specialist")
        three_ds = Dynamic3DSAgent(name="dynamic_3ds_specialist")

        pipeline = build_specialist_sdm_pipeline(
            routing_specialist=routing_sdm_adapter,
            fraud_specialist=fraud,
            three_ds_specialist=three_ds,
            global_timeout_ms=45.0,
        )

        attack_tx = {
            "amount": 10.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 20,  # Extreme velocity trigger
        }
        res_ctx = await pipeline.execute(attack_tx)

        assert res_ctx.is_short_circuited is True
        assert res_ctx.final_decision["choice"] == "decline"
        # Smart routing must NOT have been called
        assert "smart_routing_specialist" not in res_ctx.agent_results
        assert res_ctx.total_latency_ms < 45.0


