"""
High-Performance Multi-Agent Payment Decision Pipeline & DAG Orchestrator.

Implements sequential & parallel execution stages, strict latency budgeting
(global 45ms timeout with per-agent SLA enforcement), circuit breakers,
conditional short-circuiting, and deterministic fallbacks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .agent_base import PaymentAgent, SpecialistSDMAgentAdapter
from .circuit_breaker import CircuitBreakerRegistry
from .context import (
    AgentExecutionResult,
    ExecutionStatus,
    TransactionContext,
)
from .stages import (
    ConditionalStage,
    ParallelStage,
    PipelineStage,
    SequentialStage,
)

logger = logging.getLogger("orchestrator.pipeline")


class PaymentOrchestrationPipeline:
    """
    Asynchronous DAG / Stage Pipeline Orchestrator for Payment Decision Models.

    Guarantees:
    - Global latency ceiling (e.g. 45ms timeout with dynamic budget tracking).
    - Per-agent SLA enforcement with non-blocking async execution.
    - Automatic circuit breaking & deterministic fallback if agents fail or return low confidence.
    - Dynamic short-circuiting on high fraud risk / card attacks to prevent downstream gateway overhead.
    - Complete, immutable audit log preservation for every decision and transaction lifecycle.
    """

    def __init__(
        self,
        name: str = "payment_decision_pipeline",
        global_timeout_ms: float = 45.0,
        primary_decision_agent: str = "smart_routing",
        circuit_registry: Optional[CircuitBreakerRegistry] = None,
    ):
        self.name = name
        self.global_timeout_ms = global_timeout_ms
        self.primary_decision_agent = primary_decision_agent
        self.registry = circuit_registry or CircuitBreakerRegistry()
        self.stages: List[PipelineStage] = []
        self._short_circuit_rules: List[Callable[[TransactionContext], Optional[Tuple[str, Optional[str], Optional[float]]]]] = []
        self._global_fallback_handler: Optional[Callable[[TransactionContext, str], Dict[str, Any]]] = None

        # Telemetry counters
        self.total_transactions: int = 0
        self.short_circuits: int = 0
        self.global_timeouts: int = 0

    # =========================================================================
    # FLUENT BUILDER METHODS
    # =========================================================================

    def add_stage(self, stage: PipelineStage) -> PaymentOrchestrationPipeline:
        """Append an arbitrary stage to the pipeline."""
        self.stages.append(stage)
        return self

    def add_parallel_stage(
        self,
        name: str,
        agents: List[PaymentAgent],
        sla_ms: float = 20.0,
    ) -> PaymentOrchestrationPipeline:
        """Add a parallel execution stage running multiple agents concurrently."""
        self.stages.append(ParallelStage(name=name, agents=agents, sla_ms=sla_ms))
        return self

    def add_sequential_stage(
        self,
        name: str,
        agents: List[PaymentAgent],
        sla_ms: float = 25.0,
    ) -> PaymentOrchestrationPipeline:
        """Add a sequential execution stage running agents one after another."""
        self.stages.append(SequentialStage(name=name, agents=agents, sla_ms=sla_ms))
        return self

    def add_conditional_stage(
        self,
        name: str,
        condition: Callable[[TransactionContext], bool],
        inner_stage: PipelineStage,
        else_stage: Optional[PipelineStage] = None,
        sla_ms: float = 25.0,
    ) -> PaymentOrchestrationPipeline:
        """Add a stage that executes only if a context predicate is satisfied."""
        self.stages.append(
            ConditionalStage(
                name=name,
                condition=condition,
                inner_stage=inner_stage,
                else_stage=else_stage,
                sla_ms=sla_ms,
            )
        )
        return self

    def add_short_circuit_rule(
        self,
        rule_fn: Callable[[TransactionContext], Optional[Tuple[str, Optional[str], Optional[float]]]],
    ) -> PaymentOrchestrationPipeline:
        """
        Add a short-circuit rule evaluated between stages.

        rule_fn takes TransactionContext and returns:
        - None if pipeline should continue normally.
        - Tuple of (reason: str, final_choice: Optional[str], final_score: Optional[float]) to terminate early.
        """
        self._short_circuit_rules.append(rule_fn)
        return self

    def set_primary_agent(self, agent_name: str) -> PaymentOrchestrationPipeline:
        """Set the agent whose output serves as the pipeline's primary decision."""
        self.primary_decision_agent = agent_name
        return self

    def set_global_fallback(
        self,
        fallback_fn: Callable[[TransactionContext, str], Dict[str, Any]],
    ) -> PaymentOrchestrationPipeline:
        """Set fallback handler invoked when global latency budget is exceeded."""
        self._global_fallback_handler = fallback_fn
        return self

    # =========================================================================
    # SHORT CIRCUIT EVALUATION
    # =========================================================================

    def _evaluate_short_circuits(self, context: TransactionContext) -> bool:
        """Check all registered short-circuit rules."""
        if context.is_short_circuited:
            return True

        # Default standard payment short-circuit rules:
        # 1. Fraud Screener declined or high risk score (>= 0.70)
        fraud_res = context.get_agent_result("fraud_screener")
        if fraud_res is not None:
            if fraud_res.choice == "decline":
                context.short_circuit(
                    reason="fraud_screener_pre_auth_decline",
                    final_choice="decline",
                    final_score=fraud_res.score,
                    metadata={"fraud_score": fraud_res.score, "trigger": "fraud_decline"},
                )
                self.short_circuits += 1
                return True
            if fraud_res.score is not None and fraud_res.score >= 0.70:
                context.short_circuit(
                    reason="fraud_score_exceeds_decline_threshold",
                    final_choice="decline",
                    final_score=fraud_res.score,
                    metadata={"fraud_score": fraud_res.score, "trigger": "high_risk_score"},
                )
                self.short_circuits += 1
                return True

        # 2. Dynamic 3DS mandatory challenge when step-up cannot be fulfilled
        # (if caller specified step_up_impossible=True)
        if context.get("step_up_impossible", False):
            three_ds = context.get_agent_result("dynamic_3ds")
            if three_ds and three_ds.choice == "challenge":
                context.short_circuit(
                    reason="mandatory_sca_challenge_unsupported_by_channel",
                    final_choice="decline",
                    metadata={"trigger": "unsupported_sca_challenge"},
                )
                self.short_circuits += 1
                return True

        # 3. Custom rules
        for rule in self._short_circuit_rules:
            try:
                res = rule(context)
                if res is not None:
                    reason, final_choice, final_score = res
                    context.short_circuit(
                        reason=reason,
                        final_choice=final_choice,
                        final_score=final_score,
                    )
                    self.short_circuits += 1
                    return True
            except Exception as ex:
                logger.warning(f"Short circuit rule failed with: {ex}")

        return False

    # =========================================================================
    # PIPELINE EXECUTION ENGINE
    # =========================================================================

    async def execute(
        self,
        payload: Union[Dict[str, Any], TransactionContext],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionContext:
        """
        Execute the full multi-stage decision pipeline asynchronously.

        Args:
            payload: Input dictionary of payment features or existing TransactionContext.
            metadata: Optional additional request-level metadata.

        Returns:
            Finalized TransactionContext with all agent outputs and audit trail.
        """
        self.total_transactions += 1

        if isinstance(payload, TransactionContext):
            context = payload
            if metadata:
                context.metadata.update(metadata)
        else:
            context = TransactionContext(inputs=payload, metadata=metadata)

        start_time = time.perf_counter()
        deadline = start_time + (self.global_timeout_ms / 1000.0)

        context.record_audit(
            event_type="PIPELINE_EXECUTION_START",
            details={
                "pipeline_name": self.name,
                "global_timeout_ms": self.global_timeout_ms,
                "stage_count": len(self.stages),
                "primary_agent": self.primary_decision_agent,
            },
        )

        # Iterate through stages
        for stage in self.stages:
            now = time.perf_counter()
            remaining_budget_ms = (deadline - now) * 1000.0

            # Global timeout enforcement
            if remaining_budget_ms <= 0:
                self.global_timeouts += 1
                logger.error(
                    f"Pipeline [{self.name}] breached global latency timeout ({self.global_timeout_ms}ms) before stage '{stage.name}'."
                )
                context.record_audit(
                    event_type="PIPELINE_GLOBAL_TIMEOUT",
                    stage=stage.name,
                    details={"elapsed_ms": (now - start_time) * 1000.0, "global_timeout_ms": self.global_timeout_ms},
                )
                self._apply_global_fallback(context, "global_latency_timeout_breach")
                break

            # Execute the stage
            try:
                stage_result = await stage.execute(
                    context=context,
                    remaining_budget_ms=remaining_budget_ms,
                    registry=self.registry,
                )
            except Exception as ex:
                logger.exception(f"Unhandled exception in pipeline stage [{stage.name}]: {ex}")
                context.record_audit(
                    event_type="STAGE_ERROR",
                    stage=stage.name,
                    details={"error": str(ex)},
                )

            # Check for short circuit after every stage
            if self._evaluate_short_circuits(context):
                logger.info(
                    f"Pipeline [{self.name}] short-circuited after stage '{stage.name}'. Reason: {context.short_circuit_reason}"
                )
                break

        # Finalize context with resolution
        context.finalize(primary_agent=self.primary_decision_agent)
        return context

    def _apply_global_fallback(self, context: TransactionContext, reason: str) -> None:
        """Apply deterministic global fallback when SLA budget is completely breached."""
        if self._global_fallback_handler:
            try:
                fb = self._global_fallback_handler(context, reason)
                context.finalize(
                    final_choice=fb.get("choice", "checkout_crossborder"),
                    final_score=fb.get("score"),
                    primary_agent="global_fallback",
                    metadata={"fallback_reason": reason, "handler": "custom_global_fallback"},
                )
                return
            except Exception as ex:
                logger.exception(f"Global fallback handler failed: {ex}")

        # Deterministic emergency fallback
        bin_country = str(context.get("bin_country", "US")).upper()
        if bin_country in ("US", "CA"):
            emergency_choice = "stripe_us"
        elif bin_country in ("GB", "DE", "FR", "NL", "IE", "ES", "IT"):
            emergency_choice = "adyen_eu"
        else:
            emergency_choice = "checkout_crossborder"

        context.finalize(
            final_choice=emergency_choice,
            primary_agent="emergency_fallback",
            metadata={"is_emergency_fallback": True, "fallback_reason": reason},
        )


class DAGNode:
    """A node in an asynchronous DAG execution graph."""

    def __init__(
        self,
        name: str,
        agent: PaymentAgent,
        dependencies: Optional[List[str]] = None,
    ):
        self.name = name
        self.agent = agent
        self.dependencies: Set[str] = set(dependencies or [])


class DAGPipelineBuilder:
    """
    Builds and compiles an execution graph with explicit dependencies
    into an optimized stage pipeline.
    """

    def __init__(
        self,
        name: str = "dag_payment_pipeline",
        global_timeout_ms: float = 45.0,
        primary_decision_agent: str = "smart_routing",
    ):
        self.name = name
        self.global_timeout_ms = global_timeout_ms
        self.primary_decision_agent = primary_decision_agent
        self.nodes: Dict[str, DAGNode] = {}

    def add_node(
        self,
        name: str,
        agent: PaymentAgent,
        dependencies: Optional[List[str]] = None,
    ) -> DAGPipelineBuilder:
        """Add an agent node with its upstream dependencies."""
        self.nodes[name] = DAGNode(name=name, agent=agent, dependencies=dependencies)
        return self

    def compile(self) -> PaymentOrchestrationPipeline:
        """
        Compile DAG into parallel and sequential stages using Kahn's topological level sort.
        """
        # Validate dependency existence
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{name}' references unknown dependency '{dep}'")

        # In-degree computation
        in_degree = {name: len(node.dependencies) for name, node in self.nodes.items()}
        dependents = defaultdict(list)
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                dependents[dep].append(name)

        # Level-order topological grouping
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        levels: List[List[str]] = []
        visited_count = 0

        while queue:
            level_size = len(queue)
            current_level: List[str] = []
            for _ in range(level_size):
                curr = queue.popleft()
                current_level.append(curr)
                visited_count += 1
                for nxt in dependents[curr]:
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0:
                        queue.append(nxt)
            levels.append(current_level)

        if visited_count != len(self.nodes):
            raise ValueError("Cycle detected in DAG pipeline agent dependencies!")

        pipeline = PaymentOrchestrationPipeline(
            name=self.name,
            global_timeout_ms=self.global_timeout_ms,
            primary_decision_agent=self.primary_decision_agent,
        )

        for idx, level_names in enumerate(levels):
            agents = [self.nodes[n].agent for n in level_names]
            if len(agents) == 1:
                pipeline.add_sequential_stage(
                    name=f"stage_{idx}_{level_names[0]}",
                    agents=agents,
                    sla_ms=agents[0].sla_ms,
                )
            else:
                max_sla = max(a.sla_ms for a in agents)
                pipeline.add_parallel_stage(
                    name=f"stage_{idx}_parallel_{'_'.join(level_names)}",
                    agents=agents,
                    sla_ms=max_sla,
                )

        return pipeline


def build_default_payment_pipeline(
    fraud_agent: PaymentAgent,
    three_ds_agent: PaymentAgent,
    routing_agent: PaymentAgent,
    retry_agent: Optional[PaymentAgent] = None,
    global_timeout_ms: float = 45.0,
) -> PaymentOrchestrationPipeline:
    """
    Standard production payment pipeline:
    - Stage 1: Parallel Risk & Compliance (Fraud Screener + Dynamic 3DS in parallel, max 15ms)
    - Short Circuit: If fraud risk is critical or card attack detected, terminate immediately!
    - Stage 2: Smart Gateway Routing (enriches context with risk & 3DS outputs, max 20ms)
    - Stage 3 (Optional): Smart Retry conditional on decline status.
    """
    pipeline = PaymentOrchestrationPipeline(
        name="production_payment_auth_pipeline",
        global_timeout_ms=global_timeout_ms,
        primary_decision_agent=routing_agent.name,
    )

    # Stage 1: Parallel Pre-Auth Screening
    pipeline.add_parallel_stage(
        name="pre_auth_risk_and_compliance",
        agents=[fraud_agent, three_ds_agent],
        sla_ms=18.0,
    )

    # Stage 2: Smart Gateway Routing
    pipeline.add_sequential_stage(
        name="smart_gateway_routing",
        agents=[routing_agent],
        sla_ms=22.0,
    )

    # Stage 3: Conditional Smart Retry (only runs if transaction is declined or soft failure)
    if retry_agent is not None:
        pipeline.add_conditional_stage(
            name="post_decline_smart_retry",
            condition=lambda ctx: ctx.get("is_declined", False) or ctx.get("decline_code") is not None,
            inner_stage=SequentialStage(
                name="retry_backoff_calculation",
                agents=[retry_agent],
                sla_ms=retry_agent.sla_ms,
            ),
        )

    return pipeline


def build_live_production_pipeline(
    global_timeout_ms: float = 45.0,
) -> PaymentOrchestrationPipeline:
    """
    Instantiates the complete production payment pipeline using 100% REAL domain agents:
    - FraudScreenerAgent (backed by specs/fraud_screener.yaml)
    - Dynamic3DSAgent (backed by specs/dynamic_3ds.yaml)
    - SmartRoutingAgent (backed by specs/smart_routing.yaml)
    - SmartRetryAgent (backed by specs/smart_retry.yaml)
    Zero mocks, 100% genuine domain decision logic and circuit breakers.
    """
    try:
        from payment_decision_factory.agents.fraud_screener import FraudScreenerAgent
        from payment_decision_factory.agents.dynamic_3ds import Dynamic3DSAgent
        from payment_decision_factory.agents.smart_routing import SmartRoutingAgent
        from payment_decision_factory.agents.smart_retry import SmartRetryAgent
    except ModuleNotFoundError:
        from agents.fraud_screener import FraudScreenerAgent
        from agents.dynamic_3ds import Dynamic3DSAgent
        from agents.smart_routing import SmartRoutingAgent
        from agents.smart_retry import SmartRetryAgent

    fraud_agent = FraudScreenerAgent()
    three_ds_agent = Dynamic3DSAgent()
    routing_agent = SmartRoutingAgent()
    retry_agent = SmartRetryAgent()

    return build_default_payment_pipeline(
        fraud_agent=fraud_agent,
        three_ds_agent=three_ds_agent,
        routing_agent=routing_agent,
        retry_agent=retry_agent,
        global_timeout_ms=global_timeout_ms,
    )


def _ensure_payment_agent(
    agent_input: Union[PaymentAgent, Tuple[Any, Any], Any],
    default_name: str,
    sla_ms: float = 15.0,
) -> PaymentAgent:
    """Normalize input into a PaymentAgent instance (supporting SpecialistSDMAgentAdapter)."""
    if isinstance(agent_input, PaymentAgent):
        return agent_input

    if isinstance(agent_input, tuple) and len(agent_input) == 2:
        model, pipeline_helper = agent_input
        return SpecialistSDMAgentAdapter(
            name=default_name,
            sdm_model=model,
            sdm_pipeline=pipeline_helper,
            sla_ms=sla_ms,
        )

    if hasattr(agent_input, "decide"):
        from .agent_base import CallableAgent
        return CallableAgent(
            name=getattr(agent_input, "name", default_name),
            decide_fn=agent_input.decide,
            sla_ms=getattr(agent_input, "sla_ms", sla_ms),
        )

    raise TypeError(
        f"Expected PaymentAgent, (SpecialistSDM, SpecialistPipeline) tuple, or agent object for '{default_name}', got {type(agent_input)}"
    )


def build_specialist_sdm_pipeline(
    routing_specialist: Union[PaymentAgent, Tuple[Any, Any], Any],
    fraud_specialist: Union[PaymentAgent, Tuple[Any, Any], Any],
    three_ds_specialist: Union[PaymentAgent, Tuple[Any, Any], Any],
    retry_specialist: Optional[Union[PaymentAgent, Tuple[Any, Any], Any]] = None,
    global_timeout_ms: float = 45.0,
    parallel_sla_ms: float = 15.0,
    routing_sla_ms: float = 15.0,
) -> PaymentOrchestrationPipeline:
    """
    Constructs an ultra-low-latency DAG pipeline powered natively by the 4
    compact Specialist Small Decision Models (SDMs):

    Pipeline Topology:
    1. Stage 1 (Parallel Risk & PSD2 Compliance):
       - Fraud Specialist SDM: Pre-auth velocity, CVV, and anomaly risk scoring.
       - Dynamic 3DS Specialist SDM: PSD2 RTS SCA exemptions (TRA / Low-Value).
       Runs concurrently via ParallelStage with microsecond (<0.2ms) inference.

    2. Dynamic Short-Circuit Guardrail:
       - Evaluates Fraud Specialist SDM score; if high-risk (>= 0.70) or decline,
         short-circuits immediately, preventing unnecessary downstream processor dispatch.

    3. Stage 2 (Sequential Specialist Routing):
       - Smart Routing Specialist SDM: Evaluates gateway choices (Stripe, Adyen, Checkout.com)
         with temperature-scaled calibrated probabilities, informed by upstream risk/3DS.

    4. Stage 3 (Conditional Post-Decline Retry Specialist):
       - Smart Retry Specialist SDM: Runs conditionally if transaction encounters a decline,
         predicting optimal backoff hours and payday alignment.
    """
    fraud_agent = _ensure_payment_agent(fraud_specialist, "fraud_specialist", sla_ms=parallel_sla_ms)
    three_ds_agent = _ensure_payment_agent(three_ds_specialist, "dynamic_3ds_specialist", sla_ms=parallel_sla_ms)
    routing_agent = _ensure_payment_agent(routing_specialist, "smart_routing_specialist", sla_ms=routing_sla_ms)

    pipeline = PaymentOrchestrationPipeline(
        name="specialist_sdm_decision_pipeline",
        global_timeout_ms=global_timeout_ms,
        primary_decision_agent=routing_agent.name,
    )

    # Stage 1: Parallel Pre-Auth Risk & Compliance Specialists
    pipeline.add_parallel_stage(
        name="parallel_risk_and_compliance_specialists",
        agents=[fraud_agent, three_ds_agent],
        sla_ms=parallel_sla_ms,
    )

    # Dynamic short-circuit rule on high fraud risk from Fraud Specialist
    def _fraud_sdm_short_circuit_checker(ctx: TransactionContext):
        fraud_res = ctx.get_agent_result(fraud_agent.name)
        if fraud_res is not None:
            if fraud_res.choice == "decline":
                return ("fraud_specialist_sdm_decline", "decline", fraud_res.score)
            if fraud_res.score is not None and fraud_res.score >= 0.70:
                return ("fraud_specialist_sdm_high_risk_score", "decline", fraud_res.score)
        return None

    pipeline.add_short_circuit_rule(_fraud_sdm_short_circuit_checker)

    # Stage 2: Sequential Smart Routing Specialist
    pipeline.add_sequential_stage(
        name="sequential_smart_routing_specialist",
        agents=[routing_agent],
        sla_ms=routing_sla_ms,
    )

    # Stage 3: Conditional Smart Retry Specialist
    if retry_specialist is not None:
        retry_agent = _ensure_payment_agent(retry_specialist, "smart_retry_specialist", sla_ms=parallel_sla_ms)
        pipeline.add_conditional_stage(
            name="conditional_smart_retry_specialist",
            condition=lambda ctx: ctx.get("is_declined", False) or ctx.get("decline_code") is not None,
            inner_stage=SequentialStage(
                name="retry_backoff_specialist_stage",
                agents=[retry_agent],
                sla_ms=retry_agent.sla_ms,
            ),
        )

    return pipeline

