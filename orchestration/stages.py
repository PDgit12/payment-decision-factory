"""
Execution Stages for Multi-Agent Payment Decision Pipelines.

Provides Parallel, Sequential, and Conditional execution stages
with per-agent SLA enforcement, circuit-breaker wrapping, and deterministic fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .agent_base import PaymentAgent
from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState
from .context import (
    AgentExecutionResult,
    ExecutionStatus,
    StageExecutionResult,
    TransactionContext,
)

logger = logging.getLogger("orchestrator.stages")


class PipelineStage(ABC):
    """Abstract base class for a pipeline stage."""

    name: str
    stage_type: str
    sla_ms: float

    @abstractmethod
    async def execute(
        self,
        context: TransactionContext,
        remaining_budget_ms: float,
        registry: CircuitBreakerRegistry,
    ) -> StageExecutionResult:
        """
        Execute this stage with given context and remaining latency budget.

        Args:
            context: Transaction context accumulator.
            remaining_budget_ms: Milliseconds remaining in global latency budget.
            registry: Circuit breaker registry.

        Returns:
            StageExecutionResult with execution metrics and status.
        """
        raise NotImplementedError


async def _execute_agent_with_budget(
    agent: PaymentAgent,
    context: TransactionContext,
    budget_ms: float,
    registry: CircuitBreakerRegistry,
    stage_name: str,
) -> AgentExecutionResult:
    """
    Execute a single agent under strict SLA, circuit breaking, and confidence guardrails.
    """
    breaker = registry.get_or_create(
        agent.name,
        fallback_handler=agent.fallback_handler,
    )
    # Update breaker config SLA if specified on agent
    if hasattr(agent, "sla_ms") and agent.sla_ms:
        breaker.config.sla_timeout_ms = agent.sla_ms
    if hasattr(agent, "confidence_floor") and agent.confidence_floor:
        breaker.config.confidence_floor = agent.confidence_floor

    state = breaker.get_state()
    start_perf = time.perf_counter()

    # 1. Fast-fail if circuit breaker is OPEN
    if state == CircuitState.OPEN:
        logger.warning(f"Agent [{agent.name}] circuit is OPEN. Bypassing to deterministic fallback.")
        fallback_res = await breaker.execute_fallback(
            context=context,
            reason="circuit_breaker_open",
            status=ExecutionStatus.CIRCUIT_OPEN,
            start_perf=start_perf,
        )
        context.record_agent_result(
            agent_name=agent.name,
            choice=fallback_res.choice,
            score=fallback_res.score,
            confidence=fallback_res.confidence,
            probabilities=fallback_res.probabilities,
            latency_ms=fallback_res.latency_ms,
            status=ExecutionStatus.CIRCUIT_OPEN,
            is_fallback=True,
            fallback_reason=fallback_res.fallback_reason,
            feature_contributions=fallback_res.feature_contributions,
            metadata=fallback_res.metadata,
            stage=stage_name,
        )
        return fallback_res

    # 2. Compute effective timeout for this execution
    effective_timeout_ms = min(agent.sla_ms, budget_ms)
    if effective_timeout_ms <= 0:
        logger.warning(f"Agent [{agent.name}] latency budget exhausted ({effective_timeout_ms:.2f}ms).")
        await breaker.record_failure("latency_budget_exhausted", is_timeout=True)
        fallback_res = await breaker.execute_fallback(
            context=context,
            reason="latency_budget_exhausted",
            status=ExecutionStatus.TIMEOUT,
            start_perf=start_perf,
        )
        context.record_agent_result(
            agent_name=agent.name,
            choice=fallback_res.choice,
            score=fallback_res.score,
            confidence=fallback_res.confidence,
            probabilities=fallback_res.probabilities,
            latency_ms=fallback_res.latency_ms,
            status=ExecutionStatus.TIMEOUT,
            is_fallback=True,
            fallback_reason="latency_budget_exhausted",
            metadata=fallback_res.metadata,
            stage=stage_name,
        )
        return fallback_res

    timeout_sec = effective_timeout_ms / 1000.0

    # 3. Execute agent with timeout
    try:
        raw_res = await asyncio.wait_for(agent.decide(context), timeout=timeout_sec)
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0

        # Check confidence floor
        confidence = float(getattr(raw_res, "confidence", 1.0))
        if confidence < agent.confidence_floor:
            logger.info(
                f"Agent [{agent.name}] returned low confidence ({confidence:.2f} < {agent.confidence_floor:.2f}). Triggering fallback."
            )
            await breaker.record_failure("low_confidence", is_low_confidence=True)
            fallback_res = await breaker.execute_fallback(
                context=context,
                reason=f"low_confidence_{confidence:.2f}",
                status=ExecutionStatus.LOW_CONFIDENCE_FALLBACK,
                start_perf=start_perf,
            )
            context.record_agent_result(
                agent_name=agent.name,
                choice=fallback_res.choice,
                score=fallback_res.score,
                confidence=fallback_res.confidence,
                probabilities=fallback_res.probabilities,
                latency_ms=round(elapsed_ms, 3),
                status=ExecutionStatus.LOW_CONFIDENCE_FALLBACK,
                is_fallback=True,
                fallback_reason=f"low_confidence: {confidence:.2f} < {agent.confidence_floor:.2f}",
                feature_contributions=getattr(raw_res, "feature_contributions", {}),
                metadata=getattr(raw_res, "metadata", {}),
                stage=stage_name,
            )
            return fallback_res

        # Agent succeeded normally
        await breaker.record_success(elapsed_ms, confidence)
        return context.record_agent_result(
            agent_name=agent.name,
            choice=getattr(raw_res, "choice", None),
            score=getattr(raw_res, "score", None),
            confidence=confidence,
            probabilities=getattr(raw_res, "probabilities", {}),
            latency_ms=round(elapsed_ms, 3),
            status=ExecutionStatus.SUCCESS,
            is_fallback=getattr(raw_res, "is_fallback", False),
            fallback_reason=getattr(raw_res, "fallback_reason", None),
            feature_contributions=getattr(raw_res, "feature_contributions", {}),
            metadata=getattr(raw_res, "metadata", {}),
            stage=stage_name,
        )

    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        logger.warning(
            f"Agent [{agent.name}] exceeded SLA ({effective_timeout_ms:.1f}ms). Timed out after {elapsed_ms:.2f}ms."
        )
        await breaker.record_failure("sla_timeout", is_timeout=True)
        fallback_res = await breaker.execute_fallback(
            context=context,
            reason=f"sla_timeout_exceeded_{effective_timeout_ms:.1f}ms",
            status=ExecutionStatus.TIMEOUT,
            start_perf=start_perf,
        )
        return context.record_agent_result(
            agent_name=agent.name,
            choice=fallback_res.choice,
            score=fallback_res.score,
            confidence=fallback_res.confidence,
            probabilities=fallback_res.probabilities,
            latency_ms=round(elapsed_ms, 3),
            status=ExecutionStatus.TIMEOUT,
            is_fallback=True,
            fallback_reason=f"sla_timeout_exceeded_{effective_timeout_ms:.1f}ms",
            metadata=fallback_res.metadata,
            stage=stage_name,
        )

    except Exception as ex:
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        logger.exception(f"Agent [{agent.name}] raised unexpected exception: {ex}")
        await breaker.record_failure(f"exception: {type(ex).__name__}")
        fallback_res = await breaker.execute_fallback(
            context=context,
            reason=f"error: {str(ex)}",
            status=ExecutionStatus.ERROR,
            start_perf=start_perf,
        )
        return context.record_agent_result(
            agent_name=agent.name,
            choice=fallback_res.choice,
            score=fallback_res.score,
            confidence=fallback_res.confidence,
            probabilities=fallback_res.probabilities,
            latency_ms=round(elapsed_ms, 3),
            status=ExecutionStatus.ERROR,
            is_fallback=True,
            fallback_reason=f"error: {str(ex)}",
            metadata={"exception": str(ex)},
            stage=stage_name,
        )


class ParallelStage(PipelineStage):
    """
    Executes multiple agents concurrently via asyncio.gather.
    Enforces per-agent latency SLA, concurrent circuit breaking,
    and consolidates outputs into the TransactionContext.
    """

    def __init__(
        self,
        name: str,
        agents: List[PaymentAgent],
        sla_ms: float = 25.0,
    ):
        self.name = name
        self.stage_type = "parallel"
        self.agents = agents
        self.sla_ms = sla_ms

    async def execute(
        self,
        context: TransactionContext,
        remaining_budget_ms: float,
        registry: CircuitBreakerRegistry,
    ) -> StageExecutionResult:
        start_perf = time.perf_counter()
        stage_budget = min(self.sla_ms, remaining_budget_ms)

        context.record_audit(
            event_type="STAGE_START",
            stage=self.name,
            details={
                "type": self.stage_type,
                "agents": [a.name for a in self.agents],
                "stage_budget_ms": stage_budget,
            },
        )

        tasks = [
            _execute_agent_with_budget(
                agent=agent,
                context=context,
                budget_ms=stage_budget,
                registry=registry,
                stage_name=self.name,
            )
            for agent in self.agents
        ]

        # Execute all parallel agents concurrently
        results: List[AgentExecutionResult] = await asyncio.gather(*tasks, return_exceptions=False)
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0

        # Assess stage health status
        has_fallback = any(r.is_fallback for r in results)
        all_failed = all(r.status in (ExecutionStatus.TIMEOUT, ExecutionStatus.ERROR) for r in results)

        if all_failed:
            status = ExecutionStatus.ERROR
        elif has_fallback:
            status = ExecutionStatus.LOW_CONFIDENCE_FALLBACK
        else:
            status = ExecutionStatus.SUCCESS

        return context.record_stage_result(
            stage_name=self.name,
            stage_type=self.stage_type,
            latency_ms=round(elapsed_ms, 3),
            status=status,
            agents_executed=[a.name for a in self.agents],
        )


class SequentialStage(PipelineStage):
    """
    Executes agents in sequence. Each agent can inspect outputs produced
    by earlier agents in the transaction context.
    """

    def __init__(
        self,
        name: str,
        agents: List[PaymentAgent],
        sla_ms: float = 25.0,
    ):
        self.name = name
        self.stage_type = "sequential"
        self.agents = agents
        self.sla_ms = sla_ms

    async def execute(
        self,
        context: TransactionContext,
        remaining_budget_ms: float,
        registry: CircuitBreakerRegistry,
    ) -> StageExecutionResult:
        start_perf = time.perf_counter()
        stage_budget = min(self.sla_ms, remaining_budget_ms)

        context.record_audit(
            event_type="STAGE_START",
            stage=self.name,
            details={
                "type": self.stage_type,
                "agents": [a.name for a in self.agents],
                "stage_budget_ms": stage_budget,
            },
        )

        executed_names: List[str] = []
        for agent in self.agents:
            if context.is_short_circuited:
                break

            current_elapsed = (time.perf_counter() - start_perf) * 1000.0
            agent_remaining = stage_budget - current_elapsed

            await _execute_agent_with_budget(
                agent=agent,
                context=context,
                budget_ms=agent_remaining,
                registry=registry,
                stage_name=self.name,
            )
            executed_names.append(agent.name)

        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        return context.record_stage_result(
            stage_name=self.name,
            stage_type=self.stage_type,
            latency_ms=round(elapsed_ms, 3),
            status=ExecutionStatus.SUCCESS,
            agents_executed=executed_names,
            short_circuited=context.is_short_circuited,
            short_circuit_reason=context.short_circuit_reason,
        )


class ConditionalStage(PipelineStage):
    """
    Executes an inner stage only if a condition predicate evaluates to True.
    Useful for running smart retry only on declines, or skipping routing on fraud blocks.
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[TransactionContext], bool],
        inner_stage: PipelineStage,
        else_stage: Optional[PipelineStage] = None,
        sla_ms: float = 25.0,
    ):
        self.name = name
        self.stage_type = "conditional"
        self.condition = condition
        self.inner_stage = inner_stage
        self.else_stage = else_stage
        self.sla_ms = sla_ms

    async def execute(
        self,
        context: TransactionContext,
        remaining_budget_ms: float,
        registry: CircuitBreakerRegistry,
    ) -> StageExecutionResult:
        start_perf = time.perf_counter()

        condition_met = False
        try:
            condition_met = bool(self.condition(context))
        except Exception as ex:
            logger.warning(f"Conditional stage [{self.name}] condition raised: {ex}. Treating as False.")

        if condition_met:
            stage_to_run = self.inner_stage
        elif self.else_stage is not None:
            stage_to_run = self.else_stage
        else:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            return context.record_stage_result(
                stage_name=self.name,
                stage_type=self.stage_type,
                latency_ms=round(elapsed_ms, 3),
                status=ExecutionStatus.SKIPPED,
                agents_executed=[],
            )

        res = await stage_to_run.execute(context, remaining_budget_ms, registry)
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        res.latency_ms = round(elapsed_ms, 3)
        return res
