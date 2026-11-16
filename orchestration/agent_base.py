"""
Base Agent Protocol and Adapters for Payment Decision Models.

Provides unified interface for both Laya-style core decision engines
and specialized multi-agent models (Fraud Screener, Dynamic 3DS, Smart Retry, Smart Routing).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

from .circuit_breaker import FallbackHandler
from .context import AgentExecutionResult, ExecutionStatus, TransactionContext

logger = logging.getLogger("orchestrator.agent")


class PaymentAgent(ABC):
    """Abstract base class for all payment decision agents."""

    name: str
    sla_ms: float = 20.0
    confidence_floor: float = 0.40
    fallback_handler: Optional[FallbackHandler] = None

    @abstractmethod
    async def decide(self, context: TransactionContext) -> AgentExecutionResult:
        """
        Execute agent decision logic on current transaction context.

        Args:
            context: Transaction context containing inputs, history, and upstream results.

        Returns:
            AgentExecutionResult containing choice/score, confidence, and metadata.
        """
        raise NotImplementedError


class CallableAgent(PaymentAgent):
    """Wraps any async or sync callable function into a PaymentAgent."""

    def __init__(
        self,
        name: str,
        decide_fn: Callable[[TransactionContext], Any],
        sla_ms: float = 20.0,
        confidence_floor: float = 0.40,
        fallback_handler: Optional[FallbackHandler] = None,
    ):
        self.name = name
        self.decide_fn = decide_fn
        self.sla_ms = sla_ms
        self.confidence_floor = confidence_floor
        self.fallback_handler = fallback_handler

    async def decide(self, context: TransactionContext) -> AgentExecutionResult:
        start_perf = time.perf_counter()
        if asyncio.iscoroutinefunction(self.decide_fn):
            raw = await self.decide_fn(context)
        else:
            raw = self.decide_fn(context)

        latency_ms = (time.perf_counter() - start_perf) * 1000.0

        if isinstance(raw, AgentExecutionResult):
            raw.agent_name = self.name
            if raw.latency_ms <= 0:
                raw.latency_ms = round(latency_ms, 3)
            return raw

        # Convert dict or object
        choice = getattr(raw, "choice", None) or (raw.get("choice") if isinstance(raw, dict) else str(raw))
        score = getattr(raw, "score", None) or (raw.get("score") if isinstance(raw, dict) else None)
        confidence = getattr(raw, "confidence", 1.0) or (raw.get("confidence", 1.0) if isinstance(raw, dict) else 1.0)
        probabilities = getattr(raw, "probabilities", {}) or (raw.get("probabilities", {}) if isinstance(raw, dict) else {})
        metadata = getattr(raw, "metadata", {}) or (raw.get("metadata", {}) if isinstance(raw, dict) else {})
        feature_contributions = getattr(raw, "feature_contributions", {}) or (
            raw.get("feature_contributions", {}) if isinstance(raw, dict) else {}
        )

        return AgentExecutionResult(
            agent_name=self.name,
            choice=choice,
            score=score,
            confidence=float(confidence),
            probabilities=probabilities,
            latency_ms=round(latency_ms, 3),
            status=ExecutionStatus.SUCCESS,
            feature_contributions=feature_contributions,
            metadata=metadata,
        )


class EngineAgentAdapter(PaymentAgent):
    """
    Adapter that integrates core.engine.DecisionEngine or any spec-driven engine
    with the asynchronous multi-agent orchestration pipeline.
    """

    def __init__(
        self,
        name: str,
        engine: Any,
        sla_ms: float = 20.0,
        confidence_floor: float = 0.40,
        fallback_handler: Optional[FallbackHandler] = None,
    ):
        self.name = name
        self.engine = engine
        self.sla_ms = sla_ms
        self.confidence_floor = confidence_floor
        self.fallback_handler = fallback_handler

    async def decide(self, context: TransactionContext) -> AgentExecutionResult:
        start_perf = time.perf_counter()

        # Engine evaluate call (support async or sync evaluate/decide)
        eval_method = getattr(self.engine, "decide", None) or getattr(self.engine, "evaluate", None)
        if eval_method is None:
            raise AttributeError(f"Engine {self.engine} has neither 'decide' nor 'evaluate' method.")

        if asyncio.iscoroutinefunction(eval_method):
            res = await eval_method(context)
        else:
            # Run in threadpool if engine is synchronous to prevent blocking async event loop
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, eval_method, context.inputs)

        latency_ms = (time.perf_counter() - start_perf) * 1000.0

        if isinstance(res, AgentExecutionResult):
            res.agent_name = self.name
            res.latency_ms = round(latency_ms, 3)
            return res

        # Extract attributes from DecisionResult (or dict)
        choice = getattr(res, "choice", None) or (res.get("choice") if isinstance(res, dict) else None)
        score = getattr(res, "score", None) or (res.get("score") if isinstance(res, dict) else None)
        confidence = float(getattr(res, "confidence", 1.0) or (res.get("confidence", 1.0) if isinstance(res, dict) else 1.0))
        probabilities = getattr(res, "probabilities", {}) or (res.get("probabilities", {}) if isinstance(res, dict) else {})
        feature_contributions = getattr(res, "feature_contributions", {}) or (
            res.get("feature_contributions", {}) if isinstance(res, dict) else {}
        )
        is_fallback = getattr(res, "is_fallback", False) or (res.get("is_fallback", False) if isinstance(res, dict) else False)
        routing_reason = getattr(res, "routing_reason", "") or (res.get("routing_reason", "") if isinstance(res, dict) else "")
        metadata = getattr(res, "metadata", {}) or (res.get("metadata", {}) if isinstance(res, dict) else {})
        if routing_reason:
            metadata["routing_reason"] = routing_reason

        return AgentExecutionResult(
            agent_name=self.name,
            choice=choice,
            score=score,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=round(latency_ms, 3),
            status=ExecutionStatus.SUCCESS,
            is_fallback=is_fallback,
            feature_contributions=feature_contributions,
            metadata=metadata,
        )


class SpecialistSDMAgentAdapter(PaymentAgent):
    """
    Adapter that integrates a compact Specialist SDM (Small Decision Model)
    into the asynchronous multi-agent orchestration pipeline.

    Supports:
    - Direct evaluation via SpecialistPipeline and SpecialistSDM PyTorch model.
    - Automatic input extraction from TransactionContext.
    - Sub-0.2ms CPU execution with calibrated temperature-scaled probabilities.
    - Low-confidence and SLA fallback handling.
    """

    def __init__(
        self,
        name: str,
        sdm_model: Any,
        sdm_pipeline: Any,
        sla_ms: float = 15.0,
        confidence_floor: float = 0.40,
        fallback_handler: Optional[FallbackHandler] = None,
    ):
        self.name = name
        self.sdm_model = sdm_model
        self.sdm_pipeline = sdm_pipeline
        self.sla_ms = sla_ms
        self.confidence_floor = confidence_floor
        self.fallback_handler = fallback_handler

    async def decide(self, context: TransactionContext) -> AgentExecutionResult:
        start_perf = time.perf_counter()

        # Extract normalized inputs from context
        inputs = dict(context.inputs)
        if "fraud_score" not in inputs:
            fs = context.get("fraud_score")
            if fs is not None:
                inputs["fraud_score"] = float(fs)
        if "three_ds_choice" not in inputs:
            sc = context.get("three_ds_choice")
            if sc is not None:
                inputs["three_ds_choice"] = str(sc)

        # Fast in-process inference (< 0.1ms on CPU)
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            self.sdm_pipeline.evaluate_transaction,
            self.sdm_model,
            inputs,
        )

        latency_ms = (time.perf_counter() - start_perf) * 1000.0

        score_val = getattr(res, "score", None)
        if score_val is None:
            scores_dict = getattr(res, "scores", None)
            if isinstance(scores_dict, dict) and hasattr(res, "choice") and res.choice in scores_dict:
                score_val = scores_dict[res.choice]
            else:
                score_val = getattr(res, "confidence", 1.0)

        return AgentExecutionResult(
            agent_name=self.name,
            choice=getattr(res, "choice", None),
            score=score_val,
            confidence=float(getattr(res, "confidence", 1.0)),
            probabilities=getattr(res, "probabilities", {}) or {},
            latency_ms=round(latency_ms, 3),
            status=ExecutionStatus.SUCCESS,
            is_fallback=getattr(res, "is_fallback", False),
            feature_contributions=getattr(res, "feature_contributions", {}) or {},
            metadata=getattr(res, "metadata", {}) or {},
        )



class MockAgent(PaymentAgent):
    """
    Controllable mock agent for testing SLAs, timeouts, failures,
    and circuit breaking under edge conditions.
    """

    def __init__(
        self,
        name: str = "mock_agent",
        choice: Optional[str] = "approved",
        score: Optional[float] = None,
        confidence: float = 0.95,
        probabilities: Optional[Dict[str, float]] = None,
        sleep_ms: float = 2.0,
        fail: bool = False,
        fail_exception: Optional[Exception] = None,
        sla_ms: float = 20.0,
        confidence_floor: float = 0.40,
        fallback_handler: Optional[FallbackHandler] = None,
        return_choice: Optional[str] = None,
    ):
        if return_choice is not None:
            choice = return_choice
        self.name = name
        self.choice = choice
        self.score = score
        self.confidence = confidence
        self.probabilities = probabilities or {}
        self.sleep_ms = sleep_ms
        self.fail = fail
        self.fail_exception = fail_exception
        self.sla_ms = sla_ms
        self.confidence_floor = confidence_floor
        self.fallback_handler = fallback_handler
        self.call_count = 0

    async def decide(self, context: TransactionContext) -> AgentExecutionResult:
        self.call_count += 1
        start_perf = time.perf_counter()

        if self.sleep_ms > 0:
            await asyncio.sleep(self.sleep_ms / 1000.0)

        if self.fail:
            raise self.fail_exception or RuntimeError(f"Mock failure in agent {self.name}")

        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0

        return AgentExecutionResult(
            agent_name=self.name,
            choice=self.choice,
            score=self.score,
            confidence=self.confidence,
            probabilities=self.probabilities or ({self.choice: self.confidence} if self.choice else {}),
            latency_ms=round(elapsed_ms, 3),
            status=ExecutionStatus.SUCCESS,
            metadata={"call_count": self.call_count},
        )
