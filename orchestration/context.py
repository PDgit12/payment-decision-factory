"""
Payment Transaction Context Accumulator.

Preserves payment inputs, agent decisions, confidence scores, execution latencies,
and an immutable chronological audit trail across the transaction lifecycle.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionStatus(str, Enum):
    """Execution status for agent or stage execution."""
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    LOW_CONFIDENCE_FALLBACK = "LOW_CONFIDENCE_FALLBACK"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class AgentExecutionResult:
    """Detailed record of an individual agent's decision and execution metrics."""
    agent_name: str
    choice: Optional[str] = None
    score: Optional[float] = None
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    executed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent execution result to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value if isinstance(self.status, ExecutionStatus) else str(self.status)
        return data


@dataclass
class StageExecutionResult:
    """Record of a pipeline stage execution."""
    stage_name: str
    stage_type: str  # "parallel", "sequential", "conditional"
    latency_ms: float = 0.0
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    agents_executed: List[str] = field(default_factory=list)
    short_circuited: bool = False
    short_circuit_reason: Optional[str] = None
    executed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert stage execution result to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value if isinstance(self.status, ExecutionStatus) else str(self.status)
        return data


@dataclass
class AuditEvent:
    """Individual entry in the transaction audit trail."""
    timestamp: str
    elapsed_ms: float
    event_type: str
    stage: Optional[str] = None
    agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TransactionContext:
    """
    Thread-safe / async-safe payment transaction context accumulator.

    Preserves raw transaction parameters, intermediate agent decisions,
    confidence metrics, and an audit trail throughout the decision pipeline.
    """

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tx_id: Optional[str] = None,
    ):
        self.transaction_id: str = tx_id or transaction_id or f"tx_{uuid.uuid4().hex[:12]}"
        self.tx_id: str = self.transaction_id
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self._start_perf: float = time.perf_counter()

        # Inputs
        self.inputs: Dict[str, Any] = dict(inputs or {})
        self.metadata: Dict[str, Any] = dict(metadata or {})

        # Execution tracking
        self.agent_results: Dict[str, AgentExecutionResult] = {}
        self.stage_results: List[StageExecutionResult] = []
        self.audit_trail: List[AuditEvent] = []

        # Short circuit and final resolution
        self.is_short_circuited: bool = False
        self.short_circuit_reason: Optional[str] = None
        self.final_decision: Optional[Dict[str, Any]] = None
        self.total_latency_ms: float = 0.0

        # Initial audit log
        self.record_audit(
            event_type="PIPELINE_INIT",
            details={
                "transaction_id": self.transaction_id,
                "input_keys": list(self.inputs.keys()),
                "amount": self.inputs.get("amount"),
                "currency": self.inputs.get("currency"),
            },
        )

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds since context creation."""
        return (time.perf_counter() - self._start_perf) * 1000.0

    def get(self, key: str, default: Any = None) -> Any:
        """
        Hierarchical context retriever.

        Order of resolution:
        1. Direct key match in inputs (e.g., 'amount', 'currency')
        2. Dotted agent result accessor (e.g., 'fraud_screener.score', 'dynamic_3ds.choice')
        3. Common semantic aliases:
           - 'fraud_score' -> agent_results['fraud_screener'].score
           - 'fraud_choice' -> agent_results['fraud_screener'].choice
           - 'three_ds_choice' / 'sca_action' -> agent_results['dynamic_3ds'].choice
           - 'routing_choice' -> agent_results['smart_routing'].choice
        4. Key match in metadata
        5. Default value
        """
        # 1. Inputs
        if key in self.inputs:
            return self.inputs[key]

        # 2. Dotted path
        if "." in key:
            prefix, attr = key.split(".", 1)
            if prefix in self.agent_results:
                res = self.agent_results[prefix]
                if hasattr(res, attr):
                    return getattr(res, attr)
                if attr in res.metadata:
                    return res.metadata[attr]

        # 3. Common semantic aliases
        alias_map = {
            "fraud_score": ("fraud_screener", "score"),
            "fraud_decision": ("fraud_screener", "choice"),
            "fraud_choice": ("fraud_screener", "choice"),
            "three_ds_choice": ("dynamic_3ds", "choice"),
            "sca_action": ("dynamic_3ds", "choice"),
            "gateway_choice": ("smart_routing", "choice"),
            "routing_choice": ("smart_routing", "choice"),
            "retry_hours": ("smart_retry", "score"),
        }
        if key in alias_map:
            agent_name, attr = alias_map[key]
            if agent_name in self.agent_results:
                return getattr(self.agent_results[agent_name], attr, default)

        # 4. Metadata
        if key in self.metadata:
            return self.metadata[key]

        return default

    def set_input(self, key: str, value: Any) -> None:
        """Add or update an input feature."""
        self.inputs[key] = value

    def update_inputs(self, new_inputs: Dict[str, Any]) -> None:
        """Update multiple input features."""
        self.inputs.update(new_inputs)

    def record_agent_result(
        self,
        agent_name: str,
        choice: Optional[str] = None,
        score: Optional[float] = None,
        confidence: float = 0.0,
        probabilities: Optional[Dict[str, float]] = None,
        latency_ms: float = 0.0,
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        is_fallback: bool = False,
        fallback_reason: Optional[str] = None,
        feature_contributions: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None,
    ) -> AgentExecutionResult:
        """Record an agent's execution output in the context and audit trail."""
        result = AgentExecutionResult(
            agent_name=agent_name,
            choice=choice,
            score=score,
            confidence=confidence,
            probabilities=probabilities or {},
            latency_ms=latency_ms,
            status=status,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            feature_contributions=feature_contributions or {},
            metadata=metadata or {},
        )
        self.agent_results[agent_name] = result

        # Also store directly accessible keys for downstream convenience
        if choice is not None:
            self.inputs[f"{agent_name}_choice"] = choice
        if score is not None:
            self.inputs[f"{agent_name}_score"] = score
        self.inputs[f"{agent_name}_confidence"] = confidence

        # Record audit event
        self.record_audit(
            event_type="AGENT_EXECUTION_RECORDED",
            stage=stage,
            agent=agent_name,
            details={
                "choice": choice,
                "score": score,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "status": status.value if isinstance(status, ExecutionStatus) else str(status),
                "is_fallback": is_fallback,
                "fallback_reason": fallback_reason,
            },
        )
        return result

    def get_agent_result(self, agent_name: str) -> Optional[AgentExecutionResult]:
        """Retrieve execution result for a specific agent."""
        return self.agent_results.get(agent_name)

    def record_stage_result(
        self,
        stage_name: str,
        stage_type: str,
        latency_ms: float,
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        agents_executed: Optional[List[str]] = None,
        short_circuited: bool = False,
        short_circuit_reason: Optional[str] = None,
    ) -> StageExecutionResult:
        """Record a pipeline stage execution."""
        stage_result = StageExecutionResult(
            stage_name=stage_name,
            stage_type=stage_type,
            latency_ms=latency_ms,
            status=status,
            agents_executed=agents_executed or [],
            short_circuited=short_circuited,
            short_circuit_reason=short_circuit_reason,
        )
        self.stage_results.append(stage_result)

        self.record_audit(
            event_type="STAGE_COMPLETED",
            stage=stage_name,
            details={
                "stage_type": stage_type,
                "latency_ms": latency_ms,
                "status": status.value if isinstance(status, ExecutionStatus) else str(status),
                "agents": agents_executed or [],
                "short_circuited": short_circuited,
                "short_circuit_reason": short_circuit_reason,
            },
        )
        return stage_result

    def record_audit(
        self,
        event_type: str,
        stage: Optional[str] = None,
        agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Append an event to the chronological audit trail."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=round(self.elapsed_ms, 3),
            event_type=event_type,
            stage=stage,
            agent=agent,
            details=details or {},
        )
        self.audit_trail.append(event)
        return event

    def short_circuit(
        self,
        reason: str,
        final_choice: Optional[str] = None,
        final_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Trigger an early short-circuit termination of the pipeline
        (e.g., hard fraud decline, velocity attack threshold exceeded).
        """
        self.is_short_circuited = True
        self.short_circuit_reason = reason
        self.final_decision = {
            "choice": final_choice,
            "score": final_score,
            "short_circuited": True,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self.record_audit(
            event_type="PIPELINE_SHORT_CIRCUIT",
            details={
                "reason": reason,
                "final_choice": final_choice,
                "final_score": final_score,
            },
        )

    def finalize(
        self,
        final_choice: Optional[str] = None,
        final_score: Optional[float] = None,
        primary_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Finalize the transaction context and compute total latency."""
        self.total_latency_ms = round(self.elapsed_ms, 3)

        if not self.final_decision:
            # If not already short-circuited, resolve final decision from primary agent or arguments
            agent_res = self.agent_results.get(primary_agent) if primary_agent else None
            resolved_choice = final_choice or (agent_res.choice if agent_res else None)
            resolved_score = final_score if final_score is not None else (agent_res.score if agent_res else None)
            confidence = agent_res.confidence if agent_res else 1.0
            is_fallback = agent_res.is_fallback if agent_res else False

            self.final_decision = {
                "choice": resolved_choice,
                "score": resolved_score,
                "confidence": confidence,
                "primary_agent": primary_agent,
                "is_fallback": is_fallback,
                "short_circuited": False,
                "metadata": metadata or {},
            }

        self.final_decision["total_latency_ms"] = self.total_latency_ms
        self.final_decision["transaction_id"] = self.transaction_id

        self.record_audit(
            event_type="PIPELINE_FINALIZE",
            details={
                "total_latency_ms": self.total_latency_ms,
                "final_choice": self.final_decision.get("choice"),
                "is_fallback": self.final_decision.get("is_fallback"),
                "short_circuited": self.is_short_circuited,
            },
        )
        return self.final_decision

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full context to nested dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "created_at": self.created_at,
            "total_latency_ms": self.total_latency_ms or round(self.elapsed_ms, 3),
            "is_short_circuited": self.is_short_circuited,
            "short_circuit_reason": self.short_circuit_reason,
            "final_decision": self.final_decision,
            "inputs": self.inputs,
            "metadata": self.metadata,
            "agent_results": {k: v.to_dict() for k, v in self.agent_results.items()},
            "stage_results": [s.to_dict() for s in self.stage_results],
            "audit_trail": [a.to_dict() for a in self.audit_trail],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize context to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> Dict[str, Any]:
        """Concise summary for telemetry, logging, and response payloads."""
        return {
            "transaction_id": self.transaction_id,
            "choice": self.final_decision.get("choice") if self.final_decision else None,
            "score": self.final_decision.get("score") if self.final_decision else None,
            "confidence": self.final_decision.get("confidence") if self.final_decision else 0.0,
            "is_fallback": self.final_decision.get("is_fallback", False) if self.final_decision else False,
            "short_circuited": self.is_short_circuited,
            "total_latency_ms": self.total_latency_ms or round(self.elapsed_ms, 3),
            "agents_run": list(self.agent_results.keys()),
            "stages_run": [s.stage_name for s in self.stage_results],
        }
