"""
Circuit Breaker and Deterministic Fallback Mechanisms for Payment Agents.

Implements high-resilience circuit breaking (CLOSED -> OPEN -> HALF_OPEN)
with fast-fail semantics, rolling failure tracking, per-agent SLA monitoring,
and automatic deterministic backup rules.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from .context import AgentExecutionResult, ExecutionStatus, TransactionContext

logger = logging.getLogger("orchestrator.circuit_breaker")


class CircuitState(str, Enum):
    """Circuit breaker operational state."""
    CLOSED = "CLOSED"      # Normal operation: requests pass through to agent
    OPEN = "OPEN"          # Failing: fast-fail directly to deterministic fallback
    HALF_OPEN = "HALF_OPEN"# Trial recovery: allow single probe request to check agent health


@dataclass
class CircuitBreakerConfig:
    """Configuration parameters for an agent's circuit breaker."""
    failure_threshold: int = 3             # Consecutive failures before opening circuit
    recovery_timeout_sec: float = 3.0      # Time in seconds circuit remains open before half-open trial
    half_open_successes: int = 1           # Successful probe requests needed to close circuit
    confidence_floor: float = 0.40         # Low-confidence threshold triggering fallback
    sla_timeout_ms: float = 20.0           # Per-agent latency budget in milliseconds


@dataclass
class CircuitMetrics:
    """Real-time metrics for circuit health and telemetry."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    low_confidence_calls: int = 0
    circuit_opened_count: int = 0
    last_failure_reason: Optional[str] = None
    last_state_change: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "timeout_calls": self.timeout_calls,
            "low_confidence_calls": self.low_confidence_calls,
            "circuit_opened_count": self.circuit_opened_count,
            "last_failure_reason": self.last_failure_reason,
            "failure_rate": (
                round(self.failed_calls / self.total_calls, 3) if self.total_calls > 0 else 0.0
            ),
        }


# Type alias for deterministic fallback handler
FallbackHandler = Union[
    Callable[[TransactionContext, str], AgentExecutionResult],
    Callable[[TransactionContext, str], Coroutine[Any, Any, AgentExecutionResult]],
    AgentExecutionResult,
]


class CircuitBreaker:
    """
    Per-agent circuit breaker with latency SLA enforcement,
    consecutive failure tracking, and deterministic fallback execution.
    """

    def __init__(
        self,
        agent_name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_handler: Optional[FallbackHandler] = None,
    ):
        self.agent_name = agent_name
        self.config = config or CircuitBreakerConfig()
        self.fallback_handler = fallback_handler

        self.state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.consecutive_successes: int = 0
        self.last_failure_time: float = 0.0
        self.metrics = CircuitMetrics()
        self._lock = asyncio.Lock()

    def get_state(self) -> CircuitState:
        """Check current state, transitioning from OPEN to HALF_OPEN if timeout elapsed."""
        now = time.time()
        if self.state == CircuitState.OPEN:
            if (now - self.last_failure_time) >= self.config.recovery_timeout_sec:
                self.state = CircuitState.HALF_OPEN
                self.consecutive_successes = 0
                logger.info(f"CircuitBreaker[{self.agent_name}] transitioned OPEN -> HALF_OPEN (probe mode)")
        return self.state

    async def record_success(self, latency_ms: float, confidence: float) -> None:
        """Record successful call, potentially closing a HALF_OPEN circuit."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.consecutive_failures = 0

            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.config.half_open_successes:
                    self.state = CircuitState.CLOSED
                    self.consecutive_successes = 0
                    logger.info(f"CircuitBreaker[{self.agent_name}] probe succeeded; circuit CLOSED.")

    async def record_failure(self, reason: str, is_timeout: bool = False, is_low_confidence: bool = False) -> None:
        """Record failure or SLA breach, opening circuit if threshold reached."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            if is_timeout:
                self.metrics.timeout_calls += 1
            if is_low_confidence:
                self.metrics.low_confidence_calls += 1

            self.metrics.last_failure_reason = reason
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_failure_time = time.time()

            if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if self.consecutive_failures >= self.config.failure_threshold or self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.OPEN
                    self.metrics.circuit_opened_count += 1
                    logger.warning(
                        f"CircuitBreaker[{self.agent_name}] OPENED! Consecutive failures: {self.consecutive_failures}. Reason: {reason}"
                    )

    async def execute_fallback(
        self,
        context: TransactionContext,
        reason: str,
        status: ExecutionStatus = ExecutionStatus.ERROR,
        start_perf: Optional[float] = None,
    ) -> AgentExecutionResult:
        """Execute deterministic backup rule without invoking failing agent."""
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0 if start_perf else 0.1

        if self.fallback_handler is not None:
            if isinstance(self.fallback_handler, AgentExecutionResult):
                fb = self.fallback_handler
                return AgentExecutionResult(
                    agent_name=self.agent_name,
                    choice=fb.choice,
                    score=fb.score,
                    confidence=fb.confidence,
                    probabilities=dict(fb.probabilities),
                    latency_ms=round(elapsed_ms, 3),
                    status=status,
                    is_fallback=True,
                    fallback_reason=reason,
                    metadata={"fallback_source": "static_template"},
                )
            elif callable(self.fallback_handler):
                try:
                    if asyncio.iscoroutinefunction(self.fallback_handler):
                        res = await self.fallback_handler(context, reason)
                    else:
                        res = self.fallback_handler(context, reason)

                    if isinstance(res, AgentExecutionResult):
                        res.agent_name = self.agent_name
                        res.is_fallback = True
                        res.fallback_reason = reason
                        res.status = status
                        res.latency_ms = round(elapsed_ms, 3)
                        return res
                    elif isinstance(res, dict):
                        return AgentExecutionResult(
                            agent_name=self.agent_name,
                            choice=res.get("choice"),
                            score=res.get("score"),
                            confidence=res.get("confidence", 0.5),
                            probabilities=res.get("probabilities", {}),
                            latency_ms=round(elapsed_ms, 3),
                            status=status,
                            is_fallback=True,
                            fallback_reason=reason,
                            metadata=res.get("metadata", {}),
                        )
                except Exception as ex:
                    logger.exception(f"Error in fallback handler for {self.agent_name}: {ex}")

        # Default deterministic fallback when no specific handler is attached
        return self._default_deterministic_rule(context, reason, status, elapsed_ms)

    def _default_deterministic_rule(
        self,
        context: TransactionContext,
        reason: str,
        status: ExecutionStatus,
        elapsed_ms: float,
    ) -> AgentExecutionResult:
        """
        Industry-standard deterministic backup rules by agent domain.
        """
        agent = self.agent_name.lower()
        amount = float(context.get("amount", 0.0))
        currency = str(context.get("currency", "USD")).upper()
        bin_country = str(context.get("bin_country", "US")).upper()
        velocity_1h = int(context.get("velocity_1h", 0))

        if "fraud" in agent:
            # Deterministic Fraud Screener Backup:
            # High ticket or velocity triggers step-down challenge; low ticket allows
            if amount > 500.0 or velocity_1h > 5:
                choice = "challenge"
                score = 0.75
            else:
                choice = "allow"
                score = 0.15
            return AgentExecutionResult(
                agent_name=self.agent_name,
                choice=choice,
                score=score,
                confidence=0.85,
                latency_ms=round(elapsed_ms, 3),
                status=status,
                is_fallback=True,
                fallback_reason=f"deterministic_fraud_rule: {reason}",
                metadata={"rule": "amount_velocity_heuristic", "reason": reason},
            )

        elif "3ds" in agent or "sca" in agent:
            # Deterministic Dynamic 3DS Backup:
            # PSD2 safe default: challenge transactions in EEA; low value exempt if < €30
            eea_countries = {"DE", "FR", "NL", "IE", "ES", "IT", "SE", "NO", "DK", "BE", "AT", "CH", "GB"}
            if currency == "EUR" and amount < 30.0 and velocity_1h < 3:
                choice = "exemption_low_value"
            elif bin_country in eea_countries:
                choice = "challenge"
            else:
                choice = "frictionless"
            return AgentExecutionResult(
                agent_name=self.agent_name,
                choice=choice,
                confidence=0.90,
                latency_ms=round(elapsed_ms, 3),
                status=status,
                is_fallback=True,
                fallback_reason=f"deterministic_3ds_rule: {reason}",
                metadata={"rule": "psd2_compliance_heuristic", "reason": reason},
            )

        elif "retry" in agent:
            # Deterministic Smart Retry Backup:
            # Exponential backoff heuristic
            return AgentExecutionResult(
                agent_name=self.agent_name,
                choice="retry",
                score=4.0,  # 4 hours backoff
                confidence=0.75,
                latency_ms=round(elapsed_ms, 3),
                status=status,
                is_fallback=True,
                fallback_reason=f"deterministic_retry_rule: {reason}",
                metadata={"rule": "exponential_backoff", "backoff_hours": 4.0},
            )

        else:
            # Smart Gateway Routing Backup:
            # Deterministic geography & currency dispatch
            eu_countries = {"DE", "FR", "NL", "IE", "ES", "IT", "SE", "NO", "DK", "BE", "AT", "CH", "GB"}
            if bin_country in ("US", "CA") or currency == "USD":
                choice = "stripe_us"
            elif bin_country in eu_countries or currency == "EUR":
                choice = "adyen_eu"
            else:
                choice = "checkout_crossborder"

            return AgentExecutionResult(
                agent_name=self.agent_name,
                choice=choice,
                confidence=0.80,
                probabilities={choice: 0.80},
                latency_ms=round(elapsed_ms, 3),
                status=status,
                is_fallback=True,
                fallback_reason=f"deterministic_routing_rule: {reason}",
                metadata={"rule": "geo_currency_affinity", "reason": reason},
            )


class CircuitBreakerRegistry:
    """Thread-safe registry for managing multiple agent circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        agent_name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_handler: Optional[FallbackHandler] = None,
    ) -> CircuitBreaker:
        if agent_name not in self._breakers:
            self._breakers[agent_name] = CircuitBreaker(
                agent_name=agent_name,
                config=config,
                fallback_handler=fallback_handler,
            )
        return self._breakers[agent_name]

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED."""
        for cb in self._breakers.values():
            cb.state = CircuitState.CLOSED
            cb.consecutive_failures = 0
            cb.consecutive_successes = 0

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return metrics dict for all registered circuit breakers."""
        return {
            name: {
                "state": cb.get_state().value,
                "metrics": cb.metrics.to_dict(),
            }
            for name, cb in self._breakers.items()
        }
