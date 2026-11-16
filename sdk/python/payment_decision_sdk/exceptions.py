"""Exceptions for the Payment Decision SDK."""


class DecisionSDKError(Exception):
    """Base exception for all errors raised by the Payment Decision SDK."""


class InvalidContextError(DecisionSDKError):
    """Raised when payment transaction context fails validation."""


class SidecarConnectionError(DecisionSDKError):
    """Raised when the SDK cannot establish a connection to the sidecar microservice."""


class DecisionTimeoutError(DecisionSDKError):
    """Raised when decision evaluation exceeds the configured latency timeout budget."""


class EngineLoadError(DecisionSDKError):
    """Raised when the local in-process model artifact or spec fails to load."""


class CircuitBreakerOpenError(DecisionSDKError):
    """Raised when repeated failures cause the circuit breaker to trip open."""
