"""Payment Decision SDK.

High-throughput, low-latency client SDK for intelligent payment routing and decisioning.
"""

from payment_decision_sdk.client import DecisionEngineClient
from payment_decision_sdk.exceptions import (
    CircuitBreakerOpenError,
    DecisionSDKError,
    DecisionTimeoutError,
    EngineLoadError,
    InvalidContextError,
    SidecarConnectionError,
)
from payment_decision_sdk.in_process import InProcessEngine
from payment_decision_sdk.models import (
    BatchDecisionRequest,
    BatchDecisionResponse,
    DecisionResult,
    PaymentContext,
)
from payment_decision_sdk.sdm import SpecialistSDMEvaluator, load_specialist
from payment_decision_sdk.sealed import SealedPackageClient
from payment_decision_sdk.sidecar_client import SidecarClient

__version__ = "1.0.0"

__all__ = [
    "DecisionEngineClient",
    "InProcessEngine",
    "SidecarClient",
    "SpecialistSDMEvaluator",
    "SealedPackageClient",
    "load_specialist",
    "PaymentContext",
    "DecisionResult",
    "BatchDecisionRequest",
    "BatchDecisionResponse",
    "DecisionSDKError",
    "InvalidContextError",
    "SidecarConnectionError",
    "DecisionTimeoutError",
    "EngineLoadError",
    "CircuitBreakerOpenError",
]
