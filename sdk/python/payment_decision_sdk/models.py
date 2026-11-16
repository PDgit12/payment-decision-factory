"""Payment Decision SDK - Data Models.

Defines typed schemas for payment context inputs and decision evaluation outputs.
Designed for sub-millisecond serialization and strict type safety.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PaymentContext(BaseModel):
    """Context information for a payment transaction to be routed."""

    amount: float = Field(..., gt=0, description="Transaction amount (e.g., 49.99)")
    currency: str = Field(..., min_length=3, max_length=3, description="3-letter ISO currency code, e.g., USD, EUR, GBP")
    card_brand: str = Field(..., description="Card network brand: visa, mastercard, amex, discover, etc.")
    bin_country: str = Field(..., min_length=2, max_length=2, description="2-letter ISO country code of issuing bank, e.g., US, GB, DE")
    mcc: str = Field(..., description="Merchant Category Code (e.g., '5411', '5732')")
    velocity_1h: int = Field(default=1, ge=0, description="Transaction count for cardholder in the past hour")
    customer_id: Optional[str] = Field(default=None, description="Optional customer or account identifier")
    payment_method_id: Optional[str] = Field(default=None, description="Optional tokenized payment method ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom key-value pairs")

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v: Any) -> str:
        return str(v).upper().strip()

    @field_validator("bin_country", mode="before")
    @classmethod
    def normalize_bin_country(cls, v: Any) -> str:
        return str(v).upper().strip()

    @field_validator("card_brand", mode="before")
    @classmethod
    def normalize_card_brand(cls, v: Any) -> str:
        return str(v).lower().strip()

    @field_validator("mcc", mode="before")
    @classmethod
    def normalize_mcc(cls, v: Any) -> str:
        return str(v).strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PaymentContext:
        """Create a PaymentContext instance from a dictionary."""
        return cls(**data)


class DecisionResult(BaseModel):
    """Calibrated routing or authorization decision produced by the engine."""

    decision: str = Field(..., description="Selected routing action (e.g., stripe_us, adyen_eu, checkout_crossborder)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence probability for the chosen action")
    probabilities: Dict[str, float] = Field(
        default_factory=dict, description="Calibrated probabilities across all candidate choices"
    )
    latency_ms: float = Field(..., ge=0.0, description="Inference and evaluation latency in milliseconds")
    routing_reason: Optional[str] = Field(default=None, description="Explainable rationale or primary decision rule")
    feature_contributions: Optional[Dict[str, float]] = Field(
        default_factory=dict, description="SHAP-style or feature weight attribution breakdown"
    )
    is_fallback: bool = Field(default=False, description="True if decision was produced by circuit breaker or safety fallback")
    engine_mode: str = Field(default="in_process", description="Execution mode: in_process | sidecar | fallback")
    model_version: str = Field(default="v1.0.0", description="Version of the model or decision spec deployed")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of decision evaluation")

    @property
    def choice(self) -> str:
        """Alias for backward/cross-compatibility with DecisionEngine.evaluate()."""
        return self.decision

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecisionResult:
        """Create a DecisionResult instance from a dictionary."""
        return cls(**data)


class BatchDecisionRequest(BaseModel):
    """Request schema for batch evaluation."""

    transactions: List[PaymentContext] = Field(..., description="List of payment transactions to evaluate")


class BatchDecisionResponse(BaseModel):
    """Response schema for batch evaluation."""

    results: List[DecisionResult] = Field(..., description="List of decision results matching input transactions")
    total_latency_ms: float = Field(..., description="Total batch execution time in milliseconds")
    count: int = Field(..., description="Number of transactions evaluated")
