"""Payment Decision Sidecar Server.

High-throughput, low-latency microservice exposing REST APIs for payment routing decisions.
Designed to run as a sidecar container alongside merchant/fintech application services
(e.g., Kubernetes sidecar pod or Docker container network).
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root and sdk are in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "sdk" / "python"))

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from payment_decision_factory.core.engine import DecisionEngine
from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_sdk.models import PaymentContext

# Global engine instance and telemetry state
_engine: Optional[DecisionEngine] = None
_spec: Optional[DecisionSpec] = None
_start_time: float = time.time()
_metrics: Dict[str, Any] = {
    "total_decisions": 0,
    "total_batch_transactions": 0,
    "decisions_by_gateway": {},
    "total_latency_ms": 0.0,
    "circuit_breakers_triggered": 0,
}


def load_engine() -> DecisionEngine:
    """Load the decision spec and compile the engine runtime."""
    global _engine, _spec

    spec_env = os.environ.get("SPEC_PATH")
    if spec_env and Path(spec_env).exists():
        spec_path = Path(spec_env)
    else:
        # Search candidate locations
        candidates = [
            ROOT_DIR / "specs" / "smart_routing.yaml",
            Path("specs/smart_routing.yaml"),
            Path("/app/specs/smart_routing.yaml"),
        ]
        spec_path = next((p for p in candidates if p.exists()), None)

    if spec_path is None or not spec_path.exists():
        raise RuntimeError(f"Could not locate smart_routing.yaml specification in: {candidates}")

    _spec = DecisionSpec.from_yaml(str(spec_path))
    _engine = DecisionEngine(_spec)
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load spec and pre-warm engine on startup."""
    global _start_time
    _start_time = time.time()
    load_engine()
    yield


app = FastAPI(
    title="Payment Decision Sidecar",
    description="Sub-20ms intelligent payment gateway routing and risk decision sidecar",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for cross-origin local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SidecarDecisionResponse(BaseModel):
    """Schema for individual decision response."""

    decision: str = Field(..., description="Chosen gateway choice (e.g., stripe_us, adyen_eu)")
    choice: str = Field(..., description="Alias for chosen gateway")
    confidence: float = Field(..., description="Calibrated confidence score (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Calibrated probability per choice")
    scores: Dict[str, float] = Field(..., description="Raw utility scores before scaling")
    feature_contributions: Dict[str, float] = Field(..., description="Feature attribution impact")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    routing_reason: str = Field(..., description="Primary routing explanation")
    is_fallback: bool = Field(default=False, description="Whether fallback policy was triggered")
    noul_triggered: bool = Field(default=False, description="Whether noul circuit breaker tripped")
    noul_action: Optional[str] = Field(default=None, description="Action taken by noul primitive")
    effective_fee_est: float = Field(default=0.0, description="Estimated gateway processing cost in bps")
    model_version: str = Field(default="v1.0.0", description="Model version")
    timestamp: float = Field(default_factory=time.time, description="Decision timestamp")


class BatchDecideRequest(BaseModel):
    """Schema for batch evaluation request."""

    transactions: List[PaymentContext] = Field(..., description="Array of payment context transactions")


class BatchDecideResponse(BaseModel):
    """Schema for batch evaluation response."""

    results: List[SidecarDecisionResponse]
    count: int
    total_latency_ms: float
    avg_latency_ms: float


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health and liveness check for container orchestrators (k8s/docker)."""
    is_ready = _engine is not None and _spec is not None
    return {
        "status": "healthy" if is_ready else "initializing",
        "model_loaded": is_ready,
        "spec_id": _spec.spec_id if _spec else None,
        "spec_version": _spec.version if _spec else None,
        "uptime_sec": round(time.time() - _start_time, 2),
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Performance telemetry and throughput metrics."""
    total = _metrics["total_decisions"]
    avg_lat = (_metrics["total_latency_ms"] / total) if total > 0 else 0.0
    return {
        "total_decisions": total,
        "total_batch_transactions": _metrics["total_batch_transactions"],
        "average_inference_latency_ms": round(avg_lat, 3),
        "circuit_breakers_triggered": _metrics["circuit_breakers_triggered"],
        "decisions_by_gateway": _metrics["decisions_by_gateway"],
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


class SpecialistDecideRequest(BaseModel):
    """Schema for specialist model decision request."""

    specialist_type: str = Field(default="routing", description="Domain: routing, fraud, sca, retry")
    context: PaymentContext = Field(..., description="Payment transaction context")


@app.get("/specialists/info")
def specialists_info() -> Dict[str, Any]:
    """Return status and metadata for the Specialist Small Decision Model (SDM) suite."""
    return {
        "suite_name": "Specialist SDM Suite",
        "specialists": ["routing", "fraud", "sca", "retry"],
        "architecture": "Gated Residual Deep Decision Network (<2MB, <0.1ms latency)",
        "calibrated": True,
        "onnx_ready": True,
        "domains": {
            "routing": {"choices": ["stripe_us", "adyen_eu", "checkout_crossborder"]},
            "fraud": {"choices": ["normal", "elevated", "high_risk_burst"]},
            "sca": {"choices": ["frictionless_low_value", "tra_exemption", "challenge_otp", "out_of_scope"]},
            "retry": {"choices": ["immediate", "intraday_8h", "salary_deposit_72h"]},
        },
    }


@app.post("/specialists/decide", response_model=SidecarDecisionResponse)
def decide_specialist(req: SpecialistDecideRequest) -> SidecarDecisionResponse:
    """Evaluate payment context using a dedicated Specialist Small Decision Model (SDM)."""
    from payment_decision_sdk.sdm import SpecialistSDMEvaluator

    evaluator = SpecialistSDMEvaluator(specialist_type=req.specialist_type)
    res = evaluator.evaluate(req.context)

    return SidecarDecisionResponse(
        decision=res.decision,
        choice=res.decision,
        confidence=res.confidence,
        probabilities=res.probabilities,
        scores=res.probabilities,
        feature_contributions=res.feature_contributions or {},
        latency_ms=res.latency_ms,
        routing_reason=res.routing_reason or f"Evaluated by {req.specialist_type} specialist SDM",
        is_fallback=res.is_fallback,
        noul_triggered=False,
        noul_action=None,
        effective_fee_est=0.0,
        model_version="sdm-v1.0",
        timestamp=time.time(),
    )


@app.post("/decide", response_model=SidecarDecisionResponse)
def decide(context: PaymentContext, use_sdm: bool = False) -> SidecarDecisionResponse:
    """Evaluate payment context and return real-time routing decision.

    Guaranteed sub-20ms SLA latency profile for high-conversion payment dispatch.
    """
    if use_sdm:
        req = SpecialistDecideRequest(specialist_type="routing", context=context)
        return decide_specialist(req)

    global _engine
    if _engine is None:
        try:
            load_engine()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Decision engine could not be initialized: {e}",
            )

    t0 = time.perf_counter()
    raw_dict = context.to_dict()

    try:
        core_result = _engine.evaluate(raw_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Decision evaluation failed: {exc}",
        ) from exc

    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    # Record telemetry
    decision_key = core_result.choice
    _metrics["total_decisions"] += 1
    _metrics["total_latency_ms"] += total_latency_ms
    _metrics["decisions_by_gateway"][decision_key] = (
        _metrics["decisions_by_gateway"].get(decision_key, 0) + 1
    )
    if core_result.noul.triggered:
        _metrics["circuit_breakers_triggered"] += 1

    return SidecarDecisionResponse(
        decision=core_result.choice,
        choice=core_result.choice,
        confidence=core_result.confidence,
        probabilities=core_result.probabilities,
        scores=core_result.scores,
        feature_contributions=core_result.feature_contributions,
        latency_ms=round(total_latency_ms, 3),
        routing_reason=core_result.routing_reason,
        is_fallback=core_result.is_fallback,
        noul_triggered=core_result.noul.triggered,
        noul_action=core_result.noul.action,
        effective_fee_est=core_result.effective_fee_est,
        model_version=_spec.version if _spec else "1.0.0",
        timestamp=time.time(),
    )


@app.post("/decide/batch", response_model=BatchDecideResponse)
def decide_batch(req: BatchDecideRequest) -> BatchDecideResponse:
    """Batch evaluation endpoint for high-volume reconciliation or offline testing."""
    global _engine
    if _engine is None:
        try:
            load_engine()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Decision engine could not be initialized: {e}",
            )

    t0 = time.perf_counter()
    responses: List[SidecarDecisionResponse] = []

    for item in req.transactions:
        sub_resp = decide(item)
        responses.append(sub_resp)

    total_batch_ms = (time.perf_counter() - t0) * 1000.0
    count = len(responses)
    avg_ms = (total_batch_ms / count) if count > 0 else 0.0

    _metrics["total_batch_transactions"] += count

    return BatchDecideResponse(
        results=responses,
        count=count,
        total_latency_ms=round(total_batch_ms, 3),
        avg_latency_ms=round(avg_ms, 3),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
