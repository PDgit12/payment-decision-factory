"""
Base Agent Interface and Decision Result Models.
===============================================
Defines standard interfaces for specialized payment decision agents
(Fraud Screener, Dynamic 3DS, Smart Retry, Smart Routing), ensuring
seamless integration with the DAG orchestrator, core decision engine,
and backing by trained Specialist Small Decision Models (SDMs).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field

# Pre-load stdlib code module to prevent shadowing from root workspace
stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(stdlib_code) and ("code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")):
    spec_code = importlib.util.spec_from_file_location("code", stdlib_code)
    if spec_code and spec_code.loader:
        m = importlib.util.module_from_spec(spec_code)
        sys.modules["code"] = m
        spec_code.loader.exec_module(m)

# Try importing PaymentAgent from orchestration.agent_base for full typing and inheritance
try:
    from payment_decision_factory.orchestration.agent_base import PaymentAgent
    from payment_decision_factory.orchestration.context import (
        AgentExecutionResult,
        ExecutionStatus,
        TransactionContext,
    )
except ImportError:
    class PaymentAgent(ABC):  # type: ignore
        name: str
        sla_ms: float = 20.0
        confidence_floor: float = 0.40
        fallback_handler: Optional[Any] = None

        @abstractmethod
        async def decide(self, context: Any) -> Any:
            raise NotImplementedError

    TransactionContext = Any  # type: ignore
    AgentExecutionResult = Any  # type: ignore
    ExecutionStatus = Any  # type: ignore

from payment_decision_factory.core.spec import DecisionSpec


class AgentDecisionResult(BaseModel):
    """
    Standardized, calibrated decision output returned by specialized payment agents.
    Exposes unified attributes compatible with Core DecisionEngine, Orchestrator, and SDKs.
    """
    decision_id: str = Field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:12]}")
    spec_id: str = "unknown"
    agent_name: str = "agent"
    choice: Optional[str] = None
    score: Optional[float] = None
    confidence: float = 1.0
    probabilities: Dict[str, float] = Field(default_factory=dict)
    scores: Dict[str, float] = Field(default_factory=dict)
    feature_contributions: Dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    routing_reason: Optional[str] = None
    is_fallback: bool = False
    status: str = "SUCCESS"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    model_config = {
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
    }

    @property
    def decision(self) -> Optional[str]:
        """Alias for choice to support cross-SDK compatibility."""
        return self.choice

    def __getitem__(self, item: str) -> Any:
        """Support dict-style subscripting (e.g. result['choice'])."""
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.metadata:
            return self.metadata[item]
        raise KeyError(f"'{item}' not found in AgentDecisionResult")

    def get(self, item: str, default: Any = None) -> Any:
        """Safe dict-style get method."""
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision result to JSON-serializable dictionary."""
        d = self.model_dump()
        d["decision"] = self.choice
        return d


class BaseDecisionAgent(PaymentAgent):
    """
    Abstract base class for all specialized payment decision agents.

    Supports:
    - Declarative spec loading (YAML or DecisionSpec object).
    - Asynchronous evaluation (`decide`) under DAG orchestrator SLA budgeting.
    - Synchronous evaluation (`decide_sync` / `evaluate`) for unit tests and local scripts.
    - Transparent feature extraction from `TransactionContext` or raw dictionaries.
    - Seamless backing by deep, compact Specialist Small Decision Models (SDMs) or ONNX runtimes.
    """

    name: str = "base_agent"
    sla_ms: float = 20.0
    confidence_floor: float = 0.40
    fallback_handler: Optional[Any] = None

    def __init__(
        self,
        name: str,
        spec: Optional[Union[str, DecisionSpec]] = None,
        sla_ms: Optional[float] = None,
        confidence_floor: Optional[float] = None,
        fallback_handler: Optional[Any] = None,
        sdm: Optional[Any] = None,
        sdm_pipeline: Optional[Any] = None,
    ):
        self.name = name
        self.spec = self._load_spec(spec) if spec else None
        if self.spec:
            self.sla_ms = sla_ms or self.spec.sla.target_latency_ms or 20.0
            self.confidence_floor = (
                confidence_floor or self.spec.primitives.score.confidence_floor or 0.40
            )
        else:
            self.sla_ms = sla_ms or 20.0
            self.confidence_floor = confidence_floor or 0.40

        self.fallback_handler = fallback_handler
        self.sdm = sdm
        self.sdm_pipeline = sdm_pipeline

    def _load_spec(self, spec_input: Union[str, DecisionSpec]) -> Optional[DecisionSpec]:
        """Load and validate DecisionSpec from object, path, or known specs dir."""
        if isinstance(spec_input, DecisionSpec):
            return spec_input

        if isinstance(spec_input, str):
            # Check if direct file path exists
            if os.path.exists(spec_input):
                return DecisionSpec.from_yaml(spec_input)

            # Check inside payment_decision_factory/specs/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(base_dir, "specs", spec_input)
            if not candidate.endswith(".yaml") and not candidate.endswith(".yml"):
                candidate += ".yaml"
            if os.path.exists(candidate):
                return DecisionSpec.from_yaml(candidate)

        return None

    @property
    def is_sdm_backed(self) -> bool:
        """Returns True if this agent is backed by a trained Specialist SDM or ONNX model."""
        return self.sdm is not None

    def set_sdm(self, sdm: Any, pipeline: Optional[Any] = None) -> BaseDecisionAgent:
        """
        Attach a trained SpecialistSDM or ONNX inference session to this agent.
        Enables sub-0.2ms deep decision model inference.
        """
        self.sdm = sdm
        if pipeline is not None:
            self.sdm_pipeline = pipeline
        elif self.spec is not None:
            try:
                from payment_decision_factory.core.sdm import SpecialistPipeline
                self.sdm_pipeline = SpecialistPipeline(self.spec, specialist_type=self.name)
            except Exception:
                self.sdm_pipeline = None
        return self

    def load_sdm(
        self,
        model_path: str,
        license_key: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> BaseDecisionAgent:
        """Load trained Specialist SDM weights (.pt / .pth), ONNX artifact (.onnx), or sealed package (.sdmpkg)."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Specialist SDM artifact not found at: {model_path}")

        if model_path.endswith(".sdmpkg"):
            from payment_decision_factory.core.sealed_package import SealedPackage
            package = SealedPackage.load_file(model_path)
            key = license_key or os.environ.get("SDM_LICENSE_KEY", "default_license_key")
            decrypted = package.decrypt(license_key=key)
            runner = decrypted.create_runner()
            self.set_sdm(runner)
        elif model_path.endswith(".onnx"):
            try:
                from payment_decision_factory.core.sdm import ONNXSpecialistSDM
                choice_keys = list(self.spec.choices.keys()) if self.spec else []
                sdm = ONNXSpecialistSDM(model_path, choice_keys=choice_keys)
                self.set_sdm(sdm)
            except Exception:
                import onnxruntime as ort
                session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self.set_sdm(session)
        else:
            import torch
            from payment_decision_factory.core.sdm import SpecialistSDM
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, (SpecialistSDM, torch.nn.Module)):
                self.set_sdm(checkpoint)
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                if self.sdm and isinstance(self.sdm, torch.nn.Module):
                    self.sdm.load_state_dict(checkpoint["model_state_dict"])
                    self.sdm.eval()
            elif isinstance(checkpoint, dict):
                if self.sdm and isinstance(self.sdm, torch.nn.Module):
                    self.sdm.load_state_dict(checkpoint)
                    self.sdm.eval()

        return self

    def evaluate_with_sdm(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute in-flight inference using the attached Specialist SDM or ONNX session.
        Returns choice, calibrated confidence, probabilities distribution, and latency.
        """
        if self.sdm is None:
            return None

        t0 = time.perf_counter()

        # 1. Resolve vector representation from inputs
        if isinstance(inputs, (list, tuple)):
            vec = list(inputs)
        elif hasattr(inputs, "to_list"):
            vec = inputs.to_list()
        elif hasattr(inputs, "ndim"):
            vec = inputs.tolist()
        elif self.sdm_pipeline is not None and self.spec is not None and hasattr(self.spec, "validate_input"):
            valid = self.spec.validate_input(inputs)
            vec = self.sdm_pipeline.serializer.to_vector(valid)
        elif self.spec is not None and hasattr(self.spec, "validate_input"):
            from payment_decision_factory.core.serializer import PaymentSerializer
            valid = self.spec.validate_input(inputs)
            vec = PaymentSerializer(self.spec).to_vector(valid)
        elif hasattr(self, "encoder") and self.encoder is not None and hasattr(self.encoder, "transform_record"):
            vec = self.encoder.transform_record(inputs)
        elif hasattr(self.spec, "feature_schemas") and hasattr(self.spec, "numerical_features"):
            from payment_decision_factory.core.universal_factory import UniversalFeatureEncoder
            enc = UniversalFeatureEncoder(self.spec)
            vec = enc.transform_record(inputs)
        elif "amount" in inputs and ("currency" in inputs or "card_brand" in inputs):
            try:
                from payment_decision_factory.core.serializer import PaymentSerializer
                from payment_decision_factory.core.spec import DecisionSpec
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                spec_path = os.path.join(base_dir, "specs", "smart_routing.yaml")
                fallback_spec = DecisionSpec.from_yaml(spec_path)
                valid = fallback_spec.validate_input(inputs)
                vec = PaymentSerializer(fallback_spec).to_vector(valid)
            except Exception:
                input_dim = getattr(self.sdm, "input_dim", 30)
                vec = [float(v) for v in inputs.values() if isinstance(v, (int, float))]
                if len(vec) < input_dim:
                    vec.extend([0.0] * (input_dim - len(vec)))
                else:
                    vec = vec[:input_dim]
        else:
            input_dim = getattr(self.sdm, "input_dim", 30)
            vec = [float(v) for v in inputs.values() if isinstance(v, (int, float))]
            if len(vec) < input_dim:
                vec.extend([0.0] * (input_dim - len(vec)))
            else:
                vec = vec[:input_dim]

        # 2. Run model inference (ONNXSpecialistSDM, PyTorch SpecialistSDM, or ONNX InferenceSession)
        if hasattr(self.sdm, "predict"):
            import numpy as np
            _, probs = self.sdm.predict(vec)
            probs_np = probs[0] if probs.ndim == 2 else probs

        elif hasattr(self.sdm, "__call__") and hasattr(self.sdm, "eval"):
            import numpy as np
            import torch

            self.sdm.eval()
            x_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = self.sdm(x_tensor)
                if isinstance(out, (tuple, list)):
                    probs = out[1] if len(out) > 1 else out[0]
                elif isinstance(out, dict):
                    probs = out.get("probs", out.get("probabilities", out.get("logits")))
                else:
                    probs = out

                if isinstance(probs, torch.Tensor):
                    probs_np = probs.squeeze(0).cpu().numpy()
                else:
                    probs_np = np.asarray(probs)

        elif hasattr(self.sdm, "run"):
            import numpy as np
            x_array = np.array([vec], dtype=np.float32)
            input_name = self.sdm.get_inputs()[0].name
            outputs = self.sdm.run(None, {input_name: x_array})
            if len(outputs) > 1:
                probs_np = outputs[1][0] if outputs[1].ndim == 2 else outputs[1]
            else:
                raw_logits = outputs[0][0]
                shift = raw_logits - np.max(raw_logits)
                exps = np.exp(shift)
                probs_np = exps / np.sum(exps)

        else:
            return None

        # 3. Resolve choice mapping
        idx_to_choice: Dict[int, str] = {}
        if hasattr(self.sdm, "idx_to_choice") and self.sdm.idx_to_choice:
            idx_to_choice = self.sdm.idx_to_choice
        elif self.sdm_pipeline is not None and hasattr(self.sdm_pipeline, "idx_to_choice"):
            idx_to_choice = self.sdm_pipeline.idx_to_choice
        elif hasattr(self, "explicit_choices") and self.explicit_choices:
            idx_to_choice = {i: c for i, c in enumerate(self.explicit_choices)}
        elif self.spec is not None and hasattr(self.spec, "choices"):
            if isinstance(self.spec.choices, dict):
                idx_to_choice = {i: c for i, c in enumerate(self.spec.choices.keys())}
            elif isinstance(self.spec.choices, (list, tuple)):
                idx_to_choice = {i: c for i, c in enumerate(self.spec.choices)}
            else:
                idx_to_choice = {i: f"choice_{i}" for i in range(len(probs_np))}
        else:
            idx_to_choice = {i: f"choice_{i}" for i in range(len(probs_np))}

        idx = int(np.argmax(probs_np))
        choice = idx_to_choice.get(idx, f"choice_{idx}")
        conf = float(probs_np[idx])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob_dict = {
            idx_to_choice.get(i, f"choice_{i}"): float(round(probs_np[i], 4))
            for i in range(len(probs_np))
        }

        return {
            "choice": choice,
            "confidence": round(conf, 4),
            "probabilities": prob_dict,
            "latency_ms": round(latency_ms, 3),
            "raw_probs": probs_np,
        }

    def _extract_inputs(self, context: Union[TransactionContext, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract normalized key-value inputs from TransactionContext or dict.
        Resolves upstream agent outputs (e.g., fraud_score, sca_action).
        """
        if hasattr(context, "get") and hasattr(context, "inputs"):
            # It is a TransactionContext instance
            inputs = dict(context.inputs)
            # Resolve common upstream aliases if not explicitly present
            if "fraud_score" not in inputs:
                fraud_score = context.get("fraud_score")
                if fraud_score is not None:
                    inputs["fraud_score"] = float(fraud_score)
            if "three_ds_choice" not in inputs:
                sca_choice = context.get("three_ds_choice")
                if sca_choice is not None:
                    inputs["three_ds_choice"] = str(sca_choice)
            return inputs

        elif isinstance(context, dict):
            return dict(context)

        return {}

    async def decide(self, context: Union[TransactionContext, Dict[str, Any]]) -> AgentDecisionResult:
        """
        Asynchronously execute agent decision logic on current context.
        Enforces local latency tracking and standardized result schema.
        """
        start_perf = time.perf_counter()
        inputs = self._extract_inputs(context)

        result = self._compute_decision(inputs)

        latency_ms = (time.perf_counter() - start_perf) * 1000.0
        result.agent_name = self.name
        result.latency_ms = round(latency_ms, 3)
        if self.spec:
            result.spec_id = self.spec.spec_id

        return result

    def decide_sync(self, context: Union[TransactionContext, Dict[str, Any]]) -> AgentDecisionResult:
        """Synchronous wrapper for offline evaluation and unit tests."""
        start_perf = time.perf_counter()
        inputs = self._extract_inputs(context)
        result = self._compute_decision(inputs)
        latency_ms = (time.perf_counter() - start_perf) * 1000.0
        result.agent_name = self.name
        result.latency_ms = round(latency_ms, 3)
        if self.spec:
            result.spec_id = self.spec.spec_id
        return result

    def evaluate(self, payload: Dict[str, Any]) -> AgentDecisionResult:
        """Alias for `decide_sync` matching Core Engine runtime signature."""
        return self.decide_sync(payload)

    @abstractmethod
    def _compute_decision(self, inputs: Dict[str, Any]) -> AgentDecisionResult:
        """Internal domain decision calculation implemented by each agent subclass."""
        raise NotImplementedError
