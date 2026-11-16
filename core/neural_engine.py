"""
Laya Neural Decision Engine (System 1 Pretrained Model Integration)
==================================================================
Hyper-personalized payment decision engine powered by Convai Innovations' Laya
(ModernBERT-large non-autoregressive decision model).

Key Capabilities:
1. Native Pretrained Weights: Integrates directly with `convaiinnovations/laya`
   and `convaiinnovations/laya` (subfolder="typed-decisions").
2. Dynamic Spec Mapping: Automatically translates declarative `spec.yaml`
   definitions into typed Laya question schemas (`choice`, `score`, `noul`).
3. Client Hyper-Personalization:
   - Client-specific acquirers, thresholds, and corridor routing rules.
   - Temperature calibration overlays (`calibration.json`) fitted to client settlement logs.
   - Graceful hardware adaptation (Apple Silicon MPS, NVIDIA CUDA, CPU, or ONNX backend).
4. Dual-Mode Fallback: If neural model weights are not preloaded, seamlessly
   falls back to the microsecond compiled heuristic vector engine.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Ensure root package is in sys.path
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.serializer import PaymentSerializer
from payment_decision_factory.core.engine import DecisionEngine, DecisionResult

logger = logging.getLogger("payment_decision_factory.neural_engine")


class LayaNeuralEngine:
    """Hyper-personalized System 1 Decision Engine backed by Laya pretrained weights."""

    def __init__(
        self,
        spec: Union[DecisionSpec, str],
        model_id: str = "convaiinnovations/laya",
        subfolder: Optional[str] = "typed-decisions",
        device: Optional[str] = None,
        lazy_load: bool = True,
        calibration_path: Optional[str] = None,
        fallback_to_heuristic: bool = True,
    ):
        """
        Initialize the neural decision engine.

        Args:
            spec: DecisionSpec instance or path to spec.yaml.
            model_id: Hugging Face model repository ID.
            subfolder: Subfolder within repo (e.g. 'typed-decisions' or None).
            device: 'mps', 'cuda', 'cpu', or None for auto-detection.
            lazy_load: Whether to defer downloading/loading weights until first inference.
            calibration_path: Optional path to client-specific temperature calibration JSON.
            fallback_to_heuristic: Fall back to compiled vector engine if weights unavailable.
        """
        if isinstance(spec, str):
            self.spec = DecisionSpec.from_yaml(spec)
        else:
            self.spec = spec

        self.model_id = model_id
        self.subfolder = subfolder
        self.device = device
        self.calibration_path = calibration_path
        self.fallback_to_heuristic = fallback_to_heuristic

        # Serializer for turning payment states into semantic context
        self.serializer = PaymentSerializer(self.spec)

        # Baseline heuristic engine for fallback / fast verification
        self._heuristic_engine = DecisionEngine(self.spec)

        # Neural agent handle
        self._agent = None
        self._is_loaded = False
        self._questions = self._build_laya_questions()

        if not lazy_load:
            self._load_agent()

    def _build_laya_questions(self) -> Dict[str, Dict[str, Any]]:
        """Translates declarative spec choices and rubrics into Laya question schema."""
        criteria: Dict[str, str] = {}
        if self.spec.choices:
            for key, choice in self.spec.choices.items():
                currencies = ",".join(choice.preferred_currencies) if choice.preferred_currencies else "All"
                countries = ",".join(choice.preferred_countries) if choice.preferred_countries else "All"
                desc = f"{choice.display_name} ({choice.provider}, {choice.region}). Preferred currencies: {currencies}. Preferred countries: {countries}."
                criteria[key] = desc
        else:
            criteria = {"allow": "Approve transaction", "decline": "Decline transaction"}

        instructions = (
            f"Evaluate the payment state and select the optimal target gateway for {self.spec.name}. "
            f"Optimize for authorization rate, lowest fee bps, and regional currency affinity."
        )

        return {
            "routing_choice": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        }

    def _load_agent(self) -> bool:
        """Loads the Laya Agent weights into memory."""
        if self._is_loaded and self._agent is not None:
            return True

        try:
            from laya.agent import Agent

            logger.info(
                f"Loading Laya neural weights from {self.model_id} (subfolder={self.subfolder}, device={self.device})..."
            )
            self._agent = Agent(
                model_id_or_path=self.model_id,
                subfolder=self.subfolder,
                device=self.device,
                calibration=self.calibration_path,
            )
            self._is_loaded = True
            logger.info("Laya neural decision model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(
                f"Could not load Laya neural weights ({e}). "
                f"{'Falling back to heuristic vector engine.' if self.fallback_to_heuristic else 'Raising error.'}"
            )
            if not self.fallback_to_heuristic:
                raise e
            self._agent = None
            self._is_loaded = False
            return False

    def decide(self, payment_payload: Dict[str, Any]) -> DecisionResult:
        """
        Evaluate payment transaction and return calibrated DecisionResult.

        Uses real Laya ModernBERT neural forward pass if loaded,
        or compiled fast engine if offline/fallback.
        """
        start_time = time.perf_counter()

        # Step 1: Validate input against spec schema
        validated_inputs = self.spec.validate_input(payment_payload)

        # Step 2: Serialize to semantic prompt
        serialized = self.serializer.serialize(validated_inputs)
        semantic_text = serialized.semantic_prompt

        # Step 3: Neural Inference via Laya (if loaded)
        if self._load_agent() and self._agent is not None:
            try:
                # Format state for Laya
                laya_state = {"text": semantic_text}
                
                # Single forward pass through ModernBERT decision head
                pred = self._agent.predict(laya_state, self._questions)
                
                res = pred.get("answers", {}).get("routing_choice", {})
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                choice = res.get("choice") or self.spec.primitives.choice.default_choice
                confidence = float(res.get("confidence", 0.85))
                probabilities = res.get("probabilities", {choice: confidence})

                # Feature contributions from semantic parser
                features = {
                    "corridor": serialized.corridor,
                    "amount_tier": serialized.amount_tier,
                    "neural_model": f"{self.model_id}:{self.subfolder}",
                }

                return DecisionResult(
                    choice=choice,
                    score=confidence,
                    confidence=confidence,
                    probabilities=probabilities,
                    feature_contributions=features,
                    latency_ms=round(latency_ms, 3),
                    is_fallback=False,
                    metadata={"engine": "laya_neural", "corridor": features["corridor"]},
                )
            except Exception as e:
                logger.error(f"Error during Laya neural forward pass: {e}. Falling back to heuristic.")

        # Step 4: Fallback to compiled vector engine
        return self._heuristic_engine.evaluate(validated_inputs)


def create_neural_engine(spec_path: str, subfolder: str = "typed-decisions") -> LayaNeuralEngine:
    """Convenience factory to instantiate a client-personalized Laya engine."""
    return LayaNeuralEngine(spec=spec_path, subfolder=subfolder, lazy_load=True)
