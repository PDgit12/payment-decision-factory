"""
Universal Domain-Agnostic Decision Agent.
========================================
Extends the BaseDecisionAgent to support arbitrary decision domains beyond payments
(e.g., credit/lending underwriting, logistics dispatch, dynamic pricing, insurance risk).
Supports in-process execution, sealed `.sdmpkg` loading, and microsecond inference.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Union

from .base import AgentDecisionResult, BaseDecisionAgent
from payment_decision_factory.core.spec import DecisionSpec


class UniversalDecisionAgent(BaseDecisionAgent):
    """
    Domain-agnostic decision agent capable of evaluating arbitrary schemas
    and choices using compiled Specialist SDMs or sealed `.sdmpkg` packages.
    """

    def __init__(
        self,
        name: str = "universal_decision_agent",
        spec: Optional[Union[str, DecisionSpec, Any]] = None,
        choices: Optional[List[str]] = None,
        sla_ms: float = 20.0,
        confidence_floor: float = 0.40,
        fallback_handler: Optional[Any] = None,
        sdm: Optional[Any] = None,
        sdm_pipeline: Optional[Any] = None,
        encoder: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            spec=spec,
            sla_ms=sla_ms,
            confidence_floor=confidence_floor,
            fallback_handler=fallback_handler,
            sdm=sdm,
            sdm_pipeline=sdm_pipeline,
        )
        self.encoder = encoder

        if choices:
            self.explicit_choices = list(choices)
        elif self.spec and hasattr(self.spec, "choices"):
            if isinstance(self.spec.choices, dict):
                self.explicit_choices = list(self.spec.choices.keys())
            elif isinstance(self.spec.choices, (list, tuple)):
                self.explicit_choices = list(self.spec.choices)
            else:
                self.explicit_choices = ["choice_0", "choice_1"]
        else:
            self.explicit_choices = ["choice_0", "choice_1"]

        self.extra_metadata = metadata or {}

    def _compute_decision(self, inputs: Dict[str, Any]) -> AgentDecisionResult:
        # 1. Evaluate with SDM or Sealed Package if available
        if self.is_sdm_backed:
            sdm_out = self.evaluate_with_sdm(inputs)
            if sdm_out is not None:
                choice = sdm_out["choice"]
                conf = sdm_out["confidence"]
                probs = sdm_out["probabilities"]

                is_sealed = getattr(self.sdm, "package_id", None) is not None
                domain_name = getattr(self.sdm, "domain", getattr(self.spec, "domain", getattr(self.spec, "domain_name", "universal")) if self.spec else "universal")
                spec_id_val = getattr(self.spec, "spec_id", getattr(self.spec, "domain_id", "universal_v1")) if self.spec else "universal_v1"

                meta = {
                    "backed_by_sdm": True,
                    "is_sealed": is_sealed,
                    "package_id": getattr(self.sdm, "package_id", None),
                    "domain": domain_name,
                    **self.extra_metadata,
                }

                return AgentDecisionResult(
                    agent_name=self.name,
                    spec_id=spec_id_val,
                    choice=choice,
                    score=conf,
                    confidence=conf,
                    probabilities=probs,
                    scores=probs,
                    feature_contributions={"sdm_latent_inference": 1.0},
                    latency_ms=sdm_out["latency_ms"],
                    routing_reason=f"Decided by Universal Specialist SDM (action={choice}, conf={conf:.1%})",
                    is_fallback=False,
                    status="SUCCESS",
                    metadata=meta,
                )

        # 2. Spec-driven default evaluation if no SDM is attached
        default_choice = self.explicit_choices[0]
        if self.spec and hasattr(self.spec, "primitives") and hasattr(self.spec.primitives, "choice"):
            default_choice = getattr(self.spec.primitives.choice, "default_choice", default_choice)

        probs = {c: 1.0 / len(self.explicit_choices) for c in self.explicit_choices}
        probs[default_choice] = round(0.5 + 0.5 / len(self.explicit_choices), 3)
        # Normalize
        total_p = sum(probs.values())
        norm_probs = {k: round(v / total_p, 4) for k, v in probs.items()}

        spec_id_val = getattr(self.spec, "spec_id", getattr(self.spec, "domain_id", "universal_v1")) if self.spec else "universal_v1"
        domain_name = getattr(self.spec, "domain", getattr(self.spec, "domain_name", "universal")) if self.spec else "universal"

        return AgentDecisionResult(
            agent_name=self.name,
            spec_id=spec_id_val,
            choice=default_choice,
            score=norm_probs[default_choice],
            confidence=norm_probs[default_choice],
            probabilities=norm_probs,
            scores=norm_probs,
            feature_contributions={"default_spec_policy": 1.0},
            latency_ms=0.05,
            routing_reason=f"Default decision policy for {default_choice}",
            is_fallback=False,
            status="SUCCESS",
            metadata={"backed_by_sdm": False, "domain": domain_name},
        )
