"""
Pre-Authorization Fraud & Velocity Decision Agent.
==================================================
Evaluates transaction velocity, geo-IP divergence, proxy/VPN indicators,
and card verification results to generate calibrated pre-auth fraud risk scores
and discrete decisions (ALLOW, REVIEW, DECLINE).
Supports backing by deep, compact Fraud Specialist Small Decision Models (SDMs).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

from .base import AgentDecisionResult, BaseDecisionAgent


def _sigmoid(x: float) -> float:
    """Standard numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _logit(p: float) -> float:
    """Log-odds logit transform."""
    p_clamped = max(0.001, min(0.999, p))
    return math.log(p_clamped / (1.0 - p_clamped))


class FraudScreenerAgent(BaseDecisionAgent):
    """
    Specialized pre-auth risk and velocity decision agent.

    Primitives:
    - `noul`: Instant circuit breaker on rapid card testing or CVV mismatch.
    - `score`: Calibrated fraud probability score using temperature scaling or deep SDM.
    - `choice`: Discrete action (`allow`, `review`, `decline`).
    """

    def __init__(
        self,
        name: str = "fraud_screener",
        spec: Optional[str] = "fraud_screener.yaml",
        sla_ms: float = 15.0,
        confidence_floor: float = 0.40,
        fallback_handler: Optional[Any] = None,
        sdm: Optional[Any] = None,
        sdm_pipeline: Optional[Any] = None,
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

    def _compute_decision(self, inputs: Dict[str, Any]) -> AgentDecisionResult:
        # Extract features with safe defaults
        amount = float(inputs.get("amount", 50.0))
        currency = str(inputs.get("currency", "USD")).upper()
        bin_country = str(inputs.get("bin_country", "US")).upper()
        ip_country = str(inputs.get("ip_country", bin_country)).upper()
        velocity_1h = int(inputs.get("velocity_1h", 1))
        velocity_24h = int(inputs.get("velocity_24h", velocity_1h))
        is_anonymous_ip = bool(inputs.get("is_anonymous_ip", False))
        cvv_result = str(inputs.get("cvv_result", "match")).lower()
        avs_code = str(inputs.get("avs_code", "match")).lower()
        email_age_days = int(inputs.get("email_domain_age_days", 365))

        feature_contributions: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # 1. NOUL / CIRCUIT BREAKER RULES (Hard boundary violations always enforced)
        # ---------------------------------------------------------------------
        # Rule A: Extreme 1h velocity (active bot/card testing attack)
        if velocity_1h >= 12:
            return AgentDecisionResult(
                agent_name=self.name,
                choice="decline",
                score=0.98,
                confidence=0.99,
                probabilities={"allow": 0.005, "review": 0.015, "decline": 0.98},
                scores={"risk_score": 0.98, "raw_risk": 0.95},
                feature_contributions={"velocity_1h": 0.85, "attack_burst": 0.15},
                routing_reason="Circuit breaker triggered: card attempt velocity exceeds safe limit (>=12/hr).",
                is_fallback=False,
                status="SUCCESS",
                metadata={"rule_triggered": "card_testing_velocity_burst", "risk_tier": "critical"},
            )

        # Rule B: Hard CVV mismatch
        if cvv_result == "no_match":
            return AgentDecisionResult(
                agent_name=self.name,
                choice="decline",
                score=0.95,
                confidence=0.98,
                probabilities={"allow": 0.01, "review": 0.04, "decline": 0.95},
                scores={"risk_score": 0.95, "raw_risk": 0.90},
                feature_contributions={"cvv_result": 0.80, "amount": 0.10},
                routing_reason="Circuit breaker triggered: CVV code verification failed hard mismatch.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"rule_triggered": "cvv_hard_mismatch", "risk_tier": "critical"},
            )

        # Rule C: Anonymous IP + High ticket + International BIN mismatch
        if is_anonymous_ip and amount >= 5000.0 and bin_country != ip_country:
            return AgentDecisionResult(
                agent_name=self.name,
                choice="decline",
                score=0.92,
                confidence=0.95,
                probabilities={"allow": 0.02, "review": 0.06, "decline": 0.92},
                scores={"risk_score": 0.92, "raw_risk": 0.88},
                feature_contributions={"is_anonymous_ip": 0.35, "ip_bin_mismatch": 0.35, "high_amount": 0.25},
                routing_reason="Circuit breaker triggered: High ticket order originating from proxy with cross-border BIN discrepancy.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"rule_triggered": "high_ticket_anonymous_proxy", "risk_tier": "high"},
            )

        # ---------------------------------------------------------------------
        # 2. SPECIALIST SDM INFERENCE (If Backed by Trained Deep Model)
        # ---------------------------------------------------------------------
        if self.is_sdm_backed:
            sdm_out = self.evaluate_with_sdm(inputs)
            if sdm_out is not None:
                choice = sdm_out["choice"]
                conf = sdm_out["confidence"]
                probs = sdm_out["probabilities"]

                # Calibrated fraud risk score from model distribution
                decline_prob = probs.get("decline", probs.get("fraud", probs.get("class_1", 0.0)))
                review_prob = probs.get("review", 0.0)
                risk_score = round(float(decline_prob) + 0.5 * float(review_prob), 4)

                is_low_risk = choice in ("allow", "clean", "class_0")
                return AgentDecisionResult(
                    agent_name=self.name,
                    choice=choice,
                    score=risk_score,
                    confidence=conf,
                    probabilities=probs,
                    scores={"risk_score": risk_score},
                    feature_contributions={"sdm_latent_inference": 1.0},
                    routing_reason=f"Evaluated by Fraud Specialist SDM (action={choice}, conf={conf:.1%}, risk={risk_score:.3f})",
                    is_fallback=False,
                    status="SUCCESS",
                    metadata={
                        "backed_by_sdm": True,
                        "sdm_type": "fraud",
                        "risk_tier": "low" if is_low_risk else ("medium" if choice == "review" else "high"),
                        "is_sealed": getattr(self.sdm, "package_id", None) is not None,
                        "package_id": getattr(self.sdm, "package_id", None),
                    },
                )

        # ---------------------------------------------------------------------
        # 3. CONTINUOUS HEURISTIC RISK SCORING & FEATURE ATTRIBUTION (Fallback)
        # ---------------------------------------------------------------------
        risk_components: List[float] = []

        # Velocity contribution (0 to ~0.35)
        vel_weight = min(velocity_1h / 8.0, 1.0) * 0.35
        feature_contributions["velocity_1h"] = round(vel_weight, 3)
        risk_components.append(vel_weight)

        # IP vs BIN Country discrepancy (0.22 if mismatched)
        geo_mismatch = (bin_country != ip_country)
        geo_weight = 0.22 if geo_mismatch else 0.0
        feature_contributions["ip_bin_mismatch"] = round(geo_weight, 3)
        risk_components.append(geo_weight)

        # Anonymous IP (VPN/Proxy/Tor) (0.18)
        anon_weight = 0.18 if is_anonymous_ip else 0.0
        feature_contributions["is_anonymous_ip"] = round(anon_weight, 3)
        risk_components.append(anon_weight)

        # CVV / AVS check
        cvv_weight = 0.10 if cvv_result == "not_provided" else 0.0
        avs_weight = 0.12 if avs_code == "no_match" else (0.05 if avs_code == "partial_match" else 0.0)
        feature_contributions["verification_checks"] = round(cvv_weight + avs_weight, 3)
        risk_components.extend([cvv_weight, avs_weight])

        # High ticket outlier
        amount_weight = min(amount / 15000.0, 0.15)
        feature_contributions["amount_exposure"] = round(amount_weight, 3)
        risk_components.append(amount_weight)

        # Email domain age
        domain_weight = 0.08 if email_age_days < 30 else 0.0
        feature_contributions["email_domain_risk"] = round(domain_weight, 3)
        risk_components.append(domain_weight)

        # Raw linear risk score
        base_intercept = 0.05
        raw_risk = min(0.99, max(0.01, base_intercept + sum(risk_components)))

        # ---------------------------------------------------------------------
        # 4. TEMPERATURE SCALING CALIBRATION
        # ---------------------------------------------------------------------
        temperature = 0.75
        if self.spec and hasattr(self.spec.primitives.score, "temperature"):
            temperature = self.spec.primitives.score.temperature

        raw_logit = _logit(raw_risk)
        calibrated_logit = raw_logit / temperature
        calibrated_risk = round(_sigmoid(calibrated_logit), 4)

        # ---------------------------------------------------------------------
        # 5. DISCRETE DECISION CHOICE & PROBABILITIES
        # ---------------------------------------------------------------------
        allow_threshold = 0.25
        decline_threshold = 0.70

        if calibrated_risk < allow_threshold:
            choice = "allow"
            confidence = round(min(0.99, 0.75 + (allow_threshold - calibrated_risk) * 1.0), 3)
            prob_allow = round(1.0 - calibrated_risk, 3)
            prob_review = round(calibrated_risk * 0.8, 3)
            prob_decline = round(max(0.001, 1.0 - prob_allow - prob_review), 3)
            reason = f"Transaction risk score ({calibrated_risk:.2f}) within safe pre-auth bounds."

        elif calibrated_risk >= decline_threshold:
            choice = "decline"
            confidence = round(min(0.99, 0.75 + (calibrated_risk - decline_threshold) * 0.8), 3)
            prob_decline = round(calibrated_risk, 3)
            prob_review = round((1.0 - calibrated_risk) * 0.8, 3)
            prob_allow = round(max(0.001, 1.0 - prob_decline - prob_review), 3)
            reason = f"Elevated fraud indicators detected; risk score ({calibrated_risk:.2f}) exceeds decline threshold."

        else:
            choice = "review"
            confidence = round(0.85, 3)
            prob_review = round(0.65, 3)
            prob_allow = round((1.0 - calibrated_risk) * 0.5, 3)
            prob_decline = round(calibrated_risk * 0.5, 3)
            reason = f"Moderate anomaly profile (risk score {calibrated_risk:.2f}); recommended for manual review or 3DS step-up."

        total_p = prob_allow + prob_review + prob_decline
        probabilities = {
            "allow": round(prob_allow / total_p, 4),
            "review": round(prob_review / total_p, 4),
            "decline": round(prob_decline / total_p, 4),
        }

        return AgentDecisionResult(
            agent_name=self.name,
            choice=choice,
            score=calibrated_risk,
            confidence=confidence,
            probabilities=probabilities,
            scores={"risk_score": calibrated_risk, "raw_risk": round(raw_risk, 4)},
            feature_contributions=feature_contributions,
            routing_reason=reason,
            is_fallback=False,
            status="SUCCESS",
            metadata={
                "backed_by_sdm": False,
                "risk_tier": "low" if choice == "allow" else ("medium" if choice == "review" else "high"),
                "geo_mismatch": geo_mismatch,
                "is_anonymous_ip": is_anonymous_ip,
                "velocity_1h": velocity_1h,
            },
        )
