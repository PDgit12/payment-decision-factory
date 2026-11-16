"""
Dynamic 3D-Secure & SCA Exemption Decision Agent.
================================================
Evaluates transaction parameters against European PSD2 Regulatory Technical Standards (RTS),
maximizing frictionless checkout through TRA (Article 18) and Low-Value (Article 16) exemptions
while enforcing mandatory step-up challenges on high risk or out-of-scope transactions.
Supports backing by deep, compact 3DS/SCA Specialist Small Decision Models (SDMs).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Union

from .base import AgentDecisionResult, BaseDecisionAgent

# EEA Member States + UK & Switzerland subject to SCA/PSD2 conventions
EEA_COUNTRIES: Set[str] = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI", "LT", "LU",
    "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "GB", "CH",
}

# Approximate FX rates to EUR for statutory PSD2 thresholds
EUR_RATES: Dict[str, float] = {
    "EUR": 1.0,
    "USD": 0.92,
    "GBP": 1.17,
    "CAD": 0.68,
    "AUD": 0.60,
    "JPY": 0.0062,
    "CHF": 1.05,
    "SEK": 0.088,
    "NOK": 0.085,
    "SGD": 0.69,
}


class Dynamic3DSAgent(BaseDecisionAgent):
    """
    Specialized SCA/PSD2 exemption decision agent.

    Primitives:
    - `choice`: Selected 3DS action (`tra_exemption`, `low_value`, `secure_corporate`, `challenge`).
    - `score`: Predicted issuer exemption acceptance probability (0.0 - 1.0).
    - `noul`: PSD2 statutory cap boundaries (>€500 cap, high fraud cutoff).
    """

    def __init__(
        self,
        name: str = "dynamic_3ds",
        spec: Optional[str] = "dynamic_3ds.yaml",
        sla_ms: float = 15.0,
        confidence_floor: float = 0.45,
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

    def _convert_to_eur(self, amount: float, currency: str, explicit_eur: Optional[float] = None) -> float:
        """Calculate amount in EUR for PSD2 RTS compliance."""
        if explicit_eur is not None and explicit_eur > 0:
            return float(explicit_eur)
        rate = EUR_RATES.get(currency.upper(), 0.92)
        return round(amount * rate, 2)

    def _compute_decision(self, inputs: Dict[str, Any]) -> AgentDecisionResult:
        amount = float(inputs.get("amount", 49.99))
        currency = str(inputs.get("currency", "EUR")).upper()
        explicit_eur = inputs.get("amount_eur")
        amount_eur = self._convert_to_eur(amount, currency, explicit_eur)

        bin_country = str(inputs.get("bin_country", "IE")).upper()
        merchant_country = str(inputs.get("merchant_country", "IE")).upper()
        card_brand = str(inputs.get("card_brand", "visa")).lower()

        fraud_score = float(inputs.get("fraud_score", 0.10))
        acquirer_tra_rate = float(inputs.get("acquirer_tra_rate", 0.0004))
        is_recurring = bool(inputs.get("is_recurring", False))
        is_corporate = bool(inputs.get("is_corporate", False))
        delegated_auth = bool(inputs.get("delegated_auth_supported", False))

        feature_contributions: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # 1. NOUL / STATUTORY BOUNDARIES (PSD2 Law Always Enforced)
        # ---------------------------------------------------------------------
        # Boundary A: Statutory TRA Hard Cap (> €500)
        if amount_eur > 500.0:
            return AgentDecisionResult(
                agent_name=self.name,
                choice="challenge",
                score=0.88,
                confidence=0.98,
                probabilities={"tra_exemption": 0.01, "low_value": 0.0, "secure_corporate": 0.04, "challenge": 0.95},
                scores={"issuer_acceptance_prob": 0.88, "amount_eur": amount_eur},
                feature_contributions={"amount_eur": 0.90, "statutory_cap": 0.10},
                routing_reason="Statutory boundary: PSD2 RTS Article 18 strictly caps TRA exemptions at €500. Step-up mandatory.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"article": "PSD2 Article 97", "amount_eur": amount_eur, "exemption": "NONE"},
            )

        # Boundary B: High fraud score invalidates exemption criteria
        if fraud_score >= 0.40:
            return AgentDecisionResult(
                agent_name=self.name,
                choice="challenge",
                score=0.82,
                confidence=0.95,
                probabilities={"tra_exemption": 0.04, "low_value": 0.01, "secure_corporate": 0.05, "challenge": 0.90},
                scores={"issuer_acceptance_prob": 0.82, "fraud_score": fraud_score},
                feature_contributions={"fraud_score": 0.85, "amount_eur": 0.15},
                routing_reason=f"Elevated fraud score ({fraud_score:.2f}) invalidates low-risk exemption criteria; challenge required.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"article": "PSD2 Step-Up", "fraud_score": fraud_score, "exemption": "NONE"},
            )

        # Boundary C: One-Leg-Out (Issuer or Acquirer outside EEA)
        issuer_in_eea = bin_country in EEA_COUNTRIES
        merchant_in_eea = merchant_country in EEA_COUNTRIES
        if not (issuer_in_eea and merchant_in_eea):
            feature_contributions["one_leg_out_scope"] = 0.80
            return AgentDecisionResult(
                agent_name=self.name,
                choice="tra_exemption",
                score=0.96,
                confidence=0.94,
                probabilities={"tra_exemption": 0.92, "low_value": 0.02, "secure_corporate": 0.02, "challenge": 0.04},
                scores={"issuer_acceptance_prob": 0.96},
                feature_contributions=feature_contributions,
                routing_reason="Transaction is One-Leg-Out (out of EEA regulatory scope); SCA challenge not mandated.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"scope": "one_leg_out", "issuer_eea": issuer_in_eea, "merchant_eea": merchant_in_eea},
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
                acceptance_prob = round(1.0 - probs.get("challenge", 0.10), 4)

                return AgentDecisionResult(
                    agent_name=self.name,
                    choice=choice,
                    score=acceptance_prob,
                    confidence=conf,
                    probabilities=probs,
                    scores={"issuer_acceptance_prob": acceptance_prob, "amount_eur": amount_eur},
                    feature_contributions={"sdm_latent_inference": 1.0},
                    routing_reason=f"Evaluated by 3DS/SCA Specialist SDM (action={choice}, conf={conf:.1%})",
                    is_fallback=False,
                    status="SUCCESS",
                    metadata={
                        "backed_by_sdm": True,
                        "sdm_type": "sca",
                        "article": "PSD2 Specialist SDM",
                        "frictionless": choice != "challenge",
                        "is_sealed": getattr(self.sdm, "package_id", None) is not None,
                        "package_id": getattr(self.sdm, "package_id", None),
                    },
                )

        # ---------------------------------------------------------------------
        # 3. LOW VALUE EXEMPTION (PSD2 RTS Article 16)
        # ---------------------------------------------------------------------
        if amount_eur <= 30.0 and fraud_score < 0.20:
            feature_contributions["amount_tier"] = 0.60
            feature_contributions["fraud_risk"] = 0.40
            return AgentDecisionResult(
                agent_name=self.name,
                choice="low_value",
                score=0.98,
                confidence=0.96,
                probabilities={"tra_exemption": 0.08, "low_value": 0.88, "secure_corporate": 0.01, "challenge": 0.03},
                scores={"issuer_acceptance_prob": 0.98},
                feature_contributions=feature_contributions,
                routing_reason=f"Qualifies for PSD2 Article 16 Low-Value exemption (€{amount_eur:.2f} <= €30.00).",
                is_fallback=False,
                status="SUCCESS",
                metadata={"article": "PSD2 RTS Article 16", "exemption_type": "LOW_VALUE", "frictionless": True},
            )

        # ---------------------------------------------------------------------
        # 4. SECURE CORPORATE PAYMENT EXEMPTION (PSD2 RTS Article 17)
        # ---------------------------------------------------------------------
        if is_corporate:
            feature_contributions["corporate_card_rail"] = 0.85
            return AgentDecisionResult(
                agent_name=self.name,
                choice="secure_corporate",
                score=0.95,
                confidence=0.92,
                probabilities={"tra_exemption": 0.05, "low_value": 0.01, "secure_corporate": 0.90, "challenge": 0.04},
                scores={"issuer_acceptance_prob": 0.95},
                feature_contributions=feature_contributions,
                routing_reason="Qualifies for PSD2 RTS Article 17 Secure Corporate Payment exemption.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"article": "PSD2 RTS Article 17", "exemption_type": "CORPORATE", "frictionless": True},
            )

        # ---------------------------------------------------------------------
        # 5. TRANSACTION RISK ANALYSIS (TRA) EXEMPTION (PSD2 RTS Article 18)
        # ---------------------------------------------------------------------
        tra_eligible = False
        tier_label = "none"

        if amount_eur <= 100.0:
            if acquirer_tra_rate <= 0.0013 and fraud_score < 0.25:
                tra_eligible = True
                tier_label = "tier_1_sub_100"
        elif amount_eur <= 250.0:
            if acquirer_tra_rate <= 0.0006 and fraud_score < 0.20:
                tra_eligible = True
                tier_label = "tier_2_sub_250"
        elif amount_eur <= 500.0:
            if acquirer_tra_rate <= 0.0001 and fraud_score < 0.15:
                tra_eligible = True
                tier_label = "tier_3_sub_500"

        if tra_eligible:
            feature_contributions["tra_tier"] = 0.45
            feature_contributions["acquirer_fraud_rate"] = 0.35
            feature_contributions["fraud_score"] = 0.20

            acceptance_prob = round(0.95 - (fraud_score * 0.20) - (acquirer_tra_rate * 50.0), 3)
            acceptance_prob = max(0.70, min(0.98, acceptance_prob))

            return AgentDecisionResult(
                agent_name=self.name,
                choice="tra_exemption",
                score=acceptance_prob,
                confidence=0.92,
                probabilities={
                    "tra_exemption": round(acceptance_prob * 0.90, 3),
                    "low_value": 0.02,
                    "secure_corporate": 0.02,
                    "challenge": round(1.0 - (acceptance_prob * 0.90) - 0.04, 3),
                },
                scores={"issuer_acceptance_prob": acceptance_prob, "amount_eur": amount_eur},
                feature_contributions=feature_contributions,
                routing_reason=f"Eligible for PSD2 RTS Article 18 TRA Exemption ({tier_label}, €{amount_eur:.2f}).",
                is_fallback=False,
                status="SUCCESS",
                metadata={
                    "backed_by_sdm": False,
                    "article": "PSD2 RTS Article 18",
                    "exemption_type": "TRA",
                    "tier": tier_label,
                    "frictionless": True,
                },
            )

        # ---------------------------------------------------------------------
        # 6. DEFAULT: MANDATORY STEP-UP CHALLENGE
        # ---------------------------------------------------------------------
        feature_contributions["exemption_ineligibility"] = 0.60
        feature_contributions["liability_shift_preservation"] = 0.40

        return AgentDecisionResult(
            agent_name=self.name,
            choice="challenge",
            score=0.88,
            confidence=0.90,
            probabilities={"tra_exemption": 0.08, "low_value": 0.02, "secure_corporate": 0.02, "challenge": 0.88},
            scores={"issuer_acceptance_prob": 0.88, "amount_eur": amount_eur},
            feature_contributions=feature_contributions,
            routing_reason="Transaction parameters exceed TRA thresholds; requiring consumer 3DS authentication for liability shift.",
            is_fallback=False,
            status="SUCCESS",
            metadata={"backed_by_sdm": False, "article": "PSD2 Article 97", "exemption_type": "NONE", "frictionless": False},
        )
