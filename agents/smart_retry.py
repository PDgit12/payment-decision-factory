"""
Intelligent Decline Recovery & Retry Timing Decision Agent.
============================================================
Evaluates payment decline reasons, error codes, attempt counts, and cardholder
billing cycles to determine optimal retry timing windows, backoff durations,
and scheme compliance guardrails (Mastercard/Visa excessive retry caps).
Supports backing by deep, compact Retry Specialist Small Decision Models (SDMs).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Union

from .base import AgentDecisionResult, BaseDecisionAgent

# Hard decline codes where card network rules strictly forbid retry attempts
HARD_DECLINES: Set[str] = {
    "stolen_card",
    "lost_card",
    "account_closed",
    "pickup_card",
    "fraudulent_transaction",
    "invalid_account_number",
}

# Technical transient error codes safe for immediate cascade reroute
TECHNICAL_ERRORS: Set[str] = {
    "processor_timeout",
    "system_error",
    "gateway_unavailable",
    "network_error",
    "bank_offline",
}

BACKOFF_HOURS_MAP: Dict[str, float] = {
    "retry_immediate": 0.0,
    "retry_short_backoff": 8.0,
    "retry_smart_window": 48.0,
    "retry_payday_window": 96.0,
    "terminal_no_retry": -1.0,
}


class SmartRetryAgent(BaseDecisionAgent):
    """
    Specialized decline recovery and retry timing decision agent.

    Primitives:
    - `choice`: Selected recovery action (`retry_immediate`, `retry_short_backoff`,
                `retry_smart_window`, `retry_payday_window`, `terminal_no_retry`).
    - `score`: Predicted optimal backoff hours and recovery probability.
    - `noul`: Card brand scheme compliance guardrails (zero tolerance on hard declines, max 4 attempts).
    """

    def __init__(
        self,
        name: str = "smart_retry",
        spec: Optional[str] = "smart_retry.yaml",
        sla_ms: float = 20.0,
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
        decline_code = str(inputs.get("decline_code", "generic_decline")).lower().strip()
        decline_category = str(inputs.get("decline_category", "soft_decline")).lower().strip()
        retry_attempt = int(inputs.get("retry_attempt", 1))
        amount = float(inputs.get("amount", 29.99))
        currency = str(inputs.get("currency", "USD")).upper()
        billing_cycle_day = int(inputs.get("billing_cycle_day", 1))
        day_of_week = int(inputs.get("day_of_week", 2))

        feature_contributions: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # 1. NOUL / SCHEME COMPLIANCE RULES (Visa / Mastercard Rules Always Enforced)
        # ---------------------------------------------------------------------
        # Rule A: Zero-tolerance hard decline check
        if decline_category == "hard_decline" or decline_code in HARD_DECLINES:
            feature_contributions["hard_decline_rule"] = 1.0
            return AgentDecisionResult(
                agent_name=self.name,
                choice="terminal_no_retry",
                score=-1.0,
                confidence=0.99,
                probabilities={
                    "retry_immediate": 0.0,
                    "retry_short_backoff": 0.0,
                    "retry_smart_window": 0.0,
                    "retry_payday_window": 0.0,
                    "terminal_no_retry": 1.0,
                },
                scores={"optimal_backoff_hours": -1.0, "recovery_probability": 0.0},
                feature_contributions=feature_contributions,
                routing_reason=f"Card scheme compliance violation: retrying hard decline '{decline_code}' is prohibited by network rules.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"scheme_action": "PERMANENT_CEASE", "backoff_hours": -1.0, "compliance_enforced": True},
            )

        # Rule B: Excessive retry boundary (Mastercard limit: max 3 retries in 30 days)
        if retry_attempt >= 4:
            feature_contributions["excessive_attempts_cap"] = 0.95
            return AgentDecisionResult(
                agent_name=self.name,
                choice="terminal_no_retry",
                score=-1.0,
                confidence=0.96,
                probabilities={
                    "retry_immediate": 0.0,
                    "retry_short_backoff": 0.0,
                    "retry_smart_window": 0.04,
                    "retry_payday_window": 0.01,
                    "terminal_no_retry": 0.95,
                },
                scores={"optimal_backoff_hours": -1.0, "recovery_probability": 0.02},
                feature_contributions=feature_contributions,
                routing_reason=f"Excessive retry threshold reached (attempt {retry_attempt} >= 4); stopped to prevent issuer fines.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"scheme_action": "RETRY_LIMIT_EXCEEDED", "backoff_hours": -1.0},
            )

        # ---------------------------------------------------------------------
        # 2. TECHNICAL / TRANSIENT NETWORK ERRORS
        # ---------------------------------------------------------------------
        if decline_category == "technical_error" or decline_code in TECHNICAL_ERRORS:
            feature_contributions["technical_timeout_cascade"] = 0.85
            return AgentDecisionResult(
                agent_name=self.name,
                choice="retry_immediate",
                score=0.0,
                confidence=0.92,
                probabilities={
                    "retry_immediate": 0.88,
                    "retry_short_backoff": 0.08,
                    "retry_smart_window": 0.02,
                    "retry_payday_window": 0.0,
                    "terminal_no_retry": 0.02,
                },
                scores={"optimal_backoff_hours": 0.0, "recovery_probability": 0.74},
                feature_contributions=feature_contributions,
                routing_reason=f"Decline reason '{decline_code}' is transient; cascade retry immediately via secondary rail.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"action": "IMMEDIATE_CASCADE", "backoff_hours": 0.0},
            )

        # ---------------------------------------------------------------------
        # 3. SPECIALIST SDM INFERENCE (If Backed by Trained Deep Model)
        # ---------------------------------------------------------------------
        if self.is_sdm_backed:
            sdm_out = self.evaluate_with_sdm(inputs)
            if sdm_out is not None:
                choice = sdm_out["choice"]
                conf = sdm_out["confidence"]
                probs = sdm_out["probabilities"]

                backoff_hours = BACKOFF_HOURS_MAP.get(choice, 48.0)
                recovery_prob = 0.0 if choice == "terminal_no_retry" else round(max(0.20, conf * 0.85), 2)

                return AgentDecisionResult(
                    agent_name=self.name,
                    choice=choice,
                    score=backoff_hours,
                    confidence=conf,
                    probabilities=probs,
                    scores={"optimal_backoff_hours": backoff_hours, "recovery_probability": recovery_prob},
                    feature_contributions={"sdm_latent_inference": 1.0},
                    routing_reason=f"Evaluated by Retry Specialist SDM (action={choice}, backoff={backoff_hours}h, conf={conf:.1%})",
                    is_fallback=False,
                    status="SUCCESS",
                    metadata={
                        "backed_by_sdm": True,
                        "sdm_type": "retry",
                        "backoff_hours": backoff_hours,
                        "recovery_prob": recovery_prob,
                        "is_sealed": getattr(self.sdm, "package_id", None) is not None,
                        "package_id": getattr(self.sdm, "package_id", None),
                    },
                )

        # ---------------------------------------------------------------------
        # 4. SOFT DECLINES: OPTIMAL TIMING & PAYDAY ALIGNMENT (Fallback Heuristic)
        # ---------------------------------------------------------------------
        if decline_code == "card_velocity_exceeded":
            feature_contributions["velocity_limit_intraday"] = 0.75
            return AgentDecisionResult(
                agent_name=self.name,
                choice="retry_short_backoff",
                score=8.0,
                confidence=0.88,
                probabilities={
                    "retry_immediate": 0.05,
                    "retry_short_backoff": 0.82,
                    "retry_smart_window": 0.10,
                    "retry_payday_window": 0.01,
                    "terminal_no_retry": 0.02,
                },
                scores={"optimal_backoff_hours": 8.0, "recovery_probability": 0.62},
                feature_contributions=feature_contributions,
                routing_reason="Issuer velocity limit exceeded; backoff 8 hours for intraday counter reset.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"backed_by_sdm": False, "action": "INTRADAY_BACKOFF", "backoff_hours": 8.0},
            )

        is_payday_proximity = (billing_cycle_day in (1, 15, 30, 31) or day_of_week in (3, 4))
        if decline_code == "insufficient_funds" and is_payday_proximity and retry_attempt <= 2:
            backoff_hours = 72.0 if day_of_week in (3, 4) else 96.0
            recovery_prob = round(0.78 - (retry_attempt - 1) * 0.12, 2)
            feature_contributions["payday_alignment"] = 0.55
            feature_contributions["insufficient_funds_tier"] = 0.35
            feature_contributions["attempt_penalty"] = 0.10

            return AgentDecisionResult(
                agent_name=self.name,
                choice="retry_payday_window",
                score=backoff_hours,
                confidence=0.89,
                probabilities={
                    "retry_immediate": 0.02,
                    "retry_short_backoff": 0.05,
                    "retry_smart_window": 0.20,
                    "retry_payday_window": 0.70,
                    "terminal_no_retry": 0.03,
                },
                scores={"optimal_backoff_hours": backoff_hours, "recovery_probability": recovery_prob},
                feature_contributions=feature_contributions,
                routing_reason=f"Insufficient funds aligned with salary deposit cycle; scheduled for payday window in {int(backoff_hours)}h.",
                is_fallback=False,
                status="SUCCESS",
                metadata={"backed_by_sdm": False, "action": "PAYDAY_DUNNING", "backoff_hours": backoff_hours, "recovery_rate": recovery_prob},
            )

        # ---------------------------------------------------------------------
        # 5. DEFAULT SMART BACKOFF (48h liquidity replenishment)
        # ---------------------------------------------------------------------
        attempt_multiplier = 1.5 ** (retry_attempt - 1)
        backoff_hours = min(120.0, round(36.0 * attempt_multiplier, 1))
        recovery_prob = max(0.20, round(0.68 - (retry_attempt * 0.15) - min(amount / 1000.0 * 0.05, 0.15), 2))

        feature_contributions["liquidity_settlement_window"] = 0.45
        feature_contributions["attempt_decay"] = round(0.15 * retry_attempt, 2)
        feature_contributions["amount_elasticity"] = 0.15

        return AgentDecisionResult(
            agent_name=self.name,
            choice="retry_smart_window",
            score=backoff_hours,
            confidence=0.84,
            probabilities={
                "retry_immediate": 0.04,
                "retry_short_backoff": 0.12,
                "retry_smart_window": 0.72,
                "retry_payday_window": 0.08,
                "terminal_no_retry": 0.04,
            },
            scores={"optimal_backoff_hours": backoff_hours, "recovery_probability": recovery_prob},
            feature_contributions=feature_contributions,
            routing_reason=f"Soft decline '{decline_code}'; scheduled optimal smart backoff window of {backoff_hours}h (recovery prob {recovery_prob:.0%}).",
            is_fallback=False,
            status="SUCCESS",
            metadata={"backed_by_sdm": False, "action": "SMART_BACKOFF", "backoff_hours": backoff_hours, "recovery_prob": recovery_prob},
        )
