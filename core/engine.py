"""
Laya-Style Decision Engine Runtime
===================================
High-throughput, sub-20ms decision engine runtime supporting:
- `choice`: Multi-candidate calibrated routing with feature attributions.
- `score`: Continuous risk, utility, and temperature-scaled probability calibration.
- `noul`: Fast circuit breaker, step-down, and anomaly boundary detection.

Includes empirical calibration calculation (temperature scaling, Brier scoring, ECE).
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

from payment_decision_factory.core.spec import DecisionSpec, GatewayChoice
from payment_decision_factory.core.serializer import (
    PaymentSerializer,
    SerializedPaymentState,
    get_country_region,
    classify_corridor,
)


class FeatureContribution(BaseModel):
    """Individual feature impact on the chosen decision."""
    feature: str
    impact: float
    direction: str = "positive"
    description: str


class NoulResult(BaseModel):
    """Result of the 'noul' (no-action / circuit-breaker) primitive evaluation."""
    triggered: bool = False
    action: Optional[str] = None
    reason: Optional[str] = None
    rule_name: Optional[str] = None


class DecisionResult(BaseModel):
    """
    Standardized, calibrated decision output returned by DecisionEngine.
    Compatible with dict indexing and attribute access across SDK, sandbox, and orchestrator.
    """
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spec_id: str
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    scores: Dict[str, float]
    feature_contributions: Dict[str, float]
    feature_breakdown: List[FeatureContribution] = Field(default_factory=list)
    latency_ms: float
    routing_reason: str
    is_fallback: bool = False
    noul: NoulResult = Field(default_factory=NoulResult)
    effective_fee_est: float = 0.0
    state_summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    model_config = {
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
    }

    @property
    def decision(self) -> str:
        """Alias for choice to support various client conventions."""
        return self.choice

    @property
    def noul_triggered(self) -> bool:
        """Convenience property for noul trigger status."""
        return self.noul.triggered

    @property
    def noul_action(self) -> Optional[str]:
        """Convenience property for noul action."""
        return self.noul.action

    def __getitem__(self, item: str) -> Any:
        """Support dict-style subscripting for seamless client interoperability."""
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.metadata:
            return self.metadata[item]
        raise KeyError(f"'{item}' not found in DecisionResult")

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
        d["noul_triggered"] = self.noul.triggered
        d["noul_action"] = self.noul.action
        return d


class CalibrationMetrics(BaseModel):
    """Empirical calibration metrics (Brier Score, ECE)."""
    brier_score: float
    brier_target: float
    is_calibrated: bool
    expected_calibration_error: float
    sample_size: int
    temperature: float


class DecisionEngine:
    """
    Laya-style decision engine runtime.
    Executes 'noul' guardrails, 'choice' routing, and 'score' temperature calibration
    with sub-20ms latency and explanation attributions.
    """

    def __init__(self, spec: DecisionSpec):
        self.spec = spec
        self.serializer = PaymentSerializer(spec)
        self.choices = list(spec.choices.keys())
        self.default_choice = spec.primitives.choice.default_choice
        self.temperature = spec.primitives.score.temperature
        self.weights = spec.primitives.choice.weights
        self.affinities = spec.primitives.choice.affinities

    # =========================================================================
    # PRIMITIVE 1: NOUL (Circuit breaker / Step-down / Boundary guards)
    # =========================================================================
    def evaluate_noul(self, state: SerializedPaymentState) -> NoulResult:
        """
        Evaluate 'noul' primitive: checks for circuit breaker triggers,
        velocity anomalies, and high-ticket risk thresholds before routing.
        """
        noul_cfg = self.spec.primitives.noul
        if not noul_cfg.circuit_breaker_active:
            return NoulResult(triggered=False)

        # 1. High Velocity Card Testing Check
        if state.velocity_1h >= noul_cfg.max_velocity_threshold:
            return NoulResult(
                triggered=True,
                action="STEP_DOWN_CHALLENGE",
                reason=f"Hourly card velocity ({state.velocity_1h}) exceeds safe threshold ({noul_cfg.max_velocity_threshold}). Triggering 3DS step-down challenge.",
                rule_name="excessive_velocity_circuit_breaker",
            )

        # 2. High Value + Velocity Burst Check
        if state.amount >= noul_cfg.high_value_amount_threshold and state.velocity_1h >= noul_cfg.high_value_velocity_threshold:
            return NoulResult(
                triggered=True,
                action="MANUAL_REVIEW",
                reason=f"High ticket (${state.amount:.2f}) combined with elevated velocity ({state.velocity_1h}/hr) requires manual fraud review.",
                rule_name="high_value_velocity_burst",
            )

        # 3. Suspicious MCC Check
        if state.mcc in noul_cfg.suspicious_mccs and state.amount > 10000.0:
            return NoulResult(
                triggered=True,
                action="STEP_DOWN_CHALLENGE",
                reason=f"High amount in high-risk MCC ({state.mcc}) requires heightened authentication.",
                rule_name="suspicious_mcc_high_ticket",
            )

        return NoulResult(triggered=False)

    # =========================================================================
    # PRIMITIVE 2 & 3: CHOICE & SCORE (Routing Logic, Calibration & Utility)
    # =========================================================================
    def _compute_raw_logits(self, state: SerializedPaymentState) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute uncalibrated affinity logits and feature attributions for all choices.
        """
        logits: Dict[str, float] = {}
        contributions: Dict[str, Dict[str, float]] = {}

        curr_weight = self.weights.get("currency_affinity", 0.35)
        geo_weight = self.weights.get("geo_affinity", 0.40)
        brand_weight = self.weights.get("brand_affinity", 0.15)
        cost_weight = self.weights.get("cost_optimization", 0.10)

        for choice_key, choice_spec in self.spec.choices.items():
            choice_contribs: Dict[str, float] = {}

            # 1. Currency Affinity
            curr_aff = 0.4  # baseline
            if choice_key in self.affinities.get("currencies", {}):
                curr_aff = self.affinities["currencies"][choice_key].get(state.currency, 0.3)
            elif state.currency in choice_spec.preferred_currencies:
                curr_aff = 0.95
            currency_score = curr_aff * curr_weight * 3.0
            choice_contribs["currency_match"] = round(currency_score, 4)

            # 2. Geo / Regional Domicile Affinity
            geo_score = 0.0
            if choice_spec.region == "GLOBAL":
                # Global rail handles everything with high resilience
                geo_score = 0.70 * geo_weight * 3.0
            elif choice_spec.region == state.region:
                # Direct domestic/intra-region match (US to US, EU to EU)
                geo_score = 1.0 * geo_weight * 3.0
            elif state.bin_country in choice_spec.preferred_countries:
                geo_score = 0.95 * geo_weight * 3.0
            else:
                # Cross-region mismatch penalty
                geo_score = 0.15 * geo_weight * 3.0
            choice_contribs["geo_affinity"] = round(geo_score, 4)

            # 3. Card Brand Optimization
            brand_aff = 0.5
            if choice_key in self.affinities.get("brands", {}):
                brand_aff = self.affinities["brands"][choice_key].get(state.card_brand, 0.5)
            brand_score = brand_aff * brand_weight * 2.0
            choice_contribs["card_brand_fit"] = round(brand_score, 4)

            # 4. Processing Cost Efficiency
            # Effective fee in dollars: amount * (bps/10000) + (fixed_cents/100)
            fee_est = (state.amount * (choice_spec.base_fee_bps / 10000.0)) + (choice_spec.fixed_fee_cents / 100.0)
            # Cost score: inverted normalized fee (cheaper fee gets higher utility)
            fee_rate = fee_est / max(state.amount, 1.0)
            cost_efficiency = max(0.0, 1.0 - (fee_rate * 20.0))  # higher fee reduces cost efficiency
            cost_score = cost_efficiency * cost_weight * 2.0
            choice_contribs["cost_efficiency"] = round(cost_score, 4)

            # Sum into composite logit
            total_logit = currency_score + geo_score + brand_score + cost_score
            logits[choice_key] = total_logit
            contributions[choice_key] = choice_contribs

        return logits, contributions

    def _apply_temperature_scaling(self, logits: Dict[str, float]) -> Dict[str, float]:
        """
        Calibrate choice probabilities using temperature-scaled softmax.
        P(y = c) = exp(z_c / T) / sum_k exp(z_k / T)
        """
        temp = max(self.temperature, 0.05)
        keys = list(logits.keys())
        z = np.array([logits[k] for k in keys], dtype=np.float64)

        # Scale by temperature and prevent numerical overflow
        z_scaled = z / temp
        z_max = np.max(z_scaled)
        exp_z = np.exp(z_scaled - z_max)
        probs = exp_z / np.sum(exp_z)

        return {keys[i]: float(round(probs[i], 4)) for i in range(len(keys))}

    # =========================================================================
    # CORE EVALUATE / DECIDE PIPELINE
    # =========================================================================
    def evaluate(self, inputs: Union[Dict[str, Any], Any]) -> DecisionResult:
        """
        Main evaluation entrypoint for a payment transaction.
        Measures exact execution latency and returns calibrated DecisionResult.
        """
        start_time = time.perf_counter()

        # Handle object inputs (e.g. TransactionContext with .to_dict() or dict-like)
        if hasattr(inputs, "to_dict") and callable(inputs.to_dict):
            raw_payload = inputs.to_dict()
        elif isinstance(inputs, dict):
            raw_payload = inputs
        elif hasattr(inputs, "get"):
            raw_payload = {
                "amount": inputs.get("amount", 0.0),
                "currency": inputs.get("currency", "USD"),
                "card_brand": inputs.get("card_brand", "visa"),
                "bin_country": inputs.get("bin_country", "US"),
                "mcc": inputs.get("mcc", "5411"),
                "velocity_1h": inputs.get("velocity_1h", 0),
            }
        else:
            raw_payload = dict(inputs)

        # 1. Semantic serialization
        state = self.serializer.serialize(raw_payload)

        # 2. Evaluate 'noul' primitive
        noul_res = self.evaluate_noul(state)

        # 3. Compute raw logits and feature contributions
        logits, all_contributions = self._compute_raw_logits(state)

        # 4. Temperature-scaled calibrated probabilities
        probabilities = self._apply_temperature_scaling(logits)

        # 5. Determine winning choice
        sorted_choices = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        best_choice, best_prob = sorted_choices[0]

        is_fallback = False
        routing_reason = ""

        # Check circuit breaker or uncertainty floor
        noul_floor = self.spec.primitives.noul.min_confidence_floor
        if noul_res.triggered:
            is_fallback = True
            routing_reason = f"NOUL TRIGGERED [{noul_res.action}]: {noul_res.reason}"
        elif best_prob < noul_floor:
            # Low confidence boundary fallback
            is_fallback = True
            best_choice = self.default_choice
            routing_reason = (
                f"Confidence ({best_prob:.2f}) below threshold ({noul_floor:.2f}). "
                f"Routed to resilient fallback gateway '{self.default_choice}'."
            )
        else:
            choice_meta = self.spec.choices.get(best_choice)
            display = choice_meta.display_name if choice_meta else best_choice
            routing_reason = (
                f"Optimal match: {display} selected with {best_prob*100:.1f}% confidence "
                f"based on {state.corridor} corridor and currency alignment ({state.currency})."
            )

        # 6. Build explanation feature breakdown
        chosen_contribs = all_contributions.get(best_choice, {})
        feature_breakdown = [
            FeatureContribution(
                feature="currency_match",
                impact=chosen_contribs.get("currency_match", 0.0),
                description=f"Alignment of transaction currency {state.currency} with rail clearing capabilities.",
            ),
            FeatureContribution(
                feature="geo_affinity",
                impact=chosen_contribs.get("geo_affinity", 0.0),
                description=f"Geographical alignment of card issuing country {state.bin_country} with rail region.",
            ),
            FeatureContribution(
                feature="card_brand_fit",
                impact=chosen_contribs.get("card_brand_fit", 0.0),
                description=f"Interchange efficiency and acceptance rate for {state.card_brand.upper()}.",
            ),
            FeatureContribution(
                feature="cost_efficiency",
                impact=chosen_contribs.get("cost_efficiency", 0.0),
                description="Estimated interchange and scheme fee optimization.",
            ),
        ]

        # Calculate estimated fee
        choice_spec = self.spec.choices.get(best_choice)
        effective_fee = 0.0
        if choice_spec:
            effective_fee = round(
                (state.amount * (choice_spec.base_fee_bps / 10000.0)) + (choice_spec.fixed_fee_cents / 100.0), 4
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        # Round latency for reporting, guarantee positive float
        latency_ms = round(max(elapsed_ms, 0.05), 3)

        return DecisionResult(
            spec_id=self.spec.spec_id,
            choice=best_choice,
            confidence=best_prob,
            probabilities=probabilities,
            scores={k: round(v, 4) for k, v in logits.items()},
            feature_contributions=chosen_contribs,
            feature_breakdown=feature_breakdown,
            latency_ms=latency_ms,
            routing_reason=routing_reason,
            is_fallback=is_fallback,
            noul=noul_res,
            effective_fee_est=effective_fee,
            state_summary=state.semantic_prompt,
            metadata={
                "corridor": state.corridor,
                "region": state.region,
                "amount_tier": state.amount_tier,
                "risk_profile": state.risk_profile,
                "sla_target_ms": self.spec.sla.target_latency_ms,
                "sla_met": latency_ms <= self.spec.sla.max_latency_ms,
            },
        )

    def decide(self, context: Union[Dict[str, Any], Any]) -> DecisionResult:
        """Alias for evaluate matching orchestrator and agent conventions."""
        return self.evaluate(context)

    def batch_evaluate(self, payloads: List[Dict[str, Any]]) -> List[DecisionResult]:
        """Evaluate a batch of transaction payloads in sequence with low latency."""
        return [self.evaluate(p) for p in payloads]

    # =========================================================================
    # CALIBRATION AND EVALUATION METRICS (Brier Score & ECE)
    # =========================================================================
    def compute_brier_score(
        self,
        predicted_probs: List[Dict[str, float]],
        ground_truth_choices: List[str]
    ) -> float:
        """
        Compute multi-class Brier score:
        BS = (1/N) * sum_{i=1}^N sum_{c=1}^K (p_{ic} - y_{ic})^2
        Lower is better (0.0 represents perfect probabilistic calibration).
        """
        if not predicted_probs or not ground_truth_choices:
            return 0.0

        n = len(ground_truth_choices)
        total_loss = 0.0
        choices_order = self.choices

        for i in range(n):
            true_choice = ground_truth_choices[i]
            probs = predicted_probs[i]
            for c in choices_order:
                p_ic = probs.get(c, 0.0)
                y_ic = 1.0 if c == true_choice else 0.0
                total_loss += (p_ic - y_ic) ** 2

        return round(total_loss / n, 5)

    def compute_ece(
        self,
        confidences: List[float],
        accuracies: List[bool],
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE):
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        if not confidences or not accuracies:
            return 0.0

        n = len(confidences)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = [
                j for j in range(n)
                if bin_lower <= confidences[j] < bin_upper or (i == n_bins - 1 and confidences[j] == 1.0)
            ]

            bin_size = len(in_bin)
            if bin_size > 0:
                bin_acc = sum(1.0 for j in in_bin if accuracies[j]) / bin_size
                bin_conf = sum(confidences[j] for j in in_bin) / bin_size
                ece += (bin_size / n) * abs(bin_acc - bin_conf)

        return round(float(ece), 4)

    def evaluate_calibration(
        self,
        predictions: List[DecisionResult],
        ground_truth_choices: List[str]
    ) -> CalibrationMetrics:
        """
        Run a full calibration assessment against ground truth dataset.
        """
        probs_list = [p.probabilities for p in predictions]
        confs = [p.confidence for p in predictions]
        accs = [predictions[i].choice == ground_truth_choices[i] for i in range(len(predictions))]

        brier = self.compute_brier_score(probs_list, ground_truth_choices)
        ece = self.compute_ece(confs, accs)
        target = self.spec.calibration.brier_score_target

        return CalibrationMetrics(
            brier_score=brier,
            brier_target=target,
            is_calibrated=brier <= target,
            expected_calibration_error=ece,
            sample_size=len(predictions),
            temperature=self.temperature,
        )
