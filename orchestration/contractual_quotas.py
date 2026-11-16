"""Contractual Quota Manager & Constrained Multi-Objective Routing Optimizer.

Solves the Pre-Mortem Commercial Failure Mode:
"Acquirer Minimum Volume Quotas & Tiered Pricing Penalty Avoidance."

In production, enterprise merchants have contracted volume commitments with processors
(e.g., $100M/year with Adyen for Tier-1 IC+10bps pricing; falling below triggers
retroactive 8bps penalties + non-attainment fees). Naive per-transaction cost routing
can trigger catastrophic contract penalties.

This module enforces constrained optimization:
    U(acquirer) = w_auth * P_auth - w_fee * Fee_bps + lambda_quota * Quota_Urgency
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("payment_decision_factory.orchestration.quotas")


@dataclass
class AcquirerContractualQuota:
    """Contractual volume commitment and pricing tiers for an acquirer."""
    acquirer_name: str
    target_volume_usd: float
    current_volume_usd: float
    start_date: str
    deadline_date: str
    under_quota_penalty_bps: float = 12.0  # Basis points retroactive penalty if missed
    flat_penalty_fee_usd: float = 50_000.0  # Fixed non-attainment fee
    hard_minimum: bool = True               # Whether this quota must be defended at all costs

    @property
    def remaining_volume_usd(self) -> float:
        return max(0.0, self.target_volume_usd - self.current_volume_usd)

    @property
    def attainment_percentage(self) -> float:
        if self.target_volume_usd <= 0:
            return 100.0
        return (self.current_volume_usd / self.target_volume_usd) * 100.0

    def compute_quota_pacing_urgency(self, current_date_str: Optional[str] = None) -> float:
        """Computes a normalized urgency multiplier [0.0, 5.0] based on volume trajectory.
        
        If elapsed time > volume attainment, urgency scales exponentially to protect
        the contract from catastrophic retroactive penalties.
        """
        # Parse dates (fallback to standard 365-day year if unparseable)
        try:
            d_start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
            d_end = datetime.strptime(self.deadline_date, "%Y-%m-%d").date()
            d_curr = datetime.strptime(current_date_str, "%Y-%m-%d").date() if current_date_str else date.today()
            total_days = max(1, (d_end - d_start).days)
            elapsed_days = max(0, min(total_days, (d_curr - d_start).days))
            time_progress = elapsed_days / total_days
        except Exception:
            time_progress = 0.5  # Mid-year default

        vol_progress = min(1.0, self.current_volume_usd / max(1.0, self.target_volume_usd))

        # If quota is already achieved, zero urgency
        if vol_progress >= 1.0:
            return 0.0

        # Trailing trajectory calculation
        trajectory_gap = time_progress - vol_progress
        if trajectory_gap <= 0.0:
            # Ahead of schedule: low base urgency
            return 0.15

        # Trailing behind schedule: exponential urgency ramp
        # Scaled by penalty severity (e.g. 12 bps penalty + flat fee)
        urgency = 1.0 + (trajectory_gap * 4.0)
        return min(5.0, urgency)


class ConstrainedQuotaRoutingOptimizer:
    """Multi-Objective Routing Engine with Contractual Quota Defenses.
    
    Dynamically balances:
    1. Authorization Probability (Uplift)
    2. Effective Interchange & Processing Fees (Bps)
    3. Contractual Volume Attainment to prevent catastrophic non-attainment fines.
    """

    def __init__(
        self,
        quotas: Optional[Dict[str, AcquirerContractualQuota]] = None,
        default_auth_weight: float = 1.0,
        default_fee_weight: float = 0.5,
        quota_penalty_weight: float = 0.8,
    ):
        self.quotas = quotas or {}
        self.w_auth = default_auth_weight
        self.w_fee = default_fee_weight
        self.w_quota = quota_penalty_weight

    def register_quota(self, quota: AcquirerContractualQuota) -> None:
        self.quotas[quota.acquirer_name] = quota

    def record_processed_volume(self, acquirer_name: str, amount_usd: float) -> None:
        """Updates real-time accumulators when a transaction is dispatched."""
        if acquirer_name in self.quotas:
            self.quotas[acquirer_name].current_volume_usd += amount_usd

    def optimize_routing_decision(
        self,
        candidate_rails: List[Dict[str, Any]],
        tx_amount_usd: float,
        current_date_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Selects the optimal acquiring rail subject to contractual volume constraints.
        
        Candidate rail format:
        {
            "acquirer": "adyen",
            "p_auth": 0.945,        # Predicted probability of authorization
            "fee_bps": 185.0,       # Effective interchange + processing fee in bps
            "latency_p95_ms": 120.0
        }
        """
        if not candidate_rails:
            raise ValueError("No candidate rails provided for routing optimization")

        scored_candidates = []

        for rail in candidate_rails:
            acq = rail["acquirer"]
            p_auth = float(rail.get("p_auth", 0.90))
            fee_bps = float(rail.get("fee_bps", 200.0))

            # Base economic utility (100 bps fee = -1.0 utility; 1% auth = +1.0 utility)
            auth_utility = p_auth * 100.0 * self.w_auth
            fee_penalty = (fee_bps / 100.0) * self.w_fee

            # Contractual quota urgency bonus/penalty
            quota_bonus = 0.0
            quota_info = None

            if acq in self.quotas:
                q = self.quotas[acq]
                urgency = q.compute_quota_pacing_urgency(current_date_str)
                # If behind quota, applying transaction to this acquirer delivers
                # massive avoidance value against the retroactive penalty
                quota_bonus = urgency * (q.under_quota_penalty_bps / 10.0) * self.w_quota
                quota_info = {
                    "target_volume_usd": q.target_volume_usd,
                    "current_volume_usd": q.current_volume_usd,
                    "attainment_pct": round(q.attainment_percentage, 1),
                    "urgency": round(urgency, 2),
                }

            total_utility = auth_utility - fee_penalty + quota_bonus

            scored_candidates.append({
                "acquirer": acq,
                "total_utility": round(total_utility, 3),
                "auth_utility": round(auth_utility, 3),
                "fee_penalty": round(fee_penalty, 3),
                "quota_bonus": round(quota_bonus, 3),
                "p_auth": p_auth,
                "fee_bps": fee_bps,
                "quota_info": quota_info,
            })

        # Sort by descending total utility
        scored_candidates.sort(key=lambda x: x["total_utility"], reverse=True)
        winner = scored_candidates[0]

        return {
            "selected_acquirer": winner["acquirer"],
            "total_utility": winner["total_utility"],
            "p_auth": winner["p_auth"],
            "fee_bps": winner["fee_bps"],
            "quota_bonus_applied": winner["quota_bonus"],
            "all_candidates": scored_candidates,
            "quota_pacing_active": any(c["quota_bonus"] > 0.0 for c in scored_candidates),
        }
