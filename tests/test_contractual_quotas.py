"""Tests for Contractual Quota Manager & Constrained Multi-Objective Routing."""

import pytest
from payment_decision_factory.orchestration.contractual_quotas import (
    AcquirerContractualQuota,
    ConstrainedQuotaRoutingOptimizer,
)


class TestAcquirerContractualQuota:
    """Validate quota tracking, remaining balance, and urgency scaling."""

    def test_quota_metrics(self):
        quota = AcquirerContractualQuota(
            acquirer_name="adyen",
            target_volume_usd=100_000_000.0,
            current_volume_usd=40_000_000.0,
            start_date="2026-01-01",
            deadline_date="2026-12-31",
            under_quota_penalty_bps=12.0,
        )
        assert quota.remaining_volume_usd == 60_000_000.0
        assert quota.attainment_percentage == 40.0

    def test_pacing_urgency_when_behind_schedule(self):
        # 75% through the year (October 1), but only 40% volume achieved
        quota = AcquirerContractualQuota(
            acquirer_name="adyen",
            target_volume_usd=100_000_000.0,
            current_volume_usd=40_000_000.0,
            start_date="2026-01-01",
            deadline_date="2026-12-31",
            under_quota_penalty_bps=15.0,
        )
        urgency = quota.compute_quota_pacing_urgency("2026-10-01")
        assert urgency > 2.0  # High urgency to catch up

    def test_pacing_urgency_when_target_achieved(self):
        quota = AcquirerContractualQuota(
            acquirer_name="adyen",
            target_volume_usd=100_000_000.0,
            current_volume_usd=105_000_000.0,
            start_date="2026-01-01",
            deadline_date="2026-12-31",
        )
        urgency = quota.compute_quota_pacing_urgency("2026-06-01")
        assert urgency == 0.0  # Zero urgency when quota already locked in


class TestConstrainedQuotaRoutingOptimizer:
    """Validate constrained multi-objective decisioning."""

    def test_quota_defense_biasing(self):
        # Setup: Stripe is slightly cheaper (180 bps vs 190 bps) and same auth rate (95%),
        # BUT the merchant is in danger of missing their Adyen $100M commitment,
        # which would trigger a 15 bps retroactive penalty across all Adyen volume.
        adyen_quota = AcquirerContractualQuota(
            acquirer_name="adyen",
            target_volume_usd=100_000_000.0,
            current_volume_usd=50_000_000.0,
            start_date="2026-01-01",
            deadline_date="2026-12-31",
            under_quota_penalty_bps=15.0,
        )

        optimizer = ConstrainedQuotaRoutingOptimizer(
            quotas={"adyen": adyen_quota},
            quota_penalty_weight=1.0,
        )

        candidates = [
            {"acquirer": "stripe_us", "p_auth": 0.95, "fee_bps": 180.0},
            {"acquirer": "adyen", "p_auth": 0.95, "fee_bps": 190.0},
        ]

        # On November 15 (nearly end of year, trailing by 38%), quota defense kicks in
        res = optimizer.optimize_routing_decision(
            candidate_rails=candidates,
            tx_amount_usd=500.0,
            current_date_str="2026-11-15",
        )

        # Adyen should be selected despite 10 bps higher direct fee because the quota penalty avoidance bonus dominates
        assert res["selected_acquirer"] == "adyen"
        assert res["quota_pacing_active"] is True
        assert res["quota_bonus_applied"] > 0.0

    def test_overwhelming_auth_gap_overrides_quota(self):
        # If one rail has a massive auth rate advantage (e.g. 96% vs 70%),
        # auth uplift should still win over quota pacing
        adyen_quota = AcquirerContractualQuota(
            acquirer_name="adyen",
            target_volume_usd=100_000_000.0,
            current_volume_usd=50_000_000.0,
            start_date="2026-01-01",
            deadline_date="2026-12-31",
            under_quota_penalty_bps=15.0,
        )

        optimizer = ConstrainedQuotaRoutingOptimizer(
            quotas={"adyen": adyen_quota},
            quota_penalty_weight=1.0,
        )

        candidates = [
            {"acquirer": "stripe_us", "p_auth": 0.98, "fee_bps": 180.0},
            {"acquirer": "adyen", "p_auth": 0.65, "fee_bps": 190.0},
        ]

        res = optimizer.optimize_routing_decision(
            candidate_rails=candidates,
            tx_amount_usd=100.0,
            current_date_str="2026-11-15",
        )

        # Stripe should win easily because Adyen auth rate is abysmal (33% gap)
        assert res["selected_acquirer"] == "stripe_us"
