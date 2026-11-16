"""
Unit tests for Pre-Mortem Algorithmic Defenses.
==============================================
Validates:
1. TwoSpeedLabelResolver (45-90d chargeback lag & survival weighting)
2. HierarchicalEntityEmbedding (350k+ BIN cold-start Bayesian shrinkage)
3. OnlineAcquirerDriftDetector (Sudden acquirer policy shock detection & Thompson failover)
"""

import os
import sys
import time
import pytest
import numpy as np
import torch

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.premortem_defenses import (
    TwoSpeedLabelResolver,
    HierarchicalEntityEmbedding,
    OnlineAcquirerDriftDetector,
)


class TestTwoSpeedLabelResolver:
    def test_fresh_unobserved_transaction_survival_weighting(self):
        resolver = TwoSpeedLabelResolver(maturity_threshold_days=90.0, early_fraud_decay_days=30.0)
        now = time.time()

        # 1-day old transaction: unobserved, immature, receives low weight (~0.10)
        r_1d = {"transaction_id": "tx_1d", "timestamp_epoch": now - 86400, "outcome": "approved"}
        res_1d = resolver.resolve_record(r_1d, current_time_epoch=now)
        assert res_1d.is_fraud_mature is False
        assert res_1d.fraud_label == 0
        assert 0.01 < res_1d.sample_weight < 0.15

        # 45-day old transaction: partially matured, receives moderate weight (~0.75-0.80)
        r_45d = {"transaction_id": "tx_45d", "timestamp_epoch": now - (45 * 86400), "outcome": "approved"}
        res_45d = resolver.resolve_record(r_45d, current_time_epoch=now)
        assert res_45d.is_fraud_mature is False
        assert 0.70 < res_45d.sample_weight < 0.85

        # 95-day old transaction: fully matured clean transaction, full weight (1.0)
        r_95d = {"transaction_id": "tx_95d", "timestamp_epoch": now - (95 * 86400), "outcome": "approved"}
        res_95d = resolver.resolve_record(r_95d, current_time_epoch=now)
        assert res_95d.is_fraud_mature is True
        assert res_95d.sample_weight == 1.0

    def test_early_scheme_fraud_alert_overrides_lag(self):
        resolver = TwoSpeedLabelResolver()
        now = time.time()

        # 2-day old transaction with TC40 / SAFE issuer notification
        r_tc40 = {
            "transaction_id": "tx_tc40",
            "timestamp_epoch": now - (2 * 86400),
            "outcome": "approved",
            "has_early_fraud_alert": True,
        }
        res_tc40 = resolver.resolve_record(r_tc40, current_time_epoch=now)
        assert res_tc40.is_fraud_mature is True
        assert res_tc40.fraud_label == 1
        assert res_tc40.sample_weight == 1.0


class TestHierarchicalEntityEmbedding:
    def test_bayesian_shrinkage_on_cold_start_bins(self):
        emb = HierarchicalEntityEmbedding(
            num_bins=10_000,
            num_issuers=100,
            num_countries=50,
            num_brands=4,
            embedding_dim=32,
            shrinkage_tau=20.0,
        )

        # Register 1,000 observations for BIN 42 (hot frequent BIN)
        freq_bins = torch.tensor([42] * 1000)
        emb.update_frequencies(freq_bins)

        # BIN 9999 has ZERO observations (cold start)
        bin_hot = torch.tensor([42])
        bin_cold = torch.tensor([9999])
        issuer = torch.tensor([10])
        country = torch.tensor([5])
        brand = torch.tensor([1])

        out_hot = emb(bin_hot, issuer, country, brand)
        out_cold = emb(bin_cold, issuer, country, brand)

        assert out_hot.shape == (1, 32)
        assert out_cold.shape == (1, 32)
        assert torch.isfinite(out_hot).all()
        assert torch.isfinite(out_cold).all()

        # For cold-start BIN, embedding matches parent hierarchy
        parent_vec = (
            emb.issuer_emb(issuer) +
            emb.country_emb(country) +
            emb.brand_emb(brand) +
            emb.type_emb(torch.tensor([0]))
        ) / 4.0
        torch.testing.assert_close(out_cold, parent_vec, atol=1e-5, rtol=1e-5)


class TestOnlineAcquirerDriftDetector:
    def test_clean_traffic_diagnostics(self):
        detector = OnlineAcquirerDriftDetector(rails=["stripe_us", "adyen_eu"])
        detector.set_expected_rate("stripe_us", 0.95)

        # 100 successful transactions (95% expected)
        for i in range(100):
            detector.record_outcome("stripe_us", success=(i % 20 != 0))

        status = detector.diagnose_rail("stripe_us")
        assert status.is_drifting is False
        assert status.empirical_rate >= 0.90
        assert status.recommended_override is None

        choice, overridden, _ = detector.resolve_routing_choice("stripe_us")
        assert choice == "stripe_us"
        assert overridden is False

    def test_sudden_policy_shock_and_autonomous_reroute(self):
        detector = OnlineAcquirerDriftDetector(rails=["stripe_us", "adyen_eu", "checkout_crossborder"])
        detector.set_expected_rate("stripe_us", 0.95)
        detector.set_expected_rate("adyen_eu", 0.94)

        # Adyen is healthy (95% approvals)
        for _ in range(50):
            detector.record_outcome("adyen_eu", success=True)

        # Stripe suffers a policy shock / sudden 500-error collapse (drops to 40% auth)
        for i in range(60):
            detector.record_outcome("stripe_us", success=(i % 3 == 0))

        status = detector.diagnose_rail("stripe_us")
        assert status.is_drifting is True
        assert status.empirical_rate < 0.60
        assert status.z_score < -3.0
        assert status.recommended_override == "adyen_eu"

        # Autonomous resolution routes around failing rail
        choice, overridden, reason = detector.resolve_routing_choice("stripe_us")
        assert choice == "adyen_eu"
        assert overridden is True
        assert "AUTONOMOUS DRIFT OVERRIDE" in reason
