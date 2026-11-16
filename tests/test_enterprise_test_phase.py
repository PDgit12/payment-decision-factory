"""
Comprehensive Test Suite for Enterprise Sandbox Testing Phase & Readiness Certification.
========================================================================================
Tests:
1. High-Throughput Latency Stress Testing (P50/P90/P95/P99 latency profiling, SLA adherence <15ms).
2. Counterfactual Replay Stress Testing (Approval delta bps, fee savings, net economic benefit).
3. Perturbation & Noisy Feature Robustness Testing (noise levels 1%-10%, metadata drop, resilience score).
4. Autonomous Policy Self-Adjustment Phase (temperature calibration, threshold adaptation).
5. Enterprise Readiness Certification & Cryptographic Audit Verification (SHA-256 seal integrity).
6. High-level Convenience API (run_enterprise_sandbox_phase, certify_enterprise_model).
"""

from __future__ import annotations

import os
import sys
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.engine import DecisionEngine
from payment_decision_factory.sandbox.mock_traffic import MockTrafficGenerator, PaymentTransaction
from payment_decision_factory.sandbox.enterprise_test_phase import (
    EnterpriseReadinessCertificate,
    EnterpriseSandboxTestingSuite,
    EnterpriseTestConfig,
    LatencyStressProfile,
    PerturbationRobustnessReport,
    run_enterprise_sandbox_phase,
    certify_enterprise_model,
)

CORE_SPEC_PATH = os.path.join(
    os.path.dirname(__file__), "../specs/smart_routing.yaml"
)


@pytest.fixture
def spec():
    return DecisionSpec.from_yaml(CORE_SPEC_PATH)


@pytest.fixture
def engine(spec):
    return DecisionEngine(spec)


@pytest.fixture
def traffic_gen():
    return MockTrafficGenerator(seed=42)


@pytest.fixture
def sample_batch(traffic_gen):
    return traffic_gen.generate_batch("mixed", count=60)


class TestEnterpriseLatencyStress:
    """Tests for latency profiling and SLA compliance under batch conditions."""

    def test_latency_stress_profile_metrics(self, engine, spec, sample_batch):
        suite = EnterpriseSandboxTestingSuite()
        profile = suite.run_latency_stress_test(engine, sample_batch, spec=spec)

        assert isinstance(profile, LatencyStressProfile)
        assert profile.total_samples == 60
        assert profile.p50_latency_ms < 15.0
        assert profile.p95_latency_ms < 15.0
        assert profile.p99_latency_ms < 15.0
        assert profile.sla_budget_ms == 15.0
        assert profile.sla_breaches_count == 0
        assert profile.sla_adherence_pct == 100.0
        assert profile.throughput_tps > 0.0

        d = profile.to_dict()
        assert d["total_samples"] == 60
        assert "p95_latency_ms" in d
        assert "throughput_tps" in d


class TestEnterpriseCounterfactualStress:
    """Tests for counterfactual replay and economic performance."""

    def test_counterfactual_stress_test(self, engine, spec, sample_batch):
        suite = EnterpriseSandboxTestingSuite()
        report = suite.run_counterfactual_stress_test(engine, sample_batch, spec=spec)

        assert report.total_transactions == 60
        assert 0.0 <= report.baseline_approval_rate <= 1.0
        assert 0.0 <= report.model_approval_rate <= 1.0
        assert isinstance(report.approval_rate_delta_bps, float)
        assert isinstance(report.net_benefit_usd, float)


class TestEnterprisePerturbationRobustness:
    """Tests for noise tolerance and feature degradation resilience."""

    def test_perturbation_robustness_test(self, engine, spec, sample_batch):
        config = EnterpriseTestConfig(noise_levels=[0.01, 0.05])
        suite = EnterpriseSandboxTestingSuite(config=config)
        report = suite.run_perturbation_robustness_test(engine, sample_batch, spec=spec)

        assert isinstance(report, PerturbationRobustnessReport)
        assert 0.0 <= report.baseline_approval_rate <= 1.0
        assert 0.0 <= report.resilience_score <= 1.0
        assert "noise_std_0.01" in report.perturbed_metrics
        assert "noise_std_0.05" in report.perturbed_metrics
        assert isinstance(report.drift_detected, bool)

        d = report.to_dict()
        assert "resilience_score" in d
        assert "perturbed_metrics" in d


class TestEnterpriseCertification:
    """Tests for end-to-end certification and cryptographic audit validation."""

    def test_full_certification_pass(self, engine, spec, sample_batch):
        config = EnterpriseTestConfig(sla_budget_ms=15.0, min_sla_adherence_pct=95.0)
        suite = EnterpriseSandboxTestingSuite(config=config)

        cert = suite.certify(
            model=engine,
            spec=spec,
            records=sample_batch,
            model_tier="sub_billion",
            model_name="smart_routing_sub_billion_transformer",
        )

        assert isinstance(cert, EnterpriseReadinessCertificate)
        assert cert.status == "CERTIFIED_ENTERPRISE_READY"
        assert cert.is_certified is True
        assert cert.model_tier == "sub_billion"
        assert cert.total_transactions_tested == 60
        assert cert.latency_profile.p95_latency_ms < 15.0
        assert cert.counterfactual_report.total_transactions == 60
        assert cert.robustness_report.resilience_score > 0.80
        assert cert.self_adjustment_result is not None
        assert len(cert.sha256_audit_hash) == 64

        # Verify audit hash integrity
        assert cert.verify_audit_hash() is True

        # Test dictionary and JSON serialization
        d = cert.to_dict()
        assert d["certificate_id"] == cert.certificate_id
        assert d["status"] == "CERTIFIED_ENTERPRISE_READY"

        j = cert.to_json()
        assert isinstance(j, str)
        assert cert.certificate_id in j

    def test_certification_rejection_on_strict_sla_breach(self, engine, spec, sample_batch):
        # Artificially tiny SLA budget to verify rejection logic
        strict_config = EnterpriseTestConfig(sla_budget_ms=0.000001, min_sla_adherence_pct=100.0)
        suite = EnterpriseSandboxTestingSuite(config=strict_config)

        cert = suite.certify(
            model=engine,
            spec=spec,
            records=sample_batch,
            model_tier="sub_billion",
        )

        assert cert.status == "REJECTED_SLA_BREACH"
        assert cert.is_certified is False
        assert any("P95 latency breach" in r or "SLA adherence" in r for r in cert.reasons)
        assert cert.verify_audit_hash() is True


class TestConvenienceAPI:
    """Tests for run_enterprise_sandbox_phase and certify_enterprise_model."""

    def test_run_enterprise_sandbox_phase_shortcut(self, engine, spec, sample_batch):
        cert = run_enterprise_sandbox_phase(
            model=engine,
            spec=spec,
            records=sample_batch,
            model_tier="sub_billion",
        )
        assert isinstance(cert, EnterpriseReadinessCertificate)
        assert cert.is_certified is True
        assert cert.verify_audit_hash() is True

    def test_certify_enterprise_model_shortcut(self, engine, spec, sample_batch):
        cert = certify_enterprise_model(
            model=engine,
            spec=spec,
            records=sample_batch,
            model_tier="sub_billion",
        )
        assert isinstance(cert, EnterpriseReadinessCertificate)
        assert cert.is_certified is True
