"""
Comprehensive Master Integration Test Suite for Enterprise Sub-Billion Specialist SDM & Sandbox Testing Phase.
==============================================================================================================
Validates:
1. High-capacity Tabular Feature-Tokenizer Transformer (FT-Transformer) scaling from 1M to 100M+ parameters.
2. Dynamic entity embeddings for high-cardinality features (BINs, MCCs, corridors) and calibrated temperature heads.
3. Fast CPU training, self-contained ONNX artifact export, and ONNX Runtime parity with sub-10ms latency.
4. Comprehensive Sandbox Testing Phase: Latency SLA stress testing (<15ms budget), counterfactual replay,
   feature perturbation robustness, policy self-adjustment, and tamper-evident EnterpriseReadinessCertificate.
5. Sealed closed-source distribution (.sdmpkg) with AES-256-GCM authenticated encryption, AAD manifest binding,
   and in-memory zero-disk execution.
6. Closed-Loop Orchestration integration supporting tier="sub_billion" and canary deployment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

# Ensure standard library 'code' module is bound to stdlib
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code) and ("code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")):
    import importlib.util
    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SDK_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sdk/python"))
if _SDK_PYTHON not in sys.path:
    sys.path.insert(0, _SDK_PYTHON)

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.engine import DecisionEngine
from payment_decision_factory.core.enterprise_sdm import (
    EnterpriseSubBillionSDM,
    NumericalFeatureTokenizer,
    TransformerBlock,
    EnterpriseONNXWrapper,
    EnterpriseONNXRunner,
    export_enterprise_sdm_to_onnx,
)
from payment_decision_factory.core.sealed_package import (
    SealedPackage,
    DecryptedPackageContext,
    SealedModelRunner,
    LicenseAuthenticationError,
    LicenseExpiredError,
)
from payment_decision_factory.sandbox.mock_traffic import MockTrafficGenerator, PaymentTransaction
from payment_decision_factory.sandbox.simulator import CounterfactualReport
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
TEST_LICENSE_KEY = "ENT-TEST-KEY-9988-7766-5544"
TEST_CLIENT_ID = "enterprise_fintech_corp"


@pytest.fixture
def spec() -> DecisionSpec:
    return DecisionSpec.from_yaml(CORE_SPEC_PATH)


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    gen = MockTrafficGenerator(seed=1337)
    txs = gen.generate_batch(scenario="happy_path", count=60)
    records = []
    for tx in txs:
        d = tx.to_dict()
        d["historical_gateway"] = "stripe_us"
        d["optimal_gateway"] = "stripe_us" if tx.amount < 100.0 else "adyen_eu"
        records.append(d)
    return records


# =============================================================================
# 1. ENTERPRISE SUB-BILLION ARCHITECTURE & PARAMETER SCALING
# =============================================================================

class TestEnterpriseSubBillionArchitecture:
    """Validates the Tabular Transformer architecture, parameter budgets, and calibrated heads."""

    def test_model_instantiation_and_parameter_scaling(self):
        """Verify model construction, embedding layers, and parameter scaling tiers."""
        # Tier 1: Small (~1M parameters for fast local dev & CI)
        small_model = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=6,
            num_classes=4,
            categorical_dims={"bin_country": 50, "card_brand": 10, "currency": 20},
            parameter_target="small",
        )
        small_params = small_model.count_parameters()
        assert small_params > 100_000, f"Expected >100k params, got {small_params}"
        assert small_params < 5_000_000, f"Expected <5M params for small tier, got {small_params}"

        # Tier 2: Medium (~10M parameters for standard enterprise workloads)
        medium_model = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=6,
            num_classes=4,
            categorical_dims={"bin_country": 50, "card_brand": 10, "currency": 20},
            parameter_target="medium",
        )
        med_params = medium_model.count_parameters()
        assert med_params > 4_000_000, f"Expected >4M params, got {med_params}"

        # Tier 3: Large (100M+ parameters for high-scale enterprise clusters)
        large_model = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=6,
            num_classes=4,
            categorical_dims={"bin_country": 50, "card_brand": 10, "currency": 20},
            parameter_target="large",
        )
        large_params = large_model.count_parameters()
        assert large_params > 80_000_000, f"Expected >80M params for sub-billion tier, got {large_params}"

    def test_forward_pass_and_probability_calibration(self):
        """Test forward pass returning calibrated logits and valid probability distribution."""
        model = EnterpriseSubBillionSDM(
            num_continuous_features=4,
            num_classes=3,
            categorical_dims={"bin": 25, "mcc": 30},
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )
        batch_size = 5
        continuous_x = torch.randn(batch_size, 4)
        categorical_x = torch.tensor([[1, 2], [3, 4], [0, 0], [10, 15], [24, 29]], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            logits, probs = model(continuous_x, categorical_x)

        assert logits.shape == (batch_size, 3)
        assert probs.shape == (batch_size, 3)

        # Probabilities must sum to 1.0 along class dimension
        prob_sums = probs.sum(dim=-1).numpy()
        assert np.allclose(prob_sums, np.ones(batch_size), atol=1e-5)

        # All probabilities strictly between 0 and 1
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_temperature_scaling_impact(self):
        """Verify temperature scaling softens logits without changing prediction ranking."""
        model = EnterpriseSubBillionSDM(
            num_continuous_features=4,
            num_classes=3,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )
        x = torch.randn(1, 4)

        model.eval()
        with torch.no_grad():
            model.temperature.data = torch.tensor([0.5])
            _, sharp_probs = model(x)

            model.temperature.data = torch.tensor([2.0])
            _, soft_probs = model(x)

        # Ranking must be strictly identical
        assert int(torch.argmax(sharp_probs)) == int(torch.argmax(soft_probs))
        # Max confidence with high temperature (soft) must be lower than with low temperature (sharp)
        assert float(torch.max(soft_probs)) < float(torch.max(sharp_probs))


# =============================================================================
# 2. ENTERPRISE ONNX EXPORT AND IN-MEMORY RUNNER
# =============================================================================

class TestEnterpriseONNXExportAndRuntime:
    """Validates self-contained ONNX artifact generation, numerical parity, and sub-10ms SLA."""

    def test_export_and_in_memory_runtime(self):
        """Export EnterpriseSubBillionSDM to ONNX and evaluate with EnterpriseONNXRunner."""
        model = EnterpriseSubBillionSDM(
            num_continuous_features=5,
            num_classes=3,
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
        )
        choices = ["stripe_us", "adyen_eu", "checkout_uk"]

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "enterprise_sdm.onnx")
            exported_path = export_enterprise_sdm_to_onnx(
                model=model,
                output_path=onnx_path,
                input_dim=5,
                verify=True,
            )

            assert os.path.exists(exported_path)
            assert os.path.getsize(exported_path) > 1024  # Valid artifact

            # Verify no orphaned external .data files
            assert not os.path.exists(exported_path + ".data")

            # Load into high-throughput EnterpriseONNXRunner
            runner = EnterpriseONNXRunner(exported_path, choices=choices)

            # Test single-record evaluation latency SLA (<15ms)
            test_features = [150.0, 1.0, 0.0, 50.0, 2.0]
            start = time.perf_counter()
            decision = runner.evaluate(test_features)
            eval_duration_ms = (time.perf_counter() - start) * 1000.0

            assert decision["choice"] in choices
            assert 0.0 <= decision["confidence"] <= 1.0
            assert decision["tier"] == "enterprise_sub_billion"
            assert eval_duration_ms < 15.0, f"Latency SLA breach: {eval_duration_ms:.2f}ms"
            assert decision["latency_ms"] < 15.0

    def test_onnx_pytorch_parity(self):
        """Ensure ONNX runtime predictions match PyTorch forward pass outputs."""
        model = EnterpriseSubBillionSDM(
            num_continuous_features=4,
            num_classes=3,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )
        choices = ["stripe_us", "adyen_eu", "checkout_uk"]

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "parity_test.onnx")
            export_enterprise_sdm_to_onnx(model, onnx_path, input_dim=4, verify=False)

            runner = EnterpriseONNXRunner(onnx_path, choices=choices)

            test_input = np.array([[10.0, 2.5, 0.0, 1.0]], dtype=np.float32)
            model.eval()
            with torch.no_grad():
                _, pt_probs = model(torch.tensor(test_input))

            _, onnx_probs = runner.predict(test_input)

            assert np.allclose(pt_probs.numpy(), onnx_probs, atol=1e-3)


# =============================================================================
# 3. ENTERPRISE SANDBOX TESTING PHASE & READINESS CERTIFICATION
# =============================================================================

class TestEnterpriseSandboxTestingPhase:
    """Validates the comprehensive digital twin testing suite and readiness certificate."""

    def test_latency_stress_profile(self, spec, sample_records):
        """Verify batch P50/P95/P99 latency profiling against 15ms budget."""
        suite = EnterpriseSandboxTestingSuite(config=EnterpriseTestConfig(sla_budget_ms=15.0))
        engine = DecisionEngine(spec)

        profile = suite.run_latency_stress_test(engine, sample_records, spec=spec)
        assert isinstance(profile, LatencyStressProfile)
        assert profile.total_samples == len(sample_records)
        assert profile.sla_budget_ms == 15.0
        assert profile.p50_latency_ms > 0.0
        assert profile.p95_latency_ms < 15.0
        assert profile.p99_latency_ms < 25.0
        assert profile.sla_adherence_pct >= 90.0
        assert profile.throughput_tps > 0.0

    def test_counterfactual_replay_stress_test(self, spec, sample_records):
        """Verify historical replay measuring approval uplift and economic savings."""
        suite = EnterpriseSandboxTestingSuite()
        engine = DecisionEngine(spec)

        report = suite.run_counterfactual_stress_test(engine, sample_records, spec=spec)
        assert isinstance(report, CounterfactualReport)
        assert report.total_transactions == len(sample_records)
        assert report.model_approval_rate >= 0.70
        assert report.approval_rate_delta_bps is not None

    def test_perturbation_robustness_test(self, spec, sample_records):
        """Verify model decision stability under Gaussian feature noise (1%-10%)."""
        suite = EnterpriseSandboxTestingSuite()
        engine = DecisionEngine(spec)

        report = suite.run_perturbation_robustness_test(engine, sample_records, spec=spec)
        assert isinstance(report, PerturbationRobustnessReport)
        assert report.resilience_score > 0.75
        assert 0.01 in [float(k.replace("noise_std_", "")) for k in report.perturbed_metrics.keys() if "std" in k] or len(report.perturbed_metrics) > 0

    def test_enterprise_readiness_certification_lifecycle(self, spec, sample_records):
        """Test full enterprise certification generating a tamper-evident audit certificate."""
        suite = EnterpriseSandboxTestingSuite(config=EnterpriseTestConfig(sla_budget_ms=15.0))
        engine = DecisionEngine(spec)

        cert = suite.certify(
            model=engine,
            spec=spec,
            records=sample_records,
            model_tier="sub_billion",
            model_name="smart_routing_sub_billion_v1",
        )

        assert isinstance(cert, EnterpriseReadinessCertificate)
        assert cert.status == "CERTIFIED_ENTERPRISE_READY"
        assert cert.is_certified is True
        assert cert.model_tier == "sub_billion"
        assert cert.total_transactions_tested == len(sample_records)
        assert cert.latency_profile.p95_latency_ms < 15.0
        assert cert.counterfactual_report is not None
        assert cert.robustness_report is not None

        # Verify cryptographic audit seal
        assert cert.verify_audit_hash() is True

        # Test tampering detection on audit hash
        tampered_cert = copy.deepcopy(cert)
        tampered_cert.summary_kpis["p50_latency_ms"] = 999.0
        assert tampered_cert.verify_audit_hash() is False

        # Test serialization
        cert_dict = cert.to_dict()
        assert cert_dict["is_certified"] is True
        assert "sha256_audit_hash" in cert_dict
        cert_json = cert.to_json()
        assert "CERTIFIED_ENTERPRISE_READY" in cert_json

    def test_certification_fails_on_unreasonable_sla(self, spec, sample_records):
        """Verify that strict SLA failure correctly blocks certification."""
        # Unrealistic 0.001ms budget forces SLA breach
        strict_config = EnterpriseTestConfig(sla_budget_ms=0.001, min_sla_adherence_pct=99.0)
        suite = EnterpriseSandboxTestingSuite(config=strict_config)
        engine = DecisionEngine(spec)

        cert = suite.certify(engine, spec, sample_records, model_tier="sub_billion")
        assert cert.status == "REJECTED_SLA_BREACH"
        assert cert.is_certified is False
        assert any("latency breach" in r.lower() or "sla" in r.lower() for r in cert.reasons)


# =============================================================================
# 4. SEALED CLOSED-SOURCE DISTRIBUTION (.SDMPKG) FOR ENTERPRISE MODELS
# =============================================================================

class TestEnterpriseSealedPackaging:
    """Validates AES-256-GCM encrypted distribution with certificate binding."""

    def test_sealed_enterprise_package_lifecycle(self, spec, sample_records):
        """Seal enterprise ONNX model into .sdmpkg and verify zero-disk in-memory execution."""
        # Train small enterprise model
        model = EnterpriseSubBillionSDM(
            num_continuous_features=4,
            num_classes=3,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )
        choices = ["stripe_us", "adyen_eu", "checkout_uk"]

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            export_enterprise_sdm_to_onnx(model, onnx_path, input_dim=4, verify=False)
            with open(onnx_path, "rb") as f:
                onnx_bytes = f.read()

            pkg_path = os.path.join(tmpdir, "enterprise_routing.sdmpkg")

            # 1. Seal package with enterprise tier and audit hash
            audit_hash = hashlib.sha256(b"mock_audit_certificate").hexdigest()
            sealed_pkg = SealedPackage.seal(
                model_bytes_or_path=onnx_bytes,
                license_key=TEST_LICENSE_KEY,
                client_id=TEST_CLIENT_ID,
                domain=spec.domain,
                model_type="enterprise_sub_billion_sdm",
                spec_dict=spec.model_dump() if hasattr(spec, "model_dump") else spec.dict(),
                choices=choices,
                input_dim=4,
                validity_days=90,
                output_path=pkg_path,
            )

            assert os.path.exists(pkg_path)
            assert sealed_pkg.manifest["client_id"] == TEST_CLIENT_ID

            # 2. In-memory decryption and execution
            loaded_pkg = SealedPackage.load_file(pkg_path)
            decrypted = loaded_pkg.decrypt(TEST_LICENSE_KEY)
            assert decrypted.client_id == TEST_CLIENT_ID
            assert len(decrypted.model_bytes) == len(onnx_bytes)

            # Execute ONNX session directly from RAM (zero disk writes)
            runner = decrypted.create_runner()
            res = runner.evaluate([1.0, 2.0, 3.0, 4.0])
            assert res["choice"] in choices
            assert 0.0 <= res["confidence"] <= 1.0

            # 3. Security: Tampering with manifest client_id triggers LicenseAuthenticationError
            raw_dict = loaded_pkg.to_dict()
            raw_dict["manifest"]["client_id"] = "hacked_client"
            tampered_pkg = SealedPackage.from_dict(raw_dict)

            with pytest.raises(LicenseAuthenticationError):
                tampered_pkg.decrypt(TEST_LICENSE_KEY)

            # 4. Security: Invalid license key is rejected
            with pytest.raises(LicenseAuthenticationError):
                loaded_pkg.decrypt("WRONG-KEY-0000")


# =============================================================================
# 5. FULL CLOSED-LOOP ENTERPRISE PIPELINE INTEGRATION
# =============================================================================

class TestEnterpriseClosedLoopPipelineIntegration:
    """Validates end-to-end enterprise closed-loop lifecycle, certification, and canary deployment."""

    @pytest.mark.asyncio
    async def test_closed_loop_enterprise_readiness_and_canary_flow(self, spec, sample_records):
        """Validate that the enterprise sandbox phase and readiness certification wire into the pipeline."""
        from payment_decision_factory.orchestration.closed_loop import (
            ClosedLoopOrchestrator,
            RolloutMode,
            CanaryDeploymentAgent,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = ClosedLoopOrchestrator(
                client_id=TEST_CLIENT_ID,
                output_dir=tmpdir,
            )

            # Certify enterprise readiness via sandbox testing phase
            cert = run_enterprise_sandbox_phase(
                model=DecisionEngine(spec),
                spec=spec,
                records=sample_records,
                model_tier="sub_billion",
            )
            assert cert.is_certified is True

            # Execute closed loop with canary deployment on enterprise sub_billion tier
            result = await orchestrator.execute_closed_loop(
                spec=spec,
                records=sample_records,
                specialist_type="routing",
                target_field="optimal_gateway",
                rollout_mode=RolloutMode.CANARY,
                canary_percentage=15.0,
                license_key=TEST_LICENSE_KEY,
                tier="sub_billion",
            )

            assert result.status == "SUCCESS"
            assert result.readiness_certificate is not None
            assert result.readiness_certificate.is_certified is True
            assert result.training_metrics["tier"] == "sub_billion"
            assert result.training_metrics["parameter_count"] > 100_000
            assert result.sealed_package is not None
            assert os.path.exists(result.sealed_package.package_path)
            assert result.deployment_info["rollout_mode"] == "CANARY"
            assert result.deployment_info["canary_percentage"] == 15.0
