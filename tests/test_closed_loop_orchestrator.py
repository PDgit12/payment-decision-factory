"""
Comprehensive Test Suite for Universal Closed-Loop Platform Lifecycle Orchestrator.
===================================================================================
Tests:
1. Data Ingestion: Telemetry schema profiling, distributions, missing value rates.
2. Digital Twin Simulation: Counterfactual replay, KPI estimation.
3. Policy Self-Adjustment: Autonomous threshold and calibration optimization.
4. Sealed Packaging: AES-256-GCM encryption, checksum manifest, zero-disk leakage.
5. Deployment Rollout: Shadow parity, Canary traffic allocation, auto-rollback on SLA breach.
6. Full End-to-End Closed-Loop Lifecycle Execution.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.orchestration.agent_base import MockAgent
from payment_decision_factory.orchestration.closed_loop import (
    CanaryDeploymentAgent,
    ClosedLoopLifecycleResult,
    ClosedLoopOrchestrator,
    IngestionReport,
    LifecyclePhase,
    RolloutMode,
    SealedPackageMetadata,
    SelfAdjustmentDelta,
    SimulationReport,
)
from payment_decision_factory.orchestration.context import (
    ExecutionStatus,
    TransactionContext,
)
from payment_decision_factory.orchestration.pipeline import PaymentOrchestrationPipeline
from payment_decision_factory.sandbox.mock_traffic import MockTrafficGenerator

CORE_SPEC_PATH = os.path.join(
    os.path.dirname(__file__), "../specs/smart_routing.yaml"
)


@pytest.fixture
def sample_records():
    """Generate 60 realistic test transactions for lifecycle orchestration."""
    gen = MockTrafficGenerator()
    return gen.generate_batch("mixed", count=60)


@pytest.fixture
def spec():
    return DecisionSpec.from_yaml(CORE_SPEC_PATH)


class TestDataIngestion:
    """Tests for Step 1: Telemetry Data Ingestion & Profiling."""

    def test_ingest_records_profiling(self, sample_records):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = ClosedLoopOrchestrator(client_id="test_merchant", output_dir=tmpdir)
            report = orch.ingest_data(sample_records, target_field="optimal_gateway")

            assert isinstance(report, IngestionReport)
            assert report.total_records == 60
            assert "amount" in report.numeric_features
            assert "currency" in report.categorical_features
            assert "card_brand" in report.categorical_features
            assert report.missing_value_rate >= 0.0
            assert len(report.target_distribution) > 0


class TestSimulationAndSelfAdjustment:
    """Tests for Step 3 & 4: Digital Twin Simulation and Policy Self-Adjustment."""

    def test_simulation_replay_and_kpis(self, spec, sample_records):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = ClosedLoopOrchestrator(client_id="test_merchant", output_dir=tmpdir)
            from payment_decision_factory.core.engine import DecisionEngine
            engine = DecisionEngine(spec)

            sim_report = orch.run_simulation(model=engine, spec=spec, records=sample_records)
            assert isinstance(sim_report, SimulationReport)
            assert sim_report.total_simulated == 60
            assert 0.0 <= sim_report.approval_rate <= 1.0
            assert sim_report.average_confidence > 0.0
            assert sim_report.p95_latency_ms < 15.0

    def test_policy_self_adjustment_optimization(self, spec):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = ClosedLoopOrchestrator(client_id="test_merchant", output_dir=tmpdir)
            # Create a mock low-approval simulation report
            low_sim = SimulationReport(
                total_simulated=100,
                approval_rate=0.88,  # Below 0.95 target
                average_confidence=0.62,
                brier_score=0.065,
                p95_latency_ms=0.15,
                estimated_fee_bps=190.0,
                challenge_rate=0.08,  # High challenge rate > 5%
                counterfactual_uplift_pct=1.1,
                outcomes_breakdown={"stripe_us": 50, "adyen_eu": 38, "checkout_crossborder": 12},
            )

            adjusted_spec, delta = orch.self_adjust_policy(spec, low_sim, target_approval_rate=0.95)
            assert isinstance(delta, SelfAdjustmentDelta)
            # Temperature and confidence floor should be tightened/expanded
            assert adjusted_spec.primitives.score.temperature < spec.primitives.score.temperature
            assert adjusted_spec.primitives.score.confidence_floor <= spec.primitives.score.confidence_floor
            # Velocity threshold relaxed due to high challenge rate
            assert adjusted_spec.primitives.noul.max_velocity_threshold > spec.primitives.noul.max_velocity_threshold
            assert len(delta.adjusted_rules) >= 2


class TestSealedPackaging:
    """Tests for Step 5: Sealed Packaging (.sdmpkg AES-256-GCM)."""

    def test_seal_package_generation(self, spec):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = ClosedLoopOrchestrator(client_id="acme_corp", output_dir=tmpdir)
            from payment_decision_factory.core.engine import DecisionEngine
            engine = DecisionEngine(spec)

            pkg_meta = orch.seal_package(
                model=engine,
                spec=spec,
                license_key="lic_acme_corp_secret_key_12345",
            )
            assert isinstance(pkg_meta, SealedPackageMetadata)
            assert pkg_meta.client_id == "acme_corp"
            assert pkg_meta.encryption_algorithm == "AES-256-GCM"
            assert os.path.exists(pkg_meta.package_path)
            assert pkg_meta.package_path.endswith(".sdmpkg")
            assert pkg_meta.file_size_bytes > 0
            assert len(pkg_meta.sha256_checksum) == 64


class TestCanaryDeployment:
    """Tests for Step 6: Canary, Shadow, and Active Deployment Rollout."""

    @pytest.mark.asyncio
    async def test_shadow_rollout_mode(self):
        control = MockAgent(name="control_agent", choice="stripe_us", confidence=0.85)
        candidate = MockAgent(name="candidate_agent", choice="adyen_eu", confidence=0.92)

        canary_agent = CanaryDeploymentAgent(
            name="test_shadow",
            control_agent=control,
            candidate_agent=candidate,
            mode=RolloutMode.SHADOW,
        )

        ctx = TransactionContext(inputs={"amount": 100.0})
        res = await canary_agent.decide(ctx)

        # Shadow mode MUST always return control decision to caller
        assert res.choice == "stripe_us"
        # Wait a tick for async shadow task to complete
        await asyncio.sleep(0.05)
        assert canary_agent.candidate_calls >= 1

    @pytest.mark.asyncio
    async def test_canary_auto_rollback_on_errors(self):
        control = MockAgent(name="control_agent", choice="stripe_us")
        # Candidate fails 100% of the time
        failing_candidate = MockAgent(name="bad_candidate", fail=True)

        canary_agent = CanaryDeploymentAgent(
            name="failing_canary",
            control_agent=control,
            candidate_agent=failing_candidate,
            mode=RolloutMode.CANARY,
            canary_percentage=100.0,  # All traffic to candidate to trigger error threshold
            max_error_rate=0.10,
        )

        # Execute 12 transactions to trip the threshold (min 10 calls)
        for _ in range(12):
            ctx = TransactionContext(inputs={"amount": 50.0})
            await canary_agent.decide(ctx)

        assert canary_agent.is_rolled_back is True
        assert canary_agent.rollback_reason is not None

        # After rollback, subsequent calls MUST route to control safely!
        ctx_after = TransactionContext(inputs={"amount": 50.0})
        res_after = await canary_agent.decide(ctx_after)
        assert res_after.choice == "stripe_us"


class TestEndToEndClosedLoopExecution:
    """Tests full lifecycle from Ingest Data -> Train -> Simulate -> Adjust -> Seal -> Deploy."""

    @pytest.mark.asyncio
    async def test_full_closed_loop_lifecycle(self, spec, sample_records):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = ClosedLoopOrchestrator(client_id="fintech_global", output_dir=tmpdir)

            start = time.perf_counter()
            result = await orch.execute_closed_loop(
                spec=spec,
                records=sample_records,
                specialist_type="routing",
                target_field="optimal_gateway",
                rollout_mode=RolloutMode.CANARY,
                canary_percentage=20.0,
            )
            elapsed = time.perf_counter() - start

            assert isinstance(result, ClosedLoopLifecycleResult)
            assert result.status == "SUCCESS"
            assert result.phase == LifecyclePhase.COMPLETED
            assert result.duration_seconds < 25.0  # Entire loop runs in under 25 seconds!

            # Verify sub-reports
            assert result.ingestion.total_records == 60
            assert result.training_metrics is not None
            assert result.initial_simulation.total_simulated == 60
            assert result.self_adjustment is not None
            assert result.final_simulation.total_simulated == 60
            assert result.sealed_package is not None
            assert os.path.exists(result.sealed_package.package_path)
            assert result.deployment_info["rollout_mode"] == "CANARY"

            # Check JSON serialization
            json_dump = result.to_json()
            assert "fintech_global" in json_dump
            assert "AES-256-GCM" in json_dump


class TestEnterpriseSubBillionLifecycleExecution:
    """Tests enterprise tier with Tabular Transformer and Sandbox Readiness Certification."""

    @pytest.mark.asyncio
    async def test_enterprise_sub_billion_closed_loop_lifecycle(self, spec, sample_records):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = ClosedLoopOrchestrator(client_id="titan_enterprise", output_dir=tmpdir)

            start = time.perf_counter()
            result = await orch.execute_closed_loop(
                spec=spec,
                records=sample_records,
                specialist_type="routing",
                target_field="optimal_gateway",
                rollout_mode=RolloutMode.CANARY,
                canary_percentage=15.0,
                tier="sub_billion",
            )
            elapsed = time.perf_counter() - start

            assert isinstance(result, ClosedLoopLifecycleResult)
            assert result.status == "SUCCESS"
            assert result.phase == LifecyclePhase.COMPLETED
            assert result.tier == "sub_billion"
            assert elapsed < 25.0

            # Training metrics
            assert result.training_metrics["tier"] == "sub_billion"
            assert result.training_metrics["parameter_count"] > 100000

            # Readiness Certification
            assert result.readiness_certificate is not None
            cert = result.readiness_certificate
            assert cert.model_tier == "sub_billion"
            assert cert.is_certified is True
            assert cert.latency_profile.p99_latency_ms < 15.0
            assert cert.verify_audit_hash() is True

            # Simulation & Sealed Packaging
            assert result.simulation_report.total_simulated > 0
            assert result.sealed_package_metadata is not None
            assert os.path.exists(result.sealed_package_metadata.package_path)

            # JSON export
            data = result.to_dict()
            assert data["tier"] == "sub_billion"
            assert data["readiness_certificate"]["status"] == "CERTIFIED_ENTERPRISE_READY"

    def test_payment_sdm_factory_enterprise_tier(self, sample_records):
        from payment_decision_factory.factory import PaymentSDMFactory

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = PaymentSDMFactory()
            dict_records = [
                r.to_dict() if hasattr(r, "to_dict") else vars(r)
                for r in sample_records
            ]

            manifest = factory.build_client_package(
                client_name="Titan Enterprise",
                records=dict_records,
                output_dir=tmpdir,
                tier="sub_billion",
                license_key="lic_titan_enterprise_test_key_12345",
            )

            assert manifest["tier"] == "sub_billion"
            assert manifest["readiness_certificate"] is not None
            assert manifest["sla_performance"]["production_sla_threshold_ms"] == 15.0
            assert manifest["sla_performance"]["sla_passed"] is True
            assert os.path.exists(os.path.join(tmpdir, "models/routing_specialist.onnx"))
            assert os.path.exists(os.path.join(tmpdir, "models/routing_specialist.sdmpkg"))
            assert os.path.exists(os.path.join(tmpdir, "client_manifest.json"))
            assert os.path.exists(os.path.join(tmpdir, "README.md"))
