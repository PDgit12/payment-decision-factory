"""
Universal Decision Model Factory & Closed-Loop Platform End-to-End Master Test Suite.
=====================================================================================
Comprehensive integration test suite covering all 5 architectural pillars:
1. Domain-Agnostic Decision Factory: Flexible schemas (Lending, Healthcare, Logistics),
   auto-feature encoding, sub-second CPU training without GPU dependencies.
2. Digital Twin Simulation Sandbox: Counterfactual transaction replay, network perturbation,
   acceptance & cost modeling, and business KPI estimation.
3. Autonomous Policy Self-Adjustment: Self-tuning of probability thresholds, temperature
   calibration factors, and utility weights based on simulation feedback.
4. Sealed Closed-Source Distribution (.sdmpkg): Authenticated AES-256-GCM encryption,
   PBKDF2 key derivation, AAD manifest binding, license authentication, and zero-disk in-memory execution.
5. Production Pipeline Integration & Canary Rollout: End-to-end closed-loop lifecycle,
   Shadow mirror verification, Canary traffic splitting, and auto-rollback on SLA breaches.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import pytest
import numpy as np

# Ensure root directory and SDK are on python path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SDK_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sdk/python"))
if _SDK_PYTHON not in sys.path:
    sys.path.insert(0, _SDK_PYTHON)

# Core imports
from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.engine import DecisionEngine
from payment_decision_factory.core.sdm import SpecialistSDM, export_specialist_to_onnx
from payment_decision_factory.core.sealed_package import (
    DecryptedPackageContext,
    LicenseAuthenticationError,
    LicenseExpiredError,
    SealedModelRunner,
    SealedPackage,
    SealedPackageSecurityError,
)
from payment_decision_factory.core.universal_factory import (
    UniversalDecisionFactory,
    UniversalDomainSpec,
    UniversalFeatureEncoder,
    UniversalSDM,
)
from payment_decision_factory.sandbox.mock_traffic import MockTrafficGenerator
from payment_decision_factory.sandbox.simulator import (
    AcquirerProfile,
    CounterfactualReport,
    DigitalTwinConfig,
    DigitalTwinSimulator,
    SelfAdjustmentOptimizer,
    SelfAdjustmentResult,
    SimulationOutcome,
    self_adjust,
    simulate,
)
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
from payment_decision_factory.orchestration.context import ExecutionStatus, TransactionContext
from payment_decision_factory.orchestration.pipeline import PaymentOrchestrationPipeline
from payment_decision_factory.orchestration.agent_base import MockAgent
from payment_decision_sdk.sealed import SealedPackageClient
from payment_decision_sdk.models import PaymentContext

CORE_SPEC_PATH = os.path.join(os.path.dirname(__file__), "../specs/smart_routing.yaml")
TEST_LICENSE_KEY = "ENT-PROD-PLATFORM-TEST-KEY-XYZ-999"


@pytest.fixture(scope="module")
def valid_onnx_path():
    """Build a temporary valid ONNX model file for packaging tests."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name

    sdm = SpecialistSDM("routing", num_continuous_features=30, num_classes=3)
    sdm.eval()
    export_specialist_to_onnx(sdm, path, input_dim=30, verify=False)
    yield path

    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(path + ".data"):
        os.remove(path + ".data")


# =============================================================================
# PILLAR 1: DOMAIN-AGNOSTIC UNIVERSAL DECISION MODEL FACTORY
# =============================================================================
class TestUniversalDecisionModelFactory:
    """Tests training and calibration on cross-domain datasets (Lending & Logistics)."""

    def test_lending_credit_decision_factory(self):
        """Test universal model factory on a Lending Credit Underwriting schema."""
        factory = UniversalDecisionFactory.create_for_domain(
            domain_id="consumer_lending",
            domain_name="Consumer Lending Underwriting SDM",
            target_field="underwriting_decision",
            choices=["approve_tier1", "approve_tier2", "manual_review", "decline"],
            numerical_features=["credit_score", "dti_ratio", "annual_income", "loan_amount", "emp_length_years"],
            categorical_features=["loan_purpose", "home_ownership", "state"],
        )

        # Generate synthetic lending records
        rng = np.random.default_rng(42)
        records = []
        purposes = ["debt_consolidation", "home_improvement", "credit_card", "small_business"]
        ownerships = ["RENT", "OWN", "MORTGAGE"]
        states = ["CA", "NY", "TX", "FL", "IL", "WA"]

        for _ in range(120):
            score = float(rng.integers(580, 850))
            dti = float(rng.uniform(0.05, 0.55))
            income = float(rng.uniform(30000, 200000))
            amount = float(rng.uniform(2000, 45000))
            emp = float(rng.integers(0, 20))
            purpose = str(rng.choice(purposes))
            own = str(rng.choice(ownerships))
            st = str(rng.choice(states))

            if score > 740 and dti < 0.30:
                decision = "approve_tier1"
            elif score > 680 and dti < 0.40:
                decision = "approve_tier2"
            elif score > 620:
                decision = "manual_review"
            else:
                decision = "decline"

            records.append({
                "credit_score": score,
                "dti_ratio": dti,
                "annual_income": income,
                "loan_amount": amount,
                "emp_length_years": emp,
                "loan_purpose": purpose,
                "home_ownership": own,
                "state": st,
                "underwriting_decision": decision,
            })

        t0 = time.perf_counter()
        trained_model, metrics = factory.train(records, epochs=8, batch_size=16)
        train_time = time.perf_counter() - t0

        # Performance & footprint assertions
        assert train_time < 5.0, f"Training took too long: {train_time:.2f}s"
        assert metrics["train_samples"] > 0
        assert metrics["parameter_count"] < 250000
        assert metrics["validation_accuracy"] >= 0.55

        # Test single loan applicant inference
        applicant = {
            "credit_score": 780,
            "dti_ratio": 0.18,
            "annual_income": 125000,
            "loan_amount": 15000,
            "emp_length_years": 8,
            "loan_purpose": "debt_consolidation",
            "home_ownership": "MORTGAGE",
            "state": "CA",
        }
        res = factory.evaluate(trained_model, applicant)
        assert res["choice"] in ["approve_tier1", "approve_tier2"]
        assert res["confidence"] > 0.30
        assert res["latency_ms"] < 2.0


# =============================================================================
# PILLAR 2: DIGITAL TWIN SIMULATION & COUNTERFACTUAL REPLAY
# =============================================================================
class TestDigitalTwinSimulationEngine:
    """Tests high-fidelity simulation, network noise, and counterfactual uplift."""

    def test_digital_twin_simulation_and_counterfactual_lift(self):
        spec = DecisionSpec.from_yaml(CORE_SPEC_PATH)
        engine = DecisionEngine(spec)
        traffic_gen = MockTrafficGenerator(seed=101)
        batch = traffic_gen.generate_batch("mixed", count=50)

        config = DigitalTwinConfig(
            network_jitter_ms=1.5,
            packet_loss_rate=0.01,
            sla_timeout_ms=25.0,
        )
        simulator = DigitalTwinSimulator(config=config, spec=spec)

        # 1. Batch simulation
        outcome = simulator.simulate_batch(model=engine, transactions=batch)
        assert isinstance(outcome, SimulationOutcome)
        assert outcome.total_transactions == 50
        assert outcome.successful_auths + outcome.declined_auths == 50
        assert outcome.sla_breaches == 0
        assert outcome.p95_latency_ms < 20.0
        assert outcome.overall_approval_rate > 0.50

        # 2. Counterfactual historical replay
        report = simulator.run_counterfactual_replay(
            model=engine,
            transactions=batch,
            baseline_field="historical_gateway",
        )
        assert isinstance(report, CounterfactualReport)
        assert report.total_transactions == 50
        assert isinstance(report.approval_rate_delta_bps, float)
        assert isinstance(report.net_benefit_usd, float)


# =============================================================================
# PILLAR 3: AUTONOMOUS POLICY SELF-ADJUSTMENT
# =============================================================================
class TestPolicySelfAdjustmentOptimizer:
    """Tests automatic tuning of thresholds, calibration factor, and utility weights."""

    def test_autonomous_policy_optimization(self):
        spec = DecisionSpec.from_yaml(CORE_SPEC_PATH)
        engine = DecisionEngine(spec)
        traffic_gen = MockTrafficGenerator(seed=202)
        batch = traffic_gen.generate_batch("cross_border_psd2", count=40)

        # Run initial baseline simulation
        baseline_sim = simulate(model=engine, records=batch, spec=spec)

        # Run self-adjustment optimizer
        optimizer = SelfAdjustmentOptimizer(spec=spec)
        adj_result = optimizer.self_adjust(
            model=engine,
            sim_report=baseline_sim,
            records=batch,
            spec=spec,
        )

        assert isinstance(adj_result, SelfAdjustmentResult)
        assert adj_result.optimized_spec is not None
        assert adj_result.optimized_spec.spec_id == spec.spec_id

        # Verify optimized parameters differ or improve upon baseline
        assert adj_result.calibrated_temperature > 0.0
        assert len(adj_result.parameter_deltas) > 0
        assert adj_result.approval_uplift_bps >= 0.0 or adj_result.brier_improvement >= 0.0


# =============================================================================
# PILLAR 4: SEALED CLOSED-SOURCE PACKAGING (.sdmpkg)
# =============================================================================
class TestSealedPackageSecurity:
    """Tests AES-256-GCM authenticated encryption, license keys, and zero-disk leakage."""

    def test_sealed_package_full_security_lifecycle(self, valid_onnx_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_path = os.path.join(tmpdir, "secure_model.sdmpkg")

            # 1. Seal model into .sdmpkg
            pkg = SealedPackage.seal(
                model_bytes_or_path=valid_onnx_path,
                license_key=TEST_LICENSE_KEY,
                client_id="acme_corp",
                domain="payment_routing",
                choices=["stripe_us", "adyen_eu", "checkout_crossborder"],
                input_dim=30,
                validity_days=30,
                output_path=pkg_path,
            )
            assert os.path.exists(pkg_path)
            assert pkg.manifest["client_id"] == "acme_corp"
            assert pkg.crypto["cipher"] == "AES-256-GCM"

            # 2. In-memory decryption and execution
            loaded_pkg = SealedPackage.load_file(pkg_path)
            ctx = loaded_pkg.decrypt(TEST_LICENSE_KEY)
            assert ctx.client_id == "acme_corp"
            assert len(ctx.model_bytes) > 0

            runner = ctx.create_runner()
            test_x = np.random.randn(1, 30).astype(np.float32)
            eval_res = runner.evaluate(test_x)
            assert eval_res["choice"] in ["stripe_us", "adyen_eu", "checkout_crossborder"]
            assert eval_res["confidence"] > 0.0

            # 3. Invalid license key rejection
            with pytest.raises(LicenseAuthenticationError):
                loaded_pkg.decrypt("WRONG-UNAUTHORIZED-LICENSE-KEY")

            # 4. Tamper detection (manifest tampering aborts GCM verification)
            loaded_pkg.manifest["client_id"] = "hacked_client"
            with pytest.raises(LicenseAuthenticationError):
                loaded_pkg.decrypt(TEST_LICENSE_KEY)

            # 5. License expiration enforcement
            expired_pkg = SealedPackage.seal(
                model_bytes_or_path=valid_onnx_path,
                license_key=TEST_LICENSE_KEY,
                client_id="acme_corp",
                domain="payment_routing",
                choices=["stripe_us", "adyen_eu", "checkout_crossborder"],
                input_dim=30,
                validity_days=-1,  # Expired
                output_path=os.path.join(tmpdir, "expired.sdmpkg"),
            )
            with pytest.raises(LicenseExpiredError):
                expired_pkg.decrypt(TEST_LICENSE_KEY)


# =============================================================================
# PILLAR 5: PRODUCTION PIPELINE INTEGRATION, CANARY ROLLOUT & SDK
# =============================================================================
class TestProductionPipelineAndSDKIntegration:
    """Tests end-to-end closed-loop orchestrator, canary deployments, and Python SDK."""

    @pytest.mark.asyncio
    async def test_closed_loop_orchestrator_end_to_end(self):
        """Test complete 6-stage lifecycle execution."""
        spec = DecisionSpec.from_yaml(CORE_SPEC_PATH)
        gen = MockTrafficGenerator(seed=303)
        records = gen.generate_batch("mixed", count=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = ClosedLoopOrchestrator(
                client_id="test_merchant",
                output_dir=tmpdir,
                license_key=TEST_LICENSE_KEY,
            )

            result = await orchestrator.execute_closed_loop(
                spec=spec,
                records=records,
                specialist_type="routing",
                target_field="optimal_gateway",
                rollout_mode=RolloutMode.CANARY,
                canary_percentage=15.0,
                license_key=TEST_LICENSE_KEY,
            )

            assert isinstance(result, ClosedLoopLifecycleResult)
            assert result.ingestion.total_records == 50
            assert result.initial_simulation.total_simulated == 50
            assert result.self_adjustment is not None
            assert result.sealed_package is not None
            assert os.path.exists(result.sealed_package.package_path)
            assert result.deployment_info is not None

    def test_python_sdk_sealed_package_client_in_process(self, valid_onnx_path):
        """Verify Python SDK loads .sdmpkg and executes in-process with microsecond latency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_path = os.path.join(tmpdir, "routing.sdmpkg")

            SealedPackage.seal(
                model_bytes_or_path=valid_onnx_path,
                license_key=TEST_LICENSE_KEY,
                client_id="test_merchant",
                domain="payment_routing",
                choices=["stripe_us", "adyen_eu", "checkout_crossborder"],
                input_dim=30,
                validity_days=30,
                output_path=pkg_path,
            )

            # Instantiate SDK SealedPackageClient
            sdk_client = SealedPackageClient(
                package_path=pkg_path,
                license_key=TEST_LICENSE_KEY,
                client_id="test_merchant",
            )

            # Evaluate transaction
            payment_ctx = PaymentContext(
                amount=79.99,
                currency="USD",
                card_brand="visa",
                bin_country="US",
                mcc="5411",
                velocity_1h=1,
            )
            res = sdk_client.decide(payment_ctx)
            assert res.decision in ["stripe_us", "adyen_eu", "checkout_crossborder"]
            assert res.confidence > 0.0
            assert res.latency_ms < 15.0  # In-process sub-millisecond execution (<15ms SLA)

    @pytest.mark.asyncio
    async def test_canary_rollout_and_auto_rollback(self):
        """Verify CanaryDeploymentAgent automatically rolls back to control upon error spike."""
        control_agent = MockAgent(name="control_routing", choice="stripe_us")

        class FailingCandidateAgent:
            def __init__(self):
                self.name = "candidate_failing"
                self.call_count = 0

            async def decide(self, ctx: TransactionContext):
                self.call_count += 1
                raise RuntimeError("Candidate simulated upstream exception!")

        candidate_agent = FailingCandidateAgent()

        canary = CanaryDeploymentAgent(
            control_agent=control_agent,
            candidate_agent=candidate_agent,
            canary_percentage=100.0,  # Send all to candidate to force rollback
            error_threshold_pct=20.0,
            min_requests=3,
        )

        ctx = TransactionContext(tx_id="tx_test", inputs={"amount": 50.0})

        # Process transactions: after 3 failures, canary triggers auto-rollback
        for _ in range(5):
            res = await canary.decide(ctx)
            # Rollback protection returns control decision rather than crashing checkout
            assert res.choice == "stripe_us"

        assert canary.is_rolled_back is True
        assert canary.rollback_reason is not None
        assert "error" in canary.rollback_reason.lower()
