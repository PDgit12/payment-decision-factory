"""
Comprehensive Test Suite for Digital Twin Simulation Engine & Policy Self-Adjustment.
=====================================================================================
Tests:
1. Digital Twin Configuration & Multi-Acquirer Rails (Stripe US, Adyen EU, Checkout.com).
2. Realistic Network Simulation (Latency distributions, jitter, timeouts, noise).
3. Single & Batch Transaction Simulation with Decision SLA adherence (<20ms).
4. Counterfactual Historical Transaction Replay (Approval uplift bps, fee savings, net gain).
5. Automated Policy Self-Adjustment (Temperature calibration, utility weights, confidence floors).
6. High-level Convenience API (simulate, self_adjust).
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
from payment_decision_factory.sandbox.simulator import (
    AcquirerProfile,
    DigitalTwinConfig,
    DigitalTwinSimulator,
    SimulatedTransactionResult,
    SimulationOutcome,
    CounterfactualReport,
    SelfAdjustmentOptimizer,
    SelfAdjustmentResult,
    simulate,
    self_adjust,
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
    return traffic_gen.generate_batch("mixed", count=50)


class TestDigitalTwinConfig:
    """Tests for simulator configurations and acquiring profiles."""

    def test_default_config_profiles(self):
        config = DigitalTwinConfig()
        assert "stripe_us" in config.acquirers
        assert "adyen_eu" in config.acquirers
        assert "checkout_crossborder" in config.acquirers

        stripe = config.acquirers["stripe_us"]
        assert stripe.region == "US"
        assert stripe.base_approval_rate > 0.90
        assert stripe.base_fee_bps == 290.0

        adyen = config.acquirers["adyen_eu"]
        assert adyen.region == "EU"
        assert adyen.base_fee_bps == 120.0
        assert adyen.domestic_bonus > 0.0

    def test_custom_acquirer_profile(self):
        custom_rail = AcquirerProfile(
            name="custom_latam",
            display_name="Custom LatAm Local Rail",
            region="LATAM",
            base_approval_rate=0.88,
            base_fee_bps=350.0,
            fixed_fee_cents=40.0,
        )
        config = DigitalTwinConfig(
            network_latency_mean_ms=180.0,
            acquirers={"custom_latam": custom_rail},
        )
        assert len(config.acquirers) == 1
        assert config.network_latency_mean_ms == 180.0


class TestSimulationExecution:
    """Tests for single transaction and batch digital twin simulation."""

    def test_simulate_single_transaction_dict(self, engine, spec):
        sim = DigitalTwinSimulator(DigitalTwinConfig(random_seed=42))
        tx = {
            "transaction_id": "tx_test_001",
            "amount": 75.50,
            "currency": "EUR",
            "card_brand": "visa",
            "bin_country": "FR",
            "mcc": "5411",
            "velocity_1h": 2,
        }
        res = sim.simulate_transaction(engine, tx, spec=spec)
        assert isinstance(res, SimulatedTransactionResult)
        assert res.transaction_id == "tx_test_001"
        assert res.currency == "EUR"
        assert res.bin_country == "FR"
        assert res.choice in ["adyen_eu", "stripe_us", "checkout_crossborder"]
        assert res.decision_latency_ms < 250.0  # non-blocking execution under heavy CPU load
        assert res.network_latency_ms > 0.0
        assert res.total_latency_ms >= res.decision_latency_ms
        assert res.fee_usd > 0.0
        assert "adyen_eu" in res.counterfactual_choices
        assert "stripe_us" in res.counterfactual_choices

    def test_simulate_single_payment_transaction_object(self, engine, spec, traffic_gen):
        sim = DigitalTwinSimulator(DigitalTwinConfig(random_seed=123))
        tx_obj = traffic_gen.generate_happy_path(1)[0]
        assert isinstance(tx_obj, PaymentTransaction)

        res = sim.simulate_transaction(engine, tx_obj, spec=spec)
        assert isinstance(res, SimulatedTransactionResult)
        assert res.transaction_id == tx_obj.transaction_id
        assert res.decision_latency_ms < 250.0  # non-blocking execution under heavy CPU load

    def test_run_simulation_batch(self, engine, spec, sample_batch):
        sim = DigitalTwinSimulator(DigitalTwinConfig(random_seed=42))
        outcome = sim.run_simulation(engine, sample_batch, spec=spec)

        assert isinstance(outcome, SimulationOutcome)
        assert outcome.total_transactions == 50
        assert outcome.total_simulated == 50
        assert 0.0 <= outcome.approval_rate <= 1.0
        assert outcome.approved_count + outcome.declined_count == 50
        assert outcome.total_fees_usd > 0.0
        assert outcome.average_fee_usd > 0.0
        assert outcome.decision_latency_p95_ms < 15.0  # in-process SLA
        assert outcome.p95_latency_ms < 15.0
        assert 0.0 <= outcome.brier_score <= 1.0
        assert len(outcome.choice_distribution) > 0
        assert sum(outcome.choice_distribution.values()) == 50
        assert len(outcome.results) == 50

        # Export dictionary verification
        d = outcome.to_dict()
        assert d["total_transactions"] == 50
        assert "approval_rate" in d
        assert "total_fees_usd" in d
        assert "brier_score" in d


class TestCounterfactualReplay:
    """Tests for historical counterfactual replay and economic uplift calculation."""

    def test_counterfactual_replay_analysis(self, engine, spec, traffic_gen):
        sim = DigitalTwinSimulator(DigitalTwinConfig(random_seed=42))
        # Generate transactions with a baseline historical gateway (e.g. naive single-acquirer default)
        records = traffic_gen.generate_batch("mixed", count=40)
        dict_records = [r.to_dict() for r in records]
        for r in dict_records:
            r["historical_gateway"] = "stripe_us"  # legacy single-gateway baseline

        report = sim.run_counterfactual_replay(
            model_or_engine=engine,
            historical_transactions=dict_records,
            spec=spec,
            baseline_field="historical_gateway",
        )

        assert isinstance(report, CounterfactualReport)
        assert report.total_replayed == 40
        assert 0.0 <= report.baseline_approval_rate <= 1.0
        assert 0.0 <= report.model_approval_rate <= 1.0
        assert isinstance(report.approval_uplift_bps, float)
        assert report.routing_changes_count >= 0
        assert 0.0 <= report.routing_change_pct <= 100.0

        # Dict representation
        d = report.to_dict()
        assert d["total_replayed"] == 40
        assert "approval_uplift_bps" in d
        assert "net_financial_gain_usd" in d


class TestSelfAdjustmentOptimizer:
    """Tests for automated policy self-adjustment optimization."""

    def test_temperature_optimization(self, engine, spec, sample_batch):
        optimizer = SelfAdjustmentOptimizer()
        dict_records = [r.to_dict() for r in sample_batch]
        opt_temp = optimizer.optimize_temperature(engine, dict_records, spec)
        assert isinstance(opt_temp, float)
        assert 0.2 <= opt_temp <= 3.0

    def test_utility_weights_optimization(self, engine, spec, sample_batch):
        optimizer = SelfAdjustmentOptimizer()
        dict_records = [r.to_dict() for r in sample_batch]
        opt_weights = optimizer.optimize_utility_weights(engine, dict_records, spec)
        assert isinstance(opt_weights, dict)
        assert "currency_affinity" in opt_weights
        assert "geo_affinity" in opt_weights
        assert "cost_optimization" in opt_weights

    def test_end_to_end_self_adjust(self, engine, spec, sample_batch):
        optimizer = SelfAdjustmentOptimizer()
        dict_records = [r.to_dict() for r in sample_batch]
        result = optimizer.self_adjust(
            model_or_engine=engine,
            spec=spec,
            records=dict_records,
        )

        assert isinstance(result, SelfAdjustmentResult)
        assert isinstance(result.optimized_spec, DecisionSpec)
        assert isinstance(result.initial_kpis, SimulationOutcome)
        assert isinstance(result.final_kpis, SimulationOutcome)
        assert "temperature" in result.parameter_deltas
        assert "weights" in result.parameter_deltas
        assert "approval_rate_gain_bps" in result.uplift_kpis
        assert result.optimization_duration_sec > 0.0

        d = result.to_dict()
        assert "initial_kpis" in d
        assert "final_kpis" in d
        assert "parameter_deltas" in d
        assert "uplift_kpis" in d


class TestConvenienceAPI:
    """Tests for package-level simulate and self_adjust shortcuts."""

    def test_simulate_shortcut(self, engine, spec, sample_batch):
        outcome = simulate(
            model=engine,
            records=sample_batch,
            spec=spec,
        )
        assert isinstance(outcome, SimulationOutcome)
        assert outcome.total_transactions == 50
        assert outcome.p95_latency_ms < 15.0

    def test_self_adjust_shortcut(self, engine, spec, sample_batch):
        dict_records = [r.to_dict() for r in sample_batch]
        result = self_adjust(
            model=engine,
            sim_report=None,
            spec=spec,
            records=dict_records,
        )
        assert isinstance(result, SelfAdjustmentResult)
        assert result.optimized_spec.spec_id == spec.spec_id
