"""
Comprehensive Unit & Integration Test Suite for Payment Decision Factory Core
=============================================================================
Tests:
- DecisionSpec parsing, schema verification, and input validation
- PaymentSerializer semantic transformation, vectors, and prompts
- DecisionEngine routing logic, 'choice', 'score', and 'noul' primitives
- Temperature-scaled calibration, Brier scoring, and ECE
- DecisionFactory compilation, bundle export, and artifact loading
- Strict SLA latency benchmarking (<20ms)
- SDK, Sandbox, and Orchestrator interface compatibility
"""

import os
import sys

# Ensure root directory is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import shutil
import tempfile
import time
import pytest
import numpy as np

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.serializer import PaymentSerializer
from payment_decision_factory.core.engine import DecisionEngine, DecisionResult
from payment_decision_factory.core.factory import DecisionFactory

SPEC_PATH = os.path.join(os.path.dirname(__file__), "../specs/smart_routing.yaml")


class TestDecisionSpec:
    """Tests for spec loading, validation, and schema enforcement."""

    def test_load_valid_spec(self):
        spec = DecisionSpec.from_yaml(SPEC_PATH)
        assert spec.spec_id == "smart_routing_v1"
        assert spec.name == "Smart Payment Gateway Routing Specification"
        assert spec.sla.max_latency_ms == 25.0
        assert "stripe_us" in spec.choices
        assert "adyen_eu" in spec.choices
        assert "checkout_crossborder" in spec.choices

    def test_schema_input_validation(self):
        spec = DecisionSpec.from_yaml(SPEC_PATH)

        # Valid payload
        valid_payload = {
            "amount": 49.99,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 2,
        }
        cleaned = spec.validate_input(valid_payload)
        assert cleaned["amount"] == 49.99
        assert cleaned["currency"] == "USD"
        assert cleaned["card_brand"] == "visa"

        # Missing required field
        invalid_missing = valid_payload.copy()
        del invalid_missing["amount"]
        with pytest.raises(ValueError, match="Missing required field 'amount'"):
            spec.validate_input(invalid_missing)

        # Invalid currency enum
        invalid_curr = valid_payload.copy()
        invalid_curr["currency"] = "XYZ_INVALID"
        with pytest.raises(ValueError, match="not in allowed enum"):
            spec.validate_input(invalid_curr)

        # Invalid country regex
        invalid_country = valid_payload.copy()
        invalid_country["bin_country"] = "USA_TOO_LONG"
        with pytest.raises(ValueError, match="does not match regex"):
            spec.validate_input(invalid_country)

        # Amount out of range
        invalid_amount = valid_payload.copy()
        invalid_amount["amount"] = -10.0
        with pytest.raises(ValueError, match="below minimum"):
            spec.validate_input(invalid_amount)


class TestPaymentSerializer:
    """Tests for semantic state serialization and vector transformation."""

    @pytest.fixture
    def serializer(self):
        spec = DecisionSpec.from_yaml(SPEC_PATH)
        return PaymentSerializer(spec)

    def test_us_domestic_serialization(self, serializer):
        payload = {
            "amount": 55.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
        }
        state = serializer.serialize(payload)
        assert state.corridor == "DOMESTIC_US"
        assert state.region == "US"
        assert state.amount_tier == "SMALL"
        assert state.velocity_tier == "NORMAL"
        assert state.risk_profile == "LOW"
        assert "[PAYMENT_STATE]" in state.semantic_prompt
        assert "DOMESTIC_US" in state.semantic_prompt
        assert len(state.feature_vector) == 30
        assert isinstance(state.vector_array, np.ndarray)

    def test_eu_intra_serialization(self, serializer):
        payload = {
            "amount": 2500.0,
            "currency": "EUR",
            "card_brand": "mastercard",
            "bin_country": "DE",
            "mcc": "5734",
            "velocity_1h": 0,
        }
        state = serializer.serialize(payload)
        assert state.corridor == "INTRA_EU"
        assert state.region == "EU"
        assert state.amount_tier == "HIGH"
        assert state.mcc_category == "DIGITAL_SAAS"

    def test_cross_border_serialization(self, serializer):
        payload = {
            "amount": 800.0,
            "currency": "JPY",
            "card_brand": "jcb",
            "bin_country": "JP",
            "mcc": "7011",
            "velocity_1h": 1,
        }
        state = serializer.serialize(payload)
        assert state.corridor in ("INTRA_APAC", "CROSS_BORDER")
        assert state.region == "APAC"

    def test_batch_serialization(self, serializer):
        payloads = [
            {"amount": 10.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411", "velocity_1h": 0},
            {"amount": 20.0, "currency": "EUR", "card_brand": "mastercard", "bin_country": "FR", "mcc": "5411", "velocity_1h": 1},
        ]
        states = serializer.batch_serialize(payloads)
        assert len(states) == 2
        assert states[0].currency == "USD"
        assert states[1].currency == "EUR"


class TestDecisionEngine:
    """Tests for routing logic, primitives (choice, score, noul), and calibration."""

    @pytest.fixture
    def engine(self):
        spec = DecisionSpec.from_yaml(SPEC_PATH)
        return DecisionEngine(spec)

    def test_us_domestic_routing(self, engine):
        payload = {
            "amount": 42.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
        }
        res = engine.evaluate(payload)
        assert res.choice == "stripe_us"
        assert res.confidence > 0.50
        assert res.probabilities["stripe_us"] > res.probabilities["adyen_eu"]
        assert not res.is_fallback
        assert not res.noul.triggered
        assert res.latency_ms > 0
        assert len(res.feature_breakdown) == 4

    def test_eu_direct_routing(self, engine):
        payload = {
            "amount": 180.0,
            "currency": "EUR",
            "card_brand": "mastercard",
            "bin_country": "DE",
            "mcc": "5812",
            "velocity_1h": 0,
        }
        res = engine.evaluate(payload)
        assert res.choice == "adyen_eu"
        assert res.confidence > 0.50
        assert res.probabilities["adyen_eu"] > res.probabilities["stripe_us"]

    def test_cross_border_routing(self, engine):
        payload = {
            "amount": 1500.0,
            "currency": "JPY",
            "card_brand": "jcb",
            "bin_country": "JP",
            "mcc": "5734",
            "velocity_1h": 1,
        }
        res = engine.evaluate(payload)
        assert res.choice == "checkout_crossborder"
        assert res.confidence > 0.50

    def test_noul_card_testing_velocity_trigger(self, engine):
        payload = {
            "amount": 12.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 16,  # exceeds threshold 15
        }
        res = engine.evaluate(payload)
        assert res.noul.triggered is True
        assert res.noul.action == "STEP_DOWN_CHALLENGE"
        assert res.is_fallback is True
        assert "velocity" in res.routing_reason.lower()

    def test_noul_high_value_burst_trigger(self, engine):
        payload = {
            "amount": 35000.0,  # exceeds 25,000
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 8,   # exceeds 6
        }
        res = engine.evaluate(payload)
        assert res.noul.triggered is True
        assert res.noul.action == "MANUAL_REVIEW"
        assert res.is_fallback is True

    def test_calibration_and_brier_scoring(self, engine):
        probs = [
            {"stripe_us": 0.8, "adyen_eu": 0.1, "checkout_crossborder": 0.1},
            {"stripe_us": 0.1, "adyen_eu": 0.8, "checkout_crossborder": 0.1},
        ]
        truth = ["stripe_us", "adyen_eu"]
        brier = engine.compute_brier_score(probs, truth)
        assert brier < 0.10  # Very well calibrated

    def test_low_latency_sla_compliance(self, engine):
        payload = {
            "amount": 100.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
        }
        latencies = []
        for _ in range(200):
            t0 = time.perf_counter()
            engine.evaluate(payload)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        p99 = np.percentile(latencies, 99)
        avg = np.mean(latencies)
        assert p99 < 20.0, f"p99 latency {p99}ms exceeded 20ms SLA"
        assert avg < 2.0, f"avg latency {avg}ms exceeded 2ms target"

    def test_interface_compatibility(self, engine):
        """Verify seamless compatibility with SDK, Sandbox, and Orchestrator contracts."""
        payload = {
            "amount": 99.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 0,
        }
        res = engine.evaluate(payload)

        # Attribute access
        assert isinstance(res.choice, str)
        assert isinstance(res.decision, str)
        assert res.choice == res.decision
        assert isinstance(res.confidence, float)
        assert isinstance(res.probabilities, dict)
        assert isinstance(res.feature_contributions, dict)
        assert isinstance(res.latency_ms, float)
        assert isinstance(res.is_fallback, bool)

        # Dict-like indexing
        assert res["choice"] == res.choice
        assert res["decision"] == res.choice
        assert res["confidence"] == res.confidence
        assert res["probabilities"] == res.probabilities

        # Safe get method
        assert res.get("choice") == res.choice
        assert res.get("non_existent", "default") == "default"

        # to_dict conversion
        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["decision"] == res.choice


class TestDecisionFactory:
    """Tests for pipeline compilation and artifact bundle loading."""

    def test_compilation_and_bundle_loading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            factory = DecisionFactory.from_yaml_file(SPEC_PATH)
            bundle = factory.compile(output_dir=tmp_dir, create_archive=True)

            assert os.path.exists(os.path.join(tmp_dir, "spec.yaml"))
            assert os.path.exists(os.path.join(tmp_dir, "manifest.json"))
            assert os.path.exists(os.path.join(tmp_dir, "benchmark_report.json"))
            assert os.path.exists(os.path.join(tmp_dir, "bundle.json"))
            assert bundle.benchmark.sla_passed is True

            # Load compiled bundle back into an engine
            loaded_engine = DecisionFactory.load_bundle(tmp_dir)
            assert isinstance(loaded_engine, DecisionEngine)

            # Evaluate transaction on loaded engine
            payload = {
                "amount": 100.0,
                "currency": "USD",
                "card_brand": "visa",
                "bin_country": "US",
                "mcc": "5411",
                "velocity_1h": 0,
            }
            res = loaded_engine.evaluate(payload)
            assert res.choice == "stripe_us"
            assert res.confidence > 0.50
