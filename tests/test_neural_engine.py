"""
Tests for LayaNeuralEngine (System 1 Decision Engine Integration)
"""

import os
import pytest
from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.neural_engine import LayaNeuralEngine, create_neural_engine

SPEC_PATH = os.path.join(
    os.path.dirname(__file__), "../specs/smart_routing.yaml"
)


class TestLayaNeuralEngine:
    def test_initialization_from_spec_path(self):
        engine = create_neural_engine(SPEC_PATH)
        assert engine.spec.spec_id == "smart_routing_v1"
        assert engine.model_id == "convaiinnovations/laya"
        assert engine.subfolder == "typed-decisions"
        assert "routing_choice" in engine._questions
        assert "stripe_us" in engine._questions["routing_choice"]["criteria"]
        assert "adyen_eu" in engine._questions["routing_choice"]["criteria"]

    def test_decision_evaluation_happy_path_us(self):
        engine = create_neural_engine(SPEC_PATH)
        payload = {
            "amount": 49.99,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": 5411,
            "velocity_1h": 1,
        }
        result = engine.decide(payload)
        assert result.choice in ["stripe_us", "adyen_eu", "checkout_crossborder"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0.0

    def test_decision_evaluation_european_tx(self):
        engine = create_neural_engine(SPEC_PATH)
        payload = {
            "amount": 120.00,
            "currency": "EUR",
            "card_brand": "mastercard",
            "bin_country": "DE",
            "mcc": 5732,
            "velocity_1h": 2,
        }
        result = engine.decide(payload)
        assert result.choice == "adyen_eu"
        assert result.confidence > 0.50

    def test_dynamic_question_generation(self):
        engine = create_neural_engine(SPEC_PATH)
        q = engine._questions["routing_choice"]
        assert q["type"] == "choice"
        assert "stripe_us" in q["criteria"]
        assert "USD" in q["criteria"]["stripe_us"]
        assert "EUR" in q["criteria"]["adyen_eu"]
