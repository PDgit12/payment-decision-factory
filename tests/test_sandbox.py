"""Test suite for Payment Decision Factory Sandbox and Mock Traffic Generator."""

import os
import pytest
from fastapi.testclient import TestClient

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.engine import DecisionEngine
from payment_decision_factory.sandbox.app import app, state, load_spec_engine
from payment_decision_factory.sandbox.mock_traffic import (
    MockTrafficGenerator,
    DecisionSimulator,
    PaymentTransaction,
    evaluate_batch,
)

SPEC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../specs/smart_routing.yaml"))

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def routing_engine():
    spec = DecisionSpec.from_yaml(SPEC_PATH)
    return DecisionEngine(spec)


class TestMockTrafficGenerator:
    """Tests for synthetic traffic generator across scenarios."""

    def test_happy_path_generation(self):
        gen = MockTrafficGenerator(seed=123)
        batch = gen.generate_happy_path(count=40)
        assert len(batch) == 40
        for tx in batch:
            assert isinstance(tx, PaymentTransaction)
            assert tx.amount >= 10.0
            assert tx.velocity_1h <= 3
            assert tx.currency in ["USD", "EUR", "GBP"]
            assert tx.scenario == "happy_path"

    def test_card_testing_generation(self):
        gen = MockTrafficGenerator(seed=123)
        batch = gen.generate_card_testing(count=30)
        assert len(batch) == 30
        for tx in batch:
            assert tx.velocity_1h >= 14
            assert tx.amount <= 5.0
            assert tx.scenario == "card_testing"
            assert "attack_vector" in tx.metadata

    def test_cross_border_psd2_generation(self):
        gen = MockTrafficGenerator(seed=123)
        batch = gen.generate_cross_border_psd2(count=25)
        assert len(batch) == 25
        for tx in batch:
            assert tx.bin_country in MockTrafficGenerator.EEA_COUNTRIES
            assert tx.amount >= 30.0
            assert tx.scenario == "cross_border_psd2"

    def test_batch_generator_dispatch(self):
        gen = MockTrafficGenerator(seed=99)
        mixed = gen.generate_batch(scenario="mixed", count=50)
        assert len(mixed) == 50
        scenarios = {tx.scenario for tx in mixed}
        assert "happy_path" in scenarios
        assert "card_testing" in scenarios
        assert "cross_border_psd2" in scenarios


class TestBatchEvaluation:
    """Tests for batch evaluation and metric calculation."""

    def test_batch_evaluation_with_engine(self, routing_engine):
        gen = MockTrafficGenerator(seed=42)
        batch = gen.generate_batch(scenario="happy_path", count=20)
        metrics = evaluate_batch(routing_engine, batch, scenario_name="happy_path", sla_target_ms=20.0)

        assert metrics.total_transactions == 20
        assert metrics.sla_adherence_pct == 100.0
        assert metrics.average_latency_ms < 20.0
        assert metrics.latency_p95_ms < 20.0
        assert metrics.average_confidence > 0.40
        assert "stripe_us" in metrics.routing_distribution
        assert sum(metrics.routing_distribution.values()) == 20

    def test_card_testing_triggers_fallback_or_challenge(self, routing_engine):
        gen = MockTrafficGenerator(seed=42)
        batch = gen.generate_card_testing(count=15)
        metrics = evaluate_batch(routing_engine, batch, scenario_name="card_testing")

        assert metrics.total_transactions == 15
        assert metrics.high_velocity_count == 15
        # Excessive velocity triggers resilient fallback
        assert metrics.routing_distribution.get("checkout_crossborder", 0) > 0


class TestSandboxApiEndpoints:
    """Tests for Sandbox FastAPI application."""

    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert data["engine_ready"] is True
        assert data["sla_target_ms"] == 20.0

    def test_index_page(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Payment Decision Factory" in res.text
        assert "Real-Time Sync" in res.text

    def test_api_spec_details(self, client):
        res = client.get("/api/spec")
        assert res.status_code == 200
        data = res.json()
        assert data["spec_id"] == "smart_routing_v1"
        assert "choices" in data
        assert "stripe_us" in data["choices"]

    def test_api_decide_normal(self, client):
        payload = {
            "amount": 49.99,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
        }
        res = client.post("/api/decide", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["choice"] == "stripe_us"
        assert data["confidence"] > 0.50
        assert data["latency_ms"] < 20.0
        assert data["sla_met"] is True
        assert "feature_contributions" in data
        assert data["circuit_breaker"]["triggered"] is False

    def test_api_decide_circuit_breaker(self, client):
        payload = {
            "amount": 1.00,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "NG",
            "mcc": "5969",
            "velocity_1h": 25,
        }
        res = client.post("/api/decide", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["circuit_breaker"]["triggered"] is True
        assert data["circuit_breaker"]["action"] == "STEP_DOWN_CHALLENGE"
        assert data["is_fallback"] is True

    def test_api_traffic_generate(self, client):
        res = client.post("/api/traffic/generate", json={"scenario": "happy_path", "count": 15})
        assert res.status_code == 200
        data = res.json()
        assert data["total_transactions"] == 15
        assert data["sla_adherence_pct"] == 100.0
        assert len(data["transactions"]) == 15

    def test_api_traffic_stream(self, client):
        res = client.get("/api/traffic/stream?scenario=cross_border_psd2&count=4")
        assert res.status_code == 200
        data = res.json()
        assert data["scenario"] == "cross_border_psd2"
        assert len(data["items"]) == 4

    def test_api_specs_list_and_switch(self, client):
        res_list = client.get("/api/specs/list")
        assert res_list.status_code == 200
        data_list = res_list.json()
        assert len(data_list["specs"]) >= 1

        # Switch to fraud_screener.yaml
        res_switch = client.post("/api/specs/switch", json={"spec_filename": "fraud_screener.yaml"})
        assert res_switch.status_code == 200
        assert res_switch.json()["status"] == "success"

        # Switch back to smart_routing.yaml
        res_back = client.post("/api/specs/switch", json={"spec_filename": "smart_routing.yaml"})
        assert res_back.status_code == 200
        assert res_back.json()["status"] == "success"

    def test_api_benchmark(self, client):
        res = client.post("/api/benchmark?samples=25")
        assert res.status_code == 200
        data = res.json()
        assert data["samples"] == 25
        assert data["latency_p50_ms"] < 20.0
        assert data["sla_adherence_pct"] == 100.0

    def test_specialists_status(self, client):
        res = client.get("/api/specialists/status")
        assert res.status_code == 200
        data = res.json()
        assert "specialists" in data
        assert "routing" in data["specialists"]
        assert "fraud" in data["specialists"]
        assert "sca" in data["specialists"]
        assert "retry" in data["specialists"]
        assert data["fleet_target"] == 4

    def test_specialists_train_routing(self, client):
        res = client.post("/api/specialists/train", json={
            "specialist_type": "routing",
            "sample_count": 60,
            "epochs": 4,
            "export_onnx": True,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "routing" in data["reports"]
        rep = data["reports"]["routing"]
        assert rep["parameter_count"] > 10000
        assert rep["parameter_count"] < 250000
        assert rep["artifact_size_kb"] < 1500.0  # < 1.5MB requirement
        assert rep["training_duration_sec"] < 15.0  # < 15s requirement

    def test_api_decide_with_sdm_mode(self, client):
        # First ensure routing SDM is trained
        client.post("/api/specialists/train", json={
            "specialist_type": "routing",
            "sample_count": 60,
            "epochs": 4,
            "export_onnx": True,
        })
        payload = {
            "amount": 49.99,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
            "use_sdm": True,
        }
        res = client.post("/api/decide", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["sdm_evaluated"] is True
        assert data["latency_ms"] < 20.0
        assert "Specialist SDM" in data["routing_reason"]

