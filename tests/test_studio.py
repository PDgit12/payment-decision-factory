"""
Unit & Integration Tests for Payment Decision Studio & Hybrid Architecture API
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.studio.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_studio_status_endpoint(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "hardware" in data
    assert data["hardware"]["chip_model"] is not None
    assert data["hardware"]["optimal_device"] in ["mps", "cuda", "cpu"]
    assert data["models_count"] >= 1


def test_studio_marketplace_and_vault(client):
    # Test models listing
    res_models = client.get("/api/marketplace/models")
    assert res_models.status_code == 200
    models = res_models.json()
    assert isinstance(models, list)
    assert len(models) >= 1
    assert "package_path" in models[0]
    assert "sha256_audit_hash" in models[0]

    # Test key generation and listing
    res_gen = client.post(
        "/api/keys/generate",
        json={"client_id": "test_merchant_corp", "label": "Stripe/Adyen Hotpath Key"},
    )
    assert res_gen.status_code == 200
    key_data = res_gen.json()
    assert key_data["client_id"] == "test_merchant_corp"
    assert key_data["license_key"].startswith("lic_test_merchant_corp_")
    assert key_data["is_active"] is True

    # Verify key is in list
    res_keys = client.get("/api/keys")
    assert res_keys.status_code == 200
    keys = res_keys.json()
    assert any(k["license_key"] == key_data["license_key"] for k in keys)


def test_studio_hybrid_evaluation_clean_transaction(client):
    """Verify clean French card is routed to Adyen EU with System 1 & 2 approval."""
    payload = {
        "amount": 120.0,
        "currency": "EUR",
        "bin_country": "FR",
        "card_brand": "visa",
        "mcc": 5411,
        "velocity_1h": 1,
    }
    res = client.post("/api/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["final_decision"] == "authorized"
    assert data["action"] == "CAPTURE"
    assert data["routed_gateway"] == "adyen_eu"
    assert data["system1_neural"]["invoked"] is True
    assert data["system1_neural"]["confidence"] > 0.5
    assert data["system2_guardrails"]["psd2_statutory_veto_applied"] is False
    assert data["economics"]["savings_vs_default_bps"] > 0


def test_studio_hybrid_evaluation_psd2_statutory_veto(client):
    """Verify high-value EUR transaction (>€500) triggers statutory PSD2 3DS challenge."""
    payload = {
        "amount": 750.0,
        "currency": "EUR",
        "bin_country": "DE",
        "card_brand": "visa",
        "mcc": 5411,
        "velocity_1h": 1,
    }
    res = client.post("/api/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["final_decision"] == "challenge_3ds_required"
    assert data["action"] == "CHALLENGE_3DS"
    assert data["system2_guardrails"]["psd2_statutory_veto_applied"] is True
    assert "Statutory Cap" in data["system2_guardrails"]["psd2_reason"]


def test_studio_hybrid_evaluation_velocity_ddos_veto(client):
    """Verify velocity spike (>=12 tx/hr) triggers immediate System 2 Pre-Auth veto."""
    payload = {
        "amount": 25.0,
        "currency": "USD",
        "bin_country": "US",
        "card_brand": "mastercard",
        "mcc": 5411,
        "velocity_1h": 15,
    }
    res = client.post("/api/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["final_decision"] == "blocked_bot_ddos"
    assert data["action"] == "DECLINE"
    assert data["system1_neural"]["invoked"] is False
    assert data["system2_guardrails"]["velocity_check_passed"] is False


def test_studio_hybrid_evaluation_stolen_card_veto(client):
    """Verify Mastercard Cat 1 stolen card (code 41) triggers 0-retry veto."""
    payload = {
        "amount": 95.0,
        "currency": "USD",
        "bin_country": "US",
        "card_brand": "mastercard",
        "mcc": 5411,
        "velocity_1h": 1,
        "error_code": 41,
    }
    res = client.post("/api/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["final_decision"] == "blocked_stolen_card"
    assert data["action"] == "DECLINE"
    assert data["system1_neural"]["invoked"] is False
    assert data["system2_guardrails"]["cat1_stolen_check_passed"] is False


def test_studio_integration_routes_generator(client):
    """Verify generation of multi-pathway integration code snippets."""
    res = client.get("/api/integration/test_model_id")
    assert res.status_code == 200
    snippets = res.json()
    assert "python" in snippets
    assert "typescript" in snippets
    assert "gateway_middleware" in snippets
    assert "SealedPackageClient" in snippets["python"]
    assert "SealedPackageClient" in snippets["typescript"]
    assert "smart_route_payment" in snippets["gateway_middleware"]
