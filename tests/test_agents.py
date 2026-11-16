"""
Unit and Integration Tests for Specialized Payment Decision Agents.
===================================================================
Tests FraudScreenerAgent, Dynamic3DSAgent, SmartRetryAgent, and SmartRoutingAgent
under synchronous, asynchronous, and DAG context execution.
"""

import asyncio
import pytest

from payment_decision_factory.agents import (
    AgentDecisionResult,
    Dynamic3DSAgent,
    FraudScreenerAgent,
    SmartRetryAgent,
    SmartRoutingAgent,
    UniversalDecisionAgent,
    create_agent,
)
from payment_decision_factory.orchestration.context import TransactionContext


# =============================================================================
# FRAUD SCREENER TESTS
# =============================================================================

def test_fraud_screener_happy_path_approval():
    agent = FraudScreenerAgent()
    payload = {
        "amount": 45.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
        "ip_country": "US",
        "mcc": "5411",
        "velocity_1h": 1,
        "is_anonymous_ip": False,
        "cvv_result": "match",
        "avs_code": "match",
    }
    result = agent.decide_sync(payload)

    assert result.choice == "allow"
    assert result.decision == "allow"
    assert result.score < 0.25
    assert result.confidence >= 0.70
    assert "velocity_1h" in result.feature_contributions
    assert result.probabilities["allow"] > result.probabilities["decline"]
    assert result.latency_ms < 15.0


def test_fraud_screener_velocity_circuit_breaker():
    agent = FraudScreenerAgent()
    payload = {
        "amount": 25.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
        "ip_country": "US",
        "mcc": "5411",
        "velocity_1h": 15,  # Bot attack
        "cvv_result": "match",
    }
    result = agent.decide_sync(payload)

    assert result.choice == "decline"
    assert result.score >= 0.90
    assert result.confidence >= 0.95
    assert "circuit breaker" in result.routing_reason.lower()


def test_fraud_screener_cvv_hard_mismatch():
    agent = FraudScreenerAgent()
    payload = {
        "amount": 50.0,
        "currency": "USD",
        "card_brand": "mastercard",
        "bin_country": "US",
        "ip_country": "US",
        "mcc": "5732",
        "velocity_1h": 1,
        "cvv_result": "no_match",
    }
    result = agent.decide_sync(payload)

    assert result.choice == "decline"
    assert result.score >= 0.90
    assert "CVV" in result.routing_reason


def test_fraud_screener_review_tier():
    agent = FraudScreenerAgent()
    payload = {
        "amount": 350.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
        "ip_country": "GB",  # Geo mismatch
        "mcc": "5411",
        "velocity_1h": 5,    # Elevated velocity
        "is_anonymous_ip": False,
        "cvv_result": "match",
    }
    result = agent.decide_sync(payload)

    assert result.choice == "review"
    assert 0.25 <= result.score < 0.70


# =============================================================================
# DYNAMIC 3DS / SCA TESTS
# =============================================================================

def test_dynamic_3ds_low_value_exemption():
    agent = Dynamic3DSAgent()
    payload = {
        "amount": 20.0,
        "currency": "EUR",
        "bin_country": "DE",
        "merchant_country": "IE",
        "card_brand": "visa",
        "fraud_score": 0.05,
    }
    result = agent.decide_sync(payload)

    assert result.choice == "low_value"
    assert result.confidence >= 0.90
    assert "Article 16" in result.routing_reason


def test_dynamic_3ds_tra_exemption_tier_1():
    agent = Dynamic3DSAgent()
    payload = {
        "amount": 80.0,
        "currency": "EUR",
        "bin_country": "FR",
        "merchant_country": "IE",
        "card_brand": "mastercard",
        "fraud_score": 0.08,
        "acquirer_tra_rate": 0.0005,
    }
    result = agent.decide_sync(payload)

    assert result.choice == "tra_exemption"
    assert result.score >= 0.85
    assert "Article 18" in result.routing_reason


def test_dynamic_3ds_statutory_cap_above_500():
    agent = Dynamic3DSAgent()
    payload = {
        "amount": 650.0,
        "currency": "EUR",
        "bin_country": "DE",
        "merchant_country": "IE",
        "card_brand": "visa",
        "fraud_score": 0.02,
        "acquirer_tra_rate": 0.0001,
    }
    result = agent.decide_sync(payload)

    assert result.choice == "challenge"
    assert "500" in result.routing_reason


def test_dynamic_3ds_high_fraud_forces_challenge():
    agent = Dynamic3DSAgent()
    payload = {
        "amount": 40.0,
        "currency": "EUR",
        "bin_country": "NL",
        "merchant_country": "IE",
        "card_brand": "visa",
        "fraud_score": 0.55,  # High risk from screener
    }
    result = agent.decide_sync(payload)

    assert result.choice == "challenge"
    assert "fraud score" in result.routing_reason.lower()


def test_dynamic_3ds_one_leg_out():
    agent = Dynamic3DSAgent()
    payload = {
        "amount": 150.0,
        "currency": "USD",
        "bin_country": "US",
        "merchant_country": "US",
        "card_brand": "visa",
        "fraud_score": 0.05,
    }
    result = agent.decide_sync(payload)

    assert result.choice == "tra_exemption"
    assert "one-leg-out" in result.routing_reason.lower()


# =============================================================================
# SMART RETRY TESTS
# =============================================================================

def test_smart_retry_hard_decline_prohibition():
    agent = SmartRetryAgent()
    payload = {
        "decline_code": "stolen_card",
        "decline_category": "hard_decline",
        "retry_attempt": 1,
        "amount": 50.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
    }
    result = agent.decide_sync(payload)

    assert result.choice == "terminal_no_retry"
    assert result.score == -1.0
    assert result.confidence >= 0.98
    assert "prohibited" in result.routing_reason.lower()


def test_smart_retry_excessive_retry_cap():
    agent = SmartRetryAgent()
    payload = {
        "decline_code": "insufficient_funds",
        "decline_category": "soft_decline",
        "retry_attempt": 4,  # scheme limit reached
        "amount": 30.0,
        "currency": "USD",
        "card_brand": "mastercard",
        "bin_country": "US",
    }
    result = agent.decide_sync(payload)

    assert result.choice == "terminal_no_retry"
    assert result.score == -1.0
    assert "excessive" in result.routing_reason.lower()


def test_smart_retry_technical_timeout_instant_reroute():
    agent = SmartRetryAgent()
    payload = {
        "decline_code": "processor_timeout",
        "decline_category": "technical_error",
        "retry_attempt": 1,
        "amount": 75.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
    }
    result = agent.decide_sync(payload)

    assert result.choice == "retry_immediate"
    assert result.score == 0.0  # 0 hours
    assert result.scores["recovery_probability"] > 0.60


def test_smart_retry_payday_alignment():
    agent = SmartRetryAgent()
    payload = {
        "decline_code": "insufficient_funds",
        "decline_category": "soft_decline",
        "retry_attempt": 1,
        "amount": 120.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
        "billing_cycle_day": 1,  # 1st of month payday
        "day_of_week": 4,        # Friday
    }
    result = agent.decide_sync(payload)

    assert result.choice == "retry_payday_window"
    assert result.score >= 72.0
    assert "payday" in result.routing_reason.lower()


# =============================================================================
# SMART ROUTING TESTS
# =============================================================================

def test_smart_routing_domestic_us():
    agent = SmartRoutingAgent()
    payload = {
        "amount": 50.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
        "mcc": "5411",
        "velocity_1h": 1,
    }
    result = agent.decide_sync(payload)

    assert result.choice == "stripe_us"
    assert result.confidence >= 0.55


def test_smart_routing_european_direct():
    agent = SmartRoutingAgent()
    payload = {
        "amount": 75.0,
        "currency": "EUR",
        "card_brand": "mastercard",
        "bin_country": "DE",
        "mcc": "5411",
        "velocity_1h": 1,
    }
    result = agent.decide_sync(payload)

    assert result.choice == "adyen_eu"
    assert result.confidence >= 0.55


# =============================================================================
# ASYNC & CONTEXT INTEGRATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_agent_async_decide_and_context_chaining():
    # 1. Initialize context with raw transaction
    context = TransactionContext(
        inputs={
            "amount": 85.0,
            "currency": "EUR",
            "card_brand": "visa",
            "bin_country": "FR",
            "merchant_country": "IE",
            "ip_country": "FR",
            "mcc": "5411",
            "velocity_1h": 1,
            "cvv_result": "match",
            "avs_code": "match",
        }
    )

    # 2. Stage 1: Fraud Screener evaluates
    fraud_agent = create_agent("fraud_screener")
    fraud_res = await fraud_agent.decide(context)
    assert fraud_res.choice == "allow"
    context.record_agent_result(
        agent_name="fraud_screener",
        choice=fraud_res.choice,
        score=fraud_res.score,
        confidence=fraud_res.confidence,
        latency_ms=fraud_res.latency_ms,
    )

    # 3. Stage 2: Dynamic 3DS reads fraud score from context
    three_ds_agent = create_agent("dynamic_3ds")
    three_ds_res = await three_ds_agent.decide(context)
    assert three_ds_res.choice == "tra_exemption"
    context.record_agent_result(
        agent_name="dynamic_3ds",
        choice=three_ds_res.choice,
        score=three_ds_res.score,
        confidence=three_ds_res.confidence,
        latency_ms=three_ds_res.latency_ms,
    )

    # 4. Stage 3: Smart Routing dispatches
    routing_agent = create_agent("smart_routing")
    routing_res = await routing_agent.decide(context)
    assert routing_res.choice == "adyen_eu"

    # Verify context history
    assert "fraud_screener" in context.agent_results
    assert "dynamic_3ds" in context.agent_results
    assert context.get("fraud_score") == fraud_res.score


@pytest.mark.asyncio
async def test_full_dag_pipeline_with_domain_agents():
    from payment_decision_factory.orchestration.pipeline import PaymentOrchestrationPipeline

    pipeline = PaymentOrchestrationPipeline(
        name="full_checkout_pipeline",
        global_timeout_ms=45.0,
        primary_decision_agent="smart_routing",
    )

    # Stage 1: Parallel Fraud Screening & Dynamic 3DS
    fraud = FraudScreenerAgent()
    three_ds = Dynamic3DSAgent()
    pipeline.add_parallel_stage("risk_and_sca", [fraud, three_ds], sla_ms=18.0)

    # Dynamic short circuit if fraud screener declines
    pipeline.add_short_circuit_rule(
        lambda ctx: ("High risk fraud decline", "decline", 0.95)
        if ctx.get("fraud_choice") == "decline" else None
    )

    # Stage 2: Smart Gateway Routing
    routing = SmartRoutingAgent()
    pipeline.add_sequential_stage("gateway_routing", [routing], sla_ms=18.0)

    # 1. Normal US Transaction
    ctx = TransactionContext(
        inputs={
            "amount": 49.99,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "ip_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
            "cvv_result": "match",
            "avs_code": "match",
        }
    )
    res_ctx = await pipeline.execute(ctx)
    assert not res_ctx.is_short_circuited
    assert res_ctx.final_decision["choice"] == "stripe_us"
    assert res_ctx.total_latency_ms < 45.0

    # 2. Card Attack Transaction Short-Circuits
    ctx_attack = TransactionContext(
        inputs={
            "amount": 10.00,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "ip_country": "US",
            "mcc": "5411",
            "velocity_1h": 15,
            "cvv_result": "match",
        }
    )
    res_attack = await pipeline.execute(ctx_attack)
    assert res_attack.is_short_circuited
    assert res_attack.final_decision["choice"] == "decline"


# =============================================================================
# SPECIALIST SDM BACKING TESTS
# =============================================================================

def test_smart_routing_specialist_sdm_backing():
    from payment_decision_factory.core.sdm import SpecialistSDM

    agent = SmartRoutingAgent()
    assert not agent.is_sdm_backed

    sdm = SpecialistSDM("routing", num_continuous_features=30, num_classes=3)
    agent.set_sdm(sdm)
    assert agent.is_sdm_backed

    payload = {
        "amount": 50.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
        "mcc": "5411",
        "velocity_1h": 1,
    }
    # Warm-up cold start initialization
    _ = agent.decide_sync(payload)
    result = agent.decide_sync(payload)

    assert result.metadata["backed_by_sdm"] is True
    assert result.metadata["sdm_type"] == "routing"
    assert result.choice in ["stripe_us", "adyen_eu", "checkout_crossborder"]
    assert result.confidence > 0.0
    assert result.latency_ms < 20.0  # Production gateway SLA is < 20ms


def test_fraud_screener_specialist_sdm_backing_and_circuit_breakers():
    from payment_decision_factory.core.sdm import SpecialistSDM

    agent = FraudScreenerAgent()
    sdm = SpecialistSDM("fraud", num_continuous_features=30, num_classes=3)
    agent.set_sdm(sdm)
    assert agent.is_sdm_backed

    # 1. Normal evaluation backed by SDM
    payload = {
        "amount": 45.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
        "ip_country": "US",
        "mcc": "5411",
        "velocity_1h": 1,
        "cvv_result": "match",
    }
    result = agent.decide_sync(payload)
    assert result.metadata["backed_by_sdm"] is True
    assert result.choice in ["allow", "review", "decline"]
    assert result.score is not None

    # 2. Circuit breaker MUST override SDM inference during bot card testing attack
    bot_payload = {
        "amount": 25.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
        "ip_country": "US",
        "mcc": "5411",
        "velocity_1h": 16,  # Attack threshold
        "cvv_result": "match",
    }
    bot_res = agent.decide_sync(bot_payload)
    assert bot_res.choice == "decline"
    assert bot_res.score >= 0.90
    assert bot_res.metadata.get("rule_triggered") == "card_testing_velocity_burst"


def test_dynamic_3ds_specialist_sdm_backing_and_statutory_boundaries():
    from payment_decision_factory.core.sdm import SpecialistSDM

    agent = Dynamic3DSAgent()
    sdm = SpecialistSDM("sca", num_continuous_features=30, num_classes=4)
    agent.set_sdm(sdm)
    assert agent.is_sdm_backed

    # 1. Normal evaluation backed by SDM
    payload = {
        "amount": 45.0,
        "currency": "EUR",
        "card_brand": "visa",
        "bin_country": "DE",
        "merchant_country": "IE",
        "fraud_score": 0.05,
    }
    result = agent.decide_sync(payload)
    assert result.metadata["backed_by_sdm"] is True
    assert result.choice in ["tra_exemption", "low_value", "secure_corporate", "challenge"]

    # 2. Statutory cap (> 500 EUR) MUST override SDM inference under PSD2 law
    cap_payload = {
        "amount": 750.0,
        "currency": "EUR",
        "card_brand": "visa",
        "bin_country": "DE",
        "merchant_country": "IE",
        "fraud_score": 0.01,
    }
    cap_res = agent.decide_sync(cap_payload)
    assert cap_res.choice == "challenge"
    assert "500" in cap_res.routing_reason


def test_smart_retry_specialist_sdm_backing_and_scheme_rules():
    from payment_decision_factory.core.sdm import SpecialistSDM

    agent = SmartRetryAgent()
    sdm = SpecialistSDM("retry", num_continuous_features=30, num_classes=5)
    agent.set_sdm(sdm)
    assert agent.is_sdm_backed

    # 1. Soft decline evaluation backed by SDM
    payload = {
        "decline_code": "insufficient_funds",
        "decline_category": "soft_decline",
        "retry_attempt": 1,
        "amount": 50.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
    }
    result = agent.decide_sync(payload)
    assert result.metadata["backed_by_sdm"] is True
    assert result.choice in ["retry_immediate", "retry_short_backoff", "retry_smart_window", "retry_payday_window", "terminal_no_retry"]
    assert result.score in [0.0, 8.0, 48.0, 96.0, -1.0]

    # 2. Scheme rules strictly forbid retrying stolen cards even with SDM
    stolen_payload = {
        "decline_code": "stolen_card",
        "decline_category": "hard_decline",
        "retry_attempt": 1,
        "amount": 50.0,
        "currency": "USD",
        "card_brand": "visa",
        "bin_country": "US",
    }
    stolen_res = agent.decide_sync(stolen_payload)
    assert stolen_res.choice == "terminal_no_retry"
    assert stolen_res.score == -1.0
    assert stolen_res.metadata.get("compliance_enforced") is True


def test_agent_load_sdm_onnx_runtime():
    import os
    import tempfile
    from payment_decision_factory.core.sdm import SpecialistSDM, export_specialist_to_onnx

    sdm = SpecialistSDM("routing", num_continuous_features=30, num_classes=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "specialist.onnx")
        export_specialist_to_onnx(sdm, onnx_file, input_dim=30)
        assert os.path.exists(onnx_file)

        agent = SmartRoutingAgent()
        agent.load_sdm(onnx_file)
        assert agent.is_sdm_backed

        payload = {
            "amount": 75.0,
            "currency": "EUR",
            "card_brand": "mastercard",
            "bin_country": "DE",
            "mcc": "5411",
            "velocity_1h": 1,
        }
        res = agent.decide_sync(payload)
        assert res.metadata["backed_by_sdm"] is True
        assert res.choice in ["stripe_us", "adyen_eu", "checkout_crossborder"]
        assert res.latency_ms < 5.0


def test_agent_load_sdmpkg_sealed_package():
    import os
    import tempfile
    from payment_decision_factory.core.sdm import SpecialistSDM, export_specialist_to_onnx
    from payment_decision_factory.core.sealed_package import SealedPackage

    sdm = SpecialistSDM("routing", num_continuous_features=30, num_classes=3)
    license_key = "LIC-TEST-ENTERPRISE-KEY-999"
    client_id = "test_acquirer_client"

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "routing.onnx")
        pkg_file = os.path.join(tmpdir, "routing.sdmpkg")

        export_specialist_to_onnx(sdm, onnx_file, input_dim=30, verify=False)
        assert os.path.exists(onnx_file)

        # Seal into .sdmpkg
        sealed_pkg = SealedPackage.seal(
            model_bytes_or_path=onnx_file,
            license_key=license_key,
            client_id=client_id,
            domain="routing",
            choices=["stripe_us", "adyen_eu", "checkout_crossborder"],
            input_dim=30,
            validity_days=30,
            output_path=pkg_file,
        )
        assert os.path.exists(pkg_file)

        # Load into domain agent via sealed package
        agent = SmartRoutingAgent()
        agent.load_sdm(pkg_file, license_key=license_key)
        assert agent.is_sdm_backed is True

        payload = {
            "amount": 120.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
        }
        res = agent.decide_sync(payload)

        assert res.metadata["backed_by_sdm"] is True
        assert res.metadata["is_sealed"] is True
        assert res.metadata["package_id"] == sealed_pkg.manifest["package_id"]
        assert res.choice in ["stripe_us", "adyen_eu", "checkout_crossborder"]
        assert 0.0 <= res.confidence <= 1.0
        assert res.latency_ms < 5.0


def test_agent_sealed_package_authentication_failure():
    import os
    import tempfile
    from payment_decision_factory.core.sdm import SpecialistSDM, export_specialist_to_onnx
    from payment_decision_factory.core.sealed_package import SealedPackage, LicenseAuthenticationError

    sdm = SpecialistSDM("routing", num_continuous_features=30, num_classes=3)
    correct_key = "LIC-VALID-KEY"
    wrong_key = "LIC-WRONG-KEY-REJECT"

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "routing.onnx")
        pkg_file = os.path.join(tmpdir, "routing.sdmpkg")

        export_specialist_to_onnx(sdm, onnx_file, input_dim=30, verify=False)
        SealedPackage.seal(
            model_bytes_or_path=onnx_file,
            license_key=correct_key,
            client_id="client_1",
            domain="routing",
            choices=["stripe_us", "adyen_eu", "checkout_crossborder"],
            input_dim=30,
            validity_days=30,
            output_path=pkg_file,
        )

        agent = SmartRoutingAgent()
        with pytest.raises(LicenseAuthenticationError):
            agent.load_sdm(pkg_file, license_key=wrong_key)


def test_universal_decision_agent_default_policy():
    agent = create_agent(
        "universal",
        name="credit_underwriter",
        choices=["auto_approve", "manual_review", "auto_reject"],
    )
    assert isinstance(agent, UniversalDecisionAgent)
    assert agent.explicit_choices == ["auto_approve", "manual_review", "auto_reject"]

    payload = {"credit_score": 720, "annual_income": 85000, "debt_to_income": 0.28}
    result = agent.decide_sync(payload)

    assert result.agent_name == "credit_underwriter"
    assert result.choice == "auto_approve"
    assert result.confidence > 0.0
    assert "auto_approve" in result.probabilities
    assert result.metadata["backed_by_sdm"] is False


def test_universal_decision_agent_sealed_sdmpkg_execution():
    import os
    import tempfile
    from payment_decision_factory.core.sdm import SpecialistSDM, export_specialist_to_onnx
    from payment_decision_factory.core.sealed_package import SealedPackage

    choices = ["ground_economy", "express_air", "freight_liner"]
    sdm = SpecialistSDM("logistics_dispatch", num_continuous_features=10, num_classes=len(choices))
    license_key = "LIC-LOGISTICS-ENTERPRISE-2026"

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "logistics.onnx")
        pkg_file = os.path.join(tmpdir, "logistics.sdmpkg")

        export_specialist_to_onnx(sdm, onnx_file, input_dim=10, verify=False)

        sealed_pkg = SealedPackage.seal(
            model_bytes_or_path=onnx_file,
            license_key=license_key,
            client_id="freight_corp",
            domain="logistics_dispatch",
            choices=choices,
            input_dim=10,
            validity_days=60,
            output_path=pkg_file,
        )

        agent = UniversalDecisionAgent(
            name="carrier_dispatcher",
            choices=choices,
        )
        agent.load_sdm(pkg_file, license_key=license_key)
        assert agent.is_sdm_backed is True

        payload = {
            "weight_kg": 450.0,
            "distance_km": 1200.0,
            "deadline_hours": 24.0,
            "pallet_count": 4.0,
            "hazard_rating": 0.0,
        }
        res = agent.decide_sync(payload)

        assert res.agent_name == "carrier_dispatcher"
        assert res.choice in choices
        assert 0.0 <= res.confidence <= 1.0
        assert res.metadata["backed_by_sdm"] is True
        assert res.metadata["is_sealed"] is True
        assert res.metadata["domain"] == "logistics_dispatch"
        assert res.metadata["package_id"] == sealed_pkg.manifest["package_id"]
        assert res.latency_ms < 5.0


def test_agent_load_real_creditcard_sdm_package():
    import os
    import numpy as np

    pkg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "real_creditcard_sdm.sdmpkg",
    )
    if not os.path.exists(pkg_path):
        pytest.skip("real_creditcard_sdm.sdmpkg not built yet")

    license_key = "live_real_data_license_key_2026_enterprise"

    # 1. Load into UniversalDecisionAgent
    agent = UniversalDecisionAgent(name="real_creditcard_agent")
    agent.load_sdm(pkg_path, license_key=license_key)
    assert agent.is_sdm_backed is True

    # 2. Ingest 30 real numerical feature values
    real_sample = [0.0] * 30
    real_sample[0] = 120.0     # Time
    real_sample[1] = -1.3598   # V1
    real_sample[29] = 149.62   # Amount

    res = agent.decide_sync(real_sample)
    assert res.choice in ["legitimate", "fraudulent", "clean", "fraud", "allow", "decline"]
    assert 0.0 <= res.confidence <= 1.0
    assert res.metadata["backed_by_sdm"] is True
    assert res.metadata["is_sealed"] is True
    assert res.latency_ms < 5.0





