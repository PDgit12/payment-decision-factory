"""Comprehensive test suite for Payment Decision SDK."""

import sys
from pathlib import Path
import pytest
import pytest_asyncio
import httpx

# Add paths
THIS_FILE = Path(__file__).resolve()
FACTORY_ROOT = THIS_FILE.parents[3]
REPO_ROOT = THIS_FILE.parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FACTORY_ROOT))
sys.path.insert(0, str(FACTORY_ROOT / "sdk" / "python"))

from payment_decision_sdk import (
    DecisionEngineClient,
    DecisionResult,
    InProcessEngine,
    PaymentContext,
    SidecarClient,
    SpecialistSDMEvaluator,
    SealedPackageClient,
    InvalidContextError,
)
from payment_decision_factory.sdk.sidecar.server import app


class TestPaymentModels:
    """Test data schemas and validation rules."""

    def test_payment_context_normalization(self):
        ctx = PaymentContext(
            amount=49.99,
            currency="usd ",
            card_brand="VISA",
            bin_country="us",
            mcc="5411",
            velocity_1h=2,
        )
        assert ctx.currency == "USD"
        assert ctx.card_brand == "visa"
        assert ctx.bin_country == "US"
        assert ctx.mcc == "5411"
        assert ctx.to_dict()["amount"] == 49.99

    def test_payment_context_zero_pan_sanitization(self):
        """Verifies raw PAN is automatically sanitized under PCI-DSS 4.0 Requirement 3.3.1."""
        raw_data = {
            "amount": 99.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "pan": "4111111111111111",
            "card_number": "4242424242424242",
            "metadata": {
                "order_id": "ord_123",
                "full_pan": "4000000000000002",
            },
        }
        ctx = PaymentContext.from_dict(raw_data)
        d = ctx.to_dict()
        assert "pan" not in d
        assert "card_number" not in d
        assert "full_pan" not in d["metadata"]
        assert d["metadata"]["order_id"] == "ord_123"

    def test_payment_context_invalid_amount(self):
        with pytest.raises(Exception):
            PaymentContext(
                amount=-10.0,
                currency="USD",
                card_brand="visa",
                bin_country="US",
                mcc="5411",
            )

    def test_decision_result_properties(self):
        res = DecisionResult(
            decision="stripe_us",
            confidence=0.88,
            probabilities={"stripe_us": 0.88, "adyen_eu": 0.12},
            latency_ms=0.45,
            routing_reason="Domestic US optimization",
        )
        assert res.decision == "stripe_us"
        assert res.choice == "stripe_us"
        assert res.confidence == 0.88
        assert res.is_fallback is False


class TestInProcessEngine:
    """Test ultra-low latency in-process scoring."""

    @pytest.fixture
    def engine(self):
        spec_path = FACTORY_ROOT / "specs" / "smart_routing.yaml"
        return InProcessEngine(spec_path=spec_path)

    def test_evaluate_us_transaction(self, engine):
        ctx = PaymentContext(
            amount=99.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=1,
        )
        result = engine.evaluate(ctx)
        assert result.decision == "stripe_us"
        assert result.confidence > 0.50
        assert result.latency_ms < 20.0
        assert result.engine_mode == "in_process"

    def test_evaluate_european_transaction(self, engine):
        ctx = PaymentContext(
            amount=150.0,
            currency="EUR",
            card_brand="mastercard",
            bin_country="DE",
            mcc="5411",
            velocity_1h=1,
        )
        result = engine.evaluate(ctx)
        assert result.decision == "adyen_eu"
        assert result.confidence > 0.50
        assert "adyen_eu" in result.probabilities

    def test_evaluate_crossborder_transaction(self, engine):
        ctx = PaymentContext(
            amount=350.0,
            currency="AUD",
            card_brand="unionpay",
            bin_country="AU",
            mcc="5732",
            velocity_1h=1,
        )
        result = engine.evaluate(ctx)
        assert result.decision == "checkout_crossborder"
        assert result.confidence > 0.40

    def test_circuit_breaker_high_velocity(self, engine):
        ctx = PaymentContext(
            amount=25.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=20,  # Exceeds max velocity threshold 15
        )
        result = engine.evaluate(ctx)
        assert result.is_fallback is True
        assert "challenge" in result.routing_reason.lower() or "circuit" in result.routing_reason.lower()


class TestDecisionEngineClient:
    """Test unified client in both sync and async workflows."""

    def test_client_in_process_sync(self):
        spec_path = FACTORY_ROOT / "specs" / "smart_routing.yaml"
        client = DecisionEngineClient(mode="in_process", spec_path=spec_path)

        res = client.decide({
            "amount": 75.0,
            "currency": "USD",
            "card_brand": "amex",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
        })
        assert res.decision == "stripe_us"
        assert res.confidence > 0.50
        assert res.latency_ms < 20.0

    @pytest.mark.asyncio
    async def test_client_in_process_async(self):
        spec_path = FACTORY_ROOT / "specs" / "smart_routing.yaml"
        client = DecisionEngineClient(mode="in_process", spec_path=spec_path)

        res = await client.decide_async({
            "amount": 80.0,
            "currency": "EUR",
            "card_brand": "visa",
            "bin_country": "FR",
            "mcc": "5411",
            "velocity_1h": 1,
        })
        assert res.decision == "adyen_eu"

    def test_client_batch_sync(self):
        spec_path = FACTORY_ROOT / "specs" / "smart_routing.yaml"
        client = DecisionEngineClient(mode="in_process", spec_path=spec_path)

        batch = [
            {"amount": 50.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411"},
            {"amount": 120.0, "currency": "EUR", "card_brand": "mastercard", "bin_country": "DE", "mcc": "5411"},
            {"amount": 300.0, "currency": "AUD", "card_brand": "unionpay", "bin_country": "AU", "mcc": "5732"},
        ]
        results = client.decide_batch(batch)
        assert len(results) == 3
        assert results[0].decision == "stripe_us"
        assert results[1].decision == "adyen_eu"
        assert results[2].decision == "checkout_crossborder"

    @pytest.mark.asyncio
    async def test_client_batch_async(self):
        spec_path = FACTORY_ROOT / "specs" / "smart_routing.yaml"
        client = DecisionEngineClient(mode="in_process", spec_path=spec_path)

        batch = [
            {"amount": 50.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411"},
            {"amount": 120.0, "currency": "EUR", "card_brand": "mastercard", "bin_country": "DE", "mcc": "5411"},
        ]
        results = await client.decide_batch_async(batch)
        assert len(results) == 2


class TestSidecarIntegration:
    """Test sidecar client with live in-memory ASGI app and network fallback."""

    def test_sidecar_with_asgi_transport(self):
        from fastapi.testclient import TestClient
        test_client = TestClient(app)
        client = SidecarClient(base_url="http://testserver", timeout_ms=50.0)
        client._client = test_client

        ctx = PaymentContext(
            amount=65.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=1,
        )
        res = client.decide(ctx)
        assert res.decision == "stripe_us"
        assert res.engine_mode == "sidecar"
        assert res.confidence > 0.50

    def test_sidecar_offline_fallback(self):
        # Point to unreachable port to verify zero-dropout checkout fallback
        client = SidecarClient(
            base_url="http://127.0.0.1:59999",
            timeout_ms=10.0,
            fallback_decision="stripe_us",
            raise_on_error=False,
        )
        ctx = PaymentContext(
            amount=100.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=1,
        )
        res = client.decide(ctx)
        assert res.is_fallback is True
        assert res.decision == "stripe_us"
        assert "fallback" in res.routing_reason.lower()


class TestSpecialistSDMEvaluator:
    """Test the Specialist Small Decision Model (SDM) evaluator suite."""

    def test_routing_specialist_sdm(self):
        evaluator = SpecialistSDMEvaluator(specialist_type="routing")
        ctx = PaymentContext(
            amount=75.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=1,
        )
        res = evaluator.evaluate(ctx)
        assert res.decision == "stripe_us"
        assert res.confidence > 0.50
        assert res.latency_ms < 5.0
        assert res.engine_mode == "specialist_sdm"

    def test_fraud_specialist_sdm(self):
        evaluator = SpecialistSDMEvaluator(specialist_type="fraud")
        ctx = PaymentContext(
            amount=50.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=15,  # Anomaly burst
        )
        res = evaluator.evaluate(ctx)
        assert res.decision == "high_risk_burst"
        assert res.confidence > 0.50

    def test_sca_specialist_sdm(self):
        evaluator = SpecialistSDMEvaluator(specialist_type="sca")
        ctx = PaymentContext(
            amount=20.0,  # Below 30 EUR low-value threshold
            currency="EUR",
            card_brand="mastercard",
            bin_country="DE",
            mcc="5411",
            velocity_1h=1,
        )
        res = evaluator.evaluate(ctx)
        assert res.decision == "frictionless_low_value"
        assert res.confidence > 0.50

    def test_retry_specialist_sdm(self):
        evaluator = SpecialistSDMEvaluator(specialist_type="retry")
        ctx = PaymentContext(
            amount=45.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=1,
        )
        res = evaluator.evaluate(ctx)
        assert res.decision == "immediate"
        assert res.confidence > 0.50

    def test_client_mode_sdm(self):
        client = DecisionEngineClient(mode="sdm", specialist_type="routing")
        res = client.decide({
            "amount": 120.0,
            "currency": "EUR",
            "card_brand": "mastercard",
            "bin_country": "FR",
            "mcc": "5411",
            "velocity_1h": 1,
        })
        assert res.decision == "adyen_eu"
        assert res.engine_mode == "specialist_sdm"

    def test_sidecar_specialist_endpoint(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.post(
            "/specialists/decide",
            json={
                "specialist_type": "routing",
                "context": {
                    "amount": 95.0,
                    "currency": "USD",
                    "card_brand": "amex",
                    "bin_country": "US",
                    "mcc": "5411",
                    "velocity_1h": 1,
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "stripe_us"
        assert data["confidence"] > 0.50


class TestSealedPackageClient:
    """Test sealed closed-source packaging and client loading."""

    @pytest.fixture
    def sealed_package_file(self, tmp_path):
        from payment_decision_factory.core.sealed_package import SealedPackage
        import onnx
        from onnx import helper, TensorProto

        # Create a small valid ONNX model in memory with ir_version=8
        X = helper.make_tensor_value_info("continuous_features", TensorProto.FLOAT, [1, 30])
        Y_logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 30])
        Y_probs = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 30])

        node_softmax = helper.make_node("Softmax", ["continuous_features"], ["probabilities"], axis=1)
        node_id = helper.make_node("Identity", ["continuous_features"], ["logits"])
        graph = helper.make_graph([node_softmax, node_id], "test_graph", [X], [Y_logits, Y_probs])
        model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        model_bytes = model_proto.SerializeToString()

        choices = ["stripe_us", "adyen_eu", "checkout_crossborder"] + [f"choice_{i}" for i in range(3, 30)]

        pkg_path = tmp_path / "test_model.sdmpkg"
        SealedPackage.seal(
            model_bytes_or_path=model_bytes,
            license_key="test-secret-license-key-12345",
            client_id="client_test_corp",
            choices=choices,
            output_path=str(pkg_path),
        )
        return pkg_path

    def test_sealed_client_decide_happy_path(self, sealed_package_file):
        client = SealedPackageClient(
            package_path=sealed_package_file,
            license_key="test-secret-license-key-12345",
            client_id="client_test_corp",
        )
        ctx = PaymentContext(
            amount=80.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=1,
        )
        res = client.decide(ctx)
        assert res.engine_mode == "sealed_package"
        assert res.decision in ["stripe_us", "adyen_eu", "checkout_crossborder"]
        assert res.latency_ms < 10.0

    def test_sealed_client_wrong_license(self, sealed_package_file):
        with pytest.raises(Exception):
            SealedPackageClient(
                package_path=sealed_package_file,
                license_key="invalid-wrong-license-key",
            )

    def test_decision_engine_client_sealed_mode(self, sealed_package_file):
        client = DecisionEngineClient(
            mode="sealed",
            sealed_package_path=sealed_package_file,
            license_key="test-secret-license-key-12345",
            client_id="client_test_corp",
        )
        res = client.decide({
            "amount": 100.0,
            "currency": "EUR",
            "card_brand": "mastercard",
            "bin_country": "DE",
            "mcc": "5411",
            "velocity_1h": 1,
        })
        assert res.engine_mode == "sealed_package"

    def test_sealed_client_enterprise_sub_billion_tier_sub_10ms_latency(self, tmp_path):
        import os, sys
        import importlib.util
        _stdlib_code = "/opt/anaconda3/lib/python3.13/code.py"
        if os.path.exists(_stdlib_code) and ("code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")):
            _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
            _m = importlib.util.module_from_spec(_spec)
            sys.modules["code"] = _m
            _spec.loader.exec_module(_m)

        from payment_decision_factory.core.enterprise_sdm import (
            EnterpriseSubBillionSDM,
            export_enterprise_sdm_to_onnx,
        )
        from payment_decision_factory.core.sealed_package import SealedPackage

        # 1. Compile enterprise SDM
        ent_model = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=30,
            num_classes=3,
            parameter_target="small",
        )
        tmp_onnx = tmp_path / "enterprise_sdm.onnx"
        export_enterprise_sdm_to_onnx(ent_model, str(tmp_onnx), input_dim=30, verify=False)

        with open(tmp_onnx, "rb") as f:
            onnx_bytes = f.read()

        pkg_path = tmp_path / "enterprise_tier.sdmpkg"
        choices = ["stripe_us", "adyen_eu", "checkout_crossborder"]
        SealedPackage.seal(
            model_bytes_or_path=onnx_bytes,
            license_key="enterprise-secret-key-xyz",
            client_id="enterprise_bank_corp",
            choices=choices,
            tier="sub_billion",
            domain="enterprise_routing",
            output_path=str(pkg_path),
        )

        # 2. Load via SealedPackageClient
        client = SealedPackageClient(
            package_path=pkg_path,
            license_key="enterprise-secret-key-xyz",
            client_id="enterprise_bank_corp",
        )

        assert client.tier == "sub_billion"
        assert client.is_enterprise_tier is True
        assert client.domain == "enterprise_routing"
        assert client.choices == choices

        # 3. Evaluate transaction & assert sub-10ms latency SLA
        ctx = PaymentContext(
            amount=500.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=1,
        )
        res = client.decide(ctx)
        assert res.engine_mode == "sealed_package"
        assert res.decision in choices
        assert res.confidence > 0.0
        assert res.latency_ms < 10.0
        assert "sub_billion" in res.routing_reason

        # 4. Batch evaluation
        batch_res = client.decide_batch([ctx] * 5)
        assert len(batch_res) == 5
        assert all(r.latency_ms < 10.0 for r in batch_res)

    def test_sealed_client_gpu_enterprise_transformer_loading(self, tmp_path):
        import os, sys
        import importlib.util
        _stdlib_code = "/opt/anaconda3/lib/python3.13/code.py"
        if os.path.exists(_stdlib_code) and ("code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")):
            _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
            _m = importlib.util.module_from_spec(_spec)
            sys.modules["code"] = _m
            _spec.loader.exec_module(_m)

        from payment_decision_factory.core.gpu_enterprise_sdm import (
            EnterpriseGPUTabularTransformer,
            export_gpu_sdm_to_quantized_onnx,
        )
        from payment_decision_factory.core.sealed_package import SealedPackage

        # 1. Instantiate EnterpriseGPUTabularTransformer with FlashTabularAttention
        model = EnterpriseGPUTabularTransformer(
            num_continuous_features=30,
            num_classes=3,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )

        onnx_path = tmp_path / "gpu_model.onnx"
        export_gpu_sdm_to_quantized_onnx(model, str(onnx_path), input_dim=30)

        with open(onnx_path, "rb") as f:
            onnx_bytes = f.read()

        pkg_path = tmp_path / "gpu_model.sdmpkg"
        choices = ["stripe_us", "adyen_eu", "checkout_crossborder"]
        SealedPackage.seal(
            model_bytes_or_path=onnx_bytes,
            license_key="test-gpu-license-999",
            client_id="gpu_fintech_client",
            choices=choices,
            tier="sub_billion",
            domain="enterprise_gpu_routing",
            output_path=str(pkg_path),
        )

        client = SealedPackageClient(str(pkg_path), license_key="test-gpu-license-999", client_id="gpu_fintech_client")
        assert client.tier == "sub_billion"
        assert client.is_enterprise_tier is True
        assert client.domain == "enterprise_gpu_routing"

        ctx = PaymentContext(
            amount=120.0,
            currency="USD",
            card_brand="visa",
            bin_country="US",
            mcc="5411",
            velocity_1h=1,
        )
        res = client.decide(ctx)
        assert res.decision in choices
        assert res.confidence > 0.0
        assert res.latency_ms < 10.0
        assert res.engine_mode == "sealed_package"

    def test_sealed_client_real_data_benchmark_inference(self, tmp_path):
        import os, sys, time, numpy as np, pandas as pd
        import importlib.util
        _stdlib_code = "/opt/anaconda3/lib/python3.13/code.py"
        if os.path.exists(_stdlib_code) and ("code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")):
            _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
            _m = importlib.util.module_from_spec(_spec)
            sys.modules["code"] = _m
            _spec.loader.exec_module(_m)

        from payment_decision_factory.core.sealed_package import SealedPackage
        from payment_decision_factory.core.enterprise_sdm import (
            EnterpriseSubBillionSDM,
            export_enterprise_sdm_to_onnx,
        )

        csv_path = "/Users/piyushdua/payment_decision_factory/data/creditcard_sample.csv"
        assert os.path.exists(csv_path)

        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ("Class",)]
        records = df[feature_cols].values.astype(np.float32)
        num_features = records.shape[1]

        # 1. Compile and seal model
        model = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=num_features,
            num_classes=2,
            parameter_target="small",
        )
        onnx_file = tmp_path / "real_benchmark.onnx"
        export_enterprise_sdm_to_onnx(model, str(onnx_file), input_dim=num_features, verify=False)

        with open(onnx_file, "rb") as f:
            onnx_bytes = f.read()

        pkg_path = tmp_path / "real_benchmark.sdmpkg"
        choices = ["legitimate", "fraudulent"]
        license_key = "live_real_data_license_key_2026_enterprise"
        client_id = "enterprise_mercantile_corp"

        SealedPackage.seal(
            model_bytes_or_path=onnx_bytes,
            license_key=license_key,
            client_id=client_id,
            choices=choices,
            tier="sub_billion",
            domain="real_data_fraud_detection",
            input_dim=num_features,
            output_path=str(pkg_path),
        )

        # 2. In-memory decryption and client initialization
        client = SealedPackageClient(
            package_path=str(pkg_path),
            license_key=license_key,
            client_id=client_id,
        )
        assert client.tier == "sub_billion"
        assert client.is_enterprise_tier is True
        assert client.choices == choices

        # 3. Direct numpy array inference with microsecond latency
        res_arr = client.decide(records[0])
        assert res_arr.decision in choices
        assert res_arr.confidence > 0.0
        assert res_arr.latency_ms < 10.0
        assert res_arr.engine_mode == "sealed_package"

        # 4. Feature dictionary inference
        res_dict = client.decide({"features": list(records[1])})
        assert res_dict.decision in choices
        assert res_dict.engine_mode == "sealed_package"

        # 5. Fail-open fallback protection
        res_fail = client.decide_safe("corrupted_invalid_input")
        assert res_fail.is_fallback is True
        assert res_fail.decision in choices
        assert "Zero-Dropout Fail-Open Fallback" in res_fail.routing_reason

        # 6. Batch evaluation on real transactions
        batch_res = client.decide_batch([records[i] for i in range(10)])
        assert len(batch_res) == 10
        assert all(b.decision in choices for b in batch_res)





