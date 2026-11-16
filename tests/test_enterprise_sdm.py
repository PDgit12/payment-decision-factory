"""
Unit tests for Enterprise Sub-Billion Parameter Specialist SDM Architecture.
============================================================================
Validates:
1. NumericalFeatureTokenizer & TransformerBlock primitives
2. EnterpriseSubBillionSDM parameter scaling (small ~1M, medium ~5M, large 113M+, sub_billion 201M+)
3. Temperature calibration and probability output formatting
4. ONNX single-file export and EnterpriseONNXRunner in-memory execution (<10ms)
5. UniversalDecisionFactory end-to-end integration with tier="sub_billion"
6. SealedPackage AES-256-GCM containerization and unlock context manager
"""

import os
import sys
import tempfile
import time
import pytest
import numpy as np
import torch

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.enterprise_sdm import (
    NumericalFeatureTokenizer,
    TransformerBlock,
    EnterpriseSubBillionSDM,
    export_enterprise_sdm_to_onnx,
    EnterpriseONNXRunner,
)
from payment_decision_factory.core.universal_factory import (
    UniversalDecisionFactory,
    UniversalDomainSpec,
)
from payment_decision_factory.core.sealed_package import SealedPackage


class TestNumericalFeatureTokenizerAndTransformer:
    def test_numerical_tokenizer_shape(self):
        tokenizer = NumericalFeatureTokenizer(num_features=5, embedding_dim=32)
        x = torch.randn(4, 5)
        tokens = tokenizer(x)
        assert tokens.shape == (4, 5, 32)

    def test_transformer_block_forward(self):
        block = TransformerBlock(hidden_dim=32, num_heads=4, mlp_ratio=2.0)
        x = torch.randn(2, 6, 32)
        out = block(x)
        assert out.shape == (2, 6, 32)


class TestEnterpriseSubBillionSDMArchitecture:
    def test_parameter_targets_and_scaling(self):
        # Small dev tier (~800k params)
        model_small = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=10,
            num_classes=3,
            parameter_target="small",
        )
        small_params = model_small.count_parameters()
        assert 500_000 < small_params < 2_000_000

        # Medium production tier (~5M params)
        model_med = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=10,
            num_classes=3,
            parameter_target="medium",
        )
        med_params = model_med.count_parameters()
        assert 3_000_000 < med_params < 10_000_000

        # Large enterprise tier (113M+ params)
        model_large = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=10,
            num_classes=3,
            parameter_target="large",
        )
        large_params = model_large.count_parameters()
        assert large_params > 100_000_000

        # Sub-billion flagship tier (201M+ params)
        model_sub = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=10,
            num_classes=3,
            parameter_target="sub_billion",
        )
        sub_params = model_sub.count_parameters()
        assert sub_params > 200_000_000

    def test_forward_pass_with_continuous_and_categorical(self):
        cat_dims = {"bin_country": 50, "currency": 10}
        model = EnterpriseSubBillionSDM(
            num_continuous_features=6,
            num_classes=3,
            categorical_dims=cat_dims,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
        )

        batch_size = 4
        cont_x = torch.randn(batch_size, 6)
        cat_x = torch.tensor([[1, 2], [3, 4], [0, 1], [5, 0]], dtype=torch.long)

        logits, probs = model(cont_x, cat_x)
        assert logits.shape == (batch_size, 3)
        assert probs.shape == (batch_size, 3)

        # Probabilities sum to 1.0 across classes
        prob_sums = probs.sum(dim=-1)
        torch.testing.assert_close(prob_sums, torch.ones(batch_size), atol=1e-5, rtol=1e-5)

    def test_create_specialist_unified_architecture(self):
        """Verifies the 85.1M parameter Unified Enterprise Specialist SDM."""
        model = EnterpriseSubBillionSDM.create_specialist(
            num_continuous_features=30,
            num_classes=2,
            specialist_name="enterprise_routing_specialist",
        )
        assert model.hidden_dim == 768
        assert model.num_layers == 12
        assert model.num_heads == 12
        assert model.specialist_name == "enterprise_routing_specialist"

        param_count = model.count_parameters()
        assert 80_000_000 <= param_count <= 90_000_000, f"Expected ~85.1M params, got {param_count:,}"

        # Test forward pass and Platt temperature calibration
        model.eval()
        x = torch.randn(2, 30)
        with torch.no_grad():
            logits, probs = model(x)
            assert logits.shape == (2, 2)
            assert probs.shape == (2, 2)
            prob_sums = probs.sum(dim=-1)
            torch.testing.assert_close(prob_sums, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_temperature_calibration(self):
        model = EnterpriseSubBillionSDM(
            num_continuous_features=4,
            num_classes=2,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )
        model.eval()
        x = torch.randn(1, 4)
        with torch.no_grad():
            _, probs_default = model(x)

            # Scale temperature higher -> probabilities become flatter (entropy increases)
            model.temperature.data.fill_(3.0)
            _, probs_warm = model(x)
            diff_warm = abs(probs_warm[0, 0].item() - 0.5)
            diff_default = abs(probs_default[0, 0].item() - 0.5)
            assert diff_warm <= diff_default + 1e-5


class TestEnterpriseONNXExportAndRunner:
    def test_single_file_onnx_export_and_predict(self):
        model = EnterpriseSubBillionSDM.create_enterprise_tier(
            num_continuous_features=8,
            num_classes=3,
            parameter_target="small",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "enterprise_sdm.onnx")
            exported_path = export_enterprise_sdm_to_onnx(
                model=model,
                output_path=onnx_path,
                input_dim=8,
                verify=True,
            )

            assert os.path.exists(exported_path)
            # Ensure no external .onnx.data file was created
            assert not os.path.exists(exported_path + ".data")

            # In-memory execution using EnterpriseONNXRunner
            with open(exported_path, "rb") as f:
                onnx_bytes = f.read()

            runner = EnterpriseONNXRunner(onnx_bytes, choices=["stripe_us", "adyen_eu", "checkout_crossborder"])
            eval_res = runner.evaluate([0.5] * 8)

            assert eval_res["choice"] in ["stripe_us", "adyen_eu", "checkout_crossborder"]
            assert 0.0 <= eval_res["confidence"] <= 1.0
            assert eval_res["latency_ms"] < 10.0  # Sub-10ms SLA target
            assert eval_res["tier"] == "enterprise_sub_billion"


class TestUniversalFactoryEnterpriseIntegration:
    @pytest.fixture
    def sample_telemetry(self):
        return [
            {"amount": 49.99, "currency": "USD", "card_brand": "visa", "target": "stripe_us"},
            {"amount": 199.50, "currency": "EUR", "card_brand": "mastercard", "target": "adyen_eu"},
            {"amount": 890.00, "currency": "GBP", "card_brand": "amex", "target": "checkout_crossborder"},
            {"amount": 12.00, "currency": "USD", "card_brand": "visa", "target": "stripe_us"},
            {"amount": 340.00, "currency": "EUR", "card_brand": "visa", "target": "adyen_eu"},
        ] * 15

    def test_factory_train_sub_billion_tier(self, sample_telemetry):
        factory = UniversalDecisionFactory.create_for_domain(
            domain_id="enterprise_payments",
            domain_name="Enterprise Payment Gateway",
            numerical_features=["amount"],
            categorical_features=["currency", "card_brand"],
            choices=["stripe_us", "adyen_eu", "checkout_crossborder"],
        )

        model, report = factory.train(
            records=sample_telemetry,
            epochs=3,
            tier="sub_billion",
            parameter_target="small",
        )

        assert isinstance(model, EnterpriseSubBillionSDM)
        assert report.tier == "sub_billion"
        assert report.parameter_count > 500_000
        assert report.validation_accuracy >= 0.50
        assert report.brier_score >= 0.0

        # In-process single evaluation
        eval_result = factory.evaluate(
            model,
            {"amount": 75.0, "currency": "USD", "card_brand": "visa"},
        )
        assert eval_result["choice"] in ["stripe_us", "adyen_eu", "checkout_crossborder"]
        assert eval_result["confidence"] > 0.0
        assert eval_result["latency_ms"] < 15.0

    def test_factory_train_and_seal_sub_billion(self, sample_telemetry):
        factory = UniversalDecisionFactory.create_for_domain(
            domain_id="enterprise_payments",
            domain_name="Enterprise Payment Gateway",
            numerical_features=["amount"],
            categorical_features=["currency", "card_brand"],
            choices=["stripe_us", "adyen_eu", "checkout_crossborder"],
        )

        license_key = "test_enterprise_license_key_2026"
        client_id = "apex_fintech_cluster"

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_path = os.path.join(tmpdir, "enterprise_model.sdmpkg")

            sealed, report = factory.train_and_seal(
                records=sample_telemetry,
                license_key=license_key,
                client_id=client_id,
                output_package_path=pkg_path,
                epochs=3,
                tier="sub_billion",
                parameter_target="small",
                metadata={"cluster_id": "eu-central-1", "sla_target_ms": 10.0},
            )

            assert os.path.exists(pkg_path)
            assert sealed.manifest["client_id"] == client_id
            assert sealed.manifest["domain"] == "enterprise_payments"
            assert "sub_billion" in sealed.manifest["model_type"]
            assert sealed.manifest["spec"]["tier"] == "sub_billion"
            assert sealed.manifest["spec"]["metadata"]["cluster_id"] == "eu-central-1"

            # Unlock with context manager and evaluate
            with sealed.unlock(license_key) as runner:
                test_vec = [1.0] * factory.encoder.total_dim
                eval_res = runner.evaluate(test_vec)
                assert eval_res["choice"] in ["stripe_us", "adyen_eu", "checkout_crossborder"]
                assert eval_res["is_sealed"] is True
                assert eval_res["latency_ms"] < 15.0
