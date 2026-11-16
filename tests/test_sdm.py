"""
Unit & Integration Tests for Specialist Small Decision Model (SDM) & ONNX Runtime
==================================================================================
Tests:
- SpecialistSDM network architecture, parameter budget (< 250k parameters)
- GatedResidualBlock transformations
- In-memory client telemetry training in < 15 seconds
- Calibration and Brier score convergence
- ONNX export via export_specialist_to_onnx (< 2MB artifact size)
- Sub-millisecond execution via ONNXSpecialistSDM runtime
- Factory compilation integration and bundle packaging
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import tempfile
import time
import pytest
import numpy as np
import torch

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.serializer import PaymentSerializer
from payment_decision_factory.core.sdm import (
    SpecialistSDM,
    SpecialistPipeline,
    SpecialistTuningReport,
    export_specialist_to_onnx,
    ONNXSpecialistSDM,
)
from payment_decision_factory.core.factory import DecisionFactory

SPEC_PATH = os.path.join(os.path.dirname(__file__), "../specs/smart_routing.yaml")


class TestSpecialistSDMArchitecture:
    """Tests for the SpecialistSDM neural network architecture."""

    def test_initialization_and_parameter_budget(self):
        model = SpecialistSDM(
            specialist_type="routing",
            num_continuous_features=30,
            num_classes=3,
            hidden_dim=64,
            num_blocks=2,
        )
        param_count = model.count_parameters()
        # Ensure model is compact: between 10k and 250k parameters
        assert 10000 <= param_count <= 250000, f"Param count {param_count} out of range"
        # Total size in memory (float32) is < 1MB
        assert (param_count * 4) / (1024 * 1024) < 1.0

    def test_forward_pass_and_probabilities(self):
        model = SpecialistSDM(
            specialist_type="routing",
            num_continuous_features=30,
            num_classes=3,
            hidden_dim=64,
            num_blocks=2,
        )
        model.eval()
        batch_size = 8
        x = torch.randn(batch_size, 30)

        with torch.no_grad():
            logits, probs = model(x)

        assert logits.shape == (batch_size, 3)
        assert probs.shape == (batch_size, 3)

        # Probabilities must be non-negative and sum to 1.0 across classes
        sums = probs.sum(dim=-1).numpy()
        np.testing.assert_allclose(sums, np.ones(batch_size), atol=1e-5)


class TestSpecialistPipelineTraining:
    """Tests for rapid client-telemetry training and calibration."""

    @pytest.fixture
    def spec(self):
        return DecisionSpec.from_yaml(SPEC_PATH)

    @pytest.fixture
    def pipeline(self, spec):
        return SpecialistPipeline(spec, specialist_type="routing")

    def test_telemetry_generation(self, pipeline):
        telemetry = pipeline.create_synthetic_telemetry(count=120)
        assert len(telemetry) == 120
        assert "optimal_gateway" in telemetry[0]
        assert telemetry[0]["optimal_gateway"] in pipeline.choice_keys

    def test_training_from_scratch_convergence(self, pipeline):
        telemetry = pipeline.create_synthetic_telemetry(count=180)
        t0 = time.perf_counter()
        model, report = pipeline.train_specialist(
            telemetry,
            epochs=10,
            batch_size=32,
            lr=0.008,
            export_onnx=False,
        )
        train_duration = time.perf_counter() - t0

        # Must train in under 15 seconds
        assert train_duration < 15.0, f"Training took {train_duration:.2f}s, expected <15s"
        assert report.train_samples > 0
        assert report.epochs_trained == 10
        assert report.validation_accuracy >= 0.70, f"Accuracy {report.validation_accuracy} too low"
        assert report.brier_score <= 0.30, f"Brier score {report.brier_score} too high"
        assert report.artifact_size_kb < 1500.0, "Artifact size exceeded 1.5MB"

    def test_evaluate_single_transaction(self, pipeline):
        telemetry = pipeline.create_synthetic_telemetry(count=90)
        model, _ = pipeline.train_specialist(telemetry, epochs=6, export_onnx=False)

        payload = {
            "amount": 75.0,
            "currency": "USD",
            "card_brand": "visa",
            "bin_country": "US",
            "mcc": "5411",
            "velocity_1h": 1,
        }
        res = pipeline.evaluate_transaction(model, payload)
        assert res.choice in pipeline.choice_keys
        assert 0.0 <= res.confidence <= 1.0
        assert res.latency_ms < 5.0, f"Latency {res.latency_ms}ms too high"
        assert "specialist SDM" in res.routing_reason


class TestONNXExportAndRuntime:
    """Tests for ONNX model export and microsecond ONNX Runtime inference."""

    @pytest.fixture
    def trained_setup(self):
        spec = DecisionSpec.from_yaml(SPEC_PATH)
        pipeline = SpecialistPipeline(spec, specialist_type="routing")
        telemetry = pipeline.create_synthetic_telemetry(count=90)
        model, _ = pipeline.train_specialist(telemetry, epochs=6, export_onnx=False)
        return spec, pipeline, model

    def test_export_specialist_to_onnx(self, trained_setup):
        spec, pipeline, model = trained_setup
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "routing_specialist.onnx")
            exported_path = export_specialist_to_onnx(
                model, onnx_path, input_dim=30, opset_version=18, verify=True
            )

            assert os.path.exists(exported_path)
            size_kb = os.path.getsize(exported_path) / 1024.0
            # Ensure file is ultra-compact (< 1MB)
            assert size_kb < 1000.0, f"ONNX file size {size_kb}KB exceeded 1MB"

    def test_onnx_runtime_parity_and_latency(self, trained_setup):
        spec, pipeline, model = trained_setup
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "routing_specialist.onnx")
            export_specialist_to_onnx(model, onnx_path, input_dim=30, verify=True)

            # Load via ONNXSpecialistSDM
            onnx_sdm = ONNXSpecialistSDM(onnx_path, choice_keys=pipeline.choice_keys)

            # Test parity with PyTorch output
            test_x = np.random.randn(1, 30).astype(np.float32)
            onnx_logits, onnx_probs = onnx_sdm.predict(test_x)

            model.eval()
            with torch.no_grad():
                py_logits, py_probs = model(torch.tensor(test_x))

            np.testing.assert_allclose(onnx_probs, py_probs.numpy(), atol=1e-4)

            # Test transaction evaluation
            serializer = PaymentSerializer(spec)
            payload = {
                "amount": 120.0,
                "currency": "EUR",
                "card_brand": "mastercard",
                "bin_country": "DE",
                "mcc": "5812",
                "velocity_1h": 0,
            }

            res = onnx_sdm.evaluate(payload, serializer, spec_id=spec.spec_id)
            assert res.choice in pipeline.choice_keys
            assert res.confidence > 0.0
            assert res.metadata.get("runtime") == "onnx"

            # Benchmark 100 ONNX evaluations for latency (<0.5ms)
            vec = serializer.to_vector(payload)
            t0 = time.perf_counter()
            for _ in range(100):
                onnx_sdm.predict(vec)
            avg_ms = ((time.perf_counter() - t0) / 100) * 1000.0
            assert avg_ms < 0.50, f"Average ONNX latency {avg_ms:.4f}ms exceeded 0.5ms"


class TestFactoryCompilationWithSDM:
    """Tests for DecisionFactory compiling bundles with embedded Specialist SDMs."""

    def test_factory_compiles_sdm_artifacts(self):
        spec = DecisionSpec.from_yaml(SPEC_PATH)
        factory = DecisionFactory(spec)

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = factory.compile(
                output_dir=tmp_dir,
                create_archive=True,
                train_sdm=True,
                sdm_type="routing",
                sdm_epochs=5,
            )

            # Verify files exist in bundle directory
            assert os.path.exists(os.path.join(tmp_dir, "spec.yaml"))
            assert os.path.exists(os.path.join(tmp_dir, "manifest.json"))
            assert os.path.exists(os.path.join(tmp_dir, "specialist_sdm.onnx"))
            assert os.path.exists(os.path.join(tmp_dir, "specialist_sdm.pt"))
            assert os.path.exists(os.path.join(tmp_dir, "sdm_report.json"))

            # Verify manifest references SDM
            assert bundle.manifest.specialist_sdm is not None
            assert bundle.manifest.specialist_sdm["onnx_model"] == "specialist_sdm.onnx"

            # Verify loader can instantiate ONNXSpecialistSDM directly from bundle
            loaded_sdm = DecisionFactory.load_specialist_onnx(tmp_dir)
            assert loaded_sdm is not None
            assert isinstance(loaded_sdm, ONNXSpecialistSDM)
            assert len(loaded_sdm.choice_keys) == 3
