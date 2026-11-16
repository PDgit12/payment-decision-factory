"""
Unit & Integration Tests for Real-Data Tabular Transformer Pipeline
===================================================================
Verifies:
1. Device selection (Apple Silicon MPS / CUDA / CPU)
2. Ingestion and normalization of real ULB credit card benchmark telemetry
3. Real backpropagation and loss optimization (Zero Mocks)
4. Post-training calibration and metrics
5. Export to ONNX and verification with ONNX Runtime
6. Cryptographic AES-256-GCM sealing and in-memory zero-disk decryption
"""

from __future__ import annotations

import os
import sys
import tempfile
import pytest
import numpy as np
import torch

# Prevent local code.py shadowing Python standard library module
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code) and ("code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")):
    import importlib.util
    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.pipeline.real_data_pipeline import (
    get_best_compute_device,
    load_real_creditcard_data,
    compute_expected_calibration_error,
    train_flagship_sub_billion_model,
    build_real_data_enterprise_package,
    run_full_real_data_pipeline,
)
from payment_decision_factory.core.sealed_package import SealedPackage


class TestRealDataPipeline:
    """Comprehensive test suite for the Real-Data Tabular Transformer pipeline."""

    def test_compute_device_detection(self):
        """Verifies compute device discovery and MPS/CUDA/CPU detection."""
        device, info = get_best_compute_device()
        assert isinstance(device, torch.device)
        assert info["device_type"] in ["mps", "cuda", "cpu"]
        assert len(info["compute_description"]) > 0
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            assert info["device_type"] == "mps"
            assert info["mps_available"] is True

    def test_load_real_creditcard_data(self):
        """Verifies real dataset ingestion, feature extraction, and normalization."""
        X, y, feat_names = load_real_creditcard_data(max_records=1000)
        assert len(X) == 1000
        assert len(y) == 1000
        assert len(feat_names) == 30
        assert X.shape[1] == 30
        # Verify normalization mean is near zero
        assert np.all(np.abs(np.mean(X, axis=0)) < 0.2)

    def test_expected_calibration_error_calculation(self):
        """Verifies ECE math on probability distributions."""
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.1, 0.9],
        ], dtype=np.float32)
        labels = np.array([0, 0, 0, 0, 1], dtype=np.int64)
        ece = compute_expected_calibration_error(probs, labels, num_bins=5)
        assert 0.0 <= ece <= 1.0

    def test_train_flagship_sub_billion_model(self):
        """Verifies real backprop training and calibration on real data."""
        X, y, _ = load_real_creditcard_data(max_records=400)
        X_train, y_train = X[:300], y[:300]
        X_val, y_val = X[300:], y[300:]

        model, metrics = train_flagship_sub_billion_model(
            X_train, y_train, X_val, y_val,
            epochs=1, batch_size=64, parameter_target="small"
        )
        assert model is not None
        assert metrics["parameters"] > 100_000
        assert "validation_accuracy" in metrics
        assert "brier_score" in metrics

    def test_build_real_data_enterprise_package(self):
        """Verifies end-to-end build, certification, and sealed package generation."""
        manifest = build_real_data_enterprise_package(
            max_records=300,
            epochs=1,
            parameter_target="small"
        )
        assert manifest["client_name"] == "Enterprise Real Payment Corp"
        assert os.path.exists(manifest["onnx_path"])
        assert os.path.exists(manifest["sealed_package_path"])
        assert manifest["certification_status"] in ["CERTIFIED", "CONDITIONALLY_CERTIFIED", "NEEDS_OPTIMIZATION", "CERTIFIED_ENTERPRISE_READY"]
        assert len(manifest["sha256_audit_hash"]) == 64

        # Test in-memory decryption
        pkg = SealedPackage.load_file(manifest["sealed_package_path"])
        with pkg.unlock("lic_real_data_enterprise_2026") as runner:
            sample_vec = np.zeros(30, dtype=np.float32)
            res = runner.evaluate(sample_vec)
            assert res["choice"] in ["authorized", "declined"]
            assert res["latency_ms"] < 15.0
