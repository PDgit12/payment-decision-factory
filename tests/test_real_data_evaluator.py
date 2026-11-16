"""
Unit and Integration Tests for Real-Data Digital Twin Evaluator.
================================================================
Verifies real-data telemetry ingestion, empirical ROC-AUC, Brier score,
MPS/CPU latency distribution, and SHA-256 Readiness Certificate generation.
"""

import hashlib
import json
import os
import sys
import tempfile
import importlib.util
import pytest
import numpy as np

# Standard library protection against local code.py shadowing
if "code" not in sys.modules or not hasattr(sys.modules["code"], "InteractiveConsole"):
    stdlib_code = "/opt/anaconda3/lib/python3.13/code.py"
    if os.path.exists(stdlib_code):
        spec = importlib.util.spec_from_file_location("code", stdlib_code)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["code"] = mod
            spec.loader.exec_module(mod)

from payment_decision_factory.sandbox.real_data_evaluator import (
    RealDataDigitalTwinEvaluator,
    RealDataEvaluationMetrics,
    compute_expected_calibration_error,
    run_and_save_real_data_certification,
)


class TestRealDataDigitalTwinEvaluator:
    """Test suite verifying 100% real mathematical evaluation on real transaction data."""

    def test_evaluator_data_loading(self):
        evaluator = RealDataDigitalTwinEvaluator()
        assert evaluator.df is not None
        assert len(evaluator.df) == 10000
        assert len(evaluator.feature_cols) == 30
        assert "Class" in evaluator.df.columns
        assert evaluator.X_norm.shape == (10000, 30)
        assert len(evaluator.y_true) == 10000

    def test_expected_calibration_error_calculation(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_prob = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_run_evaluation_metrics_and_certificate(self):
        evaluator = RealDataDigitalTwinEvaluator()
        metrics, cert = evaluator.run_evaluation()

        # Empirical quality assertions
        assert metrics.total_transactions == 10000
        assert metrics.fraud_count == 38
        assert metrics.legitimate_count == 9962
        assert metrics.roc_auc > 0.90
        assert metrics.pr_auc > 0.50
        assert metrics.brier_score < 0.20
        assert metrics.sla_adherence_pct >= 99.0
        assert metrics.p95_latency_ms < 5.0
        assert metrics.throughput_tps > 1000.0
        assert metrics.resilience_score > 0.90

        # Certificate structure & status
        assert cert["is_certified"] is True
        assert cert["status"] == "CERTIFIED_ENTERPRISE_READY"
        assert cert["hardware_device"] in ["MPS", "CUDA", "CPU"]
        assert len(cert["sha256_audit_seal"]) == 64

    def test_sha256_audit_seal_tamper_evidence(self):
        evaluator = RealDataDigitalTwinEvaluator()
        _, cert = evaluator.run_evaluation()

        payload = cert["verification_payload"]
        canonical_str = json.dumps(payload, sort_keys=True)
        expected_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        assert cert["sha256_audit_seal"] == expected_hash

        # Tampering with any value must break the seal
        tampered_payload = dict(payload)
        tampered_payload["roc_auc"] = 0.5000
        tampered_str = json.dumps(tampered_payload, sort_keys=True)
        tampered_hash = hashlib.sha256(tampered_str.encode("utf-8")).hexdigest()
        assert tampered_hash != cert["sha256_audit_seal"]

    def test_convenience_runner_saves_certificate(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            metrics, cert = run_and_save_real_data_certification(output_cert_path=tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path, "r") as f:
                loaded = json.load(f)
            assert loaded["certificate_id"] == cert["certificate_id"]
            assert loaded["sha256_audit_seal"] == cert["sha256_audit_seal"]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
