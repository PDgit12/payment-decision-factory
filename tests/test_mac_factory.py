"""
Unit & Integration Tests for Mac GPU Specialist Factory
"""

import os
import sys
import pytest

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.mac_factory import (
    MacFactoryConfig,
    MacHardwareProfile,
    MacSpecialistFactory,
)


def test_mac_hardware_profile_detection():
    profile = MacHardwareProfile.detect()
    assert profile.chip_model is not None
    assert profile.total_ram_gb > 0
    assert profile.optimal_device in ["mps", "cuda", "cpu"]
    assert profile.pytorch_version is not None


def test_mac_specialist_factory_dry_run(tmp_path):
    data_csv = os.path.join(_ROOT_DIR, "data", "creditcard_sample.csv")
    if not os.path.exists(data_csv):
        pytest.skip("Sample credit card data not found")

    out_dir = str(tmp_path / "mac_test_out")
    cfg = MacFactoryConfig(
        client_id="unit_test_client",
        parameter_tier="small",
        dataset_mode="sample",
        data_path=data_csv,
        epochs=1,
        batch_size=64,
        output_dir=out_dir,
    )

    factory = MacSpecialistFactory(config=cfg)
    receipt = factory.run()

    assert receipt.status in ["CERTIFIED_ENTERPRISE_READY", "CERTIFIED_WITH_WARNINGS"]
    assert os.path.exists(receipt.sealed_package_path)
    assert os.path.exists(receipt.onnx_path)
    assert receipt.p99_latency_ms < 15.0
    assert receipt.parameter_count > 100_000
