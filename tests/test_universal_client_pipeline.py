"""
Unit & Integration Tests for Universal Client-Facing Pipeline & Ephemeral GPU Rental
"""

import os
import sys
import pytest

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.universal_client_pipeline import (
    CLOUD_GPU_RENTAL_CATALOG,
    ClientPipelineIntent,
    UniversalClientPipelineEngine,
    UniversalDataProfiler,
)


def test_universal_data_profiler(tmp_path):
    # Create an arbitrary client dataset with diverse columns
    csv_path = tmp_path / "client_arbitrary_data.csv"
    csv_content = """transaction_id,amount,currency,card_network,user_age,country,is_fraud
tx_101,45.50,EUR,visa,32,FR,0
tx_102,1200.00,USD,mastercard,45,US,1
tx_103,15.20,EUR,visa,22,DE,0
tx_104,89.90,GBP,amex,51,GB,0
tx_105,450.00,USD,visa,29,US,0
tx_106,990.00,EUR,mastercard,38,FR,1
tx_107,12.00,USD,visa,64,US,0
tx_108,310.00,EUR,visa,19,ES,0
"""
    csv_path.write_text(csv_content, encoding="utf-8")

    X, y, profile = UniversalDataProfiler.profile_csv(str(csv_path), target_column_hint="is_fraud")

    assert profile.total_rows == 8
    assert "amount" in profile.continuous_features
    assert "country" in profile.categorical_features or "card_network" in profile.categorical_features
    assert profile.target_column == "is_fraud"
    assert len(profile.target_classes) == 2
    assert X.shape[0] == 8
    assert len(y) == 8


def test_ephemeral_gpu_catalog():
    assert "runpod_rtx4090" in CLOUD_GPU_RENTAL_CATALOG
    assert "aws_spot_g5" in CLOUD_GPU_RENTAL_CATALOG
    assert "mac_mps_local" in CLOUD_GPU_RENTAL_CATALOG

    runpod = CLOUD_GPU_RENTAL_CATALOG["runpod_rtx4090"]
    assert runpod.hourly_rate_usd == 0.44
    assert runpod.vram_gb == 24


def test_universal_client_pipeline_end_to_end(tmp_path):
    # Use real credit card sample
    data_csv = os.path.join(_ROOT_DIR, "data", "creditcard_sample.csv")
    if not os.path.exists(data_csv):
        pytest.skip("Sample credit card data not found")

    out_dir = str(tmp_path / "universal_out")
    intent = ClientPipelineIntent(
        client_id="acme_payments_inc",
        pipeline_name="Smart Routing & Fraud Shield",
        archetype="smart_routing",
        parameter_tier="small",
        compute_provider="mac_mps_local",
        output_dir=out_dir,
    )

    engine = UniversalClientPipelineEngine(intent=intent)
    receipt = engine.run(dataset_path=data_csv, epochs=1, batch_size=64)

    assert receipt.status == "CERTIFIED_ENTERPRISE_READY"
    assert os.path.exists(receipt.sealed_package_path)
    assert receipt.client_license_key.startswith("lic_acme_payments_inc_")
    assert receipt.validation_accuracy > 0.90
    assert receipt.empirical_economic_lift_bps > 0
    assert "python" in receipt.integration_snippets
    assert "typescript" in receipt.integration_snippets
