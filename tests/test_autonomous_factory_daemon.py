"""
Unit & Integration Tests for Autonomous Factory Daemon
"""

import os
import sys
import pytest

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.orchestration.autonomous_factory_daemon import (
    AutonomousPaymentFactoryDaemon,
    FactoryEvent,
)


def test_autonomous_factory_daemon_end_to_end(tmp_path):
    data_csv = os.path.join(_ROOT_DIR, "data", "creditcard_sample.csv")
    if not os.path.exists(data_csv):
        pytest.skip("Sample credit card data not found")

    egress_dir = str(tmp_path / "dist")
    daemon = AutonomousPaymentFactoryDaemon(
        base_egress_dir=egress_dir, default_license_key="test-key-auto-2026"
    )

    events_captured = []

    def on_step(step):
        events_captured.append(step.event)

    receipt = daemon.process_client_data_event(
        client_id="acme_enterprise_test",
        data_source_path=data_csv,
        license_key="test-key-auto-2026",
        parameter_target="small",
        epochs=1,
        batch_size=64,
        on_step_callback=on_step,
    )

    assert receipt.status in ["CERTIFIED_ENTERPRISE_READY", "CERTIFIED_WITH_WARNINGS"]
    assert os.path.exists(receipt.sealed_package_path)
    assert receipt.sha256_audit_hash is not None
    assert receipt.p99_latency_ms < 15.0
    assert FactoryEvent.DATA_INGESTED in events_captured
    assert FactoryEvent.GPU_PROVISIONED in events_captured
    assert FactoryEvent.TRAINING_COMPLETED in events_captured
    assert FactoryEvent.DIGITAL_TWIN_CERTIFIED in events_captured
    assert FactoryEvent.PACKAGE_SEALED in events_captured
    assert FactoryEvent.DELIVERY_DISPATCHED in events_captured
