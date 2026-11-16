"""
Unit Tests for Sealed Package (.sdmpkg) Architecture & In-Memory Decryption
=============================================================================
Tests:
- AES-256-GCM encryption with PBKDF2-HMAC-SHA256 key derivation
- Manifest cryptographic binding (Associated Data verification)
- In-memory decryption with zero plaintext disk leakage
- License expiration enforcement
- Tamper detection and invalid key rejection
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

from payment_decision_factory.core.sdm import SpecialistSDM, export_specialist_to_onnx
from payment_decision_factory.core.sealed_package import (
    SealedPackage,
    LicenseAuthenticationError,
    LicenseExpiredError,
    SealedPackageSecurityError,
)


@pytest.fixture(scope="module")
def sample_onnx_path():
    """Build a temporary valid ONNX model file for packaging tests."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name

    sdm = SpecialistSDM("routing", num_continuous_features=30, num_classes=3)
    sdm.eval()
    export_specialist_to_onnx(sdm, path, input_dim=30, verify=False)
    yield path

    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(path + ".data"):
        os.remove(path + ".data")


class TestSealedPackage:
    """Tests for .sdmpkg sealed distribution format."""

    def test_seal_and_in_memory_decrypt(self, sample_onnx_path):
        license_key = "LIC-TEST-ENTERPRISE-2026-X1"
        client_id = "test_merchant"

        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg_path = os.path.join(tmp_dir, "test.sdmpkg")

            # Seal model into .sdmpkg
            pkg = SealedPackage.seal(
                model_bytes_or_path=sample_onnx_path,
                license_key=license_key,
                client_id=client_id,
                domain="payment_routing",
                choices=["stripe_us", "adyen_eu", "checkout_crossborder"],
                input_dim=30,
                validity_days=30,
                output_path=pkg_path,
            )

            assert os.path.exists(pkg_path)
            assert pkg.manifest["client_id"] == client_id
            assert pkg.manifest["domain"] == "payment_routing"
            assert pkg.crypto["cipher"] == "AES-256-GCM"

            # Load package and decrypt strictly in-memory
            loaded_pkg = SealedPackage.load_file(pkg_path)
            ctx = loaded_pkg.decrypt(license_key)

            assert ctx.client_id == client_id
            assert ctx.package_id == pkg.manifest["package_id"]
            assert len(ctx.model_bytes) > 0

            # Execute in-memory without writing to disk
            runner = ctx.create_runner()
            test_x = np.random.randn(1, 30).astype(np.float32)
            res = runner.evaluate(test_x)

            assert res["choice"] in ["stripe_us", "adyen_eu", "checkout_crossborder"]
            assert 0.0 <= res["confidence"] <= 1.0
            assert res["is_sealed"] is True
            assert res["latency_ms"] < 5.0

    def test_wrong_license_key_rejected(self, sample_onnx_path):
        license_key = "LIC-CORRECT-KEY"
        pkg = SealedPackage.seal(
            model_bytes_or_path=sample_onnx_path,
            license_key=license_key,
            client_id="client_a",
            choices=["choice_1", "choice_2"],
        )

        with pytest.raises(LicenseAuthenticationError, match="License authentication failed"):
            pkg.decrypt("LIC-WRONG-KEY")

    def test_tampered_manifest_rejected(self, sample_onnx_path):
        license_key = "LIC-CORRECT-KEY"
        pkg = SealedPackage.seal(
            model_bytes_or_path=sample_onnx_path,
            license_key=license_key,
            client_id="client_a",
            choices=["choice_1", "choice_2"],
        )

        # Tamper with the manifest (e.g. change client_id or expiration)
        pkg.manifest["client_id"] = "hacked_client"

        with pytest.raises(LicenseAuthenticationError):
            pkg.decrypt(license_key)

    def test_expired_license_rejected(self, sample_onnx_path):
        license_key = "LIC-CORRECT-KEY"
        pkg = SealedPackage.seal(
            model_bytes_or_path=sample_onnx_path,
            license_key=license_key,
            client_id="client_a",
            choices=["choice_1", "choice_2"],
            validity_days=-1,  # Expired yesterday
        )

        with pytest.raises(LicenseExpiredError, match="Package license expired"):
            pkg.decrypt(license_key)
