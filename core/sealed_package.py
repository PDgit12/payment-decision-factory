"""
Sealed Closed-Source Distribution Architecture (.sdmpkg)
=========================================================
Enterprise IP protection and secure model distribution format.
Encrypts compiled Specialist SDMs (ONNX/PyTorch) using AES-256-GCM authenticated encryption,
manifest cryptographic binding, and license authentication.
Ensures zero plaintext model bytes are ever written to client disks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Prevent local code.py shadowing Python standard library module
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code) and ("code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")):
    import importlib.util
    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.engine import DecisionResult


class SealedPackageSecurityError(Exception):
    """Raised when authentication tag fails, checksum mismatches, or package is tampered."""
    pass


class LicenseExpiredError(SealedPackageSecurityError):
    """Raised when package license validity timestamp has expired."""
    pass


class LicenseAuthenticationError(SealedPackageSecurityError):
    """Raised when invalid license key is provided."""
    pass


@dataclass
class DecryptedPackageContext:
    """In-memory decrypted model artifact context with zero disk persistence."""
    package_id: str
    client_id: str
    domain: str
    model_type: str
    model_bytes: bytes
    manifest: Dict[str, Any]
    choices: List[str]
    input_dim: int

    def create_inference_session(self) -> Any:
        """
        Instantiate an ONNX Runtime InferenceSession directly from memory.
        No plaintext file is ever written to disk.
        """
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            self.model_bytes,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        return session

    def create_runner(self) -> SealedModelRunner:
        """Convenience method to instantiate a ready-to-evaluate runner."""
        session = self.create_inference_session()
        return SealedModelRunner(
            session=session,
            choices=self.choices,
            domain=self.domain,
            input_dim=self.input_dim,
            package_id=self.package_id,
        )

    def __enter__(self) -> SealedModelRunner:
        return self.create_runner()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SealedModelRunner:
    """Microsecond execution runner for in-memory decrypted models."""

    def __init__(
        self,
        session: Any,
        choices: List[str],
        domain: str,
        input_dim: int,
        package_id: str
    ):
        self.session = session
        self.choices = choices
        self.domain = domain
        self.input_dim = input_dim
        self.package_id = package_id
        self.input_name = self.session.get_inputs()[0].name
        self.idx_to_choice = {i: c for i, c in enumerate(self.choices)}

    def predict(self, feature_vector: Union[List[float], Any]) -> Tuple[Any, Any]:
        """Runs fast in-memory inference returning (logits, probabilities)."""
        import numpy as np
        x = np.asarray(feature_vector, dtype=np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        # Resilient dimension alignment
        if x.shape[1] < self.input_dim:
            pad = np.zeros((x.shape[0], self.input_dim - x.shape[1]), dtype=np.float32)
            x = np.hstack([x, pad])
        elif x.shape[1] > self.input_dim:
            x = x[:, :self.input_dim]

        outputs = self.session.run(None, {self.input_name: x})
        if len(outputs) >= 2:
            return outputs[0], outputs[1]
        else:
            logits = outputs[0]
            exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            return logits, probs

    def evaluate(self, feature_vector: Union[List[float], Any]) -> Dict[str, Any]:
        """Runs evaluation and formats structured result with latency measurement."""
        import numpy as np
        t0 = time.perf_counter()
        logits, probs = self.predict(feature_vector)
        probs_1d = probs[0]
        logits_1d = logits[0]

        best_idx = int(np.argmax(probs_1d))
        confidence = float(probs_1d[best_idx])
        best_choice = self.idx_to_choice.get(best_idx, f"choice_{best_idx}")

        prob_dict = {
            self.idx_to_choice.get(i, f"choice_{i}"): float(round(probs_1d[i], 4))
            for i in range(len(probs_1d))
        }
        score_dict = {
            self.idx_to_choice.get(i, f"choice_{i}"): float(round(logits_1d[i], 4))
            for i in range(len(logits_1d))
        }

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "package_id": self.package_id,
            "choice": best_choice,
            "confidence": confidence,
            "probabilities": prob_dict,
            "scores": score_dict,
            "latency_ms": round(latency_ms, 3),
            "is_sealed": True,
            "domain": self.domain,
        }


class SealedPackage:
    """
    Sealed Model Package (.sdmpkg) Container.
    Handles AES-256-GCM encryption, manifest cryptographic binding, and client license unlocking.
    """

    FORMAT_VERSION = "SDMPKG/1.0"
    KDF_ITERATIONS = 100_000

    def __init__(
        self,
        manifest: Dict[str, Any],
        crypto: Dict[str, Any],
        ciphertext_b64: str,
    ):
        self.manifest = manifest
        self.crypto = crypto
        self.ciphertext_b64 = ciphertext_b64

    @classmethod
    def _derive_key(cls, license_key: str, salt: bytes) -> bytes:
        """Derive 256-bit AES key using PBKDF2-HMAC-SHA256."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=cls.KDF_ITERATIONS,
        )
        return kdf.derive(license_key.encode("utf-8"))

    @classmethod
    def seal(
        cls,
        model_bytes_or_path: Union[bytes, str],
        license_key: str,
        client_id: str,
        domain: str = "payment_routing",
        model_type: str = "specialist_sdm",
        spec_dict: Optional[Dict[str, Any]] = None,
        choices: Optional[List[str]] = None,
        input_dim: int = 30,
        validity_days: int = 365,
        output_path: Optional[str] = None,
        tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SealedPackage:
        """
        Encrypt model into a tamper-evident sealed package (.sdmpkg).

        Args:
            model_bytes_or_path: Plaintext ONNX bytes or file path to .onnx model.
            license_key: Secret client license string.
            client_id: Authorized enterprise client identifier.
            domain: Target business domain (e.g. payment_routing, credit_scoring).
            model_type: Model identifier.
            spec_dict: Optional declarative specification dictionary.
            choices: List of decision output classes.
            input_dim: Number of input features.
            validity_days: License validity duration in days.
            output_path: Destination path ending with .sdmpkg.
            tier: Execution tier (compact, sub_billion, enterprise).
            metadata: Optional dictionary of audit and model metadata.
        """
        if isinstance(model_bytes_or_path, str):
            # Check if external data needs to be embedded first
            import onnx
            model_proto = onnx.load(model_bytes_or_path, load_external_data=True)
            raw_bytes = model_proto.SerializeToString()
        else:
            raw_bytes = model_bytes_or_path

        # 1. SHA-256 Checksum of plaintext model
        sha256_checksum = hashlib.sha256(raw_bytes).hexdigest()

        # 2. Setup Manifest
        now = time.time()
        expires_at = now + (validity_days * 86400)
        package_id = f"sdmpkg_{uuid.uuid4().hex[:12]}"

        pkg_tier = tier or (spec_dict.get("tier") if spec_dict and isinstance(spec_dict, dict) and "tier" in spec_dict else None)
        if not pkg_tier and metadata and isinstance(metadata, dict) and "tier" in metadata:
            pkg_tier = str(metadata["tier"])

        manifest = {
            "format": cls.FORMAT_VERSION,
            "package_id": package_id,
            "client_id": client_id,
            "domain": domain,
            "model_type": model_type,
            "created_at": round(now, 2),
            "expires_at": round(expires_at, 2),
            "sha256_checksum": sha256_checksum,
            "input_dim": input_dim,
            "num_classes": len(choices) if choices else 3,
            "choices": choices or ["choice_0", "choice_1", "choice_2"],
            "spec": spec_dict or {},
        }
        if pkg_tier:
            manifest["tier"] = str(pkg_tier)
        if metadata:
            manifest["metadata"] = metadata

        # 3. Cryptographic Nonce, Salt, and Key Derivation
        salt = os.urandom(16)
        iv = os.urandom(12)  # 96-bit standard nonce for AES-GCM
        key = cls._derive_key(license_key, salt)

        # 4. Authenticated Encryption with Associated Data (AAD)
        # Bind the manifest as AAD so any tampering with manifest values breaks decryption
        aad = json.dumps(manifest, sort_keys=True).encode("utf-8")
        aesgcm = AESGCM(key)
        ciphertext_with_tag = aesgcm.encrypt(iv, raw_bytes, aad)

        # Separate ciphertext and 16-byte authentication tag
        tag = ciphertext_with_tag[-16:]
        ciphertext = ciphertext_with_tag[:-16]

        crypto = {
            "cipher": "AES-256-GCM",
            "kdf": "PBKDF2-HMAC-SHA256",
            "kdf_iterations": cls.KDF_ITERATIONS,
            "salt_hex": salt.hex(),
            "iv_hex": iv.hex(),
            "tag_hex": tag.hex(),
        }

        ciphertext_b64 = base64.b64encode(ciphertext).decode("utf-8")

        pkg = cls(manifest=manifest, crypto=crypto, ciphertext_b64=ciphertext_b64)

        if output_path:
            pkg.save(output_path)

        return pkg

    def to_dict(self) -> Dict[str, Any]:
        """Convert sealed package to serializable dictionary."""
        return {
            "format": self.FORMAT_VERSION,
            "manifest": self.manifest,
            "crypto": self.crypto,
            "ciphertext": self.ciphertext_b64,
        }

    def save(self, file_path: str):
        """Save sealed package to a .sdmpkg file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SealedPackage:
        """Instantiate a SealedPackage from a dictionary representation."""
        if data.get("format") != cls.FORMAT_VERSION:
            raise SealedPackageSecurityError(f"Unsupported package format: {data.get('format')}")

        return cls(
            manifest=data["manifest"],
            crypto=data["crypto"],
            ciphertext_b64=data["ciphertext"],
        )

    @classmethod
    def load_file(cls, file_path: str) -> SealedPackage:
        """Load a .sdmpkg file into a SealedPackage instance."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def unlock(self, license_key: str) -> DecryptedPackageContext:
        """Alias for decrypt to support `with sealed.unlock(key) as runner:` syntax."""
        return self.decrypt(license_key)

    def decrypt(self, license_key: str) -> DecryptedPackageContext:
        """
        Verify license authentication, validate tamper-evident signatures,
        and decrypt model bytes strictly into memory.
        """
        # 1. License Expiration Check
        now = time.time()
        expires_at = self.manifest.get("expires_at", 0)
        if now > expires_at:
            raise LicenseExpiredError(
                f"Package license expired at timestamp {expires_at} (current: {now:.0f})"
            )

        # 2. Re-derive AES key
        salt = bytes.fromhex(self.crypto["salt_hex"])
        iv = bytes.fromhex(self.crypto["iv_hex"])
        tag = bytes.fromhex(self.crypto["tag_hex"])
        key = self._derive_key(license_key, salt)

        # 3. Decrypt with Authenticated Associated Data
        ciphertext = base64.b64decode(self.ciphertext_b64)
        ciphertext_with_tag = ciphertext + tag
        aad = json.dumps(self.manifest, sort_keys=True).encode("utf-8")

        aesgcm = AESGCM(key)
        try:
            plaintext = aesgcm.decrypt(iv, ciphertext_with_tag, aad)
        except Exception as e:
            raise LicenseAuthenticationError(
                f"License authentication failed: invalid license key or tampered package envelope ({e})"
            )

        # 4. SHA-256 Checksum Verification
        actual_checksum = hashlib.sha256(plaintext).hexdigest()
        expected_checksum = self.manifest["sha256_checksum"]
        if actual_checksum != expected_checksum:
            raise SealedPackageSecurityError(
                f"Integrity checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
            )

        return DecryptedPackageContext(
            package_id=self.manifest["package_id"],
            client_id=self.manifest["client_id"],
            domain=self.manifest["domain"],
            model_type=self.manifest["model_type"],
            model_bytes=plaintext,
            manifest=self.manifest,
            choices=self.manifest.get("choices", []),
            input_dim=self.manifest.get("input_dim", 30),
        )
