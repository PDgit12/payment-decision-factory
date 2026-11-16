"""Sealed Package (.sdmpkg) In-Process Client for Python SDK.

Supports loading and evaluating encrypted Specialist SDMs with zero plaintext model bytes
ever written to client storage. Verifies client licensing and integrity checks in-memory.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from payment_decision_sdk.exceptions import (
    DecisionSDKError,
    EngineLoadError,
    InvalidContextError,
)
from payment_decision_sdk.models import DecisionResult, PaymentContext


class SealedPackageClient:
    """In-process evaluation client for encrypted .sdmpkg model packages."""

    def __init__(
        self,
        package_path: Union[str, Path],
        license_key: str,
        client_id: Optional[str] = None,
    ):
        """Initialize and unlock the sealed model package in-memory.

        Args:
            package_path: Path to the .sdmpkg sealed package file.
            license_key: Client secret license key string.
            client_id: Optional client ID to verify entitlement.
        """
        self.package_path = Path(package_path)
        self.license_key = license_key
        self.client_id = client_id

        self.context = None
        self.runner = None
        self.manifest = {}
        self.choices = []
        self.domain = "unknown"
        self.package_id = "unknown"
        self.tier = "compact"
        self.fallback_decision = "stripe_us"

        self._unlock_package()
        self._warmup_engine()

    def _warmup_engine(self) -> None:
        """Run a single warmup forward pass to pre-allocate memory arenas and eliminate cold starts."""
        try:
            input_dim = self.context.input_dim if self.context else 30
            dummy_vec = np.zeros(input_dim, dtype=np.float32)
            self.runner.evaluate(dummy_vec)
        except Exception:
            pass

    def _unlock_package(self) -> None:
        """Decrypt package strictly in memory and instantiate inference runner."""
        if not self.package_path.exists():
            raise EngineLoadError(f"Sealed package file not found at: {self.package_path}")

        try:
            from payment_decision_factory.core.sealed_package import (
                LicenseAuthenticationError,
                LicenseExpiredError,
                SealedPackage,
                SealedPackageSecurityError,
            )

            pkg = SealedPackage.load_file(str(self.package_path))
            self.context = pkg.decrypt(self.license_key)

            if self.client_id and self.context.client_id != self.client_id:
                raise LicenseAuthenticationError(
                    f"Client entitlement mismatch: package licensed for '{self.context.client_id}', got '{self.client_id}'"
                )

            self.manifest = self.context.manifest
            self.choices = self.context.choices
            if self.choices:
                self.fallback_decision = self.choices[0]
            self.domain = self.context.domain
            self.package_id = self.context.package_id
            self.tier = self.manifest.get("tier") or (self.manifest.get("spec", {}).get("tier", "compact") if isinstance(self.manifest.get("spec"), dict) else "compact")
            self.runner = self.context.create_runner()

        except Exception as e:
            raise DecisionSDKError(f"Failed to unlock sealed package {self.package_path.name}: {e}") from e

    @property
    def is_enterprise_tier(self) -> bool:
        """Returns True if the package is compiled for Enterprise Sub-Billion SDM tier."""
        return self.tier in ("sub_billion", "enterprise")

    def _vectorize(self, payload: Dict[str, Any]) -> np.ndarray:
        """Vectorize payload matching model input dimension."""
        # 1. Direct pre-computed feature vectors
        if "feature_vector" in payload and payload["feature_vector"] is not None:
            vec = np.asarray(payload["feature_vector"], dtype=np.float32)
            if self.context and self.context.input_dim:
                if len(vec) < self.context.input_dim:
                    vec = np.pad(vec, (0, self.context.input_dim - len(vec)), mode="constant")
                elif len(vec) > self.context.input_dim:
                    vec = vec[:self.context.input_dim]
            return vec
        if "features" in payload and payload["features"] is not None:
            vec = np.asarray(payload["features"], dtype=np.float32)
            if self.context and self.context.input_dim:
                if len(vec) < self.context.input_dim:
                    vec = np.pad(vec, (0, self.context.input_dim - len(vec)), mode="constant")
                elif len(vec) > self.context.input_dim:
                    vec = vec[:self.context.input_dim]
            return vec

        input_dim = self.context.input_dim if self.context else 30
        vec = np.zeros(input_dim, dtype=np.float32)

        amount = float(payload.get("amount", 50.0))
        vec[0] = np.log1p(max(0.0, amount)) / 10.0
        vec[1] = min(1.0, float(payload.get("velocity_1h", 1)) / 20.0)

        # Currency one-hot (indices 2..8)
        currencies = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]
        curr = str(payload.get("currency", "USD")).upper()
        if curr in currencies:
            vec[2 + currencies.index(curr)] = 1.0

        # Geo one-hot (indices 9..18)
        country = str(payload.get("bin_country", "US")).upper()
        eu = {"GB", "DE", "FR", "NL", "IE", "ES", "IT"}
        if country in ("US", "CA"):
            vec[9] = 1.0
        elif country in eu:
            vec[10] = 1.0
        else:
            vec[11] = 1.0

        # Card brand (indices 19..23)
        brand = str(payload.get("card_brand", "visa")).lower()
        brands = ["visa", "mastercard", "amex", "discover"]
        if brand in brands:
            vec[19 + brands.index(brand)] = 1.0

        # Enterprise high-dimensional feature support (e.g. MCC, custom telemetry)
        if input_dim > 24:
            mcc_val = payload.get("mcc")
            if mcc_val is not None:
                try:
                    vec[24] = (float(mcc_val) % 10000.0) / 10000.0
                except (ValueError, TypeError):
                    pass

            idx = 25
            for k, v in payload.items():
                if idx >= input_dim:
                    break
                if k not in ("amount", "velocity_1h", "currency", "bin_country", "card_brand", "mcc", "metadata"):
                    try:
                        vec[idx] = float(v)
                        idx += 1
                    except (ValueError, TypeError):
                        pass

        return vec

    def decide(
        self, context: Union[PaymentContext, Dict[str, Any], List[float], np.ndarray]
    ) -> DecisionResult:
        """Synchronously evaluate a payment transaction using the sealed model.

        Args:
            context: Validated PaymentContext, feature dictionary, or direct float array.

        Returns:
            Calibrated DecisionResult produced in-memory with sub-10ms (typically sub-1ms) latency.
        """
        t0 = time.perf_counter()

        if isinstance(context, (list, tuple, np.ndarray)):
            vec = np.asarray(context, dtype=np.float32)
            if self.context and self.context.input_dim:
                if len(vec) < self.context.input_dim:
                    vec = np.pad(vec, (0, self.context.input_dim - len(vec)), mode="constant")
                elif len(vec) > self.context.input_dim:
                    vec = vec[:self.context.input_dim]
        elif isinstance(context, dict):
            try:
                ctx = PaymentContext.from_dict(context)
                payload = ctx.to_dict()
            except Exception:
                # Fall back to raw telemetry dictionary for domain-agnostic or enterprise models
                payload = context
            vec = self._vectorize(payload)
        elif isinstance(context, PaymentContext):
            payload = context.to_dict()
            vec = self._vectorize(payload)
        else:
            raise InvalidContextError(f"Expected PaymentContext, dict, or feature array, got {type(context)}")

        # Evaluate via in-memory ONNX runner
        out = self.runner.evaluate(vec)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DecisionResult(
            decision=out["choice"],
            confidence=out["confidence"],
            probabilities=out["probabilities"],
            latency_ms=round(elapsed_ms, 3),
            routing_reason=f"Decided by encrypted sealed package [{self.package_id}] ({self.domain}, tier={self.tier})",
            feature_contributions={
                "sealed_in_memory": 1.0,
                "licensed_to": 1.0,
                "is_enterprise_tier": 1.0 if self.is_enterprise_tier else 0.0,
            },
            is_fallback=False,
            engine_mode="sealed_package",
            model_version=f"sealed-{self.package_id}",
        )

    def decide_safe(
        self,
        context: Union[PaymentContext, Dict[str, Any], List[float], np.ndarray],
        fallback_decision: Optional[str] = None,
    ) -> DecisionResult:
        """Zero-dropout fail-open evaluation wrapper. Guaranteed never to throw during checkout."""
        t0 = time.perf_counter()
        fb = fallback_decision or self.fallback_decision or (self.choices[0] if self.choices else "stripe_us")
        try:
            return self.decide(context)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DecisionResult(
                decision=fb,
                confidence=0.50,
                probabilities={fb: 1.0},
                latency_ms=round(elapsed_ms, 3),
                routing_reason=f"Zero-Dropout Fail-Open Fallback: Safe bypass triggered ({exc})",
                feature_contributions={"sealed_in_memory": 1.0, "is_fallback": 1.0},
                is_fallback=True,
                engine_mode="fail_open_fallback",
                model_version=f"sealed-{self.package_id}",
            )

    async def decide_safe_async(
        self,
        context: Union[PaymentContext, Dict[str, Any], List[float], np.ndarray],
        fallback_decision: Optional[str] = None,
    ) -> DecisionResult:
        """Asynchronously evaluate a payment transaction with zero-dropout fail-open protection."""
        return self.decide_safe(context, fallback_decision=fallback_decision)

    async def decide_async(
        self, context: Union[PaymentContext, Dict[str, Any], List[float], np.ndarray]
    ) -> DecisionResult:
        """Asynchronously evaluate a payment transaction."""
        return self.decide(context)

    def decide_batch(
        self, contexts: List[Union[PaymentContext, Dict[str, Any], List[float], np.ndarray]]
    ) -> List[DecisionResult]:
        """Evaluate a batch of payment transactions."""
        return [self.decide(c) for c in contexts]

    async def decide_batch_async(
        self, contexts: List[Union[PaymentContext, Dict[str, Any], List[float], np.ndarray]]
    ) -> List[DecisionResult]:
        """Asynchronously evaluate a batch of payment transactions."""
        return [self.decide(c) for c in contexts]

    def get_manifest(self) -> Dict[str, Any]:
        """Return the cryptographically verified package manifest."""
        return self.manifest
