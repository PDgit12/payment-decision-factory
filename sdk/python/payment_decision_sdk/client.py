"""Unified Decision Engine Client.

Provides a single, idiomatic interface for fintech developers to evaluate payment routing
decisions either in-process (<1ms) or via sidecar (<20ms).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from payment_decision_sdk.exceptions import DecisionSDKError
from payment_decision_sdk.in_process import InProcessEngine
from payment_decision_sdk.models import DecisionResult, PaymentContext
from payment_decision_sdk.sdm import SpecialistSDMEvaluator
from payment_decision_sdk.sealed import SealedPackageClient
from payment_decision_sdk.sidecar_client import SidecarClient


class DecisionEngineClient:
    """Unified client for evaluating payment gateway routing and risk decisions.

    Examples:
        In-Process Usage:
        >>> client = DecisionEngineClient(mode="in_process")
        >>> result = client.decide({"amount": 100.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411"})
        >>> print(result.decision, result.confidence, result.latency_ms)

        Sidecar Usage:
        >>> client = DecisionEngineClient(mode="sidecar", sidecar_url="http://localhost:8080")
        >>> result = client.decide({"amount": 100.0, "currency": "EUR", "card_brand": "mastercard", "bin_country": "DE", "mcc": "5411"})
    """

    def __init__(
        self,
        mode: str = "in_process",
        sidecar_url: str = "http://localhost:8080",
        timeout_ms: float = 20.0,
        fallback_decision: str = "stripe_us",
        spec_path: Optional[Union[str, Path]] = None,
        engine_instance: Optional[Any] = None,
        raise_on_error: bool = False,
        sdm_evaluator: Optional[SpecialistSDMEvaluator] = None,
        sdm_path: Optional[Union[str, Path]] = None,
        specialist_type: str = "routing",
        sealed_package_path: Optional[Union[str, Path]] = None,
        license_key: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        """Initialize the client.

        Args:
            mode: "in_process", "sidecar", "sdm", "sealed", or "auto".
            sidecar_url: URL for the sidecar container if mode="sidecar".
            timeout_ms: Max latency timeout in milliseconds (default 20.0ms).
            fallback_decision: Gateway choice to fallback to if errors occur.
            spec_path: Path to declarative decision spec YAML if mode="in_process".
            engine_instance: Optional pre-compiled DecisionEngine instance.
            raise_on_error: Whether to raise exceptions or return fallback decisions.
            sdm_evaluator: Optional pre-configured SpecialistSDMEvaluator.
            sdm_path: Optional path to ONNX or PyTorch Specialist SDM.
            specialist_type: Specialist domain ("routing", "fraud", "sca", "retry").
            sealed_package_path: Path to .sdmpkg encrypted sealed model package.
            license_key: Secret license key for sealed package unlocking.
            client_id: Optional client identifier to verify entitlement.
        """
        self.mode = mode.lower()
        self.sidecar_url = sidecar_url
        self.timeout_ms = timeout_ms
        self.fallback_decision = fallback_decision
        self.raise_on_error = raise_on_error
        self.specialist_type = specialist_type
        self.sealed_package_path = sealed_package_path
        self.license_key = license_key
        self.client_id = client_id

        # In-process engine
        self._in_process: Optional[InProcessEngine] = None
        if self.mode in ("in_process", "auto"):
            self._in_process = InProcessEngine(
                spec_path=spec_path, engine_instance=engine_instance
            )

        # Specialist SDM evaluator
        self._sdm: Optional[SpecialistSDMEvaluator] = None
        if self.mode in ("sdm", "auto") or sdm_evaluator is not None or sdm_path is not None:
            self._sdm = sdm_evaluator or SpecialistSDMEvaluator(
                model_or_path=sdm_path,
                spec_path=spec_path,
                specialist_type=specialist_type,
            )

        # Sealed package client
        self._sealed: Optional[SealedPackageClient] = None
        if (self.mode == "sealed" or sealed_package_path is not None) and license_key:
            assert sealed_package_path is not None
            self._sealed = SealedPackageClient(
                package_path=sealed_package_path,
                license_key=license_key,
                client_id=client_id,
            )

        # Sidecar client
        self._sidecar: Optional[SidecarClient] = None
        if self.mode in ("sidecar", "auto"):
            self._sidecar = SidecarClient(
                base_url=self.sidecar_url,
                timeout_ms=self.timeout_ms,
                fallback_decision=self.fallback_decision,
                raise_on_error=self.raise_on_error,
            )

    def decide(self, context: Union[PaymentContext, Dict[str, Any]]) -> DecisionResult:
        """Synchronously evaluate a payment transaction.

        Args:
            context: Payment transaction parameters as PaymentContext or dictionary.

        Returns:
            Calibrated DecisionResult object.
        """
        if self.mode == "in_process":
            assert self._in_process is not None
            return self._in_process.evaluate(context)

        elif self.mode == "sdm":
            return self.decide_sdm(context)

        elif self.mode == "sealed":
            if self._sealed is None:
                raise DecisionSDKError("Client mode is 'sealed' but sealed package was not initialized with license_key")
            return self._sealed.decide(context)

        elif self.mode == "sidecar":
            assert self._sidecar is not None
            return self._sidecar.decide(context)

        elif self.mode == "auto":
            # Auto mode: try SDM first if available, then in-process, then sidecar
            if self._sdm is not None:
                try:
                    return self._sdm.evaluate(context)
                except Exception:
                    pass
            if self._in_process is not None:
                try:
                    return self._in_process.evaluate(context)
                except Exception:
                    pass
            if self._sidecar is not None:
                return self._sidecar.decide(context)

        raise DecisionSDKError(f"Unsupported evaluation mode: {self.mode}")

    def decide_sdm(self, context: Union[PaymentContext, Dict[str, Any]]) -> DecisionResult:
        """Evaluate a payment transaction via the Specialist Small Decision Model (SDM)."""
        if self._sdm is None:
            self._sdm = SpecialistSDMEvaluator(specialist_type=self.specialist_type)
        return self._sdm.evaluate(context)

    async def decide_sdm_async(self, context: Union[PaymentContext, Dict[str, Any]]) -> DecisionResult:
        """Asynchronously evaluate a payment transaction via the Specialist SDM."""
        return self.decide_sdm(context)

    async def decide_async(
        self, context: Union[PaymentContext, Dict[str, Any]]
    ) -> DecisionResult:
        """Asynchronously evaluate a payment transaction."""
        if self.mode == "in_process":
            assert self._in_process is not None
            return self._in_process.evaluate(context)

        elif self.mode == "sdm":
            return await self.decide_sdm_async(context)

        elif self.mode == "sealed":
            if self._sealed is None:
                raise DecisionSDKError("Client mode is 'sealed' but sealed package was not initialized with license_key")
            return await self._sealed.decide_async(context)

        elif self.mode == "sidecar":
            assert self._sidecar is not None
            return await self._sidecar.decide_async(context)

        elif self.mode == "auto":
            if self._sdm is not None:
                try:
                    return self._sdm.evaluate(context)
                except Exception:
                    pass
            if self._in_process is not None:
                try:
                    return self._in_process.evaluate(context)
                except Exception:
                    pass
            if self._sidecar is not None:
                return await self._sidecar.decide_async(context)

        raise DecisionSDKError(f"Unsupported evaluation mode: {self.mode}")

    def decide_batch(
        self, contexts: List[Union[PaymentContext, Dict[str, Any]]]
    ) -> List[DecisionResult]:
        """Synchronously evaluate a batch of payment transactions."""
        if self.mode == "in_process":
            assert self._in_process is not None
            return [self._in_process.evaluate(ctx) for ctx in contexts]

        elif self.mode == "sidecar":
            assert self._sidecar is not None
            return self._sidecar.decide_batch(contexts)

        elif self.mode == "auto":
            if self._in_process is not None:
                try:
                    return [self._in_process.evaluate(ctx) for ctx in contexts]
                except Exception:
                    pass
            if self._sidecar is not None:
                return self._sidecar.decide_batch(contexts)

        return [self.decide(ctx) for ctx in contexts]

    async def decide_batch_async(
        self, contexts: List[Union[PaymentContext, Dict[str, Any]]]
    ) -> List[DecisionResult]:
        """Asynchronously evaluate a batch of payment transactions."""
        if self.mode == "in_process":
            assert self._in_process is not None
            return [self._in_process.evaluate(ctx) for ctx in contexts]

        elif self.mode == "sidecar":
            assert self._sidecar is not None
            return await self._sidecar.decide_batch_async(contexts)

        elif self.mode == "auto":
            if self._in_process is not None:
                try:
                    return [self._in_process.evaluate(ctx) for ctx in contexts]
                except Exception:
                    pass
            if self._sidecar is not None:
                return await self._sidecar.decide_batch_async(contexts)

        tasks = [self.decide_async(ctx) for ctx in contexts]
        return await asyncio.gather(*tasks)

    def health_check(self) -> Dict[str, Any]:
        """Check system health for the configured evaluation mode."""
        status = {
            "mode": self.mode,
            "status": "healthy",
            "in_process_ready": self._in_process is not None,
            "sidecar_ready": False,
        }
        if self._sidecar is not None:
            status["sidecar_ready"] = self._sidecar.health()

        if self.mode == "sidecar" and not status["sidecar_ready"]:
            status["status"] = "degraded"

        return status

    def close(self) -> None:
        """Close client connections."""
        if self._sidecar is not None:
            self._sidecar.close()

    async def close_async(self) -> None:
        """Close asynchronous client connections."""
        if self._sidecar is not None:
            await self._sidecar.close_async()

    def __enter__(self) -> DecisionEngineClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> DecisionEngineClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close_async()
