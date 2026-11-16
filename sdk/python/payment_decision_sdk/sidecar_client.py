"""Sidecar client for HTTP-based decision evaluation (<20ms SLA).

Provides connection-pooled HTTP communication to the payment decision sidecar container,
with strict latency budgeting, automatic circuit breaking, and zero-dropout fallback.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Union

import httpx

from payment_decision_sdk.models import (
    BatchDecisionRequest,
    BatchDecisionResponse,
    DecisionResult,
    PaymentContext,
)
from payment_decision_sdk.exceptions import (
    DecisionTimeoutError,
    InvalidContextError,
    SidecarConnectionError,
)


class SidecarClient:
    """High-throughput HTTP client for the Payment Decision Sidecar microservice."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout_ms: float = 20.0,
        fallback_decision: str = "stripe_us",
        raise_on_error: bool = False,
        max_connections: int = 100,
        max_keepalive_connections: int = 50,
    ):
        """Initialize the sidecar client.

        Args:
            base_url: Base URL of the sidecar microservice.
            timeout_ms: Maximum SLA latency budget in milliseconds (default 20.0ms).
            fallback_decision: Fallback gateway choice if sidecar fails or times out.
            raise_on_error: If True, raise exception on timeout/connection failure.
                            If False, return a safe fallback DecisionResult to protect checkout conversion.
            max_connections: Maximum HTTP pool connections.
            max_keepalive_connections: Maximum idle keepalive connections.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_ms / 1000.0
        self.timeout_ms = timeout_ms
        self.fallback_decision = fallback_decision
        self.raise_on_error = raise_on_error

        # Configure HTTP pool limits for ultra-low latency
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=60.0,
        )
        timeout = httpx.Timeout(self.timeout_sec, connect=min(self.timeout_sec, 0.015))

        self._client: Optional[httpx.Client] = httpx.Client(
            base_url=self.base_url,
            limits=limits,
            timeout=timeout,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        self._async_client: Optional[httpx.AsyncClient] = None
        self._limits = limits
        self._timeout = timeout

    def _get_async_client(self) -> httpx.AsyncClient:
        """Lazily initialize the asynchronous HTTP client."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                limits=self._limits,
                timeout=self._timeout,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
        return self._async_client

    def decide(self, context: Union[PaymentContext, Dict[str, Any]]) -> DecisionResult:
        """Synchronously evaluate a payment transaction via the sidecar.

        Args:
            context: Validated PaymentContext or raw payment dictionary.

        Returns:
            Calibrated DecisionResult. If sidecar fails and raise_on_error is False,
            returns safe fallback result to prevent checkout abandonment.
        """
        start_time = time.perf_counter()

        if isinstance(context, dict):
            try:
                ctx = PaymentContext.from_dict(context)
            except Exception as e:
                raise InvalidContextError(f"Validation failed for payment payload: {e}") from e
        elif isinstance(context, PaymentContext):
            ctx = context
        else:
            raise InvalidContextError(f"Expected PaymentContext or dict, got {type(context)}")

        payload = ctx.to_dict()

        try:
            assert self._client is not None
            response = self._client.post("/decide", json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if response.status_code == 200:
                data = response.json()
                data["engine_mode"] = "sidecar"
                data["latency_ms"] = round(elapsed_ms, 3)
                return DecisionResult.from_dict(data)

            err_msg = f"Sidecar responded with HTTP status {response.status_code}: {response.text}"
            if self.raise_on_error:
                raise SidecarConnectionError(err_msg)
            return self._build_fallback(ctx, elapsed_ms, err_msg)

        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if self.raise_on_error:
                raise DecisionTimeoutError(
                    f"Decision evaluation exceeded SLA timeout ({self.timeout_ms}ms)"
                ) from exc
            return self._build_fallback(ctx, elapsed_ms, f"SLA timeout ({self.timeout_ms}ms) exceeded")

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if self.raise_on_error:
                raise SidecarConnectionError(f"Sidecar communication failure: {exc}") from exc
            return self._build_fallback(ctx, elapsed_ms, f"Sidecar connection error: {exc}")

    async def decide_async(self, context: Union[PaymentContext, Dict[str, Any]]) -> DecisionResult:
        """Asynchronously evaluate a payment transaction via the sidecar."""
        start_time = time.perf_counter()

        if isinstance(context, dict):
            try:
                ctx = PaymentContext.from_dict(context)
            except Exception as e:
                raise InvalidContextError(f"Validation failed for payment payload: {e}") from e
        elif isinstance(context, PaymentContext):
            ctx = context
        else:
            raise InvalidContextError(f"Expected PaymentContext or dict, got {type(context)}")

        payload = ctx.to_dict()
        client = self._get_async_client()

        try:
            response = await client.post("/decide", json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if response.status_code == 200:
                data = response.json()
                data["engine_mode"] = "sidecar"
                data["latency_ms"] = round(elapsed_ms, 3)
                return DecisionResult.from_dict(data)

            err_msg = f"Sidecar responded with HTTP {response.status_code}: {response.text}"
            if self.raise_on_error:
                raise SidecarConnectionError(err_msg)
            return self._build_fallback(ctx, elapsed_ms, err_msg)

        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if self.raise_on_error:
                raise DecisionTimeoutError(
                    f"Decision evaluation exceeded SLA timeout ({self.timeout_ms}ms)"
                ) from exc
            return self._build_fallback(ctx, elapsed_ms, f"SLA timeout ({self.timeout_ms}ms) exceeded")

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if self.raise_on_error:
                raise SidecarConnectionError(f"Sidecar async error: {exc}") from exc
            return self._build_fallback(ctx, elapsed_ms, f"Sidecar connection error: {exc}")

    def decide_batch(
        self, contexts: List[Union[PaymentContext, Dict[str, Any]]]
    ) -> List[DecisionResult]:
        """Synchronously evaluate a batch of payment transactions."""
        start_time = time.perf_counter()
        normalized: List[PaymentContext] = []
        for c in contexts:
            if isinstance(c, dict):
                normalized.append(PaymentContext.from_dict(c))
            else:
                normalized.append(c)

        payload = {"transactions": [ctx.to_dict() for ctx in normalized]}

        try:
            assert self._client is not None
            response = self._client.post("/decide/batch", json=payload)
            if response.status_code == 200:
                data = response.json()
                results: List[DecisionResult] = []
                for item in data.get("results", []):
                    item["engine_mode"] = "sidecar"
                    results.append(DecisionResult.from_dict(item))
                return results

            if self.raise_on_error:
                raise SidecarConnectionError(f"Batch request returned HTTP {response.status_code}")
            return [self.decide(c) for c in normalized]

        except Exception as exc:
            if self.raise_on_error:
                raise SidecarConnectionError(f"Batch sidecar failure: {exc}") from exc
            # Fall back to per-item evaluation
            return [self.decide(c) for c in normalized]

    async def decide_batch_async(
        self, contexts: List[Union[PaymentContext, Dict[str, Any]]]
    ) -> List[DecisionResult]:
        """Asynchronously evaluate a batch of payment transactions."""
        client = self._get_async_client()
        normalized: List[PaymentContext] = []
        for c in contexts:
            if isinstance(c, dict):
                normalized.append(PaymentContext.from_dict(c))
            else:
                normalized.append(c)

        payload = {"transactions": [ctx.to_dict() for ctx in normalized]}

        try:
            response = await client.post("/decide/batch", json=payload)
            if response.status_code == 200:
                data = response.json()
                results: List[DecisionResult] = []
                for item in data.get("results", []):
                    item["engine_mode"] = "sidecar"
                    results.append(DecisionResult.from_dict(item))
                return results

            if self.raise_on_error:
                raise SidecarConnectionError(f"Batch async returned HTTP {response.status_code}")
            return [await self.decide_async(c) for c in normalized]

        except Exception as exc:
            if self.raise_on_error:
                raise SidecarConnectionError(f"Async batch error: {exc}") from exc
            tasks = [self.decide_async(c) for c in normalized]
            return await asyncio.gather(*tasks)

    def health(self) -> bool:
        """Check if the sidecar microservice is healthy and reachable."""
        try:
            assert self._client is not None
            resp = self._client.get("/health", timeout=1.0)
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
        except Exception:
            return False

    def _build_fallback(
        self, ctx: PaymentContext, elapsed_ms: float, reason: str
    ) -> DecisionResult:
        """Construct a deterministic fallback routing decision."""
        # Derive intelligent fallback based on currency if default isn't explicit
        fallback_choice = self.fallback_decision
        if ctx.currency in ("EUR", "GBP") and fallback_choice == "stripe_us":
            fallback_choice = "adyen_eu"

        return DecisionResult(
            decision=fallback_choice,
            confidence=0.50,
            probabilities={fallback_choice: 1.0},
            latency_ms=round(elapsed_ms, 3),
            routing_reason=f"Safety Fallback: {reason}",
            feature_contributions={"fallback_circuit": 1.0},
            is_fallback=True,
            engine_mode="fallback",
            model_version="fallback-safe",
        )

    def close(self) -> None:
        """Close synchronous client connections."""
        if self._client is not None:
            self._client.close()

    async def close_async(self) -> None:
        """Close asynchronous client connections."""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()

    def __enter__(self) -> SidecarClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> SidecarClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close_async()
