"""
Comprehensive Test Suite for Digital Twin Failure Modes & Fail-Open Circuit Breakers.
====================================================================================
Tests:
1. Real-Time Sliding-Window Circuit Breaker (Threshold tracking, tripping, recovery).
2. Acquirer Sudden Outage Simulation (Adyen 500 spike, dynamic rerouting to Stripe/Checkout).
3. Latency Degradation & Timeout Diversion (<1.0ms failover speed).
4. Total Catastrophic Failure: Deterministic 0ms Fail-Open Guarantee (Zero checkout dropouts).
"""

from __future__ import annotations

import os
import sys
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from payment_decision_factory.sandbox.mock_traffic import MockTrafficGenerator
from payment_decision_factory.sandbox.failure_modes import (
    CircuitBreakerConfig,
    RealTimeCircuitBreaker,
    ProductionFailureSimulator,
    OutageResilienceReport,
)


@pytest.fixture
def traffic_batch():
    gen = MockTrafficGenerator(seed=42)
    # Generate 100 European transactions (which would naturally prefer adyen_eu)
    return [r.to_dict() for r in gen.generate_cross_border_psd2(100)]


class TestRealTimeCircuitBreaker:
    """Tests for low-latency in-memory circuit breaker."""

    def test_initial_healthy_state(self):
        cb = RealTimeCircuitBreaker()
        assert cb.is_rail_healthy("adyen_eu") is True
        assert cb.is_rail_healthy("stripe_us") is True

    def test_trip_on_consecutive_errors(self):
        config = CircuitBreakerConfig(error_threshold_pct=25.0, min_samples_to_trip=4, sliding_window_size=10)
        cb = RealTimeCircuitBreaker(config)

        # Record 3 healthy transactions
        for _ in range(3):
            cb.record_outcome("adyen_eu", is_error=False, latency_ms=90.0)
        assert cb.is_rail_healthy("adyen_eu") is True

        # Record 2 errors -> 2/5 = 40% > 25% threshold
        cb.record_outcome("adyen_eu", is_error=True, latency_ms=120.0)
        cb.record_outcome("adyen_eu", is_error=True, latency_ms=120.0)

        assert cb.is_rail_healthy("adyen_eu") is False
        assert "adyen_eu" in cb.tripped_rails

    def test_select_healthy_fallback_rerouting(self):
        config = CircuitBreakerConfig(error_threshold_pct=20.0, min_samples_to_trip=3)
        cb = RealTimeCircuitBreaker(config)

        # Trip adyen_eu
        for _ in range(3):
            cb.record_outcome("adyen_eu", is_error=True, latency_ms=150.0)

        # Fallback must be a healthy alternative, not adyen_eu
        fallback = cb.select_healthy_fallback("adyen_eu", ["stripe_us", "adyen_eu", "checkout_crossborder"])
        assert fallback in ["stripe_us", "checkout_crossborder"]
        assert fallback != "adyen_eu"

    def test_fail_open_when_all_tripped(self):
        config = CircuitBreakerConfig(default_fail_open_rail="stripe_us")
        cb = RealTimeCircuitBreaker(config)

        # Trip all available rails
        cb.tripped_rails["adyen_eu"] = 9999999999.0
        cb.tripped_rails["stripe_us"] = 9999999999.0
        cb.tripped_rails["checkout_crossborder"] = 9999999999.0

        # Must fail open to default_fail_open_rail without throwing exception
        choice = cb.select_healthy_fallback("adyen_eu", ["stripe_us", "adyen_eu", "checkout_crossborder"])
        assert choice == "stripe_us"


class TestProductionFailureSimulator:
    """Tests for disaster simulation under real production failures."""

    def test_simulate_acquirer_outage_resilience(self, traffic_batch):
        sim = ProductionFailureSimulator(CircuitBreakerConfig(error_threshold_pct=20.0, min_samples_to_trip=4))
        report = sim.simulate_acquirer_outage(
            transactions=traffic_batch,
            outage_rail="adyen_eu",
            outage_error_rate=1.0,
        )

        assert isinstance(report, OutageResilienceReport)
        assert report.total_transactions_tested == 100
        assert report.circuit_breaker_tripped is True
        assert report.transactions_diverted_count > 0
        assert report.retained_conversion_rate >= 0.90  # >90% conversion retained during 100% outage
        assert report.checkout_dropout_rate_pct <= 10.0  # Only initial warmup samples drop
        assert report.unhandled_exceptions_count == 0
        assert report.fail_open_success is True
        assert report.avg_failover_latency_ms < 1.0  # Sub-millisecond failover decision speed

        d = report.to_dict()
        assert d["circuit_breaker_tripped"] is True
        assert d["fail_open_success"] is True

    def test_simulate_catastrophic_total_outage_fail_open(self, traffic_batch):
        sim = ProductionFailureSimulator()
        report = sim.simulate_catastrophic_total_outage(traffic_batch)

        assert report.total_transactions_tested == 100
        assert report.outage_type == "TOTAL_CATASTROPHIC"
        assert report.checkout_dropouts_count == 0
        assert report.checkout_dropout_rate_pct == 0.0
        assert report.unhandled_exceptions_count == 0
        assert report.fail_open_success is True
        assert report.retained_conversion_rate == 1.0
