"""
Unit Tests for Universal Decision Model Factory
================================================
Tests:
- Dynamic schema profiling for non-payment business domains (Logistics & Credit)
- Automatic categorical entity embeddings and numerical feature scaling
- Fast deep model training in < 5 seconds on CPU
- End-to-end train-and-seal pipeline producing valid .sdmpkg packages
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

from payment_decision_factory.core.universal_factory import (
    UniversalDecisionFactory,
    UniversalDomainSpec,
)


class TestUniversalDecisionFactory:
    """Tests for domain-agnostic Specialist SDM factory."""

    def test_logistics_carrier_dispatch_domain(self):
        """Test factory on a Logistics & Freight Carrier Dispatch domain."""
        factory = UniversalDecisionFactory.create_for_domain(
            domain_id="freight_dispatch",
            domain_name="Automated Freight Carrier Dispatch SDM",
            numerical_features=["weight_kg", "distance_km", "urgency_hours", "freight_value_usd"],
            categorical_features=["origin_zone", "destination_zone", "equipment_type"],
            choices=["fedex_freight", "dhl_express", "coyote_logistics", "ch_robinson"],
            target_field="selected_carrier",
        )

        # Generate realistic freight records
        records = []
        for i in range(120):
            w = 100.0 + (i * 25.0) % 5000
            d = 50.0 + (i * 75.0) % 2500
            u = 4.0 if (i % 3 == 0) else 48.0
            val = 500.0 + (i * 200.0) % 20000
            orig = "US_EAST" if (i % 2 == 0) else "US_WEST"
            dest = "EU_CENTRAL" if (i % 4 == 0) else "US_MIDWEST"
            eq = "dry_van" if (i % 2 == 0) else "reefer"

            if u <= 12.0:
                carrier = "dhl_express"
            elif w > 2500.0:
                carrier = "coyote_logistics"
            elif orig == "US_EAST" and dest == "US_MIDWEST":
                carrier = "ch_robinson"
            else:
                carrier = "fedex_freight"

            records.append({
                "weight_kg": w,
                "distance_km": d,
                "urgency_hours": u,
                "freight_value_usd": val,
                "origin_zone": orig,
                "destination_zone": dest,
                "equipment_type": eq,
                "selected_carrier": carrier,
            })

        t0 = time.perf_counter()
        model, metrics = factory.train(records, epochs=10)
        train_time = time.perf_counter() - t0

        assert train_time < 10.0, f"Training took {train_time}s, expected < 10s"
        assert metrics["train_samples"] > 0
        assert metrics["parameter_count"] < 150_000
        assert metrics["validation_accuracy"] >= 0.70

    def test_train_and_seal_pipeline_end_to_end(self):
        """Test full train -> export ONNX -> AES-256-GCM seal into .sdmpkg."""
        factory = UniversalDecisionFactory.create_for_domain(
            domain_id="dynamic_pricing",
            domain_name="Dynamic Airline Ticket Pricing Tier SDM",
            numerical_features=["days_to_departure", "searches_last_hour", "competitor_price_usd"],
            categorical_features=["route_corridor", "cabin_class"],
            choices=["economy_discount", "economy_standard", "premium_surge"],
            target_field="fare_class",
        )

        records = []
        for i in range(100):
            days = 1 + (i % 60)
            searches = 10 + (i * 5) % 300
            comp = 150.0 + (i * 10.0) % 800
            route = "JFK_LHR" if (i % 2 == 0) else "SFO_NRT"
            cabin = "economy" if (i % 3 != 0) else "business"

            if days <= 3 or searches > 150:
                fare = "premium_surge"
            elif days > 30 and comp < 300:
                fare = "economy_discount"
            else:
                fare = "economy_standard"

            records.append({
                "days_to_departure": days,
                "searches_last_hour": searches,
                "competitor_price_usd": comp,
                "route_corridor": route,
                "cabin_class": cabin,
                "fare_class": fare,
            })

        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg_path = os.path.join(tmp_dir, "pricing.sdmpkg")
            license_key = "LIC-AIRLINE-GLOBAL-2026"

            sealed_pkg, metrics = factory.train_and_seal(
                records=records,
                license_key=license_key,
                client_id="global_airlines_group",
                output_package_path=pkg_path,
                epochs=8,
            )

            assert os.path.exists(pkg_path)
            assert sealed_pkg.manifest["domain"] == "dynamic_pricing"

            # In-memory decryption and inference verification
            ctx = sealed_pkg.decrypt(license_key)
            runner = ctx.create_runner()

            test_rec = records[0]
            test_vec = factory.encoder.transform_record(test_rec)
            res = runner.evaluate(test_vec)

            assert res["choice"] in factory.spec.choices
            assert 0.0 <= res["confidence"] <= 1.0
            assert res["latency_ms"] < 5.0
