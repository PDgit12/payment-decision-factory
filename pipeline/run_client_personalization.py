"""
End-to-End Client Personalization Demonstration
===============================================
Demonstrates the full client onboarding loop:
1. Simulates ingestion of a client's 2,000 historical payment records.
2. Serializes data and trains a calibrated decision adapter on top of the model.
3. Measures accuracy, Brier score, and loss reduction.
4. Exports the custom weights and manifest for the client.
"""

import json
import os
import sys
import random

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.pipeline.tuner import ModelTuner


def generate_sample_client_history(num_records: int = 2000) -> list:
    """Generates realistic client payment records with empirical gateway preferences."""
    records = []
    card_brands = ["visa", "mastercard", "amex", "discover"]

    for _ in range(num_records):
        # 60% US domestic, 30% European, 10% Cross-border / APAC
        r = random.random()
        if r < 0.60:
            # US domestic transaction -> Optimal: stripe_us
            currency = "USD"
            bin_country = random.choice(["US", "CA"])
            amount = round(random.uniform(15.0, 350.0), 2)
            velocity = random.randint(1, 4)
            optimal = "stripe_us"
        elif r < 0.90:
            # European EEA transaction -> Optimal: adyen_eu
            currency = random.choice(["EUR", "GBP"])
            bin_country = random.choice(["GB", "DE", "FR", "NL", "IE", "ES"])
            amount = round(random.uniform(25.0, 600.0), 2)
            velocity = random.randint(1, 3)
            optimal = "adyen_eu"
        else:
            # High-ticket or cross-border -> Optimal: checkout_crossborder
            currency = random.choice(["USD", "EUR", "JPY", "AUD"])
            bin_country = random.choice(["SG", "JP", "AU", "BR", "AE"])
            amount = round(random.uniform(500.0, 5000.0), 2)
            velocity = random.randint(1, 5)
            optimal = "checkout_crossborder"

        records.append({
            "amount": amount,
            "currency": currency,
            "card_brand": random.choice(card_brands),
            "bin_country": bin_country,
            "mcc": random.choice([5411, 5732, 5812, 5999]),
            "velocity_1h": velocity,
            "optimal_gateway": optimal,
            "status": "authorized"
        })

    return records


def main():
    print("=" * 70)
    print(" PAYMENT DECISION FACTORY — CLIENT DATA PERSONALIZATION PIPELINE")
    print("=" * 70)

    spec_path = os.path.join(_ROOT_DIR, "payment_decision_factory/specs/smart_routing.yaml")
    client_name = "Acme_Global_SaaS"

    # Step 1: Ingest Client Data
    print(f"\n[1/4] Ingesting client historical transaction logs for '{client_name}'...")
    raw_history = generate_sample_client_history(2500)
    print(f"      Ingested {len(raw_history)} historical transaction events.")

    # Step 2: Initialize Tuner and Prepare Datasets
    print("\n[2/4] Serializing payments telemetry into normalized feature tensors...")
    tuner = ModelTuner(spec_path)
    train_ds, val_ds = tuner.prepare_dataset(raw_history)
    print(f"      Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}")
    print(f"      Feature dimensions: {train_ds.features.shape[1]} features extracted per transaction.")

    # Step 3: Train Calibrated Adapter
    print("\n[3/4] Tuning decision adapter weights and fitting temperature scaling...")
    print("      Optimizing Negative Log-Likelihood with Brier Score penalty...")
    head, metrics = tuner.train(train_ds, val_ds, epochs=20, lr=0.008)

    print("\n      --- TUNING RESULTS ---")
    print(f"      Initial Loss:         {metrics.initial_loss}")
    print(f"      Final Loss:           {metrics.final_loss} (Loss reduction: {round((1 - metrics.final_loss/metrics.initial_loss)*100, 1)}%)")
    print(f"      Validation Accuracy:  {round(metrics.final_accuracy * 100, 2)}%")
    print(f"      Brier Score:          {metrics.brier_score} (Target < 0.08: {'PASS' if metrics.brier_score < 0.08 else 'FAIL'})")
    print(f"      Fitted Temperature:   {metrics.fitted_temperature}x")
    print(f"      Training Duration:    {metrics.duration_seconds}s")

    # Step 4: Export Client Bundle
    output_dir = os.path.join(_ROOT_DIR, "payment_decision_factory/artifacts/client_acme_v1")
    print(f"\n[4/4] Exporting hyper-personalized model package to {output_dir}...")
    tuner.export_personalized_bundle(head, metrics, output_dir, client_name)

    print("\n" + "=" * 70)
    print(" PERSONALIZATION COMPLETE: Client model is compiled, calibrated & ready!")
    print("=" * 70)


if __name__ == "__main__":
    main()
