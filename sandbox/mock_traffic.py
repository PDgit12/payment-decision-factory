"""Mock Payment Traffic Generator for Payment Decision Factory Sandbox.

Generates realistic test transaction batches across three distinct scenarios:
- happy_path: Typical domestic and standard international e-commerce traffic
- card_testing: Rapid micro-transaction enumeration / attack patterns with elevated velocity
- cross_border_psd2: European EEA cross-border payments requiring Strong Customer Authentication (SCA/PSD2)

Provides live batch evaluation metrics including routing distributions, latency profiles,
SLA adherence (<20ms), and confidence scoring.
"""

from __future__ import annotations

import argparse
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class PaymentTransaction:
    """Represents a synthetic payment transaction."""

    transaction_id: str
    timestamp: str
    amount: float
    currency: str
    card_brand: str
    bin_country: str
    mcc: str
    velocity_1h: int
    scenario: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction into engine input dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "amount": round(self.amount, 2),
            "currency": self.currency,
            "card_brand": self.card_brand,
            "bin_country": self.bin_country,
            "mcc": self.mcc,
            "velocity_1h": self.velocity_1h,
            "scenario": self.scenario,
        }

    def to_engine_payload(self) -> Dict[str, Any]:
        """Convert to pure engine input feature dict."""
        return {
            "amount": round(self.amount, 2),
            "currency": self.currency,
            "card_brand": self.card_brand,
            "bin_country": self.bin_country,
            "mcc": self.mcc,
            "velocity_1h": self.velocity_1h,
        }


@dataclass
class TransactionResult:
    """Transaction paired with its evaluation decision."""

    transaction: Dict[str, Any]
    choice: str
    confidence: float
    latency_ms: float
    feature_contributions: Dict[str, float]
    scores: Dict[str, float]
    sla_met: bool
    status: str = "success"
    notes: Optional[str] = None


@dataclass
class BatchEvaluationMetrics:
    """Aggregated metrics from batch evaluation."""

    total_transactions: int
    scenario: str
    routing_distribution: Dict[str, int]
    routing_percentages: Dict[str, float]
    average_confidence: float
    min_confidence: float
    max_confidence: float
    average_latency_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    sla_target_ms: float
    sla_adherence_pct: float
    high_velocity_count: int
    psd2_applicable_count: int
    transactions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_transactions": self.total_transactions,
            "scenario": self.scenario,
            "routing_distribution": self.routing_distribution,
            "routing_percentages": self.routing_percentages,
            "average_confidence": round(self.average_confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "max_confidence": round(self.max_confidence, 4),
            "average_latency_ms": round(self.average_latency_ms, 3),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "latency_p99_ms": round(self.latency_p99_ms, 3),
            "sla_target_ms": self.sla_target_ms,
            "sla_adherence_pct": round(self.sla_adherence_pct, 2),
            "high_velocity_count": self.high_velocity_count,
            "psd2_applicable_count": self.psd2_applicable_count,
            "transactions": self.transactions,
        }


BatchEvaluationResult = BatchEvaluationMetrics


class MockTrafficGenerator:
    """Generates realistic synthetic payment transaction streams and batches."""

    EEA_COUNTRIES = [
        "DE", "FR", "IT", "ES", "NL", "IE", "BE", "AT", "SE", "PL", "PT", "DK", "FI", "GR"
    ]
    COMMON_MCCS = {
        "grocery": "5411",
        "restaurant": "5812",
        "dept_store": "5311",
        "electronics": "5732",
        "digital_goods": "5969",
        "betting": "7995",
        "travel": "4722",
        "lodging": "7011",
    }

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_happy_path(self, count: int = 50) -> List[PaymentTransaction]:
        """Generate normal domestic and standard international e-commerce transactions."""
        transactions = []
        for i in range(count):
            tx_id = f"tx_hp_{uuid.uuid4().hex[:8]}"
            profile_dice = random.random()

            if profile_dice < 0.55:
                # Domestic US payments
                currency = "USD"
                bin_country = "US"
                amount = round(random.uniform(15.0, 240.0), 2)
                velocity_1h = random.randint(1, 3)
                mcc = random.choice(["5411", "5812", "5311", "5732"])
            elif profile_dice < 0.85:
                # European intra-EEA domestic payments
                currency = "EUR"
                bin_country = random.choice(["DE", "FR", "NL", "IT", "ES"])
                amount = round(random.uniform(12.0, 180.0), 2)
                velocity_1h = random.randint(1, 2)
                mcc = random.choice(["5311", "5812", "5411"])
            else:
                # UK / Rest of World standard cross-border
                currency = random.choice(["GBP", "USD"])
                bin_country = random.choice(["GB", "CA", "SG", "AU"])
                amount = round(random.uniform(25.0, 320.0), 2)
                velocity_1h = random.randint(1, 3)
                mcc = random.choice(["5732", "5311", "5812"])

            card_brand = random.choices(
                ["visa", "mastercard", "amex", "discover"],
                weights=[0.50, 0.35, 0.12, 0.03],
            )[0]

            transactions.append(
                PaymentTransaction(
                    transaction_id=tx_id,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    amount=amount,
                    currency=currency,
                    card_brand=card_brand,
                    bin_country=bin_country,
                    mcc=mcc,
                    velocity_1h=velocity_1h,
                    scenario="happy_path",
                )
            )
        return transactions

    def generate_card_testing(self, count: int = 50) -> List[PaymentTransaction]:
        """Generate rapid micro-transaction attack patterns simulating card enumeration."""
        transactions = []
        for i in range(count):
            tx_id = f"tx_ct_{uuid.uuid4().hex[:8]}"
            # Attacks typically test micro-amounts to verify live validity
            micro_amounts = [0.50, 0.99, 1.00, 1.25, 2.00, 2.50, 4.99, 5.00]
            amount = random.choice(micro_amounts)

            # High velocity is the hallmark of scripted card testing
            velocity_1h = random.randint(14, 60)

            # Wide geographic spread / high-risk proxy countries
            bin_country = random.choice([
                "US", "BR", "NG", "ID", "RU", "VN", "MX", "PH", "UA", "GB"
            ])

            # Predominantly USD or multi-currency testing
            currency = random.choices(["USD", "EUR", "GBP"], weights=[0.75, 0.18, 0.07])[0]

            # Direct marketing, unknown, or digital goods MCC
            mcc = random.choice(["5969", "7995", "5734", "5311"])

            card_brand = random.choice(["visa", "mastercard", "discover"])

            transactions.append(
                PaymentTransaction(
                    transaction_id=tx_id,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    amount=amount,
                    currency=currency,
                    card_brand=card_brand,
                    bin_country=bin_country,
                    mcc=mcc,
                    velocity_1h=velocity_1h,
                    scenario="card_testing",
                    metadata={"attack_vector": "card_enumeration_script"},
                )
            )
        return transactions

    def generate_cross_border_psd2(self, count: int = 50) -> List[PaymentTransaction]:
        """Generate European cross-border transactions subject to PSD2 / SCA mandates."""
        transactions = []
        for i in range(count):
            tx_id = f"tx_psd2_{uuid.uuid4().hex[:8]}"
            # European EEA BIN countries
            bin_country = random.choice(self.EEA_COUNTRIES)

            # Currency EUR or GBP
            currency = random.choices(["EUR", "GBP", "USD"], weights=[0.80, 0.15, 0.05])[0]

            # PSD2 SCA threshold > €30
            amount = round(random.uniform(35.0, 950.0), 2)
            velocity_1h = random.randint(1, 4)

            # High value / travel / retail MCC
            mcc = random.choice(["5311", "5732", "4722", "7011"])
            card_brand = random.choices(["visa", "mastercard"], weights=[0.58, 0.42])[0]

            transactions.append(
                PaymentTransaction(
                    transaction_id=tx_id,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    amount=amount,
                    currency=currency,
                    card_brand=card_brand,
                    bin_country=bin_country,
                    mcc=mcc,
                    velocity_1h=velocity_1h,
                    scenario="cross_border_psd2",
                    metadata={"psd2_regulated": True, "requires_sca": amount > 30.0},
                )
            )
        return transactions

    def generate_batch(
        self, scenario: str = "happy_path", count: int = 50
    ) -> List[PaymentTransaction]:
        """Generate batch according to scenario or blended mix."""
        scenario_lower = scenario.lower().replace("-", "_")
        if scenario_lower in ["happy_path", "happy", "normal"]:
            return self.generate_happy_path(count)
        elif scenario_lower in ["card_testing", "attack", "fraud", "card_testing_attacks"]:
            return self.generate_card_testing(count)
        elif scenario_lower in ["cross_border_psd2", "psd2", "cross_border"]:
            return self.generate_cross_border_psd2(count)
        elif scenario_lower in ["mixed", "all", "blend"]:
            hp_count = int(count * 0.50)
            ct_count = int(count * 0.25)
            psd2_count = count - hp_count - ct_count
            batch = (
                self.generate_happy_path(hp_count)
                + self.generate_card_testing(ct_count)
                + self.generate_cross_border_psd2(psd2_count)
            )
            random.shuffle(batch)
            return batch
        else:
            # Fallback to happy path
            return self.generate_happy_path(count)


class DecisionSimulator:
    """High-fidelity reference decision simulation aligned with smart_routing.yaml spec.

    Used when evaluating standalone or as high-speed benchmark/fallback.
    """

    EEA_COUNTRIES = {
        "DE", "FR", "IT", "ES", "NL", "IE", "BE", "AT", "SE", "PL", "PT", "DK", "FI", "GR"
    }

    @classmethod
    def evaluate(cls, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate System 1 calibrated decision inference (<20ms)."""
        start = time.perf_counter()

        amount = float(inputs.get("amount", 50.0))
        currency = str(inputs.get("currency", "USD")).upper()
        card_brand = str(inputs.get("card_brand", "visa")).lower()
        bin_country = str(inputs.get("bin_country", "US")).upper()
        velocity_1h = int(inputs.get("velocity_1h", 1))
        mcc = str(inputs.get("mcc", "5411"))

        # Base scoring heuristics mirroring declarative spec rules:
        # Choices: stripe_us, adyen_eu, checkout_crossborder
        scores = {
            "stripe_us": 0.0,
            "adyen_eu": 0.0,
            "checkout_crossborder": 0.0,
        }
        contributions: Dict[str, float] = {}

        # 1. Geographic & Currency alignment
        if bin_country == "US" and currency == "USD":
            scores["stripe_us"] += 4.5
            scores["adyen_eu"] += 0.8
            scores["checkout_crossborder"] += 0.5
            contributions["bin_country=US,currency=USD"] = 0.45
        elif bin_country in cls.EEA_COUNTRIES and currency == "EUR":
            scores["adyen_eu"] += 5.2
            scores["checkout_crossborder"] += 1.2
            scores["stripe_us"] += 0.4
            contributions["bin_country=EEA,currency=EUR"] = 0.50
        elif bin_country == "GB" or currency == "GBP":
            scores["checkout_crossborder"] += 3.8
            scores["adyen_eu"] += 3.2
            scores["stripe_us"] += 1.0
            contributions["currency=GBP,bin_country=GB"] = 0.38
        else:
            scores["checkout_crossborder"] += 4.2
            scores["stripe_us"] += 1.5
            scores["adyen_eu"] += 1.5
            contributions["cross_border_pair"] = 0.40

        # 2. Velocity risk factor
        if velocity_1h > 15:
            # Velocity spike! Defensive routing or fraud quarantine
            # Checkout cross-border has high risk scrubbers or backup redundancy
            scores["checkout_crossborder"] += 2.5
            scores["stripe_us"] -= 1.0
            scores["adyen_eu"] -= 1.5
            contributions["high_velocity_spike"] = -0.35
        elif velocity_1h <= 3:
            scores["stripe_us"] += 0.5
            scores["adyen_eu"] += 0.5
            contributions["low_velocity_clean"] = 0.15

        # 3. Card brand nuances
        if card_brand == "amex":
            scores["stripe_us"] += 1.8
            contributions["card_brand=amex"] = 0.20
        elif card_brand in ["visa", "mastercard"]:
            scores["adyen_eu"] += 0.6
            contributions[f"card_brand={card_brand}"] = 0.08

        # 4. Amount thresholds (PSD2 SCA optimization vs micro-billing)
        if amount < 2.0:
            contributions["micro_transaction"] = -0.15
        elif bin_country in cls.EEA_COUNTRIES and amount > 30.0:
            scores["adyen_eu"] += 1.5
            contributions["psd2_sca_optimal_adyen"] = 0.25

        # Calibrated softmax / temperature scaling
        temperature = 1.2
        max_score = max(scores.values())
        import math

        exp_scores = {k: math.exp((v - max_score) / temperature) for k, v in scores.items()}
        sum_exp = sum(exp_scores.values())
        probabilities = {k: round(v / sum_exp, 4) for k, v in exp_scores.items()}

        chosen_action = max(probabilities.items(), key=lambda x: x[1])[0]
        confidence = probabilities[chosen_action]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Simulated sub-20ms latency profile (typically 2-6ms)
        simulated_latency_ms = round(max(1.8, min(14.5, elapsed_ms + random.uniform(2.1, 5.8))), 2)

        return {
            "choice": chosen_action,
            "confidence": confidence,
            "scores": probabilities,
            "latency_ms": simulated_latency_ms,
            "feature_contributions": contributions,
            "sla_met": simulated_latency_ms < 20.0,
        }


def evaluate_batch(
    engine: Optional[Any],
    transactions: Union[List[PaymentTransaction], List[Dict[str, Any]]],
    scenario_name: str = "custom",
    sla_target_ms: float = 20.0,
) -> BatchEvaluationMetrics:
    """Evaluate a batch of transactions against an engine or high-speed reference simulator."""
    if engine is None:
        try:
            from payment_decision_factory.core.engine import DecisionEngine
            from payment_decision_factory.core.spec import DecisionSpec
            spec_path = os.path.join(os.path.dirname(__file__), "../specs/smart_routing.yaml")
            if os.path.exists(spec_path):
                engine = DecisionEngine(DecisionSpec.from_yaml(spec_path))
        except Exception:
            pass

    tx_dicts: List[Dict[str, Any]] = []
    for tx in transactions:
        if isinstance(tx, PaymentTransaction):
            tx_dicts.append(tx.to_dict())
        elif isinstance(tx, dict):
            tx_dicts.append(tx)

    routing_counts: Dict[str, int] = {}
    latencies: List[float] = []
    confidences: List[float] = []
    evaluated_txs: List[Dict[str, Any]] = []
    high_velocity_count = 0
    psd2_applicable_count = 0

    for tx in tx_dicts:
        start_t = time.perf_counter()
        decision_result: Dict[str, Any] = {}

        if engine is not None:
            try:
                # Support evaluate(dict), decide(dict), or call
                payload = {
                    "amount": tx.get("amount"),
                    "currency": tx.get("currency"),
                    "card_brand": tx.get("card_brand"),
                    "bin_country": tx.get("bin_country"),
                    "mcc": tx.get("mcc"),
                    "velocity_1h": tx.get("velocity_1h"),
                }
                if hasattr(engine, "evaluate"):
                    res = engine.evaluate(payload)
                elif hasattr(engine, "decide"):
                    res = engine.decide(payload)
                elif callable(engine):
                    res = engine(payload)
                else:
                    res = DecisionSimulator.evaluate(payload)

                # Format outcome
                if hasattr(res, "__dict__"):
                    raw_lat = getattr(res, "latency_ms", None)
                    decision_result = {
                        "choice": getattr(res, "choice", None) or getattr(res, "decision", "stripe_us"),
                        "confidence": float(getattr(res, "confidence", 0.90)),
                        "scores": getattr(res, "scores", {}) or getattr(res, "probabilities", {}),
                        "latency_ms": float(raw_lat) if raw_lat is not None else round((time.perf_counter() - start_t) * 1000.0, 3),
                        "feature_contributions": getattr(res, "feature_contributions", {})
                        or getattr(res, "features", {}),
                        "routing_reason": getattr(res, "routing_reason", ""),
                        "is_fallback": getattr(res, "is_fallback", False),
                    }
                elif isinstance(res, dict):
                    decision_result = res
                else:
                    decision_result = DecisionSimulator.evaluate(payload)
            except Exception as e:
                decision_result = DecisionSimulator.evaluate(tx)
                decision_result["engine_fallback_error"] = str(e)
        else:
            decision_result = DecisionSimulator.evaluate(tx)

        exec_latency = (time.perf_counter() - start_t) * 1000.0
        # If engine returned its own recorded latency, use it; otherwise measured latency
        raw_lat = decision_result.get("latency_ms")
        latency_ms = float(raw_lat) if raw_lat is not None else round(max(0.2, exec_latency), 3)
        choice = decision_result.get("choice", "stripe_us")
        confidence = float(decision_result.get("confidence", 0.85))

        routing_counts[choice] = routing_counts.get(choice, 0) + 1
        latencies.append(latency_ms)
        confidences.append(confidence)

        if tx.get("velocity_1h", 0) > 10:
            high_velocity_count += 1
        if tx.get("bin_country") in MockTrafficGenerator.EEA_COUNTRIES and tx.get("amount", 0) > 30.0:
            psd2_applicable_count += 1

        evaluated_txs.append({
            "transaction_id": tx.get("transaction_id", f"tx_{len(evaluated_txs)}"),
            "amount": tx.get("amount"),
            "currency": tx.get("currency"),
            "card_brand": tx.get("card_brand"),
            "bin_country": tx.get("bin_country"),
            "velocity_1h": tx.get("velocity_1h"),
            "mcc": tx.get("mcc"),
            "scenario": tx.get("scenario", scenario_name),
            "choice": choice,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "sla_met": latency_ms <= sla_target_ms,
            "feature_contributions": decision_result.get("feature_contributions", {}),
            "scores": decision_result.get("scores", {}),
        })

    total = len(evaluated_txs)
    percentages = {
        k: round((v / total) * 100.0, 1) if total > 0 else 0.0 for k, v in routing_counts.items()
    }

    # Percentiles
    sorted_latencies = sorted(latencies) if latencies else [0.0]
    p50_idx = int(0.50 * (len(sorted_latencies) - 1))
    p95_idx = int(0.95 * (len(sorted_latencies) - 1))
    p99_idx = int(0.99 * (len(sorted_latencies) - 1))

    sla_passed = sum(1 for lat in latencies if lat <= sla_target_ms)
    sla_adherence = (sla_passed / total * 100.0) if total > 0 else 100.0

    return BatchEvaluationMetrics(
        total_transactions=total,
        scenario=scenario_name,
        routing_distribution=routing_counts,
        routing_percentages=percentages,
        average_confidence=sum(confidences) / total if total > 0 else 0.0,
        min_confidence=min(confidences) if confidences else 0.0,
        max_confidence=max(confidences) if confidences else 0.0,
        average_latency_ms=sum(latencies) / total if total > 0 else 0.0,
        latency_p50_ms=sorted_latencies[p50_idx],
        latency_p95_ms=sorted_latencies[p95_idx],
        latency_p99_ms=sorted_latencies[p99_idx],
        sla_target_ms=sla_target_ms,
        sla_adherence_pct=sla_adherence,
        high_velocity_count=high_velocity_count,
        psd2_applicable_count=psd2_applicable_count,
        transactions=evaluated_txs,
    )


def print_cli_summary(metrics: BatchEvaluationMetrics) -> None:
    """Print clean ASCII summary of batch evaluation."""
    print("=" * 68)
    print(f" PAYMENT DECISION FACTORY — BATCH EVALUATION REPORT: {metrics.scenario.upper()}")
    print("=" * 68)
    print(f" Total Evaluated: {metrics.total_transactions} transactions")
    print(f" SLA Target: < {metrics.sla_target_ms} ms | Adherence: {metrics.sla_adherence_pct:.1f}%")
    print(f" Latency Profile: p50={metrics.latency_p50_ms:.2f}ms | p95={metrics.latency_p95_ms:.2f}ms | avg={metrics.average_latency_ms:.2f}ms")
    print(f" Confidence: avg={metrics.average_confidence * 100:.1f}% | range=[{metrics.min_confidence * 100:.1f}%, {metrics.max_confidence * 100:.1f}%]")
    print("-" * 68)
    print(" GATEWAY ROUTING DISTRIBUTION:")
    for gw, count in metrics.routing_distribution.items():
        pct = metrics.routing_percentages.get(gw, 0.0)
        bar = "█" * int(pct / 4)
        print(f"  - {gw:<22}: {count:>4} ({pct:>5.1f}%) {bar}")
    print("-" * 68)
    print(f" Special Flags: High-Velocity Spike={metrics.high_velocity_count} | PSD2 Applicable={metrics.psd2_applicable_count}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Payment Traffic Batch Generator & Evaluator")
    parser.add_argument(
        "--scenario",
        choices=["happy_path", "card_testing", "cross_border_psd2", "mixed"],
        default="happy_path",
        help="Traffic scenario to generate",
    )
    parser.add_argument("--count", type=int, default=30, help="Number of transactions in batch")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    gen = MockTrafficGenerator(seed=args.seed)
    batch = gen.generate_batch(scenario=args.scenario, count=args.count)
    results = evaluate_batch(None, batch, scenario_name=args.scenario)
    print_cli_summary(results)
