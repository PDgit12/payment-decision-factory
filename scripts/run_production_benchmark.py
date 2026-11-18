#!/usr/bin/env python3
"""
Production Benchmark Suite for Specialist Decision Models (SDMs)
=================================================================
Performs real, empirical latency profiling and financial accuracy benchmarking
against real credit card transaction telemetry on local host hardware.

Metrics Measured:
- Memory Footprint (RAM / RSS before & after decryption)
- Cold-Start vs Warm Cache Latency
- High-Resolution Latency Distribution: P50, P90, P95, P99, Min, Max, StdDev
- Throughput (Transactions Per Second / TPS)
- Statistical Accuracy, Precision, Recall, F1 Score
- Financial Impact: Net Volume Authorized, Fraud Prevented, VDMP Dispute Rate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Standard library protection against local code.py shadowing
import importlib.util
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code):
    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PARENT_DIR = os.path.abspath(os.path.join(_REPO_ROOT, ".."))
_SDK_DIR = os.path.join(_REPO_ROOT, "sdk", "python")

for p in [_PARENT_DIR, _REPO_ROOT, _SDK_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from payment_decision_sdk.sealed import SealedPackageClient

# Terminal styling
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def get_process_memory_mb() -> float:
    """Read current process resident set size (RSS) in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # macOS gives bytes or KB


def load_benchmark_dataset(csv_path: str, max_records: int = 1000) -> List[Dict[str, Any]]:
    """Load real transactions including fraud and legitimate cases."""
    records = []
    fraud_records = []
    legit_records = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        label_idx = len(header) - 1
        amount_idx = len(header) - 2

        for row in reader:
            try:
                features = [float(row[i]) for i in range(label_idx)]
                label = int(row[label_idx].replace('"', "").strip())
                amount = float(row[amount_idx])
                item = {
                    "features": np.array(features, dtype=np.float32),
                    "label": label,  # 0: legit, 1: fraud
                    "amount": amount,
                }
                if label == 1:
                    fraud_records.append(item)
                else:
                    legit_records.append(item)
            except (ValueError, IndexError):
                continue

    # Ensure all available real fraud cases are included for statistical rigor
    records.extend(fraud_records)
    remaining = max_records - len(records)
    records.extend(legit_records[:max(0, remaining)])
    return records


def run_benchmark(
    package_path: str,
    license_key: str,
    csv_path: str,
    num_samples: int = 500,
) -> Dict[str, Any]:
    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}      REAL-WORLD PRODUCTION BENCHMARK: SEALED SPECIALIST SDM          {RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  Package Path   : {package_path}")
    print(f"  Package Size   : {os.path.getsize(package_path) / (1024*1024):.2f} MB")
    print(f"  Dataset Source : {csv_path}")
    print(f"  Sample Target  : {num_samples:,} real transactions")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    # Step 1: Memory & In-Memory Unlock
    rss_before = get_process_memory_mb()
    print(f"[{time.strftime('%H:%M:%S')}] {CYAN}Phase 1: In-Memory Cryptographic Decryption & Engine Load...{RESET}")
    t_unlock_start = time.perf_counter()
    client = SealedPackageClient(package_path=package_path, license_key=license_key)
    unlock_elapsed_ms = (time.perf_counter() - t_unlock_start) * 1000.0
    rss_after = get_process_memory_mb()

    print(f"  ✓ Decryption & Schema Binding : {GREEN}{unlock_elapsed_ms:.2f} ms{RESET}")
    print(f"  ✓ Process RSS Memory Delta   : {YELLOW}+{rss_after - rss_before:.2f} MB{RESET} (Total: {rss_after:.2f} MB)")
    print(f"  ✓ Model Domain               : {client.domain}")
    print(f"  ✓ Valid Decision Choices     : {client.choices}")

    # Step 2: Load Test Telemetry
    print(f"\n[{time.strftime('%H:%M:%S')}] {CYAN}Phase 2: Ingesting Real Transaction Telemetry...{RESET}")
    dataset = load_benchmark_dataset(csv_path, max_records=num_samples)
    total_tx = len(dataset)
    fraud_tx = sum(1 for d in dataset if d["label"] == 1)
    legit_tx = total_tx - fraud_tx
    print(f"  ✓ Loaded {total_tx:,} real transactions ({legit_tx:,} legitimate, {fraud_tx:,} confirmed fraud)")

    # Step 3: Cold Start Measurement
    print(f"\n[{time.strftime('%H:%M:%S')}] {CYAN}Phase 3: Measuring Cold-Start vs Warm Latency...{RESET}")
    cold_vec = dataset[0]["features"]
    t_cold = time.perf_counter()
    _ = client.decide(cold_vec)
    cold_latency_ms = (time.perf_counter() - t_cold) * 1000.0
    print(f"  ✓ Cold-Start First Inference : {YELLOW}{cold_latency_ms:.3f} ms{RESET}")

    # Warmup runs
    for i in range(min(5, len(dataset))):
        _ = client.decide(dataset[i]["features"])

    # Step 4: High-Resolution Empirical Profiling
    print(f"\n[{time.strftime('%H:%M:%S')}] {CYAN}Phase 4: Running Continuous Transaction Replay Loop ({total_tx:,} tx)...{RESET}")
    latencies_us: List[float] = []
    decisions: List[str] = []
    confidences: List[float] = []
    true_labels: List[int] = []
    amounts: List[float] = []

    t_loop_start = time.perf_counter()
    for idx, item in enumerate(dataset):
        vec = item["features"]
        t0 = time.perf_counter_ns()
        res = client.decide(vec)
        dt_us = (time.perf_counter_ns() - t0) / 1000.0  # microseconds

        latencies_us.append(dt_us)
        decisions.append(res.choice)
        confidences.append(res.confidence)
        true_labels.append(item["label"])
        amounts.append(item["amount"])

        if (idx + 1) % 250 == 0 or (idx + 1) == total_tx:
            pct = ((idx + 1) / total_tx) * 100.0
            print(f"  ↳ Replayed {idx + 1:,} / {total_tx:,} transactions ({pct:.1f}%) | Current P50: {np.percentile(latencies_us, 50)/1000.0:.2f} ms")

    total_bench_seconds = time.perf_counter() - t_loop_start
    throughput_tps = total_tx / max(total_bench_seconds, 1e-6)

    # Step 5: Statistical & Financial Calculations
    latencies_ms = np.array(latencies_us) / 1000.0
    p50_ms = float(np.percentile(latencies_ms, 50))
    p90_ms = float(np.percentile(latencies_ms, 90))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    mean_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms))

    # Confusion Matrix
    # Pred: 'authorized' -> 0, 'declined' -> 1
    tp, fp, tn, fn = 0, 0, 0, 0
    vol_authorized = 0.0
    vol_declined = 0.0
    fraud_dollar_prevented = 0.0
    false_decline_dollar_cost = 0.0

    for dec, conf, truth, amt in zip(decisions, confidences, true_labels, amounts):
        pred_label = 0 if dec == "authorized" else 1
        if pred_label == 0:
            vol_authorized += amt
        else:
            vol_declined += amt

        if truth == 1 and pred_label == 1:
            tp += 1
            fraud_dollar_prevented += amt
        elif truth == 0 and pred_label == 1:
            fp += 1
            false_decline_dollar_cost += amt
        elif truth == 0 and pred_label == 0:
            tn += 1
        elif truth == 1 and pred_label == 0:
            fn += 1

    accuracy = (tp + tn) / max(total_tx, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
    dispute_rate_pct = (fn / max(tn + fn, 1)) * 100.0

    # Step 6: Terminal HUD Report
    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}                   PRODUCTION BENCHMARK RESULTS                       {RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  {BOLD}Transactions Evaluated :{RESET} {total_tx:,}")
    print(f"  {BOLD}Total Benchmark Time   :{RESET} {total_bench_seconds:.2f} seconds")
    print(f"  {BOLD}Throughput Capacity    :{RESET} {GREEN}{throughput_tps:,.1f} transactions / second{RESET}")
    print(f"  {BOLD}RSS Host RAM Usage     :{RESET} {rss_after:.1f} MB (Delta: +{rss_after - rss_before:.1f} MB)")
    print(f"{DIM}----------------------------------------------------------------------{RESET}")
    print(f"  {BOLD}LATENCY PROFILE (In-Process Host RAM):{RESET}")
    print(f"    • P50 Latency (Median) : {CYAN}{p50_ms:.3f} ms{RESET} ({p50_ms * 1000.0:.1f} µs)")
    print(f"    • P90 Latency          : {CYAN}{p90_ms:.3f} ms{RESET}")
    print(f"    • P95 Latency          : {CYAN}{p95_ms:.3f} ms{RESET}")
    print(f"    • P99 Latency (Tail)   : {CYAN}{p99_ms:.3f} ms{RESET}")
    print(f"    • Min / Max Latency    : {min_ms:.3f} ms / {max_ms:.3f} ms")
    print(f"    • Mean ± Std Dev       : {mean_ms:.3f} ms ± {std_ms:.3f} ms")
    print(f"{DIM}----------------------------------------------------------------------{RESET}")
    print(f"  {BOLD}STATISTICAL ACCURACY & RECALL:{RESET}")
    print(f"    • Empirical Accuracy   : {GREEN}{accuracy * 100.0:.2f}%{RESET}")
    print(f"    • Precision Score      : {precision * 100.0:.2f}%")
    print(f"    • Recall (Fraud Catch) : {recall * 100.0:.2f}%")
    print(f"    • F1-Score             : {f1:.4f}")
    print(f"    • Confusion Matrix     : TN={tn:,} | FP={fp:,} | TP={tp:,} | FN={fn:,}")
    print(f"{DIM}----------------------------------------------------------------------{RESET}")
    print(f"  {BOLD}FINANCIAL & REGULATORY VALUE:{RESET}")
    print(f"    • Total Authorized Vol : ${vol_authorized:,.2f}")
    print(f"    • Fraud Vol Prevented  : {GREEN}${fraud_dollar_prevented:,.2f}{RESET}")
    print(f"    • False Decline Loss   : ${false_decline_dollar_cost:,.2f}")
    print(f"    • Visa VDMP Ratio      : {dispute_rate_pct:.3f}% (Cap: < 0.900%) -> {GREEN if dispute_rate_pct < 0.9 else RED}{'COMPLIANT' if dispute_rate_pct < 0.9 else 'BREACH'}{RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "package_path": package_path,
        "sample_size": total_tx,
        "throughput_tps": round(throughput_tps, 1),
        "latency_ms": {
            "p50": round(p50_ms, 3),
            "p90": round(p90_ms, 3),
            "p95": round(p95_ms, 3),
            "p99": round(p99_ms, 3),
            "min": round(min_ms, 3),
            "max": round(max_ms, 3),
            "mean": round(mean_ms, 3),
            "std": round(std_ms, 3),
            "cold_start": round(cold_latency_ms, 3),
        },
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "financials": {
            "volume_authorized": round(vol_authorized, 2),
            "fraud_prevented": round(fraud_dollar_prevented, 2),
            "false_decline_cost": round(false_decline_dollar_cost, 2),
            "vdmp_dispute_rate_pct": round(dispute_rate_pct, 4),
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Real Production Benchmark for SDM Package")
    parser.add_argument(
        "--package",
        default="artifacts/mac_factory/mac_fintech_specialist_sdm.sdmpkg",
        help="Path to .sdmpkg sealed model package",
    )
    parser.add_argument(
        "--key",
        default="lic_mac_specialist_local_2026",
        help="License decryption key",
    )
    parser.add_argument(
        "--data",
        default="data/creditcard_sample.csv",
        help="Path to real transaction dataset CSV",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of real transaction rows to benchmark (default: 500)",
    )
    parser.add_argument(
        "--out",
        default="artifacts/benchmark_report_production.json",
        help="Output path for benchmark JSON report",
    )
    args = parser.parse_args()

    pkg_path = os.path.join(_REPO_ROOT, args.package) if not os.path.isabs(args.package) else args.package
    data_path = os.path.join(_REPO_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    out_path = os.path.join(_REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out

    if not os.path.exists(pkg_path):
        print(f"[-] Error: Package not found at {pkg_path}")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"[-] Error: Dataset not found at {data_path}")
        sys.exit(1)

    summary = run_benchmark(
        package_path=pkg_path,
        license_key=args.key,
        csv_path=data_path,
        num_samples=args.samples,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Benchmark JSON report saved to: {out_path}\n")


if __name__ == "__main__":
    main()
