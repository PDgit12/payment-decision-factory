#!/usr/bin/env python3
"""
Interactive 1-Command Mac GPU Specialist Decision Factory Runner
================================================================
Executes the autonomous assembly line on your local Mac laptop using
Apple Silicon Metal Performance Shaders (MPS) at $0 cloud cost.

Usage Examples:
    # 1. Fast Specialist Build (Tier: Small ~800k params, ~5-8 seconds):
    python scripts/run_mac_specialist_factory.py --tier small

    # 2. Medium Enterprise Specialist (Tier: Medium ~5.2M params):
    python scripts/run_mac_specialist_factory.py --tier medium --epochs 3

    # 3. Full Benchmark (Using complete 284k real dataset):
    python scripts/run_mac_specialist_factory.py --tier small --dataset full
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

# Disable artificial MPS memory allocation limit on Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Robust unshadowing of standard library code module if local code.py exists
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code):
    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FACTORY_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_FACTORY_DIR, ".."))

for p in [_REPO_ROOT, _FACTORY_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from payment_decision_factory.mac_factory import (
    MacFactoryConfig,
    MacHardwareProfile,
    MacSpecialistFactory,
)

# ANSI terminal formatting
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def print_banner(hardware: MacHardwareProfile):
    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}  APPLE SILICON SPECIALIST DECISION MODEL FACTORY (MAC NATIVE){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  {BOLD}Processor / Chip  :{RESET} {hardware.chip_model}")
    print(f"  {BOLD}Operating System  :{RESET} {hardware.os_version}")
    print(f"  {BOLD}Unified Memory    :{RESET} {hardware.total_ram_gb} GB RAM (Unified Architecture)")
    print(f"  {BOLD}MPS GPU Engine    :{RESET} {'ACTIVE (Metal Performance Shaders)' if hardware.mps_available else 'CPU Vectorized'}")
    print(f"  {BOLD}PyTorch Runtime   :{RESET} {hardware.pytorch_version} | Device: {hardware.optimal_device}")
    print(f"  {BOLD}Cloud Cost        :{RESET} $0.00 (100% Free Local Hardware)")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")


def on_factory_progress(step: str, meta: dict):
    ts = time.strftime("%H:%M:%S")
    if step == "DATA_INGESTED":
        print(f"[{ts}] {GREEN}✓ Station 1 (Data Ingested):{RESET} {meta['total_records']:,} real transactions | Fraud Rate: {meta['fraud_base_rate_pct']}%")
    elif step == "GPU_PROVISIONED":
        print(f"[{ts}] {GREEN}✓ Station 2 (GPU Provision):{RESET} {meta['device'].upper()} | Tier: {meta['tier']} ({meta['parameters']:,} params, {meta['model_size_fp32_mb']} MB)")
    elif step == "TRAINING_EPOCH":
        print(f"     ↳ Epoch {meta['epoch']}: Cross-Entropy Loss = {meta['avg_loss']}")
    elif step == "TRAINING_COMPLETED":
        print(f"[{ts}] {GREEN}✓ Station 3 (MPS Training) :{RESET} Finished in {meta['train_time_seconds']}s | Val Acc: {meta['validation_accuracy']*100:.2f}% | Brier Score: {meta['brier_score']}")
    elif step == "DIGITAL_TWIN_CERTIFIED":
        print(f"[{ts}] {GREEN}✓ Station 4 (Digital Twin) :{RESET} {meta['status']} | P50: {meta['p50_latency_ms']}ms | P99: {meta['p99_latency_ms']}ms | Audit Hash: {meta['sha256_audit_hash'][:16]}...")
    elif step == "PACKAGE_SEALED":
        print(f"[{ts}] {GREEN}✓ Station 5 (Package Sealed):{RESET} AES-256-GCM package created ({meta['size_mb']} MB) at {os.path.basename(meta['sealed_package_path'])}")
    elif step == "LIVE_SDK_VERIFIED":
        print(f"[{ts}] {GREEN}✓ Station 6 (Live SDK Test) :{RESET} In-Memory Evaluation P50: {meta['in_memory_p50_ms']}ms | P99: {meta['in_memory_p99_ms']}ms")
    elif step == "DELIVERY_DISPATCHED":
        print(f"[{ts}] {CYAN}★ Assembly Line Complete   :{RESET} Total turnaround time: {meta['total_elapsed_seconds']}s\n")


def print_receipt(receipt):
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}                AUTONOMOUS FACTORY DELIVERY RECEIPT                  {RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  {BOLD}Client ID            :{RESET} {receipt.client_id}")
    print(f"  {BOLD}Certification Status :{RESET} {GREEN}{receipt.status}{RESET}")
    print(f"  {BOLD}Specialist Tier      :{RESET} {receipt.parameter_tier} ({receipt.parameter_count:,} parameters)")
    print(f"  {BOLD}Raw Model Size       :{RESET} {receipt.raw_fp32_mb} MB (FP32)")
    print(f"  {BOLD}GPU Training Elapsed :{RESET} {receipt.train_time_seconds} seconds on Apple Silicon MPS")
    print(f"  {BOLD}Empirical Accuracy   :{RESET} {receipt.validation_accuracy * 100:.2f}%")
    print(f"  {BOLD}Calibrated Brier     :{RESET} {receipt.brier_score}")
    print(f"  {BOLD}P50 Latency (Median) :{RESET} {receipt.p50_latency_ms} ms ({receipt.p50_latency_ms * 1000.0:.1f} microseconds)")
    print(f"  {BOLD}P99 Latency (SLA)    :{RESET} {receipt.p99_latency_ms} ms (Target: <15.0 ms)")
    print(f"  {BOLD}Throughput Capacity  :{RESET} {receipt.throughput_tps:,.1f} transactions / second")
    print(f"  {BOLD}Cryptographic Hash   :{RESET} SHA-256: {receipt.sha256_audit_hash}")
    print(f"  {BOLD}Sealed Binary Path   :{RESET} {receipt.sealed_package_path}")
    print(f"  {BOLD}Readiness Cert JSON  :{RESET} {receipt.readiness_cert_path}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    print(f"{BOLD}{YELLOW}Quick Client Integration Code (Python SDK):{RESET}")
    print(f"""
from payment_decision_sdk.sealed import SealedPackageClient

client = SealedPackageClient(
    package_path="{receipt.sealed_package_path}",
    license_key="lic_mac_specialist_local_2026"
)

# 0.2ms In-Memory Live Checkout Decision
result = client.evaluate(transaction_feature_vector)
print("Decision:", result.choice, "| Confidence:", result.confidence)
""")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Mac GPU Specialist Decision Model Factory")
    parser.add_argument("--tier", choices=["small", "medium", "large", "flagship"], default="small",
                        help="Specialist SDM parameter tier: small (~800k), medium (~5.2M), large (~113M), flagship (~201M)")
    parser.add_argument("--dataset", choices=["sample", "full"], default="sample",
                        help="Dataset mode: sample (10,000 real rows) or full (284,807 real rows)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for MPS DataLoader (default: 64)")
    parser.add_argument("--client", default="mac_fintech_specialist", help="Client identifier")
    parser.add_argument("--key", default="lic_mac_specialist_local_2026", help="Client license encryption key")
    parser.add_argument("--out", default="artifacts/mac_factory", help="Output directory for generated packages")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 10,
                        help="Number of CPU threads to saturate all cores (default: all 10 cores)")

    args = parser.parse_args()

    # Maximize Apple Silicon core usage
    try:
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
    except Exception:
        pass

    cfg = MacFactoryConfig(
        client_id=args.client,
        parameter_tier=args.tier,
        dataset_mode=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        license_key=args.key,
        output_dir=os.path.join(_FACTORY_DIR, args.out),
    )

    factory = MacSpecialistFactory(config=cfg)
    print_banner(factory.hardware)

    print(f"{BOLD}Executing 6-Station Autonomous Assembly Line on Apple Silicon GPU...{RESET}\n")
    receipt = factory.run(progress_callback=on_factory_progress)
    print_receipt(receipt)


if __name__ == "__main__":
    main()
