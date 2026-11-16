"""
Comprehensive Test Suite for GPU-Accelerated Digital Twin Simulation Engine.
=============================================================================
Tests:
1. Tensor Batching & In-VRAM Transaction Encoding (GPUTensorBatch).
2. Parallel Multi-Acquirer Simulation (GPUDigitalTwinSimulator.simulate_tensor_batch).
3. Ultra-High-Speed Counterfactual Replay (GPUDigitalTwinSimulator.run_gpu_counterfactual_replay).
4. GPU-Native Adversarial Perturbation (Gaussian noise, velocity surges).
5. Vectorized Policy Self-Adjustment (GPUSelfAdjustmentOptimizer).
6. Multi-Million Transaction Throughput Benchmarking (<60s SLA validation).
7. Top-Level Convenience API (gpu_simulate, gpu_counterfactual_replay).
"""

from __future__ import annotations

import os
import sys

# Standard library shielding
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code) and (
    "code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")
):
    import importlib.util

    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from payment_decision_factory.sandbox.mock_traffic import MockTrafficGenerator
from payment_decision_factory.sandbox.gpu_simulator import (
    GPUTensorBatch,
    GPUSimulationOutcome,
    GPUCounterfactualReport,
    GPUDigitalTwinSimulator,
    GPUSelfAdjustmentOptimizer,
    gpu_simulate,
    gpu_counterfactual_replay,
    get_optimal_device,
)


@pytest.fixture
def optimal_device():
    return get_optimal_device()


@pytest.fixture
def sample_records():
    gen = MockTrafficGenerator(seed=42)
    return gen.generate_batch("mixed", count=500)


@pytest.fixture
def tensor_batch(sample_records, optimal_device):
    return GPUTensorBatch.from_records(sample_records, device=optimal_device)


class TestGPUTensorBatch:
    """Tests for in-VRAM columnar transaction tensor representation."""

    def test_from_records_conversion(self, tensor_batch):
        assert tensor_batch.batch_size == 500
        assert tensor_batch.amounts.shape == (500,)
        assert tensor_batch.currencies.shape == (500,)
        assert tensor_batch.bin_countries.shape == (500,)
        assert tensor_batch.mccs.shape == (500,)
        assert tensor_batch.velocities.shape == (500,)
        assert tensor_batch.historical_rails.shape == (500,)
        assert tensor_batch.amounts.dtype == torch.float32

    def test_generate_synthetic_gpu_batch(self, optimal_device):
        batch = GPUTensorBatch.generate_synthetic_gpu_batch(
            count=10_000,
            device=optimal_device,
            seed=101,
        )
        assert batch.batch_size == 10_000
        assert batch.amounts.min() >= 2.50
        assert batch.amounts.max() <= 2500.0
        assert batch.velocities.min() >= 1


class TestGPUDigitalTwinSimulation:
    """Tests for vectorized multi-rail simulation in GPU memory."""

    def test_simulate_tensor_batch(self, tensor_batch, optimal_device):
        sim = GPUDigitalTwinSimulator(device=str(optimal_device))
        outcome = sim.simulate_tensor_batch(tensor_batch)

        assert isinstance(outcome, GPUSimulationOutcome)
        assert outcome.total_transactions == 500
        assert outcome.approved_count + outcome.declined_count == 500
        assert 0.0 <= outcome.approval_rate <= 1.0
        assert outcome.total_fees_usd > 0.0
        assert outcome.average_fee_usd > 0.0
        assert 0.0 <= outcome.brier_score <= 1.0
        assert outcome.throughput_tps > 0.0
        assert outcome.p50_latency_ms < 5.0

        d = outcome.to_dict()
        assert d["total_transactions"] == 500
        assert "throughput_tps" in d
        assert "approval_rate" in d

    def test_gpu_counterfactual_replay(self, tensor_batch, optimal_device):
        sim = GPUDigitalTwinSimulator(device=str(optimal_device))
        report = sim.run_gpu_counterfactual_replay(tensor_batch, baseline_rail_idx=0)

        assert isinstance(report, GPUCounterfactualReport)
        assert report.total_replayed == 500
        assert 0.0 <= report.baseline_approval_rate <= 1.0
        assert 0.0 <= report.model_approval_rate <= 1.0
        assert isinstance(report.approval_uplift_bps, float)
        assert report.routing_changes_count >= 0
        assert 0.0 <= report.routing_change_pct <= 100.0
        assert report.throughput_tps > 0.0

        d = report.to_dict()
        assert d["total_replayed"] == 500
        assert "approval_uplift_bps" in d
        assert "net_financial_gain_usd" in d

    def test_adversarial_perturbation(self, tensor_batch, optimal_device):
        sim = GPUDigitalTwinSimulator(device=str(optimal_device))
        perturbed = sim.run_adversarial_perturbation(
            tensor_batch,
            amount_noise_std=0.10,
            velocity_surge_prob=0.15,
        )

        assert perturbed.batch_size == tensor_batch.batch_size
        # Verify amounts shifted due to noise
        assert not torch.equal(perturbed.amounts, tensor_batch.amounts)
        # Verify velocities surged
        assert perturbed.velocities.max() >= tensor_batch.velocities.max()


class TestGPUSelfAdjustmentOptimizer:
    """Tests for parallel GPU policy calibration and weight tuning."""

    def test_optimize_temperature_gpu(self, tensor_batch, optimal_device):
        optimizer = GPUSelfAdjustmentOptimizer(
            simulator=GPUDigitalTwinSimulator(device=str(optimal_device))
        )
        opt_temp, brier = optimizer.optimize_temperature_gpu(tensor_batch)

        assert isinstance(opt_temp, float)
        assert 0.2 <= opt_temp <= 3.0
        assert 0.0 <= brier <= 1.0

    def test_self_adjust_gpu_end_to_end(self, tensor_batch, optimal_device):
        optimizer = GPUSelfAdjustmentOptimizer(
            simulator=GPUDigitalTwinSimulator(device=str(optimal_device))
        )
        res = optimizer.self_adjust_gpu(tensor_batch)

        assert "calibrated_temperature" in res
        assert "brier_score" in res
        assert "optimized_weights" in res
        assert "currency_affinity" in res["optimized_weights"]
        assert res["optimization_time_sec"] > 0.0


class TestMassiveThroughputBenchmark:
    """Tests multi-million transaction throughput benchmark (<60s SLA)."""

    def test_benchmark_sub_60s_sla(self, optimal_device):
        sim = GPUDigitalTwinSimulator(device=str(optimal_device))
        # Benchmark 100,000 transactions for fast CI verification
        res = sim.benchmark_massive_replay(total_records=100_000, chunk_size=50_000)

        assert res["total_records_replayed"] == 100_000
        assert res["total_elapsed_sec"] < 10.0
        assert res["sub_60s_sla_achieved"] is True
        assert res["throughput_tps"] > 10_000  # High-throughput line rate


class TestConvenienceAPI:
    """Tests top-level gpu_simulate and gpu_counterfactual_replay entrypoints."""

    def test_gpu_simulate_shortcut(self, sample_records, optimal_device):
        outcome = gpu_simulate(sample_records, device=str(optimal_device))
        assert isinstance(outcome, GPUSimulationOutcome)
        assert outcome.total_transactions == 500

    def test_gpu_counterfactual_replay_shortcut(self, sample_records, optimal_device):
        report = gpu_counterfactual_replay(sample_records, device=str(optimal_device))
        assert isinstance(report, GPUCounterfactualReport)
        assert report.total_replayed == 500
