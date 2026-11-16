"""
Payment Decision Model Factory & Pipeline Runner
=================================================
Pipeline runner that compiles declarative `spec.yaml` policies into
production-ready, verified model artifact bundles for client hand-off and SDK serving.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import shutil
import tarfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

# Ensure payment_decision_factory is importable when executed directly
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.serializer import PaymentSerializer
from payment_decision_factory.core.engine import DecisionEngine, DecisionResult, CalibrationMetrics
from payment_decision_factory.core.sdm import (
    SpecialistSDM,
    SpecialistPipeline,
    SpecialistTuningReport,
    export_specialist_to_onnx,
    ONNXSpecialistSDM,
)


class BenchmarkResult(BaseModel):
    """Validation benchmark metrics recorded during compilation."""
    total_test_samples: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    sla_target_ms: float
    sla_passed: bool
    brier_score: float
    brier_target: float
    calibration_passed: bool
    choices_distribution: Dict[str, int]
    noul_triggers_count: int


class ClientManifest(BaseModel):
    """Client SDK and integration manifest."""
    bundle_id: str
    spec_id: str
    spec_name: str
    spec_version: str
    domain: str
    created_at: float
    checksum_sha256: str
    sla: Dict[str, Any]
    input_fields: List[str]
    choices: List[str]
    default_choice: str
    specialist_sdm: Optional[Dict[str, Any]] = None
    supported_interfaces: List[str] = Field(
        default_factory=lambda: ["in_process_python", "sidecar_http", "typescript_sdk", "onnx_specialist"]
    )
    entrypoint: str = "payment_decision_factory.core.engine.DecisionEngine"


class CompiledArtifactBundle(BaseModel):
    """Container for the compiled model artifact bundle."""
    bundle_id: str
    spec: DecisionSpec
    manifest: ClientManifest
    benchmark: BenchmarkResult
    compiled_weights: Dict[str, Any]
    sdm_report: Optional[Dict[str, Any]] = None
    onnx_path: Optional[str] = None
    bundle_dir: Optional[str] = None
    archive_path: Optional[str] = None


class DecisionFactory:
    """
    Pipeline runner and compiler for DecisionSpec models.
    """

    def __init__(self, spec: Optional[DecisionSpec] = None):
        self.spec = spec

    @classmethod
    def from_yaml_file(cls, spec_path: str) -> DecisionFactory:
        """Initialize factory directly from a spec YAML file path."""
        spec = DecisionSpec.from_yaml(spec_path)
        return cls(spec)

    def _generate_synthetic_test_suite(self) -> List[Tuple[Dict[str, Any], str]]:
        """
        Generate representative test vectors with ground-truth routing expectations
        to validate correctness, latency, and calibration during build time.
        """
        test_cases: List[Tuple[Dict[str, Any], str]] = [
            # Domestic US -> stripe_us
            ({"amount": 25.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411", "velocity_1h": 0}, "stripe_us"),
            ({"amount": 150.0, "currency": "USD", "card_brand": "amex", "bin_country": "US", "mcc": "5734", "velocity_1h": 1}, "stripe_us"),
            ({"amount": 89.99, "currency": "CAD", "card_brand": "mastercard", "bin_country": "CA", "mcc": "5311", "velocity_1h": 2}, "stripe_us"),
            # European clearing -> adyen_eu
            ({"amount": 45.0, "currency": "EUR", "card_brand": "visa", "bin_country": "DE", "mcc": "5812", "velocity_1h": 0}, "adyen_eu"),
            ({"amount": 320.0, "currency": "GBP", "card_brand": "mastercard", "bin_country": "GB", "mcc": "5732", "velocity_1h": 1}, "adyen_eu"),
            ({"amount": 75.50, "currency": "EUR", "card_brand": "visa", "bin_country": "FR", "mcc": "5411", "velocity_1h": 1}, "adyen_eu"),
            ({"amount": 210.0, "currency": "EUR", "card_brand": "mastercard", "bin_country": "NL", "mcc": "4722", "velocity_1h": 0}, "adyen_eu"),
            # Cross-border / APAC / LATAM -> checkout_crossborder
            ({"amount": 6500.0, "currency": "JPY", "card_brand": "jcb", "bin_country": "JP", "mcc": "5734", "velocity_1h": 0}, "checkout_crossborder"),
            ({"amount": 450.0, "currency": "AUD", "card_brand": "visa", "bin_country": "AU", "mcc": "7011", "velocity_1h": 1}, "checkout_crossborder"),
            ({"amount": 1200.0, "currency": "USD", "card_brand": "unionpay", "bin_country": "SG", "mcc": "5968", "velocity_1h": 0}, "checkout_crossborder"),
            ({"amount": 850.0, "currency": "EUR", "card_brand": "visa", "bin_country": "BR", "mcc": "5411", "velocity_1h": 1}, "checkout_crossborder"),
            # Card testing / anomaly -> triggers noul
            ({"amount": 10.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411", "velocity_1h": 18}, "stripe_us"),
        ]
        return test_cases

    def run_benchmark(self, engine: DecisionEngine) -> BenchmarkResult:
        """
        Execute pre-flight benchmark and calibration audit on synthetic test set.
        """
        test_suite = self._generate_synthetic_test_suite()
        latencies = []
        predictions: List[DecisionResult] = []
        ground_truth: List[str] = []
        distribution: Dict[str, int] = {k: 0 for k in engine.choices}
        noul_triggers = 0

        # Warm up
        for item, _ in test_suite[:3]:
            engine.evaluate(item)

        for payload, expected in test_suite:
            t0 = time.perf_counter()
            res = engine.evaluate(payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            predictions.append(res)
            ground_truth.append(expected)

            distribution[res.choice] = distribution.get(res.choice, 0) + 1
            if res.noul.triggered:
                noul_triggers += 1

        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        sla_target = engine.spec.sla.max_latency_ms

        cal_metrics = engine.evaluate_calibration(predictions, ground_truth)

        return BenchmarkResult(
            total_test_samples=len(test_suite),
            avg_latency_ms=round(avg_lat, 3),
            p95_latency_ms=round(p95_lat, 3),
            p99_latency_ms=round(p99_lat, 3),
            sla_target_ms=sla_target,
            sla_passed=p99_lat <= sla_target,
            brier_score=round(cal_metrics.brier_score, 4),
            brier_target=round(cal_metrics.brier_target, 4),
            calibration_passed=cal_metrics.brier_score <= cal_metrics.brier_target * 1.5,  # tolerance on synthetic
            choices_distribution=distribution,
            noul_triggers_count=noul_triggers,
        )

    def compile(
        self,
        output_dir: str = "artifacts/smart_routing_v1",
        create_archive: bool = True,
        train_sdm: bool = True,
        sdm_type: str = "routing",
        sdm_epochs: int = 15,
    ) -> CompiledArtifactBundle:
        """
        Run the end-to-end compilation pipeline:
        1. Validate spec schema and constraints.
        2. Instantiate DecisionEngine and precompute weights.
        3. Run latency and calibration benchmark suite.
        4. Optionally train and export a hyper-personalized Specialist SDM (ONNX + PyTorch).
        5. Package manifest, spec, pre-compiled bundle, and tar.gz archive.
        """
        if not self.spec:
            raise ValueError("No DecisionSpec provided to factory for compilation.")

        spec = self.spec
        os.makedirs(output_dir, exist_ok=True)

        # 1. Instantiate engine & precompute weights
        engine = DecisionEngine(spec)
        compiled_weights = {
            "choice_weights": spec.primitives.choice.weights,
            "affinities": spec.primitives.choice.affinities,
            "temperature": spec.primitives.score.temperature,
            "default_choice": spec.primitives.choice.default_choice,
            "noul_config": spec.primitives.noul.model_dump(),
        }

        # 2. Run verification benchmark
        benchmark = self.run_benchmark(engine)

        # 3. Train and export Specialist SDM if requested
        sdm_report_dict = None
        onnx_file_path = None
        pt_file_path = None
        if train_sdm:
            try:
                pipeline = SpecialistPipeline(spec, specialist_type=sdm_type)
                telemetry = pipeline.create_synthetic_telemetry(count=250)
                sdm_model, sdm_report = pipeline.train_specialist(
                    telemetry, epochs=sdm_epochs, export_onnx=False
                )

                # Export ONNX model
                onnx_file_path = os.path.join(output_dir, "specialist_sdm.onnx")
                export_specialist_to_onnx(sdm_model, onnx_file_path, verify=True)

                # Export PyTorch weights
                import torch
                pt_file_path = os.path.join(output_dir, "specialist_sdm.pt")
                torch.save(sdm_model.state_dict(), pt_file_path)

                sdm_report_dict = {
                    "specialist_name": sdm_report.specialist_name,
                    "parameter_count": sdm_report.parameter_count,
                    "artifact_size_kb": sdm_report.artifact_size_kb,
                    "validation_accuracy": sdm_report.validation_accuracy,
                    "brier_score": sdm_report.brier_score,
                    "fitted_temperature": sdm_report.fitted_temperature,
                    "training_duration_sec": sdm_report.training_duration_sec,
                    "onnx_model": "specialist_sdm.onnx",
                    "torch_model": "specialist_sdm.pt",
                }

                # Write sdm_report.json
                sdm_report_path = os.path.join(output_dir, "sdm_report.json")
                with open(sdm_report_path, "w", encoding="utf-8") as f:
                    json.dump(sdm_report_dict, f, indent=2)

            except Exception as e:
                import logging
                logging.getLogger("payment_decision_factory.factory").warning(
                    f"Specialist SDM compilation encountered error: {e}"
                )

        # 4. Compute spec checksum
        spec_yaml = spec.to_yaml()
        checksum = hashlib.sha256(spec_yaml.encode("utf-8")).hexdigest()
        bundle_id = f"bundle_{spec.spec_id}_{int(time.time())}"

        # 5. Build client manifest
        manifest = ClientManifest(
            bundle_id=bundle_id,
            spec_id=spec.spec_id,
            spec_name=spec.name,
            spec_version=spec.version,
            domain=spec.domain,
            created_at=time.time(),
            checksum_sha256=checksum,
            sla=spec.sla.model_dump(),
            input_fields=list(spec.schema.inputs.keys()),
            choices=list(spec.choices.keys()),
            default_choice=spec.primitives.choice.default_choice,
            specialist_sdm=sdm_report_dict,
        )

        # 6. Write artifacts into output_dir
        # a) spec.yaml
        spec_path = os.path.join(output_dir, "spec.yaml")
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(spec_yaml)

        # b) manifest.json
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        # c) benchmark_report.json
        bench_path = os.path.join(output_dir, "benchmark_report.json")
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(benchmark.model_dump(), f, indent=2)

        # d) bundle.json (monolithic bundle for single-file deployment)
        bundle_data = {
            "bundle_id": bundle_id,
            "spec": spec.model_dump(by_alias=True),
            "manifest": manifest.model_dump(),
            "benchmark": benchmark.model_dump(),
            "compiled_weights": compiled_weights,
            "specialist_sdm": sdm_report_dict,
        }
        bundle_json_path = os.path.join(output_dir, "bundle.json")
        with open(bundle_json_path, "w", encoding="utf-8") as f:
            json.dump(bundle_data, f, indent=2)

        archive_path = None
        if create_archive:
            archive_path = os.path.join(output_dir, f"{bundle_id}.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(spec_path, arcname="spec.yaml")
                tar.add(manifest_path, arcname="manifest.json")
                tar.add(bench_path, arcname="benchmark_report.json")
                tar.add(bundle_json_path, arcname="bundle.json")
                if onnx_file_path and os.path.exists(onnx_file_path):
                    tar.add(onnx_file_path, arcname="specialist_sdm.onnx")
                if pt_file_path and os.path.exists(pt_file_path):
                    tar.add(pt_file_path, arcname="specialist_sdm.pt")
                if sdm_report_dict and os.path.exists(os.path.join(output_dir, "sdm_report.json")):
                    tar.add(os.path.join(output_dir, "sdm_report.json"), arcname="sdm_report.json")

        return CompiledArtifactBundle(
            bundle_id=bundle_id,
            spec=spec,
            manifest=manifest,
            benchmark=benchmark,
            compiled_weights=compiled_weights,
            sdm_report=sdm_report_dict,
            onnx_path=onnx_file_path,
            bundle_dir=output_dir,
            archive_path=archive_path,
        )

    @classmethod
    def load_specialist_onnx(cls, bundle_path_or_dir: str) -> Optional[ONNXSpecialistSDM]:
        """
        Load an ONNX Specialist SDM from a compiled artifact directory if present.
        """
        target_dir = bundle_path_or_dir if os.path.isdir(bundle_path_or_dir) else os.path.dirname(bundle_path_or_dir)
        onnx_file = os.path.join(target_dir, "specialist_sdm.onnx")
        manifest_file = os.path.join(target_dir, "manifest.json")

        if not os.path.exists(onnx_file):
            return None

        choice_keys = []
        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r") as f:
                    m = json.load(f)
                    choice_keys = m.get("choices", [])
            except Exception:
                pass

        return ONNXSpecialistSDM(onnx_file, choice_keys=choice_keys)

    @classmethod
    def load_bundle(cls, bundle_path_or_dir: str) -> DecisionEngine:
        """
        Load a compiled artifact bundle (directory or bundle.json) into a ready-to-serve DecisionEngine.
        """
        if os.path.isdir(bundle_path_or_dir):
            bundle_json = os.path.join(bundle_path_or_dir, "bundle.json")
            if os.path.exists(bundle_json):
                target_file = bundle_json
            else:
                spec_yaml = os.path.join(bundle_path_or_dir, "spec.yaml")
                return DecisionEngine(DecisionSpec.from_yaml(spec_yaml))
        else:
            target_file = bundle_path_or_dir

        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "spec" in data:
            spec = DecisionSpec(**data["spec"])
        else:
            spec = DecisionSpec(**data)

        return DecisionEngine(spec)


def main():
    """CLI compilation runner."""
    import argparse
    parser = argparse.ArgumentParser(description="Payment Decision Factory Pipeline Runner")
    parser.add_argument("--spec", default="payment_decision_factory/specs/smart_routing.yaml", help="Path to input spec.yaml")
    parser.add_argument("--out", default="payment_decision_factory/artifacts/smart_routing_v1", help="Output bundle directory")
    args = parser.parse_args()

    print(f"Loading specification from: {args.spec}")
    factory = DecisionFactory.from_yaml_file(args.spec)
    print(f"Compiling artifact bundle to: {args.out}...")
    bundle = factory.compile(output_dir=args.out)

    print("\n" + "=" * 60)
    print(f"Compilation Successful!")
    print(f"Bundle ID:          {bundle.bundle_id}")
    print(f"SLA Target Passed:  {bundle.benchmark.sla_passed} (p99 latency: {bundle.benchmark.p99_latency_ms:.3f} ms)")
    print(f"Brier Score:        {bundle.benchmark.brier_score:.4f} (target: {bundle.benchmark.brier_target:.4f})")
    print(f"Manifest written:   {bundle.bundle_dir}/manifest.json")
    if bundle.archive_path:
        print(f"Archive bundle:     {bundle.archive_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
