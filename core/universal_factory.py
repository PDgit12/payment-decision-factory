"""
Universal Decision Model Factory
=================================
Domain-agnostic Specialist SDM factory supporting arbitrary business domains:
- Payments & Smart Routing
- Credit Underwriting & Lending Decisions
- Freight Logistics & Carrier Dispatch
- Dynamic Pricing & Revenue Optimization
- Fraud & Cybersecurity Risk Scoring

Auto-profiles arbitrary client tabular/JSON telemetry, builds dynamic categorical
entity embeddings, scales continuous features, trains compact deep residual networks
in seconds without GPUs, and exports sealed, tamper-evident `.sdmpkg` packages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Prevent local code.py shadowing Python standard library module
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code) and ("code" not in sys.modules or not hasattr(sys.modules.get("code"), "InteractiveConsole")):
    import importlib.util
    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.sdm import GatedResidualBlock
from payment_decision_factory.core.sealed_package import SealedPackage
from payment_decision_factory.core.enterprise_sdm import (
    EnterpriseSubBillionSDM,
    export_enterprise_sdm_to_onnx,
    EnterpriseONNXRunner,
)

logger = logging.getLogger("payment_decision_factory.universal_factory")


@dataclass
class UniversalFeatureSchema:
    """Schema specification for universal domain features."""
    name: str
    dtype: str  # 'numerical', 'categorical', 'boolean'
    log_scale: bool = False
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    categories: List[str] = field(default_factory=list)
    cat_to_idx: Dict[str, int] = field(default_factory=dict)


@dataclass
class UniversalTrainingReport:
    """Comprehensive training metrics report supporting attribute and dict indexing."""
    domain_id: str
    train_samples: int
    val_samples: int
    input_dim: int
    parameter_count: int
    epochs: int
    validation_accuracy: float
    brier_score: float
    training_duration_sec: float
    tier: str = "compact"
    sealed_package_id: Optional[str] = None
    output_package_path: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "input_dim": self.input_dim,
            "parameter_count": self.parameter_count,
            "epochs": self.epochs,
            "validation_accuracy": self.validation_accuracy,
            "brier_score": self.brier_score,
            "training_duration_sec": self.training_duration_sec,
            "tier": self.tier,
            "sealed_package_id": self.sealed_package_id,
            "output_package_path": self.output_package_path,
        }


class UniversalDomainSpec:
    """Domain specification for any arbitrary decision model."""

    def __init__(
        self,
        domain_id: Optional[str] = None,
        domain_name: Optional[str] = None,
        task_type: str = "classification",
        target_field: str = "target",
        choices: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        sla_target_ms: float = 1.0,
        feature_schemas: Optional[Dict[str, UniversalFeatureSchema]] = None,
        domain: Optional[str] = None,
        classes: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        **kwargs,
    ):
        self.domain_id = domain_id or domain or "default_domain"
        self.domain_name = domain_name or self.domain_id
        self.task_type = task_type
        self.target_field = target_field
        self.choices = choices or classes or []
        self.numerical_features = numerical_features or numeric_features or []
        self.categorical_features = categorical_features or []
        self.sla_target_ms = sla_target_ms
        self.feature_schemas = feature_schemas or {}

    @property
    def domain(self) -> str:
        return self.domain_id

    @property
    def classes(self) -> List[str]:
        return self.choices

    @property
    def numeric_features(self) -> List[str]:
        return self.numerical_features

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "domain_name": self.domain_name,
            "task_type": self.task_type,
            "target_field": self.target_field,
            "choices": self.choices,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "sla_target_ms": self.sla_target_ms,
        }


class UniversalFeatureEncoder:
    """Auto-profiles client records and transforms arbitrary inputs into normalized tensors."""

    def __init__(self, spec: UniversalDomainSpec):
        self.spec = spec
        self.num_features = spec.numerical_features
        self.cat_features = spec.categorical_features

    def fit(self, records: List[Dict[str, Any]]) -> UniversalFeatureEncoder:
        """Profile dataset to learn categorical vocabularies and numerical statistics."""
        # 1. Profile numerical fields
        for num_f in self.num_features:
            vals = [float(r[num_f]) for r in records if num_f in r and r[num_f] is not None]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals)) or 1.0
                min_val = float(np.min(vals))
                max_val = float(np.max(vals))
            else:
                mean_val, std_val, min_val, max_val = 0.0, 1.0, 0.0, 1.0

            self.spec.feature_schemas[num_f] = UniversalFeatureSchema(
                name=num_f,
                dtype="numerical",
                mean=round(mean_val, 4),
                std=round(std_val, 4),
                min_val=round(min_val, 4),
                max_val=round(max_val, 4),
            )

        # 2. Profile categorical fields
        for cat_f in self.cat_features:
            unique_cats = sorted(list({str(r[cat_f]) for r in records if cat_f in r and r[cat_f] is not None}))
            # Index 0 is reserved for UNK (unseen categories)
            cat_map = {cat: i + 1 for i, cat in enumerate(unique_cats)}
            self.spec.feature_schemas[cat_f] = UniversalFeatureSchema(
                name=cat_f,
                dtype="categorical",
                categories=unique_cats,
                cat_to_idx=cat_map,
            )

        return self

    def transform_record(self, record: Dict[str, Any]) -> np.ndarray:
        """Convert a single dictionary record to a dense 1D float32 vector."""
        vec = []

        # Process continuous numericals (z-scored or normalized)
        for num_f in self.num_features:
            schema = self.spec.feature_schemas.get(num_f)
            raw_val = float(record.get(num_f, schema.mean if schema else 0.0))
            if schema and schema.std > 0:
                norm_val = (raw_val - schema.mean) / schema.std
            else:
                norm_val = raw_val
            vec.append(float(norm_val))

        # Process one-hot / index representation for categoricals
        for cat_f in self.cat_features:
            schema = self.spec.feature_schemas.get(cat_f)
            val = str(record.get(cat_f, "UNK"))
            num_cats = len(schema.categories) if schema else 0
            cat_idx = schema.cat_to_idx.get(val, 0) if schema else 0

            # One-hot vector per categorical field
            one_hot = [0.0] * (num_cats + 1)
            if 0 <= cat_idx < len(one_hot):
                one_hot[cat_idx] = 1.0
            vec.extend(one_hot)

        return np.array(vec, dtype=np.float32)

    @property
    def total_dim(self) -> int:
        """Total dimension of the transformed vector."""
        dim = len(self.num_features)
        for cat_f in self.cat_features:
            schema = self.spec.feature_schemas.get(cat_f)
            num_cats = len(schema.categories) if schema else 0
            dim += (num_cats + 1)
        return dim


class UniversalSDM(nn.Module):
    """
    Domain-Agnostic Small Decision Model.
    Supports classification (multi-class, binary) or continuous regression.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str = "classification",
        hidden_dim: int = 64,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            GatedResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        self.head = nn.Linear(hidden_dim, output_dim)
        if task_type == "classification":
            self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        else:
            self.temperature = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (logits, calibrated_outputs).
        For classification: (logits, probabilities).
        For regression: (raw_preds, scaled_preds).
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        raw = self.head(h)

        if self.task_type == "classification":
            temp = torch.clamp(self.temperature, min=0.2, max=4.0)
            scaled_logits = raw / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            return scaled_logits, probs
        else:
            # Regression task
            return raw, raw

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UniversalDecisionFactory:
    """
    Automated Factory Pipeline: Ingests client data across arbitrary domains,
    auto-constructs embeddings, trains in seconds on CPU, and exports sealed .sdmpkg packages.
    """

    def __init__(self, spec: UniversalDomainSpec):
        self.spec = spec
        self.encoder = UniversalFeatureEncoder(spec)
        self.choice_to_idx = {c: i for i, c in enumerate(spec.choices)}
        self.idx_to_choice = {i: c for i, c in enumerate(spec.choices)}

    @classmethod
    def create_for_domain(
        cls,
        domain_id: str,
        domain_name: str,
        numerical_features: List[str],
        categorical_features: List[str],
        choices: List[str],
        target_field: str = "target",
        task_type: str = "classification",
        sla_target_ms: float = 1.0,
    ) -> UniversalDecisionFactory:
        """Create factory instance with declarative schema."""
        spec = UniversalDomainSpec(
            domain_id=domain_id,
            domain_name=domain_name,
            task_type=task_type,
            target_field=target_field,
            choices=choices,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            sla_target_ms=sla_target_ms,
        )
        return cls(spec)

    def train(
        self,
        records: List[Dict[str, Any]],
        epochs: int = 15,
        batch_size: int = 32,
        lr: Optional[float] = None,
        tier: str = "compact",
        parameter_target: str = "small",
    ) -> Tuple[Union[UniversalSDM, EnterpriseSubBillionSDM], UniversalTrainingReport]:
        """
        Train a hyper-personalized Specialist SDM from scratch on client telemetry.
        Supports both 'compact' (<200k params) and 'sub_billion' / 'enterprise' (FT-Transformer, 1M to 100M+ params).
        """
        start_time = time.perf_counter()

        # 1. Fit feature encoder
        self.encoder.fit(records)
        input_dim = self.encoder.total_dim

        # 2. Extract X and y
        X_list = []
        y_list = []
        for r in records:
            tgt = r.get(self.spec.target_field)
            if self.spec.task_type == "classification":
                if tgt not in self.choice_to_idx:
                    continue
                y_list.append(self.choice_to_idx[tgt])
            else:
                try:
                    y_list.append(float(tgt))
                except (ValueError, TypeError):
                    continue
            X_list.append(self.encoder.transform_record(r))

        if not X_list:
            raise ValueError(f"No valid records found matching target field '{self.spec.target_field}'")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64 if self.spec.task_type == "classification" else np.float32)

        # Train / val split (80/20) with deterministic RNG
        torch.manual_seed(42)
        n = len(X)
        perm = np.random.RandomState(42).permutation(n)
        split = max(1, int(n * 0.8))
        if split >= n and n > 1:
            split = n - 1

        X_train = torch.tensor(X[perm[:split]])
        y_train = torch.tensor(y[perm[:split]])
        X_val = torch.tensor(X[perm[split:]]) if split < n else X_train
        y_val = torch.tensor(y[perm[split:]]) if split < n else y_train

        output_dim = len(self.spec.choices) if self.spec.task_type == "classification" else 1

        # 3. Instantiate model architecture based on tier
        if tier in ("sub_billion", "enterprise"):
            actual_lr = lr if lr is not None else 0.002
            model = EnterpriseSubBillionSDM.create_enterprise_tier(
                num_continuous_features=input_dim,
                num_classes=output_dim,
                parameter_target=parameter_target,
            )
        else:
            actual_lr = lr if lr is not None else 0.008
            model = UniversalSDM(
                input_dim=input_dim,
                output_dim=output_dim,
                task_type=self.spec.task_type,
                hidden_dim=64,
                num_blocks=2,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss() if self.spec.task_type == "classification" else nn.MSELoss()

        train_dataset = TensorDataset(X_train, y_train)
        act_batch = max(1, min(batch_size, len(X_train)))
        train_loader = DataLoader(train_dataset, batch_size=act_batch, shuffle=True)

        # 4. Fast Training Loop (< 15s)
        for ep in range(epochs):
            model.train()
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                logits, probs = model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # 5. Validation metrics
        model.eval()
        with torch.no_grad():
            logits_val, probs_val = model(X_val)
            if self.spec.task_type == "classification":
                preds = torch.argmax(probs_val, dim=-1)
                accuracy = (preds == y_val).float().mean().item()
                one_hot = torch.zeros_like(probs_val)
                one_hot.scatter_(1, y_val.unsqueeze(1), 1.0)
                brier_score = torch.mean((probs_val - one_hot) ** 2).item()
            else:
                accuracy = 1.0 - torch.mean(torch.abs(probs_val - y_val)).item()
                brier_score = 0.0

        duration = time.perf_counter() - start_time
        report = UniversalTrainingReport(
            domain_id=self.spec.domain_id,
            train_samples=len(X_train),
            val_samples=len(X_val),
            input_dim=input_dim,
            parameter_count=model.count_parameters(),
            epochs=epochs,
            validation_accuracy=round(accuracy, 4),
            brier_score=round(brier_score, 4),
            training_duration_sec=round(duration, 3),
            tier=tier,
        )

        return model, report

    def evaluate(self, model: Union[UniversalSDM, EnterpriseSubBillionSDM], record: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single raw record using trained UniversalSDM or EnterpriseSubBillionSDM with microsecond latency."""
        t0 = time.perf_counter()
        vec = self.encoder.transform_record(record)
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            logits, probs = model(x)
            probs_np = probs.squeeze(0).numpy()

        if self.spec.task_type == "classification":
            best_idx = int(np.argmax(probs_np))
            choice = self.idx_to_choice.get(best_idx, f"choice_{best_idx}")
            confidence = float(probs_np[best_idx])
            prob_dict = {
                self.idx_to_choice.get(i, f"choice_{i}"): float(round(probs_np[i], 4))
                for i in range(len(probs_np))
            }
        else:
            choice = str(probs_np[0])
            confidence = 1.0
            prob_dict = {"value": float(probs_np[0])}

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "domain_id": self.spec.domain_id,
            "choice": choice,
            "confidence": confidence,
            "probabilities": prob_dict,
            "latency_ms": round(latency_ms, 3),
        }

    def export_onnx(
        self,
        model: Union[UniversalSDM, EnterpriseSubBillionSDM],
        output_path: str,
        opset_version: int = 18,
    ) -> bytes:
        """Export trained model to single-file self-contained ONNX bytes."""
        import onnx
        model.eval()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if isinstance(model, EnterpriseSubBillionSDM):
            export_enterprise_sdm_to_onnx(
                model=model,
                output_path=output_path,
                input_dim=model.num_continuous_features,
                opset_version=opset_version,
                verify=True,
            )
            model_proto = onnx.load(output_path, load_external_data=True)
            return model_proto.SerializeToString()

        dummy_x = torch.randn(1, model.input_dim, dtype=torch.float32)

        torch.onnx.export(
            model,
            dummy_x,
            output_path,
            input_names=["input_features"],
            output_names=["logits", "calibrated_outputs"],
            dynamic_axes={"input_features": {0: "batch_size"}, "calibrated_outputs": {0: "batch_size"}},
            opset_version=opset_version,
            do_constant_folding=True,
        )

        # Ensure model has embedded weights (no external .data files)
        model_proto = onnx.load(output_path, load_external_data=True)
        onnx.save(model_proto, output_path, save_as_external_data=False)
        return model_proto.SerializeToString()

    def train_and_seal(
        self,
        records: List[Dict[str, Any]],
        license_key: str,
        client_id: str,
        output_package_path: str,
        epochs: int = 15,
        validity_days: int = 365,
        tier: str = "compact",
        parameter_target: str = "small",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SealedPackage, UniversalTrainingReport]:
        """
        Complete end-to-end factory pipeline:
        Trains Specialist SDM -> Exports embedded ONNX -> Seals with AES-256-GCM into .sdmpkg.
        """
        model, report = self.train(
            records,
            epochs=epochs,
            tier=tier,
            parameter_target=parameter_target,
        )

        # Temporary file for ONNX export
        tmp_onnx = output_package_path + ".tmp.onnx"
        raw_onnx_bytes = self.export_onnx(model, tmp_onnx)
        if os.path.exists(tmp_onnx):
            os.remove(tmp_onnx)
        if os.path.exists(tmp_onnx + ".data"):
            os.remove(tmp_onnx + ".data")

        # Seal into .sdmpkg
        spec_dict = self.spec.to_dict()
        spec_dict["tier"] = tier
        if metadata:
            spec_dict["metadata"] = metadata

        sealed = SealedPackage.seal(
            model_bytes_or_path=raw_onnx_bytes,
            license_key=license_key,
            client_id=client_id,
            domain=self.spec.domain_id,
            model_type=f"{self.spec.domain_id}_{tier}_specialist_sdm",
            spec_dict=spec_dict,
            choices=self.spec.choices,
            input_dim=getattr(model, "input_dim", getattr(model, "num_continuous_features", self.encoder.total_dim)),
            validity_days=validity_days,
            output_path=output_package_path,
        )

        report.sealed_package_id = sealed.manifest["package_id"]
        report.output_package_path = output_package_path
        return sealed, report
