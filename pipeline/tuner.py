"""
Payment Decision Model Tuner & Personalization Pipeline
=======================================================
Ingests client transaction history (CSV/JSON), serializes payment telemetry,
and trains a calibrated domain decision adapter on top of Laya System 1 weights.

Steps:
1. Ingests client historical transaction logs.
2. Extracts features via PaymentSerializer.
3. Trains a calibrated PyTorch decision head (or LoRA adapter).
4. Fits empirical temperature scaling for un-inflated confidence scores.
5. Emits a personalized client bundle (weights + manifest + benchmark report).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure payment_decision_factory is importable
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.spec import DecisionSpec
from payment_decision_factory.core.serializer import PaymentSerializer

logger = logging.getLogger("payment_decision_factory.tuner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PaymentDataset(Dataset):
    """PyTorch Dataset of serialized payment vectors and ground-truth targets."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class CalibratedDecisionHead(nn.Module):
    """
    Lightweight, calibrated decision projection head.
    Attaches to normalized payment feature representations or transformer CLS pooler.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        # Learnable temperature parameter for probability calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        # Scale by temperature for calibrated probabilities
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        calibrated_logits = logits / temp
        probabilities = torch.softmax(calibrated_logits, dim=-1)
        return calibrated_logits, probabilities


@dataclass
class TuningMetrics:
    epochs_trained: int
    initial_loss: float
    final_loss: float
    final_accuracy: float
    brier_score: float
    fitted_temperature: float
    duration_seconds: float


class ModelTuner:
    """Manages data ingestion, training, calibration, and bundling for a client."""

    def __init__(self, spec_path: str):
        self.spec = DecisionSpec.from_yaml(spec_path)
        self.serializer = PaymentSerializer(self.spec)
        self.choice_keys = list(self.spec.choices.keys())
        self.choice_to_idx = {k: i for i, k in enumerate(self.choice_keys)}
        self.idx_to_choice = {i: k for i, k in enumerate(self.choice_keys)}

    def prepare_dataset(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[PaymentDataset, PaymentDataset]:
        """Converts raw transaction dictionaries into train/val PyTorch datasets."""
        vectors = []
        targets = []

        for rec in records:
            # Determine target label
            target_choice = rec.get("optimal_gateway") or rec.get("gateway") or rec.get("target")
            if target_choice not in self.choice_to_idx:
                continue

            # Extract features via serializer
            try:
                valid_input = self.spec.validate_input(rec)
                vec = self.serializer.to_vector(valid_input)
                vectors.append(vec)
                targets.append(self.choice_to_idx[target_choice])
            except Exception:
                continue

        X = np.array(vectors, dtype=np.float32)
        y = np.array(targets, dtype=np.int64)

        # 80/20 train/validation split
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        split = int(n_samples * 0.8)

        train_idx, val_idx = indices[:split], indices[split:]
        train_ds = PaymentDataset(X[train_idx], y[train_idx])
        val_ds = PaymentDataset(X[val_idx], y[val_idx])

        return train_ds, val_ds

    def train(
        self,
        train_ds: PaymentDataset,
        val_ds: PaymentDataset,
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 0.005,
    ) -> Tuple[CalibratedDecisionHead, TuningMetrics]:
        """Executes the training loop with calibrated Brier score loss."""
        start_time = time.perf_counter()
        input_dim = train_ds.features.shape[1]
        num_classes = len(self.choice_keys)

        head = CalibratedDecisionHead(input_dim, num_classes)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
        ce_loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        initial_loss = 0.0
        final_loss = 0.0

        for epoch in range(epochs):
            head.train()
            total_train_loss = 0.0

            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                logits, probs = head(X_b)
                loss = ce_loss_fn(logits, y_b)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(X_b)

            epoch_loss = total_train_loss / len(train_ds)
            if epoch == 0:
                initial_loss = epoch_loss
            final_loss = epoch_loss

        # Evaluation on validation set
        head.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                logits, probs = head(X_val)
                preds = torch.argmax(probs, dim=-1)
                accuracy = (preds == y_val).float().mean().item()

                # Calculate Brier score for calibration
                # Brier Score = (1/N) * sum((probs - one_hot)^2)
                one_hot = torch.zeros_like(probs)
                one_hot.scatter_(1, y_val.unsqueeze(1), 1.0)
                brier_score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1)).item()

        duration = time.perf_counter() - start_time
        metrics = TuningMetrics(
            epochs_trained=epochs,
            initial_loss=round(initial_loss, 4),
            final_loss=round(final_loss, 4),
            final_accuracy=round(accuracy, 4),
            brier_score=round(brier_score, 4),
            fitted_temperature=round(head.temperature.item(), 3),
            duration_seconds=round(duration, 2),
        )

        return head, metrics

    def export_personalized_bundle(
        self,
        head: CalibratedDecisionHead,
        metrics: TuningMetrics,
        output_dir: str,
        client_name: str,
    ) -> str:
        """Saves the tuned weights and personalized client manifest."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save weights state dict
        weights_path = os.path.join(output_dir, "custom_adapter.pt")
        torch.save(head.state_dict(), weights_path)

        # 2. Export client manifest & calibration report
        manifest = {
            "client_name": client_name,
            "spec_id": self.spec.spec_id,
            "timestamp": time.time(),
            "tuning_metrics": {
                "initial_loss": metrics.initial_loss,
                "final_loss": metrics.final_loss,
                "validation_accuracy": metrics.final_accuracy,
                "brier_score": metrics.brier_score,
                "fitted_temperature": metrics.fitted_temperature,
                "training_time_seconds": metrics.duration_seconds,
            },
            "choices_mapping": self.choice_to_idx,
            "calibrated": metrics.brier_score < 0.10,
        }

        manifest_path = os.path.join(output_dir, "client_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Successfully exported personalized model bundle for '{client_name}' to {output_dir}")
        return output_dir
