"""
Unit Tests for SOTA Enterprise GPU Tabular Transformer Architecture & QAT Pipeline.
===================================================================================
Validates:
1. FakeQuantizeLinear STE quantization mechanics
2. UVMCachedEntityEmbedding high-cardinality entity embedding
3. FlashTabularAttentionBlock with Scaled Dot-Product Attention (SDPA)
4. EnterpriseGPUTabularTransformer scaling up to flagship tier (113M+ parameters)
5. GPUTrainingPipeline execution and Brier score temperature calibration
6. Single-file quantized ONNX export and in-memory evaluation
"""

import os
import sys
import tempfile
import time
import pytest
import numpy as np
import torch
import torch.nn as nn

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.gpu_enterprise_sdm import (
    FakeQuantizeLinear,
    UVMCachedEntityEmbedding,
    FlashTabularAttentionBlock,
    EnterpriseGPUTabularTransformer,
    GPUTrainingConfig,
    GPUTrainingPipeline,
    export_gpu_sdm_to_quantized_onnx,
)


class TestFakeQuantizeLinear:
    def test_quantization_toggle_and_forward(self):
        linear = FakeQuantizeLinear(in_features=16, out_features=32)
        x = torch.randn(4, 16)

        # Standard unquantized forward
        out_unquant = linear(x)
        assert out_unquant.shape == (4, 32)

        # Enable INT8 QAT simulation
        linear.enable_quantization(True)
        assert linear.is_quantizing is True
        out_quant = linear(x)
        assert out_quant.shape == (4, 32)

        # Check STE backward gradients flow
        loss = out_quant.sum()
        loss.backward()
        assert linear.weight.grad is not None
        assert torch.isfinite(linear.weight.grad).all()


class TestUVMCachedEntityEmbedding:
    def test_uvm_embedding_capacity_and_lookup(self):
        num_bins = 50_000
        dim = 32
        uvm = UVMCachedEntityEmbedding(
            num_embeddings=num_bins,
            embedding_dim=dim,
            cache_capacity=1_000,
            name="test_bin_table",
        )

        assert uvm.num_embeddings == 50_000
        assert uvm.cache_capacity == 1_000
        assert uvm.hot_mask[:1_000].all()
        assert not uvm.hot_mask[1_000:].any()

        # Query mixture of hot (index < 1000) and cold (index >= 1000) entities
        indices = torch.tensor([5, 42, 999, 1001, 25000, 49999], dtype=torch.long)
        embeddings = uvm(indices)
        assert embeddings.shape == (6, dim)
        assert torch.isfinite(embeddings).all()


class TestFlashTabularAttentionBlock:
    def test_sdpa_attention_forward(self):
        block = FlashTabularAttentionBlock(
            hidden_dim=64,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()


class TestEnterpriseGPUTabularTransformer:
    def test_parameter_scaling_and_flagship_capacity(self):
        # Dev scale (~1M params)
        m_dev = EnterpriseGPUTabularTransformer.create_gpu_flagship_tier(
            num_continuous_features=10,
            num_classes=3,
            scale="dev",
        )
        assert 500_000 < m_dev.count_parameters() < 2_000_000

        # Production scale (~10M params)
        m_prod = EnterpriseGPUTabularTransformer.create_gpu_flagship_tier(
            num_continuous_features=10,
            num_classes=3,
            scale="production",
        )
        assert 3_000_000 < m_prod.count_parameters() < 15_000_000

        # Flagship scale (113M+ parameters)
        m_flag = EnterpriseGPUTabularTransformer.create_gpu_flagship_tier(
            num_continuous_features=10,
            num_classes=3,
            scale="flagship",
        )
        assert m_flag.count_parameters() > 100_000_000

    def test_forward_with_continuous_and_high_cardinality_uvm(self):
        cat_dims = {"bin_ranges": 20_000, "mcc_codes": 500}
        model = EnterpriseGPUTabularTransformer(
            num_continuous_features=6,
            num_classes=3,
            categorical_dims=cat_dims,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            enable_uvm=True,
        )

        batch_size = 4
        cont_x = torch.randn(batch_size, 6)
        cat_x = torch.tensor([[100, 12], [15000, 250], [42, 5], [19999, 499]], dtype=torch.long)

        logits, probs = model(cont_x, cat_x)
        assert logits.shape == (batch_size, 3)
        assert probs.shape == (batch_size, 3)

        # Probabilities sum to 1.0 across classes
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5, rtol=1e-5)


class TestGPUTrainingPipelineAndCalibration:
    def test_training_epoch_and_calibration(self):
        model = EnterpriseGPUTabularTransformer(
            num_continuous_features=4,
            num_classes=2,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )

        config = GPUTrainingConfig(precision="fp32", target_device="cpu")
        pipeline = GPUTrainingPipeline(model=model, config=config)

        # Synthetic data
        X = torch.randn(64, 4)
        y = torch.randint(0, 2, (64,), dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss = pipeline.train_epoch(loader, optimizer, criterion, use_bf16=False)
        assert loss > 0.0

        # Calibrate temperature
        best_t = pipeline.calibrate_temperature(loader, num_classes=2)
        assert 0.2 <= best_t <= 4.0
        assert model.temperature.item() == pytest.approx(best_t, rel=1e-3)


class TestGPUQuantizedONNXExport:
    def test_export_and_in_memory_onnx_inference(self):
        model = EnterpriseGPUTabularTransformer(
            num_continuous_features=5,
            num_classes=3,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "gpu_sdm_quant.onnx")
            out_file = export_gpu_sdm_to_quantized_onnx(
                model=model,
                output_path=onnx_path,
                input_dim=5,
            )

            assert os.path.exists(out_file)
            assert not os.path.exists(out_file + ".data")

            # In-memory ONNX Runtime evaluation
            import onnxruntime as ort
            sess = ort.InferenceSession(out_file, providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            dummy_input = np.ones((1, 5), dtype=np.float32)

            t0 = time.perf_counter()
            outputs = sess.run(None, {input_name: dummy_input})
            latency_ms = (time.perf_counter() - t0) * 1000.0

            assert len(outputs) == 2
            assert outputs[1].shape == (1, 3)
            # In-memory execution < 10ms SLA
            assert latency_ms < 10.0
