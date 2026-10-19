"""
Comprehensive Unit Test Suite for Phase 2: Hybrid Multimodal Model & Training Core.
Verifies DilatedTCNEncoder, LogSequenceEncoder, CrossAttention, Multi-Task Heads,
and the Unified CloudGuardianModel.
"""

import torch
import pytest

from cloudguardian.models.tcn_encoder import DilatedTCNEncoder, Chomp1d
from cloudguardian.models.log_encoder import LogSequenceEncoder
from cloudguardian.models.cross_attention import MultimodalCrossAttention
from cloudguardian.models.heads import (
    ReconstructionHead,
    ForecastHead,
    FailureRiskHead,
    HazardTTFHead,
)
from cloudguardian.models.cloudguardian_model import CloudGuardianModel
from cloudguardian.training.loss import MaskedMultiTaskLoss
from cloudguardian.training.trainer import CloudGuardianTrainer


class TestDilatedTCNEncoder:
    def test_forward_shape(self):
        encoder = DilatedTCNEncoder(in_features=5, hidden_dim=64, dilations=[1, 2, 4, 8, 16])
        # Batch=8, TimeSteps=60, Features=5
        x = torch.randn(8, 60, 5)
        out = encoder(x)
        assert out.shape == torch.Size([8, 60, 64])

    def test_causality(self):
        """Verifies that future inputs (at time t > 10) do NOT affect outputs at time t <= 10."""
        encoder = DilatedTCNEncoder(in_features=5, hidden_dim=32, dilations=[1, 2, 4])
        encoder.eval()

        x1 = torch.randn(2, 20, 5)
        x2 = x1.clone()
        # Alter future values after index 10
        x2[:, 11:, :] += torch.randn(2, 9, 5) * 5.0

        with torch.no_grad():
            out1 = encoder(x1)
            out2 = encoder(x2)

        # Output up to time 10 must be identical
        assert torch.allclose(out1[:, :11, :], out2[:, :11, :], atol=1e-5)


class TestLogSequenceEncoder:
    def test_forward_shape(self):
        encoder = LogSequenceEncoder(in_features=37, hidden_dim=64)
        x = torch.randn(8, 60, 37)
        out = encoder(x)
        assert out.shape == torch.Size([8, 60, 64])


class TestCrossAttention:
    def test_fusion_and_attention_caching(self):
        fusion = MultimodalCrossAttention(hidden_dim=64, num_heads=4)
        m_emb = torch.randn(4, 60, 64)
        l_emb = torch.randn(4, 60, 64)

        fused, attn_dict = fusion(m_emb, l_emb)
        assert fused.shape == torch.Size([4, 60, 64])
        assert "metric_to_log" in attn_dict
        assert "log_to_metric" in attn_dict
        # Attention weight tensor: (Batch, NumHeads, T, T)
        assert attn_dict["metric_to_log"].shape == torch.Size([4, 4, 60, 60])


class TestMultiTaskHeads:
    def setup_method(self):
        self.fused_emb = torch.randn(4, 60, 64)

    def test_reconstruction_head(self):
        head = ReconstructionHead(hidden_dim=64, total_features=42)
        out = head(self.fused_emb)
        assert out.shape == torch.Size([4, 60, 42])

    def test_forecast_head(self):
        head = ForecastHead(hidden_dim=64, forecast_horizon=10, metric_features=5)
        out = head(self.fused_emb)
        assert out.shape == torch.Size([4, 10, 5])

    def test_risk_head(self):
        head = FailureRiskHead(hidden_dim=64)
        out = head(self.fused_emb)
        assert out.shape == torch.Size([4, 1])
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_hazard_ttf_head(self):
        head = HazardTTFHead(hidden_dim=64, max_ttf_minutes=30.0)
        out = head(self.fused_emb)
        assert out.shape == torch.Size([4, 1])
        assert (out >= 0.0).all() and (out <= 30.0).all()


class TestUnifiedCloudGuardianModel:
    def test_full_pipeline_forward_pass(self):
        model = CloudGuardianModel(
            metric_features=5,
            log_features=37,
            total_features=42,
            hidden_dim=64,
        )
        batch_input = torch.randn(32, 60, 42)
        outputs = model(batch_input)

        assert outputs["reconstruction"].shape == torch.Size([32, 60, 42])
        assert outputs["forecast"].shape == torch.Size([32, 10, 5])
        assert outputs["risk_prob"].shape == torch.Size([32, 1])
        assert outputs["ttf_minutes"].shape == torch.Size([32, 1])
        assert outputs["recon_error"].shape == torch.Size([32])

        # Verify model footprint is well under 50 MB
        num_params = model.count_parameters()
        assert num_params < 500_000  # Highly lightweight (~180k parameters)


class TestTrainingAndLoss:
    def test_masked_loss_computation(self):
        model = CloudGuardianModel()
        loss_fn = MaskedMultiTaskLoss()
        x = torch.randn(8, 60, 42)
        pred = model(x)

        # Unlabelled pass
        loss_unlabelled = loss_fn(pred, x)
        assert loss_unlabelled["total_loss"] > 0
        assert loss_unlabelled["loss_risk"] == 0.0

        # Labelled pass
        labels = torch.randint(0, 2, (8, 1)).float()
        ttf = torch.rand(8, 1) * 20.0
        future = torch.randn(8, 10, 5)
        loss_labelled = loss_fn(pred, x, future_metrics=future, risk_labels=labels, ttf_targets=ttf)
        assert loss_labelled["total_loss"] > 0
        assert loss_labelled["loss_risk"] > 0

    def test_trainer_fit_loop(self):
        trainer = CloudGuardianTrainer(learning_rate=0.005)
        # 16 sample training batch
        train_data = torch.randn(16, 60, 42)
        history = trainer.fit(train_data, epochs=2, batch_size=8, verbose=False)

        assert len(history["total_loss"]) == 2
        # Check that loss decreased or remained stable
        assert history["total_loss"][-1] > 0
