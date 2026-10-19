"""
CloudGuardian Unified Multimodal Deep Learning Architecture.
Combines:
1. Dilated Causal TCN Metric Encoder (5 features)
2. Log Sequence & Semantic Encoder (37 features)
3. Bidirectional Multimodal Cross-Attention Fusion
4. Multi-Task Heads: Reconstruction (Autoencoder), Forecasting, Failure Risk, and Hazard TTF.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from .tcn_encoder import DilatedTCNEncoder
from .log_encoder import LogSequenceEncoder
from .cross_attention import MultimodalCrossAttention
from .heads import ReconstructionHead, ForecastHead, FailureRiskHead, HazardTTFHead


class CloudGuardianModel(nn.Module):
    """
    End-to-End Multimodal Deep Learning Framework for Cloud Infrastructure
    Predictive Maintenance and Anomaly Detection.
    """

    def __init__(
        self,
        metric_features: int = 5,
        log_features: int = 37,
        total_features: int = 42,
        hidden_dim: int = 64,
        num_heads: int = 4,
        forecast_horizon: int = 10,
        max_ttf_minutes: float = 30.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.metric_features = metric_features
        self.log_features = log_features
        self.total_features = total_features
        self.hidden_dim = hidden_dim

        # 1. Encoders
        self.metric_encoder = DilatedTCNEncoder(
            in_features=metric_features,
            hidden_dim=hidden_dim,
            dilations=[1, 2, 4, 8, 16],
            kernel_size=3,
            dropout=dropout,
        )

        self.log_encoder = LogSequenceEncoder(
            in_features=log_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # 2. Cross-Attention Multimodal Fusion
        self.fusion = MultimodalCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout / 2.0,
        )

        # 3. Multi-Task Prediction Heads
        self.recon_head = ReconstructionHead(
            hidden_dim=hidden_dim,
            total_features=total_features,
            dropout=dropout / 2.0,
        )

        self.forecast_head = ForecastHead(
            hidden_dim=hidden_dim,
            forecast_horizon=forecast_horizon,
            metric_features=metric_features,
            dropout=dropout / 2.0,
        )

        self.risk_head = FailureRiskHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.ttf_head = HazardTTFHead(
            hidden_dim=hidden_dim,
            max_ttf_minutes=max_ttf_minutes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            x: 3D Tensor of shape (Batch, TimeSteps, total_features)
               e.g. torch.Size([32, 60, 42])
        Returns:
            Dictionary containing:
                - reconstruction: (Batch, 60, 42)
                - forecast:       (Batch, 10, 5)
                - risk_prob:      (Batch, 1)
                - ttf_minutes:    (Batch, 1)
                - recon_error:    (Batch,) mean squared deviation per sample
                - attention:      Dict of cross-attention weights
        """
        assert x.shape[-1] == self.total_features, (
            f"Expected {self.total_features} features, got {x.shape[-1]}"
        )

        # Slice modalities
        metrics_part = x[:, :, : self.metric_features]  # (B, T, 5)
        logs_part = x[:, :, self.metric_features :]     # (B, T, 37)

        # Encode both streams
        metric_emb = self.metric_encoder(metrics_part)   # (B, T, hidden_dim)
        log_emb = self.log_encoder(logs_part)            # (B, T, hidden_dim)

        # Multimodal Cross-Attention Fusion
        fused_emb, attn_dict = self.fusion(metric_emb, log_emb)  # (B, T, hidden_dim)

        # Forward through Multi-Task Heads
        reconstructed = self.recon_head(fused_emb)
        forecast = self.forecast_head(fused_emb)
        risk_prob = self.risk_head(fused_emb)
        ttf_minutes = self.ttf_head(fused_emb)

        # Compute sample-wise reconstruction error: E_i = ||x_i - x_hat_i||^2
        recon_error = torch.mean((x - reconstructed) ** 2, dim=[1, 2])

        return {
            "reconstruction": reconstructed,
            "forecast": forecast,
            "risk_prob": risk_prob,
            "ttf_minutes": ttf_minutes,
            "recon_error": recon_error,
            "fused_embedding": fused_emb,
            "attention": attn_dict,
        }

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
