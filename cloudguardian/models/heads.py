"""
Multi-Task Prediction Heads for CloudGuardian.
Implements the 4 core heads:
1. Reconstruction Head (Autoencoder Baseline: MSE Loss)
2. Forecasting Head (Future Multi-Step Metric Trajectory)
3. Failure-Risk Head (Calibrated Binary Probability: BCE Loss)
4. Hazard / Time-to-Failure (TTF) Head (Survival Regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ReconstructionHead(nn.Module):
    """
    Reconstructs the original input tensor (Batch, TimeSteps, total_features).
    Used for unsupervised anomaly detection: normal inputs reconstruct well;
    anomalous inputs exhibit large reconstruction errors.
    """

    def __init__(self, hidden_dim: int = 64, total_features: int = 42, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, total_features),
        )

    def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_emb: (Batch, TimeSteps, hidden_dim)
        Returns:
            reconstructed: (Batch, TimeSteps, total_features)
        """
        return self.decoder(fused_emb)


class ForecastHead(nn.Module):
    """
    Predicts the future values of the 5 system metrics for the next H ticks.
    Default H = 10 ticks (5 minutes into the future at 30s resolution).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        forecast_horizon: int = 10,
        metric_features: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.metric_features = metric_features

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon * metric_features),
        )

    def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
        """
        Takes the latest time step representation (fused_emb[:, -1, :])
        and outputs future metric trajectory of shape (Batch, forecast_horizon, metric_features).
        """
        latest_rep = fused_emb[:, -1, :]  # (Batch, hidden_dim)
        flat_pred = self.net(latest_rep)   # (Batch, H * 5)
        batch_size = flat_pred.shape[0]
        return flat_pred.view(batch_size, self.forecast_horizon, self.metric_features)


class FailureRiskHead(nn.Module):
    """
    Predicts the scalar probability that the system will experience a failure
    within the next 30 minutes: P(Failure) in [0, 1].
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_emb: (Batch, TimeSteps, hidden_dim)
        Returns:
            risk_prob: (Batch, 1) probability values in [0, 1]
        """
        # Pool across time dimension: (Batch, hidden_dim, TimeSteps) -> (Batch, hidden_dim, 1)
        pooled = self.pool(fused_emb.transpose(1, 2)).squeeze(-1)  # (Batch, hidden_dim)
        logits = self.classifier(pooled)
        return torch.sigmoid(logits)


class HazardTTFHead(nn.Module):
    """
    Predicts the expected Time-to-Failure (TTF) in minutes.
    If no failure is imminent, outputs max horizon (e.g. 30.0 minutes).
    """

    def __init__(self, hidden_dim: int = 64, max_ttf_minutes: float = 30.0, dropout: float = 0.2):
        super().__init__()
        self.max_ttf_minutes = max_ttf_minutes
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.ReLU(),  # TTF cannot be negative
        )

    def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_emb: (Batch, TimeSteps, hidden_dim)
        Returns:
            ttf_minutes: (Batch, 1) non-negative scalar
        """
        pooled = self.pool(fused_emb.transpose(1, 2)).squeeze(-1)
        raw_ttf = self.regressor(pooled)
        # Clamp to max_ttf_minutes
        return torch.clamp(raw_ttf, min=0.0, max=self.max_ttf_minutes)
