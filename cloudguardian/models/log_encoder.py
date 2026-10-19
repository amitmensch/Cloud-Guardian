"""
Log Sequence & Semantic Feature Encoder for CloudGuardian.
Encodes the 29 template frequency channels, 4 inter-event time deltas,
and 4 window metadata channels into a shared latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSequenceEncoder(nn.Module):
    """
    Encodes discrete template frequency distributions and timing dynamics
    into a continuous representation of shape (Batch, TimeSteps, hidden_dim).
    """

    def __init__(
        self,
        in_features: int = 37,  # 29 templates + 4 timing stats + 4 metadata
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Multi-layer projection with LayerNorm and GELU
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 1D Causal Convolution for local temporal smoothing over log counts
        self.temporal_smooth = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=2,  # will chomp 2 to make it causal
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, log_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_features: Tensor of shape (Batch, TimeSteps, in_features)
        Returns:
            Tensor of shape (Batch, TimeSteps, hidden_dim)
        """
        # Step 1: Project feature dimensions
        projected = self.projection(log_features)  # (B, T, hidden_dim)

        # Step 2: Causal temporal smoothing across consecutive ticks
        x_transposed = projected.transpose(1, 2)  # (B, hidden_dim, T)
        smoothed = self.temporal_smooth(x_transposed)
        # Chomp right padding to keep strictly causal
        smoothed = smoothed[:, :, :-2].contiguous()
        smoothed = self.dropout(F.gelu(smoothed))

        # Residual connection
        out = x_transposed + smoothed
        return out.transpose(1, 2)  # (B, T, hidden_dim)
