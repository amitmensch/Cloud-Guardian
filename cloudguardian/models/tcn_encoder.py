"""
Dilated Causal Temporal Convolutional Network (TCN) Metric Encoder.
Designed for CloudGuardian (Phase 2).

Key Properties:
1. Causal Convolutions: Enforces zero future information leakage (step t only depends on <= t).
2. Dilated Convolutions: Dilations [1, 2, 4, 8, 16] with kernel size 3 expand the receptive field.
3. Residual Connections: Skip connections stabilize gradient propagation across layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Chomp1d(nn.Module):
    """Trims trailing right-padding from standard 1D convolution to enforce causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Residual Dilated Causal Convolutional Block.
    Contains two layers of dilated causal convolutions with batch norm, ReLU, and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # First causal dilated conv layer
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal dilated conv layer
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if channel dimensions mismatch)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class DilatedTCNEncoder(nn.Module):
    """
    Dilated Temporal Convolutional Network Encoder for continuous system metrics.
    Input shape:  (Batch, in_features, TimeSteps)
    Output shape: (Batch, hidden_dim, TimeSteps)
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_dim: int = 64,
        dilations: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        layers = []
        num_levels = len(dilations)
        current_in = in_features

        for i in range(num_levels):
            d = dilations[i]
            layers.append(
                TemporalBlock(
                    in_channels=current_in,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            current_in = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, TimeSteps, in_features) or (Batch, in_features, TimeSteps)
        Returns:
            Tensor of shape (Batch, TimeSteps, hidden_dim)
        """
        # If input is (B, T, F), transpose to (B, F, T) for Conv1d
        if x.dim() == 3 and x.shape[1] != self.network[0].conv1.in_channels:
            x = x.transpose(1, 2)

        out = self.network(x)
        # Transpose back to (Batch, TimeSteps, hidden_dim)
        return out.transpose(1, 2)
