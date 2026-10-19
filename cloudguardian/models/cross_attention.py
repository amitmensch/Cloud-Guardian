"""
Multimodal Cross-Attention Fusion Layer for CloudGuardian.
Fuses Metric Latent Embeddings and Log Latent Embeddings.
Retains cross-attention weights for Phase 4 explainability (SHAP / Root-Cause Attribution).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class MultimodalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention Layer.
    Allows metric channels to attend to log template occurrences (and vice-versa).
    Stores cross-attention weight matrices for interpretability.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multihead attention: Metric queries Log
        self.metric_to_log_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Multihead attention: Log queries Metric
        self.log_to_metric_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms and projection
        self.norm_metric = nn.LayerNorm(hidden_dim)
        self.norm_log = nn.LayerNorm(hidden_dim)
        self.fuse_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Storage for downstream explainability (Phase 4)
        self.last_metric_to_log_weights: Optional[torch.Tensor] = None
        self.last_log_to_metric_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        metric_emb: torch.Tensor,
        log_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            metric_emb: (Batch, TimeSteps, hidden_dim)
            log_emb:    (Batch, TimeSteps, hidden_dim)
        Returns:
            fused_representation: (Batch, TimeSteps, hidden_dim)
            attention_dict: Dict containing saved attention weight matrices
        """
        # Metric attends to Log
        m2l_out, m2l_weights = self.metric_to_log_attn(
            query=metric_emb,
            key=log_emb,
            value=log_emb,
            need_weights=True,
            average_attn_weights=False,
        )
        norm_m = self.norm_metric(metric_emb + m2l_out)

        # Log attends to Metric
        l2m_out, l2m_weights = self.log_to_metric_attn(
            query=log_emb,
            key=metric_emb,
            value=metric_emb,
            need_weights=True,
            average_attn_weights=False,
        )
        norm_l = self.norm_log(log_emb + l2m_out)

        # Save weights for explainability
        self.last_metric_to_log_weights = m2l_weights.detach()
        self.last_log_to_metric_weights = l2m_weights.detach()

        # Concatenate and project back to hidden_dim
        combined = torch.cat([norm_m, norm_l], dim=-1)  # (B, T, 2 * hidden_dim)
        fused = self.fuse_proj(combined)  # (B, T, hidden_dim)

        attn_dict = {
            "metric_to_log": self.last_metric_to_log_weights,
            "log_to_metric": self.last_log_to_metric_weights,
        }
        return fused, attn_dict
