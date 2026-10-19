"""
Masked Multi-Task Loss Function for CloudGuardian.
Combines:
1. Unsupervised Reconstruction Loss (L_MSE)
2. Metric Forecasting Loss (L_Forecast)
3. Supervised Failure-Risk Loss (L_Risk) - only applied on labelled windows
4. Time-to-Failure Hazard Loss (L_Hazard) - only applied on precursor windows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class MaskedMultiTaskLoss(nn.Module):
    """
    Computes weighted loss across the 4 multi-task heads:
    L = alpha * L_MSE + beta * L_Forecast + mask * (gamma * L_Risk + delta * L_Hazard)
    """

    def __init__(
        self,
        alpha: float = 1.0,   # Weight for Reconstruction
        beta: float = 0.5,    # Weight for Forecasting
        gamma: float = 1.0,   # Weight for Failure Risk
        delta: float = 0.1,   # Weight for TTF Hazard
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.huber_loss = nn.SmoothL1Loss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        inputs: torch.Tensor,
        future_metrics: Optional[torch.Tensor] = None,
        risk_labels: Optional[torch.Tensor] = None,
        ttf_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Output dictionary from CloudGuardianModel
            inputs: Original input tensor (Batch, TimeSteps, 42)
            future_metrics: Optional true future metric values (Batch, H, 5)
            risk_labels: Optional binary failure indicators (Batch, 1) where 1=Failure, 0=Normal
            ttf_targets: Optional ground-truth time-to-failure in minutes (Batch, 1)
        Returns:
            Dict containing total_loss and individual loss components.
        """
        # 1. Unsupervised Reconstruction Loss (always active)
        reconstruction = predictions["reconstruction"]
        loss_recon = self.mse_loss(reconstruction, inputs)
        total_loss = self.alpha * loss_recon

        loss_dict = {
            "loss_recon": loss_recon,
            "loss_forecast": torch.tensor(0.0, device=inputs.device),
            "loss_risk": torch.tensor(0.0, device=inputs.device),
            "loss_hazard": torch.tensor(0.0, device=inputs.device),
        }

        # 2. Self-Supervised Forecasting Loss (if future ground truth available)
        if future_metrics is not None:
            forecast_pred = predictions["forecast"]
            loss_forecast = self.mse_loss(forecast_pred, future_metrics)
            total_loss = total_loss + self.beta * loss_forecast
            loss_dict["loss_forecast"] = loss_forecast

        # 3. Supervised Failure-Risk Loss (masked: only active if labels provided)
        if risk_labels is not None:
            risk_pred = predictions["risk_prob"]
            loss_risk = self.bce_loss(risk_pred, risk_labels.float())
            total_loss = total_loss + self.gamma * loss_risk
            loss_dict["loss_risk"] = loss_risk

        # 4. Supervised Hazard TTF Loss (masked: only active for positive failure examples)
        if ttf_targets is not None and risk_labels is not None:
            # Mask to only compute TTF loss for samples that actually fail
            pos_mask = (risk_labels > 0.5).squeeze(-1)
            if pos_mask.sum() > 0:
                ttf_pred = predictions["ttf_minutes"][pos_mask]
                ttf_true = ttf_targets[pos_mask]
                loss_hazard = self.huber_loss(ttf_pred, ttf_true)
                total_loss = total_loss + self.delta * loss_hazard
                loss_dict["loss_hazard"] = loss_hazard

        loss_dict["total_loss"] = total_loss
        return loss_dict
