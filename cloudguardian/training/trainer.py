"""
Training and Optimization Engine for CloudGuardian.
Manages the unsupervised baseline training loop and multi-task optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Optional
import os

from ..models.cloudguardian_model import CloudGuardianModel
from .loss import MaskedMultiTaskLoss


class CloudGuardianTrainer:
    """
    Manages model training, evaluation, and checkpointing for CloudGuardian.
    """

    def __init__(
        self,
        model: Optional[CloudGuardianModel] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        delta: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model if model is not None else CloudGuardianModel()
        self.model.to(self.device)

        self.criterion = MaskedMultiTaskLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "loss_recon": [],
            "loss_forecast": [],
            "loss_risk": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Trains the model for one full epoch across a DataLoader."""
        self.model.train()
        epoch_losses = {"total_loss": 0.0, "loss_recon": 0.0, "loss_forecast": 0.0, "loss_risk": 0.0}
        num_batches = len(dataloader)

        for batch in dataloader:
            # batch can be (x,) or (x, future_metrics, labels, ttf)
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
                future_metrics = batch[1].to(self.device) if len(batch) > 1 else None
                labels = batch[2].to(self.device) if len(batch) > 2 else None
                ttf = batch[3].to(self.device) if len(batch) > 3 else None
            else:
                x = batch.to(self.device)
                future_metrics, labels, ttf = None, None, None

            self.optimizer.zero_grad()
            predictions = self.model(x)

            loss_dict = self.criterion(
                predictions=predictions,
                inputs=x,
                future_metrics=future_metrics,
                risk_labels=labels,
                ttf_targets=ttf,
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k].item()

        # Compute average losses
        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        for k in self.history:
            self.history[k].append(avg_losses[k])

        return avg_losses

    def fit(
        self,
        train_tensor: torch.Tensor,
        epochs: int = 5,
        batch_size: int = 32,
        future_metrics: Optional[torch.Tensor] = None,
        risk_labels: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Convenience training loop over in-memory PyTorch tensors.
        """
        tensors = [train_tensor]
        if future_metrics is not None:
            tensors.append(future_metrics)
        if risk_labels is not None:
            tensors.append(risk_labels)

        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if verbose:
            print(f"Training CloudGuardian on {self.device} ({len(train_tensor)} samples, {epochs} epochs)...")

        for epoch in range(1, epochs + 1):
            avg_losses = self.train_epoch(loader)
            if verbose:
                print(
                    f"  Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Total Loss: {avg_losses['total_loss']:.4f} | "
                    f"Recon Loss: {avg_losses['loss_recon']:.4f}"
                )

        return self.history

    def evaluate_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes reconstruction errors E_i for a batch of samples.
        High values indicate anomalous operational deviation.
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            out = self.model(x)
            return out["recon_error"].cpu()

    def save_checkpoint(self, filepath: str):
        """Saves model weights and optimizer state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        """Loads model weights from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
