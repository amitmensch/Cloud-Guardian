"""
Conformal Prediction Engine for CloudGuardian.
Grounds anomaly detection in rigorous, distribution-free statistical guarantees
(Vovk et al., Algorithmic Learning in a Random World, 2005).

Converts raw reconstruction errors into well-calibrated empirical p-values.
"""

import numpy as np
import torch
from typing import Union, List, Tuple, Optional, Dict
import json
import os


class ConformalCalibrator:
    """
    Implements inductive conformal prediction for anomaly detection.
    Computes distribution-free non-conformity p-values based on a held-out
    calibration set of normal operational sequences.
    """

    def __init__(self, significance_level: float = 0.01):
        self.significance_level = significance_level
        self.calibration_scores: Optional[np.ndarray] = None
        self.adaptive_threshold: Optional[float] = None

    def fit(self, calibration_scores: Union[np.ndarray, torch.Tensor, List[float]]) -> float:
        """
        Calibrates the non-conformity distribution using a held-out set
        of reconstruction errors from normal operational sequences.
        """
        if isinstance(calibration_scores, torch.Tensor):
            scores = calibration_scores.detach().cpu().numpy()
        else:
            scores = np.asarray(calibration_scores, dtype=np.float64)

        if len(scores) == 0:
            raise ValueError("Calibration score set cannot be empty.")

        # Sort scores in ascending order for fast empirical CDF evaluation
        self.calibration_scores = np.sort(scores.flatten())
        n = len(self.calibration_scores)

        # Compute empirical (1 - significance_level) quantile threshold tau
        # With finite-sample correction
        index = int(np.ceil((1.0 - self.significance_level) * (n + 1))) - 1
        index = min(max(index, 0), n - 1)
        self.adaptive_threshold = float(self.calibration_scores[index])

        return self.adaptive_threshold

    def compute_p_value(self, test_score: float) -> float:
        """
        Calculates the conformal empirical p-value for a given test non-conformity score:
        p(x) = (1 + sum(alpha_i >= alpha(x))) / (n + 1)
        """
        if self.calibration_scores is None:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        n = len(self.calibration_scores)
        # Count how many calibration scores are greater than or equal to test_score
        greater_or_equal = np.sum(self.calibration_scores >= test_score)
        p_val = (greater_or_equal + 1.0) / (n + 1.0)
        return float(min(1.0, max(0.0, p_val)))

    def is_anomaly(self, test_score: float, custom_alpha: Optional[float] = None) -> Tuple[bool, float]:
        """
        Evaluates whether a test score is anomalous under the chosen significance level.
        Returns:
            (is_anom, p_value)
        """
        alpha = custom_alpha if custom_alpha is not None else self.significance_level
        p_val = self.compute_p_value(test_score)
        return (p_val < alpha), p_val

    def batch_evaluate(
        self, test_scores: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized evaluation of an array of test reconstruction errors.
        Returns:
            (anomaly_flags: bool array, p_values: float array)
        """
        if isinstance(test_scores, torch.Tensor):
            scores = test_scores.detach().cpu().numpy().flatten()
        else:
            scores = np.asarray(test_scores).flatten()

        p_vals = np.array([self.compute_p_value(s) for s in scores])
        flags = p_vals < self.significance_level
        return flags, p_vals

    def save(self, filepath: str):
        """Persists the calibration scores and parameters."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "significance_level": self.significance_level,
            "adaptive_threshold": self.adaptive_threshold,
            "calibration_scores": self.calibration_scores.tolist() if self.calibration_scores is not None else [],
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Loads calibrator parameters from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.significance_level = data["significance_level"]
        self.adaptive_threshold = data["adaptive_threshold"]
        self.calibration_scores = np.array(data["calibration_scores"], dtype=np.float64)
