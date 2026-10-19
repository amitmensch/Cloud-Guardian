"""
CloudGuardian Phase 5: 8-Rung Baseline Implementations
Module: experiments.baselines

Implements the 8 comparative baseline monitoring paradigms:
1. Static Threshold (e.g., CPU > 90%, Latency > 200ms)
2. Keyword / Severity Pattern Matcher (ERROR, EXCEPTION, FATAL)
3. STIDE (Sequence Time-Delay Embedding / n-gram mismatch on log templates)
4. Isolation Forest (Classical unsupervised tabular outlier detection)
5. ARIMA / Auto-Regressive Residual Forecaster
6. LSTM Autoencoder (Pure LSTM encoder-decoder)
7. DeepLog (Log-only next-template GRU predictor)
8. USAD (UnSupervised Anomaly Detection: Metric-only dual autoencoder)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class StaticThresholdDetector:
    """Rung 1: Industry-standard static threshold alert (e.g. Prometheus rules)."""

    def __init__(self, cpu_thresh: float = 0.90, latency_thresh: float = 0.50):
        self.cpu_thresh = cpu_thresh
        self.latency_thresh = latency_thresh

    def score_window(self, x: np.ndarray) -> Tuple[float, bool]:
        """
        x: (60, 42) window
        Returns (anomaly_score, is_alert)
        """
        # Channel 0: CPU, Channel 4: DB Latency
        max_cpu = np.max(x[:, 0])
        max_lat = np.max(x[:, 4])
        score = max(max_cpu / self.cpu_thresh, max_lat / self.latency_thresh)
        is_alert = (max_cpu > self.cpu_thresh) or (max_lat > self.latency_thresh)
        return float(score), bool(is_alert)


class KeywordPatternMatcher:
    """Rung 2: Keyword / Log Severity Pattern Matcher (Regex on ERROR/FATAL)."""

    def __init__(self, error_rate_threshold: float = 0.15):
        self.error_rate_threshold = error_rate_threshold

    def score_window(self, x: np.ndarray) -> Tuple[float, bool]:
        """
        Channel 39: error_level_log_ratio
        """
        error_ratio = np.max(x[:, 39]) if x.shape[1] > 39 else 0.0
        is_alert = error_ratio > self.error_rate_threshold
        return float(error_ratio), bool(is_alert)


class STIDETemplateDetector:
    """Rung 3: Sequence Time-Delay Embedding (STIDE) on log templates."""

    def __init__(self, window_size: int = 5, mismatch_threshold: float = 0.30):
        self.window_size = window_size
        self.mismatch_threshold = mismatch_threshold
        self.normal_ngrams = set()

    def fit(self, normal_sequences: List[np.ndarray]):
        for seq in normal_sequences:
            # seq: (60, 42) -> extract dominant log template per tick
            dominant_templates = np.argmax(seq[:, 5:34], axis=1)
            for i in range(len(dominant_templates) - self.window_size + 1):
                ngram = tuple(dominant_templates[i : i + self.window_size])
                self.normal_ngrams.add(ngram)

    def score_window(self, x: np.ndarray) -> Tuple[float, bool]:
        dominant_templates = np.argmax(x[:, 5:34], axis=1)
        total_ngrams = len(dominant_templates) - self.window_size + 1
        if total_ngrams <= 0:
            return 0.0, False

        mismatches = 0
        for i in range(total_ngrams):
            ngram = tuple(dominant_templates[i : i + self.window_size])
            if ngram not in self.normal_ngrams:
                mismatches += 1

        mismatch_rate = mismatches / total_ngrams
        return float(mismatch_rate), bool(mismatch_rate > self.mismatch_threshold)


class TabularIsolationForest:
    """Rung 4: Classical Unsupervised Isolation Forest on window summary statistics."""

    def __init__(self, contamination: float = 0.02, n_trees: int = 50, max_depth: int = 8):
        self.contamination = contamination
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees: List[Dict[str, Any]] = []
        self.threshold = 0.5
        self.is_fitted = False

    def _extract_features(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.mean(x, axis=0), np.max(x, axis=0), np.std(x, axis=0)])

    def fit(self, normal_windows: List[np.ndarray]):
        X = np.array([self._extract_features(w) for w in normal_windows])
        self.trees = []
        rng = np.random.RandomState(42)

        for _ in range(self.n_trees):
            tree = self._build_tree(X, 0, self.max_depth, rng)
            self.trees.append(tree)

        scores = [self._score_sample(x) for x in X]
        self.threshold = float(np.percentile(scores, 100.0 * (1.0 - self.contamination)))
        self.is_fitted = True

    def _build_tree(self, X: np.ndarray, depth: int, max_depth: int, rng) -> Dict[str, Any]:
        if depth >= max_depth or len(X) <= 1:
            return {"type": "leaf", "size": len(X), "depth": depth}

        feat_idx = rng.randint(0, X.shape[1])
        min_v, max_v = np.min(X[:, feat_idx]), np.max(X[:, feat_idx])
        if min_v == max_v:
            return {"type": "leaf", "size": len(X), "depth": depth}

        split_v = rng.uniform(min_v, max_v)
        left = X[X[:, feat_idx] < split_v]
        right = X[X[:, feat_idx] >= split_v]

        return {
            "type": "node",
            "feat": feat_idx,
            "split": split_v,
            "left": self._build_tree(left, depth + 1, max_depth, rng),
            "right": self._build_tree(right, depth + 1, max_depth, rng),
        }

    def _path_length(self, x: np.ndarray, node: Dict[str, Any], curr_depth: int) -> float:
        if node["type"] == "leaf":
            return curr_depth + (np.log(max(node["size"], 1) + 1) if node["size"] > 1 else 0.0)
        if x[node["feat"]] < node["split"]:
            return self._path_length(x, node["left"], curr_depth + 1)
        return self._path_length(x, node["right"], curr_depth + 1)

    def _score_sample(self, x: np.ndarray) -> float:
        if not self.trees:
            return 0.0
        avg_depth = np.mean([self._path_length(x, t, 0) for t in self.trees])
        return float(2.0 ** (-avg_depth / max(self.max_depth, 1)))

    def score_window(self, x: np.ndarray) -> Tuple[float, bool]:
        feat = self._extract_features(x)
        score = self._score_sample(feat)
        is_alert = score > self.threshold
        return score, bool(is_alert)


class ARIMAForecaster:
    """Rung 5: Auto-Regressive Moving Average Time-Series Residual Model."""

    def __init__(self, p: int = 3, threshold: float = 0.40):
        self.p = p
        self.threshold = threshold
        self.weights = np.ones(p) / p

    def score_window(self, x: np.ndarray) -> Tuple[float, bool]:
        # Evaluates metric channels (0..5)
        metrics = x[:, :5]
        residuals = []
        for ch in range(5):
            series = metrics[:, ch]
            for t in range(self.p, len(series)):
                pred = np.dot(series[t - self.p : t], self.weights)
                residuals.append((series[t] - pred) ** 2)

        mean_res = float(np.mean(residuals))
        return mean_res, bool(mean_res > self.threshold)


class LSTMAutoencoder(nn.Module):
    """Rung 6: Classical Recurrent LSTM Autoencoder."""

    def __init__(self, input_dim: int = 42, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        # Repeat hidden state for all time steps
        rep = h_n.repeat(x.size(1), 1, 1).transpose(0, 1)
        recon, _ = self.decoder(rep)
        return recon

    def score_window(self, x_np: np.ndarray, threshold: float = 0.05) -> Tuple[float, bool]:
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
            recon = self.forward(x_t)
            err = float(torch.mean((x_t - recon) ** 2).item())
        return err, bool(err > threshold)


class DeepLogModel(nn.Module):
    """Rung 7: DeepLog (Du et al. ACM CCS 2017) Log-only next-template GRU predictor."""

    def __init__(self, num_templates: int = 29, hidden_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_templates + 1, 16)
        self.gru = nn.GRU(16, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_templates + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        return self.fc(out)

    def score_window(self, x_np: np.ndarray, top_k: int = 3) -> Tuple[float, bool]:
        # Extract template sequence from channels 5..34
        templates = np.argmax(x_np[:, 5:34], axis=1)
        self.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(templates, dtype=torch.long).unsqueeze(0)
            logits = self.forward(t_tensor[:, :-1])
            # Check if next template is in top-k
            top_preds = torch.topk(logits, k=top_k, dim=-1).indices[0]  # (T-1, k)
            targets = t_tensor[0, 1:]
            
            anomalies = 0
            for i, target in enumerate(targets):
                if target not in top_preds[i]:
                    anomalies += 1

        anomaly_rate = anomalies / max(len(targets), 1)
        return float(anomaly_rate), bool(anomaly_rate > 0.25)


class USADMetricModel(nn.Module):
    """Rung 8: USAD (Audibert et al. ACM KDD 2020) Metric-only dual autoencoder."""

    def __init__(self, metric_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(metric_dim * 60, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, metric_dim * 60),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, metric_dim * 60),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        flat = x.reshape(batch, -1)
        z = self.encoder(flat)
        w1 = self.decoder1(z).reshape(batch, 60, 5)
        w2 = self.decoder2(z).reshape(batch, 60, 5)
        return w1, w2

    def score_window(self, x_np: np.ndarray, threshold: float = 0.08) -> Tuple[float, bool]:
        self.eval()
        with torch.no_grad():
            metrics_t = torch.tensor(x_np[:, :5], dtype=torch.float32).unsqueeze(0)
            w1, w2 = self.forward(metrics_t)
            err = 0.5 * torch.mean((metrics_t - w1) ** 2) + 0.5 * torch.mean((metrics_t - w2) ** 2)
            score = float(err.item())
        return score, bool(score > threshold)
