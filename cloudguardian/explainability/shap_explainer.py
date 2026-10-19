"""
CloudGuardian Phase 4: Saliency-Preserving Attention-Guided SHAP Explainer
Module: cloudguardian.explainability.shap_explainer

Calculates feature-level attribution and root-cause rankings for detected
anomalies by combining reconstruction residuals with cached cross-attention weights.
Computes Hit-Rate@k against ground-truth injected chaos fault channels.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn


# Canonical human-readable names for the 42 multimodal feature channels
METRIC_NAMES = [
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "disk_io_wait_ms",
    "network_rx_tx_mbps",
    "database_wait_latency_ms",
]

# Canonical 29 HDFS/Microservice event template descriptions
TEMPLATE_DESCRIPTIONS = {
    0: "E1: User authentication success",
    1: "E2: Network packet drop / socket retry",
    2: "E3: Health check ping OK",
    3: "E4: HTTP 200 GET request processed",
    4: "E5: Cache hit in Redis session store",
    5: "E6: Scheduled cron job started",
    6: "E7: JVM Garbage collection pause / memory reclaim",
    7: "E8: Database connection opened",
    8: "E9: Database connection closed",
    9: "E10: File block write completed",
    10: "E11: API Gateway request routed",
    11: "E12: Disk block replication check",
    12: "E13: Thread pool worker idle",
    13: "E14: Database connection pool exhaustion / broken pipe",
    14: "E15: Transaction committed successfully",
    15: "E16: SSL handshake completed",
    16: "E17: Slow query logged (>250ms)",
    17: "E18: Downstream microservice 503 service unavailable",
    18: "E19: Disk storage watermark high (>85%)",
    19: "E20: CPU throttling limit reached",
    20: "E21: OutOfMemoryError / killed process",
    21: "E22: Corrupted TCP packet discarded",
    22: "E23: Configuration key reload event",
    23: "E24: Redis connection timeout / circuit breaker open",
    24: "E25: Worker thread starved / deadlocked",
    25: "E26: Cassandra replica write timeout",
    26: "E27: Client aborted TCP connection",
    27: "E28: HTTP 404 resource not found",
    28: "E29: Disk write I/O timeout / device unresponsive",
}

TIMING_NAMES = [
    "mean_inter_event_time_sec",
    "std_inter_event_time_sec",
    "max_log_burst_rate_per_sec",
    "zero_log_silence_duration_sec",
]

METADATA_NAMES = [
    "active_thread_count",
    "error_level_log_ratio",
    "warn_level_log_ratio",
    "open_file_descriptors",
]


def get_all_channel_names() -> List[str]:
    """Returns canonical list of all 42 channel names."""
    names = list(METRIC_NAMES)
    for i in range(29):
        names.append(TEMPLATE_DESCRIPTIONS.get(i, f"E{i+1}: Template {i+1}"))
    names.extend(TIMING_NAMES)
    names.extend(METADATA_NAMES)
    return names


class AttentionSHAPExplainer:
    """
    Fast Attention-Guided Feature Attribution Explainer.
    
    Combines:
    1. Channel Reconstruction Residuals: (x - x_rec)^2 across the 42 channels.
    2. Cross-Attention Saliency: Saliency weights from the MultimodalCrossAttention
       layer highlighting which log events coupled most strongly with metric surges.
    """

    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.channel_names = get_all_channel_names()

    def explain(
        self,
        x: torch.Tensor,
        x_reconstructed: Optional[torch.Tensor] = None,
        attention_matrix: Optional[torch.Tensor] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Computes feature attribution and plain-English root causes for an input window.

        Args:
            x: Input tensor of shape (1, 60, 42) or (60, 42)
            x_reconstructed: Optional reconstructed tensor (1, 60, 42)
            attention_matrix: Optional attention weights (1, heads, 60, 60) or (60, 60)
            top_k: Number of top root-cause contributors to return (default 3)

        Returns:
            Dictionary containing:
            - top_contributors: List of top_k {channel, score, percentage, modality, description}
            - metric_saliency: Dict mapping 5 hardware metrics to attribution scores
            - log_saliency: Dict mapping top log templates to attribution scores
            - plain_english_summary: Concise diagnostic string for human operators
            - explanation_hit_rate_at_k: Helper to benchmark against ground-truth
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # 1. Compute reconstruction residual per channel
        if x_reconstructed is None and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                out = self.model(x)
                x_reconstructed = out.get("reconstruction", out.get("reconstructed"))
                if attention_matrix is None:
                    attention_matrix = out.get("attention")

        if x_reconstructed is not None:
            if x_reconstructed.dim() == 2:
                x_reconstructed = x_reconstructed.unsqueeze(0)
            # Channel-wise MSE across time: shape (42,)
            channel_errors = torch.mean((x - x_reconstructed) ** 2, dim=(0, 1)).detach().cpu().numpy()
        else:
            # Fallback to feature variance / deviation from baseline mean
            channel_errors = torch.var(x, dim=(0, 1)).detach().cpu().numpy()

        # 2. Extract attention saliency weight across time
        if isinstance(attention_matrix, dict):
            attention_matrix = attention_matrix.get("metric_to_log", attention_matrix.get("log_to_metric"))

        if attention_matrix is None and self.model is not None:
            if hasattr(self.model, "fusion") and hasattr(self.model.fusion, "last_metric_to_log_weights"):
                attention_matrix = self.model.fusion.last_metric_to_log_weights

        attention_weight_per_step = np.ones(60, dtype=np.float32) / 60.0
        if attention_matrix is not None:
            attn = attention_matrix.detach().cpu().numpy()
            if attn.ndim == 4:  # (B, heads, 60, 60)
                attn = np.mean(attn[0], axis=0)  # (60, 60)
            if attn.ndim == 3:  # (heads, 60, 60)
                attn = np.mean(attn, axis=0)
            # Measure attention received by each time step
            attention_weight_per_step = np.mean(attn, axis=0)
            if np.sum(attention_weight_per_step) > 0:
                attention_weight_per_step = attention_weight_per_step / np.sum(attention_weight_per_step)

        # 3. Combine channel error with temporal attention focus
        # Channels that spiked during highly-attended time steps get boosted attribution
        x_np = x[0].detach().cpu().numpy()  # (60, 42)
        temporal_weighted_activity = np.sum(np.abs(x_np) * attention_weight_per_step[:, None], axis=0)
        
        # Combined attribution score
        raw_attribution = channel_errors * (1.0 + 2.0 * temporal_weighted_activity)
        total_attr = np.sum(raw_attribution) + 1e-9
        attr_percentage = (raw_attribution / total_attr) * 100.0

        # 4. Rank top contributors
        ranked_indices = np.argsort(-raw_attribution)
        top_contributors = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            name = self.channel_names[idx]
            modality = "Hardware Metric" if idx < 5 else ("Log Template" if idx < 34 else "Telemetry Metadata")
            top_contributors.append({
                "rank": rank,
                "channel_index": int(idx),
                "channel_name": name,
                "modality": modality,
                "attribution_score": float(raw_attribution[idx]),
                "attribution_percentage": round(float(attr_percentage[idx]), 2),
                "description": name,
            })

        # 5. Modality breakdowns
        metric_saliency = {
            METRIC_NAMES[i]: round(float(attr_percentage[i]), 2)
            for i in range(5)
        }
        
        # Top log templates
        log_saliency = {}
        for i in range(5, 34):
            if attr_percentage[i] > 1.0:
                log_saliency[self.channel_names[i]] = round(float(attr_percentage[i]), 2)

        # 6. Generate plain-English diagnostic summary
        primary = top_contributors[0]
        secondary = top_contributors[1] if len(top_contributors) > 1 else None

        if primary["modality"] == "Hardware Metric":
            summary = (
                f"Incident primarily driven by {primary['channel_name']} "
                f"accounting for {primary['attribution_percentage']}% of anomalous variance."
            )
            if secondary and secondary["modality"] == "Log Template":
                summary += f" Highly correlated with software event '{secondary['channel_name']}'."
        else:
            summary = (
                f"Incident primarily triggered by software log event '{primary['channel_name']}' "
                f"({primary['attribution_percentage']}% attribution)."
            )
            if secondary:
                summary += f" Associated with surge in {secondary['channel_name']}."

        return {
            "top_contributors": top_contributors,
            "metric_saliency": metric_saliency,
            "log_saliency": log_saliency,
            "plain_english_summary": summary,
            "channel_attributions": [round(float(v), 4) for v in attr_percentage],
        }

    def evaluate_hit_rate(
        self,
        explanation: Dict[str, Any],
        ground_truth_fault_channels: List[int],
        k: int = 3,
    ) -> float:
        """
        Computes Hit-Rate@k: whether at least one ground-truth injected fault channel
        appears in the top-k explanation rankings.

        Args:
            explanation: Output dictionary from explain()
            ground_truth_fault_channels: List of channel indices known to be injected
            k: Top-k cutoff

        Returns:
            1.0 if hit, 0.0 otherwise.
        """
        top_k_indices = [item["channel_index"] for item in explanation["top_contributors"][:k]]
        for gt in ground_truth_fault_channels:
            if gt in top_k_indices:
                return 1.0
        return 0.0
