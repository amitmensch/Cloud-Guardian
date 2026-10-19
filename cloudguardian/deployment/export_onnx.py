"""
ONNX Export and High-Performance CPU Inference Engine for CloudGuardian.
Optimizes the PyTorch model for production CPU serving (< 50 MB size, < 25ms p95 latency).
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from typing import Dict, Any, Tuple

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from ..models.cloudguardian_model import CloudGuardianModel


class ExportableCloudGuardian(nn.Module):
    """
    Wrapper around CloudGuardianModel returning a tuple instead of a dictionary
    for standard ONNX graph tracing compatibility.
    """

    def __init__(self, base_model: CloudGuardianModel):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.base_model(x)
        return (
            out["reconstruction"],
            out["forecast"],
            out["risk_prob"],
            out["ttf_minutes"],
            out["recon_error"],
        )


def export_to_onnx(
    model: CloudGuardianModel,
    output_path: str = "models_onnx/cloudguardian.onnx",
    time_steps: int = 60,
    total_features: int = 42,
) -> Dict[str, Any]:
    """
    Exports CloudGuardianModel to an ONNX graph with dynamic batch dimensions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    exportable = ExportableCloudGuardian(model)
    exportable.eval()
    dummy_input = torch.randn(1, time_steps, total_features, dtype=torch.float32)

    torch.onnx.export(
        exportable,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["telemetry_window"],
        output_names=["reconstruction", "forecast", "risk_prob", "ttf_minutes", "recon_error"],
        dynamo=False,
    )

    file_size_bytes = os.path.getsize(output_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    return {
        "output_path": output_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "constraint_passed": file_size_mb < 50.0,
    }


class ONNXInferenceRunner:
    """
    Executes high-throughput, low-latency inference on CPU using ONNX Runtime.
    """

    def __init__(self, onnx_model_path: str, num_threads: int = 4):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed. Please install onnxruntime.")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, window_np: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Runs inference on a numpy array of shape (Batch, 60, 42).
        """
        if window_np.dtype != np.float32:
            window_np = window_np.astype(np.float32)

        raw_outputs = self.session.run(None, {self.input_name: window_np})
        return {
            "reconstruction": raw_outputs[0],
            "forecast": raw_outputs[1],
            "risk_prob": raw_outputs[2],
            "ttf_minutes": raw_outputs[3],
            "recon_error": raw_outputs[4],
        }

    def benchmark_latency(self, iterations: int = 50) -> Dict[str, float]:
        """
        Measures p50, p95, and mean CPU inference latency in milliseconds.
        """
        dummy = np.random.randn(1, 60, 42).astype(np.float32)
        # Warmup
        for _ in range(5):
            _ = self.predict(dummy)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = self.predict(dummy)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

        latencies = np.array(latencies)
        return {
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "min_latency_ms": round(float(np.min(latencies)), 2),
            "max_latency_ms": round(float(np.max(latencies)), 2),
        }
