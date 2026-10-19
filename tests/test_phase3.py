"""
Unit and Integration Test Suite for Phase 3:
1. Conformal Prediction Engine
2. Three-Gate Alert Governance (FDR, Hysteresis, Cooldown)
3. Chaos Testbed Simulator (8 Failure Classes)
4. ONNX Export and CPU Latency Benchmarks
"""

import numpy as np
import torch
import os
import pytest

from cloudguardian.calibration.conformal import ConformalCalibrator
from cloudguardian.calibration.governance_gates import ThreeGateAlertGovernor, OnlineFDRController
from cloudguardian.models.cloudguardian_model import CloudGuardianModel
from cloudguardian.deployment.export_onnx import export_to_onnx, ONNXInferenceRunner
from chaos_testbed.injectors.chaos_simulator import ChaosEpisode, FaultClass


class TestConformalCalibration:
    def test_p_value_and_decision(self):
        calibrator = ConformalCalibrator(significance_level=0.01)
        # 100 normal calibration scores distributed around 0.01
        normal_scores = np.random.normal(0.01, 0.002, 100)
        tau = calibrator.fit(normal_scores)
        assert tau > 0.0

        # Normal test sample (score = 0.01) should have high p-value
        p_normal = calibrator.compute_p_value(0.01)
        assert p_normal > 0.10
        is_anom, _ = calibrator.is_anomaly(0.01)
        assert not is_anom

        # Extreme anomaly sample (score = 0.50) should have very low p-value
        p_anom = calibrator.compute_p_value(0.50)
        assert p_anom < 0.01
        is_anom, _ = calibrator.is_anomaly(0.50)
        assert is_anom


class TestThreeGateAlertGovernance:
    def test_transient_noise_suppression(self):
        """Single-point spike must be blocked by Gate 2 (Hysteresis)."""
        governor = ThreeGateAlertGovernor(target_fdr=0.05, hysteresis_m=3, hysteresis_k=5)
        
        # Feed 1 isolated spike (p = 0.0001, risk = 0.90)
        dec = governor.evaluate(
            host_id="server-1",
            tick_index=1,
            timestamp=1000.0,
            p_value=0.0001,
            risk_prob=0.90,
            ttf_minutes=5.0,
            reconstruction_error=0.15,
        )
        assert dec.gate1_passed is True  # FDR detected spike
        assert dec.gate2_passed is False # Hysteresis suppressed it (only 1 of 3 needed)
        assert dec.alert_fired is False  # NO ALERT!

    def test_sustained_precursor_alert_and_cooldown(self):
        """Sustained non-conforming ticks must trigger alert, then enter cooldown."""
        governor = ThreeGateAlertGovernor(target_fdr=0.05, hysteresis_m=3, hysteresis_k=5, cooldown_ticks=10)

        # Tick 1: spike 1
        dec1 = governor.evaluate("server-1", 1, 1000.0, 0.0001, 0.70, 10.0, 0.12)
        assert dec1.alert_fired is False

        # Tick 2: spike 2
        dec2 = governor.evaluate("server-1", 2, 1030.0, 0.0001, 0.80, 8.0, 0.14)
        assert dec2.alert_fired is False

        # Tick 3: spike 3 (Now 3/5 ticks non-conforming) -> ALERT MUST FIRE!
        dec3 = governor.evaluate("server-1", 3, 1060.0, 0.0001, 0.90, 6.0, 0.16)
        assert dec3.alert_fired is True
        assert dec3.alert_payload is not None
        assert dec3.severity == "CRITICAL"

        # Tick 4: immediate next tick -> Gate 3 (Cooldown) MUST SUPPRESS
        dec4 = governor.evaluate("server-1", 4, 1090.0, 0.0001, 0.90, 6.0, 0.16)
        assert dec4.gate3_passed is False # in cooldown
        assert dec4.alert_fired is False  # Alert storm prevented!


class TestChaosSimulator:
    def test_memory_leak_generation(self):
        episode = ChaosEpisode(fault_type=FaultClass.MEMORY_LEAK_OOM, total_ticks=60, precursor_start_tick=20, onset_tick=45)
        stream = episode.generate_telemetry_stream()

        assert len(stream) == 60
        # Check that memory climbs in precursor window
        tick_10_mem = stream[10]["metrics"]["mem_util"]
        tick_40_mem = stream[40]["metrics"]["mem_util"]
        assert tick_40_mem > tick_10_mem + 20.0  # Significant RAM growth

        # Check that OOM fatal log appears at onset
        onset_logs = [msg for t_num, lvl, msg in stream[45]["logs"]]
        assert any("Out of memory" in m for m in onset_logs)


class TestONNXExportAndCPUInference:
    def test_onnx_export_and_latency(self, tmp_path):
        onnx_path = str(tmp_path / "test_cloudguardian.onnx")
        model = CloudGuardianModel(hidden_dim=32)

        # Export to ONNX
        res = export_to_onnx(model, output_path=onnx_path)
        assert os.path.exists(onnx_path)
        assert res["constraint_passed"] is True
        assert res["file_size_mb"] < 50.0  # Model is ~0.5 - 2 MB

        # Run ONNX Runtime CPU Inference
        runner = ONNXInferenceRunner(onnx_path, num_threads=2)
        dummy_input = np.random.randn(1, 60, 42).astype(np.float32)
        outputs = runner.predict(dummy_input)

        assert outputs["reconstruction"].shape == (1, 60, 42)
        assert outputs["forecast"].shape == (1, 10, 5)
        assert outputs["risk_prob"].shape == (1, 1)

        # Benchmark Latency
        bench = runner.benchmark_latency(iterations=10)
        assert bench["p95_latency_ms"] < 25.0  # Must be fast on CPU
