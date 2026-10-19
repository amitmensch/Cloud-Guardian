"""
CloudGuardian Phase 5 Test Suite: Baselines, Benchmarking, Ablations & Manuscript
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from experiments.baselines import (
    StaticThresholdDetector,
    KeywordPatternMatcher,
    STIDETemplateDetector,
    TabularIsolationForest,
    ARIMAForecaster,
    LSTMAutoencoder,
    DeepLogModel,
    USADMetricModel,
)
from experiments.benchmark_runner import BenchmarkRunner, generate_benchmark_dataset
from experiments.ablation_study import AblationStudyRunner
from experiments.leakage_audit import LeakageAuditRunner


class TestBaselines:
    """Tests the 8 comparative baseline detectors."""

    def test_static_threshold_detector(self):
        detector = StaticThresholdDetector(cpu_thresh=0.80, latency_thresh=0.40)
        normal = np.zeros((60, 42))
        normal[:, 0] = 0.30
        normal[:, 4] = 0.05
        score_norm, alert_norm = detector.score_window(normal)
        assert alert_norm is False

        fault = np.zeros((60, 42))
        fault[:, 0] = 0.95
        score_fault, alert_fault = detector.score_window(fault)
        assert alert_fault is True
        assert score_fault > score_norm

    def test_keyword_pattern_matcher(self):
        matcher = KeywordPatternMatcher(error_rate_threshold=0.10)
        normal = np.zeros((60, 42))
        normal[:, 39] = 0.01
        _, alert_norm = matcher.score_window(normal)
        assert alert_norm is False

        fault = np.zeros((60, 42))
        fault[:, 39] = 0.45
        _, alert_fault = matcher.score_window(fault)
        assert alert_fault is True

    def test_stide_detector(self):
        stide = STIDETemplateDetector(window_size=3, mismatch_threshold=0.20)
        # Train on sequences of template 5
        train_seq = [np.zeros((60, 42)) for _ in range(5)]
        for seq in train_seq:
            seq[:, 5] = 1.0
        stide.fit(train_seq)

        test_normal = np.zeros((60, 42))
        test_normal[:, 5] = 1.0
        mismatch_norm, alert_norm = stide.score_window(test_normal)
        assert alert_norm is False
        assert mismatch_norm == 0.0

        test_abnormal = np.zeros((60, 42))
        test_abnormal[:, 13] = 1.0 # Unseen template 13
        mismatch_abn, alert_abn = stide.score_window(test_abnormal)
        assert alert_abn is True
        assert mismatch_abn > 0.5

    def test_isolation_forest_and_arima(self):
        normal_wins = [np.random.normal(0.2, 0.02, size=(60, 42)) for _ in range(30)]
        iso = TabularIsolationForest(contamination=0.05)
        iso.fit(normal_wins)

        score_norm, alert_norm = iso.score_window(normal_wins[0])
        assert isinstance(alert_norm, bool)

        arima = ARIMAForecaster(p=3, threshold=0.40)
        score_arima, _ = arima.score_window(normal_wins[0])
        assert score_arima >= 0.0

    def test_neural_baselines_forward(self):
        x = torch.randn(2, 60, 42)

        # LSTM Autoencoder
        lstm_ae = LSTMAutoencoder(input_dim=42, hidden_dim=16)
        recon = lstm_ae(x)
        assert recon.shape == (2, 60, 42)

        # DeepLog
        deeplog = DeepLogModel(num_templates=29, hidden_dim=16)
        t_seq = torch.randint(0, 29, (2, 59))
        logits = deeplog(t_seq)
        assert logits.shape == (2, 59, 30)

        # USAD
        usad = USADMetricModel(metric_dim=5, hidden_dim=16)
        w1, w2 = usad(x[:, :, :5])
        assert w1.shape == (2, 60, 5)
        assert w2.shape == (2, 60, 5)


class TestBenchmarkAndAblationRunners:
    """Tests the benchmark harness, dataset generator, and ablation evaluator."""

    def test_benchmark_dataset_generation(self):
        X, y, leads = generate_benchmark_dataset(num_samples=40, fault_ratio=0.25, seed=42)
        assert X.shape == (40, 60, 42)
        assert len(y) == 40
        assert len(leads) == 40
        assert sum(y) == 10

    def test_benchmark_runner_and_latex(self):
        runner = BenchmarkRunner()
        results = runner.run_all()

        assert len(results) == 9
        # Check CloudGuardian is the 9th result
        cg_res = results[8]
        assert "CloudGuardian" in cg_res["name"]
        assert cg_res["f1_score"] > 0.90
        assert cg_res["lead_time_minutes"] > 10.0
        assert cg_res["false_alarms_per_day"] < 2.0

        latex_tab = runner.format_latex_table(results)
        assert r"\begin{table*}" in latex_tab
        assert r"\end{table*}" in latex_tab
        assert "CloudGuardian" in latex_tab

    def test_ablation_study_and_latex(self):
        ablation_runner = AblationStudyRunner()
        ablations = ablation_runner.run_ablations()
        assert len(ablations) == 6

        full_model = ablations[0]
        assert "Full" in full_model["variant"]
        assert full_model["f1_score"] > 0.94

        latex_tab = ablation_runner.format_latex_table(ablations)
        assert r"\begin{table}" in latex_tab
        assert r"\end{table}" in latex_tab

    def test_leakage_audit(self):
        audit_runner = LeakageAuditRunner()
        audit = audit_runner.run_audit()
        assert audit["overall_audit_passed"] is True
        assert audit["leakage_probability"] == 0.0
        assert len(audit["checks"]) == 4

    def test_paper_manuscript_integrity(self):
        paper_file = Path(__file__).resolve().parent.parent / "paper" / "main.tex"
        assert paper_file.exists()
        with open(paper_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "CloudGuardian" in content
        assert "Ayush Kumar Agrawal" in content
        assert "Shashank Sahu" in content
        assert "AKGEC" in content
        assert "Conformal Prediction" in content
        assert r"\begin{document}" in content
        assert r"\end{document}" in content
