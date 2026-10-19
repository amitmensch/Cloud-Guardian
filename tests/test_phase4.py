"""
CloudGuardian Phase 4 Test Suite: Explainability, Counterfactuals & Server
"""

import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient

from cloudguardian.models.cloudguardian_model import CloudGuardianModel
from cloudguardian.explainability.shap_explainer import (
    AttentionSHAPExplainer,
    METRIC_NAMES,
    TEMPLATE_DESCRIPTIONS,
    get_all_channel_names,
)
from cloudguardian.explainability.counterfactual import CounterfactualExplainer
from cloudguardian.server.api import app


class TestAttentionSHAPExplainer:
    """Tests the attention-guided SHAP explainer and Hit-Rate@k calculation."""

    def test_explainer_output_structure_and_percentages(self):
        model = CloudGuardianModel(metric_features=5, log_features=37, hidden_dim=64)
        model.eval()
        explainer = AttentionSHAPExplainer(model)

        # Create window with injected spike on channel 4 (database latency)
        x = torch.zeros(1, 60, 42)
        x[0, 30:, 4] = 2.5  # sustained DB latency surge
        x[0, 30:, 13] = 1.8 # coupled template E14 surge

        explanation = explainer.explain(x, top_k=3)

        assert "top_contributors" in explanation
        assert "metric_saliency" in explanation
        assert "log_saliency" in explanation
        assert "plain_english_summary" in explanation
        assert len(explanation["top_contributors"]) == 3

        # Check attribution percentages sum to ~100%
        channel_attrs = explanation["channel_attributions"]
        assert len(channel_attrs) == 42
        total_pct = sum(channel_attrs)
        assert abs(total_pct - 100.0) < 1.0

    def test_hit_rate_at_3_on_injected_fault(self):
        model = CloudGuardianModel(metric_features=5, log_features=37, hidden_dim=64)
        model.eval()
        explainer = AttentionSHAPExplainer(model)

        # Injected ground truth fault channels: 4 (DB latency) and 13 (Template E14)
        ground_truth = [4, 13]
        x = torch.zeros(1, 60, 42)
        x[0, 20:, 4] = 3.0
        x[0, 20:, 13] = 2.0

        explanation = explainer.explain(x, top_k=3)
        top_indices = [item["channel_index"] for item in explanation["top_contributors"]]

        # Hit-Rate@3 must be 1.0 (at least one injected channel caught in top 3)
        hit_rate = explainer.evaluate_hit_rate(explanation, ground_truth, k=3)
        assert hit_rate == 1.0
        assert (4 in top_indices) or (13 in top_indices)


class TestCounterfactualExplainer:
    """Tests the counterfactual what-if generator and operational runbooks."""

    def test_counterfactual_generation(self):
        model = CloudGuardianModel(metric_features=5, log_features=37, hidden_dim=64)
        model.eval()
        explainer = AttentionSHAPExplainer(model)
        cf_explainer = CounterfactualExplainer(model, conformal_tau=0.00849)

        # Fault window
        x = torch.zeros(1, 60, 42)
        x[0, 20:, 4] = 2.0 # DB latency
        x[0, 20:, 13] = 1.5 # E14

        explanation = explainer.explain(x, top_k=3)
        cf = cf_explainer.generate_counterfactual(x, explanation, current_risk=0.96, target_risk=0.05)

        assert "recommendations" in cf
        assert "projected_risk_reduction_pct" in cf
        assert "operational_runbook" in cf
        assert "plain_english_counterfactual" in cf
        assert cf["projected_risk_reduction_pct"] > 80.0
        assert len(cf["recommendations"]) >= 1
        assert "database" in cf["operational_runbook"].lower() or "pod" in cf["operational_runbook"].lower()


class TestFastAPIServerEndpoints:
    """Tests FastAPI REST routes for dashboard, health, scoring, and chaos."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_dashboard_ui_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "CloudGuardian Operations Center" in resp.text

    def test_health_check_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "HEALTHY"
        assert data["model_parameters"] == 193393
        assert data["p95_cpu_latency_ms"] == 3.46

    def test_chaos_injection_endpoint(self, client):
        resp = client.post("/api/chaos/inject", json={"scenario": "CONNECTION_POOL_EXHAUSTION", "host_id": "order-service-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "INJECTED"
        assert data["scenario"] == "CONNECTION_POOL_EXHAUSTION"

    def test_scoring_endpoint(self, client):
        resp = client.post("/api/score", json={"host_id": "order-service-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert "reconstruction_error" in data
        assert "conformal_p_value" in data
        assert "risk_probability" in data
        assert "explainability" in data
        assert "counterfactual" in data
        assert len(data["explainability"]["top_contributors"]) == 3

    def test_incidents_endpoint(self, client):
        resp = client.get("/api/incidents")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_incidents" in data
        assert "incidents" in data
