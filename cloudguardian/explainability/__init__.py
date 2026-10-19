"""
CloudGuardian Explainability Module
Provides saliency-preserving SHAP feature attribution and counterfactual recommendations.
"""

from cloudguardian.explainability.shap_explainer import (
    AttentionSHAPExplainer,
    METRIC_NAMES,
    TEMPLATE_DESCRIPTIONS,
    get_all_channel_names,
)
from cloudguardian.explainability.counterfactual import CounterfactualExplainer

__all__ = [
    "AttentionSHAPExplainer",
    "CounterfactualExplainer",
    "METRIC_NAMES",
    "TEMPLATE_DESCRIPTIONS",
    "get_all_channel_names",
]
