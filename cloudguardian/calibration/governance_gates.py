"""
Three-Gate Alert Governance Layer for CloudGuardian.
Implements:
1. Gate 1: Online False Discovery Rate (FDR) Controller (Alpha-Investing)
2. Gate 2: Temporal Hysteresis Buffer (m-of-k consecutive non-conforming ticks)
3. Gate 3: Per-Host Cooldown Filter (Suppresses redundant alert storms)
"""

from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import time


class GovernanceDecision:
    """Encapsulates the decision made by the three alert gates."""

    def __init__(
        self,
        alert_fired: bool,
        gate1_passed: bool,
        gate2_passed: bool,
        gate3_passed: bool,
        severity: str,
        rejection_reason: Optional[str] = None,
        alert_payload: Optional[Dict[str, Any]] = None,
    ):
        self.alert_fired = alert_fired
        self.gate1_passed = gate1_passed
        self.gate2_passed = gate2_passed
        self.gate3_passed = gate3_passed
        self.severity = severity
        self.rejection_reason = rejection_reason
        self.alert_payload = alert_payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_fired": self.alert_fired,
            "gate1_fdr_passed": self.gate1_passed,
            "gate2_hysteresis_passed": self.gate2_passed,
            "gate3_cooldown_passed": self.gate3_passed,
            "severity": self.severity,
            "rejection_reason": self.rejection_reason,
            "alert_payload": self.alert_payload,
        }


class OnlineFDRController:
    """
    Online False Discovery Rate (FDR) Control based on Alpha-Investing.
    Bounds the long-run False Discovery Proportion under stream scoring.
    """

    def __init__(self, target_fdr: float = 0.05, initial_wealth: float = 0.05):
        self.target_fdr = target_fdr
        self.wealth = initial_wealth
        self.total_tests = 0
        self.total_discoveries = 0

    def get_test_threshold(self) -> float:
        """Computes current alpha budget allocated for the next test."""
        # Allocate a fraction of current wealth
        alpha_t = min(0.05, self.wealth / 2.0)
        return max(1e-4, alpha_t)

    def update(self, p_value: float) -> Tuple[bool, float]:
        """
        Updates wealth based on test outcome.
        Returns:
            (is_discovery, allocated_alpha)
        """
        self.total_tests += 1
        alpha_t = self.get_test_threshold()

        if p_value <= alpha_t:
            # Discovery: reward wealth
            self.total_discoveries += 1
            self.wealth += (self.target_fdr - alpha_t)
            return True, alpha_t
        else:
            # Non-discovery: penalize wealth
            self.wealth -= alpha_t / (1.0 - alpha_t)
            self.wealth = max(1e-4, self.wealth)
            return False, alpha_t


class ThreeGateAlertGovernor:
    """
    Coordinates Gate 1 (Online FDR), Gate 2 (Hysteresis), and Gate 3 (Cooldown).
    Ensures that only sustained, high-confidence precursors trigger actionable alerts.
    """

    def __init__(
        self,
        target_fdr: float = 0.05,
        hysteresis_m: int = 3,       # Requires 3 out of last 5 ticks to be non-conforming
        hysteresis_k: int = 5,
        cooldown_ticks: int = 30,    # 30 ticks = 15 minutes of suppression per host
    ):
        self.fdr_controller = OnlineFDRController(target_fdr=target_fdr)
        self.hysteresis_m = hysteresis_m
        self.hysteresis_k = hysteresis_k
        self.cooldown_ticks = cooldown_ticks

        # State tracking per host
        # host_id -> deque of bools (length <= hysteresis_k)
        self.host_history: Dict[str, deque] = {}
        # host_id -> last alert tick index
        self.host_last_alert_tick: Dict[str, int] = {}
        # Track current tick counter
        self.current_tick: int = 0

    def evaluate(
        self,
        host_id: str,
        tick_index: int,
        timestamp: float,
        p_value: float,
        risk_prob: float,
        ttf_minutes: float,
        reconstruction_error: float,
    ) -> GovernanceDecision:
        """
        Passes a telemetry window through all three alert gates.
        """
        self.current_tick = tick_index

        # Initialize history queue if first time seeing host
        if host_id not in self.host_history:
            self.host_history[host_id] = deque(maxlen=self.hysteresis_k)

        # -----------------------------------------------------------------
        # GATE 1: Online False Discovery Rate (FDR) Controller
        # -----------------------------------------------------------------
        is_discovery, allocated_alpha = self.fdr_controller.update(p_value)
        gate1_passed = is_discovery

        # Record whether this tick is statistically non-conforming
        self.host_history[host_id].append(gate1_passed)

        # -----------------------------------------------------------------
        # GATE 2: Temporal Hysteresis Filter (m-of-k)
        # -----------------------------------------------------------------
        non_conforming_count = sum(self.host_history[host_id])
        gate2_passed = non_conforming_count >= self.hysteresis_m

        # -----------------------------------------------------------------
        # GATE 3: Per-Host Cooldown Filter
        # -----------------------------------------------------------------
        last_alert = self.host_last_alert_tick.get(host_id, -999999)
        ticks_since_last_alert = tick_index - last_alert
        in_cooldown = ticks_since_last_alert < self.cooldown_ticks

        # Escalation override: if risk > 95% and TTF < 3 mins, allow alert even in cooldown
        is_critical_escalation = (risk_prob > 0.95 and ttf_minutes < 3.0)
        gate3_passed = (not in_cooldown) or is_critical_escalation

        # Determine overall severity level
        if risk_prob > 0.85 or ttf_minutes < 5.0:
            severity = "CRITICAL"
        elif risk_prob > 0.50 or ttf_minutes < 15.0:
            severity = "WARNING"
        elif gate1_passed:
            severity = "ELEVATED"
        else:
            severity = "NOMINAL"

        # Final Alert Decision: All 3 gates must pass
        alert_fired = gate1_passed and gate2_passed and gate3_passed

        # Formulate rejection reason if suppressed
        rejection_reason = None
        if not alert_fired:
            if not gate1_passed:
                rejection_reason = f"Gate 1 (FDR): p-value {p_value:.4f} exceeds dynamic budget {allocated_alpha:.4f}"
            elif not gate2_passed:
                rejection_reason = f"Gate 2 (Hysteresis): {non_conforming_count}/{self.hysteresis_k} ticks non-conforming (requires {self.hysteresis_m})"
            elif not gate3_passed:
                rejection_reason = f"Gate 3 (Cooldown): Host '{host_id}' in cooldown ({ticks_since_last_alert}/{self.cooldown_ticks} ticks)"

        alert_payload = None
        if alert_fired:
            self.host_last_alert_tick[host_id] = tick_index
            alert_payload = {
                "host_id": host_id,
                "tick_index": tick_index,
                "timestamp": timestamp,
                "severity": severity,
                "p_value": p_value,
                "risk_probability": round(risk_prob * 100.0, 2),
                "time_to_failure_minutes": round(ttf_minutes, 1),
                "reconstruction_error": round(reconstruction_error, 4),
                "message": f"Precursor anomaly detected on {host_id}. Imminent failure in ~{ttf_minutes:.1f} mins.",
            }

        return GovernanceDecision(
            alert_fired=alert_fired,
            gate1_passed=gate1_passed,
            gate2_passed=gate2_passed,
            gate3_passed=gate3_passed,
            severity=severity,
            rejection_reason=rejection_reason,
            alert_payload=alert_payload,
        )
