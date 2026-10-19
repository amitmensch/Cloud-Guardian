"""
CloudGuardian Server Module
Provides FastAPI REST and WebSocket endpoints for real-time telemetry and dashboard.
"""

from cloudguardian.server.api import app, run

__all__ = ["app", "run"]
