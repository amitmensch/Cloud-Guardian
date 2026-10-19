"""
Chaos Fault Injector Engine for CloudGuardian.
Simulates and executes the 8 Canonical Infrastructure Failure Modes:
1. MEMORY_LEAK_OOM: Gradual memory accumulation leading to OOM-kill
2. DISK_EXHAUSTION: High-volume log flood filling disk space
3. CONNECTION_POOL_EXHAUSTION: Unreleased database connections causing wait timeouts
4. SLOW_QUERIES: Unindexed DB queries causing CPU spikes and request backpressure
5. NETWORK_DEGRADATION: Artificial packet jitter, latency, and socket timeouts
6. CPU_CONTENTION: Heavy background computation starving worker threads
7. CASCADING_DEPENDENCY_FAILURE: Cache service failure cascading to database overload
8. CONFIG_REGRESSION: Bad timeout/threadpool configuration causing immediate instability
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple
from enum import Enum


class FaultClass(str, Enum):
    MEMORY_LEAK_OOM = "memory_leak_oom"
    DISK_EXHAUSTION = "disk_exhaustion"
    CONNECTION_POOL_EXHAUSTION = "connection_pool_exhaustion"
    SLOW_QUERIES = "slow_queries"
    NETWORK_DEGRADATION = "network_degradation"
    CPU_CONTENTION = "cpu_contention"
    CASCADING_FAILURE = "cascading_failure"
    CONFIG_REGRESSION = "config_regression"


class ChaosEpisode:
    """Represents a simulated or live chaos fault episode with ground-truth labels."""

    def __init__(
        self,
        fault_type: FaultClass,
        total_ticks: int = 60,
        precursor_start_tick: int = 25,
        onset_tick: int = 50,
    ):
        self.fault_type = fault_type
        self.total_ticks = total_ticks
        self.precursor_start_tick = precursor_start_tick
        self.onset_tick = onset_tick
        self.ground_truth = {
            "fault_type": fault_type.value,
            "precursor_tick": precursor_start_tick,
            "onset_tick": onset_tick,
            "precursor_duration_ticks": onset_tick - precursor_start_tick,
            "precursor_duration_minutes": (onset_tick - precursor_start_tick) * 0.5,
        }

    def generate_telemetry_stream(self) -> List[Dict[str, Any]]:
        """
        Generates 60 ticks (30 minutes) of realistic metric and log telemetry
        reflecting the specific failure class.
        """
        stream = []
        base_epoch = 1700000000.0

        for t in range(self.total_ticks):
            ts = base_epoch + (t * 30.0)
            in_precursor = (self.precursor_start_tick <= t < self.onset_tick)
            in_failure = (t >= self.onset_tick)

            progress = 0.0
            if in_precursor:
                progress = (t - self.precursor_start_tick) / float(self.onset_tick - self.precursor_start_tick)
            elif in_failure:
                progress = 1.0

            # Default baseline metrics
            cpu = 25.0 + 4.0 * np.sin(t / 6.0) + np.random.normal(0, 1.0)
            mem = 38.0 + np.random.normal(0, 0.8)
            disk_lat = 5.0 + np.random.normal(0, 0.5)
            net_mb = 18.0 + np.random.normal(0, 1.5)
            conn_wait = 2.5 + np.random.normal(0, 0.3)

            logs = []  # list of (template_num, severity, message)

            # Normal baseline logs
            logs.append((1, "INFO", "dfs.DataNode: Receiving block <BLOCK_ID>"))
            logs.append((2, "INFO", "dfs.FSDataset: Packet responder <NUM> terminating"))

            # Specific fault dynamics
            if self.fault_type == FaultClass.MEMORY_LEAK_OOM:
                if in_precursor or in_failure:
                    mem = 38.0 + progress * 60.0  # RAM creeps up to 98%
                    if progress > 0.6:
                        logs.append((15, "WARN", "jvm.gc: OverheadLimitExceeded: GC pause 3400ms"))
                if in_failure:
                    logs.append((27, "FATAL", "sys.kernel: Out of memory: Kill process (order-service)"))

            elif self.fault_type == FaultClass.CONNECTION_POOL_EXHAUSTION:
                if in_precursor or in_failure:
                    conn_wait = 2.5 + progress * 45.0  # Latency climbs to 47.5ms
                    if progress > 0.4:
                        logs.append((14, "WARN", "db.pool: Connection pool wait time exceeded 3000ms"))
                if in_failure:
                    logs.append((23, "ERROR", "db.pool: PoolTimeoutException: Timeout acquiring connection"))

            elif self.fault_type == FaultClass.SLOW_QUERIES:
                if in_precursor or in_failure:
                    cpu = 25.0 + progress * 65.0
                    disk_lat = 5.0 + progress * 35.0
                    logs.append((14, "WARN", "db.engine: Slow query: full table scan on orders"))

            elif self.fault_type == FaultClass.NETWORK_DEGRADATION:
                if in_precursor or in_failure:
                    net_mb = max(1.0, 18.0 - progress * 15.0)
                    conn_wait = 2.5 + progress * 30.0
                    logs.append((23, "ERROR", "net.socket: SocketTimeoutException: Read timed out"))

            elif self.fault_type == FaultClass.CPU_CONTENTION:
                if in_precursor or in_failure:
                    cpu = 25.0 + progress * 72.0
                    logs.append((12, "WARN", "sys.throttle: CPU throttle quota exceeded on cgroup"))

            elif self.fault_type == FaultClass.DISK_EXHAUSTION:
                if in_precursor or in_failure:
                    disk_lat = 5.0 + progress * 50.0
                    logs.append((27, "FATAL", "dfs.DataNode: Disk volume full on /data"))

            elif self.fault_type == FaultClass.CASCADING_FAILURE:
                if in_precursor or in_failure:
                    conn_wait = 2.5 + progress * 38.0
                    cpu = 25.0 + progress * 40.0
                    logs.append((18, "ERROR", "cache.redis: Connection refused; failing over to db"))

            elif self.fault_type == FaultClass.CONFIG_REGRESSION:
                if in_precursor or in_failure:
                    conn_wait = 35.0
                    logs.append((12, "WARN", "app.config: Deprecated timeout configuration active"))

            stream.append(
                {
                    "tick": t,
                    "timestamp": ts,
                    "metrics": {
                        "cpu_util": float(cpu),
                        "mem_util": float(mem),
                        "disk_latency_ms": float(disk_lat),
                        "net_throughput_mb": float(net_mb),
                        "conn_wait_ms": float(conn_wait),
                    },
                    "logs": logs,
                    "is_precursor": in_precursor,
                    "is_failure": in_failure,
                }
            )

        return stream
