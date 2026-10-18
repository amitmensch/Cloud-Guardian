"""
Metric and Log Tick Synchronizer for CloudGuardian.
Aligns continuous Prometheus time-series metrics with discrete parsed log event templates
onto a uniform 30-second temporal grid.
"""

from typing import List, Dict, Any, Optional
import math


class TickData:
    """Represents aligned telemetry for a single 30-second temporal window."""

    def __init__(self, tick_id: int, start_time: float, max_templates: int = 32):
        self.tick_id = tick_id
        self.start_time = start_time
        self.max_templates = max_templates

        # Metrics accumulator
        self.metrics: Dict[str, List[float]] = {
            "cpu_util": [],
            "mem_util": [],
            "disk_latency_ms": [],
            "net_throughput_mb": [],
            "conn_wait_ms": [],
        }

        # Log template events in this tick: list of (timestamp, template_num, level)
        self.log_events: List[Dict[str, Any]] = []

    def add_metric_sample(self, metric_name: str, value: float):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(float(value))

    def add_log_event(self, timestamp: float, template_num: int, level: str = "INFO"):
        self.log_events.append(
            {
                "timestamp": timestamp,
                "template_num": template_num,
                "level": level.upper(),
            }
        )

    def compute_feature_vector(self) -> List[float]:
        """
        Constructs the feature vector for this 30s tick:
        1. 5 Mean Metric values (CPU, Mem, Disk Latency, Net Throughput, Conn Wait)
        2. Template count vector (dim = max_templates)
        3. Inter-event temporal delta stats: [mean_delta, std_delta, min_delta, max_delta]
        4. Log metadata: [total_logs, unique_templates, error_ratio]
        Total dimension = 5 + max_templates + 4 + 3 = max_templates + 12
        """
        # 1. Metric Means (impute reasonable default if missing)
        defaults = {
            "cpu_util": 20.0,
            "mem_util": 35.0,
            "disk_latency_ms": 5.0,
            "net_throughput_mb": 10.0,
            "conn_wait_ms": 2.0,
        }
        metric_means = []
        for k in [
            "cpu_util",
            "mem_util",
            "disk_latency_ms",
            "net_throughput_mb",
            "conn_wait_ms",
        ]:
            vals = self.metrics.get(k, [])
            metric_means.append(sum(vals) / len(vals) if vals else defaults[k])

        # 2. Template frequency vector
        template_counts = [0.0] * self.max_templates
        unique_seen = set()
        error_count = 0
        timestamps = []

        for evt in self.log_events:
            t_num = evt["template_num"]
            # 1-indexed to 0-indexed
            idx = t_num - 1
            if 0 <= idx < self.max_templates:
                template_counts[idx] += 1.0
            unique_seen.add(t_num)
            if evt["level"] in ("WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"):
                error_count += 1
            timestamps.append(evt["timestamp"])

        total_logs = len(self.log_events)
        unique_templates = len(unique_seen)
        error_ratio = (error_count / total_logs) if total_logs > 0 else 0.0

        # 3. Inter-event time delta statistics
        if len(timestamps) >= 2:
            timestamps.sort()
            deltas = [
                timestamps[i] - timestamps[i - 1]
                for i in range(1, len(timestamps))
            ]
            mean_delta = sum(deltas) / len(deltas)
            variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
            std_delta = math.sqrt(variance)
            min_delta = min(deltas)
            max_delta = max(deltas)
        else:
            mean_delta, std_delta, min_delta, max_delta = 0.0, 0.0, 0.0, 0.0

        temporal_stats = [mean_delta, std_delta, min_delta, max_delta]
        log_metadata = [float(total_logs), float(unique_templates), error_ratio]

        return metric_means + template_counts + temporal_stats + log_metadata


class MetricLogSynchronizer:
    """
    Manages a continuous stream of metrics and log records, segmenting them
    into aligned 30-second discrete TickData buckets.
    """

    def __init__(
        self,
        tick_duration_sec: float = 30.0,
        max_templates: int = 32,
        start_time: Optional[float] = None,
    ):
        self.tick_duration_sec = tick_duration_sec
        self.max_templates = max_templates
        self.start_time = start_time
        self.ticks: Dict[int, TickData] = {}

    def _get_tick_id(self, timestamp: float) -> int:
        if self.start_time is None:
            self.start_time = timestamp
        return int((timestamp - self.start_time) // self.tick_duration_sec)

    def ingest_metric(self, timestamp: float, metric_dict: Dict[str, float]):
        """Ingests a snapshot of Prometheus metrics."""
        tid = self._get_tick_id(timestamp)
        if tid not in self.ticks:
            start_t = self.start_time + tid * self.tick_duration_sec
            self.ticks[tid] = TickData(tid, start_t, self.max_templates)
        for k, v in metric_dict.items():
            self.ticks[tid].add_metric_sample(k, v)

    def ingest_log(
        self, timestamp: float, template_num: int, level: str = "INFO"
    ):
        """Ingests a parsed log event."""
        tid = self._get_tick_id(timestamp)
        if tid not in self.ticks:
            start_t = self.start_time + tid * self.tick_duration_sec
            self.ticks[tid] = TickData(tid, start_t, self.max_templates)
        self.ticks[tid].add_log_event(timestamp, template_num, level)

    def get_tick_features(self, tick_id: int) -> List[float]:
        """Returns the feature vector for a specific tick."""
        if tick_id in self.ticks:
            return self.ticks[tick_id].compute_feature_vector()
        # Fallback empty tick
        dummy = TickData(
            tick_id,
            (self.start_time or 0.0) + tick_id * self.tick_duration_sec,
            self.max_templates,
        )
        return dummy.compute_feature_vector()

    def get_all_ordered_ticks(self) -> List[int]:
        """Returns sorted list of all active tick IDs."""
        return sorted(self.ticks.keys())
