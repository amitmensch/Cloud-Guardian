"""
Sliding Window & Session Buffer for CloudGuardian.
Constructs 3D temporal tensors (N, TimeSteps, Features) from synchronized
30-second ticks or HDFS block sessions.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class SlidingWindowBuffer:
    """
    Maintains a temporal buffer of 30-second synchronized ticks and produces
    sliding windows of shape (WindowSize, Features).
    Default: WindowSize = 60 ticks (30 minutes of history at 30s resolution).
    """

    def __init__(self, window_size: int = 60, step_size: int = 1):
        self.window_size = window_size
        self.step_size = step_size
        self.tick_history: List[List[float]] = []
        self.tick_timestamps: List[float] = []

    def append_tick(self, feature_vector: List[float], timestamp: float):
        """Appends a new 30s tick feature vector to the buffer."""
        self.tick_history.append(feature_vector)
        self.tick_timestamps.append(timestamp)

    def is_ready(self) -> bool:
        """Checks if enough history has accumulated to produce at least one full window."""
        return len(self.tick_history) >= self.window_size

    def get_latest_window(self) -> Optional[np.ndarray]:
        """
        Extracts the most recent window of shape (1, WindowSize, Features).
        Returns None if fewer than window_size ticks have been collected.
        """
        if not self.is_ready():
            return None
        window_data = self.tick_history[-self.window_size :]
        arr = np.array(window_data, dtype=np.float32)
        # Shape: (1, T, F)
        return np.expand_dims(arr, axis=0)

    def generate_all_windows(self) -> Tuple[np.ndarray, List[float]]:
        """
        Generates all sliding windows from accumulated history.
        Returns:
            windows: np.ndarray of shape (N, WindowSize, Features)
            window_end_times: list of timestamps corresponding to the end of each window.
        """
        if not self.is_ready():
            return (
                np.empty((0, self.window_size, 0), dtype=np.float32),
                [],
            )

        windows = []
        end_times = []
        total_ticks = len(self.tick_history)

        for start_idx in range(
            0, total_ticks - self.window_size + 1, self.step_size
        ):
            end_idx = start_idx + self.window_size
            win = self.tick_history[start_idx:end_idx]
            windows.append(win)
            end_times.append(self.tick_timestamps[end_idx - 1])

        arr = np.array(windows, dtype=np.float32)
        return arr, end_times

    def clear(self):
        self.tick_history.clear()
        self.tick_timestamps.clear()


class HDFSSessionAggregator:
    """
    Aggregates log events by HDFS Block ID session.
    Extracts template occurrence counts and temporal statistics for each block session.
    """

    def __init__(self, num_templates: int = 29):
        self.num_templates = num_templates
        # block_id -> list of (timestamp, template_num)
        self.sessions: Dict[str, List[Tuple[float, int]]] = {}

    def add_event(self, block_id: str, timestamp: float, template_num: int):
        if block_id not in self.sessions:
            self.sessions[block_id] = []
        self.sessions[block_id].append((timestamp, template_num))

    def build_session_vector(self, block_id: str) -> np.ndarray:
        """
        Returns feature vector for a single block session:
        - Template counts: 29 dims
        - Temporal stats (mean, std, min, max deltas): 4 dims
        - Metadata (total events, unique events): 2 dims
        Total: 35 dims
        """
        events = self.sessions.get(block_id, [])
        counts = [0.0] * self.num_templates
        unique_templates = set()
        timestamps = []

        for ts, t_num in events:
            idx = t_num - 1
            if 0 <= idx < self.num_templates:
                counts[idx] += 1.0
            unique_templates.add(t_num)
            timestamps.append(ts)

        if len(timestamps) >= 2:
            timestamps.sort()
            deltas = [
                timestamps[i] - timestamps[i - 1]
                for i in range(1, len(timestamps))
            ]
            mean_d = float(np.mean(deltas))
            std_d = float(np.std(deltas))
            min_d = float(np.min(deltas))
            max_d = float(np.max(deltas))
        else:
            mean_d, std_d, min_d, max_d = 0.0, 0.0, 0.0, 0.0

        meta = [float(len(events)), float(len(unique_templates))]
        vec = counts + [mean_d, std_d, min_d, max_d] + meta
        return np.array(vec, dtype=np.float32)

    def total_sessions(self) -> int:
        return len(self.sessions)
