"""
Comprehensive Test Suite for Phase 1 Ingestion and Windowing Pipeline.
Verifies LogRedactor, DrainParser, MetricLogSynchronizer, and SlidingWindowBuffer.
"""

import math
import numpy as np
import pytest

from cloudguardian.ingestion.redactor import LogRedactor
from cloudguardian.ingestion.drain_parser import DrainParser
from cloudguardian.ingestion.synchronizer import MetricLogSynchronizer
from cloudguardian.features.window_buffer import (
    SlidingWindowBuffer,
    HDFSSessionAggregator,
)


class TestLogRedactor:
    def setup_method(self):
        self.redactor = LogRedactor()

    def test_ip_redaction(self):
        line = "Receiving block blk_-1608999687 src: 10.251.43.115:50010 dest: 10.251.43.115:50010"
        redacted, meta = self.redactor.process(line)
        assert "<IP>" in redacted
        assert "10.251.43.115" not in redacted
        assert "<BLOCK_ID>" in redacted
        assert meta["block_id"] == "blk_-1608999687"

    def test_hex_and_path_redaction(self):
        line = "Fatal segfault at 0x7fff5fbff8b0 in binary /usr/local/bin/hadoop-daemon.sh"
        redacted, meta = self.redactor.process(line)
        assert "<HEX>" in redacted
        assert "0x7fff5fbff8b0" not in redacted
        assert "<PATH>" in redacted
        assert meta["level"] == "FATAL"

    def test_numeric_redaction(self):
        line = "Packet responder 3 for block blk_321 terminating after 5420 bytes"
        redacted, _ = self.redactor.process(line)
        assert "<NUM>" in redacted
        assert "5420" not in redacted


class TestDrainParser:
    def setup_method(self):
        self.parser = DrainParser(depth=4, similarity_threshold=0.5)

    def test_template_creation_and_matching(self):
        line1 = "Receiving block <BLOCK_ID> src: <IP> dest: <IP>"
        res1 = self.parser.parse(line1)
        assert res1["template_id"] == "E1"

        # Identical structure should reuse E1
        line2 = "Receiving block <BLOCK_ID> src: <IP> dest: <IP>"
        res2 = self.parser.parse(line2)
        assert res2["template_id"] == "E1"

        # Different action should create new template E2
        line3 = "Packet responder <NUM> for block <BLOCK_ID> terminating"
        res3 = self.parser.parse(line3)
        assert res3["template_id"] == "E2"
        assert self.parser.total_templates() == 2

    def test_wildcard_generalization(self):
        line1 = "Verification succeeded for block <BLOCK_ID>"
        line2 = "Verification failed for block <BLOCK_ID>"
        res1 = self.parser.parse(line1)
        res2 = self.parser.parse(line2)
        assert res1["template_id"] == res2["template_id"]
        # The differing word 'succeeded' vs 'failed' should generalize to <*>
        assert "<*>" in res2["template_str"]


class TestMetricLogSynchronizer:
    def setup_method(self):
        self.sync = MetricLogSynchronizer(
            tick_duration_sec=30.0, max_templates=10, start_time=1000.0
        )

    def test_tick_alignment_and_features(self):
        t0 = 1000.0
        # Ingest metrics into tick 0 (t = 1000 to 1029)
        self.sync.ingest_metric(
            t0 + 5.0,
            {
                "cpu_util": 45.0,
                "mem_util": 60.0,
                "disk_latency_ms": 12.0,
                "net_throughput_mb": 25.0,
                "conn_wait_ms": 15.0,
            },
        )
        self.sync.ingest_metric(
            t0 + 20.0,
            {
                "cpu_util": 55.0,
                "mem_util": 62.0,
                "disk_latency_ms": 14.0,
                "net_throughput_mb": 35.0,
                "conn_wait_ms": 25.0,
            },
        )

        # Ingest log events into tick 0
        self.sync.ingest_log(t0 + 2.0, template_num=1, level="INFO")
        self.sync.ingest_log(t0 + 12.0, template_num=1, level="INFO")
        self.sync.ingest_log(t0 + 22.0, template_num=3, level="ERROR")

        vec = self.sync.get_tick_features(0)
        # Expected feature dimension: 5 metrics + 10 templates + 4 time stats + 3 log metadata = 22
        assert len(vec) == 22

        # Check metric averages: CPU (45+55)/2 = 50.0, Mem (60+62)/2 = 61.0
        assert math.isclose(vec[0], 50.0, rel_tol=1e-3)
        assert math.isclose(vec[1], 61.0, rel_tol=1e-3)

        # Check template counts: E1 count = 2, E3 count = 1
        # indices: 0..4 metrics, 5..14 templates -> vec[5] = E1, vec[7] = E3
        assert vec[5] == 2.0
        assert vec[7] == 1.0

        # Check log metadata: total logs = 3, unique = 2, error ratio = 1/3
        assert vec[-3] == 3.0
        assert vec[-2] == 2.0
        assert math.isclose(vec[-1], 1.0 / 3.0, rel_tol=1e-3)


class TestSlidingWindowBuffer:
    def test_window_generation(self):
        buf = SlidingWindowBuffer(window_size=5, step_size=1)
        # Feed 10 ticks with 4 features each
        for i in range(10):
            vec = [float(i), float(i * 2), float(i * 3), float(i * 4)]
            buf.append_tick(vec, timestamp=1000.0 + i * 30.0)

        assert buf.is_ready()
        latest = buf.get_latest_window()
        assert latest.shape == (1, 5, 4)

        all_wins, end_times = buf.generate_all_windows()
        # 10 ticks - 5 window_size + 1 = 6 windows
        assert all_wins.shape == (6, 5, 4)
        assert len(end_times) == 6


class TestHDFSSessionAggregator:
    def test_session_vector_construction(self):
        agg = HDFSSessionAggregator(num_templates=10)
        agg.add_event("blk_101", timestamp=100.0, template_num=1)
        agg.add_event("blk_101", timestamp=110.0, template_num=2)
        agg.add_event("blk_101", timestamp=125.0, template_num=1)

        vec = agg.build_session_vector("blk_101")
        # Dimension: 10 templates + 4 time stats + 2 metadata = 16
        assert len(vec) == 16
        # E1 count = 2, E2 count = 1
        assert vec[0] == 2.0
        assert vec[1] == 1.0
        # Total events = 3, unique = 2
        assert vec[-2] == 3.0
        assert vec[-1] == 2.0
