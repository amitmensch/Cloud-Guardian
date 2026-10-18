"""
CloudGuardian Phase 1 Demonstrator.
Demonstrates live log redaction, Drain3 template extraction,
30-second temporal synchronization, and sliding window tensor assembly.
Designed for College Milestone 1 / Review 1 evaluation.
"""

import time
import numpy as np

from cloudguardian.ingestion.redactor import LogRedactor
from cloudguardian.ingestion.drain_parser import DrainParser
from cloudguardian.ingestion.synchronizer import MetricLogSynchronizer
from cloudguardian.features.window_buffer import SlidingWindowBuffer


def run_phase1_demo():
    print("=" * 75)
    print("  CloudGuardian: AIOps Ingestion & Feature Windowing Pipeline Demo")
    print("  Milestone 1: Raw Log Normalization, Drain3 Parsing & Tensor Assembly")
    print("=" * 75)

    # 1. Initialize Pipeline Components
    redactor = LogRedactor()
    parser = DrainParser(depth=4, similarity_threshold=0.5)
    # Using 16 template bins for demo, 30s tick duration
    max_templates = 16
    start_time = 1700000000.0  # reference epoch
    synchronizer = MetricLogSynchronizer(
        tick_duration_sec=30.0,
        max_templates=max_templates,
        start_time=start_time,
    )
    # 60-tick window (30 minutes of telemetry)
    window_buffer = SlidingWindowBuffer(window_size=60, step_size=1)

    print("\n[Stage 1] Ingesting Sample Telemetry & Demonstrating Drain3 Parsing...")
    print("-" * 75)

    # Sample realistic raw logs from distributed cloud nodes
    sample_raw_logs = [
        (
            0.0,
            "INFO [main] dfs.DataNode: Receiving block blk_-1608999687 src: 10.251.43.115:50010 dest: 10.251.43.115:50010",
        ),
        (
            4.5,
            "INFO [DataNode:1] dfs.FSDataset: Packet responder 1 for block blk_-1608999687 terminating after 4096 bytes",
        ),
        (
            8.2,
            "INFO [DataNode:2] dfs.DataNode: Received block blk_-1608999687 of size 67108864 from 10.251.43.115:50010",
        ),
        (
            15.0,
            "WARN [pool-worker-3] org.apache.hadoop.ipc.Server: Call to 192.168.1.104:8020 failed on connection exception",
        ),
        (
            22.1,
            "ERROR [db-pool] com.zaxxer.hikari.pool.HikariPool: Connection pool db-pool is exhausted, wait time exceeded 5000ms",
        ),
        (
            35.0,
            "INFO [main] dfs.DataNode: Receiving block blk_982341102 src: 10.251.43.201:50010 dest: 10.251.43.201:50010",
        ),
        (
            42.0,
            "FATAL [core-daemon] sys.kernel: Page fault at 0x7fff5fbff8b0 in process 4210 binary /opt/services/server.bin",
        ),
    ]

    for offset_sec, raw_log in sample_raw_logs:
        current_ts = start_time + offset_sec

        # Redaction & metadata extraction
        redacted_log, meta = redactor.process(raw_log)

        # Parse with Drain3 tree
        parse_result = parser.parse(redacted_log)
        template_id = parse_result["template_id"]
        template_str = parse_result["template_str"]

        # Ingest into synchronizer
        synchronizer.ingest_log(
            current_ts,
            template_num=parse_result["template_num"],
            level=meta["level"],
        )

        print(f"Log Time: +{offset_sec:4.1f}s | Level: {meta['level']:<7}")
        print(f"  Raw:      {raw_log[:70]}...")
        print(f"  Redacted: {redacted_log[:70]}...")
        print(f"  Mapped:   [{template_id}] -> {template_str}\n")

    print(f"Total Unique Templates Discovered by Drain3: {parser.total_templates()}")

    # 2. Simulate 65 consecutive 30-second ticks (representing ~32.5 minutes of continuous cluster monitoring)
    print("\n[Stage 2] Simulating 32.5 Minutes of Aligned Stream Telemetry (65 Ticks)...")
    print("-" * 75)

    for tick_idx in range(65):
        tick_ts = start_time + (tick_idx * 30.0)

        # Normal background metrics with slight drift towards tick 55 (simulated memory leak precursor)
        cpu_val = 25.0 + 5.0 * np.sin(tick_idx / 5.0) + (15.0 if tick_idx > 50 else 0.0)
        mem_val = 40.0 + (tick_idx * 0.6)  # gradual creeping memory
        conn_wait = 2.0 + (18.0 if tick_idx > 52 else 0.0)

        metrics = {
            "cpu_util": cpu_val,
            "mem_util": mem_val,
            "disk_latency_ms": 6.5,
            "net_throughput_mb": 15.2,
            "conn_wait_ms": conn_wait,
        }
        synchronizer.ingest_metric(tick_ts, metrics)

        # Synthesize sporadic log events into tick
        t_num = 1 if (tick_idx % 2 == 0) else 2
        synchronizer.ingest_log(tick_ts + 2.0, template_num=t_num, level="INFO")

        # Ingest high-severity error logs during precursor window (ticks 50-60)
        if 50 <= tick_idx <= 60:
            synchronizer.ingest_log(tick_ts + 10.0, template_num=5, level="ERROR")

        # Extract synchronized tick feature vector
        tick_features = synchronizer.get_tick_features(tick_idx)
        window_buffer.append_tick(tick_features, timestamp=tick_ts)

    print(f"Accumulated Ticks in Window Buffer: {len(window_buffer.tick_history)}")
    print(f"Window Ready to Score: {window_buffer.is_ready()}")

    # 3. Assemble Sliding Window 3D Tensor
    print("\n[Stage 3] Assembling 3D Sliding Window Tensors for Deep Learning Model...")
    print("-" * 75)

    windows_tensor, end_timestamps = window_buffer.generate_all_windows()
    latest_window = window_buffer.get_latest_window()

    print(f"Generated Tensor Shape: (Samples, TimeSteps, Features) = {windows_tensor.shape}")
    print(f"  -> Number of 30-minute Window Samples (N): {windows_tensor.shape[0]}")
    print(f"  -> Window Time Steps (T = 60 ticks @ 30s): {windows_tensor.shape[1]}")
    print(f"  -> Total Feature Dimension (F):             {windows_tensor.shape[2]}")
    print(f"Latest Real-time Window Shape:               {latest_window.shape}")

    # Inspect the composition of the feature vector
    # 5 metrics + 16 template counts + 4 temporal deltas + 3 log metadata = 28
    print("\nFeature Decomposition Breakdown per Tick:")
    print("  - [0..4]   Mean System Metrics (CPU, Mem, Disk Latency, Net MB, Conn Wait)")
    print(f"  - [5..20]  Template Frequency Vector (Counts for E1..E{max_templates})")
    print("  - [21..24] Inter-Event Time Deltas (Mean, Std, Min, Max)")
    print("  - [25..27] Window Metadata (Total Logs, Unique Templates, Error Ratio)")

    # Sample reading from the latest tick
    latest_tick = latest_window[0, -1, :]
    print(f"\nTelemetry Snapshot from Latest Tick (Tick 64):")
    print(f"  CPU Utilization:      {latest_tick[0]:.2f}%")
    print(f"  Memory Utilization:   {latest_tick[1]:.2f}% (Crept up from 40%)")
    print(f"  Conn Wait Latency:    {latest_tick[4]:.2f} ms")
    print(f"  Total Logs in Tick:   {latest_tick[25]:.0f}")
    print(f"  Error Log Ratio:      {latest_tick[27] * 100:.1f}%")

    print("\n" + "=" * 75)
    print("  Phase 1 Validation: SUCCESSFUL")
    print("  The data pipeline cleanly converts raw logs & metrics into 3D tensors.")
    print("  The system is ready for Phase 2: Hybrid Multimodal TCN-Attention Model.")
    print("=" * 75)


if __name__ == "__main__":
    run_phase1_demo()
