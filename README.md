# CloudGuardian: AI-Powered Predictive Maintenance and Anomaly Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests Passing](https://img.shields.io/badge/tests-42%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ONNX CPU Serving](https://img.shields.io/badge/ONNX%20CPU-3.46ms%20p95-blueviolet.svg)]()
[![Explainability](https://img.shields.io/badge/XAI%20Hit--Rate%403-100%25-brightgreen.svg)]()
[![IEEE Manuscript](https://img.shields.io/badge/Paper-Camera--Ready-success.svg)](paper/main.tex)

**CloudGuardian** is an end-to-end, multimodal AIOps framework that bridges the gap between reactive static threshold monitoring (e.g., `CPU > 90%`) and proactive, calibrated failure prevention. It jointly models continuous Prometheus metrics and discrete server logs via Drain3 parsing, a Dilated Temporal Convolutional Network (TCN), Cross-Attention fusion, Conformal Prediction, Three-Gate Alert Governance, Saliency-Preserving SHAP Explainability, and an interactive Real-Time Cyberpunk Operations Dashboard.

---

## Academic Information
- **Institution:** Ajay Kumar Garg Engineering College (AKGEC), Ghaziabad
- **Affiliation:** Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow
- **Session:** 2023–2027 (Group 10)
- **Project Supervisor:** Dr. Shashank Sahu
- **Team Members:**
  - Ayush Kumar Agrawal (`2300270100043`): Ingestion & Drain3 Parsing Pipeline
  - Anubhav Hajela (`2300270100025`): Model Architecture (TCN, Attention, Multi-Task Heads)
  - Arpit Kumar (`2300270100033`): Calibration, Alert Governance & Chaos Testbed
  - Amit (`2300270100015`): Explainability Layer (SHAP/Counterfactuals), Dashboard & Paper

---

## Project Structure
```text
cloudguardian/
├── .venv/                     # Python 3.11 virtual environment
├── chaos_testbed/             # Controlled Fault Injection Environment
│   ├── docker-compose.yml     # 4-tier microservice cluster (Gateway, Orders, DB, Redis)
│   └── injectors/
│       └── chaos_simulator.py # 8 canonical failure modes (OOM, slow queries, conn exhaustion, etc.)
├── cloudguardian/             # Core Python Package
│   ├── ingestion/             # Module 1: Ingestion & Parsing
│   │   ├── redactor.py        # Regex normalizer (<IP>, <NUM>, <PATH>, <HEX>)
│   │   ├── drain_parser.py    # Fixed-depth parse tree Drain3 engine
│   │   └── synchronizer.py    # 30-second temporal grid synchronizer
│   ├── features/              # Module 2: Feature Engineering
│   │   └── window_buffer.py   # 60-tick (30-min) sliding window tensor builder
│   ├── models/                # Module 3: Deep Learning Multimodal Core
│   │   ├── tcn_encoder.py     # Dilated Causal TCN (Dilations: 1, 2, 4, 8, 16)
│   │   ├── log_encoder.py     # Template frequency & sequence encoder
│   │   ├── cross_attention.py # Bidirectional multimodal cross-attention
│   │   ├── heads.py           # Reconstruction, Forecast, Risk, and TTF heads
│   │   └── cloudguardian_model.py # Unified end-to-end architecture
│   ├── training/              # Module 4: Multi-Task Optimization
│   │   ├── loss.py            # Masked multi-task loss (MSE + Forecast + BCE + TTF)
│   │   └── trainer.py         # Training loop & checkpointing manager
│   ├── calibration/           # Module 5: Uncertainty & Governance
│   │   ├── conformal.py       # Conformal Prediction: empirical p-values & non-conformity
│   │   └── governance_gates.py# Three-Gate Alert Defense (FDR, Hysteresis, Cooldown)
│   ├── deployment/            # Module 6: Lightweight Production Serving
│   │   └── export_onnx.py     # ONNX INT8/Float export & CPU latency benchmarker
│   ├── explainability/        # Module 7: Saliency Attribution & Counterfactuals
│   │   ├── shap_explainer.py  # Fast attention-guided SHAP & Hit-Rate@3 evaluator
│   │   └── counterfactual.py  # What-If generator & actionable SRE runbooks
│   ├── server/                # Module 8: Full-Stack Serving & Operations Dashboard
│   │   ├── api.py             # FastAPI REST + WebSocket telemetry streaming engine
│   │   └── static/            # Glassmorphism Operations Center UI (HTML, CSS, JS)
│   ├── demonstrate_review1.py # Live demonstrator for Review 1
│   ├── demonstrate_review2.py # Live demonstrator for Mid-Term Review 2
│   ├── demonstrate_review3.py # Live demonstrator for Pre-Final Review 3
│   ├── demonstrate_review4.py # Live demonstrator for Pre-Viva Review 4
│   └── demonstrate_review5.py # Live demonstrator for Final Capstone Defense (Review 5)
├── experiments/               # Module 9: Benchmarking & Baseline Ladder
│   ├── baselines.py           # 8 comparative baseline monitoring paradigms
│   ├── benchmark_runner.py    # 9-rung evaluation harness & LaTeX table generator
│   ├── ablation_study.py      # Modular 6-variant ablation experiment runner
│   └── leakage_audit.py       # Formal temporal & session zero-leakage verifier
├── models_onnx/               # Exported ONNX production models (gitignored binary)
│   └── .gitkeep
├── paper/                     # Publication Manuscripts & Documentation
│   └── main.tex               # Camera-ready IEEEtran conference manuscript
├── notebooks/                 # Interactive Jupyter Notebooks
│   └── milestone1_review_demo.ipynb
├── tests/                     # Automated Test Suite (42 passing unit tests)
│   ├── test_phase1.py         # Ingestion & Windowing tests (8 tests)
│   ├── test_phase2.py         # Deep Learning Core & Multi-Task tests (11 tests)
│   ├── test_phase3.py         # Calibration, Governance, Chaos & ONNX tests (5 tests)
│   ├── test_phase4.py         # Explainability, Counterfactuals & Server tests (8 tests)
│   └── test_phase5.py         # Baselines, Benchmark, Ablations & Paper tests (10 tests)
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview & documentation
```

---

## Getting Started

### 1. Running All Automated Unit Tests (42 Passing Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected result:
```text
tests/test_phase1.py ........                                            [ 19%]
tests/test_phase2.py ...........                                         [ 45%]
tests/test_phase3.py .....                                               [ 57%]
tests/test_phase4.py ........                                            [ 76%]
tests/test_phase5.py ..........                                          [100%]
======================= 42 passed, 9 warnings in 16.70s =======================
```

### 2. Launching the Interactive Operations Dashboard
Start the FastAPI server and open your browser:
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.server.api
```
Open **`http://localhost:8000`** to view:
- **Real-Time Risk Gauge:** Animated circular SVG dial reflecting live failure probabilities.
- **Dual-Stream Synchronized Visualizer:** Canvas metric curves (CPU, RAM, Latency) synchronized with log event markers.
- **Explainability Drawer:** Top-3 root cause breakdown with animated percentage bars and plain-English diagnostics.
- **SRE Counterfactual Runbook:** Instant "What-If" remediation cards with physical parameter targets.
- **Chaos Fault Simulator Panel:** 8 interactive trigger buttons to inject live failures and watch detection in real time.
- **API Documentation:** Interactive Swagger docs at `http://localhost:8000/docs`.

### 3. Running College Review Demonstrators

#### Milestone 1 (Review 1):
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review1
```

#### Milestone 2 (Mid-Term Review 2):
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review2
```

#### Milestone 3 (Pre-Final Review 3):
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review3
```

#### Milestone 4 (Pre-Viva Review 4 — Explainability & Full-Stack):
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review4
```

#### Milestone 5 (Final Capstone Defense — Benchmarks & Manuscript):
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review5
```
Demonstrates:
- **Zero-Data-Leakage Audit:** Verifies strict chronological ordering and zero cross-split lookahead leakage.
- **9-Rung Baseline Comparison Ladder:** Benchmarks against Static Thresholds, Keyword Regex, STIDE, Isolation Forest, ARIMA, LSTM-AE, DeepLog, and USAD across 5 random seeds.
- **Systematic Ablation Study:** Validates the empirical necessity of multimodal fusion, cross-attention, and conformal calibration.
- **IEEE Conference Manuscript Audit:** Validates `paper/main.tex` readiness for submission.

---

## Empirical Benchmark Results (9-Rung Ladder)

| Monitoring Paradigm | Precision | Recall | F1-Score | Lead Time (min) | False Alarms/Day | p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Static Threshold (CPU > 85%, Latency > 400ms) | 0.412 | 0.820 | 0.548 | 1.2m | 48.2 | 0.12ms |
| 2. Keyword Pattern Matcher (Regex ERROR/FATAL) | 0.524 | 0.610 | 0.564 | 3.5m | 34.6 | 0.45ms |
| 3. STIDE (Log Template n-gram Mismatch) | 0.680 | 0.725 | 0.702 | 4.2m | 28.1 | 1.24ms |
| 4. Tabular Isolation Forest (Summary Stats) | 0.715 | 0.760 | 0.737 | 5.0m | 16.5 | 1.82ms |
| 5. ARIMA Auto-Regressive Residuals | 0.702 | 0.734 | 0.718 | 4.8m | 19.4 | 2.10ms |
| 6. LSTM Recurrent Autoencoder | 0.785 | 0.810 | 0.797 | 6.4m | 12.0 | 8.45ms |
| 7. DeepLog (Log-Only SOTA) | 0.832 | 0.805 | 0.818 | 7.1m | 9.8 | 6.20ms |
| 8. USAD (Metric-Only SOTA) | 0.854 | 0.840 | 0.847 | 8.3m | 8.2 | 4.80ms |
| **9. CloudGuardian (Proposed Multimodal Fused)** | **0.962** | **0.945** | **0.953** | **12.5m** | **0.8** | **3.46ms** |

---

## Architectural Ablation Study

| Ablation Variant | F1-Score | Lead Time (min) | False Alarms/Day | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Full CloudGuardian Pipeline** | **0.953** | **12.5m** | **0.8** | Optimal balance: maximum lead-time and sub-1 false alarm/day. |
| Ablation 1: Metrics-Only (No Logs) | 0.814 | 7.4m | 5.2 | Misses software-induced anomalies (deadlocks, thread starvation). |
| Ablation 2: Logs-Only (No Metrics) | 0.772 | 6.8m | 6.4 | Blind to silent hardware degradation (memory leaks without error logs). |
| Ablation 3: No Cross-Attention (Concat) | 0.865 | 8.9m | 3.6 | Weaker cross-modal coupling; loses attention-guided XAI interpretability. |
| Ablation 4: No Conformal Calibration | 0.838 | 11.8m | 14.8 | Fixed threshold is overly sensitive; false alarms surge 18.5x higher. |
| Ablation 5: No 3-Gate Alert Governance | 0.879 | 13.0m | 24.2 | Unfiltered single-tick glitches fire repeated alerts (alert storms). |

---

## Benchmark Performance vs. College Review Constraints

| Constraint / Benchmark | Allowed Project Budget | CloudGuardian Achieved | Status |
| :--- | :---: | :---: | :---: |
| **Model Size Footprint** | $< 50.0\text{ MB}$ | **$0.81\text{ MB}$** | **PASS (98.4% under budget)** |
| **CPU p95 Inference Latency** | $< 25.0\text{ ms}$ | **$3.46\text{ ms}$** | **PASS (7.2x faster)** |
| **Precursor Detection Lead Time** | $> 5.0\text{ mins}$ | **$12.5\text{ mins}$** | **PASS (Actionable early warning)** |
| **Explanation Hit-Rate@3** | $> 80.0\%$ | **$100.0\%$** | **PASS (Ground-truth attributed)** |
| **Counterfactual Risk Mitigation** | $> 70.0\%$ | **$93.9\%$ drop** | **PASS (SRE runbook mapped)** |
| **Transient Noise Filtering** | Suppress single-tick glitches | **100% Suppressed** | **PASS (Hysteresis Gate 2)** |
| **Alert Storm Elimination** | 0 duplicate alarms in cooldown | **100% Suppressed** | **PASS (Cooldown Gate 3)** |

---

## Development Milestones

- [x] **Phase 0:** Research Proposal Synopsis & Literature Review (Approved).
- [x] **Phase 1:** Ingestion Engine, Regex Redactor, Drain3 Parser & 30s Tensor Windowing (Completed).
- [x] **Phase 2:** Hybrid Multimodal Model (Dilated TCN, Cross-Attention & 4 Multi-Task Heads) (Completed).
- [x] **Phase 3:** Docker Compose Chaos Testbed, Conformal Calibration, 3-Gate Alerting & ONNX CPU Serving (Completed).
- [x] **Phase 4:** Saliency SHAP Explainability, Counterfactual Runbooks & Interactive Operations Dashboard (Completed).
- [x] **Phase 5:** 9-Rung Baseline Benchmarking, Leakage Audit & IEEE/IFIP DSN Conference Submission (Completed).
