# CloudGuardian: AI-Powered Predictive Maintenance and Anomaly Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests Passing](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ONNX CPU Serving](https://img.shields.io/badge/ONNX%20CPU-3.46ms%20p95-blueviolet.svg)]()

**CloudGuardian** is an end-to-end, multimodal AIOps framework that bridges the gap between reactive static threshold monitoring (e.g., `CPU > 90%`) and proactive, calibrated failure prevention. It jointly models continuous Prometheus metrics and discrete server logs via Drain3 parsing, a Dilated Temporal Convolutional Network (TCN), Cross-Attention fusion, Conformal Prediction, and Three-Gate Alert Governance.

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
│   ├── demonstrate_review1.py # Live demonstrator for Review 1
│   ├── demonstrate_review2.py # Live demonstrator for Mid-Term Review 2
│   └── demonstrate_review3.py # Live demonstrator for Pre-Final Review 3
├── models_onnx/               # Exported ONNX production models (gitignored binary)
│   └── .gitkeep
├── notebooks/                 # Interactive Jupyter Notebooks
│   └── milestone1_review_demo.ipynb
├── tests/                     # Automated Test Suite (24 passing unit tests)
│   ├── test_phase1.py         # Ingestion & Windowing tests
│   ├── test_phase2.py         # Deep Learning Core & Multi-Task tests
│   └── test_phase3.py         # Calibration, Governance, Chaos & ONNX tests
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview & documentation
```

---

## Getting Started

### 1. Running All Automated Unit Tests (24 Passing Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected result:
```text
tests/test_phase1.py ........                                            [ 33%]
tests/test_phase2.py ...........                                         [ 79%]
tests/test_phase3.py .....                                               [100%]
============================= 24 passed in 14.40s =============================
```

### 2. Running College Milestone 1 Demo (Review 1)
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review1
```

### 3. Running College Milestone 2 Demo (Mid-Term Review 2)
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review2
```
Demonstrates:
- Deep Learning Architecture audit (193k parameters, 0.74 MB footprint, CPU serving).
- Unsupervised baseline training on normal operational traces.
- Reconstruction error spike (31.4x higher on anomalies).
- Calibrated failure risk climbing to 99.2% on injected memory leak / DB saturation precursors.
- Saved multimodal cross-attention weights linking metric spikes to log error templates.

### 4. Running College Milestone 3 Demo (Pre-Final Review 3)
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review3
```
Demonstrates:
- **Conformal Prediction Calibration:** Statistical $p$-values ($p=0.5522$ on normal vs. $p=0.00498$ on fault) guaranteeing bounded false alarms with 99% confidence.
- **Three-Gate Alert Governance:**
  - *Gate 1 (Online FDR):* Dynamically bounds false discovery proportion.
  - *Gate 2 (Temporal Hysteresis $3$-of-$5$):* Suppresses single-point transient noise glitches.
  - *Gate 3 (Host Cooldown):* Suppresses duplicate alerts for 10 minutes to eliminate alert storms.
- **Controlled Chaos Engineering Testbed:** Simulates 8 real-world microservice failure classes with **12.5 minutes of actionable lead time** before database crash.
- **ONNX Production Optimization:** Exports model to **0.81 MB** and benchmarks CPU latency at **3.46 ms** (p95).

---

## Benchmark Performance vs. College Review Constraints

| Constraint / Benchmark | Allowed Project Budget | CloudGuardian Achieved | Status |
| :--- | :---: | :---: | :---: |
| **Model Size Footprint** | $< 50.0\text{ MB}$ | **$0.81\text{ MB}$** | **PASS (98.4% under budget)** |
| **CPU p95 Inference Latency** | $< 25.0\text{ ms}$ | **$3.46\text{ ms}$** | **PASS (7.2x faster)** |
| **Precursor Detection Lead Time** | $> 5.0\text{ mins}$ | **$12.5\text{ mins}$** | **PASS (Actionable early warning)** |
| **Transient Noise Filtering** | Suppress single-tick glitches | **100% Suppressed** | **PASS (Hysteresis Gate 2)** |
| **Alert Storm Elimination** | 0 duplicate alarms in cooldown | **100% Suppressed** | **PASS (Cooldown Gate 3)** |

---

## Development Milestones

- [x] **Phase 0:** Research Proposal Synopsis & Literature Review (Approved).
- [x] **Phase 1:** Ingestion Engine, Regex Redactor, Drain3 Parser & 30s Tensor Windowing (Completed).
- [x] **Phase 2:** Hybrid Multimodal Model (Dilated TCN, Cross-Attention & 4 Multi-Task Heads) (Completed).
- [x] **Phase 3:** Docker Compose Chaos Testbed, Conformal Calibration, 3-Gate Alerting & ONNX CPU Serving (Completed).
- [ ] **Phase 4:** Saliency SHAP Explainability & Real-Time React/FastAPI Dashboard.
- [ ] **Phase 5:** 9-Rung Baseline Benchmarking, Leakage Audit & IEEE/IFIP DSN Conference Submission.
