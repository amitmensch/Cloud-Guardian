# CloudGuardian: AI-Powered Predictive Maintenance and Anomaly Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests Passing](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ONNX CPU Serving](https://img.shields.io/badge/ONNX%20CPU-3.46ms%20p95-blueviolet.svg)]()
[![Explainability](https://img.shields.io/badge/XAI%20Hit--Rate%403-100%25-brightgreen.svg)]()

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
│   └── demonstrate_review4.py # Live demonstrator for Pre-Viva Review 4
├── models_onnx/               # Exported ONNX production models (gitignored binary)
│   └── .gitkeep
├── notebooks/                 # Interactive Jupyter Notebooks
│   └── milestone1_review_demo.ipynb
├── tests/                     # Automated Test Suite (32 passing unit tests)
│   ├── test_phase1.py         # Ingestion & Windowing tests (8 tests)
│   ├── test_phase2.py         # Deep Learning Core & Multi-Task tests (11 tests)
│   ├── test_phase3.py         # Calibration, Governance, Chaos & ONNX tests (5 tests)
│   └── test_phase4.py         # Explainability, Counterfactuals & Server tests (8 tests)
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview & documentation
```

---

## Getting Started

### 1. Running All Automated Unit Tests (32 Passing Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected result:
```text
tests/test_phase1.py ........                                            [ 25%]
tests/test_phase2.py ...........                                         [ 59%]
tests/test_phase3.py .....                                               [ 75%]
tests/test_phase4.py ........                                            [100%]
======================= 32 passed, 9 warnings in 17.77s =======================
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
Demonstrates:
- **Attention-Guided Feature Attribution:** Evaluates 42-channel contribution and proves **100% Hit-Rate@3** on injected chaos faults.
- **Counterfactual "What-If" Analysis:** Generates physical parameter targets (e.g., *"If DB wait drops from 1830ms to <20ms, failure risk drops from 98.4% to 4.5%"*).
- **Automated SRE Runbook Mapping:** Provides concrete operational recovery instructions for on-call engineers.
- **Full-Stack REST & WebSocket Verification:** Validates automated telemetry streaming and API health.

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
- [ ] **Phase 5:** 9-Rung Baseline Benchmarking, Leakage Audit & IEEE/IFIP DSN Conference Submission.
