# CloudGuardian: AI-Powered Predictive Maintenance and Anomaly Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests Passing](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CloudGuardian** is an end-to-end, multimodal AIOps framework that bridges the gap between reactive static threshold monitoring (e.g., `CPU > 90%`) and proactive, calibrated failure prevention. It jointly models continuous Prometheus metrics and discrete server logs via Drain3 parsing, a Dilated Temporal Convolutional Network (TCN), Cross-Attention fusion, and Conformal Prediction.

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
│   ├── demonstrate_review1.py # Live demonstrator for Review 1
│   └── demonstrate_review2.py # Live demonstrator for Mid-Term Review 2
├── notebooks/                 # Interactive Jupyter Notebooks
│   └── milestone1_review_demo.ipynb
├── tests/                     # Automated Test Suite (19 passing unit tests)
│   ├── test_phase1.py         # Ingestion & Windowing tests
│   └── test_phase2.py         # Deep Learning Core & Multi-Task tests
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview & documentation
```

---

## Getting Started

### 1. Running All Automated Unit Tests (19 Passing Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected result:
```text
============================= 19 passed in 10.34s =============================
```

### 2. Running College Milestone 1 Demo (Review 1)
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review1
```

### 3. Running College Milestone 2 Demo (Mid-Term Review 2)
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review2
```
This script demonstrates:
- Deep Learning Architecture audit (193k parameters, 0.74 MB footprint, CPU serving).
- Unsupervised baseline training on normal operational traces.
- Reconstruction error spike (31.4x higher on anomalies!).
- Calibrated failure risk climbing to 99.2% on injected memory leak / DB saturation precursors.
- Saved multimodal cross-attention weights linking metric spikes to log error templates.

---

## Development Milestones

- [x] **Phase 0:** Research Proposal Synopsis & Literature Review (Approved).
- [x] **Phase 1:** Ingestion Engine, Regex Redactor, Drain3 Parser & 30s Tensor Windowing (Completed).
- [x] **Phase 2:** Hybrid Multimodal Model (Dilated TCN, Cross-Attention & 4 Multi-Task Heads) (Completed).
- [ ] **Phase 3:** Docker Compose Chaos Testbed, Conformal Calibration & 3-Gate Alerting.
- [ ] **Phase 4:** Saliency SHAP Explainability & Real-Time React/FastAPI Dashboard.
- [ ] **Phase 5:** 9-Rung Baseline Benchmarking, Leakage Audit & IEEE/IFIP DSN Conference Submission.
