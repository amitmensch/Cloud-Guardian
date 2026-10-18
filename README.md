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
│   └── demo_phase1.py         # Live demonstrator for Review 1
├── tests/                     # Automated Test Suite
│   └── test_phase1.py         # Unit tests for Ingestion & Windowing
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview & documentation
```

---

## Getting Started

### 1. Activating the Environment
The project uses an isolated Python 3.11 virtual environment in `.venv`:

```powershell
# In PowerShell:
.\.venv\Scripts\activate
```

### 2. Running Automated Unit Tests
Verify that all parsing, normalization, and window tensor operations are functioning correctly:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected result:
```text
============================== 8 passed in 0.22s ==============================
```

### 3. Running College Milestone 1 Demo
You can run the interactive demonstrator either via terminal or as a Jupyter Notebook:

#### Option A: Terminal Live Demonstrator
```powershell
.\.venv\Scripts\python.exe -m cloudguardian.demonstrate_review1
```
This script will parse incoming logs, extract canonical HDFS templates ($E_1$ to $E_{29}$), align Prometheus metrics onto a 30-second grid, and construct and validate the 3D PyTorch tensor `torch.Size([32, 60, 42])`.

#### Option B: Interactive Jupyter Notebook
Open and run cell-by-cell in VS Code, Antigravity IDE, or Jupyter:
`notebooks/milestone1_review_demo.ipynb`

---

## Development Milestones

- [x] **Phase 0:** Research Proposal Synopsis & Literature Review (Approved).
- [x] **Phase 1:** Ingestion Engine, Regex Redactor, Drain3 Parser & 30s Tensor Windowing (Completed).
- [ ] **Phase 2:** Hybrid Multimodal Model (Dilated TCN, Cross-Attention & 4 Multi-Task Heads).
- [ ] **Phase 3:** Docker Compose Chaos Testbed, Conformal Calibration & 3-Gate Alerting.
- [ ] **Phase 4:** Saliency SHAP Explainability & Real-Time React/FastAPI Dashboard.
- [ ] **Phase 5:** 9-Rung Baseline Benchmarking, Leakage Audit & IEEE/IFIP DSN Conference Submission.
