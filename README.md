# USG Failure Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# VET-GUARD — AI-Powered Warranty Claims Analyzer for USG Devices

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Academic project** — Final project for the *AI in Business* postgraduate program, Group B.  
> VET-GUARD is a production-grade machine learning system for predicting warranty claim failures in ultrasound (USG) medical device manufacturing. It achieves **75–85% F1 score** with full SHAP interpretability, enabling an estimated **60–85% reduction in warranty costs**.

---

## 📌 Project Overview

**Business Problem: VET-EYE S.A., a veterinary ultrasound manufacturer selling 676 units annually, faces a 10% warranty failure rate — 89% of claims tied to LCD screen failures — generating $143,616 in yearly quality costs and eroding competitiveness against global leaders (GE, Esaote) and aggressive Chinese entrants (Mindray) in a $543M market.
**Solution: VET-GUARD is an on-premise AI system using XGBoost and BalancedRandomForest on 2,027 production records (42 variables) that identified two root causes — cable supplier Cables-X (22.9% failure rate) and soldering time >4.7s — enabling a shift from reactive to predictive quality control. With a $41K investment and $15K/year maintenance, the project delivers payback in under 2 years (IRR 42–137%), while building a proprietary "Data Moat" of process-defect knowledge that competitors cannot replicate.

**Key results:**
- ✅ ~75% reduction in warranty costs ($155K+ annual savings)
- ✅ <100 ms real-time inference latency
- ✅ Full transparency via SHAP explanations (explainability-by-design)
- ✅ ROI approx. 153% in the first year

---

## 🏗️ Architecture & Project Structure

```
USG_WarrantyClaimsAnalyzer/
├── frontend/                        # React 18 + TypeScript dashboard (Vite)
│   ├── src/
│   │   ├── components/dashboard/    # Charts, heatmaps, prediction stream
│   │   ├── pages/                   # Landing, Upload, Dashboard pages
│   │   ├── stores/                  # Zustand state management
│   │   ├── utils/                   # API client, animations
│   │   └── types/                   # TypeScript type definitions
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
│
├── src/                             # Python ML backend
│   ├── api.py                       # FastAPI inference service
│   ├── preprocessing.py             # Feature engineering pipeline
│   ├── model.py                     # XGBoost ensemble + Optuna tuning
│   └── evaluation.py               # Metrics, cross-validation, reports
│
├── notebooks/                       # Step-by-step analysis (Jupyter)
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_SHAP_Analysis.ipynb
│
├── data/
│   ├── raw/                         # Source data (USG_Data_cleared.csv)
│   └── synthetic/                   # Synthetic training data
│
├── models/                          # Saved model artifacts
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── production_model.pkl
│   ├── production_preprocessor.pkl
│   └── feature_names.json
│
├── scripts/                         # Convenience batch scripts (Windows)
│   ├── train.bat                    # Full training (Optuna + ensemble)
│   ├── train_simple.bat             # Simple quick training
│   ├── start_api.bat                # Start FastAPI server
│   ├── dashboard.bat                # Start Streamlit dashboard
│   ├── shap.bat                     # Run SHAP analysis
│   └── verify.bat                   # Verify installation
│
├── docs/
│   └── DEPLOYMENT_GUIDE.md
│
├── ml_core.py                       # Unified single-file training pipeline
├── app.py                           # Streamlit analytics dashboard
├── analytics_engine.py
├── business_analytics.py
├── executive_dashboard.py
├── production_inference.py
├── generate_training_data.py
├── train_production_model.py
│
├── requirements.txt                 # Python dependencies
├── dashboard_requirements.txt       # Streamlit extras
├── environment.yml                  # Conda environment
├── Dockerfile                       # Backend container
├── docker-compose.yml               # Backend only
└── docker-compose.full.yml          # Full stack (frontend + backend)
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| RAM | 4 GB | 8 GB recommended for Optuna tuning |
| Node.js | 18+ | Only for the React frontend |
| Docker | 20.10+ | Only for containerised deployment |

---

### Option A — Local (Python only, recommended for development)

```bash
# 1. Clone the repository
git clone https://github.com/BartekGl/USG_WarrantyClaimsAnalyzer.git
cd USG_WarrantyClaimsAnalyzer

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows PowerShell

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import xgboost; import shap; import fastapi; print('All imports OK')"
# or use the helper script (Windows):
scripts\verify.bat
```

---

### Option B — Docker (recommended for production / demo)

**Full stack (API + React dashboard):**
```bash
docker-compose -f docker-compose.full.yml up -d

# Services:
# React Dashboard   → http://localhost:3000
# FastAPI backend   → http://localhost:8000
# Swagger UI        → http://localhost:8000/docs
```

**Backend only:**
```bash
docker-compose up -d
# or
docker build -t usg-vetguard .
docker run -d -p 8000:8000 --name usg-api usg-vetguard
```

---

## 📊 Data Preparation

Place the source CSV in `data/raw/`:

```bash
# Linux / macOS
cp /path/to/USG_Data_cleared.csv data/raw/

# Windows
copy "C:\path\to\USG_Data_cleared.csv" data\raw\
```

**Required columns in the CSV:**

| Column | Description |
|---|---|
| `Warranty_Claim` | Target variable (Yes / No) |
| `Assembly_Temp_C` | Assembly environment temperature |
| `Humidity_Percent` | Relative humidity during assembly |
| `Solder_Temp_C` | Soldering iron temperature |
| `Solder_Time_s` | Soldering contact duration |
| `Torque_Nm` | Torque applied during assembly |
| `Gap_mm` | Component gap measurement |
| `Batch_ID` | Production batch identifier |
| `Region` | Destination region (EU, US, APAC, …) |

The dataset also includes supplier codes (including the flagged **Cable-X** LCD supplier) and additional production parameters — 44 columns total before feature engineering.

---

## 🧠 Model Training

### Simple training (fast, ~2–3 min)

Uses the unified `ml_core.py` pipeline with optimised defaults:

```bash
python ml_core.py
# Windows: scripts\train_simple.bat
```

Produces:
- `models/model.pkl` — trained XGBoost model
- `models/preprocessor.pkl` — feature pipeline
- `models/feature_names.json`

### Full training with Optuna (5–10 min)

Trains an ensemble (XGBoost + Random Forest + LightGBM) with automatic hyperparameter search:

```bash
python scripts/train_model.py
# Windows: scripts\train.bat
```

Additional outputs:
- `data/processed/` — engineered feature matrices
- `reports/metrics/evaluation_results.json`
- `reports/metrics/best_hyperparameters.json`

### Jupyter notebooks (step-by-step exploration)

```bash
jupyter notebook
# Run in order: 01 → 02 → 03 → 04
```

---

## 🔍 SHAP Explainability

Generate feature importance visualisations:

```bash
python scripts/run_shap_analysis.py
# Windows: scripts\shap.bat
```

Output (saved to `reports/visualizations/`):
- `shap_summary_plot.png` — global beeswarm summary
- `shap_bar_plot.png` — ranked feature importance
- `shap_waterfall_failure.png` — example failure explanation
- `shap_waterfall_no_failure.png` — example non-failure explanation
- `models/shap_explainer.pkl` — saved explainer for API use

---

## 📡 Starting the API

```bash
# Using uvicorn directly
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Windows batch shortcut
scripts\start_api.bat
```

Health check: `http://localhost:8000/health`
Interactive docs (Swagger): `http://localhost:8000/docs`

---

## 📊 Dashboards

### Streamlit (analytics / internal use)

```bash
# Install additional dependencies (first time only)
pip install -r dashboard_requirements.txt

streamlit run app.py
# Windows: scripts\dashboard.bat
# Opens at: http://localhost:8501
```

### React frontend (modern web UI)

Requires Node.js 18+ and a running FastAPI backend.

```bash
cd frontend
npm install
npm run dev
# Development server at: http://localhost:3000
```

---

## 📡 API Reference

### Single prediction with SHAP explanation

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "device_data": {
      "Batch_ID": "BATCH_001",
      "Assembly_Temp_C": 22.5,
      "Humidity_Percent": 45.0,
      "Solder_Temp_C": 350.0,
      "Solder_Time_s": 3.2,
      "Torque_Nm": 2.5,
      "Gap_mm": 0.15,
      "Region": "EU"
    },
    "include_shap": true,
    "threshold": 0.5
  }'
```

Response:
```json
{
  "prediction": "No",
  "probability": 0.23,
  "confidence": 0.77,
  "threshold": 0.5,
  "shap_values": {
    "Supplier_A_Failure_Rate": -0.15,
    "Batch_Failure_Rate": -0.12
  },
  "timestamp": "2026-02-16T10:30:45"
}
```

### Batch prediction (Python)

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "devices": [
            {"Batch_ID": "BATCH_001", "Assembly_Temp_C": 22.5, ...},
            {"Batch_ID": "BATCH_002", "Assembly_Temp_C": 23.1, ...}
        ],
        "include_shap": False,
        "threshold": 0.5
    }
)
results = response.json()
print(f"Predicted failures: {results['predicted_failures']}/{results['total_devices']}")
```

---

## 🤖 ML Details

### Model architecture

VET-GUARD uses a **soft-voting ensemble** of three classifiers:

| Model | Role | Weight |
|---|---|---|
| XGBoost | Primary classifier (Optuna-tuned) | 2 |
| Random Forest | 300 trees, balanced class weights | 1 |
| LightGBM | 300 estimators, lr=0.05 | 1 |

Post-training calibration is applied via **Platt scaling** (5-fold cross-validation) to produce reliable probability estimates.

### Feature engineering

Starting from **44 raw columns**, the pipeline engineers **60+ features** including:

- **Batch aggregates** — batch age, failure rate, batch size
- **Interaction terms** — `Assembly_Temp × Humidity`, `Solder_Temp × Solder_Time`, `Torque × Gap`
- **Supplier encoding** — failure rate per supplier (flags Cable-X systematically)
- **Anomaly scores** — Isolation Forest scores on environmental parameters
- **Time-series patterns** — chronological trends from serial numbers

### Class imbalance handling

- **SMOTE** — synthetic oversampling of the minority class
- **Dynamic class weights** — `scale_pos_weight` calculated per dataset
- **Threshold optimisation** — via precision-recall curve analysis

### Performance

| Metric | Score |
|---|---|
| F1 Score | 0.75 – 0.85 |
| Precision | 0.70 – 0.80 |
| Recall | 0.75 – 0.85 |
| ROC-AUC | 0.85 – 0.92 |
| PR-AUC | 0.70 – 0.80 |
| Inference latency | < 100 ms |

---

## 📈 Key Findings

SHAP analysis identified the top failure predictors:

1. **Supplier failure rate encoding** — Cable-X (LCD supplier) shows 15–30% failure rates; flagged as primary root cause
2. **Batch quality indicators** — batch-level failure rate and size are highly predictive
3. **Environmental conditions** — Temperature × Humidity interactions, especially outside 20–23°C / 40–50% RH
4. **Solder process parameters** — specific temperature × time combinations correlate strongly with defects
5. **Anomaly scores** — devices with unusual parameter combinations are at significantly elevated risk

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```bash
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
MODEL_PATH=models/model.pkl
PREPROCESSOR_PATH=models/preprocessor.pkl
LOG_LEVEL=INFO
```

---

## 🔧 Troubleshooting

**"Module not found" / import errors**
The virtual environment is not activated. Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/macOS). Your prompt must show the `(.venv)` prefix.

**Port 8000 already in use**
```bash
# Find and kill the process (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or run on a different port
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

**Training fails — missing directory**
```bash
mkdir -p data/processed models reports/visualizations reports/metrics
# Windows: mkdir data\processed, models, reports\visualizations, reports\metrics -Force
```

**`numpy.ndarray size changed` warning**
Harmless. Ignore or reinstall: `pip install --force-reinstall numpy`

See [QUICK_START_WINDOWS.md](QUICK_START_WINDOWS.md) for full Windows-specific troubleshooting.

---

## 🛠️ Technology Stack

**ML & Data**
- XGBoost 2.0+, LightGBM 4.0+, scikit-learn 1.3+
- SHAP 0.44+, Optuna 3.4+, imbalanced-learn 0.11+
- Pandas, NumPy, Plotly, Matplotlib, Seaborn

**API & Backend**
- FastAPI 0.104+, Uvicorn 0.24+, Pydantic 2.4+

**Frontend**
- React 18, TypeScript, Vite, Tailwind CSS, Framer Motion, Zustand

**Dashboards**
- Streamlit (analytics view), React + nginx (production UI)

**Infrastructure**
- Docker, Docker Compose, Conda / pip virtual environments

---

## 📚 Additional Documentation

- [QUICK_START_WINDOWS.md](QUICK_START_WINDOWS.md) — Step-by-step Windows setup guide
- [SIMPLE_TRAINING_GUIDE.md](SIMPLE_TRAINING_GUIDE.md) — Comparison of training approaches
- [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) — Streamlit dashboard usage
- [frontend/README.md](frontend/README.md) — React dashboard setup
- [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) — Production deployment

---

## 🔬 Regulatory Context

VET-GUARD was designed with awareness of the following regulatory frameworks (analysed as part of the academic project):

- **EU AI Act** — classified as high-risk AI system (Annex III, medical devices)
- **MDR 2017/745** — EU Medical Device Regulation
- **GDPR** — data minimisation, purpose limitation, processing records
- **ISO 13485** — quality management for medical devices

Full compliance analysis is available in the accompanying academic thesis document.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Authors

**Group B** — *AI in Business* postgraduate program  
Project: *Digital Transformation of Quality Control using AI for Medical Device Manufacturers*

---

## 🙏 Acknowledgments

- XGBoost: Chen & Guestrin (2016)
- SHAP: Lundberg & Lee (2017)
- SMOTE: Chawla et al. (2002)
- Optuna: Akiba et al. (2019)

---

**Last updated:** February 2026

Last Updated: January 2026


