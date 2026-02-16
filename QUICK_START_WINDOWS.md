# Quick Start Guide - Windows

Complete setup guide for Windows users to get the USG Failure Prediction system running locally.

## Prerequisites

- Python 3.10 or higher installed ([download](https://www.python.org/downloads/))
- pip package manager (included with Python)
- Git (optional, for cloning)
- 4GB+ RAM available
- Node.js 18+ (optional, only for the React frontend dashboard)

## Step 1 - Download or Clone the Project

```powershell
# Option A: Clone with git
git clone https://github.com/BartekGl/USG_WarrantyClaimsAnalyzer.git
cd USG_WarrantyClaimsAnalyzer

# Option B: Download ZIP from GitHub and extract, then cd into the folder
```

## Step 2 - Create and Activate the Virtual Environment

A Python virtual environment (`.venv`) is **required** to run this project. It isolates the project's dependencies from your system Python installation and prevents version conflicts with other projects.

```powershell
# Create the virtual environment (run once)
python -m venv .venv
```

After creating the virtual environment, **activate it**:

```powershell
.venv\Scripts\activate
```

Your terminal prompt should now show a `(.venv)` prefix, confirming the environment is active:

```
(.venv) PS C:\Users\you\USG_WarrantyClaimsAnalyzer>
```

> **Important**: You must activate the virtual environment **every time** you open a new terminal window before running any project commands. If you see `ModuleNotFoundError` or similar import errors, the most likely cause is a missing `.venv` activation.

### Why `.venv` and not `venv`?

The `.venv` naming convention (with the leading dot) keeps the environment directory hidden on Unix systems and is the Python community standard. Both names work on Windows, but `.venv` is recommended for cross-platform consistency.

## Step 3 - Install Dependencies

```powershell
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This installs ~50 packages including:
- XGBoost, LightGBM, scikit-learn (ML models)
- SHAP (model explainability)
- FastAPI, uvicorn (API server)
- Optuna (hyperparameter tuning)
- Pandas, NumPy (data processing)

### Verify the installation

```powershell
python -c "import xgboost; import shap; import fastapi; print('All imports OK')"
```

You can also run the verification script:

```powershell
scripts\verify.bat
```

## Step 4 - Prepare Your Data

Place your dataset CSV in the `data\raw\` directory:

```powershell
# Create the directory if it doesn't exist
mkdir data\raw -Force

# Copy your data file
copy "C:\path\to\your\USG_Data_cleared.csv" data\raw\
```

Verify the file is in place:

```powershell
dir data\raw\USG_Data_cleared.csv
```

**Required columns** in the CSV:
- `Warranty_Claim` (target column: Yes/No)
- Manufacturing parameters: `Assembly_Temp_C`, `Humidity_Percent`, `Solder_Temp_C`, `Solder_Time_s`, `Torque_Nm`, `Gap_mm`, `Batch_ID`, `Region`, and other production columns

## Step 5 - Train the Model

You have two training options:

### Option A: Simple training (recommended for getting started)

Uses the unified `ml_core.py` pipeline with optimized default hyperparameters. Fast and requires no tuning.

```powershell
python ml_core.py
```

Or use the batch script:

```powershell
scripts\train_simple.bat
```

This takes approximately 2-3 minutes and produces:
- `models\model.pkl` - Trained XGBoost model
- `models\preprocessor.pkl` - Feature engineering pipeline
- `models\feature_names.json` - Feature metadata

### Option B: Full training with hyperparameter optimization

Uses Optuna for hyperparameter search and trains an ensemble (XGBoost + Random Forest + LightGBM). Higher performance but slower.

```powershell
scripts\train.bat
```

This takes 5-10 minutes and additionally produces:
- `data\processed\X_processed.csv` - Processed features
- `data\processed\y_target.csv` - Target labels
- `reports\metrics\evaluation_results.json` - Performance metrics
- `reports\metrics\best_hyperparameters.json` - Optimal parameters

See `SIMPLE_TRAINING_GUIDE.md` for a detailed comparison of both approaches.

## Step 6 - Generate SHAP Explanations (Optional)

SHAP analysis generates feature importance visualizations that explain model predictions:

```powershell
scripts\shap.bat
```

Output (saved to `reports\visualizations\`):
- `shap_summary_plot.png` - Beeswarm summary
- `shap_bar_plot.png` - Feature importance ranking
- `shap_waterfall_failure.png` - Example failure explanation
- `shap_waterfall_no_failure.png` - Example non-failure explanation
- `models\shap_explainer.pkl` - Saved explainer for API use

## Step 7 - Start the API Server

With the model trained, start the FastAPI prediction server:

```powershell
scripts\start_api.bat
```

Or run uvicorn directly:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Interactive docs (Swagger)**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **API base**: http://localhost:8000

Test it by opening http://localhost:8000/health in your browser. You should see:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "shap_explainer_loaded": true
}
```

> **Note**: Keep this terminal window open while the API is running. Use a new terminal for the steps below.

## Step 8 - Start the Streamlit Dashboard (Optional)

The Streamlit dashboard provides an analytics interface for exploring model performance and SHAP explanations.

Open a **new PowerShell window**, then:

```powershell
cd USG_WarrantyClaimsAnalyzer
.venv\Scripts\activate

# Install additional dashboard dependencies (first time only)
pip install -r dashboard_requirements.txt

# Start the dashboard
scripts\dashboard.bat
```

Or run directly:

```powershell
streamlit run app.py
```

The dashboard opens at: http://localhost:8501

## Step 9 - Start the React Frontend Dashboard (Optional)

The React dashboard provides a modern web UI for uploading data and viewing predictions. Requires Node.js 18+.

Open a **new PowerShell window**, then:

```powershell
cd USG_WarrantyClaimsAnalyzer\frontend

# Install Node.js dependencies (first time only)
npm install

# Start the development server
npm run dev
```

The frontend opens at: http://localhost:3000

> **Prerequisite**: The FastAPI backend (Step 7) must be running for the React dashboard to work.

---

## Quick Reference - Daily Usage

Once the initial setup is complete, starting the application daily requires just:

```powershell
# Terminal 1: Start the API
cd USG_WarrantyClaimsAnalyzer
.venv\Scripts\activate
scripts\start_api.bat

# Terminal 2 (optional): Start the React dashboard
cd USG_WarrantyClaimsAnalyzer\frontend
npm run dev
```

## Common Issues and Solutions

### "Module not found" or import errors

The virtual environment is not activated. Activate it first:

```powershell
.venv\Scripts\activate
```

Your prompt must show the `(.venv)` prefix.

### "Data file not found"

Verify the file is in the expected location:

```powershell
dir data\raw\USG_Data_cleared.csv
```

### Training fails with "Cannot save file into a non-existent directory"

The training scripts create directories automatically, but if needed:

```powershell
mkdir data\processed -Force
mkdir models -Force
mkdir reports\visualizations -Force
mkdir reports\metrics -Force
```

### API doesn't start - port already in use

Another process is using port 8000.

**Option 1**: Find and stop the other process:

```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Option 2**: Run on a different port:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

### "numpy.ndarray size changed" warning

This is a harmless warning, not an error. You can ignore it or reinstall numpy:

```powershell
pip install --force-reinstall numpy
```

### Virtual environment was created with `venv` instead of `.venv`

If you previously created the environment as `venv\`, you can either:

1. Continue using `venv\Scripts\activate` (it works the same), or
2. Delete and recreate with the standard name:

```powershell
rmdir /s /q venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Verify Installation

```powershell
# Check Python version (3.10+ required)
python --version

# Check key packages are installed
pip list | findstr xgboost
pip list | findstr shap
pip list | findstr fastapi

# Test imports
python -c "import xgboost; import shap; import fastapi; print('All imports OK')"

# Check model files exist (after training)
dir models\model.pkl
dir models\preprocessor.pkl
```

## Project Structure After Setup

```
USG_WarrantyClaimsAnalyzer/
├── .venv/                              Virtual environment (required)
├── data/
│   ├── raw/
│   │   └── USG_Data_cleared.csv        Your data file
│   └── processed/                      Generated after training
├── models/
│   ├── model.pkl                       Generated after training
│   ├── preprocessor.pkl                Generated after training
│   ├── feature_names.json              Generated after training
│   └── shap_explainer.pkl              Generated after SHAP analysis
├── reports/
│   ├── visualizations/                 Generated after SHAP analysis
│   └── metrics/                        Generated after training
├── scripts/
│   ├── train.bat                       Full training (Optuna + ensemble)
│   ├── train_simple.bat                Simple training (ml_core.py)
│   ├── start_api.bat                   Start FastAPI server
│   ├── dashboard.bat                   Start Streamlit dashboard
│   ├── shap.bat                        Generate SHAP analysis
│   └── verify.bat                      Verify setup
├── src/
│   ├── api.py                          FastAPI server
│   ├── preprocessing.py                Feature engineering
│   ├── model.py                        Model training
│   └── evaluation.py                   Metrics and validation
├── frontend/                           React dashboard (optional)
├── ml_core.py                          Unified training pipeline
├── app.py                              Streamlit dashboard
├── requirements.txt                    Python dependencies
└── dashboard_requirements.txt          Streamlit dependencies
```

## Performance Expectations

| Metric | Value |
|--------|-------|
| Simple training time | 2-3 minutes |
| Full training time | 5-10 minutes |
| SHAP analysis | 2-3 minutes |
| API startup | < 5 seconds |
| Prediction latency | < 100 ms |
| Model F1 score | 0.78 - 0.85 |
| Model ROC-AUC | 0.85 - 0.93 |

## Next Steps

After setup is complete:

1. Explore the API docs at http://localhost:8000/docs
2. Review model metrics in `reports\metrics\evaluation_results.json`
3. Examine SHAP plots in `reports\visualizations\`
4. Read `docs\DEPLOYMENT_GUIDE.md` for production deployment options
5. See `DASHBOARD_GUIDE.md` for Streamlit dashboard usage

---

**Platform**: Windows 10/11 | **Python**: 3.10+ | **Last Updated**: February 2026
