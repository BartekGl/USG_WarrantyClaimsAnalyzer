# Quick Start Guide - Windows

Complete setup guide for Windows users to get the USG Failure Prediction system running in minutes.

## Prerequisites

- ✅ Python 3.10 or higher installed
- ✅ pip package manager
- ✅ Git (optional, for cloning)
- ✅ 4GB+ RAM available
- ✅ Node.js 18+ (for frontend dashboard - optional)

## Step-by-Step Setup

### 1. Download/Clone the Project

```powershell
# If you have git:
git clone https://github.com/BartekGl/ALK_DuzyProjekt.git
cd ALK_DuzyProjekt

# Or download ZIP from GitHub and extract
```

### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Your prompt should now show (venv)
```

**Important**: Always activate the virtual environment before running any scripts!

### 3. Install Dependencies

```powershell
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This will take 2-3 minutes and install ~50 packages including:
- XGBoost, LightGBM, scikit-learn (ML models)
- SHAP (explainability)
- FastAPI (API server)
- Optuna (hyperparameter tuning)
- Pandas, NumPy (data processing)

### 4. Prepare Your Data

Place your dataset in the correct location:

```powershell
# Create the directory if it doesn't exist
mkdir data\raw -Force

# Copy your data file
copy "C:\path\to\your\USG_Data_cleared.csv" data\raw\
```

**Required**: Your CSV file should have these columns:
- `Warranty_Claim` (target: Yes/No)
- `Batch_ID`
- `Assembly_Temp_C`
- `Humidity_Percent`
- `Solder_Temp_C`
- `Solder_Time_s`
- `Torque_Nm`
- `Gap_mm`
- `Region`
- ... and other production parameters

### 5. Train the Model

Simply run the training batch script:

```powershell
scripts\train.bat
```

**What this does:**
1. ✅ Creates necessary directories (`data/processed`, `models`, `reports`)
2. ✅ Loads and preprocesses your data (feature engineering)
3. ✅ Runs Optuna hyperparameter optimization (50 trials)
4. ✅ Trains XGBoost ensemble model (XGB + RF + LightGBM)
5. ✅ Evaluates performance with cross-validation
6. ✅ Saves model artifacts to `models/` directory

**Expected time**: 5-10 minutes (depending on your CPU)

**Output files:**
- `models/model.pkl` - Trained model
- `models/preprocessor.pkl` - Feature engineering pipeline
- `models/feature_names.json` - Feature metadata
- `data/processed/X_processed.csv` - Processed features
- `data/processed/y_target.csv` - Target labels
- `reports/metrics/evaluation_results.json` - Performance metrics
- `reports/metrics/best_hyperparameters.json` - Optimal parameters

### 6. Generate SHAP Explanations

Run the SHAP analysis script:

```powershell
scripts\shap.bat
```

**What this does:**
1. ✅ Loads trained model
2. ✅ Computes SHAP values for test set
3. ✅ Generates visualizations:
   - Summary plot (beeswarm)
   - Bar plot (feature importance)
   - Waterfall plots (example predictions)
4. ✅ Saves SHAP explainer for API use

**Expected time**: 2-3 minutes

**Output files:**
- `reports/visualizations/shap_summary_plot.png`
- `reports/visualizations/shap_bar_plot.png`
- `reports/visualizations/shap_waterfall_failure.png`
- `reports/visualizations/shap_waterfall_no_failure.png`
- `models/shap_explainer.pkl`

### 7. Start the API Server

```powershell
scripts\start_api.bat
```

The API will be available at:
- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **API base**: http://localhost:8000

**Test the API:**

Open your browser and go to http://localhost:8000/docs

Try the `/health` endpoint - you should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "shap_explainer_loaded": true
}
```

### 8. (Optional) Run the Dashboard

Open a **new PowerShell window**:

```powershell
cd frontend

# Install dependencies (first time only)
npm install

# Copy environment file
copy .env.example .env

# Start development server
npm run dev
```

The dashboard will open at: http://localhost:3000

## Common Issues & Solutions

### Issue: "Module not found" error

**Solution**: Make sure virtual environment is activated

```powershell
venv\Scripts\activate
```

Your prompt should show `(venv)` prefix.

### Issue: "Data file not found"

**Solution**: Verify file location

```powershell
# Check if file exists
dir data\raw\USG_Data_cleared.csv

# Should show your file, not "File Not Found"
```

### Issue: Training fails with "Cannot save file into a non-existent directory"

**Solution**: The script creates directories automatically, but if needed:

```powershell
mkdir data\processed -Force
mkdir models -Force
mkdir reports\visualizations -Force
mkdir reports\metrics -Force
```

### Issue: Import errors in notebooks

**Solution**: Use the standalone scripts instead of notebooks. The notebooks are for reference only. Use:
- `scripts\train.bat` instead of running notebooks manually

### Issue: "numpy.ndarray size changed" warning

**Solution**: This is just a warning, not an error. You can ignore it or reinstall numpy:

```powershell
pip install --force-reinstall numpy
```

### Issue: API doesn't start - port already in use

**Solution**: Another process is using port 8000. Either:

**Option 1**: Kill the other process
```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill it (replace PID with actual number)
taskkill /PID <PID> /F
```

**Option 2**: Use different port
```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

## Verify Installation

Run this quick test:

```powershell
# 1. Check Python version
python --version
# Should show: Python 3.10.x or higher

# 2. Check if packages are installed
pip list | findstr xgboost
pip list | findstr shap
pip list | findstr fastapi

# 3. Test imports
python -c "import xgboost; import shap; import fastapi; print('All imports OK')"
```

## Project Structure After Setup

```
ALK_DuzyProjekt/
├── data/
│   ├── raw/
│   │   └── USG_Data_cleared.csv    ✅ Your data file
│   └── processed/
│       ├── X_processed.csv          ✅ After training
│       └── y_target.csv             ✅ After training
├── models/
│   ├── model.pkl                    ✅ After training
│   ├── preprocessor.pkl             ✅ After training
│   ├── feature_names.json           ✅ After training
│   └── shap_explainer.pkl           ✅ After SHAP analysis
├── reports/
│   ├── visualizations/              ✅ After SHAP analysis
│   │   ├── shap_summary_plot.png
│   │   ├── shap_bar_plot.png
│   │   └── ...
│   └── metrics/                     ✅ After training
│       ├── evaluation_results.json
│       └── best_hyperparameters.json
├── scripts/
│   ├── train.bat                    📜 Run training
│   ├── shap.bat                     📜 Run SHAP analysis
│   ├── start_api.bat                📜 Start API server
│   ├── train_model.py               🐍 Training script
│   └── run_shap_analysis.py         🐍 SHAP script
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── api.py
└── venv/                            📦 Virtual environment
```

## Usage Workflow

**Complete workflow:**

```powershell
# 1. Activate environment (always first!)
venv\Scripts\activate

# 2. Train model (once, or when you get new data)
scripts\train.bat

# 3. Generate SHAP explanations (once, after training)
scripts\shap.bat

# 4. Start API server (keep running)
scripts\start_api.bat

# 5. (Optional) Start dashboard (in new terminal)
cd frontend
npm run dev
```

**Daily usage** (after initial setup):

```powershell
# Just start the API
venv\Scripts\activate
scripts\start_api.bat
```

## Performance Expectations

| Metric | Value |
|--------|-------|
| Training time | 5-10 minutes |
| SHAP analysis | 2-3 minutes |
| API startup | <5 seconds |
| Prediction latency | <100ms |
| Model F1 score | 75-85% |
| Model ROC-AUC | 85-92% |

## Next Steps

After successful setup:

1. **Review performance metrics**: Check `reports/metrics/evaluation_results.json`
2. **Examine SHAP plots**: Open PNG files in `reports/visualizations/`
3. **Test API**: Use interactive docs at http://localhost:8000/docs
4. **Read documentation**: See `docs/MODEL_CARD.md` for model details
5. **Business report**: See `docs/BUSINESS_REPORT.md` for ROI analysis

## Getting Help

**Before asking for help**, check:
1. ✅ Virtual environment is activated (`(venv)` in prompt)
2. ✅ All dependencies installed (`pip list` shows packages)
3. ✅ Data file exists at `data\raw\USG_Data_cleared.csv`
4. ✅ You're in the project root directory

**Common commands for debugging:**

```powershell
# Check what's installed
pip list

# Check Python path
python -c "import sys; print(sys.executable)"

# Verify imports work
python -c "import src.model; print('OK')"

# Check file structure
tree /F data
tree /F models
```

**For issues or questions:**
- GitHub Issues: https://github.com/BartekGl/ALK_DuzyProjekt/issues
- Review error messages carefully - they usually point to the problem
- Check logs in terminal output

## Tips for Success

1. **Always activate venv** before running any scripts
2. **Use batch scripts** (`.bat` files) - they're tested and reliable
3. **Don't modify notebooks** unless you know what you're doing - use scripts instead
4. **Keep data file in place** - don't move `USG_Data_cleared.csv` after setup
5. **Wait for training to complete** - don't interrupt the process
6. **Check health endpoint** before making predictions

## Troubleshooting Checklist

- [ ] Python 3.10+ installed?
- [ ] Virtual environment activated?
- [ ] Requirements installed? (`pip list | findstr xgboost`)
- [ ] Data file in correct location? (`dir data\raw\USG_Data_cleared.csv`)
- [ ] Directories created? (`mkdir data\processed models reports -Force`)
- [ ] Running from project root? (`pwd` should show ALK_DuzyProjekt)

---

**Last Updated**: January 2026
**Platform**: Windows 10/11
**Python**: 3.10+
