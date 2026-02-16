@echo off
REM Windows batch script to train the USG failure prediction model

echo ============================================================
echo USG FAILURE PREDICTION - MODEL TRAINING
echo ============================================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo ERROR: Virtual environment not activated!
    echo Please activate your virtual environment first:
    echo   venv\Scripts\activate
    echo.
    pause
    exit /b 1
)

REM Check if data file exists
if not exist "data\raw\USG_Data_cleared.csv" (
    echo ERROR: Data file not found!
    echo Please place USG_Data_cleared.csv in the data\raw\ directory
    echo.
    pause
    exit /b 1
)

echo Starting model training...
echo This may take 5-10 minutes depending on your hardware.
echo.

REM Run training script
python scripts\train_model.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo TRAINING SUCCESSFUL!
    echo ============================================================
    echo.
    echo Model saved to: models\model.pkl
    echo.
    echo Next steps:
    echo   1. Run SHAP analysis: scripts\shap.bat
    echo   2. Start API server: scripts\start_api.bat
    echo   3. View dashboard: cd frontend ^&^& npm run dev
    echo.
) else (
    echo.
    echo ============================================================
    echo TRAINING FAILED!
    echo ============================================================
    echo Please check the error messages above.
    echo.
)

pause
