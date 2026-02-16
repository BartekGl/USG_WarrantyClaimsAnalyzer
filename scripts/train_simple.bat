@echo off
REM Simple training script using unified ml_core.py

echo ============================================================
echo USG FAILURE PREDICTION - SIMPLIFIED TRAINING
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

echo Running unified training pipeline...
echo This will take approximately 2-3 minutes.
echo.

REM Run the unified pipeline
python ml_core.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo TRAINING SUCCESSFUL!
    echo ============================================================
    echo.
    echo Model artifacts saved to models/ directory
    echo.
    echo Next steps:
    echo   1. Start API: scripts\start_api.bat
    echo   2. Run SHAP analysis: scripts\shap.bat
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
