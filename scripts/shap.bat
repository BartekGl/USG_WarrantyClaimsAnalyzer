@echo off
REM Windows batch script to run SHAP analysis

echo ============================================================
echo USG FAILURE PREDICTION - SHAP ANALYSIS
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

REM Check if model exists
if not exist "models\model.pkl" (
    echo ERROR: Trained model not found!
    echo Please train the model first:
    echo   scripts\train.bat
    echo.
    pause
    exit /b 1
)

echo Running SHAP analysis...
echo This may take 2-3 minutes.
echo.

REM Run SHAP analysis
python scripts\run_shap_analysis.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SHAP ANALYSIS COMPLETE!
    echo ============================================================
    echo.
    echo Visualizations saved to: reports\visualizations\
    echo.
    echo You can now view the SHAP plots:
    echo   - reports\visualizations\shap_summary_plot.png
    echo   - reports\visualizations\shap_bar_plot.png
    echo   - reports\visualizations\shap_waterfall_failure.png
    echo.
) else (
    echo.
    echo ============================================================
    echo SHAP ANALYSIS FAILED!
    echo ============================================================
    echo Please check the error messages above.
    echo.
)

pause
