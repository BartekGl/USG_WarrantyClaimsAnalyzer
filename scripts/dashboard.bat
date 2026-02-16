@echo off
REM Launch USG Mission Control Dashboard

echo ============================================================
echo USG MISSION CONTROL DASHBOARD
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
    echo   python ml_core.py
    echo.
    pause
    exit /b 1
)

REM Check if Streamlit is installed
python -c "import streamlit" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Streamlit not found. Installing dashboard dependencies...
    pip install -r dashboard_requirements.txt
    echo.
)

echo Starting Mission Control Dashboard...
echo.
echo Dashboard will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Launch Streamlit
streamlit run app.py

pause
