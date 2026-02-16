@echo off
REM Windows batch script to start the FastAPI server

echo ============================================================
echo USG FAILURE PREDICTION - API SERVER
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

echo Starting FastAPI server...
echo.
echo API will be available at:
echo   - Local:   http://localhost:8000
echo   - Docs:    http://localhost:8000/docs
echo   - Health:  http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

pause
