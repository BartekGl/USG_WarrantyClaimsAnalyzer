@echo off
REM Windows batch script to verify setup

echo ============================================================
echo USG FAILURE PREDICTION - SETUP VERIFICATION
echo ============================================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo WARNING: Virtual environment not activated!
    echo Please activate it first: venv\Scripts\activate
    echo.
    echo Continuing anyway to check installation...
    echo.
)

REM Run verification script
python scripts\verify_setup.py

echo.
pause
