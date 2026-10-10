@echo off
setlocal enabledelayedexpansion

echo ====================================================================
echo  RailGuard Vision - Indian Railways Security Surveillance Prototype
echo  FOR HUMAN REVIEW ONLY - Does not detect explosives or narcotics
echo ====================================================================

cd /d "%~dp0"

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not found on PATH.
    echo Please install Python 3.10+ from python.org and re-run this script.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [SETUP] Creating Python virtual environment 'venv'...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SETUP] Virtual environment created successfully.
)

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Check and install dependencies
echo [SETUP] Verifying dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some dependencies failed to install via requirements.txt.
    echo Continuing to launch Streamlit...
)

REM Ensure directories exist
if not exist "data\alerts" mkdir data\alerts
if not exist "data\clips" mkdir data\clips
if not exist "data\ground_truth" mkdir data\ground_truth

echo ====================================================================
echo [LAUNCH] Starting RailGuard Vision Streamlit Dashboard...
echo Local URL: http://localhost:8501
echo Press Ctrl+C in this terminal window to terminate the server.
echo ====================================================================

streamlit run app\dashboard.py

pause
