@echo off
setlocal

echo ======================================================================
echo  RailGuard Vision - Laptop Edge Ingestion Agent (Windows)
echo ======================================================================

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

if "%RAILGUARD_CLOUD_URL%"=="" (
    set "RAILGUARD_CLOUD_URL=http://localhost:8000"
)

if "%RAILGUARD_AGENT_TOKEN%"=="" (
    set "RAILGUARD_AGENT_TOKEN=dev-agent-token-change-in-production"
)

echo Target Cloud Backend : %RAILGUARD_CLOUD_URL%
echo Video Source         : Default Webcam (0)
echo.

python run_agent.py --cloud-url "%RAILGUARD_CLOUD_URL%" --token "%RAILGUARD_AGENT_TOKEN%" --display

endlocal
