@echo off
REM Start Flask Backend API Server
echo.
echo =====================================================
echo Starting Financial CNN Backend API...
echo =====================================================
echo.
cd /d "%~dp0"
python backend_api.py
pause
