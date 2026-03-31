@echo off
REM Start React Frontend Development Server on Port 3000
echo.
echo =====================================================
echo Starting Financial CNN React Frontend on port 3000...
echo =====================================================
echo.
cd /d "%~dp0"
node "node_modules\vite\bin\vite.js" --host 0.0.0.0 --port 3000
pause
