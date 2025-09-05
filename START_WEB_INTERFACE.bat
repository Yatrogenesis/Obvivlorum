@echo off
title AI Symbiote Web Interface Launcher
color 0B

echo ========================================
echo    AI SYMBIOTE WEB INTERFACE LAUNCHER   
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 16+
    pause
    exit /b 1
)

echo [1/4] Starting Backend Server...
echo.

:: Start backend in new window
start "AI Symbiote Backend" cmd /k "cd /d %~dp0web\backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && python main.py"

:: Wait a bit for backend to start
timeout /t 5 /nobreak >nul

echo [2/4] Backend server starting on http://localhost:8000
echo.

echo [3/4] Starting Frontend Development Server...
echo.

:: Check if node_modules exists
if not exist "%~dp0web\frontend\node_modules" (
    echo Installing frontend dependencies...
    cd /d "%~dp0web\frontend"
    call npm install
)

:: Start frontend in new window
start "AI Symbiote Frontend" cmd /k "cd /d %~dp0web\frontend && npm run dev"

echo [4/4] Frontend server starting on http://localhost:3000
echo.

echo ========================================
echo    SERVERS STARTING...
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo API Docs:    http://localhost:8000/api/docs
echo Frontend UI: http://localhost:3000
echo.
echo Press any key to open the UI in your browser...
pause >nul

:: Open browser
start http://localhost:3000

echo.
echo Servers are running. Close this window to keep them running.
echo To stop servers, close the Backend and Frontend windows.
echo.
pause