@echo off
color 0A
title OBVIVLORUM AI - Real Claude Integration

echo.
echo ======================================================================
echo                      OBVIVLORUM AI - REAL CLAUDE INTEGRATION
echo ======================================================================
echo.
echo Features:
echo * Real Claude.ai integration via browser automation
echo * No API keys required - uses OAuth like Claude Code
echo * 100%% real responses from Claude
echo * Session persistence
echo * Modern GUI interface
echo * Cloudflare bypass through real browser
echo.
echo ======================================================================
echo.

cd /d "%~dp0"

echo Checking dependencies...
python -c "import playwright" 2>nul
if %errorlevel% neq 0 (
    echo Installing Playwright...
    pip install playwright
    echo Installing Chromium browser...
    playwright install chromium
)

echo Starting OBVIVLORUM AI with Real Claude Integration...
python obvivlorum_ai_real.py

if %errorlevel% neq 0 (
    echo.
    echo ======================================================================
    echo ERROR: Failed to start OBVIVLORUM AI
    echo ======================================================================
    echo.
    echo Possible solutions:
    echo 1. Install Python 3.8+ if not installed
    echo 2. Install dependencies: pip install -r requirements.txt
    echo 3. Install Playwright: pip install playwright ^&^& playwright install chromium
    echo.
    echo For support, check the README.md file or GitHub issues.
    echo ======================================================================
    echo.
    pause
    exit /b 1
)

echo.
echo OBVIVLORUM AI closed.
pause