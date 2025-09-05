@echo off
REM Safe Start Script for AI Symbiote
REM This script allows manual control of the system

echo ===============================================
echo AI Symbiote - Safe Manual Start
echo ===============================================
echo.
echo WARNING: Persistence is DISABLED by default
echo.

if exist "DISABLE_PERSISTENCE.flag" (
    echo [!] Persistence is currently DISABLED
    echo [!] Remove DISABLE_PERSISTENCE.flag to enable persistence
    echo.
)

echo Starting AI Symbiote in SAFE MODE...
echo.

REM Start without persistence
python ai_symbiote.py --no-persistence --safe-mode

pause