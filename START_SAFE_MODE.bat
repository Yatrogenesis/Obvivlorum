@echo off
echo ========================================
echo    AI Symbiote System - Safe Mode
echo ========================================
echo.

cd /d "D:\Obvivlorum"

echo Starting AI Symbiote in safe mode (no persistence)...
echo.

REM Run without persistence installation to avoid permission issues
python ai_symbiote.py --user-id default --background

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to start AI Symbiote
    echo Error code: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo AI Symbiote started successfully in safe mode
echo The system will run without automatic persistence
echo.
pause