@echo off
title AI Symbiote - Sistema Completo
echo ========================================
echo   AI SYMBIOTE - INICIANDO SISTEMA
echo ========================================
echo.

REM Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python no está instalado o no está en PATH
    pause
    exit /b 1
)

REM Iniciar componentes
echo [1/3] Iniciando núcleo del sistema...
start /min cmd /c "cd /d D:\Obvivlorum && python ai_symbiote.py --background --persistent"
timeout /t 3 /nobreak >nul

echo [2/3] Iniciando servidor web...
start /min cmd /c "cd /d D:\Obvivlorum\web\backend && python symbiote_server.py"
timeout /t 3 /nobreak >nul

echo [3/3] Iniciando interfaz GUI...
start "" "D:\Obvivlorum\ai_symbiote_gui.py"

echo.
echo [OK] Sistema iniciado correctamente
echo.
echo Interfaces disponibles:
echo - GUI Desktop: Ventana activa
echo - Web: http://localhost:8000
echo - Chat: http://localhost:8000/chat
echo.
pause
