@echo off
title AI Symbiote - Ejecutar como Administrador
echo ========================================
echo   AI SYMBIOTE - PRIVILEGIOS ELEVADOS
echo ========================================
echo.

REM Verificar si ya se ejecuta como admin
net session >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Ejecutandose con privilegios de administrador
    goto :admin_confirmed
) else (
    echo [!] Se requieren privilegios de administrador
    echo Solicitando elevacion de privilegios...
    
    REM Intentar auto-elevarse
    powershell -Command "Start-Process '%~f0' -Verb RunAs" >nul 2>&1
    if %errorlevel% == 0 (
        echo Privilegios elevados, cerrando esta ventana...
        timeout /t 3 >nul
        exit
    ) else (
        echo ERROR: No se pudieron obtener privilegios de administrador
        echo.
        echo ALTERNATIVA: Ejecuta manualmente como administrador:
        echo 1. Click derecho en START_SYSTEM_FIXED.bat
        echo 2. Selecciona "Ejecutar como administrador"
        pause
        exit /b 1
    )
)

:admin_confirmed
echo.
echo [1/4] Configurando persistencia con privilegios elevados...
python "D:\Obvivlorum\FIX_PERSISTENCE.py"

echo [2/4] Creando tareas programadas del sistema...
schtasks /create /tn "AISymbiote_System" /tr "D:\Obvivlorum\START_SYSTEM_FIXED.bat" /sc onlogon /f >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Tarea programada del sistema creada
) else (
    echo [WARNING] No se pudo crear tarea programada
)

echo [3/4] Iniciando sistema con privilegios elevados...
cd /d "D:\Obvivlorum"
start /min cmd /c "python ai_symbiote.py --background --persistent"

echo [4/4] Iniciando interfaces...
timeout /t 3 /nobreak >nul
start /min cmd /c "cd /d D:\Obvivlorum\web\backend && python symbiote_server.py"
start "" python ai_symbiote_gui.py

echo.
echo [OK] Sistema iniciado con privilegios de administrador
echo Las interfaces deberian estar funcionando sin errores de permisos
echo.
pause