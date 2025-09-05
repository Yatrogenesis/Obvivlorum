@echo off
title AI Symbiote - Diagnóstico de Emergencia
echo ============================================
echo   DIAGNOSTICO DE EMERGENCIA - AI SYMBIOTE
echo ============================================
echo.

echo [1] Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python no disponible
    goto :error
)

echo [2] Verificando archivos core...
if not exist "D:\Obvivlorum\ai_symbiote.py" (
    echo ERROR: ai_symbiote.py no encontrado
    goto :error
)

echo [3] Verificando dependencias...
python -c "import tkinter, fastapi; print('Dependencias OK')"
if %errorlevel% neq 0 (
    echo ERROR: Dependencias faltantes
    goto :error
)

echo [4] Verificando persistencia...
reg query "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run" | find "AISymbiote"
if %errorlevel% neq 0 (
    echo WARNING: Persistencia no configurada
)

echo.
echo ========== DIAGNOSTICO COMPLETO ==========
echo Sistema: OK
echo Para iniciar manualmente:
echo   python "D:\Obvivlorum\ai_symbiote.py"
echo   python "D:\Obvivlorum\ai_symbiote_gui.py"
echo ==========================================
goto :end

:error
echo.
echo ERROR: Sistema no funcional
echo Contacta soporte técnico

:end
pause
