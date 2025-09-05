@echo off
chcp 65001 >nul
color 0B
title AI Symbiote - Menu Principal

:MENU
cls
echo ================================================
echo         AI SYMBIOTE - MENU PRINCIPAL
echo ================================================
echo.
echo   Estado: PERSISTENCIA DESACTIVADA (Seguro)
echo   Version: 1.0 - Francisco Molina
echo.
echo ================================================
echo.
echo   [1] Iniciar Modo SEGURO (Sin Persistencia)
echo   [2] Iniciar GUI Desktop (Tkinter)
echo   [3] Iniciar Servidor Web (localhost:8000)
echo   [4] Reparar Sistema Windows (Admin)
echo   [5] Ver Estado del Sistema
echo   [6] Activar Persistencia (CUIDADO!)
echo   [7] Documentacion del Proyecto
echo   [8] Test Simple del Sistema
echo   [0] Salir
echo.
echo ================================================
set /p opcion="Seleccione una opcion [0-8]: "

if "%opcion%"=="1" goto SAFE_MODE
if "%opcion%"=="2" goto GUI_MODE
if "%opcion%"=="3" goto WEB_MODE
if "%opcion%"=="4" goto REPAIR
if "%opcion%"=="5" goto STATUS
if "%opcion%"=="6" goto ENABLE_PERSIST
if "%opcion%"=="7" goto DOCS
if "%opcion%"=="8" goto TEST
if "%opcion%"=="0" exit
goto MENU

:SAFE_MODE
echo.
echo Iniciando AI Symbiote en MODO SEGURO...
cd /d "D:\Obvivlorum"
if exist "DISABLE_PERSISTENCE.flag" (
    echo [OK] Persistencia desactivada - Iniciando de forma segura
)
powershell -Command "cd 'D:\Obvivlorum'; python ai_simple_working.py"
pause
goto MENU

:GUI_MODE
echo.
echo Iniciando interfaz GUI...
cd /d "D:\Obvivlorum"
start /min powershell -Command "cd 'D:\Obvivlorum'; python ai_symbiote_gui.py"
echo GUI iniciada en ventana separada
timeout /t 3 >nul
goto MENU

:WEB_MODE
echo.
echo Iniciando servidor web...
cd /d "D:\Obvivlorum\web\backend"
start cmd /k "python main.py"
timeout /t 3 >nul
start http://localhost:8000
echo Servidor web iniciado en http://localhost:8000
pause
goto MENU

:REPAIR
echo.
echo Iniciando reparacion del sistema (requiere Admin)...
powershell -Command "Start-Process 'D:\Obvivlorum\ADMIN_REPAIR.bat' -Verb RunAs"
pause
goto MENU

:STATUS
echo.
echo Verificando estado del sistema...
cd /d "D:\Obvivlorum"
echo.
powershell -Command "cd 'D:\Obvivlorum'; if (Test-Path 'DISABLE_PERSISTENCE.flag') { Write-Host '[OK] Persistencia: DESACTIVADA' -ForegroundColor Green } else { Write-Host '[!] Persistencia: ACTIVA' -ForegroundColor Yellow }; Write-Host 'Sistema: Obvivlorum v1.0'; Write-Host 'Estado: Bajo control manual'"
echo.
pause
goto MENU

:TEST
echo.
echo Ejecutando test simple del sistema...
cd /d "D:\Obvivlorum"
echo Verificando archivos principales...
if exist "ai_symbiote.py" (echo [OK] ai_symbiote.py) else (echo [ERROR] ai_symbiote.py no encontrado)
if exist "ai_symbiote_gui.py" (echo [OK] ai_symbiote_gui.py) else (echo [ERROR] ai_symbiote_gui.py no encontrado)  
if exist "ai_simple_working.py" (echo [OK] ai_simple_working.py) else (echo [ERROR] ai_simple_working.py no encontrado)
if exist "DISABLE_PERSISTENCE.flag" (echo [OK] Persistencia desactivada) else (echo [WARN] Persistencia podría estar activa)
echo.
echo Test completado
pause
goto MENU

:ENABLE_PERSIST
echo.
echo =========================================
echo   ADVERTENCIA: ACTIVAR PERSISTENCIA
echo =========================================
echo.
echo La persistencia fue desactivada porque el
echo sistema se reiniciaba agresivamente.
echo.
echo Activar solo si esta seguro!
echo.
set /p confirmar="Escriba SI para continuar: "
if /i "%confirmar%"=="SI" (
    del /f /q "D:\Obvivlorum\DISABLE_PERSISTENCE.flag" 2>nul
    echo Persistencia habilitada. Reinicie el sistema.
) else (
    echo Operacion cancelada
)
pause
goto MENU

:DOCS
echo.
echo Abriendo documentacion...
start notepad "D:\Obvivlorum\PROYECTO_OBVIVLORUM_RESUMEN.md"
goto MENU