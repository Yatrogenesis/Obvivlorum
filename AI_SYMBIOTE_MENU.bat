@echo off
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
echo   [0] Salir
echo.
echo ================================================
set /p opcion="Seleccione una opcion: "

if "%opcion%"=="1" goto SAFE_MODE
if "%opcion%"=="2" goto GUI_MODE
if "%opcion%"=="3" goto WEB_MODE
if "%opcion%"=="4" goto REPAIR
if "%opcion%"=="5" goto STATUS
if "%opcion%"=="6" goto ENABLE_PERSIST
if "%opcion%"=="7" goto DOCS
if "%opcion%"=="0" exit
goto MENU

:SAFE_MODE
echo.
echo Iniciando AI Symbiote en MODO SEGURO...
if exist "D:\Obvivlorum\DISABLE_PERSISTENCE.flag" (
    echo [OK] Persistencia desactivada - Iniciando de forma segura
)
python D:\Obvivlorum\ai_symbiote.py --no-persistence --safe-mode
pause
goto MENU

:GUI_MODE
echo.
echo Iniciando interfaz GUI...
start python D:\Obvivlorum\ai_symbiote_gui.py
echo GUI iniciada en ventana separada
timeout /t 3
goto MENU

:WEB_MODE
echo.
echo Iniciando servidor web...
cd /d D:\Obvivlorum\web\backend
start cmd /k "python main.py"
timeout /t 3
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
echo.
python D:\Obvivlorum\ai_symbiote.py --status
echo.
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
start notepad D:\Obvivlorum\PROYECTO_OBVIVLORUM_RESUMEN.md
goto MENU