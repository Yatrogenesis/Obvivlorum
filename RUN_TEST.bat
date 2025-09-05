@echo off
REM Test Script for OBVIVLORUM-NEXUS Unified System

cd /d D:\Obvivlorum

echo ================================================================================
echo                 OBVIVLORUM-NEXUS UNIFIED SYSTEM TEST                          
echo ================================================================================
echo.
echo [*] Iniciando pruebas del sistema unificado...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en PATH
    echo Por favor instala Python 3.8 o superior
    pause
    exit /b 1
)

echo [+] Python detectado
echo.

REM Run the unified system test
echo [*] Ejecutando diagnosticos del sistema...
echo.

python unified_system.py --test

echo.
echo ================================================================================
echo                         TEST COMPLETADO                                        
echo ================================================================================
echo.
echo Presiona cualquier tecla para continuar...
pause >nul
