@echo off
REM Sistema de Reparación de Windows
REM Creado: 2025-01-10
echo ========================================
echo    REPARACION DEL SISTEMA WINDOWS
echo ========================================
echo.

REM Verificar privilegios de administrador
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [!] Este script requiere permisos de administrador
    echo [!] Ejecutando con privilegios elevados...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo [+] Ejecutando con privilegios de administrador
echo.

echo [1] Verificando integridad del sistema...
sfc /scannow
echo.

echo [2] Reparando imagen de Windows...
dism /Online /Cleanup-Image /RestoreHealth
echo.

echo [3] Verificando disco...
chkdsk C: /f /r /x
echo.

echo [4] Limpiando archivos temporales...
del /q /f /s %TEMP%\* 2>nul
del /q /f /s C:\Windows\Temp\* 2>nul
echo.

echo [5] Reiniciando servicios criticos...
net stop wuauserv
net stop cryptSvc
net stop bits
net stop msiserver
net start wuauserv
net start cryptSvc
net start bits
net start msiserver
echo.

echo ========================================
echo    REPARACION COMPLETADA
echo ========================================
pause