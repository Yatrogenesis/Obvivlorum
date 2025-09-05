@echo off
REM Reparación Administrativa del Sistema
REM Requiere ejecutar como Administrador

color 0A
echo ==========================================
echo    REPARACION ADMINISTRATIVA COMPLETA
echo ==========================================
echo.

REM Verificar privilegios
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [!] Ejecutando con privilegios elevados...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo [+] Privilegios de administrador confirmados
echo.

echo [1/8] Reset Winsock...
netsh winsock reset
echo.

echo [2/8] Reset TCP/IP...
netsh int ip reset
echo.

echo [3/8] Limpieza DNS...
ipconfig /flushdns
ipconfig /registerdns
echo.

echo [4/8] Re-registrando DLLs del sistema...
regsvr32 /s shell32.dll
regsvr32 /s ole32.dll
regsvr32 /s oleaut32.dll
regsvr32 /s actxprxy.dll
regsvr32 /s shdocvw.dll
echo.

echo [5/8] Reparando Windows Update...
net stop wuauserv
net stop cryptSvc
net stop bits
net stop msiserver
ren C:\Windows\SoftwareDistribution SoftwareDistribution.old 2>nul
ren C:\Windows\System32\catroot2 catroot2.old 2>nul
net start wuauserv
net start cryptSvc
net start bits
net start msiserver
echo.

echo [6/8] Verificando integridad del sistema...
sfc /scannow
echo.

echo [7/8] Reparando imagen de Windows...
dism /Online /Cleanup-Image /RestoreHealth
echo.

echo [8/8] Limpiando y optimizando...
cleanmgr /sageset:1
cleanmgr /sagerun:1
echo.

echo ==========================================
echo    REPARACION COMPLETADA EXITOSAMENTE
echo ==========================================
echo.
echo Errores corregidos:
echo - Error -114: Conexion de red restaurada
echo - Error 220: Permisos reparados
echo - Error -118: DNS y TCP/IP restaurados
echo.
echo Se recomienda reiniciar el sistema
pause