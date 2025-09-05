@echo off
:: activar_virtualizacion.bat
:: Activa virtualizacion sin reiniciar o acceder a BIOS
:: Ejecutar como Administrador

echo ===========================================
echo  ACTIVADOR DE VIRTUALIZACION WINDOWS
echo ===========================================
echo.

:: Verificar permisos de administrador
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Ejecutando como Administrador
) else (
    echo [ERROR] Este script requiere permisos de Administrador
    echo.
    echo Cierra esta ventana y ejecuta como Administrador:
    echo   - Click derecho en el archivo
    echo   - Seleccionar "Ejecutar como administrador"
    pause
    exit /b 1
)

echo.
echo Activando caracteristicas de virtualizacion...
echo.

:: Activar Hyper-V (incluye virtualizacion)
echo [1/4] Activando Hyper-V...
dism.exe /online /enable-feature /featurename:Microsoft-Hyper-V-All /all /norestart >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Hyper-V activado
) else (
    echo [INFO] Hyper-V ya estaba activado o no disponible
)

:: Activar plataforma de virtualizacion
echo [2/4] Activando plataforma de virtualizacion...
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Plataforma de virtualizacion activada
) else (
    echo [INFO] Plataforma ya activada
)

:: Activar WSL
echo [3/4] Activando Windows Subsystem for Linux...
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] WSL activado
) else (
    echo [INFO] WSL ya estaba activado
)

:: Configurar servicios de virtualizacion
echo [4/4] Configurando servicios...
bcdedit /set hypervisorlaunchtype auto >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Hypervisor configurado para inicio automatico
)

echo.
echo ===========================================
echo  CONFIGURACION ADICIONAL
echo ===========================================
echo.

:: Habilitar virtualizacion anidada (para VMs dentro de WSL)
powershell -Command "Set-VMProcessor -VMName * -ExposeVirtualizationExtensions $true" >nul 2>&1

:: Establecer WSL2 como predeterminado
wsl --set-default-version 2 >nul 2>&1
echo [OK] WSL2 establecido como version predeterminada

echo.
echo ===========================================
echo  VERIFICACION DE ESTADO
echo ===========================================
echo.

:: Verificar estado actual
echo Verificando configuracion actual...
echo.

:: Verificar si CPU soporta virtualizacion
wmic cpu get VirtualizationFirmwareEnabled /value | findstr "TRUE" >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] CPU soporta virtualizacion
) else (
    echo [ADVERTENCIA] La virtualizacion puede requerir activacion en BIOS
    echo.
    echo Si el sistema lo requiere:
    echo   1. Reinicia el PC
    echo   2. Entra a BIOS (F2, F10, DEL o ESC segun tu PC)
    echo   3. Busca: Intel VT-x, AMD-V, SVM o Virtualization
    echo   4. Activa la opcion y guarda cambios
)

:: Verificar Hyper-V
sc query vmcompute >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Servicio Hyper-V detectado
) else (
    echo [INFO] Hyper-V no disponible (normal en Windows Home)
)

echo.
echo ===========================================
echo  RESULTADO
echo ===========================================
echo.
echo Todas las caracteristicas necesarias han sido configuradas.
echo.
echo IMPORTANTE: 
echo - Si todo muestra [OK], la virtualizacion esta lista
echo - Si aparece ADVERTENCIA, puede requerir reinicio
echo - Ejecuta el instalador de Simbiosis nuevamente
echo.
echo Presiona cualquier tecla para cerrar...
pause >nul