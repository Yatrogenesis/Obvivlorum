@echo off
REM =====================================================
REM Obvivlorum - Lanzador Principal
REM =====================================================

echo ================================================================================
echo                            OBVIVLORUM SIMBIOSIS                                
echo ================================================================================
echo.
echo Este sistema requiere ejecutarse como ADMINISTRADOR
echo.

REM Verificar si somos admin
net session >nul 2>&1
if errorlevel 1 (
    echo Elevando permisos...
    echo.
    
    REM Crear un VBScript temporal para elevar permisos
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "cmd.exe", "/c cd /d ""%~dp0"" && ""%~s0"" %*", "", "runas", 1 >> "%temp%\getadmin.vbs"
    
    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /b
)

REM Ya somos admin
cd /d D:\Obvivlorum

:MENU
cls
echo ================================================================================
echo                            OBVIVLORUM SIMBIOSIS                                
echo                     Sistema WSL2 + Kali + HoloMem + AI                         
echo ================================================================================
echo.
echo Opciones disponibles:
echo.
echo 1. Sistema Unificado (PowerShell) - RECOMENDADO
echo 2. Verificador y Lanzador (Batch)
echo 3. Scripts individuales (1-6)
echo 4. Ver archivos del proyecto
echo 5. Salir
echo.
set /p choice=Selecciona opcion (1-5): 

if "%choice%"=="1" (
    powershell -ExecutionPolicy Bypass -File "Sistema_Unificado.ps1"
    pause
    goto MENU
)

if "%choice%"=="2" (
    if exist "VERIFICAR_Y_EJECUTAR.bat" (
        call VERIFICAR_Y_EJECUTAR.bat
    ) else (
        echo Archivo no encontrado!
        pause
    )
    goto MENU
)

if "%choice%"=="3" (
    cls
    echo Scripts disponibles:
    echo.
    dir /b *.PS1
    echo.
    set /p script=Escribe el numero del script (1-6): 
    if exist "%script%.PS1" (
        powershell -ExecutionPolicy Bypass -File "%script%.PS1"
    ) else (
        echo Script no encontrado!
    )
    pause
    goto MENU
)

if "%choice%"=="4" (
    cls
    echo Contenido del proyecto:
    echo.
    dir /b
    echo.
    pause
    goto MENU
)

if "%choice%"=="5" exit /b

goto MENU
