@echo off
REM =====================================================
REM Obvivlorum - Verificador y Lanzador del Sistema
REM =====================================================

echo ================================================================================
echo                            OBVIVLORUM SIMBIOSIS                                
echo                  Sistema de Instalacion WSL2 + Kali + HoloMem                  
echo ================================================================================
echo.

REM Verificar que estamos en el directorio correcto
cd /d D:\Obvivlorum
if errorlevel 1 (
    echo ERROR: No se puede acceder a D:\Obvivlorum
    pause
    exit /b 1
)

REM Verificar permisos de administrador
net session >nul 2>&1
if errorlevel 1 (
    echo ================================================================================
    echo   ATENCION: Este sistema requiere permisos de ADMINISTRADOR
    echo ================================================================================
    echo.
    echo   Por favor, ejecuta este archivo como Administrador:
    echo   1. Clic derecho en este archivo
    echo   2. Seleccionar "Ejecutar como administrador"
    echo.
    pause
    exit /b 1
)

:MENU
cls
echo ================================================================================
echo                            OBVIVLORUM SIMBIOSIS                                
echo                  Sistema de Instalacion WSL2 + Kali + HoloMem                  
echo ================================================================================
echo.
echo Estado del Sistema:
echo -------------------

REM Verificar WSL
wsl --status >nul 2>&1
if errorlevel 1 (
    echo [X] WSL: NO INSTALADO
    set WSL_STATUS=0
) else (
    echo [✓] WSL: INSTALADO
    set WSL_STATUS=1
)

REM Verificar Kali
wsl -l | findstr /i "kali-linux" >nul 2>&1
if errorlevel 1 (
    echo [X] Kali Linux: NO INSTALADO
    set KALI_STATUS=0
) else (
    echo [✓] Kali Linux: INSTALADO
    set KALI_STATUS=1
)

REM Verificar proyecto en Kali
if "%KALI_STATUS%"=="1" (
    wsl -d kali-linux -- test -d ~/obvlivorum_simbiosis 2>nul
    if errorlevel 1 (
        echo [X] Proyecto HoloMem: NO CONFIGURADO
        set HOLOMEM_STATUS=0
    ) else (
        echo [✓] Proyecto HoloMem: CONFIGURADO
        set HOLOMEM_STATUS=1
    )
) else (
    echo [-] Proyecto HoloMem: REQUIERE KALI
    set HOLOMEM_STATUS=0
)

echo.
echo ================================================================================
echo                              MENU PRINCIPAL                                    
echo ================================================================================
echo.
echo 1. Ejecutar instalacion completa (Scripts 1-6)
echo 2. Verificar estado detallado del sistema
echo 3. Ejecutar script individual
echo 4. Reparar instalacion
echo 5. Desinstalar todo
echo 6. Ver documentacion
echo 7. Salir
echo.
set /p choice=Selecciona una opcion (1-7): 

if "%choice%"=="1" goto INSTALL_ALL
if "%choice%"=="2" goto CHECK_DETAILED
if "%choice%"=="3" goto RUN_INDIVIDUAL
if "%choice%"=="4" goto REPAIR
if "%choice%"=="5" goto UNINSTALL
if "%choice%"=="6" goto DOCS
if "%choice%"=="7" goto END

echo Opcion invalida!
pause
goto MENU

:INSTALL_ALL
cls
echo ================================================================================
echo                         INSTALACION COMPLETA                                   
echo ================================================================================
echo.
echo Este proceso instalara:
echo - WSL2 y componentes necesarios
echo - Kali Linux
echo - Modulo HoloMem
echo - TinyLLaMA
echo - Sistema completo de simbiosis
echo.
echo ADVERTENCIA: Este proceso puede tomar 30-60 minutos
echo.
set /p confirm=Continuar? (S/N): 
if /i not "%confirm%"=="S" goto MENU

REM Ejecutar scripts en orden
for %%i in (1 2 3 4 5 6) do (
    echo.
    echo ================================================================================
    echo Ejecutando Script %%i...
    echo ================================================================================
    
    if exist "%%i.PS1" (
        powershell -ExecutionPolicy Bypass -File "%%i.PS1"
        if errorlevel 1 (
            echo.
            echo ERROR: Script %%i fallo. Revisa los mensajes de error arriba.
            pause
            goto MENU
        )
    ) else (
        echo ERROR: No se encuentra %%i.PS1
        pause
        goto MENU
    )
)

echo.
echo ================================================================================
echo                      INSTALACION COMPLETA EXITOSA!                             
echo ================================================================================
pause
goto MENU

:CHECK_DETAILED
cls
echo ================================================================================
echo                        VERIFICACION DETALLADA                                  
echo ================================================================================
echo.

echo 1. WSL y Virtualizacion:
echo ------------------------
wsl --status
echo.

echo 2. Distribuciones instaladas:
echo -----------------------------
wsl -l -v
echo.

if "%KALI_STATUS%"=="1" (
    echo 3. Estado del proyecto en Kali:
    echo --------------------------------
    wsl -d kali-linux -- bash -c "
        echo 'Directorio principal:'
        ls -la ~/obvlivorum_simbiosis/ 2>/dev/null || echo 'No existe'
        echo ''
        echo 'Modulo HoloMem:'
        lsmod | grep holomem || echo 'No cargado'
        echo ''
        echo 'Archivos del modulo:'
        ls -la ~/obvlivorum_simbiosis/holomem/ 2>/dev/null || echo 'No existen'
    "
)

echo.
pause
goto MENU

:RUN_INDIVIDUAL
cls
echo ================================================================================
echo                        EJECUTAR SCRIPT INDIVIDUAL                              
echo ================================================================================
echo.
echo Scripts disponibles:
echo 1. Instalar WSL2 y Kali
echo 2. Instalar HoloMem (preparacion)
echo 3. Generar archivos fuente HoloMem
echo 4. Instalar TinyLLaMA
echo 5. Integrar sistema
echo 6. Configuracion final
echo.
set /p script=Selecciona script (1-6): 

if exist "%script%.PS1" (
    powershell -ExecutionPolicy Bypass -File "%script%.PS1"
) else (
    echo Script no encontrado!
)
pause
goto MENU

:REPAIR
cls
echo ================================================================================
echo                          REPARAR INSTALACION                                   
echo ================================================================================
echo.
echo Opciones de reparacion:
echo 1. Reinstalar actualizacion del kernel WSL2
echo 2. Reparar permisos de Kali
echo 3. Recompilar modulo HoloMem
echo 4. Verificar y reparar todo
echo.
set /p repair_choice=Selecciona opcion (1-4): 

if "%repair_choice%"=="1" (
    echo Descargando e instalando actualizacion del kernel...
    powershell -Command "Invoke-WebRequest -Uri 'https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi' -OutFile '%TEMP%\wsl_update_x64.msi'"
    msiexec /i "%TEMP%\wsl_update_x64.msi" /quiet
    echo Completado!
)

if "%repair_choice%"=="2" (
    echo Reparando permisos...
    wsl -d kali-linux -- sudo apt update
    wsl -d kali-linux -- sudo usermod -aG sudo $(whoami)
    echo Completado!
)

if "%repair_choice%"=="3" (
    echo Recompilando modulo...
    wsl -d kali-linux -- bash -c "cd ~/obvlivorum_simbiosis/holomem && make clean && make"
    echo Completado!
)

if "%repair_choice%"=="4" (
    echo Ejecutando reparacion completa...
    REM Aqui iria la logica de reparacion completa
    echo En desarrollo...
)

pause
goto MENU

:UNINSTALL
cls
echo ================================================================================
echo                             DESINSTALAR TODO                                   
echo ================================================================================
echo.
echo ADVERTENCIA: Esto eliminara:
echo - El proyecto obvlivorum_simbiosis de Kali
echo - La distribucion Kali Linux (opcional)
echo - WSL2 (opcional)
echo.
echo Los datos no se pueden recuperar!
echo.
set /p confirm=Estas seguro? (S/N): 
if /i not "%confirm%"=="S" goto MENU

echo.
echo Que deseas desinstalar?
echo 1. Solo el proyecto obvlivorum_simbiosis
echo 2. Kali Linux completo
echo 3. Todo (WSL2 + Kali)
echo.
set /p uninstall_choice=Selecciona (1-3): 

if "%uninstall_choice%"=="1" (
    wsl -d kali-linux -- rm -rf ~/obvlivorum_simbiosis
    echo Proyecto eliminado.
)

if "%uninstall_choice%"=="2" (
    wsl --unregister kali-linux
    echo Kali Linux eliminado.
)

if "%uninstall_choice%"=="3" (
    wsl --unregister kali-linux
    dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux
    dism.exe /online /disable-feature /featurename:VirtualMachinePlatform
    echo WSL2 y Kali eliminados. Reinicia el sistema.
)

pause
goto MENU

:DOCS
cls
echo ================================================================================
echo                            DOCUMENTACION                                       
echo ================================================================================
echo.
echo OBVIVLORUM SIMBIOSIS
echo --------------------
echo.
echo Este sistema crea una integracion simbiotica entre:
echo - Windows (host)
echo - WSL2 (subsistema Linux)
echo - Kali Linux (distribucion de seguridad)
echo - HoloMem (modulo de kernel para memoria compartida)
echo - TinyLLaMA (modelo de IA local)
echo.
echo FLUJO DE INSTALACION:
echo 1. Script 1: Instala WSL2 y Kali Linux
echo 2. Script 2: Prepara el entorno y dependencias
echo 3. Script 3: Genera los archivos fuente del modulo HoloMem
echo 4. Script 4: Instala TinyLLaMA
echo 5. Script 5: Integra todos los componentes
echo 6. Script 6: Configuracion final y optimizacion
echo.
echo REQUISITOS:
echo - Windows 10/11 con virtualizacion habilitada en BIOS
echo - Minimo 8GB RAM (16GB recomendado)
echo - 20GB espacio libre en disco
echo - Conexion a Internet para descargas
echo.
echo SOLUCION DE PROBLEMAS:
echo - Si WSL no funciona: Habilita virtualizacion en BIOS
echo - Si Kali no inicia: Ejecutalo manualmente desde Microsoft Store primero
echo - Si el modulo no compila: Verifica que tengas los headers del kernel
echo.
pause
goto MENU

:END
echo.
echo Gracias por usar Obvivlorum Simbiosis!
echo.
exit /b 0
