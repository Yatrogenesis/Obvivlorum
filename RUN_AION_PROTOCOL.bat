@echo off
REM =====================================================
REM AION Protocol Launcher
REM =====================================================

echo ================================================================================
echo                             AION PROTOCOL v2.0                                 
echo ================================================================================
echo.

REM Verificar que estamos en el directorio correcto
cd /d D:\Obvivlorum\AION
if errorlevel 1 (
    echo ERROR: No se puede acceder a D:\Obvivlorum\AION
    pause
    exit /b 1
)

:MENU
cls
echo ================================================================================
echo                             AION PROTOCOL v2.0                                 
echo                  Sistema Maestro de Desarrollo de Activos Digitales            
echo ================================================================================
echo.
echo Opciones disponibles:
echo.
echo 1. Inicializar AION Protocol
echo 2. Ejecutar Protocol ALPHA (Desarrollo Científico-Disruptivo)
echo 3. Ejecutar Protocol BETA (Aplicaciones Móviles)
echo 4. Ejecutar Protocol GAMMA (Software Enterprise/Desktop)
echo 5. Ejecutar Protocol DELTA (Aplicaciones Web)
echo 6. Ejecutar Protocol OMEGA (Comercialización)
echo 7. Crear puente con Obvivlorum
echo 8. Ver historial de ejecuciones
echo 9. Análisis de rendimiento
echo 10. Salir
echo.
set /p choice=Selecciona opcion (1-10): 

if "%choice%"=="1" (
    echo.
    echo Inicializando AION Protocol...
    python aion_cli.py init --config config.json
    pause
    goto MENU
)

if "%choice%"=="2" (
    echo.
    echo Ejecutando Protocol ALPHA...
    echo.
    set /p params=Parámetros (JSON o ruta de archivo): 
    python aion_cli.py execute --protocol ALPHA --params "%params%"
    pause
    goto MENU
)

if "%choice%"=="3" (
    echo.
    echo Ejecutando Protocol BETA...
    echo.
    set /p params=Parámetros (JSON o ruta de archivo): 
    python aion_cli.py execute --protocol BETA --params "%params%"
    pause
    goto MENU
)

if "%choice%"=="4" (
    echo.
    echo Ejecutando Protocol GAMMA...
    echo.
    set /p params=Parámetros (JSON o ruta de archivo): 
    python aion_cli.py execute --protocol GAMMA --params "%params%"
    pause
    goto MENU
)

if "%choice%"=="5" (
    echo.
    echo Ejecutando Protocol DELTA...
    echo.
    set /p params=Parámetros (JSON o ruta de archivo): 
    python aion_cli.py execute --protocol DELTA --params "%params%"
    pause
    goto MENU
)

if "%choice%"=="6" (
    echo.
    echo Ejecutando Protocol OMEGA...
    echo.
    set /p params=Parámetros (JSON o ruta de archivo): 
    python aion_cli.py execute --protocol OMEGA --params "%params%"
    pause
    goto MENU
)

if "%choice%"=="7" (
    echo.
    echo Integración con Obvivlorum...
    echo.
    echo Esta funcionalidad requiere acceso programático.
    echo Por favor, utilice la API de Python directamente.
    echo.
    pause
    goto MENU
)

if "%choice%"=="8" (
    echo.
    echo Historial de ejecuciones:
    echo.
    set /p limit=Número de entradas a mostrar (10): 
    if "%limit%"=="" set limit=10
    python aion_cli.py history --limit %limit%
    pause
    goto MENU
)

if "%choice%"=="9" (
    echo.
    echo Análisis de rendimiento:
    echo.
    python aion_cli.py performance
    pause
    goto MENU
)

if "%choice%"=="10" exit /b

goto MENU
