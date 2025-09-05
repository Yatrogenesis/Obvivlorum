@echo off
REM =====================================================
REM OBVIVLORUM-NEXUS UNIFIED SYSTEM LAUNCHER
REM =====================================================

cls
echo ================================================================================
echo                    OBVIVLORUM-NEXUS UNIFIED SYSTEM v2.0                       
echo ================================================================================
echo.
echo Sistema Unificado que integra:
echo - OBVIVLORUM: Framework Cuantico-Simbolico Original
echo - NEXUS AI: Motor Neuroplastico Avanzado  
echo - AION Protocol: Protocolos de Desarrollo
echo - AI Symbiote: Orquestador Adaptativo
echo.
echo Author: Francisco Molina
echo ORCID: https://orcid.org/0009-0008-6093-8267
echo.

REM Check admin privileges
net session >nul 2>&1
if errorlevel 1 (
    echo [!] Requiere privilegios de Administrador
    echo [*] Elevando permisos...
    
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\elevate.vbs"
    echo UAC.ShellExecute "cmd.exe", "/c cd /d ""%~dp0"" && ""%~s0"" %*", "", "runas", 1 >> "%temp%\elevate.vbs"
    
    "%temp%\elevate.vbs"
    del "%temp%\elevate.vbs"
    exit /b
)

cd /d D:\Obvivlorum

:MENU
cls
echo ================================================================================
echo                    OBVIVLORUM-NEXUS CONTROL CENTER                            
echo ================================================================================
echo.
echo SISTEMAS DISPONIBLES:
echo.
echo [1] Sistema Unificado (OBVIVLORUM + NEXUS) - RECOMENDADO
echo [2] Solo OBVIVLORUM (Framework Original)
echo [3] Solo NEXUS AI (Motor Neuroplastico)
echo [4] AI Symbiote (Sistema Legacy)
echo.
echo OPCIONES AVANZADAS:
echo.
echo [5] Test del Sistema Unificado
echo [6] Interfaz Web (http://localhost:8000)
echo [7] Modo Debug
echo [8] Ver Estado del Sistema
echo.
echo UTILIDADES:
echo.
echo [9] Instalar Dependencias
echo [10] Ver Logs
echo [11] Limpiar Cache
echo [12] Backup del Sistema
echo.
echo [0] Salir
echo.
set /p choice=Selecciona opcion [0-12]: 

if "%choice%"=="1" goto UNIFIED
if "%choice%"=="2" goto OBVIVLORUM_ONLY
if "%choice%"=="3" goto NEXUS_ONLY
if "%choice%"=="4" goto SYMBIOTE
if "%choice%"=="5" goto TEST_SYSTEM
if "%choice%"=="6" goto WEB_INTERFACE
if "%choice%"=="7" goto DEBUG_MODE
if "%choice%"=="8" goto SYSTEM_STATUS
if "%choice%"=="9" goto INSTALL_DEPS
if "%choice%"=="10" goto VIEW_LOGS
if "%choice%"=="11" goto CLEAR_CACHE
if "%choice%"=="12" goto BACKUP
if "%choice%"=="0" exit /b

goto MENU

:UNIFIED
echo.
echo ================================================================================
echo                    INICIANDO SISTEMA UNIFICADO                                
echo ================================================================================
echo.
echo [*] Verificando componentes...

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python no encontrado
    pause
    goto MENU
)

REM Check components
if not exist "unified_system.py" (
    echo [!] unified_system.py no encontrado
    pause
    goto MENU
)

echo [+] Componentes verificados
echo.
echo [*] Iniciando OBVIVLORUM-NEXUS...
echo.

python unified_system.py --mode unified

pause
goto MENU

:OBVIVLORUM_ONLY
echo.
echo [*] Iniciando OBVIVLORUM Framework...
if exist "ai_symbiote.py" (
    python ai_symbiote.py
) else (
    echo [!] ai_symbiote.py no encontrado
)
pause
goto MENU

:NEXUS_ONLY
echo.
echo [*] Iniciando NEXUS AI...
cd NEXUS_AI
if exist "nexus_core.py" (
    python nexus_core.py
) else (
    echo [!] nexus_core.py no encontrado
)
cd ..
pause
goto MENU

:SYMBIOTE
echo.
echo [*] Iniciando AI Symbiote (Legacy)...
if exist "ai_symbiote.py" (
    python ai_symbiote.py --status
    echo.
    set /p confirm=Iniciar AI Symbiote? (S/N): 
    if /i "%confirm%"=="S" python ai_symbiote.py
) else (
    echo [!] ai_symbiote.py no encontrado
)
pause
goto MENU

:TEST_SYSTEM
echo.
echo ================================================================================
echo                    TEST DEL SISTEMA UNIFICADO                                 
echo ================================================================================
echo.
python unified_system.py --test
pause
goto MENU

:WEB_INTERFACE
echo.
echo [*] Iniciando Interfaz Web...
echo [*] Abriendo navegador en http://localhost:8000
echo.
start "" http://localhost:8000

cd NEXUS_AI
if exist "nexus_web.py" (
    python nexus_web.py
) else (
    echo [!] nexus_web.py no encontrado
)
cd ..
pause
goto MENU

:DEBUG_MODE
echo.
echo [*] Iniciando en Modo Debug...
python unified_system.py --mode unified --debug
pause
goto MENU

:SYSTEM_STATUS
echo.
echo ================================================================================
echo                    ESTADO DEL SISTEMA                                         
echo ================================================================================
echo.

echo OBVIVLORUM Components:
if exist "ai_symbiote.py" echo [+] ai_symbiote.py - OK
if exist "AION\aion_core.py" echo [+] AION Protocol - OK
if exist "obvlivorum-implementation.py" echo [+] Obvivlorum Core - OK

echo.
echo NEXUS AI Components:
if exist "NEXUS_AI\nexus_core.py" echo [+] nexus_core.py - OK
if exist "NEXUS_AI\nexus_web.py" echo [+] nexus_web.py - OK

echo.
echo Unified System:
if exist "unified_system.py" echo [+] unified_system.py - OK
if exist "unified_system_state.json" (
    echo [+] Estado guardado encontrado
    type unified_system_state.json | findstr "timestamp"
)

echo.
echo Logs:
if exist "unified_system.log" echo [+] unified_system.log
if exist "ai_symbiote.log" echo [+] ai_symbiote.log
if exist "nexus_ai.log" echo [+] nexus_ai.log

echo.
pause
goto MENU

:INSTALL_DEPS
echo.
echo [*] Instalando dependencias...
echo.

REM Obvivlorum dependencies
pip install psutil requests cryptography numpy pandas

REM NEXUS dependencies
pip install fastapi uvicorn[standard] websockets torch torchvision

REM Additional
pip install aiofiles aiohttp python-dotenv pyyaml colorama rich

echo.
echo [+] Dependencias instaladas
pause
goto MENU

:VIEW_LOGS
echo.
echo Selecciona log:
echo 1. unified_system.log
echo 2. ai_symbiote.log
echo 3. nexus_ai.log
echo 4. Todos
echo.
set /p log=Opcion: 

if "%log%"=="1" if exist "unified_system.log" notepad unified_system.log
if "%log%"=="2" if exist "ai_symbiote.log" notepad ai_symbiote.log
if "%log%"=="3" if exist "nexus_ai.log" notepad nexus_ai.log
if "%log%"=="4" (
    if exist "unified_system.log" start notepad unified_system.log
    if exist "ai_symbiote.log" start notepad ai_symbiote.log
    if exist "nexus_ai.log" start notepad nexus_ai.log
)

goto MENU

:CLEAR_CACHE
echo.
echo [*] Limpiando cache y archivos temporales...

del /q *.log >nul 2>&1
del /q *.tmp >nul 2>&1
del /q *.bak >nul 2>&1
rmdir /s /q __pycache__ >nul 2>&1
rmdir /s /q .pytest_cache >nul 2>&1

echo [+] Cache limpiado
pause
goto MENU

:BACKUP
echo.
echo [*] Creando backup del sistema...

set backup_name=OBVIVLORUM_NEXUS_BACKUP_%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%.zip

powershell -Command "Compress-Archive -Path '*.py', '*.json', '*.md', 'AION', 'NEXUS_AI' -DestinationPath '%backup_name%'"

if exist "%backup_name%" (
    echo [+] Backup creado: %backup_name%
) else (
    echo [!] Error creando backup
)

pause
goto MENU
