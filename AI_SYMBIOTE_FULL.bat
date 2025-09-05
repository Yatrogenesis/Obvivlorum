@echo off
title AI SYMBIOTE - SISTEMA COMPLETO
color 0A

cls
echo.
echo     ===================================================
echo            AI SYMBIOTE - SISTEMA COMPLETO v1.0
echo     ===================================================
echo.
echo                   ┌─────────────┐
echo                   │  INICIANDO  │
echo                   │      🤖      │
echo                   └─────────────┘
echo.
echo     [*] Preparando entorno...
timeout /t 2 /nobreak >nul

:: Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    cls
    color 0C
    echo.
    echo     [ERROR] Python no encontrado
    echo.
    echo     Por favor instala Python 3.8+ desde:
    echo     https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Verificar Node.js
node --version >nul 2>&1
if errorlevel 1 (
    cls
    color 0C
    echo.
    echo     [ERROR] Node.js no encontrado
    echo.
    echo     Por favor instala Node.js 16+ desde:
    echo     https://nodejs.org/
    echo.
    pause
    exit /b 1
)

cls
echo.
echo     ===================================================
echo            AI SYMBIOTE - INICIANDO SERVICIOS
echo     ===================================================
echo.

:: 1. Iniciar AI Symbiote Core
echo     [1/4] Iniciando AI Symbiote Core...
start /min "AI Symbiote Core" cmd /c "cd /d %~dp0 && python ai_symbiote.py --user-id admin --background"
timeout /t 3 /nobreak >nul

:: 2. Iniciar Backend API
echo     [2/4] Iniciando Backend API...
start /min "AI Symbiote Backend" cmd /c "cd /d %~dp0web\backend && python main.py"
timeout /t 5 /nobreak >nul

:: 3. Verificar/Instalar dependencias frontend
echo     [3/4] Preparando Frontend...
if not exist "%~dp0web\frontend\node_modules" (
    echo           Instalando dependencias (primera vez)...
    cd /d "%~dp0web\frontend"
    call npm install --silent
)

:: 4. Iniciar Frontend
echo     [4/4] Iniciando Interfaz Web...
start /min "AI Symbiote Frontend" cmd /c "cd /d %~dp0web\frontend && npm run dev"
timeout /t 5 /nobreak >nul

:: Mostrar estado final
cls
color 0A
echo.
echo     ===================================================
echo              AI SYMBIOTE - SISTEMA ACTIVO
echo     ===================================================
echo.
echo     ✅ TODOS LOS SERVICIOS INICIADOS
echo.
echo     🌐 Interfaz Web:    http://localhost:3000
echo     📡 API Backend:     http://localhost:8000
echo     📚 Documentación:   http://localhost:8000/api/docs
echo.
echo     ===================================================
echo.
echo     Abriendo interfaz en 3 segundos...
timeout /t 3 /nobreak >nul

:: Abrir navegador
start http://localhost:3000

:: Crear archivo de estado
echo AI Symbiote activo desde %date% %time% > "%~dp0.symbiote_status"

:: Mantener ventana abierta pero minimizada
echo.
echo     ===================================================
echo     Sistema funcionando. Minimiza esta ventana.
echo     Para detener: Cierra esta ventana.
echo     ===================================================
echo.

:: Loop para mantener el proceso vivo
:loop
timeout /t 30 /nobreak >nul
if exist "%~dp0.symbiote_status" goto loop

:: Limpieza al cerrar
del "%~dp0.symbiote_status" 2>nul
exit