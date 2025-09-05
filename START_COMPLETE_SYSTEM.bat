@echo off
chcp 65001 >nul 2>&1
title AI Symbiote v3.0 - Sistema Completo Hibrido
color 0B

echo.
echo  ┌───────────────────────────────────────────────────────────────┐
echo  │                    AI SYMBIOTE v3.0                           │
echo  │            SISTEMA HIBRIDO COMPLETO - REAL IA                │
echo  └───────────────────────────────────────────────────────────────┘
echo.

cd /d "D:\Obvivlorum"

echo [INFO] Verificando dependencias...

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado. Instala Python 3.8+
    pause
    exit /b 1
)

echo [INFO] Instalando/actualizando dependencias...
pip install --quiet --upgrade fastapi uvicorn aiohttp requests tqdm

echo [INFO] Verificando modelo GGUF...
if not exist "models\*.gguf" (
    echo [WARNING] No se encontro modelo GGUF local
    echo [INFO] El sistema usara ChatGPT API como fallback
    echo [INFO] Para instalar modelo local, ejecuta: python download_model.py
)

echo.
echo [INFO] Iniciando sistema hibrido...
echo.

REM Start AI Symbiote core
echo [INFO] 1. Iniciando AI Symbiote Core...
start /MIN python ai_symbiote.py --background --user-id hybrid_user

REM Wait a moment
timeout /t 3 /nobreak >nul

echo [INFO] 2. Iniciando servidor web híbrido...
cd web\backend

REM Start hybrid server
start /MIN python symbiote_server.py

REM Wait for server to start
echo [INFO] Esperando que el servidor inicie...
timeout /t 8 /nobreak >nul

echo [INFO] 3. Abriendo interfaces...

REM Start GUI interface
cd ..\..
start /MIN python ai_symbiote_gui.py

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Open web interface  
start "" "D:\Obvivlorum\web\frontend\symbiote-chat.html"

echo.
echo ┌───────────────────────────────────────────────────────────────┐
echo │                    SISTEMA HIBRIDO LISTO                     │
echo ├───────────────────────────────────────────────────────────────┤
echo │  ✅ AI Symbiote Core activo                                  │
echo │  ✅ Servidor web hibrido en http://localhost:8000           │  
echo │  ✅ Interfaz GUI persistente abierta                        │
echo │  ✅ Interfaz web Claude-style disponible                    │
echo ║                                                               ║
echo │  CARACTERISTICAS HIBRIDAS:                                   │
echo │  • Modelo GGUF local (si esta disponible)                    │
echo │  • ChatGPT API gratuita (fallback automatico)               │
echo │  • Reglas inteligentes (ultimo recurso)                      │
echo │  • Modo TURBO (optimizacion de rendimiento)                  │
echo │  • Interfaz persistente estilo Visual Basic                  │
echo ║                                                               ║
echo │  COMANDOS ESPECIALES:                                        │
echo │  • "TURBO ON" - Activa modo alta velocidad                  │
echo │  • "TURBO OFF" - Desactiva modo turbo                       │
echo │  • Boton TURBO en ambas interfaces                          │
echo └───────────────────────────────────────────────────────────────┘
echo.
echo Interfaces disponibles:
echo • GUI Desktop: Ya abierta (persistente)
echo • Web Interface: http://localhost:8000 (ya abierta)
echo • API Documentation: http://localhost:8000/api/docs
echo.
echo Para instalar modelo GGUF local: python download_model.py
echo.
pause