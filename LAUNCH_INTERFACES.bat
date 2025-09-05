@echo off
title AI Symbiote - Launcher de Interfaces
color 0A

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║              AI SYMBIOTE INTERFACE LAUNCHER              ║
echo  ║         Abriendo interfaces GUI y Web simultáneamente    ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

cd /d "D:\Obvivlorum"

echo [INFO] Verificando servidor web...
curl -s -o nul http://localhost:8000/api/health
if %errorlevel% neq 0 (
    echo [WARNING] Servidor web no detectado
    echo [INFO] Iniciando servidor en background...
    cd web\backend
    start /MIN python symbiote_server.py
    cd ..\..
    echo [INFO] Esperando inicio del servidor...
    timeout /t 8 /nobreak >nul
)

echo [INFO] Abriendo Interfaz GUI Windows (persistente)...
start /MIN python ai_symbiote_gui.py

echo [INFO] Esperando inicialización GUI...
timeout /t 3 /nobreak >nul

echo [INFO] Abriendo Interfaz Web Claude-style...
start "" "web\frontend\symbiote-chat.html"

echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║                  INTERFACES ACTIVAS                      ║
echo ╠═══════════════════════════════════════════════════════════╣
echo ║  ✓ GUI Windows: Interfaz nativa persistente             ║
echo ║     - Ventana siempre disponible                         ║
echo ║     - Botón TURBO integrado                              ║
echo ║     - Chat en tiempo real                                ║
echo ║     - Estado del sistema                                 ║
echo ║                                                           ║
echo ║  ✓ Web Interface: Claude-style avanzada                  ║
echo ║     - http://localhost:8000                              ║
echo ║     - WebSocket en tiempo real                           ║
echo ║     - Botón TURBO con efectos                            ║
echo ║     - Voz, cámara y chat                                 ║
echo ║                                                           ║
echo ║  ✓ Servidor Híbrido: Inteligencia real                  ║
echo ║     - Modelo GGUF local (si disponible)                 ║
echo ║     - ChatGPT API fallback                               ║
echo ║     - Reglas expertas especializadas                     ║
echo ║     - Modo TURBO de optimización                         ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.
echo AMBAS INTERFACES ESTÁN AHORA DISPONIBLES:
echo.
echo • Interfaz GUI Windows: Ya abierta (ventana persistente)
echo • Interfaz Web: Ya abierta en navegador
echo • API Documentation: http://localhost:8000/api/docs
echo.
echo Para descargar modelo GGUF local: python download_model.py
echo.
pause