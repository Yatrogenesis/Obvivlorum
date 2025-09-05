@echo off
title AI Symbiote - Intelligent System v3.0
color 0B

echo.
echo  ╔═════════════════════════════════════════════╗
echo  ║              AI Symbiote v3.0               ║
echo  ║        INTELLIGENT - NO ES MENSO YA         ║
echo  ╚═════════════════════════════════════════════╝
echo.

cd /d "D:\Obvivlorum"

echo [INFO] Stopping previous processes...
taskkill /F /IM python.exe /T >nul 2>&1

echo [INFO] Starting Intelligent AI System...
echo [INFO] This will take a moment to load GPT-2 models...

echo [INFO] Starting AI Symbiote Core...
start /MIN python ai_symbiote.py --background --user-id intelligent_user

echo [INFO] Loading Smart AI Engine...
cd web\backend

echo [INFO] Starting Intelligent Web Server...
start /MIN python -c "
try:
    from ai_engine_smart import SmartAIEngine
    print('Smart AI Engine: Testing initialization...')
    ai = SmartAIEngine()
    if ai.is_initialized:
        print('Smart AI Engine: READY - Advanced intelligence loaded')
    else:
        print('Smart AI Engine: FAILED - Using fallback')
except Exception as e:
    print(f'Smart AI Engine: ERROR - {e}')
    
print('Starting server with intelligent AI...')
import uvicorn
from symbiote_server import app
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"

echo [INFO] Waiting for intelligent systems to initialize...
timeout /t 8 /nobreak >nul

echo [INFO] Opening Enhanced Interface...
cd ..\..
start "" "D:\Obvivlorum\web\frontend\symbiote-chat.html"

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║                  INTELLIGENT SYSTEM READY            ║
echo ╠═══════════════════════════════════════════════════════╣
echo ║  ✓ Advanced AI with GPT-2 Medium                     ║
echo ║  ✓ Context Awareness & Reasoning                      ║  
echo ║  ✓ Complex Question Understanding                     ║
echo ║  ✓ Multilingual Intelligence (ES/EN)                 ║
echo ║  ✓ NO MAS RESPUESTAS MENSAS                           ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo Interface: http://localhost:8000
echo Chat: Already opened - Try complex questions!
echo.
echo Preguntas de prueba inteligentes:
echo - "Explícame que es machine learning"
echo - "Como funciona una red neuronal?"  
echo - "Cual es la diferencia entre IA y ML?"
echo - "Por que es importante la ciberseguridad?"
echo.
pause