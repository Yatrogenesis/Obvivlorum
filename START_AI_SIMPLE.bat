@echo off
title AI Symbiote - Real AI System
color 0A

echo.
echo  ╔═══════════════════════════════════════╗
echo  ║         AI Symbiote System            ║
echo  ║        REAL AI - NO MOCKS             ║
echo  ╚═══════════════════════════════════════╝
echo.

cd /d "D:\Obvivlorum"

echo [INFO] Starting AI Symbiote Core...
start /MIN python ai_symbiote.py --background --user-id web_user

echo [INFO] Starting Real AI Web Server...
cd web\backend
start /MIN python symbiote_server.py

echo [INFO] Waiting for services to initialize...
timeout /t 8 /nobreak >nul

echo [INFO] Opening Web Interface...
cd ..\..
start "" "D:\Obvivlorum\web\frontend\symbiote-chat.html"

echo.
echo ╔════════════════════════════════════════════════════╗
echo ║                SYSTEM READY                        ║
echo ╠════════════════════════════════════════════════════╣
echo ║  ✓ Real AI Conversational Engine                   ║
echo ║  ✓ Text-to-Speech Voice Output                     ║  
echo ║  ✓ Computer Vision (Face Detection)                ║
echo ║  ✓ WebSocket Real-time Communication               ║
echo ║  ✓ NO Mock Data or Placeholders                    ║
echo ╚════════════════════════════════════════════════════╝
echo.
echo Interface: http://localhost:8000
echo Chat: Already opened in browser
echo.
echo Try typing: "Hello, tell me about yourself"
echo The AI will respond with real intelligence!
echo.
pause