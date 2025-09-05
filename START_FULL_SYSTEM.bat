@echo off
echo ========================================
echo    AI Symbiote - Full Interactive System
echo ========================================
echo.

cd /d "D:\Obvivlorum"

echo [1/4] Installing Python dependencies...
echo.

REM Install basic requirements first
pip install fastapi uvicorn websockets python-multipart pydantic aiofiles >nul 2>&1

echo Dependencies installed.
echo.

echo [2/4] Starting AI Symbiote Core...
start /B python ai_symbiote.py --background --user-id web_user

timeout /t 3 /nobreak >nul

echo [3/4] Starting Web Server...
cd web\backend
start /B python symbiote_server.py

timeout /t 3 /nobreak >nul

echo [4/4] Opening Web Interface...
cd ..\..\
start "" "http://localhost:8000"

REM Also open the chat interface directly
timeout /t 2 /nobreak >nul
start "" "D:\Obvivlorum\web\frontend\symbiote-chat.html"

echo.
echo ========================================
echo    System Started Successfully!
echo ========================================
echo.
echo Web Interface: http://localhost:8000
echo Chat Interface: Open in your browser
echo.
echo Features available:
echo - Text chat with AI
echo - Voice commands (microphone required)
echo - Face recognition (camera required)
echo - Real-time responses
echo.
echo Press Ctrl+C to stop all services
echo.

REM Keep the window open
pause >nul