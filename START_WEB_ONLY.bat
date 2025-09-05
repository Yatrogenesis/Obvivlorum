@echo off
title AI Symbiote - Solo Web Interface
color 0B

echo ========================================
echo   AI SYMBIOTE - INTERFAZ WEB SOLAMENTE
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Iniciando Backend API...
if not exist "web\backend\main.py" (
    echo ❌ Backend no encontrado
    pause
    exit /b 1
)

start "AI Symbiote Backend" cmd /c "cd web\backend && python main.py"
echo ✅ Backend iniciado

timeout /t 3 /nobreak >nul

echo [2/2] Iniciando Frontend...
if not exist "web\frontend\package.json" (
    echo ❌ Frontend no encontrado
    pause
    exit /b 1
)

cd web\frontend

if not exist "node_modules" (
    echo 📦 Instalando dependencias frontend...
    call npm install
)

start "AI Symbiote UI" cmd /c "npm run dev"
echo ✅ Frontend iniciado

cd ..\..

echo.
echo ========================================
echo       INTERFAZ WEB INICIADA
echo ========================================
echo.
echo 🌐 Frontend: http://localhost:3000
echo 📡 Backend:  http://localhost:8000
echo 📚 API Docs: http://localhost:8000/api/docs
echo.
echo ⚠️  NOTA: Esta versión solo inicia la interfaz web.
echo    El core AI Symbiote debe iniciarse por separado.
echo.

echo Abriendo navegador en 5 segundos...
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo Interfaz web funcionando.
echo Mantén esta ventana abierta.
pause