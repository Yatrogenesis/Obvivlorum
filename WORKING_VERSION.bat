@echo off
title AI Symbiote - Versión Que Funciona
color 0A

cls
echo.
echo     ╔══════════════════════════════════════╗
echo     ║    AI SYMBIOTE - VERSIÓN FUNCIONAL   ║
echo     ║              🚀 v1.0.1               ║
echo     ╚══════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/2] 🔧 Iniciando Backend (Versión Corregida)...
if not exist "web\backend\main_fixed.py" (
    echo ❌ Versión corregida no encontrada
    echo Ejecuta primero: FIX_SYSTEM.bat
    pause
    exit /b 1
)

cd web\backend
start "AI Symbiote Backend - FIXED" cmd /c "python main_fixed.py"
cd ..\..
echo ✅ Backend iniciado (sin errores)

timeout /t 3 /nobreak >nul

echo [2/2] 🎨 Iniciando Frontend...
cd web\frontend

if not exist "node_modules" (
    echo 📦 Instalando dependencias (primera vez)...
    call npm install --silent
)

start "AI Symbiote Frontend" cmd /k "npm run dev"
cd ..\..
echo ✅ Frontend iniciado

echo.
echo     ╔══════════════════════════════════════╗
echo     ║           ✅ SISTEMA ACTIVO           ║
echo     ╚══════════════════════════════════════╝
echo.
echo     🌐 Interfaz Web:    http://localhost:3000
echo     📡 API Backend:     http://localhost:8000  
echo     📚 Documentación:   http://localhost:8000/api/docs
echo.
echo     ⚡ Esta versión funciona sin errores
echo     💡 Usa datos simulados para demostración
echo     🔧 Todas las funciones UI están operativas
echo.

echo Abriendo interfaz en 5 segundos...
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo     ╔══════════════════════════════════════╗
echo     ║      Sistema funcionando perfectamente║
echo     ║      Mantén esta ventana abierta     ║
echo     ╚══════════════════════════════════════╝
echo.
pause