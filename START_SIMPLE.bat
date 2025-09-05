@echo off
title AI Symbiote - Inicio Simple
color 0B

echo ========================================
echo      AI SYMBIOTE - INICIO SIMPLE
echo ========================================
echo.

:: Ir al directorio correcto
cd /d "%~dp0"

:: Paso 1: Probar AI Symbiote básico
echo [1/3] Probando AI Symbiote básico...
python ai_symbiote.py --user-id test --test-protocol ALPHA
if errorlevel 1 (
    echo [ERROR] AI Symbiote core falló
    pause
    exit /b 1
)
echo [OK] AI Symbiote core funciona
echo.

:: Paso 2: Probar backend
echo [2/3] Iniciando backend...
cd web\backend
start "Backend" cmd /k "python main.py"
cd ..\..
echo [OK] Backend iniciado en ventana separada
echo.

:: Paso 3: Probar frontend (opcional)
echo [3/3] ¿Iniciar interfaz web? (s/n)
set /p choice="Respuesta: "
if /i "%choice%"=="s" (
    cd web\frontend
    if not exist node_modules (
        echo Instalando dependencias...
        npm install
    )
    start "Frontend" cmd /k "npm run dev"
    cd ..\..
    echo [OK] Frontend iniciado
    echo.
    echo Abriendo navegador en 5 segundos...
    timeout /t 5 /nobreak >nul
    start http://localhost:3000
)

echo.
echo ========================================
echo Sistema básico iniciado
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000 (si se inició)
echo ========================================
pause