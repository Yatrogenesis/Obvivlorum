@echo off
title AI Symbiote - Diagnóstico
color 0E

echo ========================================
echo   AI SYMBIOTE - DIAGNÓSTICO DEL SISTEMA
echo ========================================
echo.

echo [1] Verificando Python...
python --version
if errorlevel 1 (
    echo [ERROR] Python no encontrado
) else (
    echo [OK] Python disponible
)
echo.

echo [2] Verificando Node.js...
node --version
if errorlevel 1 (
    echo [ERROR] Node.js no encontrado
) else (
    echo [OK] Node.js disponible
)
echo.

echo [3] Verificando estructura de archivos...
if exist "ai_symbiote.py" (
    echo [OK] ai_symbiote.py encontrado
) else (
    echo [ERROR] ai_symbiote.py no encontrado
)

if exist "web\backend\main.py" (
    echo [OK] Backend encontrado
) else (
    echo [ERROR] Backend no encontrado
)

if exist "web\frontend\package.json" (
    echo [OK] Frontend encontrado
) else (
    echo [ERROR] Frontend no encontrado
)
echo.

echo [4] Probando AI Symbiote core...
python ai_symbiote.py --status
echo.

echo [5] Verificando puertos...
netstat -an | findstr :8000
if errorlevel 1 (
    echo [OK] Puerto 8000 disponible
) else (
    echo [WARN] Puerto 8000 ocupado
)

netstat -an | findstr :3000
if errorlevel 1 (
    echo [OK] Puerto 3000 disponible
) else (
    echo [WARN] Puerto 3000 ocupado
)
echo.

echo ========================================
echo Presiona cualquier tecla para continuar...
pause >nul