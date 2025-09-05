@echo off
title Probar Backend Solo
color 0C

echo ========================================
echo      PROBANDO SOLO EL BACKEND
echo ========================================
echo.

cd /d "%~dp0"

:: Verificar si existe el directorio backend
if not exist "web\backend" (
    echo [ERROR] Directorio web\backend no existe
    pause
    exit /b 1
)

:: Ir a backend
cd web\backend

:: Verificar archivo principal
if not exist "main.py" (
    echo [ERROR] main.py no existe en web\backend
    pause
    exit /b 1
)

:: Instalar dependencias si no existen
if not exist "venv" (
    echo [INFO] Creando entorno virtual...
    python -m venv venv
)

echo [INFO] Activando entorno virtual...
call venv\Scripts\activate

echo [INFO] Instalando/actualizando dependencias...
pip install -r requirements.txt

echo.
echo [INFO] Iniciando servidor backend...
echo Backend estará en: http://localhost:8000
echo Documentación en: http://localhost:8000/api/docs
echo.
echo Presiona Ctrl+C para detener
echo.

python main.py

pause