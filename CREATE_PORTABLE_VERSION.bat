@echo off
title AI Symbiote - Creador de Version Portable
color 0E

echo.
echo  ╔═════════════════════════════════════════════════╗
echo  ║           AI Symbiote Portable Creator          ║
echo  ║          Version Optimizada para USB           ║
echo  ╚═════════════════════════════════════════════════╝
echo.

set "PORTABLE_DIR=D:\AI_Symbiote_Portable"
set "SOURCE_DIR=D:\Obvivlorum"

echo [INFO] Creando directorio portable...
if exist "%PORTABLE_DIR%" rmdir /s /q "%PORTABLE_DIR%"
mkdir "%PORTABLE_DIR%"

echo [INFO] Copiando archivos esenciales...

REM Core AI files
copy "%SOURCE_DIR%\ai_simple_working.py" "%PORTABLE_DIR%\"
copy "%SOURCE_DIR%\ai_symbiote.py" "%PORTABLE_DIR%\"
copy "%SOURCE_DIR%\windows_persistence.py" "%PORTABLE_DIR%\"
copy "%SOURCE_DIR%\start_symbiote.py" "%PORTABLE_DIR%\"

REM Web interface
xcopy /E /I /Q "%SOURCE_DIR%\web" "%PORTABLE_DIR%\web"

REM AION core (sin logs pesados)
xcopy /E /I /Q "%SOURCE_DIR%\AION\*.py" "%PORTABLE_DIR%\AION\"

REM Scripts de inicio
copy "%SOURCE_DIR%\START_INTELLIGENT_AI.bat" "%PORTABLE_DIR%\"
copy "%SOURCE_DIR%\START_SIMPLE.bat" "%PORTABLE_DIR%\"

echo [INFO] Creando launcher portable...
echo @echo off > "%PORTABLE_DIR%\START_PORTABLE.bat"
echo title AI Symbiote Portable >> "%PORTABLE_DIR%\START_PORTABLE.bat"
echo color 0B >> "%PORTABLE_DIR%\START_PORTABLE.bat"
echo. >> "%PORTABLE_DIR%\START_PORTABLE.bat"
echo echo Iniciando AI Symbiote desde USB... >> "%PORTABLE_DIR%\START_PORTABLE.bat"
echo cd /d "%%~dp0" >> "%PORTABLE_DIR%\START_PORTABLE.bat"
echo python ai_simple_working.py >> "%PORTABLE_DIR%\START_PORTABLE.bat"
echo pause >> "%PORTABLE_DIR%\START_PORTABLE.bat"

REM Web server portable
echo @echo off > "%PORTABLE_DIR%\START_WEB_PORTABLE.bat"
echo title AI Symbiote Web Server Portable >> "%PORTABLE_DIR%\START_WEB_PORTABLE.bat"
echo color 0B >> "%PORTABLE_DIR%\START_WEB_PORTABLE.bat"
echo. >> "%PORTABLE_DIR%\START_WEB_PORTABLE.bat"
echo echo Iniciando servidor web desde USB... >> "%PORTABLE_DIR%\START_WEB_PORTABLE.bat"
echo cd /d "%%~dp0\web\backend" >> "%PORTABLE_DIR%\START_WEB_PORTABLE.bat"
echo python symbiote_server.py >> "%PORTABLE_DIR%\START_WEB_PORTABLE.bat"
echo pause >> "%PORTABLE_DIR%\START_WEB_PORTABLE.bat"

echo [INFO] Creando requirements.txt portable...
echo fastapi >> "%PORTABLE_DIR%\requirements.txt"
echo uvicorn >> "%PORTABLE_DIR%\requirements.txt"
echo opencv-python >> "%PORTABLE_DIR%\requirements.txt"
echo speechrecognition >> "%PORTABLE_DIR%\requirements.txt"
echo pyttsx3 >> "%PORTABLE_DIR%\requirements.txt"
echo numpy >> "%PORTABLE_DIR%\requirements.txt"
echo requests >> "%PORTABLE_DIR%\requirements.txt"

echo [INFO] Creando README portable...
echo # AI Symbiote Portable > "%PORTABLE_DIR%\README.md"
echo. >> "%PORTABLE_DIR%\README.md"
echo ## Instalacion rapida: >> "%PORTABLE_DIR%\README.md"
echo 1. Instalar Python 3.8+ >> "%PORTABLE_DIR%\README.md"
echo 2. pip install -r requirements.txt >> "%PORTABLE_DIR%\README.md"
echo 3. Ejecutar START_WEB_PORTABLE.bat >> "%PORTABLE_DIR%\README.md"
echo 4. Abrir http://localhost:8000 >> "%PORTABLE_DIR%\README.md"
echo. >> "%PORTABLE_DIR%\README.md"
echo Tamaño: ~50MB (sin modelos pesados) >> "%PORTABLE_DIR%\README.md"

echo.
echo [SUCCESS] Version portable creada en: %PORTABLE_DIR%
echo.
echo Contenido incluido:
echo ✓ AI inteligente sin dependencias pesadas
echo ✓ Interfaz web completa
echo ✓ Scripts de inicio automaticos
echo ✓ Requirements.txt para instalacion rapida
echo.
echo Tamaño estimado: ~50MB
echo.
pause