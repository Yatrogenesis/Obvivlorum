@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo 🚀 OBVIVLORUM - CREATING DESKTOP SHORTCUTS
echo ============================================================

REM Get current directory
set "PROJECT_DIR=%~dp0"

REM Usuario actual
set "USERNAME=%USERNAME%"
echo Usuario detectado: %USERNAME%

REM Rutas de Desktop (C y D)
set "DESKTOP_C=C:\Users\%USERNAME%\Desktop"
set "DESKTOP_D=D:\Users\%USERNAME%\Desktop"

echo.
echo Verificando ubicaciones de Desktop...

REM Verificar Desktop en C:
if exist "%DESKTOP_C%" (
    echo ✅ Desktop encontrado en C: %DESKTOP_C%
    set "HAS_DESKTOP_C=1"
) else (
    echo ❌ Desktop NO encontrado en C:
    set "HAS_DESKTOP_C=0"
)

REM Verificar Desktop en D:
if exist "%DESKTOP_D%" (
    echo ✅ Desktop encontrado en D: %DESKTOP_D%
    set "HAS_DESKTOP_D=1"
) else (
    echo ❌ Desktop NO encontrado en D:
    set "HAS_DESKTOP_D=0"
)

echo.
echo ============================================================
echo 🔥 CREANDO ACCESOS DIRECTOS BRUTALES
echo ============================================================

REM Función para crear shortcuts
call :CreateShortcuts

echo.
echo ✅ PROCESO COMPLETADO
echo ============================================================
pause
exit /b 0

:CreateShortcuts
    REM Acceso directo principal - OBVIVLORUM LAUNCHER
    call :CreateShortcut "🧠 OBVIVLORUM - BRUTAL LAUNCHER" "SINGLE_CLICK_LAUNCHER.py" "Lanzador unificado con configuración completa"
    
    REM Acceso directo - Brutal UI Desktop
    call :CreateShortcut "🔥 OBVIVLORUM - BRUTAL DESKTOP" "brutal_ui_desktop.py" "Interfaz desktop brutalmente impresionante"
    
    REM Acceso directo - ParrotOS Integration
    call :CreateShortcut "🦜 OBVIVLORUM - PARROTOS MODE" "parrot_gui_integration.py" "Modo ParrotOS sin restricciones"
    
    REM Acceso directo - AI Symbiote GUI
    call :CreateShortcut "🤖 OBVIVLORUM - AI SYMBIOTE" "ai_symbiote_gui.py" "Interfaz AI Symbiote tradicional"
    
    REM Acceso directo - Web Server
    call :CreateShortcut "🌐 OBVIVLORUM - WEB SERVER" "web\backend\symbiote_server.py" "Servidor web con API completa"
    
exit /b 0

:CreateShortcut
    set "SHORTCUT_NAME=%~1"
    set "TARGET_FILE=%~2"
    set "DESCRIPTION=%~3"
    
    echo.
    echo Creando: %SHORTCUT_NAME%
    echo Target: %TARGET_FILE%
    
    REM PowerShell script para crear shortcuts
    set "PS_SCRIPT="
    set "PS_SCRIPT=!PS_SCRIPT!$WshShell = New-Object -ComObject WScript.Shell; "
    set "PS_SCRIPT=!PS_SCRIPT!$projectDir = '%PROJECT_DIR%'; "
    set "PS_SCRIPT=!PS_SCRIPT!$targetFile = Join-Path $projectDir '%TARGET_FILE%'; "
    
    REM Crear en Desktop C: si existe
    if %HAS_DESKTOP_C%==1 (
        set "DESKTOP_PATH_C=%DESKTOP_C%\%SHORTCUT_NAME%.lnk"
        set "PS_CMD_C=!PS_SCRIPT!$Shortcut = $WshShell.CreateShortcut('!DESKTOP_PATH_C!'); $Shortcut.TargetPath = 'python'; $Shortcut.Arguments = '$targetFile'; $Shortcut.WorkingDirectory = '$projectDir'; $Shortcut.Description = '%DESCRIPTION%'; $Shortcut.Save();"
        
        powershell -Command "!PS_CMD_C!"
        if exist "!DESKTOP_PATH_C!" (
            echo   ✅ Creado en C: Desktop
        ) else (
            echo   ❌ Error creando en C: Desktop
        )
    )
    
    REM Crear en Desktop D: si existe
    if %HAS_DESKTOP_D%==1 (
        set "DESKTOP_PATH_D=%DESKTOP_D%\%SHORTCUT_NAME%.lnk"
        set "PS_CMD_D=!PS_SCRIPT!$Shortcut = $WshShell.CreateShortcut('!DESKTOP_PATH_D!'); $Shortcut.TargetPath = 'python'; $Shortcut.Arguments = '$targetFile'; $Shortcut.WorkingDirectory = '$projectDir'; $Shortcut.Description = '%DESCRIPTION%'; $Shortcut.Save();"
        
        powershell -Command "!PS_CMD_D!"
        if exist "!DESKTOP_PATH_D!" (
            echo   ✅ Creado en D: Desktop
        ) else (
            echo   ❌ Error creando en D: Desktop
        )
    )
    
exit /b 0