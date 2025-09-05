@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo OBVIVLORUM - CREATING DESCRIPTIVE SHORTCUTS
echo ============================================================

set "PROJECT_DIR=%~dp0"
set "USERNAME=%USERNAME%"
set "DESKTOP_C=C:\Users\%USERNAME%\Desktop"
set "DESKTOP_D=D:\Users\%USERNAME%\Desktop"

echo Creando shortcuts con descripciones CLARAS...

call :CreateDescriptiveShortcuts

echo.
echo SHORTCUTS CON BUENAS DESCRIPCIONES CREADOS
echo ============================================================
pause
exit /b 0

:CreateDescriptiveShortcuts
    REM ULTRA LIGHT - Para tu i5 + 12GB
    call :CreateShortcut "OBVIVLORUM - ULTRA LIGHT (Recomendado i5)" "ai_simple_working.py" "Version LIGERA para i5+12GB - Arranque rapido, bajo consumo RAM"
    
    REM BRUTAL FIXED - UI sin blinking
    call :CreateShortcut "OBVIVLORUM - BRUTAL UI FIXED (Sin parpadeo)" "brutal_ui_fixed.py" "Interfaz grafica FIJA - Sin blinking, configuracion completa"
    
    REM PARROTOS - Testing sin restricciones  
    call :CreateShortcut "OBVIVLORUM - PARROTOS UNLIMITED (Testing)" "parrot_gui_integration.py" "Modo testing SIN RESTRICCIONES - Herramientas completas"
    
    REM AI SYMBIOTE TRADICIONAL - El que ya probaste
    call :CreateShortcut "OBVIVLORUM - AI TRADICIONAL (El que probaste)" "ai_symbiote_gui.py" "Version tradicional GUI - La que ya usaste antes"
    
    REM WEB SERVER - Para acceso remoto
    call :CreateShortcut "OBVIVLORUM - WEB SERVER (Acceso navegador)" "web\backend\symbiote_server.py" "Servidor web - Acceso desde navegador en localhost:8000"
    
    REM SELECTOR DE MODO - El launcher
    call :CreateShortcut "OBVIVLORUM - SELECTOR DE MODO (Elige tu modo)" "SINGLE_CLICK_LAUNCHER.py" "Menu para elegir Desktop vs Web mode"
    
    REM SIMPLE TEST - Para debug
    call :CreateShortcut "OBVIVLORUM - SIMPLE TEST (Solo testing)" "brutal_ui_simple.py" "Version simple para probar que la GUI funciona"
    
exit /b 0

:CreateShortcut
    set "SHORTCUT_NAME=%~1"
    set "TARGET_FILE=%~2"
    set "DESCRIPTION=%~3"
    
    echo.
    echo Creando: %SHORTCUT_NAME%
    echo Descripcion: %DESCRIPTION%
    
    REM Crear en ambos desktops
    if exist "%DESKTOP_C%" (
        set "SHORTCUT_PATH=%DESKTOP_C%\%SHORTCUT_NAME%.lnk"
        powershell -Command "
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut('!SHORTCUT_PATH!')
        $Shortcut.TargetPath = 'python'
        $Shortcut.Arguments = '%PROJECT_DIR%%TARGET_FILE%'
        $Shortcut.WorkingDirectory = '%PROJECT_DIR%'
        $Shortcut.Description = '%DESCRIPTION%'
        $Shortcut.Save()
        "
        if exist "!SHORTCUT_PATH!" (
            echo   OK - C: Desktop
        )
    )
    
    if exist "%DESKTOP_D%" (
        set "SHORTCUT_PATH=%DESKTOP_D%\%SHORTCUT_NAME%.lnk"
        powershell -Command "
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut('!SHORTCUT_PATH!')
        $Shortcut.TargetPath = 'python'
        $Shortcut.Arguments = '%PROJECT_DIR%%TARGET_FILE%'
        $Shortcut.WorkingDirectory = '%PROJECT_DIR%'
        $Shortcut.Description = '%DESCRIPTION%'
        $Shortcut.Save()
        "
        if exist "!SHORTCUT_PATH!" (
            echo   OK - D: Desktop  
        )
    )
    
exit /b 0