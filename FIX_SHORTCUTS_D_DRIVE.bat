@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo FIXING D: DRIVE SHORTCUTS
echo ============================================================

set "PROJECT_DIR=%~dp0"
set "USERNAME=%USERNAME%"
set "DESKTOP_D=D:\Users\%USERNAME%\Desktop"

echo Project Directory: %PROJECT_DIR%
echo Username: %USERNAME%
echo Target Desktop: %DESKTOP_D%

echo.
echo Fixing shortcuts with correct paths...

REM Function to fix shortcuts
call :FixShortcuts

echo.
echo SHORTCUTS FIXED ON D: DRIVE
echo ============================================================
pause
exit /b 0

:FixShortcuts
    call :FixShortcut "OBVIVLORUM - BRUTAL LAUNCHER" "SINGLE_CLICK_LAUNCHER.py"
    call :FixShortcut "OBVIVLORUM - BRUTAL DESKTOP" "brutal_ui_desktop.py"
    call :FixShortcut "OBVIVLORUM - PARROTOS MODE" "parrot_gui_integration.py"
    call :FixShortcut "OBVIVLORUM - AI SYMBIOTE" "ai_symbiote_gui.py"
    call :FixShortcut "OBVIVLORUM - WEB SERVER" "web\backend\symbiote_server.py"
exit /b 0

:FixShortcut
    set "SHORTCUT_NAME=%~1"
    set "TARGET_FILE=%~2"
    set "SHORTCUT_PATH=%DESKTOP_D%\%SHORTCUT_NAME%.lnk"
    set "FULL_TARGET=%PROJECT_DIR%%TARGET_FILE%"
    
    echo.
    echo Fixing: %SHORTCUT_NAME%
    echo Path: %SHORTCUT_PATH%
    echo Target: %FULL_TARGET%
    
    REM Create corrected shortcut
    powershell -Command "
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%')
    $Shortcut.TargetPath = 'python'
    $Shortcut.Arguments = '\"%FULL_TARGET%\"'
    $Shortcut.WorkingDirectory = '%PROJECT_DIR%'
    $Shortcut.Description = '%SHORTCUT_NAME%'
    $Shortcut.Save()
    "
    
    if exist "%SHORTCUT_PATH%" (
        echo   OK - Fixed successfully
    ) else (
        echo   ERROR - Fix failed
    )
    
exit /b 0