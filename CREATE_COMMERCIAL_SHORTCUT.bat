@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo OBVIVLORUM - CREATING COMMERCIAL-STYLE LAUNCHER SHORTCUT
echo ============================================================

set "PROJECT_DIR=%~dp0"
set "USERNAME=%USERNAME%"
set "DESKTOP_C=C:\Users\%USERNAME%\Desktop"
set "DESKTOP_D=D:\Users\%USERNAME%\Desktop"

echo Creating commercial launcher shortcut...

call :CreateCommercialShortcut

echo.
echo COMMERCIAL LAUNCHER SHORTCUT CREATED
echo ============================================================
pause
exit /b 0

:CreateCommercialShortcut
    set "SHORTCUT_NAME=OBVIVLORUM AI - Commercial Launcher"
    set "TARGET_FILE=OBVIVLORUM_LAUNCHER.py"
    set "DESCRIPTION=Commercial-style persistent startup with mode selection and performance optimization"
    
    echo.
    echo Creating: %SHORTCUT_NAME%
    echo Description: %DESCRIPTION%
    
    REM Create on C: Desktop if exists
    if exist "%DESKTOP_C%" (
        set "SHORTCUT_PATH=%DESKTOP_C%\%SHORTCUT_NAME%.lnk"
        powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP_C%\%SHORTCUT_NAME%.lnk'); $Shortcut.TargetPath = 'python'; $Shortcut.Arguments = '%PROJECT_DIR%%TARGET_FILE%'; $Shortcut.WorkingDirectory = '%PROJECT_DIR%'; $Shortcut.Description = '%DESCRIPTION%'; $Shortcut.IconLocation = 'shell32.dll,137'; $Shortcut.WindowStyle = 1; $Shortcut.Save()"
        if exist "%DESKTOP_C%\%SHORTCUT_NAME%.lnk" (
            echo   OK - C: Desktop created
        )
    )
    
    REM Create on D: Desktop if exists
    if exist "%DESKTOP_D%" (
        set "SHORTCUT_PATH=%DESKTOP_D%\%SHORTCUT_NAME%.lnk"
        powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP_D%\%SHORTCUT_NAME%.lnk'); $Shortcut.TargetPath = 'python'; $Shortcut.Arguments = '%PROJECT_DIR%%TARGET_FILE%'; $Shortcut.WorkingDirectory = '%PROJECT_DIR%'; $Shortcut.Description = '%DESCRIPTION%'; $Shortcut.IconLocation = 'shell32.dll,137'; $Shortcut.WindowStyle = 1; $Shortcut.Save()"
        if exist "%DESKTOP_D%\%SHORTCUT_NAME%.lnk" (
            echo   OK - D: Desktop created
        )
    )
    
    REM Also create Start Menu shortcut
    set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
    if exist "%STARTMENU%" (
        set "SHORTCUT_PATH=%STARTMENU%\%SHORTCUT_NAME%.lnk"
        powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%STARTMENU%\%SHORTCUT_NAME%.lnk'); $Shortcut.TargetPath = 'python'; $Shortcut.Arguments = '%PROJECT_DIR%%TARGET_FILE%'; $Shortcut.WorkingDirectory = '%PROJECT_DIR%'; $Shortcut.Description = '%DESCRIPTION%'; $Shortcut.IconLocation = 'shell32.dll,137'; $Shortcut.WindowStyle = 1; $Shortcut.Save()"
        if exist "%STARTMENU%\%SHORTCUT_NAME%.lnk" (
            echo   OK - Start Menu created
        )
    )
    
exit /b 0