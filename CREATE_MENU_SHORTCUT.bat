@echo off
REM Crear acceso directo con menú de opciones
echo Creando acceso directo con menu de opciones...

powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([System.Environment]::GetFolderPath('Desktop') + '\AI Symbiote.lnk'); $Shortcut.TargetPath = 'D:\Obvivlorum\AI_SYMBIOTE_MENU.bat'; $Shortcut.WorkingDirectory = 'D:\Obvivlorum'; $Shortcut.IconLocation = 'C:\Windows\System32\shell32.dll, 13'; $Shortcut.Description = 'AI Symbiote - Sistema de IA Adaptativo'; $Shortcut.Save()"

echo Acceso directo creado en el escritorio
pause