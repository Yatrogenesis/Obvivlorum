# Actualizar acceso directo con menú corregido
$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut("$Desktop\AI Symbiote.lnk")
$Shortcut.TargetPath = "D:\Obvivlorum\AI_SYMBIOTE_MENU_FIXED.bat"
$Shortcut.WorkingDirectory = "D:\Obvivlorum"
$Shortcut.IconLocation = "C:\Windows\System32\shell32.dll,13"
$Shortcut.Description = "AI Symbiote - Menu Principal (Unicode Corregido)"
$Shortcut.Save()
Write-Host "Acceso directo actualizado con menu Unicode corregido"