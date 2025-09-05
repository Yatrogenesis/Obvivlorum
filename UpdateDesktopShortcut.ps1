# Script PowerShell para actualizar el acceso directo del AI Symbiote
$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [Environment]::GetFolderPath("Desktop")

# Eliminar acceso directo antiguo
$OldShortcut = "$Desktop\AI Symbiote - Modo Persistente.lnk"
if (Test-Path $OldShortcut) {
    Remove-Item $OldShortcut -Force
    Write-Host "Acceso directo antiguo eliminado"
}

# Crear nuevo acceso directo con menú
$NewShortcut = $WshShell.CreateShortcut("$Desktop\AI Symbiote.lnk")
$NewShortcut.TargetPath = "D:\Obvivlorum\AI_SYMBIOTE_MENU.bat"
$NewShortcut.WorkingDirectory = "D:\Obvivlorum"
$NewShortcut.IconLocation = "C:\Windows\System32\shell32.dll,13"
$NewShortcut.Description = "AI Symbiote - Sistema de IA Adaptativo (Menu Principal)"
$NewShortcut.Hotkey = "CTRL+SHIFT+S"
$NewShortcut.Save()

Write-Host "Acceso directo 'AI Symbiote.lnk' creado exitosamente en el escritorio"
Write-Host "Presione CTRL+SHIFT+S para acceso rapido"
Write-Host ""
Write-Host "El nuevo acceso directo abre un menu con opciones:"
Write-Host "- Modo Seguro (sin persistencia)"
Write-Host "- GUI Desktop"
Write-Host "- Servidor Web"
Write-Host "- Reparacion del sistema"
Write-Host "- Y mas opciones..."