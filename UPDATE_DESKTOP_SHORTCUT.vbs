' Script para actualizar el acceso directo del AI Symbiote
' Actualiza el acceso directo para usar el modo seguro sin persistencia

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

desktopPath = WshShell.SpecialFolders("Desktop")
shortcutPath = desktopPath & "\AI Symbiote - Modo Seguro.lnk"
oldShortcutPath = desktopPath & "\AI Symbiote - Modo Persistente.lnk"

' Eliminar el acceso directo antiguo si existe
If fso.FileExists(oldShortcutPath) Then
    fso.DeleteFile oldShortcutPath
End If

' Crear nuevo acceso directo actualizado
Set oShortcut = WshShell.CreateShortcut(shortcutPath)
oShortcut.TargetPath = "D:\Obvivlorum\SAFE_START.bat"
oShortcut.Arguments = ""
oShortcut.WorkingDirectory = "D:\Obvivlorum"
oShortcut.IconLocation = "C:\Windows\System32\cmd.exe, 0"
oShortcut.Description = "AI Symbiote - Inicio Manual Seguro (Sin Persistencia)"
oShortcut.WindowStyle = 1
oShortcut.Hotkey = "CTRL+SHIFT+A"
oShortcut.Save

WScript.Echo "Acceso directo actualizado exitosamente: AI Symbiote - Modo Seguro"