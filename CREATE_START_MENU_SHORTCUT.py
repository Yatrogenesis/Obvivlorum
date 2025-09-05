#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crear acceso directo en menu inicio para AI Symbiote
"""
import os
import winshell
from win32com.client import Dispatch
from pathlib import Path

def create_start_menu_shortcut():
    """Crear acceso directo en menu inicio."""
    print("[*] Creando acceso directo en menu inicio...")
    
    try:
        # Paths
        programs_folder = Path(os.environ['APPDATA']) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        symbiote_folder = programs_folder / "AI Symbiote"
        
        # Crear carpeta especifica para AI Symbiote
        symbiote_folder.mkdir(parents=True, exist_ok=True)
        
        shell = Dispatch('WScript.Shell')
        
        # Acceso directo principal - Sistema completo
        shortcut_path = str(symbiote_folder / "AI Symbiote - Sistema Completo.lnk")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = r"D:\Obvivlorum\START_SYSTEM_FIXED.bat"
        shortcut.WorkingDirectory = r"D:\Obvivlorum"
        shortcut.IconLocation = r"D:\Obvivlorum\START_SYSTEM_FIXED.bat"
        shortcut.Description = "AI Symbiote - Sistema Completo con todas las interfaces"
        shortcut.save()
        print(f"[OK] Acceso directo principal: {shortcut_path}")
        
        # Acceso directo - Modo seguro
        shortcut_safe_path = str(symbiote_folder / "AI Symbiote - Modo Seguro.lnk")
        shortcut_safe = shell.CreateShortCut(shortcut_safe_path)
        shortcut_safe.Targetpath = r"D:\Obvivlorum\START_SAFE_MODE_FIXED.bat"
        shortcut_safe.WorkingDirectory = r"D:\Obvivlorum"
        shortcut_safe.IconLocation = r"D:\Obvivlorum\START_SAFE_MODE_FIXED.bat"
        shortcut_safe.Description = "AI Symbiote - Modo Seguro para diagnostico"
        shortcut_safe.save()
        print(f"[OK] Acceso directo modo seguro: {shortcut_safe_path}")
        
        # Acceso directo - Solo GUI
        shortcut_gui_path = str(symbiote_folder / "AI Symbiote - Solo GUI.lnk")
        shortcut_gui = shell.CreateShortCut(shortcut_gui_path)
        shortcut_gui.Targetpath = r"python.exe"
        shortcut_gui.Arguments = r'"D:\Obvivlorum\ai_symbiote_gui.py"'
        shortcut_gui.WorkingDirectory = r"D:\Obvivlorum"
        shortcut_gui.Description = "AI Symbiote - Solo interfaz GUI de escritorio"
        shortcut_gui.save()
        print(f"[OK] Acceso directo GUI: {shortcut_gui_path}")
        
        # Acceso directo - Verificacion del sistema
        shortcut_verify_path = str(symbiote_folder / "AI Symbiote - Verificar Sistema.lnk")
        shortcut_verify = shell.CreateShortCut(shortcut_verify_path)
        shortcut_verify.Targetpath = r"python.exe"
        shortcut_verify.Arguments = r'"D:\Obvivlorum\ai_symbiote.py" --status'
        shortcut_verify.WorkingDirectory = r"D:\Obvivlorum"
        shortcut_verify.Description = "Verificar estado del sistema AI Symbiote"
        shortcut_verify.save()
        print(f"[OK] Acceso directo verificacion: {shortcut_verify_path}")
        
        # Crear tambien en el escritorio
        desktop = winshell.desktop()
        desktop_shortcut = Path(desktop) / "AI Symbiote.lnk"
        shortcut_desktop = shell.CreateShortCut(str(desktop_shortcut))
        shortcut_desktop.Targetpath = r"D:\Obvivlorum\START_SYSTEM_FIXED.bat"
        shortcut_desktop.WorkingDirectory = r"D:\Obvivlorum"
        shortcut_desktop.Description = "AI Symbiote - Sistema Completo"
        shortcut_desktop.save()
        print(f"[OK] Acceso directo en escritorio: {desktop_shortcut}")
        
        print("\n=== ACCESOS DIRECTOS CREADOS ===")
        print("Menu Inicio > Programas > AI Symbiote:")
        print("- Sistema Completo")
        print("- Modo Seguro") 
        print("- Solo GUI")
        print("- Verificar Sistema")
        print("- Escritorio: AI Symbiote.lnk")
        
    except ImportError:
        print("[ERROR] Modulos requeridos no disponibles")
        print("Instalando pywin32 y winshell...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32", "winshell"])
        print("Reinicia el script tras instalar dependencias")
    except Exception as e:
        print(f"[ERROR] No se pudieron crear accesos directos: {e}")
        # Metodo alternativo con PowerShell
        create_shortcuts_powershell()

def create_shortcuts_powershell():
    """Metodo alternativo usando PowerShell."""
    print("[*] Usando metodo alternativo PowerShell...")
    
    ps_script = '''
$WshShell = New-Object -comObject WScript.Shell
$ProgramsPath = [Environment]::GetFolderPath("Programs")
$SymbioteFolder = "$ProgramsPath\\AI Symbiote"

# Crear carpeta
New-Item -ItemType Directory -Force -Path $SymbioteFolder

# Acceso directo principal
$Shortcut = $WshShell.CreateShortcut("$SymbioteFolder\\AI Symbiote - Sistema Completo.lnk")
$Shortcut.TargetPath = "D:\\Obvivlorum\\START_SYSTEM_FIXED.bat"
$Shortcut.WorkingDirectory = "D:\\Obvivlorum"
$Shortcut.Description = "AI Symbiote - Sistema Completo"
$Shortcut.Save()

# Acceso directo en escritorio
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutDesktop = $WshShell.CreateShortcut("$Desktop\\AI Symbiote.lnk")
$ShortcutDesktop.TargetPath = "D:\\Obvivlorum\\START_SYSTEM_FIXED.bat"
$ShortcutDesktop.WorkingDirectory = "D:\\Obvivlorum"
$ShortcutDesktop.Description = "AI Symbiote - Sistema Completo"
$ShortcutDesktop.Save()

Write-Host "Accesos directos creados correctamente"
'''
    
    import subprocess
    result = subprocess.run(['powershell', '-Command', ps_script], 
                          capture_output=True, text=True, shell=True)
    
    if result.returncode == 0:
        print("[OK] Accesos directos creados con PowerShell")
    else:
        print(f"[ERROR] PowerShell fallo: {result.stderr}")

if __name__ == "__main__":
    create_start_menu_shortcut()