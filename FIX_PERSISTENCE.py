#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix para persistencia con caracteres correctos y redundancia
"""
import os
import winreg
from pathlib import Path

def fix_persistence():
    """Corregir persistencia con caracteres correctos."""
    print("[*] Corrigiendo persistencia...")
    
    # Paths correctos
    appdata_folder = Path(os.environ['APPDATA']) / "AISymbiote"
    startup_folder = Path(os.environ['APPDATA']) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    
    # Crear carpetas si no existen
    appdata_folder.mkdir(parents=True, exist_ok=True)
    startup_folder.mkdir(parents=True, exist_ok=True)
    
    # Script BAT corregido
    batch_content = """@echo off
REM AI Symbiote Auto Start
cd /d "D:\\Obvivlorum"
python "D:\\Obvivlorum\\ai_symbiote.py" --background --persistent
exit
"""
    
    batch_file = appdata_folder / "start_symbiote_fixed.bat"
    with open(batch_file, 'w', encoding='utf-8') as f:
        f.write(batch_content)
    print(f"[OK] Batch corregido: {batch_file}")
    
    # VBS corregido para ejecucion silenciosa
    vbs_content = f"""Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "{batch_file}" & Chr(34), 0
Set WshShell = Nothing
"""
    
    vbs_file = appdata_folder / "start_symbiote_fixed.vbs"
    with open(vbs_file, 'w', encoding='utf-8') as f:
        f.write(vbs_content)
    print(f"[OK] VBS silencioso: {vbs_file}")
    
    # Actualizar registro con path corregido
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, "AISymbiote", 0, winreg.REG_SZ, str(vbs_file))
        winreg.CloseKey(key)
        print("[OK] Registro actualizado con VBS corregido")
    except Exception as e:
        print(f"[ERROR] No se pudo actualizar registro: {e}")
    
    # Crear acceso directo en carpeta de inicio
    startup_batch = startup_folder / "AI_Symbiote_Fixed.bat"
    with open(startup_batch, 'w', encoding='utf-8') as f:
        f.write(batch_content)
    print(f"[OK] Startup folder: {startup_batch}")
    
    # Crear script de verificacion
    verify_script = appdata_folder / "verify_system.bat"
    verify_content = """@echo off
title AI Symbiote - Verificacion
echo Verificando sistema AI Symbiote...
echo.

REM Verificar Python
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python no disponible
    pause
    exit /b 1
)

REM Verificar archivos
if not exist "D:\\Obvivlorum\\ai_symbiote.py" (
    echo ERROR: ai_symbiote.py no encontrado
    pause
    exit /b 1
)

echo [OK] Sistema verificado correctamente
echo Ejecutando sistema...
cd /d "D:\\Obvivlorum"
python "D:\\Obvivlorum\\ai_symbiote.py" --status
pause
"""
    
    with open(verify_script, 'w', encoding='utf-8') as f:
        f.write(verify_content)
    print(f"[OK] Script verificacion: {verify_script}")
    
    print("\n=== REDUNDANCIA CREADA ===")
    print("1. Registro: VBS silencioso")
    print("2. Startup folder: Batch directo") 
    print("3. Verificacion: Script diagnostico")
    print("4. Encoding: UTF-8 en todos los archivos")

if __name__ == "__main__":
    fix_persistence()
    input("\nPresiona Enter para continuar...")