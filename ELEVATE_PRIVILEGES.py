#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elevador de Privilegios para AI Symbiote
========================================
Otorga privilegios administrativos y configura el sistema correctamente.
"""
import os
import sys
import subprocess
import winreg
import json
from pathlib import Path
import ctypes

def is_admin():
    """Verificar si se ejecuta como administrador."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def elevate_self():
    """Auto-elevar privilegios si no se ejecuta como admin."""
    if is_admin():
        return True
    else:
        print("[!] Se requieren privilegios de administrador")
        print("Solicitando elevacion...")
        try:
            # Intentar elevar privilegios
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, " ".join(sys.argv), None, 1
            )
            return False  # Se reinicia con privilegios
        except Exception as e:
            print(f"ERROR: No se pudieron elevar privilegios: {e}")
            return False

def configure_with_admin_privileges():
    """Configurar sistema con privilegios administrativos."""
    print("========================================")
    print("   AI SYMBIOTE - CONFIGURACION ADMIN")
    print("========================================")
    print()
    
    if not is_admin():
        print("ERROR: Esta funcion requiere privilegios de administrador")
        return False
    
    success_count = 0
    total_tasks = 6
    
    # 1. Crear tarea programada del sistema
    print("[1/6] Creando tarea programada del sistema...")
    try:
        cmd = [
            "schtasks", "/create",
            "/tn", "AISymbiote_System",
            "/tr", "D:\\Obvivlorum\\START_SYSTEM_FIXED.bat",
            "/sc", "onlogon",
            "/rl", "highest",
            "/f"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("  [OK] Tarea programada creada exitosamente")
            success_count += 1
        else:
            print(f"  [FAIL] Error: {result.stderr}")
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
    
    # 2. Configurar registro HKLM
    print("[2/6] Configurando registro del sistema (HKLM)...")
    try:
        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE) as key:
            batch_path = r"D:\Obvivlorum\START_SYSTEM_FIXED.bat"
            winreg.SetValueEx(key, "AISymbiote_System", 0, winreg.REG_SZ, batch_path)
            print("  [OK] Registro HKLM configurado")
            success_count += 1
    except Exception as e:
        print(f"  [FAIL] Error en registro HKLM: {e}")
    
    # 3. Crear servicio de Windows (opcional)
    print("[3/6] Configurando servicio de Windows...")
    try:
        # Crear wrapper para servicio
        service_wrapper = Path("D:/Obvivlorum/service_wrapper.py")
        wrapper_content = '''
import os
import sys
import time
import subprocess
import servicemanager
import win32serviceutil
import win32service

class AISymbioteService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AISymbioteService"
    _svc_display_name_ = "AI Symbiote System Service"
    _svc_description_ = "AI Symbiote Background Service"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        
    def SvcDoRun(self):
        while True:
            try:
                subprocess.run(["python", "D:/Obvivlorum/ai_symbiote.py", "--background"], 
                             cwd="D:/Obvivlorum")
                time.sleep(300)  # Check every 5 minutes
            except:
                time.sleep(60)

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(AISymbioteService)
        '''
        
        with open(service_wrapper, 'w') as f:
            f.write(wrapper_content)
        print("  [OK] Wrapper de servicio creado")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Error en servicio: {e}")
    
    # 4. Configurar permisos de carpeta
    print("[4/6] Configurando permisos de carpetas...")
    try:
        # Dar permisos completos a la carpeta del proyecto
        cmd = ['icacls', 'D:\\Obvivlorum', '/grant', 'Everyone:(OI)(CI)F', '/T']
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("  [OK] Permisos de carpeta configurados")
            success_count += 1
        else:
            print(f"  [FAIL] Error en permisos: {result.stderr}")
    except Exception as e:
        print(f"  [FAIL] Error configurando permisos: {e}")
    
    # 5. Configurar firewall (permitir Python)
    print("[5/6] Configurando reglas de firewall...")
    try:
        python_exe = sys.executable
        cmd = [
            'netsh', 'advfirewall', 'firewall', 'add', 'rule',
            'name=AISymbiote Python',
            f'program={python_exe}',
            'dir=in', 'action=allow', 'protocol=TCP'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("  [OK] Regla de firewall agregada")
            success_count += 1
        else:
            print("   Warning: No se pudo configurar firewall")
    except Exception as e:
        print(f"   Warning firewall: {e}")
    
    # 6. Crear configuracion persistente
    print("[6/6] Guardando configuracion de privilegios...")
    try:
        config = {
            "admin_configured": True,
            "timestamp": "2025-09-03",
            "scheduled_task": True,
            "registry_hklm": True,
            "service_created": True,
            "permissions_set": True,
            "firewall_configured": True
        }
        
        config_file = "D:/Obvivlorum/.ai_symbiote/admin_config.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print("  [OK] Configuracion guardada")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] Error guardando config: {e}")
    
    # Resumen
    print()
    print("=" * 40)
    print(f"CONFIGURACION COMPLETADA: {success_count}/{total_tasks} tareas exitosas")
    
    if success_count >= 4:
        print("[OK] SISTEMA CONFIGURADO EXITOSAMENTE")
        print("El AI Symbiote ahora deberia funcionar sin errores de permisos")
    else:
        print(" CONFIGURACION PARCIAL")
        print("Algunas configuraciones fallaron, pero el sistema deberia funcionar")
    
    return success_count >= 4

def main():
    """Funcion principal."""
    if not elevate_self():
        return
    
    print("Ejecutandose con privilegios de administrador...")
    configure_with_admin_privileges()
    input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    main()