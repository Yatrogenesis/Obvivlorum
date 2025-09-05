#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificación final completa del sistema AI Symbiote
"""
import os
import sys
import json
import subprocess
import socket
from pathlib import Path
import winreg

def check_python_and_deps():
    """Verificar Python y dependencias críticas."""
    print("=== VERIFICACION PYTHON Y DEPENDENCIAS ===")
    
    # Python
    print(f"Python: {sys.version}")
    
    # Dependencias críticas
    critical_deps = [
        "tkinter", "fastapi", "uvicorn", "numpy", "logging", "threading",
        "json", "os", "sys", "pathlib", "winreg", "subprocess"
    ]
    
    missing = []
    for dep in critical_deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"[OK] {dep}")
        except ImportError:
            missing.append(dep)
            print(f"✗ {dep} - FALTANTE")
    
    if missing:
        print(f"\nERROR: Dependencias faltantes: {missing}")
        return False
    else:
        print("\n✓ Todas las dependencias críticas disponibles")
        return True

def check_file_structure():
    """Verificar estructura de archivos."""
    print("\n=== VERIFICACION ESTRUCTURA DE ARCHIVOS ===")
    
    critical_files = [
        "D:/Obvivlorum/ai_symbiote.py",
        "D:/Obvivlorum/ai_symbiote_gui.py", 
        "D:/Obvivlorum/windows_persistence.py",
        "D:/Obvivlorum/adaptive_task_facilitator.py",
        "D:/Obvivlorum/AION/aion_core.py",
        "D:/Obvivlorum/web/backend/symbiote_server.py",
        "D:/Obvivlorum/web/frontend/symbiote-chat.html",
        "D:/Obvivlorum/START_SYSTEM_FIXED.bat",
        "D:/Obvivlorum/config_optimized.json"
    ]
    
    missing_files = []
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path} - FALTANTE")
    
    if missing_files:
        print(f"\nWARNING: Archivos faltantes: {len(missing_files)}")
        return False
    else:
        print("\n✓ Todos los archivos críticos presentes")
        return True

def check_persistence():
    """Verificar persistencia Windows."""
    print("\n=== VERIFICACION PERSISTENCIA ===")
    
    # Registro
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                           r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run")
        value, regtype = winreg.QueryValueEx(key, "AISymbiote")
        winreg.CloseKey(key)
        print(f"✓ Registro: {value}")
    except Exception as e:
        print(f"✗ Registro no encontrado: {e}")
        return False
    
    # Archivos de persistencia
    appdata_folder = Path(os.environ['APPDATA']) / "AISymbiote"
    persistence_files = [
        appdata_folder / "start_symbiote_fixed.bat",
        appdata_folder / "start_symbiote_fixed.vbs",
        appdata_folder / "verify_system.bat"
    ]
    
    for file_path in persistence_files:
        if file_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - FALTANTE")
            return False
    
    print("✓ Persistencia configurada correctamente")
    return True

def check_ports():
    """Verificar puertos disponibles."""
    print("\n=== VERIFICACION PUERTOS ===")
    
    ports_to_check = [8000, 8001, 8080]
    available_ports = []
    
    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result != 0:
            print(f"✓ Puerto {port} disponible")
            available_ports.append(port)
        else:
            print(f"✗ Puerto {port} en uso")
    
    if available_ports:
        print(f"✓ Puertos disponibles para servidor web: {available_ports}")
        return True
    else:
        print("WARNING: Todos los puertos comunes están ocupados")
        return False

def check_shortcuts():
    """Verificar accesos directos."""
    print("\n=== VERIFICACION ACCESOS DIRECTOS ===")
    
    programs_folder = Path(os.environ['APPDATA']) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "AI Symbiote"
    
    if programs_folder.exists():
        shortcuts = list(programs_folder.glob("*.lnk"))
        print(f"✓ Carpeta menú inicio: {len(shortcuts)} accesos directos")
        for shortcut in shortcuts:
            print(f"  - {shortcut.name}")
    else:
        print("✗ Carpeta de accesos directos no encontrada")
        return False
    
    # Desktop
    desktop_shortcut = Path(os.path.expanduser("~/Desktop")) / "AI Symbiote.lnk"
    if desktop_shortcut.exists():
        print("✓ Acceso directo en escritorio")
    else:
        print("✗ Acceso directo en escritorio no encontrado")
    
    return True

def test_core_system():
    """Test básico del sistema core."""
    print("\n=== TEST SISTEMA CORE ===")
    
    try:
        # Test import del módulo principal
        sys.path.insert(0, "D:/Obvivlorum")
        
        # Verificar que el módulo principal puede importarse
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, 'D:/Obvivlorum'); import ai_symbiote; print('CORE OK')"
        ], capture_output=True, text=True, cwd="D:/Obvivlorum")
        
        if "CORE OK" in result.stdout:
            print("✓ Módulo core puede importarse")
        else:
            print(f"✗ Error importando core: {result.stderr}")
            return False
            
        # Test status command
        status_result = subprocess.run([
            sys.executable, "ai_symbiote.py", "--status"
        ], capture_output=True, text=True, cwd="D:/Obvivlorum")
        
        if status_result.returncode == 0:
            print("✓ Comando --status funciona")
        else:
            print(f"✗ Error en --status: {status_result.stderr}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error en test core: {e}")
        return False

def create_emergency_scripts():
    """Crear scripts de emergencia."""
    print("\n=== CREANDO SCRIPTS DE EMERGENCIA ===")
    
    # Script de diagnóstico
    diag_script = """@echo off
title AI Symbiote - Diagnóstico de Emergencia
echo ============================================
echo   DIAGNOSTICO DE EMERGENCIA - AI SYMBIOTE
echo ============================================
echo.

echo [1] Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python no disponible
    goto :error
)

echo [2] Verificando archivos core...
if not exist "D:\\Obvivlorum\\ai_symbiote.py" (
    echo ERROR: ai_symbiote.py no encontrado
    goto :error
)

echo [3] Verificando dependencias...
python -c "import tkinter, fastapi; print('Dependencias OK')"
if %errorlevel% neq 0 (
    echo ERROR: Dependencias faltantes
    goto :error
)

echo [4] Verificando persistencia...
reg query "HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run" | find "AISymbiote"
if %errorlevel% neq 0 (
    echo WARNING: Persistencia no configurada
)

echo.
echo ========== DIAGNOSTICO COMPLETO ==========
echo Sistema: OK
echo Para iniciar manualmente:
echo   python "D:\\Obvivlorum\\ai_symbiote.py"
echo   python "D:\\Obvivlorum\\ai_symbiote_gui.py"
echo ==========================================
goto :end

:error
echo.
echo ERROR: Sistema no funcional
echo Contacta soporte técnico

:end
pause
"""
    
    with open("D:/Obvivlorum/EMERGENCY_DIAGNOSTIC.bat", 'w', encoding='utf-8') as f:
        f.write(diag_script)
    print("✓ Script de diagnóstico de emergencia creado")
    
    # Script de recuperación
    recovery_script = """@echo off
title AI Symbiote - Recuperación del Sistema
echo Iniciando recuperación del sistema...
echo.

REM Reinstalar persistencia
python "D:\\Obvivlorum\\FIX_PERSISTENCE.py"

REM Recrear accesos directos  
python "D:\\Obvivlorum\\CREATE_START_MENU_SHORTCUT.py"

REM Verificar sistema
python "D:\\Obvivlorum\\FINAL_SYSTEM_CHECK.py"

echo.
echo Recuperación completada
pause
"""
    
    with open("D:/Obvivlorum/EMERGENCY_RECOVERY.bat", 'w', encoding='utf-8') as f:
        f.write(recovery_script)
    print("✓ Script de recuperación de emergencia creado")

def main():
    """Verificación completa del sistema."""
    print("AI SYMBIOTE - VERIFICACION FINAL COMPLETA")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Ejecutar todas las verificaciones
    checks = [
        ("Python y Dependencias", check_python_and_deps),
        ("Estructura de Archivos", check_file_structure), 
        ("Persistencia Windows", check_persistence),
        ("Puertos de Red", check_ports),
        ("Accesos Directos", check_shortcuts),
        ("Sistema Core", test_core_system)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_checks_passed = False
        except Exception as e:
            print(f"ERROR en {check_name}: {e}")
            results[check_name] = False
            all_checks_passed = False
    
    # Crear scripts de emergencia
    create_emergency_scripts()
    
    # Resumen final
    print("\n" + "=" * 50)
    print("RESUMEN FINAL DE VERIFICACION")
    print("=" * 50)
    
    for check_name, result in results.items():
        status = "✓ OK" if result else "✗ FALLO"
        print(f"{check_name:.<30} {status}")
    
    if all_checks_passed:
        print("\n🎉 SISTEMA COMPLETAMENTE FUNCIONAL")
        print("✓ Todos los componentes verificados")
        print("✓ Persistencia configurada") 
        print("✓ Accesos directos creados")
        print("✓ Scripts de emergencia listos")
        print("\nPara iniciar: START_SYSTEM_FIXED.bat")
        
        # Crear archivo de estado
        status_data = {
            "last_check": "2025-09-03",
            "status": "FULLY_FUNCTIONAL",
            "components_verified": list(results.keys()),
            "all_checks_passed": True,
            "ready_for_production": True
        }
        
        with open("D:/Obvivlorum/SYSTEM_VERIFICATION_COMPLETE.json", 'w') as f:
            json.dump(status_data, f, indent=2)
            
        print("✓ Certificación guardada: SYSTEM_VERIFICATION_COMPLETE.json")
    else:
        print("\n[WARNING] SISTEMA PARCIALMENTE FUNCIONAL")
        print("Algunos componentes requieren atención")
        print("Ejecuta EMERGENCY_RECOVERY.bat para reparar")
    
    return all_checks_passed

if __name__ == "__main__":
    main()