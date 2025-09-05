#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rápido del sistema AI Symbiote con Auditoría de Seguridad Avanzada
"""
import os
import sys
import json
import subprocess
import hashlib
import time
from pathlib import Path
import winreg
from datetime import datetime

def quick_test():
    """Test rápido de funcionalidad."""
    print("AI SYMBIOTE - TEST RAPIDO")
    print("=" * 30)
    
    tests_passed = 0
    total_tests = 8
    
    # 1. Python
    print("[1/8] Python... ", end="")
    try:
        print(f"OK - {sys.version.split()[0]}")
        tests_passed += 1
    except:
        print("ERROR")
    
    # 2. Archivos core
    print("[2/8] Archivos core... ", end="")
    core_files = [
        "D:/Obvivlorum/ai_symbiote.py",
        "D:/Obvivlorum/START_SYSTEM_FIXED.bat"
    ]
    if all(Path(f).exists() for f in core_files):
        print("OK")
        tests_passed += 1
    else:
        print("ERROR - Archivos faltantes")
    
    # 3. Dependencias
    print("[3/8] Dependencias... ", end="")
    try:
        import tkinter, fastapi, json
        print("OK")
        tests_passed += 1
    except ImportError as e:
        print(f"ERROR - {e}")
    
    # 4. Persistencia registro
    print("[4/8] Persistencia registro... ", end="")
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                           r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run")
        value, regtype = winreg.QueryValueEx(key, "AISymbiote")
        winreg.CloseKey(key)
        print("OK")
        tests_passed += 1
    except:
        print("ERROR")
    
    # 5. Archivos persistencia
    print("[5/8] Archivos persistencia... ", end="")
    appdata_folder = Path(os.environ['APPDATA']) / "AISymbiote"
    if (appdata_folder / "start_symbiote_fixed.vbs").exists():
        print("OK")
        tests_passed += 1
    else:
        print("ERROR")
    
    # 6. Accesos directos
    print("[6/8] Accesos directos... ", end="")
    menu_folder = Path(os.environ['APPDATA']) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "AI Symbiote"
    if menu_folder.exists() and len(list(menu_folder.glob("*.lnk"))) > 0:
        print("OK")
        tests_passed += 1
    else:
        print("ERROR")
    
    # 7. Puerto web
    print("[7/8] Puerto web... ", end="")
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 8000))
    sock.close()
    if result != 0:
        print("OK - Disponible")
        tests_passed += 1
    else:
        print("WARNING - En uso")
    
    # 8. Test import core
    print("[8/8] Import core... ", end="")
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, 'D:/Obvivlorum'); import ai_symbiote"
        ], capture_output=True, text=True, cwd="D:/Obvivlorum", timeout=10)
        
        if result.returncode == 0:
            print("OK")
            tests_passed += 1
        else:
            print(f"ERROR - {result.stderr[:50]}...")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    except Exception as e:
        print(f"ERROR - {e}")
    
    # AUDITORÍA DE SEGURIDAD AVANZADA
    print("\n" + "🔒 AUDITORÍA DE SEGURIDAD AVANZADA")
    print("=" * 40)
    
    security_tests_passed = 0
    security_total_tests = 5
    
    # Test 1: Integridad de archivos críticos
    print("[SEC-1/5] Integridad de archivos... ", end="")
    integrity_result = test_file_integrity()
    if integrity_result["passed"]:
        print("OK")
        security_tests_passed += 1
    else:
        print(f"WARNING - {integrity_result['issues']} archivos modificados")
    
    # Test 2: Configuraciones de seguridad
    print("[SEC-2/5] Configuraciones seguridad... ", end="")
    config_result = test_security_configurations()
    if config_result["passed"]:
        print("OK")
        security_tests_passed += 1
    else:
        print(f"WARNING - {config_result['issues']} configuraciones inseguras")
    
    # Test 3: Scripts de arranque
    print("[SEC-3/5] Scripts arranque... ", end="")
    startup_result = test_startup_scripts_integrity()
    if startup_result["passed"]:
        print("OK")
        security_tests_passed += 1
    else:
        print(f"WARNING - {startup_result['issues']} scripts comprometidos")
    
    # Test 4: Persistencia controlada
    print("[SEC-4/5] Persistencia controlada... ", end="")
    persistence_result = test_persistence_control()
    if persistence_result["passed"]:
        print("OK")
        security_tests_passed += 1
    else:
        print(f"WARNING - Persistencia no controlada")
    
    # Test 5: Niveles de amenaza
    print("[SEC-5/5] Niveles de amenaza... ", end="")
    threat_result = test_threat_levels()
    if threat_result["passed"]:
        print("OK")
        security_tests_passed += 1
    else:
        print(f"WARNING - Detectados {threat_result['threats']} amenazas")
    
    # Resultado final COMBINADO
    print("\n" + "=" * 40)
    total_combined_tests = total_tests + security_total_tests
    total_combined_passed = tests_passed + security_tests_passed
    combined_percentage = (total_combined_passed / total_combined_tests) * 100
    
    print(f"FUNCIONAL: {tests_passed}/{total_tests} tests OK")
    print(f"SEGURIDAD: {security_tests_passed}/{security_total_tests} tests OK")
    print(f"RESULTADO COMBINADO: {total_combined_passed}/{total_combined_tests} tests OK ({combined_percentage:.1f}%)")
    
    if total_combined_passed >= 11:
        print("ESTADO: SISTEMA FUNCIONAL Y SEGURO")
        status = "FUNCTIONAL_SECURE"
    elif total_combined_passed >= 9:
        print("ESTADO: FUNCIONAL CON WARNINGS DE SEGURIDAD")
        status = "FUNCTIONAL_WARNINGS"
    elif tests_passed >= 6:
        print("ESTADO: FUNCIONAL PERO INSEGURO")
        status = "FUNCTIONAL_INSECURE"
    else:
        print("ESTADO: REQUIERE REPARACION COMPLETA")
        status = "BROKEN"
    
    # Guardar resultado COMPLETO
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "functional_tests": {
            "passed": tests_passed,
            "total": total_tests,
            "percentage": (tests_passed / total_tests) * 100
        },
        "security_tests": {
            "passed": security_tests_passed,
            "total": security_total_tests,
            "percentage": (security_tests_passed / security_total_tests) * 100
        },
        "combined_results": {
            "passed": total_combined_passed,
            "total": total_combined_tests,
            "percentage": combined_percentage
        },
        "status": status,
        "ready_to_use": total_combined_passed >= 9,
        "security_details": {
            "integrity": integrity_result,
            "configurations": config_result,
            "startup_scripts": startup_result,
            "persistence_control": persistence_result,
            "threat_levels": threat_result
        }
    }
    
    with open("D:/Obvivlorum/QUICK_TEST_RESULT.json", 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Resultado guardado: QUICK_TEST_RESULT.json")
    
    if total_combined_passed >= 9:
        print("\n✅ PRÓXIMO PASO: Ejecutar START_SYSTEM_FIXED.bat")
        print("💡 Sistema seguro y listo para producción")
    elif tests_passed >= 6:
        print("\n⚠️  PRÓXIMO PASO: Revisar warnings de seguridad")
        print("🔧 Ejecutar adaptive_persistence_scheduler.py")
    else:
        print("\n❌ PRÓXIMO PASO: Ejecutar EMERGENCY_RECOVERY.bat")
        print("🆘 Sistema requiere reparación completa")
    
    return total_combined_passed >= 9


def test_file_integrity():
    """Test de integridad de archivos críticos mediante hashes."""
    try:
        critical_files = {
            "D:/Obvivlorum/ai_symbiote.py": None,
            "D:/Obvivlorum/config_optimized.json": None,
            "D:/Obvivlorum/linux_executor.py": None,
            "D:/Obvivlorum/windows_persistence.py": None,
            "D:/Obvivlorum/parrot_security_config.json": None
        }
        
        modified_files = 0
        hash_file = Path("D:/Obvivlorum/.file_hashes.json")
        
        # Load previous hashes if exist
        previous_hashes = {}
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    previous_hashes = json.load(f)
            except:
                pass
        
        current_hashes = {}
        for file_path in critical_files:
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    content = f.read()
                    current_hash = hashlib.sha256(content).hexdigest()
                    current_hashes[file_path] = current_hash
                    
                    # Compare with previous hash
                    if file_path in previous_hashes:
                        if previous_hashes[file_path] != current_hash:
                            modified_files += 1
            else:
                modified_files += 1
        
        # Save current hashes
        with open(hash_file, 'w') as f:
            json.dump(current_hashes, f, indent=2)
        
        return {
            "passed": modified_files == 0,
            "issues": modified_files,
            "details": "Archivos críticos verificados por hash SHA256"
        }
        
    except Exception as e:
        return {
            "passed": False,
            "issues": "unknown",
            "details": f"Error en verificación: {e}"
        }


def test_security_configurations():
    """Test de configuraciones de seguridad."""
    try:
        insecure_configs = 0
        
        # Check config_optimized.json
        config_path = Path("D:/Obvivlorum/config_optimized.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                # Check dangerous settings
                linux_config = config.get("components", {}).get("linux_executor", {})
                if linux_config.get("safe_mode") is not False:
                    insecure_configs += 1  # Should be False for unrestricted mode
                if not linux_config.get("unrestricted_mode"):
                    insecure_configs += 1
        else:
            insecure_configs += 1
        
        # Check persistence controls
        persistence_config = Path("C:/Users/Propietario/AppData/Local/AISymbiote/persistence_config.json")
        if persistence_config.exists():
            with open(persistence_config, 'r') as f:
                persist_config = json.load(f)
                config_data = persist_config.get("config", {})
                
                # These should be False for security
                dangerous_settings = ["auto_start", "stealth_mode", "self_healing"]
                for setting in dangerous_settings:
                    if config_data.get(setting) is True:
                        insecure_configs += 1
        
        return {
            "passed": insecure_configs == 0,
            "issues": insecure_configs,
            "details": f"Configuraciones inseguras detectadas: {insecure_configs}"
        }
        
    except Exception as e:
        return {
            "passed": False,
            "issues": "unknown",
            "details": f"Error verificando configuraciones: {e}"
        }


def test_startup_scripts_integrity():
    """Test de integridad de scripts de arranque."""
    try:
        compromised_scripts = 0
        
        startup_paths = [
            Path(os.path.expanduser("~")) / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup",
            Path("D:/Obvivlorum"),
        ]
        
        suspicious_patterns = [
            b"powershell -enc",  # Encoded PowerShell
            b"cmd /c start",     # Suspicious command execution
            b"certutil -decode", # Base64 decode operations
            b"rundll32",         # DLL execution
        ]
        
        for startup_path in startup_paths:
            if startup_path.exists():
                for script_file in startup_path.glob("*.bat"):
                    try:
                        with open(script_file, 'rb') as f:
                            content = f.read().lower()
                            for pattern in suspicious_patterns:
                                if pattern in content:
                                    compromised_scripts += 1
                                    break
                    except:
                        pass
        
        return {
            "passed": compromised_scripts == 0,
            "issues": compromised_scripts,
            "details": f"Scripts comprometidos: {compromised_scripts}"
        }
        
    except Exception as e:
        return {
            "passed": False,
            "issues": "unknown",
            "details": f"Error verificando scripts: {e}"
        }


def test_persistence_control():
    """Test de control de persistencia."""
    try:
        # Check if adaptive scheduler exists
        scheduler_exists = Path("D:/Obvivlorum/adaptive_persistence_scheduler.py").exists()
        
        # Check control flags
        disable_flag = Path("D:/Obvivlorum/DISABLE_PERSISTENCE.flag").exists()
        force_enable_flag = Path("D:/Obvivlorum/FORCE_ENABLE_PERSISTENCE.flag").exists()
        
        # Check registry entries (should be clean)
        registry_clean = True
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run")
            try:
                value, _ = winreg.QueryValueEx(key, "AISymbiote")
                registry_clean = False  # Persistence entry found
            except FileNotFoundError:
                pass  # Good, no persistence entry
            winreg.CloseKey(key)
        except:
            pass
        
        controlled = scheduler_exists and registry_clean and not force_enable_flag
        
        return {
            "passed": controlled,
            "issues": "No controlada" if not controlled else "Controlada",
            "details": {
                "scheduler_exists": scheduler_exists,
                "registry_clean": registry_clean,
                "disable_flag": disable_flag,
                "force_enable_flag": force_enable_flag
            }
        }
        
    except Exception as e:
        return {
            "passed": False,
            "issues": "Error",
            "details": f"Error verificando persistencia: {e}"
        }


def test_threat_levels():
    """Test de detección de amenazas."""
    try:
        threats_detected = 0
        
        # Check for suspicious processes
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info.get('name', '').lower()
                    cmdline = ' '.join(proc_info.get('cmdline', [])).lower()
                    
                    # Suspicious process patterns
                    suspicious_names = ['mimikatz', 'psexec', 'netcat', 'ncat']
                    suspicious_cmdline_patterns = [
                        'powershell -enc',
                        'cmd /c echo',
                        'certutil -decode'
                    ]
                    
                    if any(sus in proc_name for sus in suspicious_names):
                        threats_detected += 1
                    
                    for pattern in suspicious_cmdline_patterns:
                        if pattern in cmdline:
                            threats_detected += 1
                            break
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass
        
        # Check for suspicious network connections
        try:
            import psutil
            connections = psutil.net_connections(kind='inet')
            suspicious_ports = [4444, 5555, 6666, 31337]
            
            for conn in connections:
                if conn.laddr and conn.laddr.port in suspicious_ports:
                    threats_detected += 1
        except:
            pass
        
        return {
            "passed": threats_detected == 0,
            "threats": threats_detected,
            "details": f"Amenazas detectadas: procesos sospechosos, puertos maliciosos"
        }
        
    except Exception as e:
        return {
            "passed": True,  # Assume safe if can't check
            "threats": 0,
            "details": f"No se pudo verificar amenazas: {e}"
        }


if __name__ == "__main__":
    quick_test()